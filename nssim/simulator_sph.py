import taichi as ti
import numpy as np

DIM = 2

'''
A differentiable simulator program which 
performs fluid simulation and optimized the 
particle field towards a target image.
'''
@ti.data_oriented
class SPHSimulator2D:

    buffer = 0.02

    '''
    Initializes the simulator which runs on 
    device DEVICE.
    Note: OpenGL and CUDA are different Taichi devices.
    '''
    def __init__(self, device, num_particles=32768, grid_size=192,
            dt = 0.0001, lr=0.1):
        device = device.lower()
        if device == "cuda":
            self.device = ti.cuda
        elif device == "opengl":
            self.device = ti.opengl
        else:
            self.device = ti.cpu
        ti.init(arch=self.device)

        # simulation parameters
        self.num_particles = num_particles
        self.K_smoothingRadius = 0.02
        self.K_stiff = 100  # stiffness
        self.K_stiffN = 200  # stiffness near
        self.viscosity_a = 20
        self.viscosity_b = 4
        self.K_restDensity = 8 # rest density
        self.restitution = -0.5
        self.domain_size = 1
        self.grid_size = grid_size#self.domain_size // self.K_smoothingRadius
        self.dt = dt
        self.lr_sim = lr
        self.lr_init = lr * 50
        self.gravity = -9.8

        # other quantities
        self.inv_dx = self.grid_size / self.domain_size
        self.particle_mass = 1
        # density of particles
        self.dens = ti.field(dtype=ti.f32, shape=num_particles)  # density
        self.densN = ti.field(dtype=ti.f32, shape=num_particles)  # near density
        # pressure of particles
        self.press = ti.field(dtype=ti.f32, shape=num_particles)  # density
        self.pressN = ti.field(dtype=ti.f32, shape=num_particles)  # near density
        # pairs
        max_pairs = 128  # max neighbor of one particle
        self.pair = ti.field(dtype=ti.i32, shape=(num_particles - 1, max_pairs))
        self.dist = ti.field(dtype=ti.f32, shape=(num_particles - 1, max_pairs))  # store distance for each pair
        self.num_pair = ti.field(dtype=ti.i32, shape=(num_particles - 1,))  # number of pairs

        self.vlim = self.domain_size / (self.grid_size * DIM * self.dt)
        self.frame = 0

        # Taichi fields
        self.pos = ti.Vector.field(DIM, dtype=ti.f32, shape=self.num_particles, needs_grad=True)  # current position
        self.oldPos = ti.Vector.field(DIM, dtype=ti.f32, shape=self.num_particles, needs_grad=True) # old position
        self.vel = ti.Vector.field(DIM, dtype=ti.f32, shape=self.num_particles, needs_grad=True)
        self.col = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles, needs_grad=True)

        self.grid_v = ti.Vector.field(DIM
        , dtype=ti.f32, shape=(self.grid_size, self.grid_size), needs_grad=True)
        self.grid_m = ti.field(dtype=ti.f32, shape=(self.grid_size, self.grid_size), needs_grad=True)
        self.grid_c = ti.Vector.field(3, dtype=ti.f32, shape=(self.grid_size, self.grid_size), needs_grad=True)

        self.target = ti.Vector.field(3, dtype=ti.f32, shape=(self.grid_size, self.grid_size))
        self.loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)
                
    
    '''
    Initializes the simulation system. This involves
    materialization of Taichi kernels, after which no
    new Taichi fields can be created.
    '''
    def initialize(self):
        self.particle_init()
        
    '''
    Sets the target of the optimization to a given image.
    '''
    def set_target(self, target_img):
        if isinstance(target_img, np.ndarray):
            self.target.from_numpy(target_img)
        else:
            self.target.from_torch(target_img)

    '''
    Updates the simulation by one frame. One frame
    may include multiple substeps.
    '''
    def step(self, substep=5):
        for _ in range(substep):
            self.clear_state()
            self.predict_velocity()  # initial prediction of velocity
            with ti.Tape(self.loss):
                #self.particle_update() # update with predicted velocity
                self.p2g() # convert to grid representation
                self.grid_step()
            self.grad_step(False, self.lr_sim) # apply gradient descend to color and velocity
            #self.restore_position() # undo the original position update
            self.particle_update() # update position with the new velocity
        self.frame += 1

    '''
    Performs one step of gradient descent of the simulation
    towards the specified loss.
    '''
    def optimize(self, substep=20):
        for _ in range(substep):
            self.clear_state()
            with ti.Tape(self.loss):
                self.p2g()  # convert to grid representation
                self.grid_step()
            self.grad_step(True, self.lr_init)
        self.frame += 1
    
    '''
    Returns the input to stylization.
    '''
    def get_fields(self):
        return self.density_field()

    '''
    Sets the optimization target
    '''
    def set_target(self, target):
        self.target.from_numpy(target)

    '''
    Returns the density field of the simulation
    as a numpy array.
    '''
    def density_field(self) -> np.ndarray:
        return self.grid_m.to_numpy() / self.particle_mass

    '''
    Returns the color field of the simulation
    as a numpy array.
    '''
    def color_field(self) -> np.ndarray:
        return self.grid_c.to_numpy()

    '''
    ====== Taichi kernels and functions =======
    '''
    @ti.kernel
    def particle_init(self):
        for i in range(self.num_particles):
            length = self.num_particles ** (1 / DIM)
            # place particles around the center of the domain
            self.pos[i] = ti.Vector([i % length / length, i // length / length])  # 0 - 1
            self.pos[i] = (self.pos[i] + 1) / 2 - 0.05# + ti.random() * 0.001  # 0.25 - 0.75 (centered)
            self.pos[i][1] -= 0.35
            self.pos[i] *= self.domain_size  # scale to fit the domain
            self.oldPos[i] = self.pos[i]

    @ti.kernel
    def clear_state(self):
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0
            self.grid_v[I].fill(0)
            self.grid_c[I].fill(0)
        for i in range(self.num_particles):
            self.dens[i] = 0
            self.densN[i] = 0
            self.num_pair[i] = 0

    @ti.kernel
    def predict_velocity(self):
        # predict velocity
        for i in range(self.num_particles):
            # compute new velocity
            self.vel[i] = (self.pos[i] - self.oldPos[i]) / self.dt
            # collision handling?
            self.boundary_collision(i)
            # save previous position
            self.oldPos[i] = self.pos[i]
            # apply gravity
            self.vel[i][1] += (self.gravity * self.dt)

        # apply viscosity
        for i in range(self.num_particles - 1):
            for j in range(i + 1, self.num_particles):
                q = (self.pos[i] - self.pos[j]).norm() / self.K_smoothingRadius
                if q < 1:
                    r_ij = ti.normalized(self.pos[i] - self.pos[j])
                    u = ti.dot((self.vel[i] - self.vel[j]), r_ij)
                    if u > 0:
                        I = self.dt * (1 - q) * (self.viscosity_a * u + self.viscosity_b * u * u) * r_ij
                        self.vel[i] -= I / 2
                        self.vel[j] += I / 2

    @ti.kernel
    def particle_update(self):

        # predict new position
        for i in range(self.num_particles):
            # advance to new position
            self.pos[i] += (self.vel[i] * self.dt)

        # compute pair info
        for i in range(self.num_particles - 1):
            for j in range(i + 1, self.num_particles):
                distance = (self.pos[i] - self.pos[j]).norm()
                if distance < self.K_smoothingRadius:
                    self.pair[i, self.num_pair[i]] = j
                    self.dist[i, self.num_pair[i]] = distance
                    self.num_pair[i] += 1

        # compute density
        for i in range(self.num_particles):
            self.dens[i] = 1
            self.densN[i] = 1

        for i in range(self.num_particles - 1):
            for j in range(self.num_pair[i]):
                q = 1 - self.dist[i, j] / self.K_smoothingRadius
                q2 = q * q
                q3 = q2 * q
                # print(dist[i], q)
                self.dens[i] += q2
                self.dens[self.pair[i, j]] += q2
                self.densN[i] += q3
                self.densN[self.pair[i, j]] += q3

        # update pressure
        for i in range(self.num_particles):
            self.press[i] = self.K_stiff * (self.dens[i] - self.K_restDensity)
            self.pressN[i] = self.K_stiffN * self.densN[i]

        # apply pressure
        for i in range(self.num_particles - 1):
            for j in range(self.num_pair[i]):
                p = self.press[i] + self.press[self.pair[i, j]]
                pN = self.pressN[i] + self.pressN[self.pair[i, j]]

                q = 1 - self.dist[i, j] / self.K_smoothingRadius
                q2 = q * q

                displace = (p * q + pN * q2) * (self.dt ** 2)
                a2bN = (self.pos[i] - self.pos[self.pair[i, j]]) / self.dist[i, j]

                self.pos[i] += displace * a2bN
                self.pos[self.pair[i, j]] -= displace * a2bN

    @ti.kernel
    def restore_position(self):
        for i in range(self.num_particles):
            self.pos[i] = self.oldPos[i]

    ''' handle particle collision with boundary '''

    @ti.func
    def boundary_collision(self, index):

        # x boundary
        if self.pos[index][0] < self.buffer:
            self.pos[index][0] = self.buffer
            self.vel[index][0] *= self.restitution
        elif self.pos[index][0] > self.domain_size - self.buffer:
            self.pos[index][0] = self.domain_size - self.buffer
            self.vel[index][0] *= self.restitution

        # y boundary
        if self.pos[index][1] < self.buffer:
            self.pos[index][1] = self.buffer
            self.vel[index][1] *= self.restitution
        elif self.pos[index][1] > self.domain_size - self.buffer:
            self.pos[index][1] = self.domain_size - self.buffer
            self.vel[index][1] *= self.restitution

    ''' update particle position '''

    @ti.kernel
    def position_update(self):
        for i in self.pos:
            self.pos[i] += self.vel[i] * self.dt

    ''' gradient descent'''
    @ti.kernel
    def grad_step(self, step_velocity : ti.Template(), lr: ti.f32):
        for i in self.col:
            self.col[i] -= lr * self.col.grad[i]
            self.col[i] = min(max(0, self.col[i]), 1) # prevent negative colors
            if step_velocity:
                self.vel[i] -= self.dt * lr * self.vel.grad[i]
                for d in ti.static(range(DIM)):
                    self.vel[i][d] = min(max(self.vel[i][d], -self.vlim), self.vlim)

    ''' Lagrangian to Eularian representation '''

    @ti.kernel
    def p2g(self):
        for i in range(self.num_particles):
            newpos = self.pos[i] + self.vel[i] * self.dt
            base = (newpos * self.inv_dx - 0.5).cast(int)
            fx = newpos * self.inv_dx - base.cast(float)
            # quadratic B-spline
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            for dX in ti.static(ti.ndrange(3, 3)): 
                weight = w[dX[0]][0] * w[dX[1]][1]
                offset_x = (dX - fx) / self.inv_dx
                self.grid_v[base + dX] += weight * self.vel[i]
                self.grid_m[base + dX] += weight
                self.grid_c[base + dX] += weight * self.col[i]


    @ti.kernel
    def grid_step(self): # compute loss
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_c[I] = self.grid_c[I] / self.grid_m[I]
                self.loss[None] += ((self.target[I] - self.grid_c[I]) ** 2).sum() / self.grid_size


# main function for debugging
if __name__ == '__main__':
    gui = ti.GUI('SPH Fluid', 768, background_color=0x000000)
    # gui_g1 = ti.GUI('grid_m', grid_size, background_color = 0x000000)
    # gui_g2 = ti.GUI('grid_v', grid_size, background_color = 0x000000)
    sph = SPH_Simulator("cuda")

    while True:
        sph.initialize()
        gui.clear(0x000000)
        for _ in range(15):
            sph.step()

        gui.circles(sph.pos.to_numpy() / sph.domain_size, radius=6, color=0xC0D9D9)
        # for _ in range(1):

        # draw particle

        # grid_w_np = grid_w.to_numpy()
        gui.show()