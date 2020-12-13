import taichi as ti
import numpy as np

DIM = 2

'''
A differentiable simulator program which 
performs fluid simulation and optimized the 
particle field towards a target image.
'''
@ti.data_oriented
class Simulator:

    '''
    Initializes the simulator which runs on 
    device DEVICE.
    Note: OpenGL and CUDA are different Taichi devices.
    '''
    def __init__(self, device, num_particles=32768, grid_size=192,
            dt = 0.0001):
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
        self.grid_size = grid_size
        self.buffer = int(grid_size / 32)
        self.domain_size = 1
        self.dt = dt
        self.lr = 3
        self.gravity = -9.8
        # other quantities
        self.inv_dx = self.grid_size / self.domain_size
        self.particle_vol = (0.5 * self.domain_size / self.grid_size) ** DIM
        self.particle_rho = 1
        self.particle_mass = self.particle_vol * self.particle_rho
        self.E = 400
        self.vlim = self.domain_size / (self.grid_size * DIM * self.dt)
        self.frame = 0

        # Taichi fields
        self.pos = ti.Vector.field(DIM, dtype=ti.f32, shape=self.num_particles, needs_grad=True)  # current position
        self.vel = ti.Vector.field(DIM, dtype=ti.f32, shape=self.num_particles, needs_grad=True)
        self.aff = ti.Matrix.field(DIM, DIM, dtype=ti.f32, shape=self.num_particles, needs_grad=True)
        self.col = ti.Vector.field(3, dtype=ti.f32, shape=self.num_particles, needs_grad=True)
        self.J = ti.field(dtype=ti.f32, shape=self.num_particles)

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
    def step(self, substep=20):
        for _ in range(substep):
            self.clear_grid()
            with ti.Tape(self.loss):
                self.p2g()
                self.grid_step()
            self.grad_step(False)
            self.particle_update()
            self.g2p()
        self.frame += 1

    '''
    Performs one step of gradient descent of the simulation
    towards the specified loss.
    '''
    def optimize(self, substep=20):
        for _ in range(substep):
            self.clear_grid()
            with ti.Tape(self.loss):
                self.p2g()
                self.grid_step()
            self.grad_step(True)
        self.frame += 1
            

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
            self.J[i] = 1

    @ti.kernel
    def clear_grid(self):
        for I in ti.grouped(self.grid_m):
            self.grid_m[I] = 0
            self.grid_v[I].fill(0)
            self.grid_c[I].fill(0)

    @ti.kernel
    def particle_update(self):
        for i in self.pos:
            self.pos[i] += self.vel[i] * self.dt

    ''' gradient descent'''
    @ti.kernel
    def grad_step(self, step_velocity : ti.Template()):
        for i in self.col:
            self.col[i] -= self.lr * self.col.grad[i]
            self.col[i] = min(max(0, self.col[i]), 1) # prevent negative colors
            if step_velocity:
                self.vel[i] -= 0.001 * self.lr * self.vel.grad[i]
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
            stress = -self.dt * self.particle_vol * (self.J[i] - 1) * 4 * self.E * self.inv_dx * self.inv_dx
            affine = ti.Matrix.identity(float, DIM) * stress + self.aff[i] * self.particle_mass
            for dX in ti.static(ti.ndrange(3, 3)): 
                weight = w[dX[0]][0] * w[dX[1]][1]
                offset_x = (dX - fx) / self.inv_dx
                self.grid_v[base + dX] += weight * (self.particle_mass * self.vel[i] + affine @ offset_x)
                self.grid_m[base + dX] += weight * self.particle_mass
                self.grid_c[base + dX] += weight * self.col[i]


    @ti.kernel
    def grid_step(self):
        for I in ti.grouped(self.grid_m):
            if self.grid_m[I] > 0:
                self.grid_v[I] = self.grid_v[I] / self.grid_m[I]  # change this grid_v as well
                self.grid_v[I][1] += self.gravity * self.dt
                self.grid_c[I] = self.grid_c[I] * self.particle_mass / self.grid_m[I]
                # boundary condition
                for d in ti.static(range(DIM)):
                    cond = (I[d] < self.buffer and self.grid_v[I][d] < 0) \
                        or (I[d] > self.grid_size - self.buffer and self.grid_v[I][d] > 0)
                    if cond:
                        self.grid_v[I][d] = 0
            # ================= compute loss ====================
            self.loss[None] += ((self.target[I] - self.grid_c[I]) ** 2).sum() / (self.grid_size ** 2)
            # ===================================================

    @ti.kernel
    def g2p(self):
        for i in range(self.num_particles):
            base = (self.pos[i] * self.inv_dx - 0.5).cast(int)
            fx = self.pos[i] * self.inv_dx - base.cast(float)
            # quadratic B-spline
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, DIM)
            #=====================================
            #new_c = ti.Vector.zero(float, 3)
            # =====================================
            new_A = ti.Matrix.zero(float, DIM, DIM)
            for dX in ti.static(ti.ndrange(3, 3)):
                offset_X = dX - fx
                weight = w[dX[0]][0] * w[dX[1]][1]
                new_v += weight * self.grid_v[base + dX]
                # =====================================
                #new_c += weight * grid_c[base + dX]
                # =====================================
                new_A += 4 * weight * self.grid_v[base + dX].outer_product(offset_X) * self.inv_dx
            self.vel[i] = new_v
            self.aff[i] = new_A
            self.J[i] *= 1 + self.dt * self.aff[i].trace()

    