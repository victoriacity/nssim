''' Combine MPM simulation with particle velocity and color optimization '''

import numpy as np
import taichi as ti

DIM = 2

ti.init(arch=ti.cuda)

''' parameters '''

num_particles = 32768
grid_size = 192

domain_size = 1

inv_dx = grid_size / domain_size

particle_vol = (0.5 * domain_size / grid_size) ** DIM
particle_rho = 1
particle_mass = particle_vol * particle_rho

E = 400

# delta t
lr = 10
dt = 0.00008

gravity = -9.8

restitution = 0


# velocity limit by CFL condition
vlim = domain_size / (grid_size * DIM * dt)

print(vlim)

# img = np.transpose(ti.imread("starry.jpg"), (1, 0, 2)).reshape(-1, 3)
img = np.zeros((grid_size, grid_size, 3)).reshape(-1, 3) + 255
# rgb 2 hex
img = img[:, 0] * 65536 + img[:, 1] * 256 + img[:, 2]

''' Fields '''
# positions of particles
pos = ti.Vector.field(2, dtype=ti.f32, shape=num_particles, needs_grad=True)  # current position

# velocities of particles
vel = ti.Vector.field(2, dtype=ti.f32, shape=num_particles, needs_grad=True)
# affine velocity field
aff = ti.Matrix.field(DIM, DIM, dtype=ti.f32, shape=num_particles, needs_grad=True)
# volume deformation
J = ti.field(dtype=ti.f32, shape=num_particles)

# Eularian grid

grid_v = ti.Vector.field(2, dtype=ti.f32, shape=(grid_size, grid_size), needs_grad=True)
grid_m = ti.field(dtype=ti.f32, shape=(grid_size, grid_size), needs_grad=True)

# =======================================================
grid_c = ti.Vector.field(3, dtype=ti.f32, shape=(grid_size, grid_size), needs_grad=True)

# color of particles
col = ti.Vector.field(3, dtype=ti.f32, shape=num_particles, needs_grad=True)

# target image
target = ti.Vector.field(3, dtype=ti.f32, shape=(grid_size, grid_size))

# loss
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

img = ti.imread("pattern1_256.png")
# =======================================================


''' initialize particle position & velocity '''

def init():
    particle_init()
    target.from_numpy(img / 255)
    print("dam grid spacing:", 0.5 / (num_particles) ** (1 / 2) * domain_size)
    print("init density:", num_particles / (0.5 * domain_size) ** 2)

@ti.kernel
def particle_init():
    for i in range(num_particles):
        length = ti.sqrt(num_particles)
        # place particles around the center of the domain
        pos[i] = ti.Vector([i % length / length, i // length / length])  # 0 - 1
        pos[i] = (pos[i] + 1) / 2 - 0.05# + ti.random() * 0.001  # 0.25 - 0.75 (centered)
        pos[i] *= domain_size  # scale to fit the domain
        J[i] = 1


# ====================================================



def optimize():
    clear_grid()
    with ti.Tape(loss):
        p2g()
        grid_step()
    step_color()

''' update particle state '''
def update():
    clear_grid()
    with ti.Tape(loss):
        p2g()
        grid_step()
    grad_step()
    particle_update()
    g2p()
    
    
    # advect particles
    

@ti.kernel
def clear_grid():
    for I in ti.grouped(grid_m):
        grid_m[I] = 0
        grid_v[I].fill(0)
        grid_c[I].fill(0)



@ti.kernel
def particle_update():
    for i in pos:
        pos[i] += vel[i] * dt


''' gradient descend'''
@ti.kernel
def grad_step():
    for i in col:
        #col[i] -= lr * col.grad[i]
        #col[i] = min(max(0, col[i]), 1) # prevent negative colors
        vel[i] -= lr * vel.grad[i]
        for d in ti.static(range(DIM)):
            vel[i][d] = min(max(vel[i][d], -vlim), vlim)
    #for I in ti.grouped(grid_v):
        #print(grid_v.grad[I])
    #    grid_v[I] -= 50 * lr * grid_v.grad[I]
# ====================================================

@ti.kernel
def step_color():
    for i in col:
        col[i] -= lr * col.grad[i]
        col[i] = min(max(0, col[i]), 1)

''' helper function to compute distance between two points '''


@ti.func
def distance(v1, v2):
    dv = v1 - v2
    dis = dv.norm()
    return dis


buffer = 6  # buffer grids for boundary conditions

''' Lagrangian to Eularian representation '''


@ti.kernel
def p2g():
    for i in range(num_particles):
        newpos = pos[i] + vel[i] * dt
        base = (newpos * inv_dx - 0.5).cast(int)
        fx = newpos * inv_dx - base.cast(float)
        # quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * particle_vol * (J[i] - 1) * 4 * E * inv_dx * inv_dx
        affine = ti.Matrix.identity(float, DIM) * stress + aff[i] * particle_mass
        for dX in ti.static(ti.ndrange(3, 3)): 
            weight = w[dX[0]][0] * w[dX[1]][1]
            offset_x = (dX - fx) / inv_dx
            grid_v[base + dX] += weight * (particle_mass * vel[i] + affine @ offset_x)
            grid_m[base + dX] += weight * particle_mass
            grid_c[base + dX] += weight * col[i]


@ti.kernel
def grid_step():
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] = grid_v[I] / grid_m[I]  # change this grid_v as well
            grid_v[I][1] += gravity * dt
            grid_c[I] = grid_c[I] * particle_mass / grid_m[I]
            # boundary condition
            for d in ti.static(range(DIM)):
                cond = (I[d] < buffer and grid_v[I][d] < 0) \
                       or (I[d] > grid_size - buffer and grid_v[I][d] > 0)
                if cond:
                    grid_v[I][d] = 0
        # ================= compute loss ====================
        loss[None] += ((target[I] - grid_c[I]) ** 2).sum() / (grid_size ** 2)
        # ===================================================

@ti.kernel
def g2p():
    for i in range(num_particles):
        base = (pos[i] * inv_dx - 0.5).cast(int)
        fx = pos[i] * inv_dx - base.cast(float)
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
            new_v += weight * grid_v[base + dX]
            # =====================================
            #new_c += weight * grid_c[base + dX]
            # =====================================
            new_A += 4 * weight * grid_v[base + dX].outer_product(offset_X) * inv_dx
        vel[i] = new_v
        aff[i] = new_A
        J[i] *= 1 + dt * aff[i].trace()
        # =====================================
        #col[i] = new_c
        # =====================================


def main():
    gui = ti.GUI('Fluid', 768, background_color=0x112F41)
    gui_g1 = ti.GUI('grid_m', grid_size, background_color=0x000000)
    gui_g2 = ti.GUI('grid_c', grid_size, background_color = 0x000000)
    init()
    frame = 0
    while True:
        gui.clear(0x112F41)
        for _ in range(20):
            if frame < 100:
                optimize()
            else:
                update()
        # =======================================================
        print(loss[None])
        col_np = (col.to_numpy() * 255).astype(int)
        col_np = col_np[:, 0] * 65536 + col_np[:, 1] * 256 + col_np[:, 2]

        gui.circles(pos.to_numpy() / domain_size, radius=2, color=col_np)
        # =======================================================

        #gui.circles(pos.to_numpy() / domain_size, radius=2, color=0x068587)

        grid_m_np = grid_m.to_numpy() / particle_mass / 3
        gui.show()
        gui_g1.set_image(grid_m_np)
        gui_g2.set_image(grid_c)
        gui_g1.show()
        gui_g2.show()
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 's':
                ti.imwrite(grid_m_np, "grid_m.png")
        frame += 1


if __name__ == '__main__':
    main()