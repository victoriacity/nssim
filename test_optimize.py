'''
Particle coloring test using differentiable P2G
'''
import numpy as np
import taichi as ti

DIM = 2

ti.init(arch=ti.cuda)
num_particles = 32768
grid_size = 256
inv_dx = grid_size

lr = 5

# positions of particles
pos = ti.Vector.field(DIM, dtype=ti.f32, shape=num_particles, needs_grad=True)
# color of particles
col = ti.Vector.field(3, dtype=ti.f32, shape=num_particles, needs_grad=True)
 
# grid properties
grid_m = ti.field(dtype = ti.f32, shape=(grid_size, grid_size))
grid_c = ti.Vector.field(3, dtype=ti.f32, shape=(grid_size, grid_size), needs_grad=True)

# target image
target = ti.Vector.field(3, dtype=ti.f32, shape=(grid_size, grid_size))

# loss
loss = ti.field(dtype=ti.f32, shape=(), needs_grad=True)

img = ti.imread("starry_256.png")

def init():
    pos.from_numpy(np.random.rand(num_particles, DIM).astype(np.float32))
    col.from_numpy(np.random.rand(num_particles, 3).astype(np.float32))
    target.from_numpy(img / 255)

''' update particle state '''
def substep():
    # clear grid
    grid_m.fill(0)
    grid_c.fill(0)
    with ti.Tape(loss): # loss is always cleared when entering the tape
        p2g()
        grid_step()
    #optimize_grid()
    grad_step()
    #g2p()


@ti.kernel
def grad_step():
    for i in col:
        col[i] -= lr * col.grad[i]
        pos[i] -= lr * 0.2 * pos.grad[i]
        col[i] = max(0, col[i]) # prevent negative colors
    #for I in ti.grouped(grid_c):
    #    grid_c[I] -= lr * grid_c.grad[I]
    
@ti.kernel
def optimize_grid():
    for I in ti.grouped(grid_c):
        grid_c[I] -= lr * grid_c.grad[I]   

@ti.kernel
def p2g():
    for i in range(num_particles):
        base = (pos[i] * inv_dx - 0.5).cast(int)
        fx = pos[i] * inv_dx - base.cast(float)
        # quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        for dX in ti.static(ti.ndrange(3, 3)):
            weight = w[dX[0]][0] * w[dX[1]][1]
            grid_c[base + dX] += weight * col[i]
            grid_m[base + dX] += weight
        # regularization of color intensity
        loss[None] += (col[i] ** 2).sum() / num_particles * 0.05

@ti.kernel
def grid_step():
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            loss[None] += ((target[I] - grid_c[I]) ** 2).sum() / grid_size ** 2
            #grid_c[I] = grid_c[I] / grid_m[I]

            

@ti.kernel
def g2p():
    for i in range(num_particles):
        base = (pos[i] * inv_dx - 0.5).cast(int)
        fx = pos[i] * inv_dx - base.cast(float)
        # quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_c = ti.Vector.zero(float, 3)
        for dX in ti.static(ti.ndrange(3, 3)):
            weight = w[dX[0]][0] * w[dX[1]][1]
            new_c += weight * grid_c[base + dX]
        # clamp color
        col[i] = new_c #min(max(0, new_c), 1)

  

def main():
    gui = ti.GUI('Fluid', 512, background_color = 0x112F41)
    gui_g1 = ti.GUI('grid_c', grid_size, background_color = 0x000000)
    gui_g2 = ti.GUI('grid_m', grid_size, background_color = 0x000000)
    init()
    while True:
        gui.clear(0x112F41)
        for _ in range(10):
            substep()
        print(loss[None])
        col_np = (col.to_numpy() * 255).astype(int)
        col_np = col_np[:, 0] * 65536 + col_np[:, 1] * 256 + col_np[:, 2]
        
        gui.circles(pos.to_numpy(), radius=2, color=col_np)
        
        gui.show()
        gui_g1.set_image(grid_c.to_numpy())
        gui_g1.show()
        gui_g2.set_image(grid_m.to_numpy())
        gui_g2.show()
            
        
if __name__ == '__main__':
    main()
