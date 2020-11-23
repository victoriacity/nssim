
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import taichi as ti
import torch
from photo_wct import PhotoWCT

DIM = 2

ti.init(arch = ti.cuda)

''' parameters '''

num_particles = 8192
grid_size = 128

domain_size = 1

particle_vol = (0.5 * domain_size / grid_size) ** DIM
particle_rho = 1
particle_mass = particle_vol * particle_rho

E = 400

#delta t
dt = 0.0001

gravity = -9.8

restitution = 0

#img = np.transpose(ti.imread("starry.jpg"), (1, 0, 2)).reshape(-1, 3)
img = np.zeros((grid_size, grid_size, 3)).reshape(-1, 3) + 255
# rgb 2 hex
img = img[:, 0] * 65536 + img[:, 1] * 256 + img[:, 2]

''' Fields '''
# positions of particles
pos = ti.Vector.field(2, dtype = ti.f32, shape = num_particles) # current position

# velocities of particles
vel = ti.Vector.field(2, dtype = ti.f32, shape = num_particles)
# affine velocity field
aff = ti.Matrix.field(DIM, DIM, dtype = ti.f32, shape = num_particles)
# volume deformation
J = ti.field(dtype = ti.f32, shape = num_particles)


# Eularian grid

grid_v = ti.Vector.field(2, dtype = ti.f32, shape = (grid_size, grid_size))
grid_m = ti.field(dtype = ti.f32, shape = (grid_size, grid_size))

inv_dx = grid_size / domain_size

''' initialize particle position & velocity '''
@ti.kernel
def init():
    length = ti.sqrt(num_particles)
    for i in range(num_particles):
        # place particles around the center of the domain
        pos[i] = ti.Vector([i%length/length, i//length/length]) # 0 - 1
        pos[i] = (pos[i] + 1)/2 - 0.05 + ti.random() * 0.001 # 0.25 - 0.75 (centered)
        pos[i] *= domain_size # scale to fit the domain    
        J[i] = 1
    print("dam grid spacing:", 0.5 / (num_particles) ** (1/2) * domain_size)
    print("init density:", num_particles / (0.5 * domain_size) ** 2 )
        

''' update particle state '''
@ti.kernel
def update():
    # clear grid
    for I in ti.grouped(grid_m):
        grid_m[I] = 0
        grid_v[I].fill(0)

    p2g()
    grid_step()
    g2p() 

    # advect particles
    for i in pos:
        pos[i] += vel[i] * dt
    
        
''' helper function to compute distance between two points '''         
@ti.func
def distance(v1, v2):
    dv = v1 - v2
    dis = dv.norm()
    return dis
    
buffer = 3 # buffer grids for boundary conditions

''' Lagrangian to Eularian representation '''
@ti.func
def p2g():

    for i in range(num_particles):
        base = (pos[i] * inv_dx - 0.5).cast(int)
        fx = pos[i] * inv_dx - base.cast(float)
        # quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        stress = -dt * particle_vol * (J[i] - 1) * 4 * E * inv_dx * inv_dx
        affine = ti.Matrix.identity(float, DIM) * stress + aff[i] * particle_mass
        for dX in ti.static(ti.ndrange(3, 3)):
            weight = w[dX[0]][0] * w[dX[1]][1]
            offset_x = (dX - fx) / inv_dx
            grid_v[base + dX] += weight * (particle_mass * vel[i] + affine @ offset_x)
            grid_m[base + dX] += weight * particle_mass

@ti.func
def grid_step():
    for I in ti.grouped(grid_m):
        if grid_m[I] > 0:
            grid_v[I] = grid_v[I] / grid_m[I] # change this grid_v as well
            grid_v[I][1] += gravity * dt
            # boundary condition
            for d in ti.static(range(DIM)):
                cond = (I[d] < buffer and grid_v[I][d] < 0) \
                    or (I[d] > grid_size - buffer and grid_v[I][d] > 0)
                if cond:
                    grid_v[I][d] = restitution * grid_v[I][d]

@ti.func
def g2p():
    for i in range(num_particles):
        base = (pos[i] * inv_dx - 0.5).cast(int)
        fx = pos[i] * inv_dx - base.cast(float)
        # quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, DIM)
        new_C = ti.Matrix.zero(float, DIM, DIM)
        for dX in ti.static(ti.ndrange(3, 3)):
            offset_X = dX - fx
            weight = w[dX[0]][0] * w[dX[1]][1]
            new_v += weight * grid_v[base + dX]
            new_C += 4 * weight * grid_v[base + dX].outer_product(offset_X) * inv_dx
        vel[i] = new_v
        aff[i] = new_C
        J[i] *= 1 + dt * aff[i].trace()

    

def main():
    p_wct = PhotoWCT().cuda()
    p_wct.load_state_dict(torch.load("model.pth"))
    style_img = Image.open("starry_256.png").convert('RGB')
    style_img = transforms.ToTensor()(style_img).unsqueeze(0).cuda()

    gui = ti.GUI('Fluid', 768, background_color = 0x112F41)
    gui_g1 = ti.GUI('grid_m', grid_size, background_color = 0x000000)
    #gui_g2 = ti.GUI('grid_v', grid_size, background_color = 0x000000)
    init()
    frame = 0
    while True:
        gui.clear(0x112F41)
        for _ in range(20):
            update()
        gui.circles(pos.to_numpy() / domain_size, radius=2, color=0x068587)
        
        grid_m_np = grid_m.to_numpy() / particle_mass / 3
        
        grid_m_np = grid_m_np * 255
        if frame > 300:
            cont_img = torch.from_numpy(np.tile(grid_m_np, (3, 1, 1))).unsqueeze(0).cuda()
            with torch.no_grad():
                result_img = p_wct.transform(cont_img, style_img).squeeze(0).cpu().numpy()
                gui_g1.set_image(np.transpose(result_img, (1, 2, 0)) * (grid_m_np > 0)[..., None])

        gui.show()
        
        #gui_g2.set_image(grid_v)
        gui_g1.show()
        #gui_g2.show()
        if gui.get_event(ti.GUI.PRESS):
            if gui.event.key == 's':
                ti.imwrite(grid_m_np, "grid_m.png")
        frame += 1
            
        
if __name__ == '__main__':
    main()