"""
Taichi SPH
"""
import numpy as np
import taichi as ti

ti.init(arch = ti.cuda)

''' parameters '''

num_particles = 2500

#delta t
dt = 0.0001

gravity = -50

# interaction radius
K_smoothingRadius = 0.02

# stiffness
K_stiff = 100 # stiffness
K_stiffN = 200 # stiffness near

# rest density
K_restDensity = 8

restitution = -0.5

# domain scale (0, 0) - (domain_size, domain_size)
# used to convert positions into canvas coordinates
domain_size = 1



''' Fields '''
# positions of particles
pos = ti.Vector.field(2, dtype = ti.f32, shape = num_particles) # current position
oldPos = ti.Vector.field(2, dtype = ti.f32, shape = num_particles) # old position

# velocities of particles
vel = ti.Vector.field(2, dtype = ti.f32, shape = num_particles)

col = ti.Vector.field(3, dtype = ti.i32, shape=num_particles)

# density of particles
dens = ti.field(dtype = ti.f32, shape=num_particles) # density
densN = ti.field(dtype = ti.f32, shape=num_particles) # near density

# pressure of particles
press = ti.field(dtype = ti.f32, shape=num_particles) # density
pressN = ti.field(dtype = ti.f32, shape=num_particles) # near density

# pairs
max_pairs = 128 # max neighbor of one particle
pair = ti.field(dtype=ti.i32, shape=(num_particles - 1, max_pairs))
dist = ti.field(dtype=ti.f32, shape=(num_particles - 1, max_pairs)) # store distance for each pair
num_pair = ti.field(dtype=ti.i32, shape=(num_particles - 1,)) # number of pairs

# Eularian grid
grid_size = int(domain_size / K_smoothingRadius) # The grid has size (grid_size, grid_size)
print("grid size:", grid_size)
grid_v = ti.Vector.field(2, dtype = ti.f32, shape = (grid_size, grid_size)) # grid to store P2G attributes
grid_w = ti.field(dtype = ti.f32, shape = (grid_size, grid_size)) # grid to store the sum of weights
#img = np.transpose(ti.imread("starry.jpg"), (1, 0, 2)).reshape(-1, 3)
#img = np.zeros((grid_size, grid_size, 3)).reshape(-1, 3) + 255
img = np.ones((num_particles, 3)) * 255
# rgb 2 hex
img = img[:, 0] * 65536 + img[:, 1] * 256 + img[:, 2]
print(img.shape)

''' initialize particle position & velocity '''
@ti.kernel
def init():

    length = ti.sqrt(num_particles)

    for i in range(num_particles):

        # place particles around the center of the domain
        pos[i] = ti.Vector([i%length/length, i//length/length]) # 0 - 1
        pos[i] = (pos[i] + 1)/2 - 0.05 + ti.random() * 0.001 # 0.25 - 0.75 (centered)
        pos[i] *= domain_size # scale to fit the domain

        oldPos[i] = pos[i]
    print("dam grid spacing:", 0.5 / (num_particles) ** (1/2) * domain_size)
    print("init density:", num_particles / (0.5 * domain_size) ** 2 )

        # initialize velocity to 0
        #vel[i] = ti.Vector([0, 0])


''' update particle state '''
@ti.kernel
def update():

    # state update
    for i in range(num_particles):
        # clear density
        dens[i] = 0
        densN[i] = 0
        num_pair[i] = 0

        # compute new velocity
        vel[i] = (pos[i] - oldPos[i])/dt

        # collision handling?
        boundary_collision(i)

        # save previous position
        oldPos[i] = pos[i]
        # apply gravity
        vel[i][1] += (gravity * dt)

    # apply viscosity
    a = 2000
    b = 40

    for i in range(num_particles - 1):
        for j in range(i + 1, num_particles):
            distance = (pos[i] - pos[j]).norm()
            q = distance / K_smoothingRadius
            if q < 1:
                r_ij = ti.normalized(pos[i] - pos[j])
                u = ti.dot((vel[i] - vel[j]), r_ij)
                if u > 0:
                    I = dt * (1-q) * (a*u + b*u*u) * r_ij
                    vel[i] -= I/2
                    vel[j] += I/2


    # position update
    for i in range(num_particles):
        # advance to new position
        pos[i] += (vel[i] * dt)

    # compute pair info
    for i in range(num_particles - 1):
        for j in range(i + 1, num_particles):
            distance = (pos[i] - pos[j]).norm()
            if distance < K_smoothingRadius:
                pair[i, num_pair[i]] = j
                dist[i, num_pair[i]] = distance
                num_pair[i] += 1


    # compute density
    for i in range(num_particles):
        dens[i] = 1
        densN[i] = 1

    for i in range(num_particles - 1):
        for j in range(num_pair[i]):
            q = 1 - dist[i, j] / K_smoothingRadius
            q2 = q * q
            q3 = q2 * q
            # print(dist[i], q)
            dens[i] += q2
            dens[pair[i, j]] += q2
            densN[i] += q3
            densN[pair[i, j]] += q3


    # update pressure
    for i in range(num_particles):
        press[i] = K_stiff * (dens[i] - K_restDensity)
        pressN[i] = K_stiffN * densN[i]


    # apply pressure
    for i in range(num_particles - 1):
        for j in range(num_pair[i]):
            p = press[i] + press[pair[i, j]]
            pN = pressN[i] + pressN[pair[i, j]]

            q = 1 - dist[i, j] / K_smoothingRadius
            q2 = q * q

            displace = (p * q + pN * q2) * (dt * dt)
            a2bN = (pos[i] - pos[pair[i, j]]) / dist[i, j]

            pos[i] += displace * a2bN
            pos[pair[i, j]] -= displace * a2bN


''' handle particle collision with boundary '''
@ti.func
def boundary_collision(index):

    # x boundary
    if pos[index][0] < 0:
        pos[index][0] = 0
        vel[index][0] *= restitution
    elif pos[index][0] > domain_size:
        pos[index][0] = domain_size
        vel[index][0] *= restitution

    # y boundary
    if pos[index][1] < 0:
        pos[index][1] = 0
        vel[index][1] *= restitution
    elif pos[index][1] > domain_size:
        pos[index][1] = domain_size
        vel[index][1] *= restitution
    
    

def main():
    gui = ti.GUI('SPH Fluid', 512, background_color = 0x000000)
    #gui_g1 = ti.GUI('grid_m', grid_size, background_color = 0x000000)
    #gui_g2 = ti.GUI('grid_v', grid_size, background_color = 0x000000)
    init()
    
    while True:
        
        gui.clear(0x000000)
        for _ in range(15):
            grid_w.fill(0)
            grid_v.fill(0)
            update()
        gui.circles(pos.to_numpy() / domain_size, radius=6, color=img)
        #for _ in range(1):
        
        # draw particle
        
        #grid_w_np = grid_w.to_numpy()
        gui.show()
        #gui_g1.set_image(grid_w_np / np.max(grid_w_np))
        #gui_g2.set_image(grid_v)
        #gui_g1.show()
        #gui_g2.show()
            
        
if __name__ == '__main__':
    main()