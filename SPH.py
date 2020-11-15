"""
Taichi SPH
"""

import taichi as ti
import math

ti.init(arch = ti.cuda)

num_particles = 256

#delta t
dt = 0.01


gravity = -10


# positions of particles
pos = ti.Vector.field(2, dtype = ti.f32, shape = num_particles)

# velocities of particles
vel = ti.Vector.field(2, dtype = ti.f32, shape = num_particles)


@ti.kernel
# initialize particle position & velocity
def init():
    length = math.sqrt(num_particles)
    
    for i in range(num_particles):
        
        # place particles around the center
        pos[i] = ti.Vector([i%length/length, i//length/length])
        pos[i] = (pos[i] + 1)/2.0 - 0.25
        
        # initialize velocity to 0
        vel[i] = ti.Vector([0, 0])
        

@ti.kernel
# update particle state
def update():
    # apply gravity
    for i in range(num_particles):
        vel[i][1] += (gravity * dt)
        pos[i] += (vel[i] * dt)
        

@ti.kernel
# handle particle collision with boundary
def boundary_collision():
    for i in range(num_particles):
    

def main():
    gui = ti.GUI('SPH Fluid', background_color = 0xDDDDDD)
    
    init()
    
    while True:
        
        P = pos.to_numpy()
        
        update()
        
        for p in P:
            gui.circle(p, color = 0x33BBFF, radius = 5)
            
        gui.show()
            
        
if __name__ == '__main__':
    main()