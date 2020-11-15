"""
Taichi SPH
"""

import taichi as ti
import math

ti.init(arch = ti.cuda)

''' parameters '''

num_particles = 256

#delta t
dt = 0.001

gravity = -10

# interaction radius
K_smoothingRadius = 0.1

# stiffness
K_stiff = 0.0
K_stiffN = 0.0

# rest density
K_restDensity = 5

restitution = -0.5

''' Fields '''
# positions of particles
pos = ti.Vector.field(2, dtype = ti.f32, shape = num_particles)

# old positions
oldPos = ti.Vector.field(2, dtype = ti.f32, shape = num_particles)

# velocities of particles
vel = ti.Vector.field(2, dtype = ti.f32, shape = num_particles)

# density of particles
dens = ti.field(dtype = ti.f32, shape = num_particles) # density
densN = ti.field(dtype = ti.f32, shape = num_particles) # near density

# pressure of particles
press = ti.field(dtype = ti.f32, shape = num_particles) # density
pressN = ti.field(dtype = ti.f32, shape = num_particles) # near density

# pairs
''' CANNOT USE COMPUTED RESULT HERE '''
#max_pairs = (1 + num_particles)*num_particles/2
max_pairs = 32896
pair = ti.Vector.field(2, dtype = ti.i32, shape = max_pairs)
dist = ti.field(dtype = ti.f32, shape = max_pairs) # store distance for each pair
num_pairs = 0


''' initialize particle position & velocity '''
@ti.kernel
def init():
    length = math.sqrt(num_particles)
    
    for i in range(num_particles):
        
        # place particles around the center (0-1) scale
        pos[i] = ti.Vector([i%length/length, i//length/length])
        pos[i] = (pos[i] + 1)/2.0 - 0.25
        
        oldPos[i] = pos[i]
        
        # initialize velocity to 0
        vel[i] = ti.Vector([0, 0])
        

''' update particle state '''
@ti.kernel
def update():
    
    for i in range(num_particles):
        # compute new velocity
        vel[i] = (pos[i] - oldPos[i])/dt
        # save previous position
        oldPos[i] = pos[i]
        # apply gravity
        vel[i][1] += (gravity * dt)
        # advance to new position
        pos[i] += (vel[i] * dt)
        # clear density
        dens[i] = 0
        densN[i] = 0
        
       
    '''
    # compute pairs
    num_pairs = 0
    for i in range(num_particles):
        for j in range(i, num_particles):
            dis = distance(pos[i], pos[j])
            if  dis < K_smoothingRadius:
                pair[num_pairs] = ti.Vector([i, j]) # indices
                dist[num_pairs] = dis # distance
                num_pairs += 1
    
    # compute density
    for i in range(num_pairs):
        q = 1 - dist[i] / K_smoothingRadius
        q2 = q * q
        q3 = q2 * q
        
        dens[pair[i][0]] += q2
        dens[pair[i][1]] += q2
        densN[pair[i][0]] += q3
        densN[pair[i][1]] += q3
        
    # update pressure
    for i in range(num_particles):
        press[i] = K_stiff * (dens[i] - K_restDensity)
        pressN[i] = K_stiffN * densN[i]
        
    # apply pressure
    for i in range(num_pairs):
        # index of particle a & b
        a = pair[i][0]
        b = pair[i][1]
        
        p = press[a] + press[b]
        pN = pressN[a] + pressN[b]
        
        q = 1 - dist[i] / K_smoothingRadius
        q2 = q * q
        
        displace = (p * q + pN * q2) * (dt * dt)
        a2bN = (pos[pair[i][1]] - pos[pair[i][0]])/dist[i]
        
        pos[a] += displace * a2bN
        pos[b] -= displace * a2bN
        
    # boundary collisoin
    
    for i in range(num_particles):
        boundary_collision(i)
    '''
    
        


''' handle particle collision with boundary '''
@ti.func
def boundary_collision(index):
        
    # x boundary
    if pos[index][0] < 0:
        pos[index][0] = 0
        vel[index][0] *= restitution
    elif pos[index][0] > 1:
        pos[index][0] = 1
        vel[index][0] *= restitution
        
    # y boundary
    if pos[index][1] < 0:
        pos[index][1] = 0
        vel[index][1] *= restitution
    elif pos[index][1] > 1:
        pos[index][1] = 1
        vel[index][1] *= restitution
        
            
@ti.func
def distance(v1, v2):
    dv = v1 - v2
    dis = ti.sqrt(dv[0]*dv[0] + dv[1]*dv[1])
    return dis
    

def main():
    gui = ti.GUI('SPH Fluid', background_color = 0xDDDDDD)
    
    init()
    
    while True:
        
        P = pos.to_numpy()
        
        update()
        
        # draw particle
        for p in P:
            gui.circle(p, color = 0x33BBFF, radius = 5)
            
        gui.show()
            
        
if __name__ == '__main__':
    main()