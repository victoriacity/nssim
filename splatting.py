'''
Particle splatting Renderer

Given the positions & colors of particles + camera parameters
Return the 2D image
'''

import taichi as ti
from camera import *
#from MPM_optimize import *

''' MPM_optimization is currently 2D '''
''' Create some place-holder fields and a camera for now '''
num_particles = 3
# positions of particles
pos = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
# color of particles
col = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
camera = Camera()

''' directional light '''
light = ti.Vector([1, -1, 4]).normalized()

''' background color '''
bkg_color = ti.Vector([0, 0, 0])

''' fields '''
z_buffer = ti.field(dtype=ti.f32, shape=camera.res)
# fragment color
frag_c = ti.Vector.field(3, dtype=ti.f32, shape=camera.res)
# fragment normal
frag_n = ti.Vector.field(3, dtype=ti.f32, shape=camera.res)
# fragment weight
frag_w = ti.field(dtype=ti.f32, shape=camera.res)

pos_view_space = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
pos_img_space = ti.Vector.field(2, dtype=ti.f32, shape=num_particles)
r_projected = ti.field(dtype=ti.f32, shape=num_particles)

#@ti.function
@ti.kernel
def render(particle_radius: ti.f32, epsilon: ti.f32): # particle_position_field, particle_color_field

    # clear buffer
    for I in ti.grouped(camera.img):
        z_buffer[I] = float("inf")
        frag_w[I] = 0
        frag_c[I].fill(0)
        frag_n[I].fill(0)
        camera.img[I].fill(0.0)

    W2V = camera.L2W.inverse()
    padding = 3 # extra padding to avoid cropping some of the projected sphere

    # first pass: visibility splatting
    for i in range(num_particles):

        # particle center coordinate transfer
        pos_view_space[i] = xyz(W2V @ position(pos[i])) # particle position view space 4d homogeneous coord [x, y, z, 1]
        pos_img_space[i] = camera.uncook(pos_view_space[i]) # 2d image space position (x, y) in pixel unit
        #print(pos_img_space[i])
        # find the projected radius in image space
        ref_view_space = ti.Vector([pos_view_space[i][0] + particle_radius, pos_view_space[i][1], pos_view_space[i][2]])
        ref_img_space = camera.uncook(ref_view_space)
        r_projected[i] = abs(ref_img_space[0] - pos_img_space[i][0]) + padding # projected radius in pixel unit

        # process projected fragments and compute depth
        for row in range(int(pos_img_space[i][0] - r_projected[i]), int(pos_img_space[i][0] + r_projected[i])):
            for column in range(int(pos_img_space[i][1] - r_projected[i]), int(pos_img_space[i][1] + r_projected[i])):
                if row < 0 or row >= camera.res[0] or column < 0 or column >= camera.res[1]:
                    continue
                # discard fragment if its distance to particle center > projected radius
                frag_view_space = ti.Vector([row, column, pos_view_space[i][2]]).cast(float)
                frag_view_space = camera.cook(frag_view_space) # 3d position in view space
                dis_projected = (frag_view_space - pos_view_space[i]).norm()
                if dis_projected > particle_radius:
                    #print("!")
                    continue
                # compute depth value for valid fragment
                depth = pos_view_space[i][2] - ti.sqrt(particle_radius ** 2 - dis_projected ** 2)
                z = depth + epsilon # epsilon shifted depth
                # overwrite if closer
                if z < z_buffer[row, column]:
                    z_buffer[row, column] = z
                    #frag_n[row, column] = ti.Vector([frag_view_space[0], frag_view_space[1], depth]) - pos_view_space[i]
                    #frag_c[row, column] = col[i]
                    #frag_w[row, column] = 1.0
                
    
    # second pass: attribute blending
    for i in range(num_particles):
        # compute weights for each fragment
        for row in range(int(pos_img_space[i][0] - r_projected[i]), int(pos_img_space[i][0] + r_projected[i])):
            for column in range(int(pos_img_space[i][1] - r_projected[i]), int(pos_img_space[i][1] + r_projected[i])):
                if row < 0 or row >= camera.res[0] or column < 0 or column >= camera.res[1]:
                    continue
                frag_view_space = ti.Vector([row, column, pos_view_space[i][2]]).cast(float)
                frag_view_space = camera.cook(frag_view_space)  # 3d position in view space

                dis_projected = (frag_view_space - pos_view_space[i]).norm() # view space

                #if dis_projected > particle_radius:
                #    continue
                #w1 = 1 - dis_projected / particle_radius

                dis_img_space = (ti.Vector([row, column]) - pos_img_space[i]).norm()
                if dis_img_space > r_projected[i]:
                    continue
                w1 = 1 - dis_img_space / r_projected[i]

                depth = (pos_view_space[i][2] - ti.sqrt(particle_radius ** 2 - dis_projected ** 2))
                w2 = (z_buffer[row, column] - depth) / epsilon
                if w2 > 0:
                    weight = w1 * w2
                    frag_w[row, column] += weight # sum weights
                    frag_c[row, column] += weight * col[i] # blend color
                    normal = ti.Vector([frag_view_space[0], frag_view_space[1], depth]) - pos_view_space[i]
                    frag_n[row, column] += weight * normal # blend normal
    
    # third pass: shading
    for i in range(camera.res[0]):
        for j in range(camera.res[1]):
            if frag_w[i, j] > 0:
                normal = frag_n[i, j].normalized()
                color = frag_c[i, j] / frag_w[i, j]
                camera.img[i, j] = shade_simple(color, normal)
                #camera.img[i, j] = shade_flat(color)
                #print(color[0], color[1], color[2])
            else:
                camera.img[i, j] = bkg_color


@ti.func
def shade_simple(color, normal):
    fac = normal.dot(-light)
    diffuse = color * fac
    ambient = color * 0.2
    return diffuse + ambient

@ti.func
def shade_flat(color):
    return color

''' Debugging Tools '''
# initialize fields
def init():
    for i in range(num_particles):
        pos[i] = ti.Vector([0.0, 0.0, (i - 1.5) / 5])
        col[i] = ti.Vector([0.0, 1 - i / 3, i / 3]) # all blue
    #print(pos.to_numpy())
    return

if __name__ == '__main__':

    init()


    # display img
    gui = ti.GUI('Camera View', camera.res[0], background_color=0x000000)
    #image = (camera.img.to_numpy() * 255).astype(int)
    #image = image[:, 0] * 65536 + image[:, 1] * 256 + image[:, 2]

    radius = 0.2
    epsilon = 0.1

    while gui.running:
        gui.get_event()
        gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
        # render frame
        camera.from_mouse(gui)
        #print(camera.ctl.pos, camera.ctl.target)
        #print(camera.ctl.trans)
        #print(pos_view_space.to_numpy())
        #print(camera.ctl.pos, camera.ctl.target, pos_view_space.to_numpy())
        if gui.is_pressed(ti.GUI.UP):
            epsilon += 0.001
            print(epsilon)
        elif gui.is_pressed(ti.GUI.DOWN):
            epsilon -= 0.001
            print(epsilon)
        render(radius, epsilon)
        gui.set_image(camera.img)
        gui.show()