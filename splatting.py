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
num_particles = 1
# positions of particles
pos = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
# color of particles
col = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
camera = Camera()

''' directional light '''
light = ti.Vector([1, 1, 1]).normalized()

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
    for i in range(camera.res[0]):
        for j in range(camera.res[1]):
            z_buffer[i, j] = float("inf")
            frag_w[i, j] = 0

    # first pass: visibility splatting
    for i in range(num_particles):

        # particle center coordinate transfer
        tmp = camera.L2W @ position(pos[i]) # particle position view space 4d homogeneous coord
        pos_view_space[i] = ti.Vector([tmp[0]/tmp[3], tmp[1]/tmp[3], tmp[2]/tmp[3]]) # divide by homogeneous coord to obtain 3d view space position (x, y, z)
        pos_img_space[i] = camera.uncook(pos_view_space[i]) # 2d image space position (x, y) in pixel unit

        # find the projected radius in image space
        ref_view_space = ti.Vector([pos_view_space[i][0] + particle_radius, pos_view_space[i][1], pos_view_space[i][2]])
        ref_img_space = camera.uncook(ref_view_space)
        r_projected[i] = abs(ref_img_space[0] - pos_img_space[i][0]) # projected radius in pixel unit

        # process projected fragments and compute depth
        for row in range(int(pos_img_space[i][1] - r_projected[i]), int(pos_img_space[i][1] + r_projected[i])):
            for column in range(int(pos_img_space[i][0] - r_projected[i]), int(pos_img_space[i][0] + r_projected[i])):
                # discard fragment if its distance to particle center > projected radius
                frag_view_space = camera.cook(ti.Vector([row, column, pos_view_space[i][2]])) # 3d position in view space
                dis_projected = (frag_view_space - pos_view_space[i]).norm()
                if (dis_projected > particle_radius): continue
                # compute depth value for valid fragment
                depth = pos_view_space[i][2] - ti.sqrt(particle_radius ** 2 - dis_projected ** 2)
                z = depth + epsilon # epsilon shifted depth
                # overwrite if closer
                if (z < z_buffer[row, column]): z_buffer[row, column] = z

    # second pass: attribute blending
    for i in range(num_particles):
        # compute weights for each fragment
        for row in range(int(pos_img_space[i][1] - r_projected[i]), int(pos_img_space[i][1] + r_projected[i])):
            for column in range(int(pos_img_space[i][0] - r_projected[i]), int(pos_img_space[i][0] + r_projected[i])):

                frag_view_space = camera.cook(ti.Vector([row, column, pos_view_space[i][2]]))  # 3d position in view space
                dis_projected = (frag_view_space - pos_view_space[i]).norm()

                if (dis_projected > particle_radius): continue
                w1 = 1 - dis_projected / r_projected[i]
                depth = (pos_view_space[i][2] - ti.sqrt(particle_radius ** 2 - dis_projected ** 2))
                w2 = (z_buffer[row, column] - depth) / epsilon

                weight = w1 * w2
                frag_w[row, column] += weight # sum weights
                frag_c[row, column] += weight * col[i] # blend color
                normal = ti.normalized(ti.Vector([frag_view_space[0], frag_view_space[1], depth]) - pos_view_space[i])
                frag_n[row, column] += weight * normal # blend normal

    # third pass: shading
    for i in range(camera.res[0]):
        for j in range(camera.res[1]):
            if (frag_w[i, j] > 0):
                normal = ti.normalized(frag_n[i, j])
                color = frag_c[i, j] / frag_w[i, j]
                #camera.img[i, j] = shade_simple(color, normal)
                camera.img[i, j] = shade_flat(color)
                print("!")
            else:
                camera.img[i, j] = bkg_color


def shade_simple(color, normal):
    fac = normal.dot(-light)
    if fac < 0: fac = 0
    diffuse = color * fac
    ambient = color * .2
    return diffuse + ambient

def shade_flat(color):
    return color

''' Debugging Tools '''
# initialize fields
def init():
    for i in range(num_particles):
        pos[i] = ti.Vector([0, 0, 1])
        col[i] = ti.Vector([0, 0, 1]) # all blue
    return

if __name__ == '__main__':

    init()

    # render frame
    render(1.0, 1.0)

    # display img
    gui = ti.GUI('Camera View', camera.res[0], background_color=0x000000)
    #image = (camera.img.to_numpy() * 255).astype(int)
    #image = image[:, 0] * 65536 + image[:, 1] * 256 + image[:, 2]
    gui.set_image(camera.img)
    while True:
        gui.show()