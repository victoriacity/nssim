'''
Particle splatting Renderer

Given the positions & colors of particles + camera parameters
Return the 2D image
'''

import taichi as ti
from camera import *
#from MPM_optimize import *

ti.init(arch=ti.cuda)

''' simulation parameters'''
DIM = 3
num_particles = 32768
grid_size = 64
domain_size = 1
dam_length = 0.5
particle_vol = (domain_size / grid_size) ** DIM
particle_rho = 1
particle_mass = particle_vol * particle_rho
E = 400
dt = 2e-4
gravity = -9.8

pos = ti.Vector.field(DIM, dtype = ti.f32, shape = num_particles)
vel = ti.Vector.field(DIM, dtype = ti.f32, shape = num_particles)
aff = ti.Matrix.field(DIM, DIM, dtype = ti.f32, shape = num_particles)
J = ti.field(dtype = ti.f32, shape = num_particles)
grid_v = ti.Vector.field(DIM, dtype = ti.f32, shape = (grid_size, grid_size, grid_size))
grid_m = ti.field(dtype = ti.f32, shape = (grid_size, grid_size, grid_size))
inv_dx = grid_size / domain_size



'''
Rendering parameters
'''
col = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
camera = Camera()

''' directional light '''
light = ti.Vector([-1, -1, -4]).normalized()

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

''' initialize particle position & velocity '''
@ti.kernel
def init():
    n_axis = ti.ceil(num_particles ** (1/3))
    for i in range(num_particles):
        # place particles around the center of the domain
        pos[i][2] = i // n_axis ** 2
        pos[i][1] = i % n_axis ** 2 // n_axis
        pos[i][0] = i % n_axis
        pos[i] = dam_length * pos[i] / n_axis + 0.1 # put fluid block to the corner
        pos[i] *= domain_size # scale to fit the domain    
        J[i] = 1
        col[i] = ti.Vector([0.3, 0.6, 1])


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
    
        
buffer = 4 # buffer grids for boundary conditions

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
        for dX in  ti.static(ti.ndrange(3, 3, 3)):
            weight = w[dX[0]][0] * w[dX[1]][1] * w[dX[2]][2]
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
                    grid_v[I][d] = 0

@ti.func
def g2p():
    for i in range(num_particles):
        base = (pos[i] * inv_dx - 0.5).cast(int)
        fx = pos[i] * inv_dx - base.cast(float)
        # quadratic B-spline
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, DIM)
        new_C = ti.Matrix.zero(float, DIM, DIM)
        for dX in ti.static(ti.ndrange(3, 3, 3)):
            offset_X = dX - fx
            weight = w[dX[0]][0] * w[dX[1]][1] * w[dX[2]][2]
            new_v += weight * grid_v[base + dX]
            new_C += 4 * weight * grid_v[base + dX].outer_product(offset_X) * inv_dx
        vel[i] = new_v
        aff[i] = new_C
        J[i] *= 1 + dt * aff[i].trace()

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
                    continue
                # compute depth value for valid fragment
                depth = pos_view_space[i][2] - ti.sqrt(particle_radius ** 2 - dis_projected ** 2)
                z = depth + epsilon # epsilon shifted depth
                # overwrite if closer
                if z < z_buffer[row, column]:
                    z_buffer[row, column] = z
                
    
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
                dis_img_space = (ti.Vector([row, column]) - pos_img_space[i]).norm()

                if dis_img_space > r_projected[i] - padding:
                    continue
                w1 = 1 - dis_img_space / (r_projected[i] - padding)

                depth = (pos_view_space[i][2] - ti.sqrt(particle_radius ** 2 - dis_projected ** 2))
                w2 = (z_buffer[row, column] - depth) / epsilon
                if w2 > 0:
                    weight = w1 * w2
                    frag_w[row, column] += weight # sum weights
                    frag_c[row, column] += weight * col[i] # blend color
                    normal = ti.Vector([frag_view_space[0], frag_view_space[1], depth]) - pos_view_space[i]
                    normal_world = xyz(camera.L2W @ direction(normal))
                    # transform normal to world space
                    frag_n[row, column] += weight * normal_world # blend normal
    
    # third pass: shading
    for I in ti.grouped(camera.img):
        _, viewdir = camera.pixel_ray(I)
        if frag_w[I] > 0:
            #normal = #xyz(camera.L2W @ direction(frag_n[I].normalized()))
            normal = frag_n[I].normalized()
            color = frag_c[I] / frag_w[I]
            camera.img[I] = shade_cooktorrance(
                color, normal, -light, -viewdir)
            # reflection
            refldir = viewdir - 2 * viewdir.dot(normal) * normal
            if refldir[1] > 0:
                camera.img[I] += min(0.1, refldir[1] * 5) * ti.Vector([0.8, 0.9, 0.95])
        else:
            # use simple raycast to render background
            if viewdir[1] > 0:
                camera.img[I] = ti.Vector([0.8, 0.9, 0.95]) # sky color
            else:
                floor_color = ti.Vector([1.0, 1.0, 1.0])
                # shade floor
                camera.img[I] = shade_cooktorrance(
                    floor_color, ti.Vector([0.0, 1.0, 0.0]), 
                    -light, -viewdir)


@ti.func
def shade_simple(color, normal, lightdir, viewdir):
    fac = max(0, normal.dot(lightdir))
    diffuse = color * fac
    ambient = color * 0.2
    return diffuse + ambient

@ti.func
def shade_flat(color):
    return color


EPS = 1e-4
roughness = 1.0
metallic = 0.0
specular = 1.0
ks = 3.0
kd = 2.0



@ti.func
def ischlick(cost):
    k = (roughness + 1) ** 2 / 8
    return k + (1 - k) * cost

@ti.func
def fresnel(f0, HoV):
    return f0 + (1 - f0) * (1 - HoV)**5

@ti.func
def brdf_cooktorrance2(color, normal, lightdir, viewdir):
    ks = 1.0
    kd = 1.0
    halfway = (lightdir + viewdir).normalized()
    NoH = max(EPS, ti.dot(normal, halfway))
    NoL = max(EPS, ti.dot(normal, lightdir))
    NoV = max(EPS, ti.dot(normal, viewdir))
    HoV = min(1 - EPS, max(EPS, ti.dot(halfway, viewdir)))
    ndf = roughness**2 / (NoH**2 * (roughness**2 - 1) + 1)**2
    vdf = 0.25 / (ischlick(NoL) * ischlick(NoV))
    f0 = metallic * color + (1 - metallic) * 0.16 * specular**2
    fdf = fresnel(f0, NoV)
    strength = (1 - f0) * (1 - metallic) * kd * color \
        + f0 * ks * fdf * vdf * ndf / math.pi
    return strength


@ti.func
def brdf_cooktorrance(color, normal, lightdir, viewdir):
    halfway = (viewdir + lightdir).normalized()
    ndotv = max(ti.dot(viewdir, normal), EPS)
    ndotl = max(ti.dot(lightdir, normal), EPS)
    diffuse = kd * color / math.pi
    specular = microfacet(normal, halfway)\
                * frensel(viewdir, halfway, color)\
                * geometry(ndotv, ndotl)
    specular /= 4 * ndotv * ndotl
    return diffuse + specular

'''
Trowbridge-Reitz GGX microfacet distribution
'''
@ti.func
def microfacet(normal, halfway):
    alpha = roughness
    ggx = alpha ** 2 / math.pi
    ggx /= (ti.dot(normal, halfway)**2 * (alpha**2 - 1.0) + 1.0) ** 2
    return ggx

'''
Fresnel-Schlick approximation
'''
@ti.func
def frensel(view, halfway, color):
    specular_vec = ti.Vector([specular] * 3)
    f0 = specular_vec * metallic + color * (1 - metallic)
    hdotv = min(1, max(ti.dot(halfway, view), 0))
    return (f0 + (1.0 - f0) * (1.0 - hdotv) ** 5) * ks

'''
Smith's method with Schlick-GGX
'''
@ti.func
def geometry(ndotv, ndotl):
    k = (roughness + 1.0) ** 2 / 8
    geom = ndotv * ndotl\
        / (ndotv * (1.0 - k) + k) / (ndotl * (1.0 - k) + k)
    return max(0, geom)


def shade_cooktorrance(color, normal, lightdir, viewdir):
    costheta = max(0, ti.dot(normal, lightdir))
    l_out = ti.Vector([0.0, 0.0, 0.0])
    if costheta > 0:
        l_out = brdf_cooktorrance(color, normal, lightdir, viewdir)\
                * costheta
    return l_out



def main():
    gui = ti.GUI('Camera View', camera.res[0], background_color=0x000000)
    
    #gui_g2 = ti.GUI('grid_v', grid_size, background_color = 0x000000)
    init()
    radius = 0.025
    epsilon = radius * 1.5
    while gui.running:
        gui.get_event()
        gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
        for _ in range(10):
            update()
        # render frame
        camera.from_mouse(gui)
        if gui.is_pressed(ti.GUI.UP):
            epsilon += 0.001
            print(epsilon)
        elif gui.is_pressed(ti.GUI.DOWN):
            epsilon -= 0.001
            print(epsilon)
        render(radius, epsilon)
        gui.set_image(camera.img)
        gui.show()
            
        
if __name__ == '__main__':
    main()
