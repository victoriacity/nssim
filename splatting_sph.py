'''
Particle splatting Renderer

Given the positions & colors of particles + camera parameters
Return the 2D image
'''

import taichi as ti
from camera import *

ti.init(arch=ti.cuda)

''' simulation parameters'''
num_particles = 2744
DIM = 3
dt = 0.0001

gravity = -50

# interaction radius
K_smoothingRadius = 0.2

# stiffness
K_stiff = 100 # stiffness
K_stiffN = 200 # stiffness near

# rest density
K_restDensity = 21952

restitution = -0.5

# domain scale (0, 0) - (domain_size, domain_size)
# used to convert positions into canvas coordinates
domain_size = 1
dam_length = 0.5

''' Fields '''
pos = ti.Vector.field(DIM, dtype = ti.f32, shape = num_particles) # current position
oldPos = ti.Vector.field(DIM, dtype = ti.f32, shape = num_particles) # old position
vel = ti.Vector.field(DIM, dtype = ti.f32, shape = num_particles)
col = ti.Vector.field(3, dtype = ti.i32, shape=num_particles)
dens = ti.field(dtype = ti.f32, shape=num_particles) # density
densN = ti.field(dtype = ti.f32, shape=num_particles) # near density
press = ti.field(dtype = ti.f32, shape=num_particles) # density
pressN = ti.field(dtype = ti.f32, shape=num_particles) # near density

# pairs
max_pairs = 128 # max neighbor of one particle
pair = ti.field(dtype=ti.i32, shape=(num_particles - 1, max_pairs))
dist = ti.field(dtype=ti.f32, shape=(num_particles - 1, max_pairs)) # store distance for each pair
num_pair = ti.field(dtype=ti.i32, shape=(num_particles - 1,)) # number of pairs

# Eularian grid
grid_size = int(domain_size / K_smoothingRadius) # The grid has size (grid_size, grid_size)
grid_v = ti.Vector.field(DIM, dtype = ti.f32, shape = (grid_size, grid_size)) # grid to store P2G attributes
grid_m = ti.field(dtype = ti.f32, shape = (grid_size, grid_size)) # grid to store the sum of weights


'''
Rendering parameters
'''
col = ti.Vector.field(3, dtype=ti.f32, shape=num_particles)
camera = Camera(pos=[0.5, 0.5, 2], target=[0.5, 0.5, 0.5])

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
        col[i] = ti.Vector([0.2, 0.45, 1])
    print("init density:", num_particles / (0.5 * domain_size) ** 3)


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
    for i in ti.static(range(DIM)):
        if pos[index][i] < 0:
            pos[index][i] = 0
            vel[index][i] *= restitution
        elif pos[index][i] > domain_size:
            pos[index][i] = domain_size
            vel[index][i] *= restitution


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
        
        # fragment ranges to render
        xmin = int(min(max(0, pos_img_space[i][0] - r_projected[i]), camera.res[0]))
        xmax = int(min(max(0, pos_img_space[i][0] + r_projected[i]), camera.res[0]))
        ymin = int(min(max(0, pos_img_space[i][1] - r_projected[i]), camera.res[1]))
        ymax = int(min(max(0, pos_img_space[i][1] + r_projected[i]), camera.res[1]))
        if xmin > xmax or ymin > ymax:
            continue

        # process projected fragments and compute depth
        for row in range(xmin, xmax):
            for column in range(ymin, ymax):
                #if row < 0 or row >= camera.res[0] or column < 0 or column >= camera.res[1]:
                #    continue
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
        # fragment ranges to render
        xmin = int(min(max(0, pos_img_space[i][0] - r_projected[i]), camera.res[0]))
        xmax = int(min(max(0, pos_img_space[i][0] + r_projected[i]), camera.res[0]))
        ymin = int(min(max(0, pos_img_space[i][1] - r_projected[i]), camera.res[1]))
        ymax = int(min(max(0, pos_img_space[i][1] + r_projected[i]), camera.res[1]))
        if xmin > xmax or ymin > ymax:
            continue

        # compute weights for each fragment
        for row in range(xmin, xmax):
            for column in range(ymin, ymax):
                #if row < 0 or row >= camera.res[0] or column < 0 or column >= camera.res[1]:
                 #   continue
                frag_view_space = ti.Vector([row, column, pos_view_space[i][2]]).cast(float)
                frag_view_space = camera.cook(frag_view_space)  # 3d position in view space

                dis_projected = (frag_view_space - pos_view_space[i]).norm() # view space
                dis_img_space = (ti.Vector([row, column]) - pos_img_space[i]).norm()

                if dis_img_space > r_projected[i] - padding:
                    continue
                w1 = 1 - dis_img_space / (r_projected[i] - padding)

                depth = (pos_view_space[i][2] - ti.sqrt(particle_radius ** 2 - dis_projected ** 2))
                # updates the depth of the fragment
                frag_view_space = ti.Vector([frag_view_space[0], frag_view_space[1], depth])
                w2 = (z_buffer[row, column] - depth) / epsilon
                if w2 > 0:
                    weight = w1 * w2
                    frag_w[row, column] += weight # sum weights
                    frag_c[row, column] += weight * col[i] # blend color
                    normal = (frag_view_space - pos_view_space[i]).normalized()
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
                camera.img[I] += min(0.02, refldir[1] * 5) * ti.Vector([0.8, 0.9, 0.95])
        else:
            # use simple raycast to render background
            if viewdir[1] > 0:
                camera.img[I] = ti.Vector([0.8, 0.9, 0.95]) # sky color
            else:
                floor_color = ti.Vector([1.0, 1.0, 1.0])
                # shade floor
                camera.img[I] = shade_simple(
                    floor_color, ti.Vector([0.0, 1.0, 0.0]), 
                    -light, -viewdir)
        # gamma correction
        camera.img[I] = camera.img[I] ** (1/2.2)


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
roughness = 0.5
metallic = 0.0
specular = 0.04
ambient = 0.05

@ti.func
def brdf_cooktorrance(color, normal, lightdir, viewdir):
    halfway = (viewdir + lightdir).normalized()
    ndotv = max(ti.dot(viewdir, normal), EPS)
    ndotl = max(ti.dot(lightdir, normal), EPS)
    ndf = microfacet(normal, halfway)
    geom = geometry(ndotv, ndotl)
    f = frensel(viewdir, halfway, color)
    ks = f
    kd = 1 - ks
    kd *= 1 - metallic
    diffuse = kd * color / math.pi
    specular = ndf * geom * f / (4 * ndotv * ndotl)
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
    f0 = specular_vec * (1 - metallic) + color * metallic
    hdotv = min(1, max(ti.dot(halfway, view), 0))
    return f0 + (1.0 - f0) * (1.0 - hdotv) ** 5

'''
Smith's method with Schlick-GGX
'''
@ti.func
def geometry(ndotv, ndotl):
    k = (roughness + 1.0) ** 2 / 8
    geom = ndotv * ndotl\
        / (ndotv * (1.0 - k) + k) / (ndotl * (1.0 - k) + k)
    return max(0, geom)

@ti.func
def shade_cooktorrance(color, normal, lightdir, viewdir):
    costheta = max(0, ti.dot(normal, lightdir))
    radiance = 2
    l_out = ambient * color
    if costheta > 0:
        l_out += brdf_cooktorrance(color, normal, lightdir, viewdir)\
                * costheta * radiance
    return l_out



def main():
    gui = ti.GUI('Camera View', camera.res[0], background_color=0x000000)
    paused = False
    # keyboard response may take a few frames, a flag is used to avoid repetitive events
    in_event = False
    #gui_g2 = ti.GUI('grid_v', grid_size, background_color = 0x000000)
    init()
    radius = 0.025
    epsilon = radius * 1.5
    while gui.running:
        gui.get_event()
        gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
        if not paused:
            for _ in range(10):
                update()
        # render frame
        camera.from_mouse(gui)

        if gui.is_pressed(ti.GUI.UP):
            if not in_event:
                in_event = True
                epsilon += 0.001
                print(epsilon)
        elif gui.is_pressed(ti.GUI.DOWN):
            if not in_event:
                in_event = True
                epsilon -= 0.001
                print(epsilon)
        elif gui.is_pressed(ti.GUI.SPACE):
            if not in_event:
                in_event = True
                paused = not paused
        else:
            in_event = False
        render(radius, epsilon)
        gui.set_image(camera.img)
        gui.show()
            
        
if __name__ == '__main__':
    main()
