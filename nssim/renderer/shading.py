import taichi as ti
import math

EPS = 1e-4
roughness = 0.2
metallic = 0.0
specular = 0.04
ambient = 0.1

floor_color_a = ti.Vector([1.0, 1.0, 1.0])
floor_color_b = ti.Vector([0.6, 0.6, 0.6])
floor_delta = 0.1

@ti.func
def brdf_cooktorrance(color, normal, lightdir, viewdir):
    halfway = (viewdir + lightdir).normalized()
    ndotv = max(viewdir.dot(normal), EPS)
    ndotl = max(lightdir.dot(normal), EPS)
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
    ggx /= (normal.dot(halfway)**2 * (alpha**2 - 1.0) + 1.0) ** 2
    return ggx

'''
Fresnel-Schlick approximation
'''
@ti.func
def frensel(view, halfway, color):
    specular_vec = ti.Vector([specular] * 3)
    f0 = specular_vec * (1 - metallic) + color * metallic
    hdotv = min(1, max(halfway.dot(view), 0))
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
    costheta = max(0, normal.dot(lightdir))
    radiance = 4
    l_out = ambient * color
    if costheta > 0:
        l_out += brdf_cooktorrance(color, normal, lightdir, viewdir)\
                * costheta * radiance
    return l_out

@ti.func
def shade_simple(color, normal, lightdir, viewdir):
    fac = max(0, normal.dot(lightdir))
    diffuse = color * fac
    ambient = color * 0.2
    return diffuse + ambient

@ti.func
def shade_flat(color, normal, lightdir, viewdir):
    return color


@ti.func
def sample_sky(rayorig, raydir):
    l = max(0, raydir[1])
    return ti.Vector([0.1, 0.6, 0.95]) * (1 - l) ** 3 + ti.Vector([0.98, 0.98, 1]) * (1 - (1 - l) ** 3)

@ti.func
def sample_floor(floor_height, rayorig, raydir, light):
    raylength = (floor_height - rayorig[1]) / raydir[1]
    intersect = rayorig + raylength * raydir
    # checkerboard texture
    tex_idx = int(ti.floor(intersect[0] / floor_delta) + ti.floor(intersect[2] / floor_delta))
    floor_color = floor_color_b
    if tex_idx % 2 == 0:
        floor_color = floor_color_a
    col = shade_cooktorrance(floor_color, ti.Vector([0.0, 1.0, 0.0]), light, -raydir)
    return col, intersect
