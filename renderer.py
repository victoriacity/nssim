import taichi as ti
import numpy as np
import matplotlib.cm
from camera import Camera

INF = 100000.0
cmap = matplotlib.cm.get_cmap('viridis')
colors = [list(cmap(x)[:3]) for x in [0.0, 0.5, 1.0]]


@ti.func
def lerp(x, y, t):
    return x * (1 - t) + y * t

@ti.func
def ray_aabb_intersection(box_min, box_max, o, d):
    intersect = 1

    near_int = -INF
    far_int = INF

    for i in ti.static(range(3)):
        if d[i] == 0:
            if o[i] < box_min[i] or o[i] > box_max[i]:
                intersect = 0
        else:
            i1 = (box_min[i] - o[i]) / d[i]
            i2 = (box_max[i] - o[i]) / d[i]

            new_far_int = ti.max(i1, i2)
            new_near_int = ti.min(i1, i2)

            far_int = ti.min(new_far_int, far_int)
            near_int = ti.max(new_near_int, near_int)

    if near_int > far_int:
        intersect = 0
    return intersect, near_int, far_int

box_min = [0, 0, 0]
box_max = [1, 1, 1]

@ti.data_oriented
class SceneVol():

    def __init__(self, res, field, camera):
        self.field = ti.field(ti.f32, shape=(res, res, res))
        self.field_np = field 
        self.res = res
        self.camera = camera

        @ti.materialize_callback
        def init_field():
            self.field.from_numpy(self.field_np)

    @ti.func
    def get(self, X, delta):
        # trilinear interpolation
        X0 = max(0, min(int(X), self.res))
        X1 = max(0, min(X0 + ti.ceil(abs(delta)), self.res))
        l = ti.Vector([0.0, 0.0, 0.0])
        for i in ti.static(range(3)):
            if X1[i] != X0[i]:
                l[i] = (X[i] - X0[i]) / (X1[i] - X0[i])
        #print(X, X0, X1, l)
        # lerp x
        c00 = lerp(self.field[X0], self.field[X1.x, X0.y, X0.z], l.x)
        c01 = lerp(self.field[X0.x, X0.y, X1.z], self.field[X1.x, X0.y, X1.z], l.x)
        c10 = lerp(self.field[X0.x, X1.y, X0.z], self.field[X1.x, X1.y, X0.z], l.x)
        c11 = lerp(self.field[X0.x, X1.y, X1.z], self.field[X1], l.x)
        # lerp y
        c0 = lerp(c00, c10, l.y)
        c1 = lerp(c10, c11, l.y)
        # lerp z
        return lerp(c0, c1, l.z)


    @ti.func
    def colormap(self, x):
        color = ti.Vector([0.0, 0.0, 0.0])
        if x > 0.5:
            color = lerp(ti.Vector(colors[1]), ti.Vector(colors[2]), (x - 0.5) * 2)
        else:
            color = lerp(ti.Vector(colors[0]), ti.Vector(colors[1]), x * 2)
        return color

    @ti.func
    def color_at(self, coor, camera):
        o, d = camera.pixel_ray(coor)
        #print(o, d)
        intersect, near_int, far_int = ray_aabb_intersection(box_min, box_max, o, d)
        color = ti.Vector([0.0, 0.0, 0.0])
        #print(o, d, intersect)
        if intersect:
            intensity = 0.0
            ind_start = (o + near_int * d) * self.res
            ind_end = (o + far_int * d) * self.res
            step = max(50, abs(ind_end - ind_start).max())
            delta = (ind_end - ind_start) / step
            for i in range(step):
                X = ind_start + i * delta
                #intensity = self.get(X, delta)
                #if intensity > 0.3:
                #    break
                #else:
                #    intensity = 0
                intensity += self.get(X, delta)
            #color = min(color, 0.99)
            intensity /= 50
            color = self.colormap(intensity)
        return color

    @ti.kernel
    def render(self):
        for X in ti.grouped(self.camera.img):
            self.camera.img[X] = self.color_at(X, camera)

ti.init(ti.cuda)

vol_field = np.load("volume.npy")
# normalized
vol_field = (vol_field - np.min(vol_field)) / (np.max(vol_field) - np.min(vol_field))
print(np.max(vol_field), np.sum(vol_field))
gridsize = vol_field.shape[0]
vol_field *= 10 # ad hoc adjustment for better look



camera = Camera(pos=[0.5, 0.5, 1], target=[0.5, 0.5, 0], fov=30, res=(1024, 1024))
scene = SceneVol(gridsize, vol_field, camera)

gui = ti.GUI('View', camera.res)
while gui.running:
    gui.get_event()
    gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
    camera.from_mouse(gui)
    scene.render()
    gui.set_image(camera.img)
    gui.show()