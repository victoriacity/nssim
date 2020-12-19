'''
Minimialist perspective camera module based on taichi-three.
'''
import numpy as np
import taichi as ti
import math

'''
Homogeneous coordinate helpers
'''
@ti.pyfunc
def position(x): return ti.Vector([x.x, x.y, x.z, 1.])
@ti.pyfunc
def direction(x): return ti.Vector([x.x, x.y, x.z, 0.])
@ti.pyfunc
def xyz(h): return ti.Vector([h[0], h[1], h[2]])

def make_homogeneous(linear, offset):
    if not isinstance(linear, ti.Matrix) and not isinstance(linear, np.ndarray):
        linear = ti.Matrix(linear)
    return ti.Matrix([
            [linear[0, 0], linear[0, 1], linear[0, 2], offset[0]],
            [linear[1, 0], linear[1, 1], linear[1, 2], offset[1]],
            [linear[2, 0], linear[2, 1], linear[2, 2], offset[2]],
            [           0,            0,            0,         1],
           ])


@ti.data_oriented
class Camera:

    def __init__(self, res=None, fov=30, *args, **kwargs):
        self.res = res or (512, 512)
        self.L2W = ti.Matrix.field(4, 4, float, ())
        self.W2V = ti.Matrix.field(4, 4, float, ())
        self.intrinsic = ti.Matrix.field(3, 3, float, ())
        self.fov = math.radians(fov)
        self.buffers = []
        self.add_buffer("img", dim=3, dtype=float)

        minres = min(self.res)
        self.cx = self.res[0] / 2
        self.cy = self.res[1] / 2
        self.fx = minres / (2 * math.tan(self.fov))
        self.fy = minres / (2 * math.tan(self.fov))

        self.ctl = CameraCtl(**kwargs)

        @ti.materialize_callback
        def init_camera_ctl():
            self.ctl.apply(self)

        @ti.materialize_callback
        @ti.kernel
        def init_intrinsic():
            self.intrinsic[None][0, 0] = self.fx
            self.intrinsic[None][0, 2] = self.cx
            self.intrinsic[None][1, 1] = self.fy
            self.intrinsic[None][1, 2] = self.cy
            self.intrinsic[None][2, 2] = 1.0

    def add_buffer(self, name, dim=3, dtype=float):
        if not dim:
            buffer = ti.field(dtype, self.res)
        else:
            buffer = ti.Vector.field(dim, dtype, self.res)
        setattr(self, name, buffer)
        self.buffers.append(buffer)

    def from_mouse(self, gui):
        changed = self.ctl.from_mouse(gui)
        if changed:
            self.ctl.apply(self)
        return changed

    @ti.func
    def cook(self, pos, translate=True):
        pos[0] *= abs(pos[2])
        pos[1] *= abs(pos[2])
        pos = self.intrinsic[None].inverse() @ pos
        return pos
    
    @ti.func
    def uncook(self, pos, translate: ti.template() = True):
        if ti.static(translate):
            pos = self.intrinsic[None] @ pos
        else:
            pos[0] *= self.intrinsic[None][0, 0]
            pos[1] *= self.intrinsic[None][1, 1]
        pos[0] /= abs(pos[2])
        pos[1] /= abs(pos[2])
        return ti.Vector([pos[0], pos[1]])

    @ti.func
    def pixel_ray(self, X):
        uv = self.cook(ti.Vector([X.x, X.y, 1]).cast(float))
        ray_orig = xyz((self.L2W[None] @ ti.Vector([0, 0, 0, 1])))
        ray_dir = uv.normalized()
        ray_dir = xyz((self.L2W[None] @ direction(ray_dir)))
        return ray_orig, ray_dir


    ''' 
    Unit Converter
    '''

    def toPixelUnit(self, x):  # x is in world unit
        return x * self.res / (self.fx * math.tan(self.fov))

    def toWorldUnit(self, x):  # x is in pixel unit
        return x * self.fx * math.tan(self.fov) / self.res

'''
Grouped camera array for higher performance
'''
@ti.data_oriented
class CameraArray:

    def __init__(self, n, res=None, fov=30, pos=None, target=None):
        self.n = n
        self.res = res or (512, 512)
        self.L2W = ti.Matrix.field(4, 4, float, (n,))
        self.W2V = ti.Matrix.field(4, 4, float, (n,))
        # Intrinsic matrix is the same for all cameras
        self.intrinsic = ti.Matrix.field(3, 3, float, ())
        self.fov = math.radians(fov)
        self.buffers = []
        self.add_buffer("img", dim=3, dtype=float)
        
        self.pos_np = pos if pos is not None else np.zeros((n, 3))
        self.target_np = target if target is not None else np.zeros((n, 3)) + np.array([[0, 0, 3]])
        self.up_np = np.zeros((n, 3)) + np.array([[0, 1, 0]])

        minres = min(self.res)
        self.cx = self.res[0] / 2
        self.cy = self.res[1] / 2
        self.fx = minres / (2 * math.tan(self.fov))
        self.fy = minres / (2 * math.tan(self.fov))

        @ti.materialize_callback
        def init_pos():
            self.set()

        @ti.materialize_callback
        @ti.kernel
        def init_intrinsic():
            self.intrinsic[None][0, 0] = self.fx
            self.intrinsic[None][0, 2] = self.cx
            self.intrinsic[None][1, 1] = self.fy
            self.intrinsic[None][1, 2] = self.cy
            self.intrinsic[None][2, 2] = 1.0

    def add_buffer(self, name, dim=3, dtype=float):
        if not dim:
            buffer = ti.field(dtype, (self.n, *self.res), needs_grad=True)
        else:
            buffer = ti.Vector.field(dim, dtype, (self.n, *self.res), needs_grad=True)
        setattr(self, name, buffer)
        self.buffers.append(buffer)

    @ti.func
    def cook(self, pos, translate=True):
        pos[0] *= abs(pos[2])
        pos[1] *= abs(pos[2])
        pos = self.intrinsic[None].inverse() @ pos
        return pos
    
    @ti.func
    def uncook(self, pos, translate: ti.template() = True):
        if ti.static(translate):
            pos = self.intrinsic[None] @ pos
        else:
            pos[0] *= self.intrinsic[None][0, 0]
            pos[1] *= self.intrinsic[None][1, 1]
        pos[0] /= abs(pos[2])
        pos[1] /= abs(pos[2])
        return ti.Vector([pos[0], pos[1]])

    @ti.func
    def pixel_ray(self, i, X):
        coor = ti.Vector([(X.x - self.cx) / self.fx, (X.y - self.cy) / self.fy])
        uv = coor * self.fov
        ray_orig = xyz((self.L2W[i] @ ti.Vector([0, 0, 0, 1])))
        ray_dir = ti.Vector([uv.x, uv.y, 1]).normalized()
        ray_dir = xyz((self.L2W[i] @ direction(ray_dir)))
        return ray_orig, ray_dir

    def set(self, pos=None, target=None, up=None):
        pos = pos or self.pos_np
        target = target or self.target_np
        up = up or self.up_np

        # right-hand coordinates
        # -Z axis points FROM the camera TOWARDS the scene
        
        fwd = (target - pos) / np.linalg.norm(target - pos, axis=1, keepdims=True)  # -Z
        right = np.cross(fwd, up) # +X
        right /= np.linalg.norm(right, axis=1, keepdims=True)
        up = np.cross(right, fwd) # +Y
        up /= np.linalg.norm(up, axis=1, keepdims=True)
        self.trans = np.transpose(np.stack([right, up, fwd], axis=1), (0, 2, 1))
        self.pos_np = pos
        self.target_np = target
        for i in range(self.n):
            self.L2W[i] = make_homogeneous(self.trans[i], self.pos_np[i])
        
    ''' 
    Unit Converter
    '''

    def toPixelUnit(self, x):  # x is in world unit
        return x * self.res / (self.fx * math.tan(self.fov))

    def toWorldUnit(self, x):  # x is in pixel unit
        return x * self.fx * math.tan(self.fov) / self.res

class CameraCtl:
    def __init__(self, pos=None, target=None, up=None):
        # python scope camera transformations
        self.pos = pos or [0, 0, 3]
        self.target = target or [0, 0, 0]
        self.up = up or [0, 1, 0]
        self.trans = None
        self.set()
        # mouse position for camera control
        self.mpos = (0, 0)

    def from_mouse(self, gui):
        changed = False
        is_alter_move = gui.is_pressed(ti.GUI.CTRL)
        if gui.is_pressed(ti.GUI.LMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.orbit((mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]),
                    pov=is_alter_move)
            self.mpos = mpos
            changed = True
        elif gui.is_pressed(ti.GUI.RMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.zoom_by_mouse(mpos, (mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]),
                        dolly=is_alter_move)
            self.mpos = mpos
            changed = True
        elif gui.is_pressed(ti.GUI.MMB):
            mpos = gui.get_cursor_pos()
            if self.mpos != (0, 0):
                self.pan((mpos[0] - self.mpos[0], mpos[1] - self.mpos[1]))
            self.mpos = mpos
            changed = True
        else:
            if gui.event and gui.event.key == ti.GUI.WHEEL:
                # one mouse wheel unit is (0, 120)
                self.zoom(-gui.event.delta[1] / 1200,
                    dolly=is_alter_move)
                gui.event = None
                changed = True
            mpos = (0, 0)
        self.mpos = mpos
        return changed

    def set(self, pos=None, target=None, up=None):
        pos = ti.Vector(pos or self.pos)
        target = ti.Vector(target or self.target)
        # right-hand coordinates
        # -Z axis points FROM the camera TOWARDS the scene
        up = ti.Vector(up or self.up)      # +Y
        fwd = (target - pos).normalized()  # -Z
        right = fwd.cross(up).normalized() # +X
        up = right.cross(fwd)              # +Y
        trans = ti.Matrix([right.entries, up.entries, fwd.entries]).transpose()
        self.trans = [[trans[i, j] for j in range(3)] for i in range(3)]
        self.pos = pos.entries
        self.target = target.entries

    def apply(self, camera):
        camera.L2W[None] = make_homogeneous(self.trans, self.pos)

    def orbit(self, delta, sensitivity=2.75, pov=False):
        ds, dt = delta
        if ds != 0 or dt != 0:
            dis = math.sqrt(sum((self.target[i] - self.pos[i]) ** 2 for i in range(3)))
            ds, dt = ds * sensitivity, dt * sensitivity
            newdir = ti.Vector([ds, dt, 1]).normalized()
            newdir = [sum(self.trans[i][j] * newdir[j] for j in range(3))\
                        for i in range(3)]
            if pov:
                newtarget = [self.pos[i] + dis * newdir[i] for i in range(3)]
                self.set(target=newtarget)
            else:
                newpos = [self.target[i] - dis * newdir[i] for i in range(3)]
                self.set(pos=newpos)

    def zoom_by_mouse(self, pos, delta, sensitivity=1.75, dolly=False):
        ds, dt = delta
        if ds != 0 or dt != 0:
            z = math.sqrt(ds ** 2 + dt ** 2) * sensitivity
            if (pos[0] - 0.5) * ds + (pos[1] - 0.5) * dt > 0:
                z *= -1
            self.zoom(z, dolly)
    
    def zoom(self, z, dolly=False):
        newpos = [(1 + z) * self.pos[i] - z * self.target[i] for i in range(3)]
        focus = sum(newpos[i] - self.target[i] for i in range(3))
        if focus < 1:
            dolly = True
        if dolly:
            newtarget = [z * self.pos[i] + (1 - z) * self.target[i] for i in range(3)]
            self.set(pos=newpos, target=newtarget)
        else:
            self.set(pos=newpos)

    def pan(self, delta, sensitivity=1.25):
        ds, dt = delta
        if ds != 0 or dt != 0:
            dis = math.sqrt(sum((self.target[i] - self.pos[i]) ** 2 for i in range(3)))
            ds, dt = ds * sensitivity, dt * sensitivity
            newdir = ti.Vector([-ds, -dt, 1]).normalized()
            newdir = [sum(self.trans[i][j] * newdir[j] for j in range(3))\
                        for i in range(3)]
            newtarget = [self.pos[i] + dis * newdir[i] for i in range(3)]
            newpos = [self.pos[i] + newtarget[i] - self.target[i] for i in range(3)]
            self.set(pos=newpos, target=newtarget)