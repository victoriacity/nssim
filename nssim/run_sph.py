'''
Runs a real-time stylized fluid simulation 
using two MPI ranks
'''
import numpy as np
from mpi4py import MPI
import matplotlib.pyplot as plt

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
nproc = comm.Get_size()


gridsize = 192
num_particles = 7000
img_src = "../fire_192.png"

if nproc > 2:
    print("2 MPI ranks are required. usage: mpiexec -np 2 python run.py")
    exit()

print("init", rank)

if rank == 0:
    '''
    Rank 0 performs fluid simulation in Taichi
    '''
    import taichi as ti
    
    from skimage.transform import resize
    from simulator import Simulator
    from simulator_sph import SPH_Simulator
    #sim = Simulator("cuda", grid_size=gridsize, 
    #    num_particles=num_particles, dt=1e-4)
    sim = SPH_Simulator("cuda",
        num_particles=num_particles, dt=4e-4)
    print("grid_size =", sim.grid_size)
    sim.initialize()
    gui = ti.GUI('Fluid', 512, background_color=0x00000)
    res = 384
    gui_g1 = ti.GUI('Target image', gridsize, background_color=0x000000)
    target_img = ti.imread(img_src).astype(np.float32) / 255 # initial color field
    tag = 0
    
    ready = True # the ready flag controls whether to send a new input to styler
    request_recv = None
    sim.set_target(target_img)
    sim.col.fill(1)

    while True:

        gui.clear(0x000000)
        if sim.frame < 100:
            sim.optimize()
        else:
            sim.step()
        col_np = (sim.col.to_numpy() * 255).astype(int)
        col_np = col_np[:, 0] * 65536 + col_np[:, 1] * 256 + col_np[:, 2]

        gui.circles(sim.pos.to_numpy() / sim.domain_size, radius=2.5, color=col_np)
        
        gui_g1.set_image(sim.target.to_numpy())
        gui_g1.show()
        if 600 < sim.frame <= 696:
            gui.show("frames/frame%d.png" % (sim.frame - 600))
        else:
            gui.show()

        '''
        Exchange MPI messages before a step. In the READY state,
        an image is available in TARGET_IMG which can be set as the
        optimization target of the simulation. After setting the 
        target image the current density field is sent to the stylization process.
        '''
        if ready: 
            if sim.frame > 160:
                sim.set_target(target_img)
                target_img = np.zeros((gridsize, gridsize, 3)).astype(np.float32)
                msg = sim.density_field()
                request_send = comm.Isend(msg, dest=1, tag=tag)
                tag += 1
                request_send.wait() # continue after the message is sent
                request_recv = comm.Irecv(target_img, source=1, tag=tag)
                tag += 1
                ready = False
        else:
            # check if the receiving has completed
            if request_recv is not None and request_recv.Test():
                ready = True
        
else:
    '''
    Rank 1 performs stylization in PyTorch
    '''
    import time
    from stylizer import Stylizer
    tag = 0
    cont_img = np.zeros((gridsize, gridsize)).astype(np.float32) 
    sty = Stylizer("cpu:0")
    sty.load_style(img_src)
    while True:
        # the stylization process first receives the inpyt
        request_recv = comm.Irecv(cont_img, source=0, tag=tag)
        request_recv.wait()
        tag += 1
        
        # performs stylization
        sty.load_content(cont_img)
        result_img = sty.run()

        # sends the result to simulation process
        request_send = comm.Isend(result_img, dest=0, tag=tag)
        request_send.wait()
        tag += 1
