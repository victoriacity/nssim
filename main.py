'''
Runs a real-time stylized fluid simulation 
using two MPI ranks
'''
import numpy as np
from mpi4py import MPI
from skimage.transform import resize
import argparse


TAG_START = 32768

def parse_args():
    parser = argparse.ArgumentParser(description='Stylized fluid simulation')
    parser.add_argument('style_img', type=str, nargs="+", help="Path to style images")
    parser.add_argument('-d', '--dim', type=int, default=3, choices=[2, 3],
        help="Dimensions")
    parser.add_argument('-m', '--method', type=str, default="mpm", choices=["sph", "mpm"],
        help="Simulation method")
    parser.add_argument('-n', '--npart', type=int, default=-1,
        help="Number of particles")
    parser.add_argument('-g', '--gridsize', type=int, default=-1,
        help="Grid size")
    parser.add_argument('-dt', type=float, default=-1,
        help="Time step")
    parser.add_argument('--lr', type=float, default=-1, help="Base learning rate")
    parser.add_argument('--res', type=int, default=768, help="Window resolution")
    parser.add_argument('--views', type=int, default=8, 
        help="Number of views for differentiable rendering (3D only)")
    parser.add_argument('--viewres', type=int, default=256, 
        help="Views resolution for differentiable rendering (3D only)")
    parser.add_argument('--warmup', type=int, default=20, 
        help="Number of warmup frames")
    args = parser.parse_args()
    # dimension-dependent defaults
    if args.npart == -1:
        args.npart = 8192 if args.dim == 2 else 32768
    if args.gridsize == -1:
        args.gridsize = 192 if args.dim == 2 else 64
    if args.dt == -1:
        args.dt = 3e-4 if args.dim == 3 else 1e-4
        if args.method == "sph":
            args.dt *= 4
    if args.lr == -1:
        args.lr = 0.2 if args.dim == 3 else 0.05
    if args.dim == 2:
        args.viewres = args.gridsize
    return args

if __name__ == "__main__":

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    nproc = comm.Get_size()

    args = parse_args()

    if nproc != 2:
        print("2 MPI ranks are required. usage: mpiexec -np 2 python run.py")
        exit()

    if rank == 0:
        '''
        Rank 0 performs fluid simulation in Taichi
        '''
        import taichi as ti
        

        if args.dim == 3:
            if args.method == "mpm":
                from nssim import MPMSimulator3D
                sim = MPMSimulator3D("cuda", grid_size=args.gridsize, 
                    num_particles=args.npart, dt=args.dt, lr=args.lr)
            else:
                raise NotImplementedError("3D SPH will be implemented in the future")
            from nssim import ParticleSplattingRenderer
            renderer = ParticleSplattingRenderer(sim, 
                main_res=args.res, n_views=args.views, sty_res=args.viewres)
        else:
            if args.method == "mpm":
                from nssim import MPMSimulator2D
                sim = MPMSimulator2D("cuda", grid_size=args.gridsize, 
                    num_particles=args.npart, dt=args.dt, lr=args.lr)
            else:
                from nssim import SPHSimulator2D
                sim = SPHSimulator2D("cuda", grid_size=args.gridsize, 
                    num_particles=args.npart, dt=args.dt, lr=args.lr)
            from nssim import CanvasRenderer
            renderer = CanvasRenderer(sim, main_res=args.res)
        sim.initialize()
        gui = ti.GUI('Fluid', args.res, background_color=0x00000)

        target_img = ti.imread(args.style_img[0]).astype(np.float32) / 255 # initial color field
        # resize image if necessary
        if target_img.shape[0] != args.viewres:
            target_img = resize(target_img, (args.viewres, args.viewres), anti_aliasing=True)
        if args.dim == 3:
            target_img = np.tile(target_img, (args.views, 1, 1, 1))
        main_tag = TAG_START # tag to send/recv stylized images
        switch_tag = 1 # tag to switch images
        
        ready = True # the ready flag controls whether to send a new input to styler
        paused = False 
        # keyboard response may take a few frames, a flag is used to avoid repetitive events
        in_event = False

        request_recv = None
        sim.set_target(target_img)
        # initialize particle color to the mean of target image
        sim.col.fill(np.mean(target_img.reshape(-1, 3), axis=0).tolist())


        while gui.running:

            gui.get_event()
            gui.running = not gui.is_pressed(ti.GUI.ESCAPE)
            if not paused:
                if 2 < sim.frame < args.warmup:
                    sim.optimize()
                else:
                    sim.step()
                '''
                Exchange MPI messages before a step. In the READY state,
                an image is available in TARGET_IMG which can be set as the
                optimization target of the simulation. After setting the 
                target image the current density field is sent to the stylization process.
                '''
                if ready: 
                    if sim.frame > args.warmup:
                        sim.set_target(target_img)
                        if args.dim == 3:
                            target_img = np.zeros((args.views, args.viewres, args.viewres, 3)).astype(np.float32)
                        else:
                            target_img = np.zeros((args.viewres, args.viewres, 3)).astype(np.float32)
                        msg = sim.get_fields()
                        request_send = comm.Isend(msg, dest=1, tag=main_tag)
                        main_tag += 1
                        request_send.wait() # continue after the message is sent
                        request_recv = comm.Irecv(target_img, source=1, tag=main_tag)
                        main_tag += 1
                        ready = False
                else:
                    # check if the receiving has completed
                    if request_recv is not None and request_recv.Test():
                        ready = True
            # process GUI events
            if gui.is_pressed(ti.GUI.SPACE):
                if not in_event:
                    in_event = True
                    paused = not paused
            elif gui.is_pressed(ti.GUI.RETURN):
                if not in_event:
                    in_event = True
                    # send a message to rank 1 to switch image
                    request_switch = comm.Isend(np.zeros((1)), dest=1, tag=switch_tag)
                    switch_tag += 1
            else:
                in_event = False

            renderer.render(gui)
            gui.show()
        
        # send exit signal
        request_exit = comm.Isend(np.zeros((1)), dest=1, tag=0)
        request_exit.wait()
            
    else:
        '''
        Rank 1 performs stylization in PyTorch
        '''
        from nssim import Stylizer
        imgid = 0
        main_tag = TAG_START
        switch_tag = 1
        if args.dim == 3:
            cont_img = np.zeros((args.views, args.viewres, args.viewres)).astype(np.float32)
        else:
            cont_img = np.zeros((args.viewres, args.viewres)).astype(np.float32)
        sty = Stylizer("cpu", args.viewres)
        sty.load_style(args.style_img[imgid])
        a = np.zeros((1))
        request_exit = comm.Irecv(a, source=0, tag=0)
        request_switch = comm.Irecv(a, source=0, tag=switch_tag) # value in buffer does not matter
        while not request_exit.Test():
            # whether to switch style image
            if request_switch.Test():
                imgid = (imgid + 1) % len(args.style_img)
                sty.load_style(args.style_img[imgid])
                switch_tag += 1
                request_switch = comm.Irecv(a, source=0, tag=switch_tag)

            # the stylization process first receives the inpyt
            request_recv = comm.Irecv(cont_img, source=0, tag=main_tag)
            request_recv.wait()
            main_tag += 1
            
            # performs stylization
            sty.load_content(cont_img)
            result_img = sty.run()

            # sends the result to simulation process
            request_send = comm.Isend(result_img, dest=0, tag=main_tag)
            request_send.wait()
            main_tag += 1

    print("Rank %d exits successfully." % rank)



