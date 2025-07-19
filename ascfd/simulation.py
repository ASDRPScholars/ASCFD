from ascfd.grid import Grid2D
from ascfd.constants import Constants
import ascfd.ics as ics
import glob
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import copy

import sys
from ascfd.flux import Flux


from ascfd.euler import Euler

from ascfd.bcs import BoundaryConditions

import numpy as np
import matplotlib.pyplot as plt
import os
class Simulation:
    def __init__(self, a_inputs):
        self.inp = a_inputs
        self.c = Constants(a_inputs)
        

        self.grid = Grid2D(self.inp.xlim, self.inp.ylim, self.inp.nx, self.inp.ny, self.inp.numghosts, self.c.NUMQ)
        self.flux = Flux(self.c, self.inp.flux)

        self.applyICS()

        self.bcs.apply_bcs()
        self.grid.check_grid(self.c)
        #setup initial time to be the starting time from the inputs file.
        #The starting timestep will always be 0.
        self.t = self.inp.t0
        self.timestepNum = 0

 
        #-1 is no output. Always output ICs if we are outputting.
        if self.inp.output_freq >= 0:
            self.output()

    def run(self):
        while (self.t < self.inp.t_finish) and self.timestepNum < self.inp.nt:
            print(f"Timestep: {self.timestepNum}, Current time: {self.t:.5e}")


            self.grid.assert_variable_type("prim")
            
            # Parameters
            dt = self.inp.dt if hasattr(self.inp, "dt") else 1e-6 #fixed small dt or small timestep

            density - self.grid.grid[self.c.RHOCOMP]
            vx = self.grid.grid[self.c.UCOMP]
            vy = self.grid.grid[self.c.VCOMP]
            pressure = self.grid.grid[self.c.PCOMP]

            #Electric field, Magnetic field, collision frequency
            Ex = getattr(self.inp, "Ex", 0.0)
            Ey = getattr(self.inp, "Ey", 0.0)
            Bz = getattr(self.inp, "Bz", 0.1)
            collision_freq = getattr(self.inp, "collision_freq", 1e5)

            #Ionization Parameters
            k_ion = getattr(self.inp, "k_ion", 1e-14) #ionization rate coefficient
            nn = getattr(self.inp, "nn", 1e19) #neutral density (m^-3)
            
            #Ionization rate 
            S_ion = k_ion * nn * density #rate of ion creation

            #Update ion density
            density += dt * S_ion 

            #Lorentz Force + E-field + collisions 
            q_i = 1.0 #unit ion charge

            JxB_x = q_i * density * vx * Bz
            JxB_y = -q_i * density * vx * Bz

            Eforce_x = q_i * density * Ex
            Eforce_y = q_i * density * Ey

            collision_drag_x = -collision_freq * density * vx #applies collisional momentum drag
            collision_drag_y = -collision_freq * density * vy

              # Momentum update, no convective derivative
            self.grid.grid[self.c.UCOMP] += dt * (Eforce_x + JxB_x + collision_drag_x) / density
            self.grid.grid[self.c.VCOMP] += dt * (Eforce_y + JxB_y + collision_drag_y) / density

            self.bcs.apply_bcs()
            self.timestepNum += 1
            self.t += dt

            if (self.timestepNum % self.inp.output_freq == 0) or (self.t >= self.inp.t_finish):
                self.output()

            self.grid.check_grid(self.c)

        print("Simulation completed successfully.")
           
    def applyICS(self):
        # default uniform conditions suitable for Hall thruster test
        density_ic = 1e18    # m^-3
        vx_ic = 0.0
        vy_ic = 0.0
        pressure_ic = 0.1

        X, Y = np.meshgrid(self.grid.x, self.grid.y, indexing='ij')

        self.grid.grid[self.c.RHOCOMP, :, :] = density_ic
        self.grid.grid[self.c.UCOMP, :, :] = vx_ic
        self.grid.grid[self.c.VCOMP, :, :] = vy_ic
        self.grid.grid[self.c.PCOMP, :, :] = pressure_ic

        self.grid.variables = "prim"

    def output(self):
        frames_dir = os.path.join(self.inp.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        output_plotname = os.path.join(frames_dir, f"output_{str(self.timestepNum).zfill(6)}.png")

        fig, axs = plt.subplots(1, 3, figsize=(18, 5))
        titles = ["Density", "X-Velocity", "Y-Velocity"]
        comps = [self.c.RHOCOMP, self.c.UCOMP, self.c.VCOMP]

        for ax, comp, title in zip(axs, comps, titles):
            data = self.grid.grid[comp, self.grid.Nghost:-self.grid.Nghost, self.grid.Nghost:-self.grid.Nghost].T
            extent = [self.grid.x[self.grid.Nghost], self.grid.x[-self.grid.Nghost - 1],
                      self.grid.y[self.grid.Nghost], self.grid.y[-self.grid.Nghost - 1]]
            im = ax.imshow(data, origin='lower', extent=extent, aspect='auto')
            plt.colorbar(im, ax=ax)
            ax.set_title(f"{title} at t={self.t:.5e}")
            ax.set_xlabel("x")
            ax.set_ylabel("y")

        plt.tight_layout()
        fig.savefig(output_plotname)
        plt.close()
       


        # if self.c.NUMQ == 3:
        #     fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        # elif self.c.NUMQ == 4:
        #     fig, axs = plt.subplots(2,2, figsize=(10, 15))
        # else:
        #     print("self.c.NUMQ: ", self.c.NUMQ)
        #     raise RuntimeError("System not implemented for movie.")
        
        # axs = axs.ravel()  # Flatten the array to index by i




        # def update_plot(file):
        #     data = np.loadtxt(file, delimiter=',', skiprows=2)
        #     x = data[:, 0]

        #     #data is indexed by
        #     # data[:,i] where i = 0 for x, i = 1 for icomp1, i=2 for icomp2

        #     with open(file, 'r') as f:
        #         lines = f.readlines()
        #         time_line = lines[0]
        #         time = float(time_line.split(':')[1].strip())

        #     timestep = int(file.split('_')[-1].split('.')[0])


        #     for i in range(self.c.NUMQ):
        #         axs[i].clear()
                

        #         axs[i].scatter(x, data[:,i+1], c="black")
        #         axs[i].set_ylabel(self.c.variable_names[i])


        #     axs[0].set_title(f"Time: {time:.4f}, Timestep: {timestep}")
        
        # # Create an animation by updating the plot for each output file
        # ani = animation.FuncAnimation(fig, update_plot, frames=output_files, repeat=False)

        # # Save the animation as a movie file using ffmpeg
        # movie_filename = os.path.join(self.inp.output_dir, "simulation_movie.mp4")
        # ani.save(movie_filename, writer='ffmpeg', fps=10)

        # print(f"Movie saved as {movie_filename}")
