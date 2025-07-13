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
        self.euler = Euler(self.c)

        self.grid = Grid2D(self.inp.xlim, self.inp.ylim, self.inp.nx, self.inp.ny, self.inp.numghosts, self.c.NUMQ)
        self.bcs = BoundaryConditions(self.grid, self.inp.bcs_lo, self.inp.bcs_hi)
        self.flux = Flux(self.c, self.inp.flux)

        self.apply_ics()

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
            print(f"Timestep: {self.timestepNum}, Current time: {self.t}")

            self.bcs.apply_bcs()

            self.grid.assert_variable_type("prim")
            
            # Determine timestep dt based on CFL condition
            if self.inp.system == "euler2d":
                density = self.grid.grid[self.c.RHOCOMP]
                pressure = self.grid.grid[self.c.PCOMP]
                u = self.grid.grid[self.c.UCOMP]
                v = self.grid.grid[self.c.VCOMP]
                # Ensure pressure and density are positive before sqrt
                pressure = np.maximum(pressure, 1e-12)
                density = np.maximum(density, 1e-12)
                a = np.sqrt(self.c.gamma * pressure / density) # Sound speed
                max_speed_x = np.max(np.abs(u) + a)
                max_speed_y = np.max(np.abs(v) + a)
                max_speed = max(max_speed_x, max_speed_y) # More robust estimate
            
            elif self.inp.system == "mhd2d":
                density = self.grid.grid[self.c.RHOCOMP]
                pressure = self.grid.grid[self.c.PCOMP]
                u = self.grid.grid[self.c.UCOMP]
                v = self.grid.grid[self.c.VCOMP]
                Bx = self.grid.grid[self.c.BXCOMP]
                By = self.grid.grid[self.c.BYCOMP]
                
                # Ensure pressure and density are positive
                pressure = np.maximum(pressure, 1e-12)
                density = np.maximum(density, 1e-12)
                
                a = np.sqrt(self.c.gamma * pressure / density) # Sound speed
                # Alfven speed squared components
                ca_sq_x = Bx**2 / density
                ca_sq_y = By**2 / density
                ca_sq_tot = ca_sq_x + ca_sq_y
                
                # Fast magnetosonic speed squared (cf^2)
                # cf^2 = 0.5 * ( (a^2 + ca_tot^2) + sqrt( max( (a^2 + ca_tot^2)^2 - 4*a^2*ca_x^2 , 0.0 ) ) )
                term_under_sqrt = (a**2 + ca_sq_tot)**2 - 4 * a**2 * ca_sq_x
                cf_sq_x = 0.5 * ( (a**2 + ca_sq_tot) + np.sqrt(np.maximum(term_under_sqrt, 0.0)) )
                cf_x = np.sqrt(cf_sq_x)
                
                term_under_sqrt = (a**2 + ca_sq_tot)**2 - 4 * a**2 * ca_sq_y # Use ca_sq_y for y-direction cf
                cf_sq_y = 0.5 * ( (a**2 + ca_sq_tot) + np.sqrt(np.maximum(term_under_sqrt, 0.0)) )
                cf_y = np.sqrt(cf_sq_y)
                
                # Max signal speed is max(|u|+cf_x, |v|+cf_y)
                max_signal_x = np.max(np.abs(u) + cf_x)
                max_signal_y = np.max(np.abs(v) + cf_y)
                max_speed = max(max_signal_x, max_signal_y)
            
            else:
                raise RuntimeError(f"System {self.inp.system} not supported for dt calculation.")

            # Calculate dt, ensuring it doesn't overshoot t_finish
            dt = min(self.inp.cfl * min(self.grid.dx, self.grid.dy) / max_speed, self.inp.t_finish - self.t)
            if dt <= 0: 
                 raise ValueError(f"Calculated dt is zero or negative ({dt}). Check simulation parameters or state.")

            if  self.inp.timeStepper == "RK1":
                #returns numerical flux and conservative variables at interface
                self.grid.assert_variable_type("prim")
                consU, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus = self.flux.getFlux(self.grid.grid, self.grid.Nx, self.grid.Ny, self.grid.Nghost)

                U_new = np.copy(consU)  # Start with the current conservative variables

                #FLUID UPDATE
                for i in range(self.grid.Nghost, self.grid.Nx + self.grid.Nghost):
                    for j in range(self.grid.Nghost, self.grid.Ny + self.grid.Nghost):
                        for icomp in range(self.c.NUMQ):
                            U_new[icomp, i, j] = consU[icomp, i, j] - (
                                (dt / self.grid.dx) * (numFluxX_plus[icomp, i, j] - numFluxX_minus[icomp, i, j]) +
                                (dt / self.grid.dy) * (numFluxY_plus[icomp, i, j] - numFluxY_minus[icomp, i, j])
                            )

                #Then, update flow field based on the embedded boundary.
                
                #take a step in particles
                


                            
            else:
                raise RuntimeError("Timestepping method not supported.")

            #self.grid.plot()
            # Update the grid with the new conservative variables
            #self.grid.set(U_new)
            self.grid.grid = self.euler.cons_to_prim(U_new)
            self.grid.variables = "prim"

            # Convert back to primitive variables
            #self.grid.transform(self.euler.cons_to_prim, "prim")


            self.bcs.apply_bcs()

            # assert np.all(np.isfinite(self.grid.grid)), f"Invalid values in grid at timestep {self.timestepNum}"
            # assert np.all(self.grid.grid[self.c.PCOMP] > 0), f"Negative pressure detected at timestep {self.timestepNum}"

            self.timestepNum += 1
            self.t += dt
            
            #always output the last timestep.
            if (self.timestepNum % self.inp.output_freq == 0) or (self.timestepNum == self.inp.nt-1):
                self.output()
 
 
            # DEBUG
            # self.grid.plot()
            self.grid.check_grid(self.c)

        if self.inp.make_movie:
            self.generate_movie()
    
        print("SUCCESS!")
        return self.grid
 
    def plot(self):
        if not os.path.exists(self.inp.output_dir):
            os.makedirs(self.inp.output_dir)

        if self.inp.system == "euler2d":
            fig, axs = plt.subplots(3, 1, figsize=(10, 15))
            axs[0].scatter(self.grid.x, self.grid.grid[self.c.RHOCOMP, :], c="black")
            axs[0].set_ylabel("Density")
 
            axs[1].scatter(self.grid.x, self.grid.grid[self.c.UCOMP, :],  c="black")
            axs[1].set_ylabel("Velocity")
 
            axs[2].scatter(self.grid.x, self.grid.grid[self.c.PCOMP, :],  c="black")
            axs[2].set_ylabel("Pressure")
 
        elif self.inp.system == "mhd2d":
            fig, axs = plt.subplots(3, 1, figsize=(10, 15))
            axs[0].scatter(self.grid.x, self.grid.grid[self.c.RHOCOMP, :], c="black")
            axs[0].set_ylabel("Density")
 
            axs[1].scatter(self.grid.x, self.grid.grid[self.c.UCOMP, :],  c="black")
            axs[1].set_ylabel("Velocity")
 
            axs[2].scatter(self.grid.x, self.grid.grid[self.c.PCOMP, :],  c="black")
            axs[2].set_ylabel("Pressure")
            
            axs[3].scatter(self.grid.x, self.grid.grid[self.c.B_XCOMP, :],  c="black")
            axs[3].set_ylabel("Magnetic Field")
 
        axs[0].set_title(f"Time: {self.t:.4f}")
        plt.savefig(f"{self.inp.output_dir}/plot_dt{str(self.timestepNum).zfill(6)}")
        plt.close()
 
   
 
    def apply_ics(self):

        if self.inp.system == "euler2d":
            if self.inp.ics == "diagonal_advection":
                self.grid.fill_grid(ics.diagonal_advection_2d)
            elif self.inp.ics == "kelvin_helmholtz":
                self.grid.fill_grid(ics.kelvin_helmholtz_2d)
            elif self.inp.ics == "double_mach_reflection":
                self.grid.fill_grid(ics.double_mach_reflection_2d)
            elif self.inp.ics == "riemann_problem":
                self.grid.fill_grid(ics.riemann_2d)
            else:
                raise RuntimeError("[FLUID] ICS not valid.")
           
        elif self.inp.system == "mhd2d":
            if self.inp.ics == "orszag_tang":
                self.grid.fill_grid(ics.orszag_tang_2d)
           
        else:
            raise RuntimeError("[FLUID] ICS not valid.")

    def applyParticles(self):
        """Particle Setup"""
        # I assume this should function similar to apply ics?
        # check if self.inp.particle_ic = ... 
        pass
        

    def output(self):
        # Ensure the base output directory exists
        os.makedirs(self.inp.output_dir, exist_ok=True)

        # Ensure the frames subdirectory exists
        frames_dir = os.path.join(self.inp.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)
        
        data_dir = os.path.join(self.inp.output_dir, "raw data")
        os.makedirs(data_dir, exist_ok=True)

        # File naming convention: output_timestepNum.txt
        output_filename = os.path.join(
            data_dir, f"output_{str(self.timestepNum).zfill(6)}.txt")
        output_plotname = os.path.join(
            frames_dir, f"output_{str(self.timestepNum).zfill(6)}.png")

        with open(output_filename, 'w') as f:
            # Write header
            f.write(f"# Time: {self.t:.4f}\n")
            f.write("# x, y, density, x-velocity, y-velocity, pressure\n")

            for i in range(self.grid.Nghost, self.grid.Nx - self.grid.Nghost):
                for j in range(self.grid.Nghost, self.grid.Ny - self.grid.Nghost):
                    x = self.grid.x[i]
                    y = self.grid.y[j]
                    components = [self.grid.grid[q, i, j] for q in range(self.c.NUMQ)]
                    f.write(f"{x:.12f}, {y:.12f}, " + ", ".join(f"{comp:.8f}" for comp in components) + "\n")

        if self.inp.system == "euler2d":
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        elif self.inp.system == "mhd2d":
            fig, axs = plt.subplots(2, 4, figsize=(36, 18))
        axs = axs.ravel()  # Flatten the array to index by i
        
        for q in range(self.c.NUMQ):
            # Exclude ghost cells from the plot
            plot_data = self.grid.grid[q, self.grid.Nghost:-self.grid.Nghost, self.grid.Nghost:-self.grid.Nghost].T # TODO: why transpose?
            # plot_data = self.grid.grid[q, :, :].T
            
            extent = [self.grid.x[self.grid.Nghost], self.grid.x[-self.grid.Nghost-1],
                    self.grid.y[self.grid.Nghost], self.grid.y[-self.grid.Nghost-1]]

            im = axs[q].imshow(plot_data, origin='lower', extent=extent, cmap='magma')
            
            plt.colorbar(im, ax=axs[q])
            
            axs[q].set_title(self.c.variable_names[q])
            axs[q].set_xlabel('x')
            axs[q].set_ylabel('y')

        fig.suptitle(f"Time: {self.t:.4f}, Timestep: {self.timestepNum}")
        plt.tight_layout()
        fig.savefig(output_plotname)
        plt.close()
   
 
 
 
 
 
    def generate_movie(self):
        # Create a directory for the frames if it doesn't exist
        frames_dir = os.path.join(self.inp.output_dir, "frames")
        # if not os.path.exists(frames_dir):
        #     os.makedirs(frames_dir)

        # List all the output files and sort them
        #output_files = sorted(glob.glob(os.path.join(self.inp.output_dir, "output_*.png")))
            
        #movie_filename = os.path.join(self.inp.output_dir, "simulation_movie.mp4")


        ffmpeg_command = f"ffmpeg -y -framerate 24 -i {frames_dir}/output_%06d.png -c:v libx264 -pix_fmt yuv420p {self.inp.output_dir}/movie.mp4"
        os.system(ffmpeg_command)



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
