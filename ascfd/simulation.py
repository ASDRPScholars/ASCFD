from ascfd.constants import Constants
from ascfd.bcs import BoundaryConditions
from ascfd.species.params import SpeciesParams
from ascfd.species.fluid import FluidSpecies
from ascfd.species.particles import ParticleSpecies
from ascfd.inputs import Inputs
from ascfd.fields import Fields

import numpy as np
import matplotlib.pyplot as plt
import os

class Simulation:
    def __init__(self, a_inputs: Inputs):
        self.inp = a_inputs
        self.c = Constants(a_inputs)
        
        self.fields = Fields(self.inp)
        
        # TODO: verify params
        e_params = SpeciesParams(-1.6e-19, 9.1e-31, 5/3, "e")
        xe_i_params = SpeciesParams(1.6e-19, 2.18e-25, 5/3, "xe_i")
        xe_n_params = SpeciesParams(0, 2.18e-25, 5/3, "xe_n")
        
        self.electrons = FluidSpecies(e_params, self.inp, self.fields)
        self.neutrals = ParticleSpecies(xe_n_params, self.inp, self.fields)
        self.ions = ParticleSpecies(xe_i_params, self.inp, self.fields)
        
        # loop through this list if you need to do something to all 3 species
        self.all_species = [self.electrons, self.neutrals, self.ions]
        
        #setup initial time to be the starting time from the inputs file.
        #The starting timestep will always be 0.
        
        self.t = self.inp.t0
        self.timestep = 0
        
        self.dt = self.get_dt()
        
        # TODO: use diff timestep for particles?
        for species in self.all_species:
            species.dt = self.dt
 
        #-1 is no output. Always output ICs if we are outputting.
        if self.inp.output_freq >= 0:
            self.output()
        

    def run(self):
        while (self.t < self.inp.t_finish) and self.timestep < self.inp.nt:
            print("\033[1m" + f"Timestep: {self.timestep}, Current time: {self.t}" + "\033[0m")
            
            if self.inp.timeStepper == "RK1":
                self.electrons.update()
                self.ions.update()
                self.neutrals.update()
                
            else:
                raise RuntimeError("Timestepping method not supported.")

            # assert np.all(np.isfinite(self.electrons.grid)), f"Invalid values in grid at timestep {self.timestep}"
            # assert np.all(self.electrons.grid[self.c.PCOMP] > 0), f"Negative pressure detected at timestep {self.timestep}"

            self.timestep += 1
            self.t += self.dt
            
            #always output the last timestep.
            if (self.timestep % self.inp.output_freq == 0) or (self.timestep == self.inp.nt-1):
                self.output()
 
            self.electrons.check_grid(self.c)

        if self.inp.make_movie:
            self.generate_movie()
    
        print("SUCCESS!")
        return self.grid
        
        
    def get_dt(self):
        # TODO: consider particles + fields as well when calculating dt
        
        # Determine timestep dt based on CFL condition
        if self.inp.system == "euler2d":
            density = self.electrons.grid[self.c.RHOCOMP]
            pressure = self.electrons.grid[self.c.PCOMP]
            u = self.electrons.grid[self.c.UCOMP]
            v = self.electrons.grid[self.c.VCOMP]
            # Ensure pressure and density are positive before sqrt
            pressure = np.maximum(pressure, 1e-12)
            density = np.maximum(density, 1e-12)
            a = np.sqrt(self.c.gamma * pressure / density) # Sound speed
            max_speed_x = np.max(np.abs(u) + a)
            max_speed_y = np.max(np.abs(v) + a)
            max_speed = max(max_speed_x, max_speed_y) # More robust estimate
        
        elif self.inp.system == "mhd2d":
            density = self.electrons.grid[self.c.RHOCOMP]
            pressure = self.electrons.grid[self.c.PCOMP]
            u = self.electrons.grid[self.c.UCOMP]
            v = self.electrons.grid[self.c.VCOMP]
            Bx = self.electrons.grid[self.c.BXCOMP]
            By = self.electrons.grid[self.c.BYCOMP]
            
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
        dt = min(self.inp.cfl * min(self.electrons.dx, self.electrons.dy) / max_speed, self.inp.t_finish - self.t)
        
        if dt <= 0: 
            raise ValueError(f"Calculated dt is zero or negative ({dt}). Check simulation parameters or state.")
        
        return dt
        

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
            data_dir, f"output_{str(self.timestep).zfill(6)}.txt")
        output_plotname = os.path.join(
            frames_dir, f"output_{str(self.timestep).zfill(6)}.png")

        with open(output_filename, 'w') as f:
            f.write(f"# Time: {self.t:.4f}\n")
            f.write("# x, y, density, x-velocity, y-velocity, pressure\n")

            for i in range(self.inp.numghosts, self.inp.nx - self.inp.numghosts):
                for j in range(self.inp.numghosts, self.inp.ny - self.inp.numghosts):
                    x = self.electrons.grid_x[i]
                    y = self.electrons.grid_y[j]
                    components = [self.electrons.grid[q, i, j] for q in range(self.c.NUMQ)]
                    f.write(f"{x:.12f}, {y:.12f}, " + ", ".join(f"{comp:.8f}" for comp in components) + "\n")

        if self.inp.system == "euler2d":
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        elif self.inp.system == "mhd2d":
            fig, axs = plt.subplots(2, 4, figsize=(36, 18))
        axs = axs.ravel()  # Flatten the array to index by i
        
        for q in range(self.c.NUMQ):
            # Exclude ghost cells from the plot
            plot_data = self.electrons.grid[q, self.inp.numghosts:-self.inp.numghosts, self.inp.numghosts:-self.inp.numghosts].T # TODO: why transpose?
            # plot_data = self.electrons.grid[q, :, :].T
            
            extent = [self.electrons.grid_x[self.inp.numghosts], self.electrons.grid_y[-self.inp.numghosts-1],
                      self.electrons.grid_y[self.inp.numghosts], self.electrons.grid_y[-self.inp.numghosts-1]]

            im = axs[q].imshow(plot_data, origin='lower', extent=extent, cmap='magma')
            
            plt.colorbar(im, ax=axs[q])
            
            axs[q].set_title(self.c.variable_names[q])
            axs[q].set_xlabel('x')
            axs[q].set_ylabel('y')

        fig.suptitle(f"Time: {self.t:.4f}, Timestep: {self.timestep}")
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
