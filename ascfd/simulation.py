from ascfd.fluid.constants import FluidConstants
from ascfd.particle.constants import ParticleConstants
from ascfd.fluid.bcs import FluidBoundaryConditions
from ascfd.params import SpeciesParams
from ascfd.fluid.species import FluidSpecies
from ascfd.particle.species import ParticleSpecies
from ascfd.inputs import Inputs
from ascfd.fields.fields import Fields

import numpy as np
import matplotlib.pyplot as plt
import os
import subprocess

class Simulation:
    def __init__(self, a_inputs: Inputs):
        self.inp = a_inputs
        self.c = FluidConstants(a_inputs)
        self.pc = ParticleConstants()
        self.fields = Fields(self.inp)
        
        # xenon species parameters
        e_params = SpeciesParams(-1.6e-19, 9.1e-31, 5/3, "e", density=1e18, temperature=1.0)  # electrons
        xe_i_params = SpeciesParams(1.6e-19, 2.18e-25, 5/3, "i", density=1e18, temperature=1.0)  # Xe+ ions
        xe_n_params = SpeciesParams(0.0, 2.18e-25, 5/3, "n", density=1e20, temperature=1.0)  # Xe neutrals
        
        self.electrons = FluidSpecies(e_params, self.inp, self.fields)
        
        if self.inp.particle_ics is not None:
            self.neutrals = ParticleSpecies(xe_n_params, self.inp, self.fields)
            self.ions = ParticleSpecies(xe_i_params, self.inp, self.fields)
            
            # simulation reference so species can access each other
            self.neutrals.set_simulation(self)
            self.ions.set_simulation(self)
            
            self.all_species = [self.electrons, self.neutrals, self.ions]
        else:
            self.all_species = [self.electrons]
        
        # setup initial time to be the starting time from the inputs file.
        self.t = self.inp.t0
        self.timestep = 0
        self.dt = self.get_dt()
        
        # Set timestep for all species
        for species in self.all_species:
            # species.dt = self.dt
            species.dt = 1.283e-8
        
        # -1 is no output. Always output ICs if we are outputting.
        if self.inp.output_freq >= 0:
            self.output()
        

    def run(self):
        # !TEMP!
        while (self.t < self.inp.t_finish) and self.timestep < self.inp.nt:
            print("\033[1m" + f"Timestep: {self.timestep}, Current time: {self.t}" + "\033[0m")
            
            if self.inp.timeStepper == "RK1":
                self.electrons.update()
                
                # new_particles = []
                
                # for species in self.all_species:
                #     new_particles_from_species = species.update()
                    # if new_particles_from_species:
                    #     new_particles.extend(new_particles_from_species)
                
                # for particle in new_particles:
                #     if hasattr(particle, '__len__') and len(particle) == self.pc.NUMQ + 1:
                #         particle_charge = particle[self.pc.NUMQ - 1] if hasattr(self.pc, 'CHARGE') else 0.0
                #         particle_mass = getattr(particle, 'mass', None)
                        
                #         if hasattr(self, 'ions') and particle_charge > 0:
                #             self.ions.add_particle(particle)
                #         elif hasattr(self, 'electrons') and particle_charge < 0:
                #             pass
                #         elif hasattr(self, 'neutrals') and abs(particle_charge) < 1e-20:
                #             self.neutrals.add_particle(particle)      
                
                # self.electrons.add_particles(new_particles)
                
            else:
                raise ValueError(f"Unknown time stepper: {self.inp.timeStepper}")
            
            self.t += self.dt
            self.timestep += 1
            
            if self.inp.output_freq > 0 and self.timestep % self.inp.output_freq == 0:
                self.output()
        
        # if self.inp.output_freq >= 0:
        #     self.output()
            
        # # !TEMP!
        # self.output()        
        print(f"\nSimulation completed at time {self.t} after {self.timestep} timesteps")

        if self.inp.make_movie:
            self.generate_movie()
        
        
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
        dt = min(self.inp.cfl * min(self.inp.dx, self.inp.dy) / max_speed, self.inp.t_finish - self.t)
        
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

            for i in range(self.inp.ng, self.inp.nx - self.inp.ng):
                for j in range(self.inp.ng, self.inp.ny - self.inp.ng):
                    x = self.inp.grid_x[i]
                    y = self.inp.grid_y[j]
                    components = [self.electrons.grid[q, i, j] for q in range(self.c.NUMQ)]
                    f.write(f"{x:.12f}, {y:.12f}, " + ", ".join(f"{comp:.8f}" for comp in components) + "\n")
                    
        if self.inp.system == "euler2d":
            fig, axs = plt.subplots(2, 4, figsize=(35, 15))
        elif self.inp.system == "mhd2d":
            fig, axs = plt.subplots(2, 4, figsize=(36, 18))
        axs = axs.ravel()  # Flatten the array to index by i
        
        for q in range(self.c.NUMQ):
            # Exclude ghost cells from the plot
            # plot_data = self.electrons.grid[q, self.inp.ng*5:-self.inp.ng*5, self.inp.ng*5:-self.inp.ng*5].T # TODO: why transpose?
            plot_data = np.flipud(self.electrons.grid[q, self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]) # TODO: why transpose?
            # plot_data = self.electrons.grid[q, :, :].T
            
            # !TEMP!
            # extent = [self.inp.grid_x[self.inp.ng*5], self.inp.grid_y[-self.inp.ng*5-1],
            #           self.inp.grid_y[self.inp.ng*5], self.inp.grid_y[-self.inp.ng*5-1]]
            extent = [self.inp.grid_x[self.inp.ng], self.inp.grid_y[-self.inp.ng-1],
                      self.inp.grid_y[self.inp.ng], self.inp.grid_y[-self.inp.ng-1]]

            im = axs[q].imshow(plot_data, origin='lower', extent=extent, cmap='magma')
            
            plt.colorbar(im, ax=axs[q])
            
            # if self.inp.particle_ics is not None:
            #     axs[q].scatter(self.ions.particles[self.pc.XCOMP], self.ions.particles[self.pc.YCOMP], s=50, color='blue')
            
            axs[q].set_title(self.c.variable_names[q])
            axs[q].set_xlabel('x')
            axs[q].set_ylabel('y')
            
            
        im = axs[4].imshow(self.fields.E[:, :, 0], cmap='coolwarm')
        plt.colorbar(im, ax=axs[4])
        axs[4].set_title("electric field x")
        
        im = axs[5].imshow(self.fields.E[:, :, 1], cmap='coolwarm')
        plt.colorbar(im, ax=axs[5])
        axs[5].set_title("electric field y")
        
        im = axs[6].imshow(np.sqrt(self.fields.E[:, :, 0] ** 2 +self.fields.E[:, :, 1]), cmap='coolwarm')
        plt.colorbar(im, ax=axs[6])
        axs[6].set_title("electric field magnitude")
        
        im = axs[7].imshow(self.fields.B[:, :, 1], cmap='coolwarm')
        plt.colorbar(im, ax=axs[7])
        axs[7].set_title("magnetic field y")
        

        fig.suptitle(f"Time: {self.t:.4f}, Timestep: {self.timestep}")
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


    def generate_movie(self):
        frames_directory = os.path.join(self.inp.output_dir, "frames")

        if not os.path.exists(frames_directory):
            os.makedirs(frames_directory)

        
        test_ffmpeg_command = [
            "ffmpeg",
            "-version"
        ]

        try:
            subprocess.run(test_ffmpeg_command, check=True)
            print("ffmpeg installed correctly!")
        except subprocess.CalledProcessError as error:
            print("[SIMULATION.PY]: ffmpeg is not installed or not found in system PATH")
            print("Please install ffmpeg or ensure it is accessible from command line")
            
            return


        formatted_ffmpeg_command = [
            "ffmpeg",
            "-y",
            "-framerate",
            "24",
            "-i",
            os.path.join(frames_directory, "output_%06d.png"),
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            os.path.join(self.inp.output_dir, "movie.mp4")
        ]

        try:
            subprocess.run(formatted_ffmpeg_command, check=True)
            print("MOVIE successfully made!")
        except subprocess.CalledProcessError as error:
            print("[SIMULATION.PY]: FAILED to make movie")
            if error.stderr:
                print(error.stderr.decode())


