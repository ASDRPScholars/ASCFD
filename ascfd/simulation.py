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
import matplotlib as mpl
import os
import sys
import subprocess
import scienceplots
import cmocean

plt.rcParams['text.usetex'] = True
plt.style.use(['science','ieee'])
plt.rcParams['text.usetex'] = True  # Ensure this stays set
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Computer Modern Roman']

class Simulation:
    def __init__(self, a_inputs: Inputs):
        self.inp = a_inputs
        self.c = FluidConstants(a_inputs)
        self.pc = ParticleConstants()
        self.fields = Fields(self.inp)
        
        # Initialize plasma normalization
        from ascfd.fluid.plasma_refs import PlasmaReferences
        plasma_refs = PlasmaReferences(n0=1e18, T0=1000.0, species_mass=9.1e-31, species_charge=1.6e-19)
        
        # Normalized species parameters (dimensionless)
        
        # !APPROX! assuming m_i / m_e is only 100
        e_params = SpeciesParams(-1.0, 1.0, 5/3, "e", density=1.0, temperature=100.0)  # electrons (normalized)
        xe_i_params = SpeciesParams(1.0, 100, 5/3, "i", density=0.5, temperature=10.0)  # Xe+ ions  
        
        # TODO: 99? 100? does it make a difference?
        xe_n_params = SpeciesParams(0.0, 99, 5/3, "n", density=5.0, temperature=10.0)  # Xe neutrals
        
        self.electrons = FluidSpecies(e_params, self.inp, self.fields, self)
        
        if self.inp.particle_ics is not None:
            self.neutrals = ParticleSpecies(xe_n_params, self.inp, self.fields, self)
            self.ions = ParticleSpecies(xe_i_params, self.inp, self.fields, self)
            self.pelectrons = ParticleSpecies(e_params, self.inp, self.fields, self)

            self.electrons.pelectrons = self.pelectrons
            
            # simulation reference so species can access each other
            # self.neutrals.set_simulation(self)
            # self.ions.set_simulation(self)
            # self.pelectrons.set_simulation(self)
            
            self.all_species = [self.electrons, self.neutrals, self.ions]
        else:
            self.all_species = [self.electrons]
        
        # setup initial time to be the starting time from the inputs file.
        self.t = self.inp.t0
        self.timestep = 0
        self.dt = self.get_dt()
        # self.dt = 0.002
        
        # Set timestep for all species
        for species in self.all_species:
            species.dt = self.dt
            # species.dt = 0.002

        self.pelectrons.dt = self.dt
        
        # -1 is no output. Always output ICs if we are outputting.
        if self.inp.output_freq >= 0:
            self.output()
        

    def run(self):
        while (self.t < self.inp.t_finish) and self.timestep < self.inp.nt:
            print("do we enter run loop?")
            print("\033[1m" + f"Timestep: {self.timestep}, Current time: {self.t}" + "\033[0m")
            
            if self.inp.timeStepper == "RK1":
                new_particles = []
                
                for species in self.all_species:
                    new_particles_from_species = species.update()
                    if new_particles_from_species:
                        new_particles.extend(new_particles_from_species)
                
                print("NUMBER OF NEW PARTICLES:", len(new_particles))
                
                # Add all new particles at once (more efficient than per-particle loop)
                if new_particles:
                    # Add to ions and fluid electrons as before
                    for particle in new_particles:
                        self.ions.add_particle(particle)
                    # Add all particles to fluid electrons at once
                    self.electrons.add_particles(new_particles)   
                
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
        # Ensure output directories exist
        os.makedirs(self.inp.output_dir, exist_ok=True)

        frames_dir = os.path.join(self.inp.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        data_dir = os.path.join(self.inp.output_dir, "raw data")
        os.makedirs(data_dir, exist_ok=True)

        lineplot_dir = os.path.join(self.inp.output_dir, "1d frames")
        os.makedirs(lineplot_dir, exist_ok=True)

        # File names
        output_filename = os.path.join(data_dir, f"output_{str(self.timestep).zfill(6)}.txt")
        output_plotname = os.path.join(frames_dir, f"output_{str(self.timestep).zfill(6)}.pdf")
        output_lineplotname = os.path.join(lineplot_dir, f"output_{str(self.timestep).zfill(6)}.pdf")

        # Write raw text data
        with open(output_filename, 'w') as f:
            f.write(f"# Time: {self.t:.4f}\n")
            f.write("# x, y, density, x-velocity, y-velocity, pressure\n")
            for i in range(self.inp.ng, self.inp.nx - self.inp.ng):
                for j in range(self.inp.ng, self.inp.ny - self.inp.ng):
                    x = self.inp.grid_x[i]
                    y = self.inp.grid_y[j]
                    components = [self.electrons.grid[q, i, j] for q in range(self.c.NUMQ)]
                    f.write(f"{x:.12f}, {y:.12f}, " + ", ".join(f"{comp:.8f}" for comp in components) + "\n")

        # Set up 2D field plots
        if self.inp.system == "euler2d":
            fig, axs = plt.subplots(3, 4, figsize=(20, 7))
        elif self.inp.system == "mhd2d":
            fig, axs = plt.subplots(2, 4, figsize=(36, 18))
        axs = axs.ravel()

        # Set up 1D plots
        fig1d, axs1d = plt.subplots(1, 5, figsize=(15, 2))

        for q in range(self.c.NUMQ):
            extent = [self.inp.xlim[0], self.inp.xlim[1], self.inp.ylim[0], self.inp.ylim[1]]

            # 2D plot
            plot_data = self.electrons.grid[q, self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
            # plot_data = self.electrons.grid[q]
            im = axs[q].imshow(plot_data.T, extent=extent, origin='lower', cmap='magma')
            plt.colorbar(im, ax=axs[q])

            print("!&! PLOT SHAPE", self.c.variable_names[q], np.shape(plot_data))

            # if self.inp.particle_ics is not None:
                # np.set_printoptions(threshold=sys.maxsize)
                # print("!!SIMULATION!! p electrons x:", self.pelectrons.particles[self.pc.XCOMP])
                # print("!!SIMULATION!! p electrons y:", self.pelectrons.particles[self.pc.YCOMP])
                # print("!!SIMULATION!! p electrons u:", self.pelectrons.particles[self.pc.UCOMP])
                # print("!!SIMULATION!! p electrons v:", self.pelectrons.particles[self.pc.VCOMP])
                # print("!!SIMULATION!! p electrons w:", self.pelectrons.particles[self.pc.WCOMP])

                # axs[q].scatter(self.neutrals.particles[self.pc.XCOMP], self.neutrals.particles[self.pc.YCOMP], s=5, color='gray', alpha=0.2)
                # axs[q].scatter(self.ions.particles[self.pc.XCOMP], self.ions.particles[self.pc.YCOMP], s=5, color='blue', alpha=0.2)

            axs[q].set_xlim(self.inp.xlim)
            axs[q].set_ylim(self.inp.ylim)
            axs[q].set_title(self.c.variable_names[q], weight='bold')

        # 1D line plot (center slice)
        iy = round(self.inp.ny / 2)
        
        plot_data_1d = self.electrons.grid[0, self.inp.ng:-self.inp.ng, iy]
        axs1d[0].plot(plot_data_1d)
        axs1d[0].set_ylim(top=1.2)
        axs1d[0].set_title("Electron Density", weight='bold')
        
        energy_1d = 0.5 * self.electrons.grid[0, self.inp.ng:-self.inp.ng, iy] * (self.electrons.grid[1, self.inp.ng:-self.inp.ng, iy] ** 2 + self.electrons.grid[2, self.inp.ng:-self.inp.ng, iy] ** 2 + self.electrons.grid[3, self.inp.ng:-self.inp.ng, iy] ** 2)
        axs1d[1].plot(energy_1d)
        axs1d[1].set_title("Electron Energy (eV)", weight='bold')
        
        # Ionization frequency vs X position
        if hasattr(self, 'pelectrons') and hasattr(self.pelectrons, 'ionization_positions_x'):
            if self.pelectrons.ionization_positions_x:
                # Convert positions to grid indices and create histogram
                x_positions = np.array(self.pelectrons.ionization_positions_x)
                # Convert physical positions to grid coordinates
                x_indices = ((x_positions - self.inp.grid_x[0]) / self.inp.dx).astype(int)
                # Create histogram bins for x grid points
                x_bins = np.arange(self.inp.ng, self.inp.nx - self.inp.ng + 1)
                ionization_freq, _ = np.histogram(x_indices, bins=x_bins)
                
                # Smooth the data with a moving average
                window_size = min(5, len(ionization_freq) // 3)  # Adaptive window size
                if window_size >= 3:
                    from scipy.ndimage import gaussian_filter1d
                    ionization_freq_smooth = gaussian_filter1d(ionization_freq.astype(float), sigma=4)
                else:
                    ionization_freq_smooth = ionization_freq
                
                axs1d[2].plot(x_bins[:-1], ionization_freq_smooth)
                axs1d[2].set_title("Electron-Neutral Collision Frequency", weight='bold')
        
        # Ion density 1D cross-section
        ion_density_data = self.ions._compute_particle_density_field(self.ions)[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        ion_density_1d = ion_density_data[:, iy]
        from scipy.ndimage import gaussian_filter1d
        ion_density_1d_smooth = gaussian_filter1d(ion_density_1d.astype(float), sigma=4)
        axs1d[3].plot(ion_density_1d_smooth)
        axs1d[3].set_title("Ion Density", weight='bold')
        
        # Neutral density 1D cross-section  
        neutral_density_data = self.neutrals._compute_particle_density_field(self.neutrals)[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        neutral_density_1d = neutral_density_data[:, iy]
        neutral_density_1d_smooth = gaussian_filter1d(neutral_density_1d.astype(float), sigma=4)
        axs1d[4].plot(neutral_density_1d_smooth)
        axs1d[4].set_title("Neutral Density", weight='bold')

        #     else:
        #         axs1d[2].plot([])
        #         axs1d[2].set_title("ionization frequency vs X (no data)", weight='bold')
        #         axs1d[2].set_xlabel('x grid index')
        #         axs1d[2].set_ylabel('ionizations per timestep')
        # else:
        #     axs1d[2].plot([])
        #     axs1d[2].set_title("ionization frequency vs X (no particles)", weight='bold')
        #     axs1d[2].set_xlabel('x grid index')
        #     axs1d[2].set_ylabel('ionizations per timestep')

        im = axs[5].imshow(self.electrons.grid[0, self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng].T, extent=extent, origin='lower', cmap='magma')
        plt.colorbar(im, ax=axs[5])
        axs[5].scatter(self.ions.particles[self.pc.XCOMP], self.ions.particles[self.pc.YCOMP], s=5, color='blue', alpha=0.2, clip_on=True)
        axs[5].set_xlim(self.inp.xlim)
        axs[5].set_ylim(self.inp.ylim)
        axs[5].set_title("Ion Distribution", weight='bold')
        # print("ION DENSITY", self.ions._compute_particle_density_field(self.ions))

        # Field overlays
        im = axs[6].imshow(self.fields.E[:, :, 0].T, extent=extent, origin='lower', cmap='coolwarm')
        plt.colorbar(im, ax=axs[6])
        axs[6].set_title("Electric Field X", weight='bold')

        print("!&! E SHAPE", np.shape(self.fields.E))

        im = axs[7].imshow(self.fields.potential.T, extent=extent, origin='lower', cmap='coolwarm')
        plt.colorbar(im, ax=axs[7])
        axs[7].set_title("Electric Potential", weight='bold')

        from scipy.ndimage import gaussian_filter
        charge_density_data = self.fields.charge_density[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        charge_density_smooth = gaussian_filter(charge_density_data, sigma=3.0)
        im = axs[8].imshow(charge_density_smooth.T, extent=extent, origin='lower', cmap='coolwarm')
        plt.colorbar(im, ax=axs[8])
        axs[8].set_title("Charge Density", weight='bold')

        ion_density_data = self.ions._compute_particle_density_field(self.ions)[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        ion_density_smooth = gaussian_filter(ion_density_data, sigma=3.0)
        im = axs[9].imshow(ion_density_smooth.T, extent=extent, origin='lower', cmap=mpl.cm.Blues)
        plt.colorbar(im, ax=axs[9])
        axs[9].set_title("Ion Density", weight='bold')

        # im = axs[10].imshow(ion_density_smooth.T, extent=extent, origin='lower', cmap=mpl.cm.Blues)
        # plt.colorbar(im, ax=axs[10])
        # axs[10].scatter(self.ions.particles[self.pc.XCOMP], self.ions.particles[self.pc.YCOMP], s=5, color='blue', alpha=0.2, clip_on=True)
        # axs[10].set_title("Ion Density", weight='bold')

        # energy_data = self.electrons.euler.prim_to_cons(self.electrons.grid)[self.c.ECOMP, self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        # im = axs[10].imshow(energy_data.T, extent=extent, origin='lower', cmap='magma')
        # plt.colorbar(im, ax=axs[10])
        # axs[10].set_title("Energy", weight='bold')

        neutral_density = self.neutrals._compute_particle_density_field(self.neutrals)[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        neutral_smooth = gaussian_filter(neutral_density, sigma=3)
        im = axs[10].imshow(neutral_smooth.T, extent=extent, origin='lower', cmap=mpl.cm.Greys)
        plt.colorbar(im, ax=axs[10])
        axs[10].set_title("Neutral Density", weight='bold')

        sigma = self.electrons.pelectrons.cross_section_grid
        sigma_smooth = gaussian_filter(sigma, sigma=3)
        im = axs[11].imshow(sigma_smooth.T, extent=extent, origin='lower', cmap='coolwarm')
        plt.colorbar(im, ax=axs[11])
        axs[11].set_title("Electron Collision Cross-Sections", weight='bold')



        # Save both figures
        fig.suptitle(f"Time: {self.t:.4f}, Timestep: {self.timestep}")
        fig.tight_layout()
        fig.savefig(output_plotname)
        plt.close(fig)

        fig1d.tight_layout()
        fig1d.savefig(output_lineplotname)
        plt.close(fig1d)
       


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


