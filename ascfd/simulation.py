from ascfd.fluid.constants import FluidConstants
from ascfd.particle.constants import ParticleConstants
from ascfd.fluid.bcs import FluidBoundaryConditions
from ascfd.params import SpeciesParams
from ascfd.fluid.species import FluidSpecies
from ascfd.particle.species import ParticleSpecies
from ascfd.inputs import Inputs
from ascfd.fields.fields import Fields
from ascfd.plasma_refs import PlasmaReferences

import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import subprocess
import scienceplots
import cmocean
import copy

# plt.rcParams['text.usetex'] = True
# plt.style.use(['science','ieee'])
# plt.rcParams['text.usetex'] = True  # Ensure this stays set
# plt.rcParams['font.family'] = 'serif'
# plt.rcParams['font.serif'] = ['Computer Modern Roman']

class Simulation:
    def __init__(self, a_inputs: Inputs):
        self.ref = PlasmaReferences()
        self.inp = a_inputs
        self.inp = self._phys_to_normal()
        
        self.c = FluidConstants(a_inputs)
        self.pc = ParticleConstants()
        self.fields = Fields(self.inp)
        
        # TODO: move density + temperature to PARTICLE ics, don't keep in multispecies params
        xe_i_params = SpeciesParams(self.inp.q, self.inp.m_i, 5/3, "i", density=0.5, temperature=300.0) # TODO kelvin or eV # Xe+ ions  
        xe_n_params = SpeciesParams(0, self.inp.m_n, 5/3, "n", density=5.0, temperature=0)  # Xe neutrals
        e_params = SpeciesParams(-self.inp.q, self.inp.m_e, 5/3, "e", density=1.0, temperature=30000.0) # TODO kelvin or eV
        
        self.electrons = FluidSpecies(e_params, self.inp, self.fields, self)
        
        if self.inp.particle_ics is not None:
            self.neutrals = ParticleSpecies(xe_n_params, self.inp, self.fields, self)
            self.ions = ParticleSpecies(xe_i_params, self.inp, self.fields, self)
            self.pelectrons = ParticleSpecies(e_params, self.inp, self.fields, self)

            self.electrons.pelectrons = self.pelectrons
            
            self.all_species = [self.electrons, self.neutrals, self.ions]
            self.particle_species = [self.neutrals, self.ions]
        else:
            self.all_species = [self.electrons]
        
        # setup initial time to be the starting time from the inputs file.
        self.t = self.inp.t0
        self.timestep = 0
        # self.dt = self.get_dt()
        self.dt_physical = 1e-11

        self.dt_ratio = 5

        self.dt_particle = self.dt_physical * self.dt_ratio

        self.dt_norm = self.dt_physical / self.ref.dt
        self.dt_particle_norm = self.dt_particle / self.ref.dt
        
        # Set timestep for all species

        self.electrons.dt = self.dt_norm

        for species in self.particle_species :
            species.dt = self.dt_particle_norm

        # for species in self.all_species:
        #     species.dt = self.dt_norm

        # self.pelectrons.dt = self.dt_norm
        
        # -1 is no output. Always output ICs if we are outputting.
        if self.inp.output_freq >= 0:
            self.output()
        
        
    def _phys_to_normal(self) -> Inputs:
        print("CALLED PHYS TO NORMAL")
        normal_inp = copy.deepcopy(self.inp)
        
        ref = self.ref
        inp = self.inp
        
        normal_inp.m_e = inp.m_e / ref.m
        # normal_inp.m_i = inp.m_i / ref.m
        # normal_inp.m_n = inp.m_n / ref.m
        
        # INITIAL CONDITIONS
        normal_inp.rho_e = (ref.L**3/ref.m) * inp.rho_e
        normal_inp.p_e = ((ref.dt ** 2 * ref.L) / ref.m ) * inp.p_e
        # normal_inp.n_n = ref.L ** 3 * inp.n_n
        
        # TODO: add pressure/temperature normalization
        
        normal_inp.q = inp.q / ref.q
        
        normal_inp.xlim = (inp.xlim[0] / ref.L, inp.xlim[1] / ref.L)
        normal_inp.ylim = (inp.ylim[0] / ref.L, inp.ylim[1] / ref.L)
        
        normal_inp.dx = (normal_inp.xlim[1] - normal_inp.xlim[0]) / inp.nx
        normal_inp.dy = (normal_inp.ylim[1] - normal_inp.ylim[0]) / inp.ny
        
        print("normal_inp.dx", normal_inp.dx)
        
        print("normal_inp.xlim", normal_inp.xlim)
        normal_inp.t_finish = inp.t_finish / self.ref.dt
        # v should automatically be normalized from x and t normalization
        
        normal_inp.V_anode = (ref.q / (ref.m * ref.v ** 2)) * inp.V_anode 
        normal_inp.V_cathode = (ref.q / (ref.m * ref.v ** 2)) * inp.V_cathode
        normal_inp.B_max = ((ref.q * ref.L) / (ref.m * ref.v)) * inp.B_max
        
        # normal_inp.cross_sections = inp.cross_sections / (ref.L ** 2)
        
        # TODO: how to do ics?? add ic values into inputs? add a multiplier to put into apply_ics??
        
        return normal_inp
    
    
    # TODO: scale back to real dimensions without messing up other grid bounds?
    # def _normal_to_phys(self):
    #     phys_inp = copy.deepcopy(self.inp)
        
    #     inp = self.inp
    #     ref = self.ref
        
    #     phys_inp.xlim = (inp.xlim[0] * ref.L, inp.xlim[1] * ref.L)
    #     phys_inp.ylim = (inp.ylim[0] * ref.L, inp.ylim[1] * ref.L)
        
    #     return phys_inp


    def run(self):
        while (self.t < self.inp.t_finish) and self.timestep < self.inp.nt:
            print("do we enter run loop?")
            print("\033[1m" + f"Timestep: {self.timestep}, Current time: {self.t}" + "\033[0m")
            
            if self.inp.timeStepper == "RK1":
                new_particles = []
                
                print(f"Electron update at timestep '{self.timestep}'")
                new_particles_from_electrons = self.electrons.update()
                if new_particles_from_electrons:
                    new_particles.extend(new_particles_from_electrons)

                # for species in self.all_species:
                #     new_particles_from_species = species.update()
                #     if new_particles_from_species:
                #         new_particles.extend(new_particles_from_species)
                
                print("NUMBER OF NEW PARTICLES:", len(new_particles))

                if (self.timestep % self.dt_ratio == 0):
                    print(f"Particle update at timestep '{self.timestep}'")

                    for species in self.particle_species:
                        new_particles_from_species = species.update()
                        if new_particles_from_species:
                            new_particles.extend(new_particles_from_species)

                # Add all new particles at once (more efficient than per-particle loop)
                if new_particles:
                    # Add to ions and fluid electrons as before
                    for particle in new_particles:
                        self.ions.add_particle(particle)
                    # Add all particles to fluid electrons at once
                    self.electrons.add_particles(new_particles)   
                
            else:
                raise ValueError(f"Unknown time stepper: {self.inp.timeStepper}")
            
            self.t += self.dt_physical
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
            
        if self.inp.system == "euler1d":
            density = self.electrons.grid[self.c.RHOCOMP]
            pressure = self.electrons.grid[self.c.PCOMP]
            u = self.electrons.grid[self.c.UCOMP]

            pressure = np.maximum(pressure, 1e-12)
            density = np.maximum(density, 1e-12)
            
            a = np.sqrt(self.c.gamma * pressure / density) # Sound speed
            max_speed = np.max(np.abs(u) + a)
        
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
        
    def plot_2d_data(self, ax, data, extent, title, cmap='magma', scatter_data=None):
            """
            Helper function to plot 2D data on a given axis.
            """

            im = ax.imshow(data.T, extent=extent, origin='lower', cmap=cmap)
            plt.colorbar(im, ax=ax)
            ax.set_title(title, weight='bold')

            if scatter_data is not None:
                x, y = scatter_data
                ax.scatter(x, y, s=5, color='blue', alpha=0.2, clip_on=True)

    def plot_1d_data(self, ax, x_data, y_data, title, ylabel='', scatter_data=None, color='blue'):
        """
        Helper function to plot 1D line data on a given axis.
        """
        ax.plot(x_data, y_data, color=color, linewidth=1.5)
        ax.set_xlabel('Position (m)', weight='bold')
        ax.set_ylabel(ylabel, weight='bold')
        ax.set_title(title, weight='bold')
        ax.grid(True, alpha=0.3)

        if scatter_data is not None:
            x, y = scatter_data
            ax.scatter(x, y, s=10, color='red', alpha=0.5, zorder=5)

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

        ### 2D PLOTS
        # Set up 2D field plots
        if self.inp.system in ["euler2d", "euler1d"]:
            fig, axs = plt.subplots(3, 4, figsize=(24, 7))
        elif self.inp.system == "mhd2d":
            fig, axs = plt.subplots(2, 4, figsize=(36, 18))
        axs = axs.ravel()

        # Remove duplicate plotting: only use plot_vars_2d loop
        norm_data = self.electrons.normal_to_phys()[:,self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng]
        i_scatter_data = (self.ions.particles[self.pc.XCOMP] * self.ref.L, self.ions.particles[self.pc.YCOMP] * self.ref.L)
        n_scatter_data = (self.neutrals.particles[self.pc.XCOMP] * self.ref.L, self.neutrals.particles[self.pc.YCOMP] * self.ref.L)
        # norm_E, norm_B, norm_potential = self.fields.normal_to_phys()

        plot_vars_2d = self.inp.data_2d
        
        print("PLOT VARS ARE", plot_vars_2d)

        # Precompute extent
        extent = [self.inp.xlim[0] * self.ref.L, self.inp.xlim[1] * self.ref.L, self.inp.ylim[0] * self.ref.L, self.inp.ylim[1] * self.ref.L]
        
        from scipy.ndimage import gaussian_filter

        # Define plotting logic in a dictionary (like a switch-case)
        plot_map = {
            "rho_e": lambda idx: self.plot_2d_data(axs[idx], norm_data[0], extent, "Electron Mass Density"),
            "n_e": lambda idx: self.plot_2d_data(axs[idx], norm_data[0]/self.inp.m_e, extent, "Electron Number Density"),
            "u_e": lambda idx: self.plot_2d_data(axs[idx], norm_data[1], extent, "Electron Axial Velocity"),
            "v_e": lambda idx: self.plot_2d_data(axs[idx], norm_data[2], extent, "Electron Radial Velocity"),
            "w_e": lambda idx: self.plot_2d_data(axs[idx], norm_data[3], extent, "Electron Azimuthal Velocity"),
            "mu_e": lambda idx: self.plot_2d_data(axs[idx], norm_data[0] * norm_data[1], extent, "Electron Axial Momentum"),
            "mv_e": lambda idx: self.plot_2d_data(axs[idx], norm_data[0] * norm_data[2], extent, "Electron Radial Momentum"),
            "mw_e": lambda idx: self.plot_2d_data(axs[idx], norm_data[0] * norm_data[3], extent, "Electron Azimuthal Momentum"),
            "p_e": lambda idx: self.plot_2d_data(axs[idx], norm_data[4], extent, "Electron Pressure"),
            
            "Ex": lambda idx: self.plot_2d_data(axs[idx], self.fields.E[:,:,0], extent, "Electric Field X", cmap='coolwarm'),
            "Ey": lambda idx: self.plot_2d_data(axs[idx], self.fields.E[:,:,1], extent, "Electric Field Y", cmap='coolwarm'),
            "By": lambda idx: self.plot_2d_data(axs[idx], self.fields.B[:,:,1], extent, "Magnetic Field Y", cmap='magma'),
            
            "phi": lambda idx: self.plot_2d_data(axs[idx], self.fields.potential, extent, "Electric Potential", cmap='coolwarm'),
            
            "i": lambda idx: self.plot_2d_data(axs[idx], self.ions._compute_particle_density_field(self.ions)[self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], extent, "Ion Density (Scatter)", cmap=mpl.cm.Blues, scatter_data=(self.ions.particles[self.pc.XCOMP] * self.ref.L, self.ions.particles[self.pc.YCOMP] * self.ref.L)),
            "n": lambda idx: self.plot_2d_data(axs[idx], self.neutrals._compute_particle_density_field(self.neutrals)[self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], extent, "Neutral Density (Scatter)", cmap=mpl.cm.Greys, scatter_data=(self.neutrals.particles[self.pc.XCOMP] * self.ref.L, self.neutrals.particles[self.pc.YCOMP] * self.ref.L)),
            "rho_i": lambda idx: self.plot_2d_data(axs[idx], gaussian_filter(self.ions._compute_particle_density_field(self.ions)[self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], sigma=4.5), extent, "Ion Number Density", cmap=mpl.cm.Blues),
            "rho_n": lambda idx: self.plot_2d_data(axs[idx], gaussian_filter(self.neutrals._compute_particle_density_field(self.neutrals)[self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], sigma=4.5), extent, "Neutral Number Density", cmap=mpl.cm.Greys),
            "rho_q": lambda idx: self.plot_2d_data(axs[idx], gaussian_filter(self.fields.charge_density[self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], sigma=4.5), extent, "Charge Density", cmap='coolwarm'),
            "energy": lambda idx: self.plot_2d_data(axs[idx], self.electrons.euler.prim_to_cons(self.electrons.grid)[self.c.ECOMP,self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], extent, "Energy"),
            #TODO: temp change for plotting nu instead of sigma
            "sigma": lambda idx: self.plot_2d_data(axs[idx], gaussian_filter(self.electrons.pelectrons.cross_section_grid, sigma=3), extent, "Electron Collision Frequency", cmap='coolwarm'),
        }
        
        # print("ALL NEUTRALS X", self.neutrals.particles[self.pc.XCOMP])
        # print("SELF.REF.L", self.ref.L)

        # Loop over variables and call the corresponding plotting function
        for idx, var in enumerate(plot_vars_2d):
            if var in plot_map:
                plot_map[var](idx)
                axs[idx].set_xlim(self.inp.xlim[0] * self.ref.L, self.inp.xlim[1] * self.ref.L)
                axs[idx].set_ylim(self.inp.ylim[0] * self.ref.L, self.inp.ylim[1] * self.ref.L)
                
                print("X LIM", self.inp.xlim[1])
                print("y LIM", self.inp.ylim[1])
            else:
                print(f"Warning: Unknown plot variable '{var}'")

        fig.suptitle(f"Time: {self.t:.4f}, Timestep: {self.timestep}")
        fig.tight_layout()
        fig.savefig(output_plotname)
        plt.close(fig)

        ### 1D PLOTS
        
        # Generate 1D line plots if ny=1 or for middle slice
        if self.inp.ny == 1:
            iy = 0  # For 1D, only one y-index
        else:
            iy = round(self.inp.ny / 2)  # Middle slice for 2D

        # Set up 1D plots
        n_plots = len(plot_vars_2d)
        n_cols = 4
        n_rows = (n_plots + n_cols - 1) // n_cols  # Ceiling division
        fig1d, axs1d = plt.subplots(n_rows, n_cols, figsize=(20, 4*n_rows))
        axs1d = axs1d.ravel()

        # X-coordinate for 1D plots (physical units)
        x_phys = self.inp.grid_x[self.inp.ng:-self.inp.ng] * self.ref.L

        from scipy.ndimage import gaussian_filter1d

        # Define 1D plotting logic
        plot_map_1d = {
            "rho_e": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, norm_data[0, :, iy], "Electron Mass Density", ylabel="$\\rho_e$ (kg/m³)"),
            "n_e": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, norm_data[0, :, iy]/self.inp.m_e, "Electron Number Density", ylabel="$n_e$ (1/m³)"),
            "u_e": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, norm_data[1, :, iy], "Electron Axial Velocity", ylabel="$u_e$ (m/s)"),
            "v_e": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, norm_data[2, :, iy], "Electron Radial Velocity", ylabel="$v_e$ (m/s)"),
            "w_e": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, norm_data[3, :, iy], "Electron Azimuthal Velocity", ylabel="$w_e$ (m/s)"),
            "mu_e": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, norm_data[0, :, iy] * norm_data[1, :, iy], "Electron Axial Momentum", ylabel="$\\rho u$ (kg/(m²s))"),
            "mv_e": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, norm_data[0, :, iy] * norm_data[2, :, iy], "Electron Radial Momentum", ylabel="$\\rho v$ (kg/(m²s))"),
            "mw_e": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, norm_data[0, :, iy] * norm_data[3, :, iy], "Electron Azimuthal Momentum", ylabel="$\\rho w$ (kg/(m²s))"),
            "p_e": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, norm_data[4, :, iy], "Electron Pressure", ylabel="$p_e$ (Pa)"),

            "Ex": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, self.fields.E[:, iy, 0], "Electric Field X", ylabel="$E_x$ (V/m)", color='red'),
            "Ey": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, self.fields.E[:, iy, 1], "Electric Field Y", ylabel="$E_y$ (V/m)", color='red'),
            "By": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, self.fields.B[:, iy, 1], "Magnetic Field Y", ylabel="$B_y$ (T)", color='purple'),

            "phi": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, self.fields.potential[:, iy], "Electric Potential", ylabel="$\\phi$ (V)", color='orange'),

            "i": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, gaussian_filter1d(self.ions._compute_particle_density_field(self.ions)[self.inp.ng:-self.inp.ng, iy], sigma=4), "Ion Density", ylabel="$n_i$ (1/m³)", color='blue'),
            "n": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, gaussian_filter1d(self.neutrals._compute_particle_density_field(self.neutrals)[self.inp.ng:-self.inp.ng, iy], sigma=4), "Neutral Density", ylabel="$n_n$ (1/m³)", color='gray'),
            "rho_i": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, gaussian_filter1d(self.ions._compute_particle_density_field(self.ions)[self.inp.ng:-self.inp.ng, iy], sigma=4.5), "Ion Number Density", ylabel="$n_i$ (1/m³)", color='blue'),
            "rho_n": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, gaussian_filter1d(self.neutrals._compute_particle_density_field(self.neutrals)[self.inp.ng:-self.inp.ng, iy], sigma=4.5), "Neutral Number Density", ylabel="$n_n$ (1/m³)", color='gray'),
            "rho_q": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, gaussian_filter1d(self.fields.charge_density[:, iy], sigma=4.5), "Charge Density", ylabel="$\\rho_q$ (C/m³)", color='purple'),
            "energy": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, self.electrons.euler.prim_to_cons(self.electrons.grid)[self.c.ECOMP, self.inp.ng:-self.inp.ng, iy], "Energy", ylabel="Energy (J/m³)"),
            "sigma": lambda idx: self.plot_1d_data(axs1d[idx], x_phys, gaussian_filter1d(self.electrons.pelectrons.cross_section_grid[:, iy], sigma=3), "Electron Collision Frequency", ylabel="$\\nu$ (1/s)", color='green'),
        }

        # Loop over variables and call the corresponding 1D plotting function
        for idx, var in enumerate(plot_vars_2d):
            if var in plot_map_1d:
                plot_map_1d[var](idx)
                axs1d[idx].set_xlim(self.inp.xlim[0] * self.ref.L, self.inp.xlim[1] * self.ref.L)
            else:
                print(f"Warning: Unknown 1D plot variable '{var}'")

        # Hide unused subplots
        for idx in range(len(plot_vars_2d), len(axs1d)):
            axs1d[idx].axis('off')

        fig1d.suptitle(f"1D Profiles - Time: {self.t:.4f}, Timestep: {self.timestep}")
        fig1d.tight_layout()
        fig1d.savefig(output_lineplotname)
        plt.close(fig1d)
        

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


