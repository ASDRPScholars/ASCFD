from ascfd.fluid.constants import FluidConstants
from ascfd.particle.constants import ParticleConstants
from ascfd.fluid.bcs import FluidBoundaryConditions
from ascfd.params import SpeciesParams
from ascfd.fluid.species import FluidSpecies
from ascfd.fluid.quasineutral.species import QNFluidSpecies
from ascfd.particle.species import ParticleSpecies
from ascfd.inputs import Inputs
from ascfd.fields.fields import Fields
from ascfd.plasma_refs import PlasmaReferences
from ascfd.cross_sections.xe import XenonCollisionData

import numpy as np
import pandas as pd
from scipy.integrate import quad
import matplotlib.pyplot as plt
import matplotlib as mpl
import os
import sys
import subprocess
import scienceplots
import cmocean
import copy
import os

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
        
        self.c = FluidConstants(self.inp.e_system)
        # self.p_c = FluidConstants(self.inp.i_system)
        self.pc = ParticleConstants()
        self.fields = Fields(self.inp)
        
        # == INIT
        
        ## -- ELECTRONS:
        e_params = SpeciesParams(-self.inp.q, self.inp.m_e, 5/3, "e", density=1.0, temperature=100.0)
        
        if self.inp.e_system in ["euler2d", "mhd2d"]:
            self.electrons = FluidSpecies(self.c, e_params, self.inp, self.fields, self)
        elif self.inp.e_system == "quasineutral":
            self.electrons = QNFluidSpecies(self.c, e_params, self.inp, self.fields, self)
            
        if self.inp.propellant == "xe":
            
            ## -- COLLISIONS:
            self.collisions = XenonCollisionData({
                'elastic': 'ascfd/cross_sections/elastic.txt',
                'exc1': 'ascfd/cross_sections/exc1.txt',
                'exc2': 'ascfd/cross_sections/exc2.txt',
                'exc3': 'ascfd/cross_sections/exc3.txt',
                'exc4': 'ascfd/cross_sections/exc4.txt',
                'ionization': 'ascfd/cross_sections/ionization.txt',
                'ion_elastic': 'ascfd/cross_sections/ion_elastic.txt',
                'ion_backward': 'ascfd/cross_sections/ion_backward.txt'
            })
            
            ## -- IONS:
            xe_i_params = SpeciesParams(self.inp.q, self.inp.m_i, 5/3, "i", density=0.5, temperature=10.0)  # Xe+ ions  
            
            if self.inp.i_system == "euler2d":
                self.ions = FluidSpecies(self.p_c, xe_i_params, self.inp, self.fields, self) # TODO p_c
            elif self.inp.i_system == "pic":
                self.ions = ParticleSpecies(self.pc, xe_i_params, self.inp, self.fields, self.collisions, self)
                
            ## -- NEUTRALS:
            xe_n_params = SpeciesParams(0, self.inp.m_n, 5/3, "n", density=5.0, temperature=10.0)  # Xe neutrals
            
            if self.inp.n_system == "advection1d":
                self.neutrals = FluidSpecies(self.c, xe_n_params, self.inp, self.fields, self)
            elif self.inp.n_system == "pic":
                self.neutrals = ParticleSpecies(self.pc, xe_n_params, self.inp, self.fields, self.collisions, self)

            self.all_species = [self.electrons, self.neutrals, self.ions]
                
            
        elif self.inp.propellant is None:
            self.all_species = [self.electrons]
        
        else:
            raise ValueError(f"PROPELLANT TYPE NOT SUPPORTED: {self.inp.propellant}")
        
        # setup initial time to be the starting time from the inputs file.
        self.t = self.inp.t0
        self.timestep = 0
        self.dt = self.inp.dt
        
        for species in self.all_species:
            species.dt = self.dt

        # self.pelectrons.dt = self.dt_norm
        
        # -1 is no output. Always output ICs if we are outputting.
        if self.inp.output_freq >= 0:
            self.output()
            
        for entry_name in os.listdir("output/debug"):
            path = os.path.join("output/debug", entry_name)
            with open (path, "w"):
                pass
        
        
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
    
    def get_collision_frequency(self, collision_type):
        '''DEPRECATED - NOT IN USE CURRENTLY'''
        # NOTE: a good eV/k_B range for electrons is 1-30 eV
        # TODO: THAT MEANS GRAPHS THAT SAY ENERGY (eV) BUT SHOW VALUES BETWEEN 1-30 ARE REALLY TALKING ABOUT TEMP eV/k_B (?)
        
        sigma_grid = np.zeros_like(self.inp.internal_grid)
        n_e = self.get_species_number_density("e")
        n_n = self.get_species_number_density("n")
        T_e = self.get_species_temperature("e")
        v_e = self.get_species_velocity("e")
        # v_e = np.sqrt((3 * self.inp.k_B * T_e)/(self.inp.m_e))
        E_e = T_e # TODO: let's assume the cross-section graphs are talking about energy as eV/k_B... - so we don't multiply by k_B for now...
        
        if collision_type == "en":
            for i in range (self.inp.nx):
                for j in range (self.inp.ny):
                    energy = E_e[i, j]
                    sigma_grid[i, j] = self.collisions.get_total_en_cross_section(energy)
    
        print(np.shape(n_e), np.shape(n_n), np.shape(sigma_grid))
        nu_grid = n_n
        nu_grid *= sigma_grid
        nu_grid *= v_e
        
        return np.maximum(nu_grid, 1e-12)
    
    
    def get_k_ionization(self):
        k_iz_grid = np.zeros_like(self.inp.internal_grid)
        
        T_e = self.get_species_temperature("e")
        v_e = self.get_species_velocity("e")
        m_e = self.inp.m_e
        k_B = self.inp.k_B
        
        kinetic_e = 1/2 * m_e * v_e**2 
        kinetic_e /= 1.60218e-19 # J -> eV
        energy_e = kinetic_e + (T_e * k_B) # kinetic + thermal
        energy_e = int(round(energy_e))
        
        if energy_e < 0 or not isinstance(energy_e, int):
            assert ValueError("[IONIZE] NEGATIVE TOTAL ENERGY - CANNOT TABLUATE RATE COEFF")

        df = pd.read_csv('rates/xe_kiz_K.csv')
        
        for i in range (self.inp.nx):
            for j in range (self.inp.ny):
                row = df.loc[df['epsilon_eV'] == energy_e[i, j]]
                k_iz_grid[i, j] = row['k_iz']
        
        print("got all k_iz")
    
    def get_species_number_density(self, species):
        # TODO: test does this work
        ng = self.inp.ng
        
        if species == "i":
            try:
                return np.maximum(self.ions.compute_particle_density_field()[ng:-ng, ng:-ng], 1e16)
            except:
                return self.ions.grid[self.ions.c.RHOCOMP, ng:-ng, ng:-ng]/self.inp.m_i
        elif species == "n":
            return np.maximum(self.neutrals.compute_particle_density_field()[ng:-ng, ng:-ng], 1e16)
            # except:
            #     return np.maximum(self.neutrals.grid[self.neutrals.c.RHOCOMP, ng:-ng, ng:-ng]/self.inp.m_n, 1e-12)
        elif species == "e" and self.inp.e_system == "quasineutral":
            return np.maximum(self.electrons.grid[self.electrons.c.NCOMP, ng:-ng, ng:-ng], 1e16)
        
        
    def get_species_current_density(self, species):
        """Retrieve 1D (axial) current from charged species"""
        
        if species == "i":
            try:
                return self.ions.compute_particle_density_field() * self.ions.compute_particle_x_velocity_field() * self.inp.q
            except:
                ng = self.inp.ng
                return self.ions.grid[self.ions.c.RHOCOMP, ng:-ng, ng:-ng] * self.ions.grid[self.ions.c.UCOMP, ng:-ng, ng:-ng] * -self.inp.q
    
        elif species == "e" and self.inp.e_system == "quasineutral":
            ng = self.inp.ng
            return self.electrons.grid[self.electrons.c.NCOMP, ng:-ng, ng:-ng] * self.electrons.grid[self.electrons.c.UCOMP, ng:-ng, ng:-ng] * -self.inp.q
    
    
    def get_species_velocity(self, species, direction=None):
        # TODO: complete implementation for particles
        
        if species == "i":
            pass
        elif species == "n":
            pass
        elif species == "e" and self.inp.e_system == "quasineutral":
            ng = self.inp.ng
            n_e = self.get_species_number_density("e")
            q_e = -self.inp.q
            
            u = self.electrons.grid[self.electrons.c.UCOMP, ng:-ng, ng:-ng] #/ (n_e * q_e)
            v = self.electrons.grid[self.electrons.c.VCOMP, ng:-ng, ng:-ng] #/ (n_e * q_e)
            w = self.electrons.grid[self.electrons.c.WCOMP, ng:-ng, ng:-ng] #/ (n_e * q_e)
            vel = np.sqrt(u**2 + v**2 + w**2)
        
            value = {"u": u, "v": v, "w": w}.get(direction, vel)
            
            return value
        
    def get_species_temperature(self, species):
        # TODO: complete implementation for particles
        
        if species == "i":
            pass
        elif species == "n":
            pass
        elif species == "e" and self.inp.e_system == "quasineutral":
            ng = self.inp.ng
            return self.electrons.grid[self.electrons.c.TCOMP, ng:-ng, ng:-ng]
        
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
            
            for entry_name in os.listdir("output/debug"):
                path = os.path.join("output/debug", entry_name)
                with open (path, "a") as f:
                    f.write(f"\n--Timestep: {self.timestep}--")
                
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
        if self.inp.e_system == "euler2d":
            density = self.electrons.grid[self.electrons.c.RHOCOMP]
            pressure = self.electrons.grid[self.electrons.c.PCOMP]
            u = self.electrons.grid[self.electrons.c.UCOMP]
            v = self.electrons.grid[self.electrons.c.VCOMP]
            # Ensure pressure and density are positive before sqrt
            pressure = np.maximum(pressure, 1e-12)
            density = np.maximum(density, 1e-12)
            a = np.sqrt(self.electrons.c.gamma * pressure / density) # Sound speed
            max_speed_x = np.max(np.abs(u) + a)
            max_speed_y = np.max(np.abs(v) + a)
            max_speed = max(max_speed_x, max_speed_y) # More robust estimate
        
        elif self.inp.e_system == "mhd2d":
            density = self.electrons.grid[self.electrons.c.RHOCOMP]
            pressure = self.electrons.grid[self.electrons.c.PCOMP]
            u = self.electrons.grid[self.electrons.c.UCOMP]
            v = self.electrons.grid[self.electrons.c.VCOMP]
            Bx = self.electrons.grid[self.electrons.c.BXCOMP]
            By = self.electrons.grid[self.electrons.c.BYCOMP]
            
            # Ensure pressure and density are positive
            pressure = np.maximum(pressure, 1e-12)
            density = np.maximum(density, 1e-12)
            
            a = np.sqrt(self.electrons.c.gamma * pressure / density) # Sound speed
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
            raise RuntimeError(f"System {self.inp.e_system} not supported for dt calculation.")

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
                    components = [self.electrons.grid[q, i, j] for q in range(self.electrons.c.NUMQ)]
                    f.write(f"{x:.12f}, {y:.12f}, " + ", ".join(f"{comp:.8f}" for comp in components) + "\n")

        # Set up 2D field plots
        if self.inp.e_system == "euler2d":
            fig, axs = plt.subplots(3, 4, figsize=(24, 7))
        elif self.inp.e_system == "quasineutral":
            fig, axs = plt.subplots(3, 4, figsize=(24, 7))
        elif self.inp.e_system == "mhd2d":
            fig, axs = plt.subplots(2, 4, figsize=(36, 18))
        axs = axs.ravel()

        # Remove duplicate plotting: only use plot_vars_2d loop
        # norm_data = self.electrons.normal_to_phys()[:,self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng]
        # i_scatter_data = (self.ions.particles[self.pc.XCOMP] * self.ref.L, self.ions.particles[self.pc.YCOMP] * self.ref.L)
        # n_scatter_data = (self.neutrals.particles[self.pc.XCOMP] * self.ref.L, self.neutrals.particles[self.pc.YCOMP] * self.ref.L)
        # norm_E, norm_B, norm_potential = self.fields.normal_to_phys()
        
        plot_data = self.electrons.grid[:,self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng]
        E_data, B_data, potential_data = self.fields.E, self.fields.B, self.fields.potential

        plot_vars_2d = self.inp.data_2d
        
        print("PLOT VARS ARE", plot_vars_2d)

        # Precompute extent
        extent = [self.inp.xlim[0] * self.ref.L, self.inp.xlim[1] * self.ref.L, self.inp.ylim[0] * self.ref.L, self.inp.ylim[1] * self.ref.L]
        print("!EXTENT!", extent)
        
        from scipy.ndimage import gaussian_filter
        
        print("L IS", self.ref.L)

        # Define plotting logic in a dictionary (like a switch-case)
        plot_map = {
            "rho_e": lambda idx: self.plot_2d_data(axs[idx], plot_data[0]*self.inp.m_e if self.inp.e_system == "quasineutral" else plot_data[0], extent, "Electron Mass Density"),
            "n_e": lambda idx: self.plot_2d_data(axs[idx], plot_data[0] if self.inp.e_system == "quasineutral" else plot_data[0]/self.inp.m_e, extent, "Electron Number Density"),
            "u_e": lambda idx: self.plot_2d_data(axs[idx], plot_data[1], extent, "Electron Axial Velocity"),
            "v_e": lambda idx: self.plot_2d_data(axs[idx], plot_data[2], extent, "Electron Radial Velocity"),
            "w_e": lambda idx: self.plot_2d_data(axs[idx], plot_data[3], extent, "Electron Azimuthal Velocity"),
            "mu_e": lambda idx: self.plot_2d_data(axs[idx], plot_data[0] * plot_data[1], extent, "Electron Axial Momentum"),
            "mv_e": lambda idx: self.plot_2d_data(axs[idx], plot_data[0] * plot_data[2], extent, "Electron Radial Momentum"),
            "mw_e": lambda idx: self.plot_2d_data(axs[idx], plot_data[0] * plot_data[3], extent, "Electron Azimuthal Momentum"),
            "p_e": lambda idx: self.plot_2d_data(axs[idx], plot_data[4], extent, "Electron Pressure"),
            
            "jx_e": lambda idx: self.plot_2d_data(axs[idx], plot_data[1], extent, "Electron Axial Current", cmap='winter'),
            "jy_e": lambda idx: self.plot_2d_data(axs[idx], plot_data[2], extent, "Electron Radial Current"),
            "jz_e": lambda idx: self.plot_2d_data(axs[idx], plot_data[3], extent, "Electron Azimuthal Current"),
            "T_e": lambda idx: self.plot_2d_data(axs[idx], plot_data[4], extent, "Electron Temperature"),
            
            "Ex": lambda idx: self.plot_2d_data(axs[idx], E_data[:,:,0], extent, "Electric Field X", cmap='coolwarm'),
            "Ey": lambda idx: self.plot_2d_data(axs[idx], E_data[:,:,1], extent, "Electric Field Y", cmap='coolwarm'),
            "By": lambda idx: self.plot_2d_data(axs[idx], B_data[:,:,1], extent, "Magnetic Field Y", cmap='magma'),
            
            "phi": lambda idx: self.plot_2d_data(axs[idx], potential_data, extent, "Electric Potential", cmap='coolwarm'),
            
            "i": lambda idx: self.plot_2d_data(axs[idx], self.get_species_number_density("i")[self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], extent, "Ion Density (Scatter)", cmap=mpl.cm.Blues, scatter_data=(self.ions.particles[self.pc.XCOMP] * self.ref.L, self.ions.particles[self.pc.YCOMP] * self.ref.L)),
            "n": lambda idx: self.plot_2d_data(axs[idx], self.get_species_number_density("n")[self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], extent, "Neutral Density (Scatter)", cmap=mpl.cm.Greys, scatter_data=(self.neutrals.particles[self.pc.XCOMP] * self.ref.L, self.neutrals.particles[self.pc.YCOMP] * self.ref.L)),
            "n_i": lambda idx: self.plot_2d_data(axs[idx], self.get_species_number_density("i")[self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], extent, "Ion Number Density", cmap=mpl.cm.Blues),
            "rho_i": lambda idx: self.plot_2d_data(axs[idx], self.ions.grid[self.ions.c.RHOCOMP, self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], extent, "Ion Mass Density", cmap=mpl.cm.Blues),
            "u_i": lambda idx: self.plot_2d_data(axs[idx], self.ions.grid[self.ions.c.UCOMP, self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], extent, "Ion Axial Velocity", cmap="magma"),
            "p_i": lambda idx: self.plot_2d_data(axs[idx], self.ions.grid[self.ions.c.PCOMP, self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], extent, "Ion Pressure", cmap="magma"),
            "n_n": lambda idx: self.plot_2d_data(axs[idx], self.get_species_number_density("n")[self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], extent, "Neutral Number Density", cmap=mpl.cm.Greys),
            "rho_q": lambda idx: self.plot_2d_data(axs[idx], self.fields.charge_density[self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], extent, "Charge Density", cmap='coolwarm'),
            "energy": lambda idx: self.plot_2d_data(axs[idx], self.electrons.euler.prim_to_cons(self.electrons.grid)[self.electrons.c.ECOMP,self.inp.ng:-self.inp.ng,self.inp.ng:-self.inp.ng], extent, "Energy"),
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
                
                # masked_array = np.ma.array (plot, mask=np.isnan(plot))
                # cmap = mpl.cm.jet
                # cmap.set_bad('white',1.)
                # axs.imshow(masked_array, interpolation='nearest', cmap=cmap)
                
                print("X LIM", self.inp.xlim[1])
                print("y LIM", self.inp.ylim[1])
            else:
                print(f"Warning: Unknown plot variable '{var}'")

        # 1D line plot (center slice)
        iy = round(self.inp.ny / 2)
        
        # plot_data_1d = self.electrons.grid[0, self.inp.ng:-self.inp.ng, iy]
        # axs1d[0].plot(plot_data_1d)
        # axs1d[0].set_ylim(top=1.2)
        # axs1d[0].set_title("Electron Density", weight='bold')
        
        # energy_1d = 0.5 * self.electrons.grid[0, self.inp.ng:-self.inp.ng, iy] * (self.electrons.grid[1, self.inp.ng:-self.inp.ng, iy] ** 2 + self.electrons.grid[2, self.inp.ng:-self.inp.ng, iy] ** 2 + self.electrons.grid[3, self.inp.ng:-self.inp.ng, iy] ** 2)
        # axs1d[1].plot(energy_1d)
        # axs1d[1].set_title("Electron Energy (eV)", weight='bold')
        
        # # Ionization frequency vs X position
        # if hasattr(self, 'pelectrons') and hasattr(self.pelectrons, 'ionization_positions_x'):
        #     if self.pelectrons.ionization_positions_x:
        #         # Convert positions to grid indices and create histogram
        #         x_positions = np.array(self.pelectrons.ionization_positions_x)
        #         # Convert physical positions to grid coordinates
        #         x_indices = ((x_positions - self.inp.grid_x[0]) / self.inp.dx).astype(int)
        #         # Create histogram bins for x grid points
        #         x_bins = np.arange(self.inp.ng, self.inp.nx - self.inp.ng + 1)
        #         ionization_freq, _ = np.histogram(x_indices, bins=x_bins)
                
        #         # Smooth the data with a moving average
        #         window_size = min(5, len(ionization_freq) // 3)  # Adaptive window size
        #         if window_size >= 3:
        #             from scipy.ndimage import gaussian_filter1d
        #             ionization_freq_smooth = gaussian_filter1d(ionization_freq.astype(float), sigma=4)
        #         else:
        #             ionization_freq_smooth = ionization_freq
                
        #         axs1d[2].plot(x_bins[:-1], ionization_freq_smooth)
        #         axs1d[2].set_title("Electron-Neutral Collision Frequency", weight='bold')
        
        # # Ion density 1D cross-section
        # ion_density_data = self.ions._compute_particle_density_field(self.ions)[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        # ion_density_1d = ion_density_data[:, iy]
        # from scipy.ndimage import gaussian_filter1d
        # ion_density_1d_smooth = gaussian_filter1d(ion_density_1d.astype(float), sigma=4)
        # axs1d[3].plot(ion_density_1d_smooth)
        # axs1d[3].set_title("Ion Density", weight='bold')
        
        # # Neutral density 1D cross-section  
        # neutral_density_data = self.neutrals._compute_particle_density_field(self.neutrals)[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        # neutral_density_1d = neutral_density_data[:, iy]
        # neutral_density_1d_smooth = gaussian_filter1d(neutral_density_1d.astype(float), sigma=4)
        # axs1d[4].plot(neutral_density_1d_smooth)
        # axs1d[4].set_title("Neutral Density", weight='bold')

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

        fig.suptitle(f"Time: {self.t:.4f}, Timestep: {self.timestep}")
        fig.tight_layout()
        fig.savefig(output_plotname)
        plt.close(fig)
        

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


