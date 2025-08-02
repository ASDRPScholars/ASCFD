from ascfd.fluid.constants import FluidConstants
from ascfd.particle.constants import ParticleConstants
from ascfd.fluid.euler import FluidEuler
from ascfd.fluid.ics import FluidInitialConditions
from ascfd.inputs import Inputs
from ascfd.params import SpeciesParams
from ascfd.fluid.bcs import FluidBoundaryConditions
from ascfd.fluid.flux import FluidFlux
from ascfd.fields.fields import Fields

# from ascfd.simulation import Simulation

import ascfd.fluid.ics as ics

import numpy as np
import matplotlib.pyplot as plt

import sys

class FluidSpecies:
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields, simulation):
        print("INITIALIZED ELECTRONS")
        self.c = FluidConstants(a_inputs)
        self.pc = ParticleConstants()
        self.euler = FluidEuler(self.c)
        self.flux = FluidFlux(self.c, a_inputs.flux)
        
        self.fields = fields
        self.simulation = simulation
        self.pelectrons = None
        
        self.inp = a_inputs
        self.params = params
        self.dt = None
        
        self.grid = np.zeros((self.c.NUMQ, self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        
        self.bcs = FluidBoundaryConditions(self.grid, self.inp.bcs_lo, self.inp.bcs_hi, self.inp)
        self.ics = FluidInitialConditions(self.grid, self.inp, self.params)
        
        self.grid[:] = self.ics.apply_ics()
        
        # Apply boundary conditions AFTER setting initial conditions
        self.bcs.apply_bcs()
        
        self.check_grid(self.c)
        
        
    def update(self):
        
        print("dt is", self.dt)
        consU = self.euler.prim_to_cons(self.grid)

        _, right_flux, left_flux, top_flux, bottom_flux = self.flux.getFlux(self.grid, self.inp.nx, self.inp.ny, self.inp.ng)
                
        for i in range(self.inp.ng, self.inp.nx + self.inp.ng):
            for j in range(self.inp.ng, self.inp.ny + self.inp.ng):
                for icomp in range(self.c.NUMQ):
                    
                    delta = (
                        (self.dt / self.inp.dx) * (right_flux[icomp, i, j] - left_flux[icomp, i, j]) +
                        (self.dt / self.inp.dy) * (top_flux[icomp, i, j] - bottom_flux[icomp, i, j]))
                        
                    consU[icomp, i, j] = consU[icomp, i, j] - delta
                    
        ## --LORENTZ UPDATE--
        self._apply_lorentz_source_terms(consU)

        ## --P ELECTRONS UPDATE + COLLISIONAL DAMPING--
        new_particle_array = self.convert_to_particles()
        
        # Clear existing particles and properly initialize with new ones
        self.pelectrons.active_count = 0
        self.pelectrons.is_active.fill(False)
        self.pelectrons.free_slots.clear()
        
        # Add particles from converted array
        n_new_particles = new_particle_array.shape[1]
        for i in range(n_new_particles):
            if i < self.pelectrons.capacity:
                self.pelectrons.particles[:, i] = new_particle_array[:, i]
                self.pelectrons.is_active[i] = True
                self.pelectrons.active_count += 1
        
        new_particles = self.pelectrons.update()
        self.pelectrons.update_cross_section_grid()
        sigma = self.pelectrons.cross_section_grid
        
        self._apply_damping_source_terms(sigma, consU)

        ##
        self.grid[:] = self.euler.cons_to_prim(consU)
        self.bcs.apply_bcs()
        
        ## --ELECTRIC FIELD UPDATE--
        charge_density = self.get_charge_density()
        
        self.fields.clear_charge_density()
        self.fields.add_charge_density(charge_density) # -!- TOGGLE -!-

        self.fields.update_E()

        return new_particles
    

    def _apply_damping_source_terms(self, sigma, consU_new):
            
        # OK ALL IT IS: a * sigma(e) * n_n * rho_e * V_e
        
        # --COLLISION DAMPING--
        n_n = self.simulation.neutrals._compute_particle_density_field(self.simulation.neutrals)[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng] + 0.01
        rho_e = self.grid[self.c.RHOCOMP, self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        V_x = self._get_V()[:, :, 0]

        # print("!DEBUG DAMPING! n-n is", n_n)
        # print("!DEBUG DAMPING! sigma is", sigma)

        # np.set_printoptions(threshold=sys.maxsize)
        # print("SIGMA", sigma)
        # print("n_n", n_n)
        # print("rho_e", rho_e)
        # print("V_x", V_x)

        damping_source = sigma * n_n * rho_e * V_x

        # print("!*! MIN MAX OF damping_source IS", np.min(damping_source), np.max(damping_source))

        # Collision damping: reduced from 100× to 50× for more natural electron dynamics
        # Excessive damping was causing unphysical electron behavior
        consU_new[self.c.MUCOMP, self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng] += damping_source * self.dt * 50

    
    def _apply_lorentz_source_terms(self, consU_new):

        E = self.fields.E
        B = self.fields.B
        # V = self._get_V()
        
        charge_density = self.params.charge * self.grid[self.c.RHOCOMP] / self.params.mass
        
        ## --MOMENTUM UPDATE--
        # lorentz_force = (E + np.cross(V, B))
        # x_mom_source = charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] * lorentz_force[:, :, 0]
        # y_mom_source = charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] * lorentz_force[:, :, 0]
        
        # Apply Lorentz force: F = ρq(E + v×B) to x, y, and z momentum
        x_mom_source = charge_density[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng] * E[:, :, 0]
        y_mom_source = charge_density[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng] * E[:, :, 1]

        np.set_printoptions(threshold=sys.maxsize)
        # print("!LORENTZ DEBUG! charge_density arr is", charge_density[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng])
        # print("!LORENTZ DEBUG! Ex arr is", E[:, :, 0])
        
        # Z-momentum can be deposited from v×B cross product (azimuthal component)
        # For now, using electric field z-component if available, otherwise zero
        # if E.shape[2] > 2:
        #     z_mom_source = charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] * E[:, :, 2]
        if True:
            # Magnetic field cross product could deposit z-momentum even with 2D E-field
            # v×B = (vy*Bz - vz*By, vz*Bx - vx*Bz, vx*By - vy*Bx)
            # For 2D simulation, typically Bz is the only non-zero B component
            prim_current = self.euler.cons_to_prim(consU_new)
            
            # Check for NaN propagation from cons_to_prim conversion
            if np.any(np.isnan(prim_current)):
                print(f"!ERROR! NaN in prim_current after cons_to_prim!")
                print(f"Density: min={np.min(prim_current[self.c.RHOCOMP]):.3e}, nan_count={np.sum(np.isnan(prim_current[self.c.RHOCOMP]))}")
                print(f"Pressure: min={np.min(prim_current[self.c.PCOMP]):.3e}, nan_count={np.sum(np.isnan(prim_current[self.c.PCOMP]))}")
                print(f"Energy check - before cons_to_prim, consU energy: min={np.min(consU_new[self.c.ECOMP]):.3e}")
            
            vx = prim_current[self.c.UCOMP][self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
            vy = prim_current[self.c.VCOMP][self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
            vz = prim_current[self.c.WCOMP][self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
            if B.shape[2] > 2:
                Bz = B[:, :, 2]
                cross_product = vx * B[:, :, 1] - vy * B[:, :, 0]
                
                # Physical saturation: momentum source decreases as vz increases
                # This represents realistic Hall thruster physics where azimuthal velocity 
                # eventually saturates due to collisions, geometry, etc.

                # !GOODENOUGH!
                v_sat = 0.2  # Saturation velocity scale
                saturation_factor = 1.0 / (1.0 + (np.abs(vz) / v_sat)**2)  # Smooth saturation
                z_mom_source = charge_density[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng] * cross_product * saturation_factor * 100
                
                # Debug saturation effect
                if np.any(saturation_factor < 0.9):
                    print(f"!SATURATION! Min factor: {np.min(saturation_factor):.3f}, Max vz: {np.max(np.abs(vz)):.3e}")
                
                # Check z_mom_source immediately after calculation
                if np.any(np.isnan(z_mom_source)):
                    print(f"!ERROR! NaN in z_mom_source calculation!")
                    print(f"charge_density: min={np.min(charge_density):.3e}, max={np.max(charge_density):.3e}, nan_count={np.sum(np.isnan(charge_density))}")
                    print(f"vx: min={np.min(vx):.3e}, max={np.max(vx):.3e}, nan_count={np.sum(np.isnan(vx))}")
                    print(f"vy: min={np.min(vy):.3e}, max={np.max(vy):.3e}, nan_count={np.sum(np.isnan(vy))}")
            else:
                z_mom_source = np.zeros_like(x_mom_source)
        
        # Lorentz forces: rebalanced for more physical force ratios
        # Reduced z-force from 100× to 50× to prevent energy profile drift
        # Increased x,y forces from 10× to 20× for better axial/radial dynamics
        consU_new[self.c.MUCOMP, self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng] += x_mom_source * self.dt * 20
        consU_new[self.c.MVCOMP, self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng] += y_mom_source * self.dt * 20
        consU_new[self.c.MWCOMP, self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng] += z_mom_source * self.dt * 50

        # print("!LORENTZ DEBUG! new MUCOMP:", consU_new[self.c.MUCOMP, self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng] )
        print(f"!@! Electromagnetic coupling strengths:")
        print(f"!@! Max |Ex|: {np.max(np.abs(E[:, :, 0])):.3e}")
        print(f"!@! Max |Ey|: {np.max(np.abs(E[:, :, 1])):.3e}")
        # print(f"!@! Max |x_mom_source|: {np.max(np.abs(x_mom_source)):.3e}")
        # print(f"!@! Max |y_mom_source|: {np.max(np.abs(y_mom_source)):.3e}")
        print(f"!@! Max |z_mom_source|: {np.max(np.abs(z_mom_source)):.3e}")
        print(f"!@! Max |charge_density|: {np.max(np.abs(charge_density)):.3e}")
        print(f"!@! Max |z_velocity|: {np.max(np.abs(self.grid[self.c.WCOMP])):.3e}")

        # --ENERGY UPDATE--
        # V = self._get_V()
        # energy_source = charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] * \
        #     (E[:, :, 0] * V[:, :, 0] + E[:, :, 1] * V[:, :, 1] + E[:, :, 2] * V[:, :, 2]) # cursed vector dot product on two (100, 100, 3 matricies)
        
        # Energy source: ρq(E·v) + work done by magnetic acceleration
        prim_new = self.euler.cons_to_prim(consU_new)
        vx = prim_new[self.c.UCOMP][self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        vy = prim_new[self.c.VCOMP][self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        vz = prim_new[self.c.WCOMP][self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        
        # Physical electric field work: J⋅E heating
        # Remove arbitrary /10 scaling for more realistic energy transfer
        if E.shape[2] > 2:
            electric_work = charge_density[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng] * \
                (E[:, :, 0] * vx + E[:, :, 1] * vy + E[:, :, 2] * vz) * 2.0
        else:
            electric_work = charge_density[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng] * \
                (E[:, :, 0] * vx + E[:, :, 1] * vy) * 2.0
        
        # CRITICAL: Add work done by z-momentum acceleration  
        # Work = F_z · v_z, but be careful with multiplier scaling!
        if B.shape[2] > 2:
            rho_interior = prim_new[self.c.RHOCOMP][self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
            # z_mom_source already has *100, so remove it for work calculation
            z_mom_source_unscaled = z_mom_source / 100  # Remove the arbitrary multiplier
            z_force_per_volume = z_mom_source_unscaled / self.dt  # Force per unit volume (proper units)
            z_force_per_mass = z_force_per_volume / (rho_interior + 1e-12)  # Force per unit mass
            z_work = z_force_per_mass * vz  # Work per unit mass per unit time
            z_work *= rho_interior  # Convert back to work per unit volume
            print(f"!DEBUG! Z-work stats: max={np.max(np.abs(z_work)):.3e}, z_mom_max={np.max(np.abs(z_mom_source)):.3e}")
        else:
            z_work = np.zeros_like(vx)
        
        # Total energy source
        energy_source = electric_work + z_work 
            
        print(f"!@! Max |energy_source|: {np.max(np.abs(energy_source)):.3e}")
        print(f"!@! Energy source range: [{np.min(energy_source):.3e}, {np.max(energy_source):.3e}]")

        # Energy source: reduced from 100× to 50× for consistency with momentum rebalancing
        # Maintains energy-momentum coupling while reducing excessive heating
        consU_new[self.c.ECOMP, self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng] += energy_source * self.dt * 50
        
    
    def get_charge_density(self):
        charge_density = self.params.charge * self.get_number_density()
        return charge_density
    
    
    def get_number_density(self):
        number_density = self.grid[self.c.RHOCOMP] / self.params.mass
        return number_density
    
    
    def convert_to_particles(self):
        "Converts Eulerian fluid to particles, assuming a drifting Maxwellian velocity distribution."
        
        WEIGHT = self.pc.NUMQ
        
        # TODO: DO WE ACTUALLY NEED n_ppc particle electrons??
        n_particles = self.inp.n_ppc * self.inp.nx * self.inp.ny
        
        # Particle array: [x, y, vx, vy, vz, weight] = 6 components
        ic_particles = np.zeros((self.pc.NUMQ + 1, n_particles))
        
        # kB = 1.380649e-23
        kB = 1
        v_th = np.sqrt(2 * kB * self.params.temperature / self.params.mass)
        
        p_idx = 0 
        
        while p_idx < n_particles:
            for i in range (self.inp.ng, self.inp.nx + self.inp.ng):
                for j in range(self.inp.ng, self.inp.ny + self.inp.ng):
                    
                    vx_drift = self.grid[self.c.UCOMP, i, j]
                    vy_drift = self.grid[self.c.VCOMP, i, j]
                    vz_drift = self.grid[self.c.WCOMP, i, j]  # Extract z-velocity from fluid grid
                    
                    rho = self.grid[self.c.RHOCOMP, i, j]
                    p = self.grid[self.c.PCOMP, i, j]
                    
                    # # TODO: use global v_th or use this?
                    # v_th = np.sqrt(2 * p / rho)
                    
                    weight = rho * self.inp.dx * self.inp.dy / self.inp.n_ppc
                    
                    for n in range (self.inp.n_ppc):
                        R1, R2 = np.random.rand(2)
                        R3, R4 = np.random.rand(2)

                        vx = v_th * np.sqrt(-1 * np.log(R1)) * np.cos(2 * np.pi * R2)
                        vy = v_th * np.sqrt(-1 * np.log(R1)) * np.sin(2 * np.pi * R2)
                        vz = 0.0  # TEMP: Remove thermal z-velocity to see spatial drift
                        
                        # Debug: Check thermal vs drift magnitudes
                        # if abs(vz_drift) > 1e-10 or (i % 10 == 0 and j % 10 == 0):  # Sample some cells
                        #     print(f"!DEBUG! Cell ({i-self.inp.ng}, {j-self.inp.ng}): v_th={v_th:.3e}, vz_thermal={vz:.3e}, vz_drift={vz_drift:.3e}, ratio={abs(vz_drift/vz) if abs(vz) > 1e-12 else 0:.3e}")
                        
                        x_offset = np.random.uniform(-0.4999, 0.5)
                        y_offset = np.random.uniform(-0.4999, 0.5)
                        
                        ic_particles[self.pc.XCOMP, p_idx] = (i + x_offset - self.inp.ng) * self.inp.dx 
                        ic_particles[self.pc.YCOMP, p_idx] = (j + y_offset - self.inp.ng) * self.inp.dy 
                        ic_particles[self.pc.UCOMP, p_idx] = vx + vx_drift
                        ic_particles[self.pc.VCOMP, p_idx] = vy + vy_drift
                        ic_particles[self.pc.WCOMP, p_idx] = vz + vz_drift  # Include z-drift from fluid grid
                        ic_particles[WEIGHT, p_idx] = weight
                        
                        # Debug: Print vz values for particles in magnetic field region
                        # if abs(vz_drift) > 1e-10:  # Only print where there's significant z-drift
                        #     print(f"!DEBUG! Particle at ({i-self.inp.ng}, {j-self.inp.ng}): vz_drift={vz_drift:.3e}, total_vz={vz + vz_drift:.3e}")
                        
                        p_idx += 1

        # Debug: Check vz statistics in final particle array
        vz_particles = ic_particles[self.pc.WCOMP, :]
        # print(f"!DEBUG! convert_to_particles() vz stats: min={np.min(vz_particles):.3e}, max={np.max(vz_particles):.3e}, nonzero_count={np.sum(np.abs(vz_particles) > 1e-10)}")
        
        return ic_particles
    
    
    def add_particles(self, new_particles_data):
        """Adds new particles from ionization to grid by adding conserved quantity fields."""
        
        if isinstance(new_particles_data, np.ndarray):
            
            consU = self.euler.prim_to_cons(self.grid)
            
            # Add mass density (rho)
            density_field = self._compute_bulk_quantity_field(new_particles_data, self.c.RHOCOMP)
            consU[self.c.RHOCOMP] += density_field

            # Add momentum densities (rho*u, rho*v, rho*w)
            for momentum_component in [self.c.MUCOMP, self.c.MVCOMP, self.c.MWCOMP]:
                momentum_field = self._compute_bulk_quantity_field(new_particles_data, momentum_component)
                consU[momentum_component] += momentum_field

            # Add total energy density
            energy_field = self._compute_bulk_quantity_field(new_particles_data, self.c.ECOMP)
            consU[self.c.ECOMP] += energy_field
            
            self.grid[:] = self.euler.cons_to_prim(consU)
            
            
    def _compute_bulk_quantity_field(self, particles_data, var):
        """Compute a conserved quantity field (mass, momentum, or energy) from particle data."""
        if particles_data.shape[1] == 0:
            return np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))

        field = np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        cell_area = self.inp.dx * self.inp.dy
        m = self.params.mass

        for i in range(particles_data.shape[1]):
            x = particles_data[self.pc.XCOMP, i]
            y = particles_data[self.pc.YCOMP, i]
            u = particles_data[self.pc.UCOMP, i]
            v = particles_data[self.pc.VCOMP, i]
            w = particles_data[self.pc.WCOMP, i]
            weight = particles_data[self.WEIGHT, i]

            ix = int((x - self.inp.grid_x[0]) / self.inp.dx)
            iy = int((y - self.inp.grid_y[0]) / self.inp.dy)

            if 0 <= ix < self.inp.nx_with_ghosts and 0 <= iy < self.inp.ny_with_ghosts:
                if var == self.c.RHOCOMP:
                    field[ix, iy] += m * weight / cell_area
                elif var == self.c.MUCOMP:
                    field[ix, iy] += m * u * weight / cell_area
                elif var == self.c.MVCOMP:
                    field[ix, iy] += m * v * weight / cell_area
                elif var == self.c.MWCOMP:
                    field[ix, iy] += m * w * weight / cell_area
                elif var == self.c.ECOMP:
                    kinetic_energy = 0.5 * m * (u**2 + v**2 + w**2)
                    field[ix, iy] += kinetic_energy * weight / cell_area

        return field
    
    
    # TODO: make assert_variable_type -> prim or cons work
    # TODO: check particles ("check_grid()") for neutrals and ions too
    def check_grid(self, prim=False, cons=False):
        # Check for negative or invalid values in the grid
        for i in range(self.inp.nx + 2 * self.inp.ng):
            for j in range(self.inp.ny + 2 * self.inp.ng):
                # if prim:
                #     # Check for negative pressure
                #     if self.grid[self.c.PCOMP, i, j] <= 0:
                #         print(f"Negative Pressure - Bad cell: ({i}, {j})")
                #         assert False

                #     # Check for negative density
                #     if self.grid[self.c.RHOCOMP, i, j] <= 0:
                #         print(f"Negative Density - Bad cell: ({i}, {j})")
                #         assert False

                # if cons:
                #     # Check for negative energy
                #     if self.grid[self.c.ECOMP, i, j] <= 0:
                #         print(f"Negative Energy - Bad cell: ({i}, {j})")
                #         assert False

                # Check for NaN values
                for icomp in range(self.c.NUMQ):
                    if np.isnan(self.grid[icomp, i, j]):
                        print(f"NaN value - Bad cell: ({i}, {j}), component: {icomp}")
                        assert False
                        
    
    def _get_V(self):
        
        # Use actual z-velocity from grid instead of zeros
        V = np.array([self.grid[self.c.UCOMP], self.grid[self.c.VCOMP], self.grid[self.c.WCOMP]])
        
        # change (3, 104, 104) to (100, 100, 3)
        V = np.transpose(V, (1, 2, 0))
        V = V[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng:]
        
        return V
    