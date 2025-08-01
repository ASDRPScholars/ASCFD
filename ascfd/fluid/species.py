from ascfd.fluid.constants import FluidConstants
from ascfd.particle.constants import ParticleConstants
from ascfd.fluid.euler import FluidEuler
from ascfd.fluid.ics import FluidInitialConditions
from ascfd.inputs import Inputs
from ascfd.params import SpeciesParams
from ascfd.fluid.bcs import FluidBoundaryConditions
from ascfd.fluid.flux import FluidFlux
from ascfd.fields.fields import Fields

import ascfd.fluid.ics as ics

import numpy as np
import matplotlib.pyplot as plt

import sys

class FluidSpecies:
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields):
        print("INITIALIZED ELECTRONS")
        self.c = FluidConstants(a_inputs)
        self.pc = ParticleConstants()
        self.euler = FluidEuler(self.c)
        self.flux = FluidFlux(self.c, a_inputs.flux)
        
        self.fields = fields
        
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
        
        
    def plot_a(self, title):
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))

        # Titles and components
        components = [
            (self.c.RHOCOMP, "rho"),
            (self.c.UCOMP, "u"),
            (self.c.VCOMP, "v"),
            (self.c.PCOMP, "p")
        ]

        for ax, (comp, label) in zip(axes.flat, components):
            # if comp == self.c.RHOCOMP:
            #     im = ax.imshow(self.grid[comp, 2*self.inp.ng:-2*self.inp.ng, 2*self.inp.ng:-2*self.inp.ng])
            # else:
            im = ax.imshow(self.grid[comp])
            ax.set_title(f"{label} {title}")
            plt.colorbar(im, ax=ax)
            # plt.scatter(1, 5)
            # plt.scatter(1, 3)
            # plt.scatter(1, 7)

        plt.tight_layout()
        plt.show()
        
    def update(self):
        print("dt is", self.dt)
        
        self.bcs.apply_bcs()
        consU = self.euler.prim_to_cons(self.grid)
        
        self.plot_a(title="before flux")

        _, right_flux, left_flux, top_flux, bottom_flux = self.flux.getFlux(self.grid, self.inp.nx, self.inp.ny, self.inp.ng)
        
        for i in range(self.inp.ng, self.inp.nx + self.inp.ng):
            for j in range(self.inp.ng, self.inp.ny + self.inp.ng):
                for icomp in range(self.c.NUMQ):
                    
                    # Boundary-aware flux correction for all conservative variables
                    if (icomp == self.c.RHOCOMP and (i == self.inp.ng or i == self.inp.nx + self.inp.ng - 1) and
                        (self.inp.bcs_lo[0] == "neumann" or self.inp.bcs_hi[0] == "neumann")):
                        
                        # Apply consistent flux correction to all conservative variables
                        # at boundary-adjacent cells to maintain thermodynamic consistency
                        delta_x = self._compute_conservative_flux_delta(
                            right_flux[icomp, i, j], left_flux[icomp, i, j], i
                        )
                    else:
                        # Standard flux difference for interior cells
                        delta_x = right_flux[icomp, i, j] - left_flux[icomp, i, j]
                    
                    # Keep y-direction commented out as it was originally
                    # delta_y = top_flux[icomp, i, j] - bottom_flux[icomp, i, j]
                    
                    delta = (self.dt / self.inp.dx) * delta_x
                    # + (self.dt / self.inp.dy) * delta_y
                    
                    print(f"FLUX DELTA FOR {i}, {j}", delta_x)
                    print("TOTAL DELTA", delta)
                    print()
                        
                    consU[icomp, i, j] = consU[icomp, i, j] - delta
        
        self.grid[:] = self.euler.cons_to_prim(consU)
        # self.grid[:, 1] = 1000
        self.plot_a(title="RIGHT BEFORE LORENTZ RIGHT AFTER FLUX")
        
        self._apply_lorentz_source_terms(consU)
        self.grid[:] = self.euler.cons_to_prim(consU)
        
        self.plot_a("AFTER LORENTZ")
        
        # Apply BCs again after flux updates (flux also modifies interior cells only)
        self.bcs.apply_bcs()
        self.plot_a("AFTER FINAL BCS")
        
        ## --ELECTRIC FIELD UPDATE--
        charge_density = self.get_charge_density()
        self.fields.add_charge_density(charge_density) # -!- TOGGLE -!-
        
        self.fields.update_E()
        
        # TODO: call self.ebs.apply_ebs() once embedded boundaries are brought in
    
    
    def _apply_lorentz_source_terms(self, consU_new):

        E = self.fields.E
        B = self.fields.B
        V = self._get_V()
        
        charge_density = self.params.charge * self.grid[self.c.RHOCOMP] / self.params.mass
        
        ## --MOMENTUM UPDATE--
        # lorentz_force = (E + np.cross(V, B))
        # x_mom_source = charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] * lorentz_force[:, :, 0]
        # y_mom_source = charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] * lorentz_force[:, :, 0]
        
        # Apply Lorentz force: F = ρq(E + v×B) to both x and y momentum
        x_mom_source = charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] * E[:, :, 0]
        y_mom_source = charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] * E[:, :, 1]
        
        consU_new[self.c.MUCOMP, self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] += x_mom_source
        consU_new[self.c.MVCOMP, self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] += y_mom_source

        print(f"!@! Electromagnetic coupling strengths:")
        print(f"!@! Max |Ex|: {np.max(np.abs(E[:, :, 0])):.3e}")
        print(f"!@! Max |Ey|: {np.max(np.abs(E[:, :, 1])):.3e}")
        print(f"!@! Max |x_mom_source|: {np.max(np.abs(x_mom_source)):.3e}")
        print(f"!@! Max |y_mom_source|: {np.max(np.abs(y_mom_source)):.3e}")
        print(f"!@! Max |charge_density|: {np.max(np.abs(charge_density)):.3e}")
    
        # --ENERGY UPDATE--
        # V = self._get_V()
        # energy_source = charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] * \
        #     (E[:, :, 0] * V[:, :, 0] + E[:, :, 1] * V[:, :, 1] + E[:, :, 2] * V[:, :, 2]) # cursed vector dot product on two (100, 100, 3 matricies)
        
        # Energy source: ρq(E·v) = ρq(Ex*vx + Ey*vy)  
        prim_new = self.euler.cons_to_prim(consU_new)
        vx = prim_new[self.c.UCOMP][self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        vy = prim_new[self.c.VCOMP][self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
        
        energy_source = charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] * \
            (E[:, :, 0] * vx + E[:, :, 1] * vy)
            
        print(f"!@! Max |energy_source|: {np.max(np.abs(energy_source)):.3e}")
        print(f"!@! Energy source range: [{np.min(energy_source):.3e}, {np.max(energy_source):.3e}]")
        
        consU_new[self.c.ECOMP, self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] += energy_source
        
    
    def get_charge_density(self):
        charge_density = self.params.charge * self.get_number_density()
        return charge_density
    
    
    def get_number_density(self):
        number_density = self.grid[self.c.RHOCOMP] / self.params.mass
        return number_density
    
    
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
        
        w_array = np.zeros_like(self.grid[self.c.UCOMP])
        V = np.array([self.grid[self.c.UCOMP], self.grid[self.c.VCOMP], w_array])
        
        # change (3, 104, 104) to (100, 100, 3)
        V = np.transpose(V, (1, 2, 0))
        V = V[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng:]
        
        return V
    
    def _compute_conservative_flux_delta(self, flux_plus, flux_minus, i):
        """
        Conservative flux correction for all variables at boundary-adjacent cells with Neumann BC.
        
        The issue: Ghost cells are artificially smoothed by Neumann BC, creating artificial
        gradients that contaminate flux calculations. This affects all conservative variables
        and creates inconsistencies when converting back to primitive variables.
        
        The fix: Apply consistent flux dampening to all conservative variables at boundary-
        adjacent cells to maintain thermodynamic consistency and prevent primitive variable
        discontinuities caused by inconsistent conservative updates.
        
        Physics: Neumann BC represents ∂u/∂n = 0, so artificial gradients from ghost cells
        should be dampened while preserving interior physics.
        
        Args:
            flux_plus: Flux on the positive side (right)
            flux_minus: Flux on the negative side (left)
            i: Cell index in x-direction
            
        Returns:
            Corrected flux difference for all conservative variables
        """
        # Left boundary-adjacent cell
        if i == self.inp.ng and self.inp.bcs_lo[0] == "neumann":
            # flux_minus involves ghost cells, so reduce its influence
            # Apply consistent 50% weighting to all conservative variables
            return flux_plus - 0.5 * flux_minus
            
        # Right boundary-adjacent cell  
        elif i == self.inp.nx + self.inp.ng - 1 and self.inp.bcs_hi[0] == "neumann":
            # flux_plus involves ghost cells, so reduce its influence
            # Apply consistent 50% weighting to all conservative variables
            return 0.5 * flux_plus - flux_minus
        
        # Fallback to standard calculation for interior cells
        return flux_plus - flux_minus
    