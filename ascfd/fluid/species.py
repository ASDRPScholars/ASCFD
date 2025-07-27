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
        
        self.bcs = FluidBoundaryConditions(self.grid, self.inp.bcs_lo, self.inp.bcs_hi, self.inp, embedded_boundaries="hallthruster")
        self.ics = FluidInitialConditions(self.grid, self.inp, self.params)
        
        self.bcs.apply_bcs()
        
        self.check_grid(self.c)
        
        # TODO: DELETE UNNECESSARY VALUES HERE LATER WHEN ICS.APPLY_ICS() IS DONE
        # boring ascfd.logistics stuff for apply_ics()
        
        self.grid = self.ics.apply_ics()
        
        # print("HI OK THIS IS DENSITY AT ICS:", self.grid[self.c.RHOCOMP])
        
        
    def update(self):
        
        print("dt is", self.dt)
        consU = self.euler.prim_to_cons(self.grid)
        consU_new = self.euler.prim_to_cons(self.grid)
        
        # print("!START! MU", self.grid[self.c.MUCOMP])
        # print("!START! RHO", self.grid[self.c.RHOCOMP])
        
        _, right_flux, left_flux, top_flux, bottom_flux = self.flux.getFlux(self.grid, self.inp.nx, self.inp.ny, self.inp.ng)
        
        self.bcs.apply_bcs()
        
        for i in range(self.inp.ng, self.inp.nx + self.inp.ng):
            for j in range(self.inp.ng, self.inp.ny + self.inp.ng):
                for icomp in range(self.c.NUMQ):
                    
                    delta = (
                        (self.dt / self.inp.dx) * (right_flux[icomp, i, j] - left_flux[icomp, i, j]) +
                        (self.dt / self.inp.dy) * (top_flux[icomp, i, j] - bottom_flux[icomp, i, j]))
                    
                    # print("RIGHT LEFT TOP BOTTOM:", right_flux[icomp, i, j], left_flux[icomp, i, j], top_flux[icomp, i, j], bottom_flux[icomp, i, j])
                    
                    # if icomp in [self.c.RHOCOMP, self.c.ECOMP]:
                    #     consU_new[icomp, i, j] = max(1e-6, consU[icomp, i, j] - delta)
                    # else:
                    consU_new[icomp, i, j] = consU[icomp, i, j] - delta
                    
                    # print("DELTA:", delta)
                    
        # print("!AFTER FLUX! MU TAKING IN", self.grid[self.c.MUCOMP])
        # print("!AFTER FLUX! RHO TAKING IN", self.grid[self.c.RHOCOMP])
        
        self._apply_lorentz_source_terms(consU_new)
        
        self.bcs.apply_bcs()
        
        plt.figure()
        plt.imshow(self.grid[self.c.RHOCOMP])
        plt.title("electron density")
        plt.show()
        
        fig, axs = plt.subplots(2, 2, figsize=(10, 10))  # 1 row, 3 columns
        axs = axs.flatten()
        
        # print("!FINAL BEFORE PLOT! MU TAKING IN", self.grid[self.c.MUCOMP])
        # print("!FINAL BEFORE PLOT! RHO TAKING IN", self.grid[self.c.RHOCOMP])

        # Plot charge density
        im0 = axs[0].imshow(self.grid[self.c.RHOCOMP], cmap="magma")
        axs[0].set_title("density")
        fig.colorbar(im0, ax=axs[0])

        # Plot Ex
        im1 = axs[1].imshow(self.grid[self.c.MUCOMP], cmap="magma")
        axs[1].set_title("mu")
        fig.colorbar(im1, ax=axs[1])

        # Plot Ey
        im2 = axs[2].imshow(self.grid[self.c.MVCOMP], cmap="magma")
        axs[2].set_title("mv")
        fig.colorbar(im2, ax=axs[2])
        
        # Plot E
        im3 = axs[3].imshow(self.grid[self.c.ECOMP], cmap="magma")
        axs[3].set_title("energy")
        fig.colorbar(im3, ax=axs[3])
        
        plt.tight_layout()
        plt.show()
        
        # print("!!AFTER BCS!! MU TAKING IN", self.grid[self.c.MUCOMP])
        # print("!!AFTER BCS!! RHO TAKING IN", self.grid[self.c.RHOCOMP])

        
        self.grid = self.euler.cons_to_prim(consU_new)
        
        # print("!!AFTER REAL FLUX UPDATE!! MU TAKING IN", self.grid[self.c.MUCOMP])
        # print("!!AFTER REAL FLUX UPDATE!! RHO TAKING IN", self.grid[self.c.RHOCOMP])
        
        # ELECTRIC FIELD UPDATE
        charge_density = self.get_charge_density()
        self.fields.add_charge_density(charge_density)
        
        # print("ELECTRONS ADDED CHARGE DENSITY:", charge_density)
        
        self.fields.update_E()
        
        # TODO: call self.ebs.apply_ebs() once embedded boundaries are brought in
    
    
    def _apply_lorentz_source_terms(self, consU_new):
        
        print("DEBUG mass density:", np.max(self.grid[self.c.RHOCOMP]))
        print("DEBUG params.mass:", self.params.mass)  
        print("DEBUG params.charge:", self.params.charge)
                
        n_substeps = 10
        dt_sub = self.dt / n_substeps
    
        # for i in range(n_substeps):
        E = self.fields.E
        B = self.fields.B
        V = self._get_V()
        
        charge_density = self.get_charge_density()
        
        print("charge_density range:", np.min(charge_density), np.max(charge_density))
        
        print("DEBUG charge density:", np.max(charge_density))
        print("DEBUG: max abs charge density:", np.max(np.abs(charge_density)))
        print("DEBUG: max charge density:", np.max(charge_density))  
        print("DEBUG: min charge density:", np.min(charge_density))
        
        # assert not np.isnan(E).any(), "NaN in E"
        # assert not np.isnan(B).any(), "NaN in B"
        # assert not np.isnan(V).any(), "NaN in V"
        
        lorentz_force = (E + np.cross(V, B))
        print("DEBUG: max abs lorentz force:", np.max(np.abs(lorentz_force)))
        
        ## MOMENTUM UPDATE
        x_mom_source = charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] * lorentz_force[:, :, 0]
        print("DEBUG: max abs x_mom_source:", np.max(np.abs(x_mom_source)))
        y_mom_source = charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] * lorentz_force[:, :, 1]
        
        # print("X MOM SOURCE TERMS:", x_mom_source)
        # print("Y MOM SOURCE TERMS:", y_mom_source)
        
        np.set_printoptions(threshold=sys.maxsize)
        
        # print("charge_density", charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng])
        # print("x lorentz force", lorentz_force[:, :, 0])
        # print("x_mom", np.max(x_mom_source))
        # print("y_mom", np.max(y_mom_source))
        
        # print("WHAT ARE WE FEEDING (X):", x_mom_source)
        # # assert np.any(lorentz_source_x >= 0)
        
        # print("WHAT ARE WE FEEDING (Y):", y_mom_source)
        # # assert np.any(lorentz_source_y >= 0)
        
        # !!C!!
        consU_new[self.c.MUCOMP, self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] += x_mom_source * self.dt
        consU_new[self.c.MVCOMP, self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] += y_mom_source * self.dt
        
        # print("!!C!! Max x_mom_source added:", np.max(np.abs(x_mom_source)))
        # print("!!C!! Max y_mom_source added:", np.max(np.abs(y_mom_source)))
        # print("!!C!! Max momentum after update:", np.max(np.abs(consU_new[self.c.MUCOMP])))
        # print("!!C!! Min/Max density:", np.min(consU_new[self.c.RHOCOMP]), np.max(consU_new[self.c.RHOCOMP]))

        # !!C!! 
        V = self._get_V()
        
        # print("!AFTER MOM UPDATE! MU TAKING IN", self.grid[self.c.MUCOMP])
        # print("!AFTER MOM UPDATE! RHO TAKING IN", self.grid[self.c.RHOCOMP])
        # print("!AFTER MOM UPDATE! V COMING OUT", V)
        
        # ENERGY UPDATE
        energy_source = charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] * (E[:, :, 0] * V[:, :, 0] + E[:, :, 1] * V[:, :, 1] + E[:, :, 2] * V[:, :, 2]) # cursed vector dot product on two (100, 100, 3 matricies)
        
        # # Current density
        # J_e = charge_density[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng] * V  # electron current density

        # # Joule heating (energy source)
        # energy_source = J_e[:,:,0] * E[:,:,0] + J_e[:,:,1] * E[:,:,1] + J_e[:,:,2] * E[:,:,2]
        
        #####
        
        # energy_source = np.clip(energy_source, -1e3, 1e3)
        
        # print("x energy", E[:, :, 0])
        # print("REAL x velocity", self.grid[self.c.MUCOMP])
        # print("x velocity", V[:, :, 0])
                
        # print("ENERGY source", energy_source)
        # print("ENERGY charge density", charge_density[self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng])
        # print("ENERGY E field", E[:, :, 0])
        print("ENERGY V field", V[:, :, 0])
        
        # print("energy", np.max(energy_source))
        
        # !!C!!
        consU_new[self.c.ECOMP, self.inp.ng:-self.inp.ng:, self.inp.ng:-self.inp.ng] += energy_source * self.dt
        
        print("!!C!! Max charge density:", np.max(np.abs(charge_density)))
        print("!!C!! Max E field:", np.max(np.abs(E)))
        print("!!C!! Max V field:", np.max(np.abs(V)))
        print("!!C!! Max Lorentz force:", np.max(np.abs(lorentz_force)))
        
    
    def get_charge_density(self):
        charge_density = self.params.charge * self.get_number_density()
        # print("HI THIS IS DENSITY:", self.grid[self.c.RHOCOMP])
        # print("HI THIS IS NUMBER DENSITY:", self.get_number_density())
        # print("HI THIS IS CHARGE DENSITY:", charge_density)
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
            
            self.grid = self.euler.cons_to_prim(consU)
            
            
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
    