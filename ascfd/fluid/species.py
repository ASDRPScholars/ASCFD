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
        self.ics = FluidInitialConditions(self.grid, self.inp)
        
        self.bcs.apply_bcs()
        
        self.check_grid(self.c)
        
        # TODO: DELETE UNNECESSARY VALUES HERE LATER WHEN ICS.APPLY_ICS() IS DONE
        # boring ascfd.logistics stuff for apply_ics()
        
        self.grid = self.ics.apply_ics()
        
        print("HI OK THIS IS DENSITY AT ICS:", self.grid[self.c.RHOCOMP])
        
        
    def update(self):
        
        E = self.fields.E
        B = self.fields.B
        
        consU = self.euler.prim_to_cons(self.grid)
        consU_new = self.euler.prim_to_cons(self.grid)
        
        _, right_flux, left_flux, top_flux, bottom_flux = self.flux.getFlux(self.grid, self.inp.nx, self.inp.ny, self.inp.numghosts)
        
        self.bcs.apply_bcs()
        
        for i in range(self.inp.numghosts, self.inp.nx + self.inp.numghosts):
            for j in range(self.inp.numghosts, self.inp.ny + self.inp.numghosts):
                for icomp in range(self.c.NUMQ):
                    
                    delta = (
                        (self.dt / self.inp.dx) * (right_flux[icomp, i, j] - left_flux[icomp, i, j]) +
                        (self.dt / self.inp.dy) * (top_flux[icomp, i, j] - bottom_flux[icomp, i, j]))
                    
                    print("RIGHT LEFT TOP BOTTOM:", right_flux[icomp, i, j], left_flux[icomp, i, j], top_flux[icomp, i, j], bottom_flux[icomp, i, j])
                    
                    if icomp in [self.c.RHOCOMP, self.c.ECOMP]:
                        consU_new[icomp, i, j] = max(1e-6, consU[icomp, i, j] - delta)
                    else:
                        consU_new[icomp, i, j] = consU[icomp, i, j] - delta
                    
                    print("DELTA:", delta)
        
        plt.figure()
        plt.imshow(self.grid[self.c.RHOCOMP])
        plt.title("electron density")
        plt.show()
        
        self.grid = self.euler.cons_to_prim(consU_new)
        
        self.bcs.apply_bcs()
        
        charge_density = self.get_charge_density()
        
        # ELECTRIC FIELD UPDATE
        self.fields.add_charge_density(charge_density)
        
        print("ELECTRONS ADDED CHARGE DENSITY:", charge_density)
        
        self.fields.update_E()
        
        # TODO: call self.ebs.apply_ebs() once embedded boundaries are brought in
        

    
    
    def get_charge_density(self):
        charge_density = self.params.charge * self.get_number_density()
        print("HI THIS IS DENSITY:", self.grid[self.c.RHOCOMP])
        print("HI THIS IS NUMBER DENSITY:", self.get_number_density())
        print("HI THIS IS CHARGE DENSITY:", charge_density)
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
        for i in range(self.inp.nx + 2 * self.inp.numghosts):
            for j in range(self.inp.ny + 2 * self.inp.numghosts):
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
                        