from ascfd.fluid.constants import FluidConstants
from ascfd.fluid.euler import FluidEuler
from ascfd.fluid.ics import FluidInitialConditions
from ascfd.logistics.inputs import Inputs
from ascfd.species.params import SpeciesParams
from ascfd.fluid.bcs import FluidBoundaryConditions
from ascfd.fluid.flux import FluidFlux
from ascfd.fields.fields import Fields

import ascfd.fluid.ics as ics

import numpy as np


class FluidSpecies:
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields):
        self.c = FluidConstants(a_inputs)
        self.euler = FluidEuler(self.c)
        self.flux = FluidFlux(self.c, a_inputs.flux)
        
        self.fields = Fields(a_inputs)
        
        self.inp = a_inputs
        self.params = params
        self.dt = None
        
        self.dx = (self.inp.xlim[1] - self.inp.xlim[0]) / (self.inp.nx - 1)
        self.dy = (self.inp.ylim[1] - self.inp.ylim[0]) / (self.inp.ny - 1)
        self.grid_x = np.linspace(self.inp.xlim[0] - self.dx * self.inp.numghosts, self.inp.xlim[1] + self.dx * self.inp.numghosts, self.inp.nx + 2 * self.inp.numghosts)
        self.grid_y = np.linspace(self.inp.ylim[0] - self.dy * self.inp.numghosts, self.inp.ylim[1] + self.dy * self.inp.numghosts, self.inp.ny + 2 * self.inp.numghosts)
        
        self.nx_with_ghosts = self.inp.nx + 2 * self.inp.numghosts
        self.ny_with_ghosts = self.inp.ny + 2 * self.inp.numghosts
        
        self.grid = np.zeros((self.c.NUMQ, self.nx_with_ghosts, self.ny_with_ghosts))
        
        self.bcs = FluidBoundaryConditions(self.grid, self.inp.bcs_lo, self.inp.bcs_hi, self.inp)
        self.ics = FluidInitialConditions(self.grid, self.inp)
        
        self.bcs.apply_bcs()
        
        self.check_grid(self.c)
        
        # TODO: DELETE UNNECESSARY VALUES HERE LATER WHEN ICS.APPLY_ICS() IS DONE
        # boring ascfd.logistics stuff for apply_ics()
        
        self.ics.apply_ics()
        
        
    def update(self):
        
        consU = self.euler.prim_to_cons(self.grid)
        consU_new = self.euler.prim_to_cons(self.grid)
        
        _, right_flux, left_flux, top_flux, bottom_flux = self.flux.getFlux(self.grid, self.inp.nx, self.inp.ny, self.inp.numghosts)
        
        self.bcs.apply_bcs()
        
        for i in range(self.inp.numghosts, self.inp.nx + self.inp.numghosts):
            for j in range(self.inp.numghosts, self.inp.ny + self.inp.numghosts):
                for icomp in range(self.c.NUMQ):
                    consU_new[icomp, i, j] = consU[icomp, i, j] - (
                        (self.dt / self.dx) * (right_flux[icomp, i, j] - left_flux[icomp, i, j]) +
                        (self.dt / self.dy) * (top_flux[icomp, i, j] - bottom_flux[icomp, i, j]))
                    
        self.bcs.apply_bcs()
        
        charge_density = self.get_charge_density()
        self.fields.update_E(charge_density)
        
        # TODO: call self.ebs.apply_ebs() once embedded boundaries are brought in
        
        self.grid = self.euler.cons_to_prim(consU_new)
    
    
    def get_charge_density(self):
        pass
    
    
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
                        