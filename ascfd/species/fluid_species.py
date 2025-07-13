from ascfd.grid import Grid2D
from ascfd.constants import Constants
from ascfd.euler import Euler

import numpy as np


class FluidSpecies:
    def __init__(self, params, a_inputs):
        self.c = Constants(a_inputs)
        self.euler = Euler(self.c)
        
        self.inp = a_inputs
        self.params = params

        self.grid_data = Grid2D(self.inp.xlim, self.inp.ylim, self.inp.nx, self.inp.ny, self.inp.numghosts, self.c.NUMQ)
        
    def update(self):
        
        consU = self.euler.prim_to_cons(self.grid_data)
        consU_new = self.euler.prim_to_cons(self.grid_data)
        
        _, right_flux, left_flux, top_flux, bottom_flux = self.flux.getFlux(self.grid_data.grid, self.grid_data.Nx, self.grid_data.Ny, self.grid_data.Nghost)
        
        for i in range(self.grid_data.Nghost, self.grid_data.Nx + self.grid_data.Nghost):
            for j in range(self.grid_data.Nghost, self.grid_data.Ny + self.grid_data.Nghost):
                for icomp in range(self.c.NUMQ):
                    consU_new[icomp, i, j] = consU[icomp, i, j] - (
                        (self.params.dt / self.grid_data.dx) * (right_flux[icomp, i, j] - left_flux[icomp, i, j]) +
                        (self.params.dt / self.grid_data.dy) * (top_flux[icomp, i, j] - bottom_flux[icomp, i, j]))
                    
        self.grid_data = self.euler.cons_to_prim(consU_new)
    
    def get_charge_density(self):
        pass
    