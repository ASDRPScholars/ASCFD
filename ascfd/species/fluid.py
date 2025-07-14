from ascfd.grid import Grid2D
from ascfd.constants import Constants
from ascfd.euler import Euler

import ascfd.ics as ics

import numpy as np


class FluidSpecies:
    def __init__(self, params, a_inputs, dt):
        self.c = Constants(a_inputs)
        self.euler = Euler(self.c)
        
        self.inp = a_inputs
        self.params = params
        
        self.dt = dt

        self.grid_data = Grid2D(self.inp.xlim, self.inp.ylim, self.inp.nx, self.inp.ny, self.inp.numghosts, self.c.NUMQ)
        
        self.apply_ics()
        
        
    def update(self):
        
        consU = self.euler.prim_to_cons(self.grid_data)
        consU_new = self.euler.prim_to_cons(self.grid_data)
        
        _, right_flux, left_flux, top_flux, bottom_flux = self.flux.getFlux(self.grid_data.grid, self.grid_data.Nx, self.grid_data.Ny, self.grid_data.Nghost)
        
        for i in range(self.grid_data.Nghost, self.grid_data.Nx + self.grid_data.Nghost):
            for j in range(self.grid_data.Nghost, self.grid_data.Ny + self.grid_data.Nghost):
                for icomp in range(self.c.NUMQ):
                    consU_new[icomp, i, j] = consU[icomp, i, j] - (
                        (self.dt / self.grid_data.dx) * (right_flux[icomp, i, j] - left_flux[icomp, i, j]) +
                        (self.dt / self.grid_data.dy) * (top_flux[icomp, i, j] - bottom_flux[icomp, i, j]))
                    
        self.grid_data = self.euler.cons_to_prim(consU_new)
    
    def get_charge_density(self):
        pass
    
    def apply_ics(self):
        if self.inp.system == "euler2d":
            if self.inp.ics == "diagonal_advection":
                self.grid_data.fill_grid(ics.diagonal_advection_2d)
            elif self.inp.ics == "kelvin_helmholtz":
                self.grid_data.fill_grid(ics.kelvin_helmholtz_2d)
            elif self.inp.ics == "double_mach_reflection":
                self.grid_data.fill_grid(ics.double_mach_reflection_2d)
            elif self.inp.ics == "riemann_problem":
                self.grid_data.fill_grid(ics.riemann_2d)
            else:
                raise RuntimeError("[FLUID] ICS not valid.")
           
        elif self.inp.system == "mhd2d":
            if self.inp.ics == "orszag_tang":
                self.grid_data.fill_grid(ics.orszag_tang_2d)
           
        else:
            raise RuntimeError("[FLUID] ICS not valid.")