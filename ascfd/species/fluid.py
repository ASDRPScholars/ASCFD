from ascfd.grid import Grid2D
from ascfd.constants import Constants
from ascfd.euler import Euler
from ascfd.inputs import Inputs

import ascfd.ics as ics

import numpy as np


class FluidSpecies:
    def __init__(self, params, a_inputs: Inputs, dt):
        self.c = Constants(a_inputs)
        self.euler = Euler(self.c)
        
        self.inp = a_inputs
        self.params = params
        
        self.dt = dt

        # TODO: move Grid2D functionality into each species class? or Grid -> FluidGrid, FieldGrid polymorphism?
        # self.grid_data = Grid2D(self.inp.xlim, self.inp.ylim, self.inp.nx, self.inp.ny, self.inp.numghosts, self.c.NUMQ)
        
        self.grid_data = np.zeros((self.c.NUMQ, self.inp.nx + 2 * self.inp.numghosts, self.inp.nx + 2 * self.inp.numghosts))
        
        # boring logistics stuff for apply_ics()
        dx = (self.inp.xlim[1] - self.inp.xlim[0]) / (self.inp.nx - 1)
        dy = (self.inp.ylim[1] - self.inp.ylim[0]) / (self.inp.ny - 1)
        x = np.linspace(self.inp.xlim[0] - dx * self.inp.numghosts, self.inp.xlim[1] + dx * self.inp.numghosts, self.inp.nx + 2 * self.inp.numghosts)
        y = np.linspace(self.inp.ylim[0] - dy * self.inp.numghosts, self.inp.ylim[1] + dy * self.inp.numghosts, self.inp.ny + 2 * self.inp.numghosts)
        
        self.mesh_x, self.mesh_y = np.meshgrid(x, y)
        
        self.apply_ics()
        
        
    def apply_ics(self):
            
        if self.inp.system == "euler2d":
            if self.inp.ics == "diagonal_advection":
                f = ics.diagonal_advection_2d
            elif self.inp.ics == "kelvin_helmholtz":
                f = ics.kelvin_helmholtz_2d
            elif self.inp.ics == "double_mach_reflection":
                f = ics.double_mach_reflection_2d
            elif self.inp.ics == "riemann_problem":
                f = ics.riemann_2d
            else:
                raise RuntimeError("[FLUID] ICS not valid.")
           
        elif self.inp.system == "mhd2d":
            if self.inp.ics == "orszag_tang":
                f = ics.orszag_tang_2d
           
        else:
            raise RuntimeError("[FLUID] ICS not valid.")
        
        for var in range(self.num_vars):
            self.grid_data[var] = f(self.mesh_x, self.mesh_y, var)
        
        
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
    
    
    def check_grid(self, constants, prim=False, cons=False):
        # Check for negative or invalid values in the grid
        for i in range(self.Nx + 2 * self.Nghost):
            for j in range(self.Ny + 2 * self.Nghost):
                if prim:
                    # Check for negative pressure
                    if self.grid[constants.PCOMP, i, j] <= 0:
                        print(f"Negative Pressure - Bad cell: ({i}, {j})")
                        assert False

                    # Check for negative density
                    if self.grid[constants.RHOCOMP, i, j] <= 0:
                        print(f"Negative Density - Bad cell: ({i}, {j})")
                        assert False

                if cons:
                    # Check for negative energy
                    if self.grid[constants.ECOMP, i, j] <= 0:
                        print(f"Negative Energy - Bad cell: ({i}, {j})")
                        assert False

                # Check for NaN values
                for icomp in range(constants.NUMQ):
                    if np.isnan(self.grid[icomp, i, j]):
                        print(f"NaN value - Bad cell: ({i}, {j}), component: {icomp}")
                        assert False
                        