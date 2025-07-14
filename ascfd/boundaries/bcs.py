from ascfd.logistics.inputs import Inputs
from ascfd.logistics.constants import Constants
from typing import Callable
import numpy as np

"""
Notes:
        - bcs -> boundary conditions
        - ghost cells -> extra layers of cells around the grid to calculate bcs using numerical methods
        - dirichlet boundary -> the value in the bcs is constant
        - neumann boundaries -> derivative at bcs is zero
"""
class BoundaryConditions:
    
    def __init__(self, grid, types_lo: tuple[str, str], types_hi: tuple[str, str], a_inputs: Inputs) -> None:
        # store boundary condition types and associate grid object
        self.types_lo = types_lo
        self.types_hi = types_hi
        self.grid = grid
        self.inp = a_inputs
        self.c = Constants(a_inputs)

        # boundary condition function mappings
        self.f_lo = [self.null_bcs, self.null_bcs]
        self.f_hi = [self.null_bcs, self.null_bcs]

        # dirichlet & neumann for low boundary conditions
        for idim in range(2):  # two dimensions
            if self.types_lo[idim] == "dirichlet":
                self.f_lo[idim] = self.dirichlet_lo
            elif self.types_lo[idim] == "neumann":
                self.f_lo[idim] = self.neumann_lo
            elif self.types_lo[idim] == "periodic":
                self.f_lo[idim] = self.periodic_lo
            else:
                # error if unsupported boundary condition type is provided
                raise RuntimeError(f"BC Lo Type not supported: {self.types_lo[idim]}")

            # dirichlet & neumann for high boundary conditions
            if self.types_hi[idim] == "dirichlet":
                self.f_hi[idim] = self.dirichlet_hi
            elif self.types_hi[idim] == "neumann":
                self.f_hi[idim] = self.neumann_hi
            elif self.types_hi[idim] == "periodic":
                self.f_hi[idim] = self.periodic_hi
            else:
                # error if unsupported boundary condition type is provided
                raise RuntimeError(f"BC Hi Type not supported: {self.types_hi[idim]}")

    def apply_bcs(self) -> None:
         # apply boundary conditions for both low and high boundaries
        self.apply_lo()
        self.apply_hi()

    def apply_lo(self) -> None:
        # apply low boundary conditions for all dimensions
        for idim in range(2): # two dimensions
            self.f_lo[idim](self.grid, idim)
        
    def apply_hi(self) -> None:
        # apply high boundary conditions for all dimensions
        for idim in range(2): # two dimensions
            self.f_hi[idim](self.grid, idim)

    def null_bcs(self, grid, dim: int) -> None:
        # function for unsupported or uninitialized boundary conditions
        raise RuntimeError("Null boundary conditions should not be called.")

    
    def dirichlet_lo(self, grid, dim: int) -> None:
        # dirichlet boundary condition at the low boundary
        for var in range(self.c.NUMQ): # all variables in the grid
            if dim == 0: # low boundary in x-direction
                for i in range(self.inp.numghosts): # ghost cells
                    for j in range(self.inp.numghosts, self.inp.ny + self.inp.numghosts): # valid y-range
                        grid[var, i, j] = grid[var, self.inp.numghosts, j]
            else:  # low boundary in y-direction
                for i in range(self.inp.numghosts, self.inp.nx + self.inp.numghosts): # valid x-range
                    for j in range(self.inp.numghosts): # ghost cells
                        grid[var, i, j] = grid[var, i, self.inp.numghosts]

    
    def dirichlet_hi(self, grid, dim: int) -> None:
        # dirichlet boundary condition at the high boundary
        for var in range(self.c.NUMQ): # all variables in the grid
            if dim == 0:  # high boundary in x-direction
                for i in range(self.inp.nx + self.inp.numghosts, self.inp.nx + 2*self.inp.numghosts): # ghost x-range
                    for j in range(self.inp.numghosts, self.inp.ny + self.inp.numghosts):  # valid y-range
                        grid[var, i, j] = grid[var, self.inp.nx + self.inp.numghosts - 1, j]
            else:  # high boundary in y-direction
                for i in range(self.inp.numghosts, self.inp.nx + self.inp.numghosts): # valid x-range
                    for j in range(self.inp.ny + self.inp.numghosts, self.inp.ny + 2*self.inp.numghosts): #ghost x-range
                        grid[var, i, j] = grid[var, i, self.inp.ny + self.inp.numghosts - 1]

    
    def neumann_lo(self, grid, dim: int) -> None:
        # neumann boundary condition at the low boundary
        for var in range(self.c.NUMQ): # all variables in the grid
            if dim == 0:  # low boundary in x-direction
                for i in range(self.inp.numghosts): # ghost cells
                    for j in range(self.inp.numghosts, self.inp.ny + self.inp.numghosts):  # valid y-range
                        grid[var, i, j] = grid[var, self.inp.numghosts, j]
            else:  # low boundary in y-direction
                for i in range(self.inp.numghosts, self.inp.nx + self.inp.numghosts): # valid x-range
                    for j in range(self.inp.numghosts): # ghost cells
                        grid[var, i, j] = grid[var, i, self.inp.numghosts]

    
    def neumann_hi(self, grid, dim: int) -> None:
         # neumann boundary condition at the high boundary
        for var in range(self.c.NUMQ): # all variables in the grid
            if dim == 0:  # high boundary in x-direction
                for i in range(self.inp.nx + self.inp.numghosts, self.inp.nx + 2*self.inp.numghosts): # ghost cells
                    for j in range(self.inp.numghosts, self.inp.ny + self.inp.numghosts): # valid y-range
                        grid[var, i, j] = grid[var, self.inp.nx + self.inp.numghosts - 1, j]
            else:  # y-direction
                for i in range(self.inp.numghosts, self.inp.nx + self.inp.numghosts):
                    for j in range(self.inp.ny + self.inp.numghosts, self.inp.ny + 2*self.inp.numghosts): # valid x-range
                        grid[var, i, j] = grid[var, i, self.inp.ny + self.inp.numghosts - 1] # ghost cells

    
    def periodic_lo(self, grid, dim: int) -> None:
        # periodic boundary condition at the low boundary
        for var in range(self.c.NUMQ):  # all variables in the grid
            if dim == 0:  # low boundary in x-direction
                for i in range(self.inp.numghosts): # ghost cells in low x-range
                    for j in range(self.inp.numghosts, self.inp.ny + self.inp.numghosts): # valid y-range
                        grid[var, i, j] = grid[var, self.inp.nx + i, j]
            else:  # low boundary in y-direction
                for i in range(self.inp.numghosts, self.inp.nx + self.inp.numghosts): #valid x-range
                    for j in range(self.inp.numghosts): # ghost cells in low y-range
                        grid[var, i, j] = grid[var, i, self.inp.ny + j]

    
    def periodic_hi(self, grid, dim: int) -> None:
        # periodic boundary condition at the high boundary
        for var in range(self.c.NUMQ): #all variables in the grid
            if dim == 0:  # high boundary in x-direction
                for i in range(self.inp.nx + self.inp.numghosts, self.inp.nx + 2*self.inp.numghosts): # ghost cells in high x-range
                    for j in range(self.inp.numghosts, self.inp.ny + self.inp.numghosts):  # valid y-range
                        grid[var, i, j] = grid[var, i - self.inp.nx, j]
            else:  # H=high boundary in y-direction
                for i in range(self.inp.numghosts, self.inp.nx + self.inp.numghosts): #valid x-range
                    for j in range(self.inp.ny + self.inp.numghosts, self.inp.ny + 2*self.inp.numghosts): # ghost cells in high y-range
                        grid[var, i, j] = grid[var, i, j - self.inp.ny]
