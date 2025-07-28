from ascfd.inputs import Inputs
from ascfd.fluid.constants import FluidConstants
from typing import Callable
import numpy as np

"""
Notes:
        - bcs -> boundary conditions
        - ghost cells -> extra layers of cells around the grid to calculate bcs using numerical methods
        - dirichlet boundary -> the value in the bcs is constant
        - neumann boundaries -> derivative at bcs is zero
"""
class FluidBoundaryConditions:
    
    def __init__(self, grid, types_lo: tuple[str, str], types_hi: tuple[str, str], a_inputs: Inputs) -> None:
        # store boundary condition types and associate grid object
        self.types_lo = types_lo
        self.types_hi = types_hi
        self.grid = grid
        self.inp = a_inputs
        self.c = FluidConstants(a_inputs)

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
            elif self.types_lo[idim] == "inflow":
                self.f_lo[idim] = self.inflow_lo
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
            elif self.types_hi[idim] == "inflow":
                self.f_hi[idim] = self.inflow_hi
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
                for i in range(self.inp.ng): # ghost cells
                    for j in range(self.inp.ng, self.inp.ny + self.inp.ng): # valid y-range
                        grid[var, i, j] = grid[var, self.inp.ng, j]
            else:  # low boundary in y-direction
                for i in range(self.inp.ng, self.inp.nx + self.inp.ng): # valid x-range
                    for j in range(self.inp.ng): # ghost cells
                        grid[var, i, j] = grid[var, i, self.inp.ng]

    
    def dirichlet_hi(self, grid, dim: int) -> None:
        # dirichlet boundary condition at the high boundary
        for var in range(self.c.NUMQ): # all variables in the grid
            if dim == 0:  # high boundary in x-direction
                for i in range(self.inp.nx + self.inp.ng, self.inp.nx + 2*self.inp.ng): # ghost x-range
                    for j in range(self.inp.ng, self.inp.ny + self.inp.ng):  # valid y-range
                        grid[var, i, j] = grid[var, self.inp.nx + self.inp.ng - 1, j]
            else:  # high boundary in y-direction
                for i in range(self.inp.ng, self.inp.nx + self.inp.ng): # valid x-range
                    for j in range(self.inp.ny + self.inp.ng, self.inp.ny + 2*self.inp.ng): #ghost x-range
                        grid[var, i, j] = grid[var, i, self.inp.ny + self.inp.ng - 1]

    
    def neumann_lo(self, grid, dim: int) -> None:
        print("!BCS! CALLED NEUMANN LO")
        print("!BCS! CAN I READ THE MIDDLE OF GRID??", grid[1, 52, 52])
        # neumann boundary condition at the low boundary
        for var in range(self.c.NUMQ): # all variables in the grid
            if dim == 0:  # low boundary in x-direction
                for i in range(self.inp.ng): # ghost cells
                    for j in range(self.inp.ng, self.inp.ny + self.inp.ng):  # valid y-range
                        grid[var, i, j] = grid[var, self.inp.ng, j]
                        print(f"set grid[{var}, {i}, {j}] from {grid[var, i, j]} to {grid[var, self.inp.ng, j]}")
            else:  # low boundary in y-direction
                for i in range(self.inp.ng, self.inp.nx + self.inp.ng): # valid x-range
                    for j in range(self.inp.ng): # ghost cells
                        grid[var, i, j] = grid[var, i, self.inp.ng]

    
    def neumann_hi(self, grid, dim: int) -> None:
        print("!BCS! CALLED NEUMANN HI")
         # neumann boundary condition at the high boundary
        for var in range(self.c.NUMQ): # all variables in the grid
            if dim == 0:  # high boundary in x-direction
                for i in range(self.inp.nx + self.inp.ng, self.inp.nx + 2*self.inp.ng): # ghost cells
                    for j in range(self.inp.ng, self.inp.ny + self.inp.ng): # valid y-range
                        grid[var, i, j] = grid[var, self.inp.nx + self.inp.ng - 1, j]
            else:  # y-direction
                for i in range(self.inp.ng, self.inp.nx + self.inp.ng):
                    for j in range(self.inp.ny + self.inp.ng, self.inp.ny + 2*self.inp.ng): # valid x-range
                        grid[var, i, j] = grid[var, i, self.inp.ny + self.inp.ng - 1] # ghost cells

    
    def periodic_lo(self, grid, dim: int) -> None:
        # periodic boundary condition at the low boundary
        for var in range(self.c.NUMQ):  # all variables in the grid
            if dim == 0:  # low boundary in x-direction
                for i in range(self.inp.ng): # ghost cells in low x-range
                    for j in range(self.inp.ng, self.inp.ny + self.inp.ng): # valid y-range
                        grid[var, i, j] = grid[var, self.inp.nx + i, j]
            else:  # low boundary in y-direction
                for i in range(self.inp.ng, self.inp.nx + self.inp.ng): #valid x-range
                    for j in range(self.inp.ng): # ghost cells in low y-range
                        grid[var, i, j] = grid[var, i, self.inp.ny + j]

    
    def periodic_hi(self, grid, dim: int) -> None:
        # periodic boundary condition at the high boundary
        for var in range(self.c.NUMQ): #all variables in the grid
            if dim == 0:  # high boundary in x-direction
                for i in range(self.inp.nx + self.inp.ng, self.inp.nx + 2*self.inp.ng): # ghost cells in high x-range
                    for j in range(self.inp.ng, self.inp.ny + self.inp.ng):  # valid y-range
                        grid[var, i, j] = grid[var, i - self.inp.nx, j]
            else:  # H=high boundary in y-direction
                for i in range(self.inp.ng, self.inp.nx + self.inp.ng): #valid x-range
                    for j in range(self.inp.ny + self.inp.ng, self.inp.ny + 2*self.inp.ng): # ghost cells in high y-range
                        grid[var, i, j] = grid[var, i, j - self.inp.ny]
                        
                        
    def inflow_lo(self, grid, dim: int) -> None:

        if dim == 0:  # low boundary in x-direction
            for i in range(self.inp.ng): # ghost cells
                for j in range(self.inp.ng, self.inp.ny + self.inp.ng):  # valid y-range
                    grid[self.c.RHOCOMP, i, j] = 1e-9
                    grid[self.c.PCOMP, i, j] = 150
                    grid[self.c.UCOMP, i, j] = 10
                    grid[self.c.VCOMP, i, j] = 0
                    
        else:  # low boundary in y-direction
            for i in range(self.inp.ng, self.inp.nx + self.inp.ng): # valid x-range
                for j in range(self.inp.ng): # ghost cells
                    grid[self.c.RHOCOMP, i, j] = 1e-9
                    grid[self.c.PCOMP, i, j] = 6.5217391304e40
                    grid[self.c.UCOMP, i, j] = 0
                    grid[self.c.VCOMP, i, j] = 10
                    
                    
    def inflow_hi(self, grid, dim: int) -> None:

        # neumann boundary condition at the low boundary

        if dim == 0:  # hi boundary in x-direction
            # pass
            print("CALLING HIGH INFLOW X")
            for i in range(self.inp.nx + self.inp.ng, self.inp.nx + 2*self.inp.ng): # ghost cells in high x-range
                for j in range(self.inp.ng, self.inp.ny + self.inp.ng):  # valid y-range
                    grid[self.c.RHOCOMP, i, j] = 1e-9
                    grid[self.c.PCOMP, i, j] = 150
                    grid[self.c.UCOMP, i, j] = -10
                    grid[self.c.VCOMP, i, j] = 0
                    
        else:  # hi boundary in y-direction
            # print("!!! WELRE ACLLING THIS RIGHT")
            for i in range(self.inp.ng, self.inp.nx + self.inp.ng): #valid x-range
                for j in range(self.inp.ny + self.inp.ng, self.inp.ny + 2*self.inp.ng): # ghost cells in high y-range
                    grid[self.c.RHOCOMP, i, j] = 1e-10
                    grid[self.c.PCOMP, i, j] = 6.5217391304e39
                    grid[self.c.UCOMP, i, j] = 0
                    grid[self.c.VCOMP, i, j] = -10
                    