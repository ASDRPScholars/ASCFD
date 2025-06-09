from ascfd.grid import Grid2D
from ascfd.constants import Constants
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

    def __init__(self, grid: Grid2D, types_lo: tuple[str, str], types_hi: tuple[str, str], a_constants: Constants) -> None:
        # store boundary condition types and associate grid object
        self.c = a_constants
        self.types_lo = types_lo
        self.types_hi = types_hi
        self.grid = grid

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
            elif self.types_lo[idim] == "double_inflow":
                self.f_lo[idim] = self.double_inflow_lo
            else:
                # error if unsupported boundary condition type is provided
                raise RuntimeError(
                    f"BC Lo Type not supported: {self.types_lo[idim]}")

            # dirichlet & neumann for high boundary conditions
            if self.types_hi[idim] == "dirichlet":
                self.f_hi[idim] = self.dirichlet_hi
            elif self.types_hi[idim] == "neumann":
                self.f_hi[idim] = self.neumann_hi
            elif self.types_hi[idim] == "periodic":
                self.f_hi[idim] = self.periodic_hi
            else:
                # error if unsupported boundary condition type is provided
                raise RuntimeError(
                    f"BC Hi Type not supported: {self.types_hi[idim]}")

    def apply_bcs(self) -> None:
        # apply boundary conditions for both low and high boundaries
        print("called BoundaryConditions.apply_bcs()!")
        self.apply_lo()
        self.apply_hi()

    def apply_lo(self) -> None:
        # apply low boundary conditions for all dimensions
        for idim in range(2):  # two dimensions
            self.f_lo[idim](self.grid, idim)

    def apply_hi(self) -> None:
        # apply high boundary conditions for all dimensions
        for idim in range(2):  # two dimensions
            self.f_hi[idim](self.grid, idim)

    def null_bcs(self, grid: Grid2D, dim: int) -> None:
        # function for unsupported or uninitialized boundary conditions
        raise RuntimeError("Null boundary conditions should not be called.")

    @staticmethod
    def dirichlet_lo(grid: Grid2D, dim: int) -> None:
        # dirichlet boundary condition at the low boundary
        for var in range(grid.num_vars):  # all variables in the grid
            if dim == 0:  # low boundary in x-direction
                for i in range(grid.Nghost):  # ghost cells
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):  # valid y-range
                        grid.grid[var, i, j] = grid.grid[var, grid.Nghost, j]
            else:  # low boundary in y-direction
                for i in range(grid.Nghost, grid.Nx + grid.Nghost):  # valid x-range
                    for j in range(grid.Nghost):  # ghost cells
                        grid.grid[var, i, j] = grid.grid[var, i, grid.Nghost]

    @staticmethod
    def dirichlet_hi(grid: Grid2D, dim: int) -> None:
        # dirichlet boundary condition at the high boundary
        for var in range(grid.num_vars):  # all variables in the grid
            if dim == 0:  # high boundary in x-direction
                for i in range(grid.Nx + grid.Nghost, grid.Nx + 2*grid.Nghost):  # ghost x-range
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):  # valid y-range
                        grid.grid[var, i, j] = grid.grid[var,
                                                         grid.Nx + grid.Nghost - 1, j]
            else:  # high boundary in y-direction
                for i in range(grid.Nghost, grid.Nx + grid.Nghost):  # valid x-range
                    for j in range(grid.Ny + grid.Nghost, grid.Ny + 2*grid.Nghost):  # ghost x-range
                        grid.grid[var, i, j] = grid.grid[var,
                                                         i, grid.Ny + grid.Nghost - 1]

    def inflow_lo(self, grid: Grid2D, dim: int):
        for var in range(grid.num_vars):
            if dim == 0:  # inflow at low x boundary
                for i in range(grid.Nghost):  # ghost cells on the left
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):
                        # Set inflow values here. Example:
                        if var == 0:
                            grid.grid[var, i, j] = 1.0
                        elif var == 1:  # mu
                            grid.grid[var, i, j] = 5
                        elif var == 2:  # mv
                            grid.grid[var, i, j] = 0.0
                        elif var == 3:  # pressure
                            grid.grid[var, i, j] = 3.0
                        else:  # other variables (Bx, By, etc.)
                            grid.grid[var, i, j] = 1.0  # copy from edge
            
            # TODO: Y IMPLEMENTATION BELOW IS BROKEN DON'T USE:
            else:
                for i in range(grid.Nghost):  # ghost cells on the left
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):
                        # Set inflow values here. Example:
                        if var == 0:  # density
                            grid.grid[var, i, j] = 1.0
                        elif var == 1:  # u (vx)
                            grid.grid[var, i, j] = 0.0
                        elif var == 2:  # v (vy)
                            grid.grid[var, i, j] = 0.0

    def double_inflow_lo(self, grid: Grid2D, dim: int):
        for var in range(grid.num_vars):
            if dim == 0:  # inflow at low x boundary
                for i in range(grid.Nghost):  # ghost cells on the left
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):
                        # Set inflow values here. Example:
                        upper_bound = grid.Ny * (4/5)
                        lower_bound = grid.Ny * (1/5)

                        pole_top = grid.Ny * (3/5)
                        pole_bottom = grid.Ny * (2/5)

                        # set inflow to only come out of the 2 hall thruster channels (top and bottom of the donut)
                        if (pole_top < j < upper_bound) or (lower_bound < j < pole_bottom):
                            if var == 0:
                                grid.grid[var, i, j] = 10
                            elif var == 1:  # mu
                                grid.grid[var, i, j] = 10
                            elif var == 2:  # mv
                                grid.grid[var, i, j] = 0.0
                            elif var == 3:  # energy
                                grid.grid[var, i, j] = 525
                        else:
                            if var == 0:
                                grid.grid[var, i, j] = 1.0
                            elif var == 1:  # mu
                                grid.grid[var, i, j] = 0.0
                            elif var == 2:  # mv
                                grid.grid[var, i, j] = 0.0
                            elif var == 3:  # energy
                                grid.grid[var, i, j] = 2.5

    @staticmethod
    def neumann_lo(grid: Grid2D, dim: int) -> None:
        print("called BoundaryConditions.neumann_lo()!")
        # neumann boundary condition at the low boundary
        for var in range(grid.num_vars):  # all variables in the grid
            if dim == 0:  # low boundary in x-direction
                for i in range(grid.Nghost):  # ghost cells
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):  # valid y-range
                        grid.grid[var, i, j] = grid.grid[var, grid.Nghost, j]
            else:  # low boundary in y-direction
                for i in range(grid.Nghost, grid.Nx + grid.Nghost):  # valid x-range
                    for j in range(grid.Nghost):  # ghost cells
                        grid.grid[var, i, j] = grid.grid[var, i, grid.Nghost]

    @staticmethod
    def neumann_hi(grid: Grid2D, dim: int) -> None:
        print("called BoundaryConditions.neumann_hi()!")
        # neumann boundary condition at the high boundary
        for var in range(grid.num_vars):  # all variables in the grid
            if dim == 0:  # high boundary in x-direction
                for i in range(grid.Nx + grid.Nghost, grid.Nx + 2*grid.Nghost):  # ghost cells
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):  # valid y-range
                        grid.grid[var, i, j] = grid.grid[var,
                                                         grid.Nx + grid.Nghost - 1, j]
            else:  # y-direction
                for i in range(grid.Nghost, grid.Nx + grid.Nghost):
                    for j in range(grid.Ny + grid.Nghost, grid.Ny + 2*grid.Nghost):  # valid x-range
                        grid.grid[var, i, j] = grid.grid[var, i,
                                                         grid.Ny + grid.Nghost - 1]  # ghost cells

    @staticmethod
    def periodic_lo(grid: Grid2D, dim: int) -> None:
        print("called BoundaryConditions.periodic_lo()!")
        # periodic boundary condition at the low boundary
        for var in range(grid.num_vars):  # all variables in the grid
            if dim == 0:  # low boundary in x-direction
                for i in range(grid.Nghost):  # ghost cells in low x-range
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):  # valid y-range
                        grid.grid[var, i, j] = grid.grid[var, grid.Nx + i, j]
            else:  # low boundary in y-direction
                for i in range(grid.Nghost, grid.Nx + grid.Nghost):  # valid x-range
                    for j in range(grid.Nghost):  # ghost cells in low y-range
                        grid.grid[var, i, j] = grid.grid[var, i, grid.Ny + j]

    @staticmethod
    def periodic_hi(grid: Grid2D, dim: int) -> None:
        print("called BoundaryConditions.periodic_hi()!")
        # periodic boundary condition at the high boundary
        for var in range(grid.num_vars):  # all variables in the grid
            if dim == 0:  # high boundary in x-direction
                # ghost cells in high x-range
                for i in range(grid.Nx + grid.Nghost, grid.Nx + 2*grid.Nghost):
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):  # valid y-range
                        grid.grid[var, i, j] = grid.grid[var, i - grid.Nx, j]
            else:  # H=high boundary in y-direction
                for i in range(grid.Nghost, grid.Nx + grid.Nghost):  # valid x-range
                    # ghost cells in high y-range
                    for j in range(grid.Ny + grid.Nghost, grid.Ny + 2*grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, i, j - grid.Ny]
