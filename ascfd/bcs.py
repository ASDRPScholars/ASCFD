from ascfd.grid import Grid2D
from typing import Callable
import numpy as np

class BoundaryConditions:
    
    def __init__(self, grid: Grid2D, types_lo: tuple[str, str], types_hi: tuple[str, str]) -> None:
        self.types_lo = types_lo
        self.types_hi = types_hi
        self.grid = grid

        self.f_lo = [self.null_bcs, self.null_bcs]
        self.f_hi = [self.null_bcs, self.null_bcs]

        for idim in range(2):  # two dimensions
            if self.types_lo[idim] == "dirichlet":
                self.f_lo[idim] = self.dirichlet_lo
            elif self.types_lo[idim] == "neumann":
                self.f_lo[idim] = self.neumann_lo
            elif self.types_lo[idim] == "periodic":
                self.f_lo[idim] = self.periodic_lo
            else:
                raise RuntimeError(f"BC Lo Type not supported: {self.types_lo[idim]}")

            if self.types_hi[idim] == "dirichlet":
                self.f_hi[idim] = self.dirichlet_hi
            elif self.types_hi[idim] == "neumann":
                self.f_hi[idim] = self.neumann_hi
            elif self.types_hi[idim] == "periodic":
                self.f_hi[idim] = self.periodic_hi
            else:
                raise RuntimeError(f"BC Hi Type not supported: {self.types_hi[idim]}")

    def apply_bcs(self) -> None:
        self.apply_lo()
        self.apply_hi()

    def apply_lo(self) -> None:
        for idim in range(2):
            self.f_lo[idim](self.grid, idim)
        
    def apply_hi(self) -> None:
        for idim in range(2):
            self.f_hi[idim](self.grid, idim)

    def null_bcs(self, grid: Grid2D, dim: int) -> None:
        raise RuntimeError("Null boundary conditions should not be called.")

    @staticmethod
    def dirichlet_lo(grid: Grid2D, dim: int) -> None:
        for var in range(grid.num_vars):
            if dim == 0:  # x-direction
                for i in range(grid.Nghost):
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, grid.Nghost, j]
            else:  # y-direction
                for i in range(grid.Nghost, grid.Nx + grid.Nghost):
                    for j in range(grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, i, grid.Nghost]

    @staticmethod
    def dirichlet_hi(grid: Grid2D, dim: int) -> None:
        for var in range(grid.num_vars):
            if dim == 0:  # x-direction
                for i in range(grid.Nx + grid.Nghost, grid.Nx + 2*grid.Nghost):
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, grid.Nx + grid.Nghost - 1, j]
            else:  # y-direction
                for i in range(grid.Nghost, grid.Nx + grid.Nghost):
                    for j in range(grid.Ny + grid.Nghost, grid.Ny + 2*grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, i, grid.Ny + grid.Nghost - 1]

    @staticmethod
    def neumann_lo(grid: Grid2D, dim: int) -> None:
        for var in range(grid.num_vars):
            if dim == 0:  # x-direction
                for i in range(grid.Nghost):
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, grid.Nghost, j]
            else:  # y-direction
                for i in range(grid.Nghost, grid.Nx + grid.Nghost):
                    for j in range(grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, i, grid.Nghost]

    @staticmethod
    def neumann_hi(grid: Grid2D, dim: int) -> None:
        for var in range(grid.num_vars):
            if dim == 0:  # x-direction
                for i in range(grid.Nx + grid.Nghost, grid.Nx + 2*grid.Nghost):
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, grid.Nx + grid.Nghost - 1, j]
            else:  # y-direction
                for i in range(grid.Nghost, grid.Nx + grid.Nghost):
                    for j in range(grid.Ny + grid.Nghost, grid.Ny + 2*grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, i, grid.Ny + grid.Nghost - 1]

    @staticmethod
    def periodic_lo(grid: Grid2D, dim: int) -> None:
        for var in range(grid.num_vars):
            if dim == 0:  # x-direction
                for i in range(grid.Nghost):
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, grid.Nx + i, j]
            else:  # y-direction
                for i in range(grid.Nghost, grid.Nx + grid.Nghost):
                    for j in range(grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, i, grid.Ny + j]

    @staticmethod
    def periodic_hi(grid: Grid2D, dim: int) -> None:
        for var in range(grid.num_vars):
            if dim == 0:  # x-direction
                for i in range(grid.Nx + grid.Nghost, grid.Nx + 2*grid.Nghost):
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, i - grid.Nx, j]
            else:  # y-direction
                for i in range(grid.Nghost, grid.Nx + grid.Nghost):
                    for j in range(grid.Ny + grid.Nghost, grid.Ny + 2*grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, i, j - grid.Ny]
