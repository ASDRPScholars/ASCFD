from ascfd.grid import Grid2D
from typing import Callable
import numpy as np

class BoundaryConditions:
    
    def __init__(self, grid: Grid2D, types_lo: tuple[str, str], types_hi: tuple[str, str]) -> None:
        # Initialize the boundary conditions for the low and high sides of the grid
        self.types_lo = types_lo # Boundary condition types for the low side
        self.types_hi = types_hi # Boundary condition types for the high side
        self.grid = grid # Grid object containing the simulation data

         # Default boundary condition functions
        self.f_lo = [self.null_bcs, self.null_bcs] # Boundary functions for low side
        self.f_hi = [self.null_bcs, self.null_bcs] # Boundary functions for high side
        
        # Loop over the x and y
        for idim in range(2):  # two dimensions

            # Set boundary conditions for low
            if self.types_lo[idim] == "dirichlet":
                self.f_lo[idim] = self.dirichlet_lo # Dirichlet boundary on low side
            elif self.types_lo[idim] == "neumann": 
                self.f_lo[idim] = self.neumann_lo # Neumann boundary on low side
            elif self.types_lo[idim] == "periodic": # Periodic boundary on low side
                self.f_lo[idim] = self.periodic_lo
            else:
                raise RuntimeError(f"BC Lo Type not supported: {self.types_lo[idim]}")
            
            # Set boundary conditions for the high
            if self.types_hi[idim] == "dirichlet":
                self.f_hi[idim] = self.dirichlet_hi # Dirichlet boundary on high side
            elif self.types_hi[idim] == "neumann":
                self.f_hi[idim] = self.neumann_hi # Neumann boundary on high side
            elif self.types_hi[idim] == "periodic":
                self.f_hi[idim] = self.periodic_hi  # Periodic boundary on high side
            else:
                raise RuntimeError(f"BC Hi Type not supported: {self.types_hi[idim]}")

    def apply_bcs(self) -> None:
        # Apply the boundary conditions for both the low and high sides
        self.apply_lo()
        self.apply_hi()

    def apply_lo(self) -> None:
        # Apply the boundary conditions for the low side
        for idim in range(2):
            self.f_lo[idim](self.grid, idim)
        
    def apply_hi(self) -> None:
        # Apply the boundary conditions for the high side
        for idim in range(2):
            self.f_hi[idim](self.grid, idim)

    def null_bcs(self, grid: Grid2D, dim: int) -> None:
        # It raises an error for invalid boundary conditions
        raise RuntimeError("Null boundary conditions should not be called.")

    @staticmethod
    def dirichlet_lo(grid: Grid2D, dim: int) -> None:
        # Apply Dirichlet boundary condition on the low
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
        # Apply Dirichlet boundary condition on the high
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
        # Apply Neumann (derivative) boundary condition on the low
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
        # Apply Neumann (derivative) boundary condition on the high
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
          # Apply Periodic boundary condition on the low
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
        # Apply Periodic boundary condition on the high
        for var in range(grid.num_vars):
            if dim == 0:  # x-direction
                for i in range(grid.Nx + grid.Nghost, grid.Nx + 2*grid.Nghost):
                    for j in range(grid.Nghost, grid.Ny + grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, i - grid.Nx, j]
            else:  # y-direction
                for i in range(grid.Nghost, grid.Nx + grid.Nghost):
                    for j in range(grid.Ny + grid.Nghost, grid.Ny + 2*grid.Nghost):
                        grid.grid[var, i, j] = grid.grid[var, i, j - grid.Ny]
