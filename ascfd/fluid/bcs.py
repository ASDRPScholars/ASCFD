from ascfd.inputs import Inputs
from ascfd.fluid.constants import FluidConstants
from ascfd.plasma_refs import PlasmaReferences
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
        
        self.ref = PlasmaReferences()

        # boundary condition function mappings
        self.f_lo = [self.null_bcs, self.null_bcs]
        self.f_hi = [self.null_bcs, self.null_bcs]

        # dirichlet & neumann for low boundary conditions
        for idim in range(2):  # two dimensions
            if self.types_lo[idim] == "dirichlet":
                self.f_lo[idim] = self.dirichlet_lo
            elif self.types_lo[idim] == "neumann":
                self.f_lo[idim] = self.neumann_lo
            elif self.types_lo[idim] == "neumann_der_constant":
                self.f_lo[idim] = self.neumann_der_constant_lo
            elif self.types_lo[idim] == "periodic":
                self.f_lo[idim] = self.periodic_lo
            elif self.types_lo[idim] == "tame_inflow":
                self.f_lo[idim] = self.tame_inflow_lo
            else:
                # error if unsupported boundary condition type is provided
                raise RuntimeError(f"BC Lo Type not supported: {self.types_lo[idim]}")

            # dirichlet & neumann for high boundary conditions
            if self.types_hi[idim] == "dirichlet":
                self.f_hi[idim] = self.dirichlet_hi
            elif self.types_hi[idim] == "neumann":
                self.f_hi[idim] = self.neumann_hi
            elif self.types_hi[idim] == "neumann_der_constant":
                self.f_hi[idim] = self.neumann_der_constant_hi
            elif self.types_hi[idim] == "periodic":
                self.f_hi[idim] = self.periodic_hi
            elif self.types_hi[idim] == "tame_inflow":
                self.f_hi[idim] = self.tame_inflow_hi
            else:
                # error if unsupported boundary condition type is provided
                raise RuntimeError(f"BC Hi Type not supported: {self.types_hi[idim]}")

    def apply_bcs(self) -> None:
         # apply boundary conditions for both low and high boundaries
        self.apply_lo()
        self.apply_hi()

    def apply_lo(self) -> None:
        # apply low boundary conditions for all dimensions
        for idim in range(1, -1, -1): # two dimensions
            self.f_lo[idim](self.grid, idim)
        
    def apply_hi(self) -> None:
        # apply high boundary conditions for all dimensions
        for idim in range(1, -1, -1): # two dimensions
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

        # neumann boundary condition at the low boundary
        for var in range(self.c.NUMQ): # all variables in the grid
            if dim == 0:  # low boundary in x-direction
                for i in range(self.inp.ng): # ghost cells
                    for j in range(self.inp.ng, self.inp.ny + self.inp.ng):  # valid y-range
                        grid[var, i, j] = grid[var, self.inp.ng, j]
                        
            else:  # low boundary in y-direction
                for i in range(self.inp.ng, self.inp.nx + self.inp.ng): # valid x-range
                    for j in range(self.inp.ng): # ghost cells
                        grid[var, i, j] = grid[var, i, self.inp.ng]
                        
        # Handle corners for low boundaries
        for var in range(self.c.NUMQ):
            # Bottom-left corner (lo-x, lo-y)
            for i in range(self.inp.ng):
                for j in range(self.inp.ng):
                    grid[var, i, j] = grid[var, self.inp.ng, self.inp.ng]

    
    def neumann_hi(self, grid, dim: int) -> None:
        # print("!BCS! CALLED NEUMANN HI")
         # neumann boundary condition at the high boundary
        for var in range(self.c.NUMQ): # all variables in the grid
            if dim == 0:  # high boundary in x-direction
                for i in range(self.inp.nx + self.inp.ng, self.inp.nx + 2*self.inp.ng): # ghost cells
                    for j in range(self.inp.ng, self.inp.ny + self.inp.ng): # valid y-range
                        grid[var, i, j] = grid[var, self.inp.nx + self.inp.ng - 1, j]
            else:  # y-direction
                for i in range(self.inp.ng, self.inp.nx + self.inp.ng):
                    for j in range(self.inp.ny + self.inp.ng, self.inp.ny + 2*self.inp.ng): # ghost cells
                        grid[var, i, j] = grid[var, i, self.inp.ny + self.inp.ng - 1]
                        
        # Handle corners for high boundaries
        for var in range(self.c.NUMQ):
            # Top-right corner (hi-x, hi-y)
            for i in range(self.inp.nx + self.inp.ng, self.inp.nx + 2*self.inp.ng):
                for j in range(self.inp.ny + self.inp.ng, self.inp.ny + 2*self.inp.ng):
                    grid[var, i, j] = grid[var, self.inp.nx + self.inp.ng - 1, self.inp.ny + self.inp.ng - 1]
            
            # Top-left corner (lo-x, hi-y)
            for i in range(self.inp.ng):
                for j in range(self.inp.ny + self.inp.ng, self.inp.ny + 2*self.inp.ng):
                    grid[var, i, j] = grid[var, self.inp.ng, self.inp.ny + self.inp.ng - 1]
            
            # Bottom-right corner (hi-x, lo-y)
            for i in range(self.inp.nx + self.inp.ng, self.inp.nx + 2*self.inp.ng):
                for j in range(self.inp.ng):
                    grid[var, i, j] = grid[var, self.inp.nx + self.inp.ng - 1, self.inp.ng]

    
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


    def neumann_der_constant_lo(self, grid, dim: int) -> None:
        # neumann boundary condition that maintains constant derivative at low boundary
        for var in range(self.c.NUMQ): # all variables in the grid
            if dim == 0:  # low boundary in x-direction
                for i in range(self.inp.ng): # ghost cells
                    for j in range(self.inp.ng, self.inp.ny + self.inp.ng):  # valid y-range
                        # extrapolate: ghost = 2*boundary - interior
                        boundary_idx = self.inp.ng
                        interior_idx = self.inp.ng + 1
                        grid[var, i, j] = 2 * grid[var, boundary_idx, j] - grid[var, interior_idx, j]
                        
            else:  # low boundary in y-direction
                for i in range(self.inp.ng, self.inp.nx + self.inp.ng): # valid x-range
                    for j in range(self.inp.ng): # ghost cells
                        # extrapolate: ghost = 2*boundary - interior
                        boundary_idx = self.inp.ng
                        interior_idx = self.inp.ng + 1
                        grid[var, i, j] = 2 * grid[var, i, boundary_idx] - grid[var, i, interior_idx]
                        print(f"NEUMANN SET {i}, {j} to", grid[self.c.UCOMP, i, j])
                        
        # Handle corners for low boundaries
        for var in range(self.c.NUMQ):
            # Bottom-left corner (lo-x, lo-y)
            for i in range(self.inp.ng):
                for j in range(self.inp.ng):
                    # Extrapolate from the domain corner
                    boundary_x = self.inp.ng
                    boundary_y = self.inp.ng
                    interior_x = self.inp.ng + 1
                    interior_y = self.inp.ng + 1
                    # Average of x and y extrapolations
                    extrap_x = 2 * grid[var, boundary_x, boundary_y] - grid[var, interior_x, boundary_y]
                    extrap_y = 2 * grid[var, boundary_x, boundary_y] - grid[var, boundary_x, interior_y]
                    grid[var, i, j] = 0.5 * (extrap_x + extrap_y)
                    print(f"NEUMANN SET {i}, {j} to", grid[self.c.UCOMP, i, j])

    
    def neumann_der_constant_hi(self, grid, dim: int) -> None:
        # neumann boundary condition that maintains constant derivative at high boundary
        for var in range(self.c.NUMQ): # all variables in the grid
            if dim == 0:  # high boundary in x-direction
                for i in range(self.inp.nx + self.inp.ng, self.inp.nx + 2*self.inp.ng): # ghost cells
                    for j in range(self.inp.ng, self.inp.ny + self.inp.ng): # valid y-range
                        # extrapolate: ghost = 2*boundary - interior
                        boundary_idx = self.inp.nx + self.inp.ng - 1
                        interior_idx = self.inp.nx + self.inp.ng - 2
                        grid[var, i, j] = 2 * grid[var, boundary_idx, j] - grid[var, interior_idx, j]
            else:  # high boundary in y-direction
                for i in range(self.inp.ng, self.inp.nx + self.inp.ng): # valid x-range
                    for j in range(self.inp.ny + self.inp.ng, self.inp.ny + 2*self.inp.ng): # ghost cells
                        # extrapolate: ghost = 2*boundary - interior
                        boundary_idx = self.inp.ny + self.inp.ng - 1
                        interior_idx = self.inp.ny + self.inp.ng - 2
                        grid[var, i, j] = 2 * grid[var, i, boundary_idx] - grid[var, i, interior_idx]
                        
        # Handle corners for high boundaries
        for var in range(self.c.NUMQ):
            # Top-right corner (hi-x, hi-y)
            for i in range(self.inp.nx + self.inp.ng, self.inp.nx + 2*self.inp.ng):
                for j in range(self.inp.ny + self.inp.ng, self.inp.ny + 2*self.inp.ng):
                    boundary_x = self.inp.nx + self.inp.ng - 1
                    boundary_y = self.inp.ny + self.inp.ng - 1
                    interior_x = self.inp.nx + self.inp.ng - 2
                    interior_y = self.inp.ny + self.inp.ng - 2
                    # Average of x and y extrapolations
                    extrap_x = 2 * grid[var, boundary_x, boundary_y] - grid[var, interior_x, boundary_y]
                    extrap_y = 2 * grid[var, boundary_x, boundary_y] - grid[var, boundary_x, interior_y]
                    grid[var, i, j] = 0.5 * (extrap_x + extrap_y)
            
            # Top-left corner (lo-x, hi-y)
            for i in range(self.inp.ng):
                for j in range(self.inp.ny + self.inp.ng, self.inp.ny + 2*self.inp.ng):
                    boundary_x = self.inp.ng
                    boundary_y = self.inp.ny + self.inp.ng - 1
                    interior_x = self.inp.ng + 1
                    interior_y = self.inp.ny + self.inp.ng - 2
                    # Average of x and y extrapolations
                    extrap_x = 2 * grid[var, boundary_x, boundary_y] - grid[var, interior_x, boundary_y]
                    extrap_y = 2 * grid[var, boundary_x, boundary_y] - grid[var, boundary_x, interior_y]
                    grid[var, i, j] = 0.5 * (extrap_x + extrap_y)
            
            # Bottom-right corner (hi-x, lo-y)
            for i in range(self.inp.nx + self.inp.ng, self.inp.nx + 2*self.inp.ng):
                for j in range(self.inp.ng):
                    boundary_x = self.inp.nx + self.inp.ng - 1
                    boundary_y = self.inp.ng
                    interior_x = self.inp.nx + self.inp.ng - 2
                    interior_y = self.inp.ng + 1
                    # Average of x and y extrapolations
                    extrap_x = 2 * grid[var, boundary_x, boundary_y] - grid[var, interior_x, boundary_y]
                    extrap_y = 2 * grid[var, boundary_x, boundary_y] - grid[var, boundary_x, interior_y]
                    grid[var, i, j] = 0.5 * (extrap_x + extrap_y)

                    print(f"NEUMANN SET {i}, {j} to", grid[self.c.UCOMP, i, j])

    
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
                        
                        
    def tame_inflow_lo(self, grid, dim: int) -> None:

        if dim == 0:  # low boundary in x-direction
            # Fill main boundary region
            for i in range(self.inp.ng): # ghost cells
                for j in range(self.inp.ny_with_ghosts):  # main boundary
                    grid[self.c.RHOCOMP, i, j] = 1
                    grid[self.c.PCOMP, i, j] = 1
                    grid[self.c.UCOMP, i, j] = 1
                    grid[self.c.VCOMP, i, j] = 0
            
            # Fill corners (bottom-left and top-left)
            for i in range(self.inp.ng):
                # Bottom-left corner
                for j in range(self.inp.ng):
                    grid[self.c.RHOCOMP, i, j] = 1
                    grid[self.c.PCOMP, i, j] = 1
                    grid[self.c.UCOMP, i, j] = 1  # x-direction dominates
                    grid[self.c.VCOMP, i, j] = 1  # y-direction dominates
                # Top-left corner  
                for j in range(self.inp.ny + self.inp.ng, self.inp.ny + 2*self.inp.ng):
                    grid[self.c.RHOCOMP, i, j] = 1
                    grid[self.c.PCOMP, i, j] = 1
                    grid[self.c.UCOMP, i, j] = 1  # x-direction dominates
                    grid[self.c.VCOMP, i, j] = -1  # y-direction dominates
                    
        else:  # low boundary in y-direction
            # Fill main boundary region only (corners handled by x-direction)
            for i in range(self.inp.nx_with_ghosts): # main boundary
                for j in range(self.inp.ng): # ghost cells
                    grid[self.c.RHOCOMP, i, j] = 1
                    grid[self.c.PCOMP, i, j] = 1
                    grid[self.c.UCOMP, i, j] = 0
                    grid[self.c.VCOMP, i, j] = 1


    def tame_inflow_hi(self, grid, dim: int) -> None:
        
        ref = self.ref

        if dim == 0:  # hi boundary in x-direction
            # center_y = 0.5  # vertical midpoint
            # sigma = 0.35   # controls sharpness (smaller = narrower)

            # for i in range(self.inp.nx + self.inp.ng, self.inp.nx + 2*self.inp.ng):  # ghost cells in high x-range
            #     for j in range(self.inp.ny_with_ghosts):
            #         y_normalized = j / (self.inp.ny_with_ghosts - 1)

            #         # Gaussian profile centered at 0.5
            #         taper = np.exp(-((y_normalized - center_y) ** 2) / (2 * sigma ** 2))

            #         # Scale rho and p to max 0.3, and u to max -1
            #         grid[self.c.RHOCOMP, i, j] = 1.5 * taper
            #         grid[self.c.PCOMP, i, j]   = 1.5 * taper
            #         grid[self.c.UCOMP, i, j]   = -1.0 * taper
            #         grid[self.c.VCOMP, i, j]   = 0

            for i in range(self.inp.nx + self.inp.ng, self.inp.nx + 2*self.inp.ng):  # ghost cells in high x-range
                for j in range(self.inp.ny_with_ghosts):
                    grid[self.c.RHOCOMP, i, j] = self.inp.rho_e * (ref.L**3 / ref.m)
                    grid[self.c.PCOMP, i, j]   = self.inp.p_e * ((ref.dt ** 2 * ref.L) / ref.m )
                    grid[self.c.UCOMP, i, j]   = -1
                    grid[self.c.VCOMP, i, j]   = 0
                    
        else:  # hi boundary in y-direction
            # Fill main boundary region only (corners handled by x-direction)
            for i in range(self.inp.nx_with_ghosts): # main boundary
                for j in range(self.inp.ny + self.inp.ng, self.inp.ny + 2*self.inp.ng): # ghost cells in high y-range
                    grid[self.c.RHOCOMP, i, j] = 1
                    grid[self.c.PCOMP, i, j] = 1
                    grid[self.c.UCOMP, i, j] = 0
                    grid[self.c.VCOMP, i, j] = -1
                    