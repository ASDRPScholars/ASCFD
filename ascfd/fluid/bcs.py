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
    
    def __init__(self, grid, types_lo: tuple[str, str], types_hi: tuple[str, str], a_inputs: Inputs, embedded_boundaries: str) -> None:
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
            
        # EMBEDDED BOUNDARIES
        
        if embedded_boundaries == "hallthruster":
            self.embedded_boundaries = [
                {
                    'type': 'neumann',
                    'i_range': (0, 50),
                    'j_range': (0, 40)
                },
                {
                    'type': 'neumann',
                    'i_range': (0, 50),
                    'j_range': (60, 100)
                }
            ]
            self.embedded_masks = {}
            
            self._populate_embedded_masks()
            
        else:
            print("no embedded boundary!")
        

    def _populate_embedded_masks(self):
        for id, embedded_boundary in enumerate(self.embedded_boundaries):
            mask = self._create_embedded_mask(embedded_boundary)
            self.embedded_masks[id] = mask
            
        
    def _create_embedded_mask(self, embedded_boundary):
        """Draw a rectangular mask based on (start, end) where start and end are inclusive."""
        
        mask = np.zeros((self.inp.nx, self.inp.ny), dtype=bool)
        
        i_start, i_end = embedded_boundary["i_range"]
        j_start, j_end = embedded_boundary["j_range"]
        
        mask[i_start:i_end+1, j_start:j_end+1] = True
        
        return mask
    
    
    def _apply_embedded_neumann(self, eb_id: int) -> None:
        gradient_value = 0.0
        
        first_inside = self.get_first_inside(eb_id)
        boundary_mask = self.embedded_masks[eb_id]
        
        for var in range(self.c.NUMQ):
            for i, j in first_inside:
                # Find adjacent fluid cells for extrapolation
                fluid_neighbors = []
                
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ni, nj = i + di, j + dj
                    
                    # Check array bounds first, then physical domain bounds
                    if (0 <= ni < boundary_mask.shape[0] and 
                        0 <= nj < boundary_mask.shape[1] and
                        self.inp.ng <= ni < self.inp.nx + self.inp.ng and 
                        self.inp.ng <= nj < self.inp.ny + self.inp.ng and
                        not boundary_mask[ni, nj]):
                        fluid_neighbors.append((ni, nj))
                
                if fluid_neighbors:
                    # Extrapolate from fluid cells with specified gradient
                    fluid_values = [self.grid[var, fi, fj] for fi, fj in fluid_neighbors]
                    avg_fluid = np.mean(fluid_values)
                    
                    # Apply Neumann condition (usually zero gradient)
                    self.grid[var, i, j] = avg_fluid + gradient_value


    def _apply_embedded_dirichlet(self, id):
        mask = self.embedded_masks[id]
        
        for var in range(self.c.NUMQ):
            boundary_indices = np.where(mask)
        
            for idx in zip(*boundary_indices):
                i, j = idx
                self.grid[var, i, j] = 0
        
        
    def get_first_inside(self, eb_id: int) -> list[tuple[int, int]]:
        """
        Get the outermost edge (first layer) of cells INSIDE the embedded boundary.
        These are the boundary cells that are adjacent to fluid cells.
        
        Args:
            eb_id: Embedded boundary ID
            
        Returns:
            List of (i, j) tuples representing the first layer of boundary cells
            (excluding ghost cells)
        """
        if eb_id not in self.embedded_masks:
            raise ValueError(f"Embedded boundary {eb_id} not found")
        
        boundary_mask = self.embedded_masks[eb_id]
        first_inside = []
        
        for i in range(self.inp.nx):
            for j in range(self.inp.ny):
                # Skip if this cell is NOT part of the boundary
                if not boundary_mask[i, j]:
                    continue
                    
                # Check if this boundary cell is adjacent to fluid
                is_edge = False
                for di, dj in [(-1, 0), (1, 0), (0, -1), (0, 1)]:  # 4-connectivity
                    ni, nj = i + di, j + dj
                    
                    # Check bounds to avoid index errors
                    if (0 <= ni < boundary_mask.shape[0] and 
                        0 <= nj < boundary_mask.shape[1]):
                        
                        if not boundary_mask[ni, nj]:  # neighbor is fluid
                            is_edge = True
                            break
                
                if is_edge:
                    first_inside.append((i, j))
        
        return first_inside
            
        
    def apply_ebs(self):
        for eb_id, embedded_boundary in enumerate(self.embedded_boundaries):
            if embedded_boundary['type'] == 'neumann':
                self._apply_embedded_neumann(eb_id)
            elif embedded_boundary['type'] == 'dirichlet':
                self._apply_embedded_dirichlet(eb_id)
                
        
    def apply_bcs(self) -> None:
         # apply boundary conditions for both low and high boundaries
        self.apply_lo()
        self.apply_hi()
        # self.apply_ebs()


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

    
    def neumann_hi(self, grid, dim: int) -> None:
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
