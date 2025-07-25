import numpy as np
import sympy as sp
import scipy.sparse
import types
import matplotlib.pyplot as plt
import seaborn as sns

from . import functional, helpers

# inner region conditions 
# laplacian diag matrix construction

class Poisson2DRectangle:
    """ 
        Solve 2D Poisson Equation on a rectangle
    """
    def __init__(self, rect, interior, boundary, X=100, Y=100, zero_mean=False):
        """
        Args:
            rect:
            interior: sympy expresson or numpy array
            boundary:    
            N: Rectangle to N * N grid #
        """
        self.x1, self.y1 = rect[0] # top-left corner
        self.x2, self.y2 = rect[1] # bottom-right corner
        self.X, self.Y = X, Y

        self.x = np.linspace(self.x1, self.x2, self.X)
        self.y = np.linspace(self.y1, self.y2, self.Y)
        
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        self._x_grid, self._y_grid = np.meshgrid(self.x, self.y)
        self.xs = self.x_grid.flatten()
        self.ys = self.y_grid.flatten()

        self.interior = interior
        self.boundary = boundary
        for bd, (_, mode) in self.boundary.items():
            assert bd in ["left", "right", "top", "bottom"] 
            assert mode in ["dirichlet", "neumann_x", "neumann_y"]
        
        self.zero_mean = zero_mean
        
        self.A, self.b = self.build_linear_system()

    @property
    def x_grid(self):
        return self._x_grid

    @property
    def y_grid(self):
        return self._y_grid

    def build_linear_system(self):

        self.interior_ids = np.arange(self.X + 1, 2 * self.X - 1) + self.X * np.expand_dims(np.arange(self.Y - 2), 1)
        self.interior_ids = self.interior_ids.flatten()
        boundary_ids = {
            "left": self.X * np.arange(1, self.Y - 1),
            "right": self.X * np.arange(1, self.Y - 1) + (self.X - 1),
            "top": np.arange(1, self.X - 1),
            "bottom": np.arange(self.X * self.Y - self.X + 1, self.X * self.Y - 1)
        }

        self.all_ids = np.concatenate([self.interior_ids, np.concatenate(list(boundary_ids.values()))]) 
        self.all_ids.sort()

        if self.zero_mean:
            A = scipy.sparse.lil_matrix((len(self.all_ids) + 1, len(self.all_ids) + 1))
            b = np.zeros(len(self.all_ids) + 1)
        else:
            A = scipy.sparse.lil_matrix((len(self.all_ids), len(self.all_ids)))
            b = np.zeros(len(self.all_ids))

        self.interior_pos = np.searchsorted(self.all_ids, self.interior_ids)
        boundary_pos = {
            bd: np.searchsorted(self.all_ids, boundary_ids[bd]) for bd in boundary_ids
        }

        # interior - laplacian
        n1_pos = np.searchsorted(self.all_ids, self.interior_ids - 1)
        n2_pos = np.searchsorted(self.all_ids, self.interior_ids + 1)
        n3_pos = np.searchsorted(self.all_ids, self.interior_ids - self.X)
        n4_pos = np.searchsorted(self.all_ids, self.interior_ids + self.X)
        
        # Discrete laplacian here important!
        A[self.interior_pos, n1_pos] = 1 / (self.dx**2)
        A[self.interior_pos, n2_pos] = 1 / (self.dx**2)
        A[self.interior_pos, n3_pos] = 1 / (self.dy**2)
        A[self.interior_pos, n4_pos] = 1 / (self.dy**2)
        A[self.interior_pos, self.interior_pos] = -2 / (self.dx**2) + -2 / (self.dy**2)

        if isinstance(self.interior, types.FunctionType):
            b[self.interior_pos] = self.interior(self.xs[self.interior_ids], self.ys[self.interior_ids])
        elif isinstance(self.interior, (int, float)):
            b[self.interior_pos] = self.interior
        
        if self.zero_mean:
            A[-1, :] = 1.0 
            A[:, -1] = 1.0
            A[-1, -1] = 0.0
            b[-1] = 0

        for bd, (bd_func, mode) in self.boundary.items():
            bd_pos = boundary_pos[bd]
            bd_ids = boundary_ids[bd]

            if isinstance(bd_func, types.FunctionType):
                b[bd_pos] = bd_func(self.xs[bd_ids], self.ys[bd_ids])
            elif isinstance(bd_func, (int, float)):
                b[bd_pos] = bd_func

            if mode == "dirichlet":
                A[bd_pos, bd_pos] = 1
            elif mode == "neumann_x":
                if bd == "left":
                    n_ids = bd_ids + 1
                    n_pos = np.searchsorted(self.all_ids, n_ids)
                    A[bd_pos, bd_pos] = -1 / self.dx
                    A[bd_pos, n_pos] = 1 / self.dx
                else: # right
                    n_ids = bd_ids - 1
                    n_pos = np.searchsorted(self.all_ids, n_ids)
                    A[bd_pos, bd_pos] = 1 / self.dx
                    A[bd_pos, n_pos] = -1 / self.dx
            elif mode == "neumann_y":
                if bd == "top":
                    n_ids = bd_ids + self.X
                    n_pos = np.searchsorted(self.all_ids, n_ids)
                    A[bd_pos, bd_pos] = -1 / self.dy
                    A[bd_pos, n_pos] = 1 / self.dy
                else: 
                    n_ids = bd_ids - self.X
                    n_pos = np.searchsorted(self.all_ids, n_ids)
                    A[bd_pos, bd_pos] = 1 / self.dy
                    A[bd_pos, n_pos] = -1 / self.dy
        return A.tocsr(), b

    def solve(self):
        # multigrid solver result bad?
        #x = scipy.sparse.linalg.bicg(A, b)[0]
        x = scipy.sparse.linalg.spsolve(self.A, self.b)
        if self.zero_mean:
            x = x[:-1]

        grid_ids = np.arange(self.Y * self.X)
        all_pos = np.searchsorted(grid_ids, self.all_ids)

        solution_grid = np.zeros(self.Y * self.X)
        solution_grid[all_pos] = x
        solution_grid = solution_grid.reshape(self.Y, self.X)
        return solution_grid
    
    def estimated_laplacian(self, f):
        val = f(self.x_grid, self.y_grid).flatten()
        est_lap = (self.A @ val[self.all_ids])[self.interior_pos]
        grid_ids = np.arange(self.Y * self.X)
        interior_pos = np.searchsorted(grid_ids, self.interior_ids)
        solution_grid = np.zeros(self.Y * self.X)
        solution_grid[interior_pos] = est_lap
        solution_grid = solution_grid.reshape(self.Y, self.X)
        return solution_grid
        
class Poisson2DRegion:
    """
        Solve 2D Poisson Equation on a region with arbitrary shape.
    """
    def __init__(self, region, interior, boundary=None, boundary_conditions=None, boundary_segments=None, rect=None):
        self.region = region # region mask
        self.Y, self.X = self.region.shape

        if rect:
            x0, y0 = rect[0]
            x1, y1 = rect[1]
            self.x = np.linspace(x0, x1, self.X)
            self.y = np.linspace(y0, y1, self.Y)
        else:
            self.x = np.arange(self.X)
            self.y = np.arange(self.Y)
        self.x_grid, self.y_grid = np.meshgrid(self.x, self.y)
        self.xs = self.x_grid.flatten()
        self.ys = self.y_grid.flatten()
        
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        if isinstance(interior, types.FunctionType):
           self.interior = interior(self.x_grid, self.y_grid)
        elif isinstance(interior, (int, float)):
            self.interior = np.ones_like(region) * interior
        else:
            self.interior = interior

        # Handle boundary conditions - support both old and new format
        if boundary_conditions is not None:
            # New format: boundary_conditions dict + boundary_segments mask
            assert boundary_segments is not None, "boundary_segments required when using boundary_conditions"
            self.boundary_conditions = boundary_conditions
            self.boundary_segments = boundary_segments
            
            # Validate boundary condition modes
            for segment_id, (_, mode) in self.boundary_conditions.items():
                assert mode in ["dirichlet", "neumann_x", "neumann_y"], f"Invalid mode: {mode}"
        else:
            # Old format: single boundary value for all boundaries (Dirichlet only)
            assert boundary is not None, "Either boundary or boundary_conditions must be provided"
            if isinstance(boundary, types.FunctionType):
                self.boundary = boundary(self.x_grid, self.y_grid)
            elif isinstance(boundary, (int, float)):
                self.boundary = np.ones_like(region) * boundary
            else:
                self.boundary = boundary
            
            self.boundary_conditions = None
            self.boundary_segments = None

        self.A, self.b = self.build_linear_system()
    
    def build_linear_system(self):
        self.inner_region, self.boundary_region = helpers.process_mask(self.region)
        self.grid_ids = helpers.get_grid_ids(self.X, self.Y)
        

        self.inner_ids = helpers.get_selected_values(self.grid_ids, self.inner_region).flatten()
        self.boundary_ids = helpers.get_selected_values(self.grid_ids, self.boundary_region).flatten()
        self.region_ids = helpers.get_selected_values(self.grid_ids, self.region).flatten() # boundary + inner

        self.inner_pos = np.searchsorted(self.region_ids, self.inner_ids) 
        self.boundary_pos = np.searchsorted(self.region_ids, self.boundary_ids)
        self.region_pos = np.searchsorted(self.grid_ids.flatten(), self.region_ids)

        # Find neighbor positions with bounds checking
        n1_ids = self.inner_ids - 1  # left neighbors
        n2_ids = self.inner_ids + 1  # right neighbors  
        n3_ids = self.inner_ids - self.X  # bottom neighbors
        n4_ids = self.inner_ids + self.X  # top neighbors
        
        n1_pos = np.searchsorted(self.region_ids, n1_ids)
        n2_pos = np.searchsorted(self.region_ids, n2_ids)
        n3_pos = np.searchsorted(self.region_ids, n3_ids)
        n4_pos = np.searchsorted(self.region_ids, n4_ids)
        
        # Check bounds and validate that the neighbors actually exist in region_ids
        n1_valid = np.zeros(len(n1_pos), dtype=bool)
        n2_valid = np.zeros(len(n2_pos), dtype=bool)
        n3_valid = np.zeros(len(n3_pos), dtype=bool)
        n4_valid = np.zeros(len(n4_pos), dtype=bool)
        
        # Only check equality for in-bounds indices
        n1_in_bounds = n1_pos < len(self.region_ids)
        n2_in_bounds = n2_pos < len(self.region_ids)
        n3_in_bounds = n3_pos < len(self.region_ids)
        n4_in_bounds = n4_pos < len(self.region_ids)
        
        n1_valid[n1_in_bounds] = self.region_ids[n1_pos[n1_in_bounds]] == n1_ids[n1_in_bounds]
        n2_valid[n2_in_bounds] = self.region_ids[n2_pos[n2_in_bounds]] == n2_ids[n2_in_bounds]
        n3_valid[n3_in_bounds] = self.region_ids[n3_pos[n3_in_bounds]] == n3_ids[n3_in_bounds]
        n4_valid[n4_in_bounds] = self.region_ids[n4_pos[n4_in_bounds]] == n4_ids[n4_in_bounds]

        A = scipy.sparse.lil_matrix((len(self.region_ids), len(self.region_ids)))
        
        # Only set coefficients for valid neighbors
        A[self.inner_pos[n1_valid], n1_pos[n1_valid]] = 1 / (self.dx**2)
        A[self.inner_pos[n2_valid], n2_pos[n2_valid]] = 1 / (self.dx**2)
        A[self.inner_pos[n3_valid], n3_pos[n3_valid]] = 1 / (self.dy**2)
        A[self.inner_pos[n4_valid], n4_pos[n4_valid]] = 1 / (self.dy**2)
        A[self.inner_pos, self.inner_pos] = -2 / (self.dx**2) + -2 / (self.dy**2)

        # Handle boundary conditions
        if self.boundary_conditions is not None:
            # New format: mixed boundary conditions
            # Get boundary segment mask aligned with boundary region
            boundary_segments_on_boundary = helpers.get_selected_values(self.boundary_segments, self.boundary_region).flatten()
            
            for segment_id, (bd_func, mode) in self.boundary_conditions.items():
                # Find boundary points belonging to this segment
                segment_mask = boundary_segments_on_boundary == segment_id
                segment_boundary_pos = self.boundary_pos[segment_mask]
                segment_boundary_ids = self.boundary_ids[segment_mask]
                
                if len(segment_boundary_pos) == 0:
                    continue
                
                if mode == "dirichlet":
                    A[segment_boundary_pos, segment_boundary_pos] = 1
                elif mode == "neumann_x":
                    # For Neumann x conditions, we need to find interior neighbors
                    # Try left and right neighbors
                    left_ids = segment_boundary_ids - 1
                    right_ids = segment_boundary_ids + 1
                    
                    # Check which neighbors are in the region
                    left_pos = np.searchsorted(self.region_ids, left_ids)
                    right_pos = np.searchsorted(self.region_ids, right_ids)
                    
                    left_valid = (left_pos < len(self.region_ids)) & (self.region_ids[left_pos] == left_ids)
                    right_valid = (right_pos < len(self.region_ids)) & (self.region_ids[right_pos] == right_ids)
                    
                    # Apply finite difference for normal derivative
                    for i in range(len(segment_boundary_pos)):
                        if left_valid[i]:
                            A[segment_boundary_pos[i], segment_boundary_pos[i]] = -1 / self.dx
                            A[segment_boundary_pos[i], left_pos[i]] = 1 / self.dx
                        elif right_valid[i]:
                            A[segment_boundary_pos[i], segment_boundary_pos[i]] = 1 / self.dx
                            A[segment_boundary_pos[i], right_pos[i]] = -1 / self.dx
                        else:
                            # Fallback to Dirichlet if no valid neighbor
                            A[segment_boundary_pos[i], segment_boundary_pos[i]] = 1
                            
                elif mode == "neumann_y":
                    # For Neumann y conditions
                    top_ids = segment_boundary_ids - self.X
                    bottom_ids = segment_boundary_ids + self.X
                    
                    top_pos = np.searchsorted(self.region_ids, top_ids)
                    bottom_pos = np.searchsorted(self.region_ids, bottom_ids)
                    
                    top_valid = (top_pos < len(self.region_ids)) & (self.region_ids[top_pos] == top_ids)
                    bottom_valid = (bottom_pos < len(self.region_ids)) & (self.region_ids[bottom_pos] == bottom_ids)
                    
                    for i in range(len(segment_boundary_pos)):
                        if top_valid[i]:
                            A[segment_boundary_pos[i], segment_boundary_pos[i]] = -1 / self.dy
                            A[segment_boundary_pos[i], top_pos[i]] = 1 / self.dy
                        elif bottom_valid[i]:
                            A[segment_boundary_pos[i], segment_boundary_pos[i]] = 1 / self.dy
                            A[segment_boundary_pos[i], bottom_pos[i]] = -1 / self.dy
                        else:
                            # Fallback to Dirichlet if no valid neighbor
                            A[segment_boundary_pos[i], segment_boundary_pos[i]] = 1
        else:
            # Old format: single Dirichlet boundary condition
            A[self.boundary_pos, self.boundary_pos] = 1
            
        A = A.tocsr()

        # Build right-hand side vector
        interior_laplacians = helpers.get_selected_values(self.interior, self.inner_region).flatten()
        b = np.zeros(len(self.region_ids))
        b[self.inner_pos] = interior_laplacians
        
        if self.boundary_conditions is not None:
            # New format: apply boundary values per segment
            boundary_segments_on_boundary = helpers.get_selected_values(self.boundary_segments, self.boundary_region).flatten()
            
            for segment_id, (bd_func, mode) in self.boundary_conditions.items():
                segment_mask = boundary_segments_on_boundary == segment_id
                segment_boundary_pos = self.boundary_pos[segment_mask]
                segment_boundary_ids = self.boundary_ids[segment_mask]
                
                if len(segment_boundary_pos) == 0:
                    continue
                
                # Get boundary values
                if isinstance(bd_func, types.FunctionType):
                    boundary_values = bd_func(self.xs[segment_boundary_ids], self.ys[segment_boundary_ids])
                elif isinstance(bd_func, (int, float)):
                    boundary_values = bd_func
                else:
                    boundary_values = bd_func
                
                b[segment_boundary_pos] = boundary_values
        else:
            # Old format: single boundary value
            boundary_conditions = helpers.get_selected_values(self.boundary, self.boundary_region).flatten()
            b[self.boundary_pos] = boundary_conditions

        return A, b

    def solve(self):
        # multigrid solver result bad?
        #x = scipy.sparse.linalg.bicg(A, b)[0]
        x = scipy.sparse.linalg.spsolve(self.A, self.b)

        solution_grid = np.zeros(self.Y * self.X)
        solution_grid[self.region_pos] = x
        solution_grid = solution_grid.reshape(self.Y, self.X)
        return solution_grid
