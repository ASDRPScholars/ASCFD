import numpy as np
import sympy as sp
import scipy.sparse.linalg
import types
import matplotlib.pyplot as plt
import os

from poissonpy import functional, helpers
import matplotlib as mpl

# inner region conditions 
# laplacian diag matrix construction

class Poisson2DRectangle:
    """ 
        Solve 2D Poisson Equation on a rectangle
    """
    
    # TODO: ADJUST FOR LIKE OUR ACTUAL GRID IDK
    def __init__(self, rect, interior, boundary, X, Y, zero_mean=False):
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
        # Interior points (excluding boundary rows/cols)
        self.interior_ids = np.arange(self.X + 1, 2 * self.X - 1) + self.X * np.expand_dims(np.arange(self.Y - 2), 1)
        self.interior_ids = self.interior_ids.flatten()

        # Boundary edges (excluding corners)
        boundary_ids = {
            "left": self.X * np.arange(1, self.Y - 1),
            "right": self.X * np.arange(1, self.Y - 1) + (self.X - 1),
            "top": np.arange(1, self.X - 1),
            "bottom": np.arange(self.X * self.Y - self.X + 1, self.X * self.Y - 1)
        }

        # Explicitly include corner points
        corner_ids = {
            "top_left": 0,
            "top_right": self.X - 1,
            "bottom_left": self.X * (self.Y - 1),
            "bottom_right": self.X * self.Y - 1
        }

        # Combine all IDs
        all_boundary_ids = list(boundary_ids.values()) + [np.array(list(corner_ids.values()))]
        self.all_ids = np.concatenate([self.interior_ids] + all_boundary_ids)
        self.all_ids = np.unique(self.all_ids)

        # Setup matrix
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
        corner_pos = {
            bd: np.searchsorted(self.all_ids, [corner_ids[bd]])[0] for bd in corner_ids
        }

        # Build Laplacian for interior
        n1_pos = np.searchsorted(self.all_ids, self.interior_ids - 1)
        n2_pos = np.searchsorted(self.all_ids, self.interior_ids + 1)
        n3_pos = np.searchsorted(self.all_ids, self.interior_ids - self.X)
        n4_pos = np.searchsorted(self.all_ids, self.interior_ids + self.X)

        A[self.interior_pos, n1_pos] = 1 / (self.dx**2)
        A[self.interior_pos, n2_pos] = 1 / (self.dx**2)
        A[self.interior_pos, n3_pos] = 1 / (self.dy**2)
        A[self.interior_pos, n4_pos] = 1 / (self.dy**2)
        A[self.interior_pos, self.interior_pos] = -2 / (self.dx**2) - 2 / (self.dy**2)

        # Fill in source term
        if isinstance(self.interior, types.FunctionType):
            b[self.interior_pos] = self.interior(self.xs[self.interior_ids], self.ys[self.interior_ids])
        elif isinstance(self.interior, (int, float)):
            b[self.interior_pos] = self.interior
        elif isinstance(self.interior, np.ndarray):
            b[self.interior_pos] = self.interior.flatten()[self.interior_ids]
        else:
            raise ValueError("[POISSONPY] Unsupported interior type")

        # Apply zero-mean constraint if needed
        if self.zero_mean:
            A[-1, :] = 1.0 
            A[:, -1] = 1.0
            A[-1, -1] = 0.0
            b[-1] = 0

        # Boundary conditions
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
                else:  # right
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
                else:  # bottom
                    n_ids = bd_ids - self.X
                    n_pos = np.searchsorted(self.all_ids, n_ids)
                    A[bd_pos, bd_pos] = 1 / self.dy
                    A[bd_pos, n_pos] = -1 / self.dy


        corner_neumann = {
            "top_right": self.X - 1,
            "bottom_right": self.X * self.Y - 1,
            "top_left": 0,
            "bottom_left": self.X * (self.Y - 1)
        }

        for name, cid in corner_neumann.items():
            pos = np.searchsorted(self.all_ids, cid)
            
            if 'right' in name:
                neighbor_id = cid - 1  # one step left
            else:  # left corners
                neighbor_id = cid + 1  # one step right
            
            neighbor_pos = np.searchsorted(self.all_ids, neighbor_id)

            A[pos, pos] = 1 / self.dx
            A[pos, neighbor_pos] = -1 / self.dx
            b[pos] = 0  # zero gradient
    
        return A.tocsr(), b


    def solve(self):
        # multigrid solver result bad?
        # x = scipy.sparse.linalg.bicg(self.A, self.b)[0]
        
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
    def __init__(self, region, interior, boundary, rect=None):
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
        
        self.dx = self.x[1] - self.x[0]
        self.dy = self.y[1] - self.y[0]

        if isinstance(interior, types.FunctionType):
           self.interior = interior(self.x_grid, self.y_grid)
        elif isinstance(interior, (int, float)):
            self.interior = np.ones_like(region) * interior
        else:
            self.interior = interior

        if isinstance(boundary, types.FunctionType):
            self.boundary = boundary(self.x_grid, self.y_grid)
        elif isinstance(boundary, (int, float)):
            self.boundary = np.ones_like(region) * boundary
        else:
            self.boundary = boundary

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

        n1_pos = np.searchsorted(self.region_ids, self.inner_ids - 1)
        n2_pos = np.searchsorted(self.region_ids, self.inner_ids + 1)
        n3_pos = np.searchsorted(self.region_ids, self.inner_ids - self.X)
        n4_pos = np.searchsorted(self.region_ids, self.inner_ids + self.X)

        A = scipy.sparse.lil_matrix((len(self.region_ids), len(self.region_ids)))
        A[self.inner_pos, n1_pos] = 1 / (self.dx**2)
        A[self.inner_pos, n2_pos] = 1 / (self.dx**2)
        A[self.inner_pos, n3_pos] = 1 / (self.dy**2)
        A[self.inner_pos, n4_pos] = 1 / (self.dy**2)
        A[self.inner_pos, self.inner_pos] = -2 / (self.dx**2) + -2 / (self.dy**2)

        A[self.boundary_pos, self.boundary_pos] = 1 # only dirichlet for now
        A = A.tocsr()

        boundary_conditions = helpers.get_selected_values(self.boundary, self.boundary_region).flatten()
        interior_laplacians = helpers.get_selected_values(self.interior, self.inner_region).flatten()
        b = np.zeros(len(self.region_ids))
        b[self.inner_pos] = interior_laplacians
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


def test_poisson_solver():
    """
    Test function that plots the result of the Poisson solver with specified
    boundary conditions on a 200x200 mesh.

    Boundary conditions:
    - left: Neumann (gradient = 0)
    - right: Neumann (gradient = 0)
    - top: Dirichlet (value = 300)
    - bottom: Dirichlet (value = 0)
    """
    nx, ny = 200, 200

    rect = [(0, 0), (1, 1)]  # unit square

    def top_boundary(x, y):
        return np.where(x <= 0.5, 300 * (1 - 2*x), 0)

    def bottom_boundary(x, y):
        return np.where(x <= 0.5, 300 * (1 - 2*x), 0)

    # Define boundary conditions
    boundary = {
        "left": (300, "dirichlet"),                    # Left edge (zero gradient)
        "right": (0, "dirichlet"),                   # Right edge (zero gradient)
        "top": (top_boundary, "dirichlet"),         # Top edge (linear drop profile)
        "bottom": (bottom_boundary, "dirichlet")      # Bottom edge (linear drop profile)
    }

    rhs = 0

    solver = Poisson2DRectangle(rect=rect, interior=rhs, boundary=boundary, X=nx, Y=ny)
    solution = solver.solve()

    Ex = -np.gradient(solution, solver.dx, axis=1)  # axis=1 is x-direction

    # Plot results
    fig, axes = plt.subplots(1, 4, figsize=(24, 5))

    # Contour line plot
    contour = axes[0].contour(solver.x_grid, solver.y_grid, solution, levels=20, colors='black', linewidths=0.8)
    axes[0].clabel(contour, inline=True, fontsize=8)
    axes[0].set_xlabel('x')
    axes[0].set_ylabel('y')
    axes[0].set_title('Potential (Contour Lines)')
    axes[0].set_aspect('equal')

    # Potential surface plot
    im1 = axes[1].imshow(solution, extent=[rect[0][0], rect[1][0], rect[0][1], rect[1][1]],
                        origin='lower', cmap=mpl.cm.Blues, aspect='equal')
    axes[1].set_xlabel('x')
    axes[1].set_ylabel('y')
    axes[1].set_title('Potential')
    plt.colorbar(im1, ax=axes[1], label='V')

    #  Ex plot
    im2 = axes[2].imshow(Ex, extent=[rect[0][0], rect[1][0], rect[0][1], rect[1][1]],
                        origin='lower', cmap=mpl.cm.Reds, aspect='equal')
    axes[2].set_xlabel('x')
    axes[2].set_ylabel('y')
    axes[2].set_title('Electric Field Ex = -dV/dx')
    plt.colorbar(im2, ax=axes[2], label='Ex (V/m)')

    # 1D plot along y-centerline (x-axis)
    y_center_idx = ny // 2
    axes[3].plot(solver.x, solution[y_center_idx, :], 'b-', linewidth=2)
    axes[3].set_xlabel('x')
    axes[3].set_ylabel('Potential (V)')
    axes[3].set_title(f'1D Potential Profile at y = {solver.y[y_center_idx]:.2f}')
    axes[3].grid(True, alpha=0.3)

    # create output directory if it doesn't exist
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output', 'poisson')
    os.makedirs(output_dir, exist_ok=True)

    output_path = os.path.join(output_dir, 'poisson_test_result.pdf')
    plt.tight_layout()
    plt.savefig(output_path, bbox_inches='tight')
    print(f"Plot saved to '{output_path}'")
    plt.show()

    return solution, solver

if __name__ == '__main__':
    test_poisson_solver()
