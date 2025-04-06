import numpy as np
import matplotlib.pyplot as plt
from ascfd.constants import Constants




class Grid2D:
    def __init__(self, xlim, ylim, Nx, Ny, Nghost, num_vars):
        # domain limits and grid resolution
        self.xlim = xlim
        self.ylim = ylim
        self.Nx = Nx
        self.Ny = Ny
        self.Nghost = Nghost # number of ghost cells
        self.num_vars = num_vars   # number of variables in the grid (e.g., density, velocity, pressure)
        
          # cell size
        self.dx = (xlim[1] - xlim[0]) / (Nx - 1)
        self.dy = (ylim[1] - ylim[0]) / (Ny - 1)
        
        # x and y coordinates, including ghost cells
        self.x = np.linspace(
            xlim[0] - self.dx * Nghost, xlim[1] + self.dx * Nghost, Nx + 2 * Nghost
        )
        self.y = np.linspace(
            ylim[0] - self.dy * Nghost, ylim[1] + self.dy * Nghost, Ny + 2 * Nghost
        )

        # 2D mesh grids
        self.meshX, self.meshY = np.meshgrid(self.x, self.y)

        self.ndim = 2 #dimensions
        
        # Internal x (no ghost cells)
        self.x_int = self.x[self.Nghost : self.Nx + self.Nghost]
        self.y_int = self.y[self.Nghost : self.Ny + self.Nghost]

        self.meshIntX, self.meshIntY = np.meshgrid(self.x_int, self.y_int)

         # grid with zeros
        self.grid = np.zeros((num_vars, Nx + 2 * Nghost, Ny + 2 * Nghost))
        self.variables = "prim"

    def fill_grid(self, f):
        # Fill the grid using a user-supplied function `f`
        for var in range(self.num_vars):
            # we can jsut fill ghost cells too, i don't think it matters we're just going to override anyways
            self.grid[var] = f(self.meshX, self.meshY, var)

    def plot(self):
        # Create a figure with 4 subplots arranged in 2x2
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()  # Flatten the 2x2 grid into a 1D array of axes

        # Titles for each subplot corresponding to each variable
        titles = ["Density", "UVel", "VVel", "Pressure"]

        for i in range(self.num_vars):
            # create color plots for each variable
            pcm = axs[i].pcolormesh(
                self.meshX, self.meshY, self.grid[i].T, cmap="inferno", shading="auto"
            )
            fig.colorbar(
                pcm, ax=axs[i], orientation="vertical" # color bar
            )
            axs[i].set_title(titles[i]) # plot title
            axs[i].set_xlabel("X") # label X-axis
            axs[i].set_ylabel("Y") # label Y-axis

        plt.tight_layout()  # layout to prevent overlap
        plt.show()  # display the plot

    def apply_periodic_bcs(self):
        # periodic boundary conditions
        for var in range(self.num_vars):
            for i in range(self.Nghost):
                # data from the opposite side for x boundaries
                self.grid[var, i, :] = self.grid[
                    var, self.Nx + self.Nghost - (self.Nghost - i), :
                ]
                self.grid[var, self.Nx + self.Nghost + i, :] = self.grid[
                    var, self.Nghost + i, :
                ]

            for j in range(self.Nghost):
                # data from the opposite side for y boundaries
                self.grid[var, :, j] = self.grid[
                    var, :, self.Ny + self.Nghost - (self.Nghost - j)
                ]
                self.grid[var, :, self.Ny + self.Nghost + j] = self.grid[
                    var, :, self.Nghost + j
                ]

    def apply_zero_gradient_bcs(self):
        # zero-gradient boundary conditions
        for var in range(self.num_vars):
            for ighost in range(self.Nghost):
                self.grid[var, ighost, :] = self
                # ghost cells equal to the nearest internal value 
                self.grid[var, self.Nx + self.Nghost + ighost, :] = self.grid[
                    var, self.Nx + self.Nghost - 1, :
                ]

            for jghost in range(self.Nghost):
                # ghost cells equal to the nearest internal value 
                self.grid[var, :, jghost] = self.grid[var, :, self.Nghost]
                self.grid[var, :, self.Ny + self.Nghost + jghost] = self.grid[
                    var, :, self.Ny + self.Nghost - 1
                ]

    def return_internal_grid(self):
        pass

    def return_grid(self):
        pass

    def scratch_internal(self):
        pass

    def scratch(self):
        pass

    def transform(self, f, name):
        pass

    def set_internal(self, U_new):
        pass

    def set(self, U_new):
        pass

    def set_grid(self, U_new):
        self.grid = U_new

    def internal_bounds(self):
        pass

    def assert_variable_type(self, str):
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
