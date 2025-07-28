import matplotlib.pyplot as plt
from ascfd.inputs import Inputs
from ascfd.fields.ics import FieldInitialConditions
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection only
from sympy import sin, cos
from sympy.abc import x, y
import sympy as sp

import sys

from poissonpy import solvers, functional
import poissonpy


class Fields:
    def __init__(self, a_inputs: Inputs):
        self.inp = a_inputs
        
        self.E = np.zeros((self.inp.nx, self.inp.ny, 3))
        self.B = np.zeros((self.inp.nx, self.inp.ny, 3))
        
        self.ics = FieldInitialConditions(self.E, self.B, self.inp)
        
        self.B = self.ics.apply_B_ics()
        self.ics.apply_E_ics()
        
        plt.figure()
        im = plt.imshow(self.B[:, :, 1])
        plt.title("apply_B_ics() magnetic field")
        plt.colorbar(im)
        plt.show()
        
        self.charge_density = np.zeros((self.inp.nx, self.inp.ny))
        self.potential = np.zeros((self.inp.nx, self.inp.ny))

        self.dx = (self.inp.xlim[1] - self.inp.xlim[0]) / (self.inp.nx - 1)
        self.dy = (self.inp.ylim[1] - self.inp.ylim[0]) / (self.inp.ny - 1)
        
        self.eps0 = 8.854e-12  # TODO: CHECK — Permittivity of free space
        # self.eps0 = 1e-50
                
                
    def update_E(self):
        # print("UPDATE E")
        
        # print("THIS IS CHARGE DENSITY TAKEN IN BY POISSON:", self.charge_density)
        
        self.solve_poisson()
        self._compute_electric_field()
    
        # print("NEW CALCULATED EX IS:", self.E[:, :, 0])
        # print("NEW CALCULATED EY IS:", self.E[:, :, 1])
        
                
        print("E", self.E)
        # print("V", V)
        print("B", self.B)

        fig, axs = plt.subplots(2, 2, figsize=(15, 15))  # 1 row, 3 columns
        axs = axs.flatten()

        # Plot charge density
        im0 = axs[0].imshow(self.charge_density, cmap="coolwarm")
        axs[0].set_title("Charge Density")
        fig.colorbar(im0, ax=axs[0])

        # Plot Ex
        im1 = axs[1].imshow(self.E[:, :, 0], cmap="coolwarm")
        axs[1].set_title("Electric Field Ex")
        fig.colorbar(im1, ax=axs[1])

        # Plot Ey
        im2 = axs[2].imshow(self.E[:, :, 1], cmap="coolwarm")
        axs[2].set_title("Electric Field Ey")
        fig.colorbar(im2, ax=axs[2])
        
        # Plot E
        im3 = axs[3].imshow(np.sqrt(self.E[:, :, 0]**2 + self.E[:, :, 1]**2), cmap="coolwarm")
        axs[3].set_title("Electric Field Magnitude")
        fig.colorbar(im3, ax=axs[3])
        
        plt.tight_layout()
        plt.show()
        
        ## 3D PLOT
        # Compute E magnitude
        E_mag = np.sqrt(self.E[5:-5, 5:-5, 0]**2 + self.E[5:-5, 5:-5, 1]**2)

        # Create meshgrid for X and Y
        nx, ny = E_mag.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(y, x)  # careful: meshgrid uses (cols, rows) order

        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_surface(X, Y, E_mag, cmap='coolwarm')

        # Add colorbar and labels
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        ax.set_title("ours")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("|E|")

        plt.tight_layout()
        plt.show()
        
                
    def add_charge_density(self, species_charge_density):
        self.charge_density[:] = species_charge_density[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
    
    
    def solve_poisson(self):
        
        rhs = - self.charge_density / self.eps0
        
        # rhs = np.zeros_like(self.charge_density)
        
        rect = ((self.inp.xlim[0], self.inp.ylim[0]), (self.inp.xlim[1], self.inp.ylim[1]))

        boundary = {
            "left": (30, "dirichlet"), # high voltage
            "right": (0, "neumann_x"), # to low voltage
            "top": (0, "neumann_y"),
            "bottom": (0, "neumann_y")
        }
        
        # # x, y = sp.symbols('x y')
        # f_expr = sp.exp(-x)
        
        # # f_expr = sin(x) + cos(y) # create sympy function expression
        # laplacian_expr = functional.get_sp_laplacian_expr(f_expr) # create sympy laplacian function expression

        # f = functional.get_sp_function(f_expr) # create sympy function
        # laplacian = functional.get_sp_function(laplacian_expr) # create sympy function
        
        # interior = laplacian
        # boundary = {
        #     "left": (f, "dirichlet"),
        #     "right": (f, "dirichlet"),
        #     "top": (f, "dirichlet"),
        #     "bottom": (f, "dirichlet")
        # }
        
        # boundary = {
        #     "left": (0, "dirichlet"),
        #     "right": (0, "neumann_x"),
        #     "top": (0, "neumann_y"),
        #     "bottom": (0, "neumann_y")
        # }
        
        np.set_printoptions(threshold=sys.maxsize)
        print("rhs", rhs)
        
        # solve potential from poisson !!WITH GHOSTS!! this makes computing E = -grad(phi) easier
        solver = solvers.Poisson2DRectangle(rect=rect, interior=rhs, boundary=boundary, X=self.inp.nx, Y=self.inp.ny)
        
        self.potential[:] = solver.solve()
        
        # print("POTENTIAL FROM SOLVED POISSON IS:", self.potential)
        # print(np.shape(self.potential))
        
        # f_expr = sin(x) + cos(y) # create sympy function expression
        # laplacian_expr = functional.get_sp_laplacian_expr(f_expr) # create sympy laplacian function expression

        # f = functional.get_sp_function(f_expr) # create sympy function
        # laplacian = functional.get_sp_function(laplacian_expr) # create sympy function
        
        # interior = laplacian
        # boundary = {
        #     "left": (f, "dirichlet"),
        #     "right": (f, "dirichlet"),
        #     "top": (f, "dirichlet"),
        #     "bottom": (f, "dirichlet")
        # }
        
        # solver = solvers.Poisson2DRectangle(((-2*np.pi, -2*np.pi), (2*np.pi, 2*np.pi)), 
        #     interior, boundary, X=100, Y=100)
        # solution = solver.solve()
        
        # self.potential = solution

            
    def _compute_electric_field(self):
        """Compute E = -grad(phi) using central differences with ghost cells"""
        # ng = self.inp.ng
        # nx, ny = self.inp.nx_with_ghosts, self.inp.ny_with_ghosts
    
        
        # # Central differences only for interior points to avoid boundary artifacts
        # Ex = np.zeros((nx, ny))
        # Ey = np.zeros((nx, ny))
        
        # # Use central differences only for interior points in both directions
        # Ex = -(self.potential[3:-1, 3:-1] - self.potential[1:-3, 1:-3]) / (2 * self.dx)
        # Ey = -(self.potential[3:-1, 3:-1] - self.potential[1:-3, 1:-3]) / (2 * self.dy)
        
        # print(np.shape(Ex))
        
        # grad = -np.gradient(self.potential)
        # print("gradiemt", grad, np.shape(grad))
        
        # Ey = -np.gradient(self.potential)
        
        # # Extract interior domain for storage
        # self.E[:, :, 0] = Ex
        # self.E[:, :, 1] = Ey
        
        phi = self.potential  # shape (200, 200)
        
        plt.figure()
        im = plt.imshow(phi)
        plt.colorbar(im)
        plt.title("potential")
        plt.show()
        
        dx, dy = self.dx, self.dy
        
        dphi_dy, dphi_dx = np.gradient(phi, dy, dx)  # Mind the order: (rows, cols) → (y, x)

        self.E[:, :, 0] = -dphi_dx  # Ex
        self.E[:, :, 1] = -dphi_dy  # Ey
        
        plt.quiver(self.E[::5, ::5, 0], self.E[::5, ::5, 1])
    
    
    def check_E_field(self):
        pass
    