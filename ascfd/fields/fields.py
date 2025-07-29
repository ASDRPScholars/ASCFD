import matplotlib.pyplot as plt
from ascfd.inputs import Inputs
from ascfd.fields.ics import FieldInitialConditions
from ascfd.fluid.plasma_refs import PlasmaReferences
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
        
        # Initialize plasma normalization
        self.plasma_refs = PlasmaReferences(
            n0=1e18, T0=1000.0, 
            species_mass=9.1e-31, species_charge=1.6e-19
        )
        
        # Normalized permittivity: ε₀ = 1 in plasma units
        self.eps0_normalized = 1.0
                
    def update_E(self):
        self.solve_poisson()
        self._compute_electric_field()
        self.limit_electric_field()  # Prevent runaway field growth
        
                
    def add_charge_density(self, species_charge_density):
        self.charge_density[:] = species_charge_density[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
    
    
    def solve_poisson(self):
        
        rhs = - self.charge_density / self.eps0_normalized        
        rect = ((self.inp.xlim[0], self.inp.ylim[0]), (self.inp.xlim[1], self.inp.ylim[1]))

        ## GRID IS ROTATED TO LINE UP WITH FLUID GRID so these are a bit jank:
        boundary = {
            "left": (0, "neumann_x"), # BECOMES BOTTOM
            "right": (0, "neumann_x"), # BECOMES TOP
            "top": (3, "dirichlet"), # BECOMES LEFT
            "bottom": (0, "neumann_y") # BECOMES RIGHT
        }
        
        solver = solvers.Poisson2DRectangle(rect=rect, interior=rhs, boundary=boundary, X=self.inp.nx, Y=self.inp.ny)
        
        self.potential[:] = solver.solve()

            
    def _compute_electric_field(self):
        """Compute E = -grad(phi) using central differences with ghost cells"""
        
        phi = self.potential  # shape (200, 200)
        
        dx, dy = self.dx, self.dy
        
        dphi_dy, dphi_dx = np.gradient(phi, dy, dx)  # Mind the order: (rows, cols) → (y, x)

        # self.E[:, :, 1] = -dphi_dx  # Ey
        self.E[:, :, 0] = -dphi_dy  # Ex
    
    
    def limit_electric_field(self, E_max_normalized=10.0):
        """Limit electric field to prevent runaway growth"""
        E_magnitude = np.sqrt(self.E[:,:,0]**2 + self.E[:,:,1]**2)
        
        # Find locations where |E| > E_max
        large_field_mask = E_magnitude > E_max_normalized
        
        if np.any(large_field_mask):
            # Normalize large fields to E_max while preserving direction
            normalization_factor = E_max_normalized / E_magnitude
            normalization_factor = np.where(large_field_mask, normalization_factor, 1.0)
            
            self.E[:,:,0] *= normalization_factor
            self.E[:,:,1] *= normalization_factor
            
            print(f"Warning: Limited {np.sum(large_field_mask)} cells with |E| > {E_max_normalized}")
    
    def check_E_field(self):
        pass
    