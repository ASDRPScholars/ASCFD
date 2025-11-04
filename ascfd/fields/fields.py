import matplotlib.pyplot as plt
from ascfd.inputs import Inputs
from ascfd.fields.ics import FieldInitialConditions
from ascfd.plasma_refs import PlasmaReferences
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection only
from sympy import sin, cos
from sympy.abc import x, y
import sympy as sp
import copy

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
        
        self.ref = PlasmaReferences()
        
        plt.figure()
        im = plt.imshow(self.B[:, :, 1])
        plt.title("apply_B_ics() magnetic field")
        plt.colorbar(im)
        plt.show()
        
        self.charge_density = np.zeros((self.inp.nx, self.inp.ny))
        self.potential = np.zeros((self.inp.nx, self.inp.ny))

        self.dx = self.inp.dx
        self.dy = self.inp.dy
        
        self.eps0 = 8.854e-12  # TODO: CHECK — Permittivity of free space
        # self.eps0 = 1e-50
        
        # Normalized permittivity: ε₀ = 1 in plasma units
        self.eps0_normalized = 1.0
                
    def update_E(self):
        self.solve_poisson()
        self._compute_electric_field()

        
    def normal_to_phys(self):
        E = copy.copy(self.E)
        B = copy.copy(self.B)
        potential = copy.copy(self.potential)
        ref = self.ref
        
        E = self.E * (ref.m * ref.v**2 / (ref.q * ref.L))
        B = self.B * (ref.m * ref.v / (ref.q * ref.L))
        potential = self.potential * (ref.m * ref.v**2 / ref.q)
        
        return E, B, potential
        
                
    def add_charge_density(self, species_charge_density):
        self.charge_density[:] += species_charge_density[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
    
    def clear_charge_density(self):
        self.charge_density[:] = 0
    
    def solve_poisson(self):
        
        rhs = - self.charge_density / self.eps0_normalized        
        rect = ((self.inp.xlim[0], self.inp.ylim[0]), (self.inp.xlim[1], self.inp.ylim[1]))

        ## GRID IS ROTATED TO LINE UP WITH FLUID GRID so these are a bit jank:
        boundary = {
            "left": (0, "neumann_x"), # BECOMES BOTTOM
            "right": (0, "neumann_x"), # BECOMES TOP
            "top": (self.inp.V_anode, "dirichlet"), # BECOMES LEFT
            "bottom": (self.inp.V_cathode, "dirichlet") # BECOMES RIGHT
        }
        
        solver = solvers.Poisson2DRectangle(rect=rect, interior=rhs, boundary=boundary, X=self.inp.ny, Y=self.inp.nx)
        
        self.potential[:] = solver.solve()

            
    def _compute_electric_field(self):
        """Compute E = -grad(phi) using central differences with ghost cells"""
        
        phi = self.potential  # shape (200, 200)
        
        dx, dy = self.dx, self.dy
        
        dphi_dy, dphi_dx = np.gradient(phi, dy, dx)  # Mind the order: (rows, cols) → (y, x)

        #TODO: GET RID OF MULTIPLIER
        self.E[:, :, 1] = -dphi_dx#/1000  # Ey
        self.E[:, :, 0] = -dphi_dy#/1000  # Ex

    
    def check_E_field(self):
        pass
    