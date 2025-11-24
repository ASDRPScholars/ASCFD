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
        self.E = self.ics.apply_E_ics()
        
        self.ref = PlasmaReferences()

        self.charge_density = np.zeros((self.inp.nx, self.inp.ny))
        self.potential = np.zeros((self.inp.nx, self.inp.ny))
        self.potential_star = np.zeros((self.inp.nx, self.inp.ny))

        self.dx = self.inp.dx
        self.dy = self.inp.dy
        
        self.eps0 = 8.854e-12  # TODO: CHECK — Permittivity of free space
        # self.eps0 = 1e-50
        
        # Normalized permittivity: ε₀ = 1 in plasma units
        self.eps0_normalized = 1.0

        # RF discharge parameters
        self.rf_enabled = (self.inp.rf_frequency is not None and
                          self.inp.rf_amplitude is not None and
                          self.inp.rf_boundary is not None)
        if self.rf_enabled:
            self.rf_omega = 2 * np.pi * self.inp.rf_frequency  # angular frequency

    def update_E(self, t=0.0):
        self.solve_poisson(t)
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
        # Handle 1D case where ny=1
        if self.inp.ny == 1:
            # For 1D: slice in x-direction, take the middle slice in y (avoid ghost cells in both dims)
            self.charge_density[:] += species_charge_density[self.inp.ng:-self.inp.ng, self.inp.ng:self.inp.ng+self.inp.ny]
        else:
            # For 2D: slice in both directions
            self.charge_density[:] += species_charge_density[self.inp.ng:-self.inp.ng, self.inp.ng:-self.inp.ng]
    
    def clear_charge_density(self):
        self.charge_density[:] = 0
    
    def solve_poisson(self, t=0.0):

        rhs = - self.charge_density / self.eps0_normalized
        rect = ((self.inp.xlim[0], self.inp.ylim[0]), (self.inp.xlim[1], self.inp.ylim[1]))

        ## GRID IS ROTATED TO LINE UP WITH FLUID GRID so these are a bit jank:
        # Calculate boundary values (potentially time-varying for RF discharge)
        V_anode = self.inp.V_anode
        V_cathode = self.inp.V_cathode

        # Apply RF oscillation to specified boundary
        if self.rf_enabled:
            # φ(t) = φ_DC + φ_RF * sin(ω*t)
            rf_voltage = self.inp.rf_amplitude * np.sin(self.rf_omega * t)

            if self.inp.rf_boundary.lower() == "anode" or self.inp.rf_boundary.lower() == "top":
                V_anode = self.inp.V_anode + rf_voltage
            elif self.inp.rf_boundary.lower() == "cathode" or self.inp.rf_boundary.lower() == "bottom":
                V_cathode = self.inp.V_cathode + rf_voltage
            elif self.inp.rf_boundary.lower() == "right":
                # For right boundary (maps to "right" in rotated coords)
                V_anode = self.inp.V_anode + rf_voltage
            elif self.inp.rf_boundary.lower() == "left":
                # For left boundary (maps to "left" in rotated coords)
                V_cathode = self.inp.V_cathode + rf_voltage

        boundary = {
            "left": (V_anode, "dirichlet"), # BECOMES BOTTOM
            "right": (V_cathode, "dirichlet"), # BECOMES TOP
            "top": (0, "neumann_y"), # BECOMES LEFT
            "bottom": (0, "neumann_y") # BECOMES RIGHT
        }

        solver = solvers.Poisson2DRectangle(rect=rect, interior=rhs, boundary=boundary, X=self.inp.nx, Y=self.inp.ny)

        self.potential[:] = solver.solve().T # TODO this may screw things up - if it does change X back to ny and Y back to nx

            
    def _compute_electric_field(self):
        """Compute E = -grad(phi) using central differences with ghost cells"""

        phi = self.potential

        dx, dy = self.dx, self.dy

        if self.inp.ny == 1:
            # 1D case: only compute Ex (derivative in x-direction)
            dphi_dx = np.gradient(phi, dx, axis=0)  # derivative along x-axis
            self.E[:, :, 0] = -dphi_dx  # Ex
            self.E[:, :, 1] = 0  # Ey = 0 in 1D
        else:
            # 2D case: compute both Ex and Ey
            dphi_dy, dphi_dx = np.gradient(phi, dy, dx)  # Mind the order: (rows, cols) → (y, x)
            self.E[:, :, 1] = -dphi_dx  # Ey
            self.E[:, :, 0] = -dphi_dy  # Ex


    def populate_E(self, E_field):
        self.E[:, :, 0] = E_field
        
    def populate_V(self, V_field, V_star=None):
        self.potential[:, :] = V_field
        
        if V_star is not None:
            self.potential_star[:, :] = V_star
    
    def check_E_field(self):
        pass
    