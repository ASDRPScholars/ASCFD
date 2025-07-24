from ascfd.inputs import Inputs
from ascfd.fields.ics import FieldInitialConditions
import numpy as np
# from sympy import sin, cos
# from sympy.abc import x, y

from poissonpy import solvers


class Fields:
    def __init__(self, a_inputs: Inputs):
        self.inp = a_inputs
        self.ics = FieldInitialConditions(self.inp)
        
        self.ics.apply_B_ics()
        self.ics.apply_E_ics()
        
        self.E = np.zeros((self.inp.nx, self.inp.ny, 3))
        self.B = np.zeros((self.inp.nx, self.inp.ny, 3))
        
        self.charge_density = np.zeros((self.inp.nx, self.inp.ny))
        self.potential = np.zeros((self.inp.nx, self.inp.ny))

        self.dx = (self.inp.xlim[1] - self.inp.xlim[0]) / (self.inp.nx - 1)
        self.dy = (self.inp.ylim[1] - self.inp.ylim[0]) / (self.inp.ny - 1)
        
        self.eps0 = 8.854e-12  # TODO: CHECK — Permittivity of free space
                
                
    def update_E(self):
        self.solve_poisson()
        self._compute_electric_field()
        self._clear_calculation_grids()
                
                
    def add_charge_density(self, species_charge_density):
        ng = self.inp.numghosts
        
        if ng != 0:
            self.charge_density += species_charge_density[ng:-ng, ng:-ng]
        
        
    def _clear_calculation_grids(self):
        self.charge_density.fill(0)
        self.potential.fill(0)
    
    
    def solve_poisson(self):
        rhs = - self.charge_density / self.eps0
        
        mask = np.ones_like(rhs)
        rect = ((0, self.inp.xlim), (0, self.inp.ylim))
        
        boundary = {
            "left": (lambda x, y: 0.0, "neumann_x"),
            "right": (lambda x, y: 0.0, "neumann_x"),
            "top": (lambda x, y: 0.0, "neumann_y"),
            "bottom": (lambda x, y: 0.0, "neumann_y")
        }
        
        solver = solvers.Poisson2DRegion(mask, rhs, boundary, rect)
        
        self.potential = solver.solve()
        
            
    def _compute_electric_field(self):
        """Compute E = -grad(phi) using central differences with ghost cells"""
        ng = self.inp.numghosts
        nx, ny = self.inp.nx_with_ghosts, self.inp.ny_with_ghosts
        
        # Create potential array with ghost cells
        phi_ext = np.zeros((nx, ny))
        phi_ext[ng:-ng, ng:-ng] = self.potential  # Interior domain
        
        # Apply Neumann BCs to ghost cells: ∂φ/∂n = 0
        # This means ghost cells mirror interior values across boundary
        
        # Left boundary ghost cells
        for i in range(ng):
            phi_ext[i, ng:-ng] = phi_ext[2*ng-1-i, ng:-ng]
        
        # Right boundary ghost cells  
        for i in range(ng):
            phi_ext[-1-i, ng:-ng] = phi_ext[-2*ng+i, ng:-ng]
        
        # Bottom boundary ghost cells
        for j in range(ng):
            phi_ext[ng:-ng, j] = phi_ext[ng:-ng, 2*ng-1-j] 
        
        # Top boundary ghost cells
        for j in range(ng):
            phi_ext[ng:-ng, -1-j] = phi_ext[ng:-ng, -2*ng+j]
        
        # Handle corner ghost cells (simple averaging)
        for i in range(ng):
            for j in range(ng):
                # Bottom-left corner
                phi_ext[i, j] = 0.5 * (phi_ext[i, ng] + phi_ext[ng, j])
                # Bottom-right corner  
                phi_ext[i, -1-j] = 0.5 * (phi_ext[i, -1-ng] + phi_ext[ng, -1-j])
                # Top-left corner
                phi_ext[-1-i, j] = 0.5 * (phi_ext[-1-i, ng] + phi_ext[-1-ng, j])
                # Top-right corner
                phi_ext[-1-i, -1-j] = 0.5 * (phi_ext[-1-i, -1-ng] + phi_ext[-1-ng, -1-j])
        
        # Central differences everywhere (interior + boundaries)
        Ex = np.zeros((nx, ny))
        Ey = np.zeros((nx, ny))
        
        # Can now use central differences for all interior points including boundaries
        Ex[1:-1, :] = -(phi_ext[2:, :] - phi_ext[:-2, :]) / (2 * self.dx)
        Ey[:, 1:-1] = -(phi_ext[:, 2:] - phi_ext[:, :-2]) / (2 * self.dy)
        
        # Extract interior domain for storage
        self.E[:, :, 0] = Ex[ng:-ng, ng:-ng]
        self.E[:, :, 1] = Ey[ng:-ng, ng:-ng]
    
    
    def check_E_field(self):
        pass
    