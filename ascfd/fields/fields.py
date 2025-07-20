from ascfd.inputs import Inputs
import numpy as np
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from ascfd.fields.ics import FieldInitialConditions

# TODO: write all the logic lol
class Fields:
    def __init__(self, a_inputs: Inputs):
        self.inp = a_inputs
        self.ics = FieldInitialConditions(self.inp)
        
        self.ics.apply_B_ics()
        self.ics.apply_E_ics()
        
        self.E = np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts, 3))
        self.B = np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts, 3))
        
        self.charge_density = np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        self.potential = np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))

        self.dx = (self.inp.xlim[1] - self.inp.xlim[0]) / (self.inp.nx - 1)
        self.dy = (self.inp.ylim[1] - self.inp.ylim[0]) / (self.inp.ny - 1)
        
        self.eps0 = 8.854e-12  # Permittivity of free space
                
    
    def update_E(self):
        self.solve_poisson()
        self.compute_electric_field()
        self.clear_calculation_grids()
                
    def add_charge_density(self, species_charge_density):
        self.charge_density += species_charge_density
        
    def clear_calculation_grids(self):
        self.charge_density.fill(0)
        self.potential.fill(0)
    
    def solve_poisson(self):
        """Solve Laplacian(phi) = -rho/eps0 using finite differences"""
        nx, ny = self.inp.nx_with_ghosts, self.inp.ny_with_ghosts
        
        # Create coefficient matrix for 2D finite difference Laplacian
        # Using 5-point stencil: (phi[i+1,j] + phi[i-1,j] + phi[i,j+1] + phi[i,j-1] - 4*phi[i,j])
        N = nx * ny
        
        # Main diagonal: -4/(dx^2) - 4/(dy^2)
        main_diag = -2.0 * (1.0/self.dx**2 + 1.0/self.dy**2) * np.ones(N)
        
        # Off-diagonals for x-direction: 1/(dx^2)
        x_diag = (1.0/self.dx**2) * np.ones(N-1)
        # Remove connections across x-boundaries
        for i in range(nx-1, N-1, nx):
            x_diag[i] = 0
            
        # Off-diagonals for y-direction: 1/(dy^2)
        y_diag = (1.0/self.dy**2) * np.ones(N-nx)
        
        # Construct sparse matrix
        A = diags([y_diag, x_diag, main_diag, x_diag, y_diag], 
                  [-nx, -1, 0, 1, nx], shape=(N, N), format='csr')
        
        # Right-hand side: -rho/eps0
        rhs = (-self.charge_density / self.eps0).flatten()
        
        # Apply boundary conditions (Dirichlet: phi = 0 on boundaries)
        for i in range(nx):  # Bottom and top boundaries
            A[i, :] = 0
            A[i, i] = 1
            rhs[i] = 0
            
            A[N-1-i, :] = 0
            A[N-1-i, N-1-i] = 1
            rhs[N-1-i] = 0
            
        for j in range(ny):  # Left and right boundaries
            idx = j * nx
            A[idx, :] = 0
            A[idx, idx] = 1
            rhs[idx] = 0
            
            idx = j * nx + nx - 1
            A[idx, :] = 0
            A[idx, idx] = 1
            rhs[idx] = 0
        
        # Solve the system
        phi_flat = spsolve(A, rhs)
        self.potential = phi_flat.reshape((nx, ny))
        
    def compute_electric_field(self):
        """Compute E = -grad(phi) using central differences"""
        nx, ny = self.inp.nx_with_ghosts, self.inp.ny_with_ghosts
        
        # Initialize E field components
        Ex = np.zeros((nx, ny))
        Ey = np.zeros((nx, ny))
        
        # Central differences for interior points
        Ex[1:-1, :] = -(self.potential[2:, :] - self.potential[:-2, :]) / (2 * self.dx)
        Ey[:, 1:-1] = -(self.potential[:, 2:] - self.potential[:, :-2]) / (2 * self.dy)
        
        # Forward/backward differences for boundaries
        Ex[0, :] = -(self.potential[1, :] - self.potential[0, :]) / self.dx
        Ex[-1, :] = -(self.potential[-1, :] - self.potential[-2, :]) / self.dx
        Ey[:, 0] = -(self.potential[:, 1] - self.potential[:, 0]) / self.dy
        Ey[:, -1] = -(self.potential[:, -1] - self.potential[:, -2]) / self.dy
        
        # Store magnitude for now (you can modify to store components)
        self.E = np.sqrt(Ex**2 + Ey**2)
        self.Ex = Ex
        self.Ey = Ey
    
    
    def check_E_field(self):
        pass