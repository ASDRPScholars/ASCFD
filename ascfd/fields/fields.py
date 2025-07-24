from ascfd.inputs import Inputs
import numpy as np
from scipy.sparse import diags
from scipy.sparse import csc_array
from scipy.sparse.linalg import cg
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
        """Solve Laplacian(phi) = -rho/eps0 using finite differences with resymmetrization"""
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
        P = diags([y_diag, x_diag, main_diag, x_diag, y_diag], 
                  [-nx, -1, 0, 1, nx], shape=(N, N), format='csr')
        
        A = csc_array(P)
        
        # Right-hand side: -rho/eps0
        rhs = (-self.charge_density / self.eps0).flatten()
        
        # Identify boundary nodes for resymmetrization
        boundary_nodes = set()
        
        # Bottom and top boundaries
        for i in range(nx):
            boundary_nodes.add(i)  # Bottom
            boundary_nodes.add(N-1-i)  # Top
            
        # Left and right boundaries  
        for j in range(ny):
            boundary_nodes.add(j * nx)  # Left
            boundary_nodes.add(j * nx + nx - 1)  # Right
            
        boundary_nodes = list(boundary_nodes)
        
        # Apply boundary conditions and perform resymmetrization
        # First store the original matrix before boundary modifications
        A_orig = A.copy()
        
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
        
        # Resymmetrization correction
        A, rhs = self._apply_resymmetrization(A, A_orig, rhs, boundary_nodes)
        
        # Solve the system
        phi_flat, exit_code = cg(A, rhs, atol=1e-5)
        
        print(exit_code) # 0 means convergence was successful
        
        self.potential = phi_flat.reshape((nx, ny))
    
    def _apply_resymmetrization(self, A, A_orig, rhs, boundary_nodes, boundary_potential=0.0):
        """
        Apply resymmetrization correction to maintain matrix symmetry after boundary conditions.
        
        Based on the paper methodology:
        1. Identify asymmetric terms in columns corresponding to Dirichlet boundary nodes
        2. Remove these terms from A and add their contribution to the RHS via correction matrix B
        3. Final system: A' * x = b' where A' is resymmetrized and b' = b + B * boundary_values
        """
        import scipy.sparse as sp
        
        # Convert to lil_matrix for efficient modification
        A = A.tolil()
        A_orig = A_orig.tolil() 
        
        # Create correction matrix B (initially zero)
        N = A.shape[0]
        B = sp.lil_matrix((N, N))
        
        # Boundary values vector 
        boundary_values = np.zeros(N)
        for idx in boundary_nodes:
            boundary_values[idx] = boundary_potential
        
        # For each boundary node, find all non-zero entries in its column
        # and move them to the correction matrix
        for boundary_idx in boundary_nodes:
            # Find all rows that have non-zero entries in this boundary column
            # We need to check the original matrix (before boundary conditions were applied)
            col_data = A_orig.getcol(boundary_idx)
            rows, _ = col_data.nonzero()
            
            for row in rows:
                # Skip if this is a boundary row (already set to identity)
                if row in boundary_nodes:
                    continue
                    
                # Get the original matrix entry
                entry_value = A_orig[row, boundary_idx]
                
                if abs(entry_value) > 1e-14:  # Only process non-zero entries
                    # Remove this entry from the main matrix
                    A[row, boundary_idx] = 0
                    
                    # Add the opposite to the correction matrix
                    B[row, boundary_idx] = -entry_value
        
        # Convert back to csc_matrix for efficient operations
        A = A.tocsc()
        B = B.tocsc()
        
        # Apply correction to RHS: b' = b + B * boundary_values
        corrected_rhs = rhs + B.dot(boundary_values)
        
        return A, corrected_rhs
        
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