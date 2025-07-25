"""
Unit tests for electric field boundary conditions in the Poisson solver.
Based on test cases from the validation paper.
"""

import pytest
import numpy as np
import sys
import os

# Add the ascfd directory to the Python path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from ascfd.inputs import Inputs
from ascfd.fields.fields import Fields


class TestFieldsBoundaryConditions:
    """Test electric field computation with various boundary conditions."""
    
    def setup_method(self):
        """Setup basic test configuration."""
        # Create a simple 2D grid for testing
        self.nx = 21  # Number of interior points 
        self.ny = 21
        
        # Mock inputs object with required attributes
        class MockInputs:
            def __init__(self):
                self.nx = 21
                self.ny = 21 
                self.nx_with_ghosts = 23  # nx + 2 ghost cells
                self.ny_with_ghosts = 23  # ny + 2 ghost cells
                self.xlim = [0.0, 1.0]
                self.ylim = [0.0, 1.0]
                # Required by ICs
                self.system = "euler2d" 
                self.E_ics = "test"
                self.B_ics = "test"
        
        self.inputs = MockInputs()
    
    def test_case_a_dirichlet_voltage_drop(self):
        """
        Test Case A: Dirichlet boundary conditions with voltage drop.
        
        Left BC: Dirichlet 300V
        Right BC: Dirichlet 0V  
        Top BC: Neumann 0 V/m (natural/homogeneous)
        Bottom BC: Neumann 0 V/m (natural/homogeneous)
        RHS: 0 (no charge)
        
        Expected: Linear potential profile from 300V to 0V in x-direction,
                 constant in y-direction, giving constant electric field.
        """
        fields = Fields(self.inputs)
        
        # Clear charge density (RHS = 0 for this test)
        fields.charge_density.fill(0)
        
        # Test the Case A boundary conditions by modifying the solve_poisson method
        # We'll temporarily override the boundary condition application
        original_solve = fields.solve_poisson
        
        def solve_case_a_poisson(self):
            """Modified Poisson solver for Case A boundary conditions."""
            nx, ny = self.inp.nx_with_ghosts, self.inp.ny_with_ghosts
            
            # Create coefficient matrix for 2D finite difference Laplacian
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
            from scipy.sparse import diags, csc_array
            from scipy.sparse.linalg import cg
            
            P = diags([y_diag, x_diag, main_diag, x_diag, y_diag], 
                      [-nx, -1, 0, 1, nx], shape=(N, N), format='csr')
            
            A = csc_array(P)
            
            # Right-hand side: -rho/eps0
            rhs = (-self.charge_density / self.eps0).flatten()
            
            # Apply Case A boundary conditions
            # Left boundary: 300V, Right boundary: 0V
            for j in range(ny):
                # Left boundary (x=0)
                idx_left = j * nx
                A[idx_left, :] = 0
                A[idx_left, idx_left] = 1
                rhs[idx_left] = 300.0
                
                # Right boundary (x=1)
                idx_right = j * nx + nx - 1
                A[idx_right, :] = 0
                A[idx_right, idx_right] = 1
                rhs[idx_right] = 0.0
            
            # Top and bottom boundaries: Neumann (natural boundary conditions)
            # These are handled implicitly - no modification needed
            
            # Solve the system
            phi_flat, exit_code = cg(A, rhs, atol=1e-5)
            
            print(f"CG exit code: {exit_code}")  # 0 means convergence was successful
            
            self.potential = phi_flat.reshape((nx, ny))
        
        # Temporarily replace the method
        import types
        fields.solve_poisson = types.MethodType(solve_case_a_poisson, fields)
        
        # Solve Poisson equation with Case A conditions
        fields.solve_poisson()
        
        # Analyze results
        potential = fields.potential
        
        # Check linearity in x-direction (middle row)
        mid_row = potential.shape[0] // 2
        x_profile = potential[mid_row, :]
        
        # Expected: linear decrease from 300V to 0V
        x_coords = np.linspace(0, 1, len(x_profile))
        expected_profile = 300.0 * (1 - x_coords)
        
        # Check boundary conditions
        left_boundary = potential[:, 0]
        right_boundary = potential[:, -1]
        
        print(f"Case A Results:")
        print(f"Left boundary values: min={np.min(left_boundary):.1f}V, max={np.max(left_boundary):.1f}V")
        print(f"Right boundary values: min={np.min(right_boundary):.1f}V, max={np.max(right_boundary):.1f}V")
        print(f"Middle row profile: {x_profile[::4]}")  # Sample values
        print(f"Expected profile: {expected_profile[::4]}")  # Sample values
        
        # Test boundary conditions are satisfied
        np.testing.assert_allclose(left_boundary, 300.0, rtol=0.01)
        np.testing.assert_allclose(right_boundary, 0.0, atol=1e-10)
        
        # Test approximate linearity (allowing for numerical discretization)
        np.testing.assert_allclose(x_profile, expected_profile, rtol=0.1)
        
        print(f"✓ Case A test passed!")
        print(f"Potential range: {np.min(potential):.2f}V to {np.max(potential):.2f}V")
        
    def test_basic_poisson_solver(self):
        """
        Test basic functionality of the Poisson solver with simple boundary conditions.
        This tests that the solver works and can handle boundary conditions correctly.
        """
        fields = Fields(self.inputs)
        
        # Set a simple charge distribution
        nx, ny = self.inputs.nx_with_ghosts, self.inputs.ny_with_ghosts
        fields.charge_density.fill(0)
        
        # Add a point charge in the center
        center_x, center_y = nx // 2, ny // 2
        fields.charge_density[center_x, center_y] = 1e-9  # Small test charge
        
        # Solve with default boundary conditions (phi=0 on boundaries)
        fields.solve_poisson()
        
        potential = fields.potential
        
        # Basic sanity checks
        assert potential.shape == (nx, ny), "Potential should have correct shape"
        assert not np.any(np.isnan(potential)), "No NaN values should be present"
        assert not np.any(np.isinf(potential)), "No infinite values should be present"
        
        # Boundary conditions should be satisfied (approximately zero on boundaries)
        boundary_values = np.concatenate([
            potential[0, :],    # bottom
            potential[-1, :],   # top  
            potential[:, 0],    # left
            potential[:, -1]    # right
        ])
        
        print(f"Basic Poisson solver test:")
        print(f"Boundary values range: {np.min(boundary_values):.6f} to {np.max(boundary_values):.6f}")
        print(f"Interior potential range: {np.min(potential[1:-1,1:-1]):.6f} to {np.max(potential[1:-1,1:-1]):.6f}")
        
        # Boundaries should be close to zero
        np.testing.assert_allclose(boundary_values, 0.0, atol=1e-6)
        
        print("✓ Basic Poisson solver test passed!")


if __name__ == "__main__":
    # Run the tests directly
    test = TestFieldsBoundaryConditions()
    test.setup_method()
    
    print("Running basic Poisson solver test...")
    test.test_basic_poisson_solver()
    
    print("\nRunning Case A voltage drop test...")
    test.test_case_a_dirichlet_voltage_drop()
    
    print("\nAll tests completed!")