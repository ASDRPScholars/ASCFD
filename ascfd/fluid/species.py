from ascfd.fluid.constants import FluidConstants
from ascfd.fluid.euler import FluidEuler
from ascfd.fluid.ics import FluidInitialConditions
from ascfd.inputs import Inputs
from ascfd.params import SpeciesParams
from ascfd.fluid.bcs import FluidBoundaryConditions
from ascfd.fluid.flux import FluidFlux
from ascfd.fields.fields import Fields

import ascfd.fluid.ics as ics

import numpy as np
from scipy.sparse import csr_matrix, diags, bmat
from scipy.sparse.linalg import spsolve


class FluidSpecies:
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields):
        self.c = FluidConstants(a_inputs)
        # Keep legacy components but don't use them for hall_thruster
        self.euler = FluidEuler(self.c)
        self.flux = FluidFlux(self.c, a_inputs.flux)
        
        self.fields = fields
        self.inp = a_inputs
        self.params = params
        self.dt = None
        
        self.grid = np.zeros((self.c.NUMQ, self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        
        # Physical constants for hall thruster solver
        self.e = 1.602e-19      # Elementary charge [C]
        self.m_e = 9.109e-31    # Electron mass [kg]
        self.k_B = 1.381e-23    # Boltzmann constant [J/K]
        self.epsilon_0 = 8.854e-12  # Permittivity [F/m]
        
        # Initialize differential operators for hall thruster
        if a_inputs.system == "hall_thruster":
            self._build_differential_operators()
        
        self.bcs = FluidBoundaryConditions(self.grid, self.inp.bcs_lo, self.inp.bcs_hi, self.inp)
        self.ics = FluidInitialConditions(self.grid, self.inp)
        
        self.bcs.apply_bcs()
        self.check_grid(self.c)
        self.ics.apply_ics()
        
        
    def update(self, ion_species_data=None):
        """
        Update fluid species - uses different methods based on system type
        
        Args:
            ion_species_data: For hall_thruster, list of dicts with 'density', 'velocity', 'charge'
        """
        if self.inp.system == "hall_thruster":
            self._update_hall_thruster(ion_species_data)
        else:
            self._update_euler()  # Legacy Euler solver
    
    def _update_euler(self):
        """Legacy Euler solver update"""
        E = self.fields.E
        B = self.fields.B
        
        consU = self.euler.prim_to_cons(self.grid)
        consU_new = self.euler.prim_to_cons(self.grid)
        
        _, right_flux, left_flux, top_flux, bottom_flux = self.flux.getFlux(self.grid, self.inp.nx, self.inp.ny, self.inp.numghosts)
        
        self.bcs.apply_bcs()
        
        for i in range(self.inp.numghosts, self.inp.nx + self.inp.numghosts):
            for j in range(self.inp.numghosts, self.inp.ny + self.inp.numghosts):
                for icomp in range(self.c.NUMQ):
                    consU_new[icomp, i, j] = consU[icomp, i, j] - (
                        (self.dt / self.inp.dx) * (right_flux[icomp, i, j] - left_flux[icomp, i, j]) +
                        (self.dt / self.inp.dy) * (top_flux[icomp, i, j] - bottom_flux[icomp, i, j]))
                    
        self.bcs.apply_bcs()
        self.grid = self.euler.cons_to_prim(consU_new)
    
    def _update_hall_thruster(self, ion_species_data):
        """
        Update electron fluid using quasineutral + quasistatic solver
        
        Args:
            ion_species_data: List of dicts with 'density', 'velocity', 'charge' for each ion species
        """
        if ion_species_data is None:
            raise ValueError("ion_species_data required for hall_thruster system")
        
        # Extract ion data
        n_i_species = [species['density'] for species in ion_species_data]
        u_i_species = [species['velocity'] for species in ion_species_data] 
        z_i_species = [species['charge'] for species in ion_species_data]
        
        # Previous electron temperature
        T_e_old = self.grid[self.c.T_E_COMP]
        
        # Solve coupled 6-variable system
        solution = self.solve_coupled_electron_system(n_i_species, u_i_species, z_i_species, T_e_old, self.dt)
        
        # Update grid with solution (solution is interior-only, map to ghost-inclusive grid)
        ng = self.inp.numghosts
        self.grid[self.c.N_E_COMP, ng:-ng, ng:-ng] = solution['n_e']
        self.grid[self.c.E_PAR_COMP, ng:-ng, ng:-ng] = solution['E_parallel']
        self.grid[self.c.E_PERP_COMP, ng:-ng, ng:-ng] = solution['E_perp']
        self.grid[self.c.J_E_PAR_COMP, ng:-ng, ng:-ng] = solution['j_e_parallel']
        self.grid[self.c.J_E_PERP_COMP, ng:-ng, ng:-ng] = solution['j_e_perp']
        self.grid[self.c.T_E_COMP, ng:-ng, ng:-ng] = solution['T_e']
        
        # Update fields with new electric field components
        self.fields.E = np.sqrt(solution['E_parallel']**2 + solution['E_perp']**2)
        self.fields.Ex = solution['E_parallel']  # Assuming parallel is x-direction
        self.fields.Ey = solution['E_perp']      # Assuming perp is y-direction
        
        self.bcs.apply_bcs()
    
    
    def get_charge_density(self):
        if self.inp.system == "hall_thruster":
            charge_density = self.params.charge * self.get_number_density()
        else:
            charge_density = self.params.charge * self.get_number_density()
        return charge_density
    
    def get_number_density(self):
        if self.inp.system == "hall_thruster":
            number_density = self.grid[self.c.N_E_COMP]  # Electron density directly stored
        else:
            number_density = self.grid[self.c.RHOCOMP] / self.params.mass
        return number_density
    
    
    # TODO: make assert_variable_type -> prim or cons work
    # TODO: check particles ("check_grid()") for neutrals and ions too
    def check_grid(self, prim=False, cons=False):
        # Check for negative or invalid values in the grid
        for i in range(self.inp.nx + 2 * self.inp.numghosts):
            for j in range(self.inp.ny + 2 * self.inp.numghosts):
                # if prim:
                #     # Check for negative pressure
                #     if self.grid[self.c.PCOMP, i, j] <= 0:
                #         print(f"Negative Pressure - Bad cell: ({i}, {j})")
                #         assert False

                #     # Check for negative density
                #     if self.grid[self.c.RHOCOMP, i, j] <= 0:
                #         print(f"Negative Density - Bad cell: ({i}, {j})")
                #         assert False

                # if cons:
                #     # Check for negative energy
                #     if self.grid[self.c.ECOMP, i, j] <= 0:
                #         print(f"Negative Energy - Bad cell: ({i}, {j})")
                #         assert False

                # Check for NaN values
                for icomp in range(self.c.NUMQ):
                    if np.isnan(self.grid[icomp, i, j]):
                        print(f"NaN value - Bad cell: ({i}, {j}), component: {icomp}")
                        assert False
    
    def _build_differential_operators(self):
        """Build discrete differential operators for hall thruster solver"""
        n_total = self.inp.nx * self.inp.ny
        
        # Build gradient operators (central difference)
        self.grad_x = self._build_gradient_x()
        self.grad_y = self._build_gradient_y()
        
        # Divergence operator
        self.div_x = self.grad_x.T  # Divergence is transpose of gradient
        self.div_y = self.grad_y.T
    
    def _build_gradient_x(self):
        """Build x-gradient operator matrix"""
        nx, ny = self.inp.nx, self.inp.ny
        n_total = nx * ny
        data = []
        row_ind = []
        col_ind = []
        
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                
                if i == 0:  # Forward difference
                    row_ind.extend([idx, idx])
                    col_ind.extend([idx, (i+1)*ny + j])
                    data.extend([-1/self.inp.dx, 1/self.inp.dx])
                elif i == nx-1:  # Backward difference
                    row_ind.extend([idx, idx])
                    col_ind.extend([(i-1)*ny + j, idx])
                    data.extend([-1/self.inp.dx, 1/self.inp.dx])
                else:  # Central difference
                    row_ind.extend([idx, idx])
                    col_ind.extend([(i-1)*ny + j, (i+1)*ny + j])
                    data.extend([-1/(2*self.inp.dx), 1/(2*self.inp.dx)])
                    
        return csr_matrix((data, (row_ind, col_ind)), shape=(n_total, n_total))
    
    def _build_gradient_y(self):
        """Build y-gradient operator matrix"""
        nx, ny = self.inp.nx, self.inp.ny
        n_total = nx * ny
        data = []
        row_ind = []
        col_ind = []
        
        for i in range(nx):
            for j in range(ny):
                idx = i * ny + j
                
                if j == 0:  # Forward difference
                    row_ind.extend([idx, idx])
                    col_ind.extend([idx, i*ny + (j+1)])
                    data.extend([-1/self.inp.dy, 1/self.inp.dy])
                elif j == ny-1:  # Backward difference
                    row_ind.extend([idx, idx])
                    col_ind.extend([i*ny + (j-1), idx])
                    data.extend([-1/self.inp.dy, 1/self.inp.dy])
                else:  # Central difference
                    row_ind.extend([idx, idx])
                    col_ind.extend([i*ny + (j-1), i*ny + (j+1)])
                    data.extend([-1/(2*self.inp.dy), 1/(2*self.inp.dy)])
                    
        return csr_matrix((data, (row_ind, col_ind)), shape=(n_total, n_total))
    
    def solve_coupled_electron_system(self, n_i_species, u_i_species, z_i_species, T_e_old, dt):
        """
        Solve the complete coupled system for all 6 electron variables
        
        Args:
            n_i_species: List of ion densities [m^-3] for each charge state
            u_i_species: List of ion velocities [m/s] for each charge state
            z_i_species: List of ion charges for each charge state
            T_e_old: Previous electron temperature [eV]
            dt: Timestep [s]
            
        Returns:
            solution: Dictionary with all 6 variables
        """
        n_total = self.inp.nx * self.inp.ny
        n_vars = 6
        n_total_unknowns = n_vars * n_total
        
        # Build the full system matrix and RHS
        A_matrix = self._build_coupled_matrix(n_i_species, u_i_species, z_i_species, T_e_old, dt)
        b_vector = self._build_coupled_rhs(n_i_species, u_i_species, z_i_species, T_e_old, dt)
        
        # Solve the linear system
        solution_vector = spsolve(A_matrix, b_vector)
        
        # Extract individual variables
        solution = self._extract_solution(solution_vector)
        
        return solution
    
    def _build_coupled_matrix(self, n_i_species, u_i_species, z_i_species, T_e_old, dt):
        """Build the full coupled system matrix"""
        n_total = self.inp.nx * self.inp.ny
        
        # Initialize block matrix (6x6 blocks)
        blocks = [[None for _ in range(6)] for _ in range(6)]
        
        # Get physical parameters
        B_field = np.mean(self.fields.B)  # Simplified - use mean B field
        omega_ce = self._compute_cyclotron_frequency(B_field)
        nu_e = self._compute_collision_frequency(T_e_old)
        
        # Block (0,0): Quasineutrality equation
        # n_e = Σ Z_s * n_i,s  →  n_e - Σ Z_s * n_i,s = 0
        blocks[0][0] = self._identity_matrix(n_total)  # dn_e coefficient
        
        # Block (1,3) and (1,4): Current conservation
        # ∇·(j_e + j_i) = 0  →  ∇·j_e = -∇·j_i
        # Simplified: just set j_e = 0 for now to avoid singularity
        blocks[1][3] = self._identity_matrix(n_total)  # j_e_par = 0
        blocks[1][4] = self._zero_matrix(n_total)
        
        # Block (2,0), (2,1), (2,3): Ohm's law parallel component
        blocks[2][0], blocks[2][1], blocks[2][3] = self._ohms_law_parallel_blocks(nu_e, n_total)
        
        # Block (3,0), (3,2), (3,4): Ohm's law perpendicular component  
        blocks[3][0], blocks[3][2], blocks[3][4] = self._ohms_law_perp_blocks(omega_ce, nu_e, n_total)
        
        # Block (4,4): Azimuthal current relation
        blocks[4][4] = self._identity_matrix(n_total)  # j_e_perp coefficient
        
        # Block (5,0), (5,5): Electron energy equation
        blocks[5][0], blocks[5][5] = self._energy_equation_blocks(T_e_old, dt, n_total)
        
        # Fill zero blocks
        for i in range(6):
            for j in range(6):
                if blocks[i][j] is None:
                    blocks[i][j] = self._zero_matrix(n_total)
        
        # Assemble block matrix
        A_matrix = bmat(blocks, format='csr')
        
        return A_matrix
    
    def _build_coupled_rhs(self, n_i_species, u_i_species, z_i_species, T_e_old, dt):
        """Build the right-hand side vector"""
        n_total = self.inp.nx * self.inp.ny
        rhs = np.zeros(6 * n_total)
        
        # RHS for quasineutrality (block 0)
        rhs[0*n_total:(0+1)*n_total] = self._quasineutrality_rhs(n_i_species, z_i_species)
        
        # RHS for current conservation (block 1) - simplified to j_e = 0
        rhs[1*n_total:(1+1)*n_total] = np.zeros(n_total)
        
        # RHS for Ohm's law parallel (block 2)
        rhs[2*n_total:(2+1)*n_total] = self._ohms_parallel_rhs(n_total)
        
        # RHS for Ohm's law perpendicular (block 3)
        rhs[3*n_total:(3+1)*n_total] = self._ohms_perp_rhs(n_total)
        
        # RHS for azimuthal current (block 4)
        rhs[4*n_total:(4+1)*n_total] = np.zeros(n_total)
        
        # RHS for energy equation (block 5)
        rhs[5*n_total:(5+1)*n_total] = self._energy_equation_rhs(T_e_old, dt)
        
        return rhs
    
    def _extract_solution(self, solution_vector):
        """Extract individual variables from solution vector"""
        nx, ny = self.inp.nx, self.inp.ny
        n_total = nx * ny
        
        solution = {
            'n_e': solution_vector[0*n_total:(0+1)*n_total].reshape(nx, ny),
            'E_parallel': solution_vector[1*n_total:(1+1)*n_total].reshape(nx, ny),
            'E_perp': solution_vector[2*n_total:(2+1)*n_total].reshape(nx, ny),
            'j_e_parallel': solution_vector[3*n_total:(3+1)*n_total].reshape(nx, ny),
            'j_e_perp': solution_vector[4*n_total:(4+1)*n_total].reshape(nx, ny),
            'T_e': solution_vector[5*n_total:(5+1)*n_total].reshape(nx, ny)
        }
        
        return solution
    
    # Helper methods for building matrix blocks
    def _identity_matrix(self, n_total):
        return csr_matrix(np.eye(n_total))
    
    def _zero_matrix(self, n_total):
        return csr_matrix((n_total, n_total))
    
    def _compute_cyclotron_frequency(self, B_field):
        """Compute electron cyclotron frequency"""
        return self.e * B_field / self.m_e
    
    def _compute_collision_frequency(self, T_e):
        """Compute electron collision frequency"""
        # Simplified collision frequency
        return np.ones_like(T_e) * 1e8  # [1/s]
    
    def _ohms_law_parallel_blocks(self, nu_e, n_total):
        """Build matrix blocks for parallel Ohm's law"""
        # j_e,|| = (q_e^2 n_e)/(m_e nu_e) * (E_|| + ∇||p_e/(q_e n_e))
        
        # Simplified blocks
        n_e_block = self._zero_matrix(n_total)  
        E_par_block = self._identity_matrix(n_total)  
        j_par_block = -self._identity_matrix(n_total)
        
        return n_e_block, E_par_block, j_par_block
    
    def _ohms_law_perp_blocks(self, omega_ce, nu_e, n_total):
        """Build matrix blocks for perpendicular Ohm's law"""
        # Similar structure to parallel
        n_e_block = self._zero_matrix(n_total)
        E_perp_block = self._identity_matrix(n_total)
        j_perp_block = -self._identity_matrix(n_total)
        
        return n_e_block, E_perp_block, j_perp_block
    
    def _energy_equation_blocks(self, T_e_old, dt, n_total):
        """Build matrix blocks for energy equation"""
        # Simplified energy equation blocks
        n_e_block = self._zero_matrix(n_total)
        T_e_block = self._identity_matrix(n_total)
        
        return n_e_block, T_e_block
    
    def _quasineutrality_rhs(self, n_i_species, z_i_species):
        """RHS for quasineutrality: Σ Z_s * n_i,s"""
        nx, ny = self.inp.nx, self.inp.ny
        rhs = np.zeros(nx * ny)
        
        for Z, n_i in zip(z_i_species, n_i_species):
            # Convert from ghost-inclusive grid to interior grid
            n_i_interior = n_i[self.inp.numghosts:self.inp.numghosts+nx, 
                               self.inp.numghosts:self.inp.numghosts+ny]
            rhs += Z * n_i_interior.flatten()
        return rhs
    
    def _current_conservation_rhs(self, n_i_species, u_i_species, z_i_species):
        """RHS for current conservation: -∇·j_i"""
        nx, ny = self.inp.nx, self.inp.ny
        j_i_total = np.zeros((nx, ny, 2))  # x and y components
        
        for Z, n_i, u_i in zip(z_i_species, n_i_species, u_i_species):
            # Extract interior grids
            n_i_int = n_i[self.inp.numghosts:self.inp.numghosts+nx, 
                          self.inp.numghosts:self.inp.numghosts+ny]
            u_i_int = u_i[self.inp.numghosts:self.inp.numghosts+nx, 
                          self.inp.numghosts:self.inp.numghosts+ny]
            
            j_i_total[:,:,0] += Z * self.e * n_i_int * u_i_int[:,:,0]  # x-component
            j_i_total[:,:,1] += Z * self.e * n_i_int * u_i_int[:,:,1]  # y-component
        
        # Compute divergence
        div_j_i_x = self.div_x @ j_i_total[:,:,0].flatten()
        div_j_i_y = self.div_y @ j_i_total[:,:,1].flatten()
        
        return -(div_j_i_x + div_j_i_y)
    
    def _ohms_parallel_rhs(self, n_total):
        """RHS for parallel Ohm's law"""
        return np.zeros(n_total)
    
    def _ohms_perp_rhs(self, n_total):
        """RHS for perpendicular Ohm's law"""
        return np.zeros(n_total)
    
    def _energy_equation_rhs(self, T_e_old, dt):
        """RHS for energy equation"""
        nx, ny = self.inp.nx, self.inp.ny
        T_e_interior = T_e_old[self.inp.numghosts:self.inp.numghosts+nx, 
                               self.inp.numghosts:self.inp.numghosts+ny]
        return T_e_interior.flatten() / dt
                        