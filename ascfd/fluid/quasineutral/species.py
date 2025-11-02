from ascfd.fluid.species import FluidSpecies
from ascfd.params import SpeciesParams
from ascfd.inputs import Inputs
from ascfd.fields.fields import Fields

import numpy as np
import sys
import warnings

class QNFluidSpecies(FluidSpecies):
    def __init__(self, c, params: SpeciesParams, a_inputs: Inputs, fields: Fields, simulation):
        super().__init__(c, params, a_inputs, fields, simulation)
        
        self.var_grids = {}
        self.check_grid()
        
    # TODO: DOUBLE CHECK WE'RE HANDLING GHOSTS HERE RIGHT CUZ THIS ALWAYS TRIPS ME UP
    
    def update(self):
        """Update electron fluid with ion continuity, Ohm's law momentum, and Euler energy flux."""

        # CONSTANTS
        e = self.c.e 
        m_e = self.inp.m_e

        r_1d = np.linspace(self.inp.ylim[0], self.inp.ylim[1], self.inp.ny, endpoint=False)
        r = r_1d[np.newaxis, :]
        k_m = 2.5e-13
        
        nu_w = np.zeros_like(self.inp.internal_grid)
        nu_w[0:self.inp.L_x] = 1e7
                
        beta = np.ones_like(self.inp.internal_grid)
        beta[0:self.inp.L_x] = 0.1
        
        V_a = self.inp.V_anode # TODO or get from grid?
        V_c = self.inp.V_cathode
        
        ###
        B = self.fields.B[:, :, 1]
        
        # SPECIES PROPERTIES
        n_n_plus = self.simulation.get_species_number_density("n") # get from particle
        n_e_plus = self.simulation.get_species_number_density("i") # get n_i_plus from particle
        n_e_a_plus = np.mean(n_e_plus[self.inp.ng, :])
        n_e_c_plus = np.mean(n_e_plus[-self.inp.ng, :])
        n_0 = 1e18 # TODO
        
        flux_i_plus = self.simulation.get_species_flux("i")
        flux_e_plus = self.simulation.get_species_flux("e")
        
        # CALCULATED PROPERTIES
        nu = n_n_plus * k_m + nu_w + (beta/16) * (e*B / m_e)
        mu_perp = (e / m_e) * nu / (nu**2 + (e*B / m_e)**2) # TODO is perp only?
        
        ###
        
        V = np.linspace(V_a, V_c, self.inp.nx)[:, np.newaxis] * np.ones((self.inp.nx, self.inp.ny))
        V_star = V.copy()
        
        energy_e = 3.0 * np.ones_like(self.inp.internal_grid)  # [eV] - initial mean energy ~3 eV
        energy_e_a = 3.0  # [eV] at anode
        energy_e_c = 5.0

        # TERMS
        c_1_plus = flux_i_plus
        c_2_plus = r * B * mu_perp * n_e_plus
        c_3_plus = r * B * mu_perp * n_e_plus * (np.log(n_e_plus / n_0) - 1)
        c_4_plus = n_e_plus
        c_5_plus = n_n_plus * n_e_plus

        # c_6: Use gradient for robust differentiation (maintains array size)
        dV_dx = np.gradient(V, self.inp.dx, axis=0)
        dV_star_dx = np.gradient(V_star, self.inp.dx, axis=0)
        dE_dx = np.gradient(energy_e, self.inp.dx, axis=0)

        c_6_plus = e * mu_perp * n_e_plus * dV_dx * (dV_star_dx + (2/3) * (np.log(n_e_plus/n_0) - 1) * dE_dx)
        
        ###
        I_plus = \
            e * (
                V_a - V_c - \
                2/(3*e) * energy_e_a * np.log(n_e_a_plus / n_0) + \
                2/(3*e) * energy_e_c * np.log(n_e_c_plus / n_0) + \
                np.trapz(c_1_plus/(mu_perp * n_e_plus), None, self.inp.dx) - \
                2/(3*e) * np.trapz(c_3_plus/(mu_perp * n_e_plus), None, self.inp.dx) * (energy_e_c - energy_e_a)
                ) / np.trapz((beta/(mu_perp * n_e_plus)), None, self.inp.dx)

            
        def solve_electron_energy(eps_old, n_e, E, B, j_current, dt, dz, N, c1, c3, c5, c6, beta_coeff):
            """
            Solve electron energy equation semi-implicitly (vectorized for multiple radial locations)

            Parameters:
            -----------
            eps_old : array, shape (N,) or (N, M) - electron energy at old time [eV]
            n_e : array, shape (N,) or (N, M) - electron density [m^-3]
            E : array, shape (N,) or (N, M) - electric field [V/m]
            B : array, shape (N,) or (N, M) - magnetic field [T]
            j_current : scalar or array - discharge current [A]
            dt : float - timestep [s]
            dz : float - grid spacing [m]
            N : int - number of grid points (axial direction)
            c1, c3, c5, c6 : arrays, shape (N,) or (N, M) - coefficient arrays from LANDMARK model
            beta_coeff : array, shape (N,) or (N, M) - beta coefficient

            Returns:
            --------
            eps_new : array, shape (N,) or (N, M) - electron energy at new time [eV]

            Notes:
            ------
            If inputs are 2D (N, M), solves M independent energy equations in parallel
            """

            e = 1.602e-19  # elementary charge [C]
            m_e = 9.109e-31  # electron mass [kg]

            # Determine shape for arrays
            shape = eps_old.shape

            # Initialize arrays for tridiagonal system
            A = np.zeros(shape)  # lower diagonal
            B = np.zeros(shape)  # main diagonal
            C = np.zeros(shape)  # upper diagonal
            D = np.zeros(shape)  # right-hand side

            # Compute coefficients at cell centers
            c4_old = n_e  # simplified: c4 ~ n_e
            c4_new = n_e  # assume density doesn't change much

            # Compute thermal conductivity at cell faces (k+1/2)
            # Shape is (N+1,) or (N+1, M)
            if len(shape) == 1:
                kappa_face = np.zeros(N+1)
            else:
                kappa_face = np.zeros((N+1, shape[1]))

            for k in range(1, N):
                # Average epsilon at face (vectorized)
                eps_face = 0.5 * (eps_old[k-1] + eps_old[k])

                # Braginskii parallel thermal conductivity
                # kappa_parallel ~ 3.2 * n_e * T_e^(5/2) / (m_e * nu_ei)
                # Simplified here - use appropriate formula for your case
                T_e_face = eps_face * (2.0/3.0)  # Convert energy to temperature [eV]

                # Classical conductivity (simplified) - vectorized
                nu_ei = 2.9e-12 * n_e[k] * 15.0 / T_e_face**1.5  # Coulomb collision freq
                kappa_classical = 3.16e-5 * T_e_face**2.5 / (nu_ei / n_e[k])  # [W/m/eV]

                # Anomalous enhancement factor (tuning parameter!)
                alpha_anom = 10.0  # typical for Hall thrusters

                # Effective conductivity coefficient
                kappa_face[k] = (10.0/(9.0*e)) * alpha_anom * kappa_classical / dz

            # Build tridiagonal system for interior points
            for k in range(1, N-1):
                # Compute explicit source terms (Joule heating + convection + sources)
                # Note: W and U are placeholders - add proper physics if needed
                S_k = compute_sources(
                    k, eps_old, n_e, c1, c3, c5, c6, beta_coeff,
                    j_current, kappa_classical, W=0, U=E, dz=dz
                )

                # Tridiagonal coefficients (vectorized)
                A[k] = -kappa_face[k]                              # coefficient of eps[k-1]
                B[k] = c4_new[k]/dt + kappa_face[k] + kappa_face[k+1]  # coefficient of eps[k]
                C[k] = -kappa_face[k+1]                            # coefficient of eps[k+1]
                D[k] = c4_old[k]/dt * eps_old[k] + S_k            # RHS

            # Boundary conditions
            # Left boundary (k=0): fixed temperature or flux
            A[0] = 0.0
            B[0] = 1.0
            C[0] = 0.0
            D[0] = 3.0  # e.g., 3 eV at anode

            # Right boundary (k=N-1): zero gradient or fixed value
            A[N-1] = -1.0
            B[N-1] = 1.0
            C[N-1] = 0.0
            D[N-1] = 0.0  # zero gradient: eps[N-1] = eps[N-2]

            # Solve tridiagonal system using Thomas algorithm (handles vectorization)
            eps_new = thomas_algorithm(A, B, C, D, N)

            # Apply physical limiters
            eps_new = np.clip(eps_new, 0.5, 50.0)  # Keep in [0.5, 50] eV range

            return eps_new


        def compute_sources(k, eps_old, n_e, c1, c3, c5, c6, beta, I_plus, kappa, W, U, dz):
            """
            Compute explicit source terms for energy equation (equation 21)
            All convection, heating, and source terms except thermal conduction

            Parameters:
            -----------
            k : int - grid index
            eps_old : array, shape (N,) or (N, M) - old energy values
            All other arrays can be 1D or 2D, vectorized over second dimension if 2D
            """
            e = 1.602e-19

            # Get values at k+1/2 and k-1/2 faces
            # Vectorized to handle both 1D and 2D arrays
            c1_kp = 0.5 * (c1[k] + c1[k+1]) if k < c1.shape[0]-1 else c1[k]
            c1_km = 0.5 * (c1[k-1] + c1[k]) if k > 0 else c1[k]

            eps_kp = 0.5 * (eps_old[k] + eps_old[k+1]) if k < eps_old.shape[0]-1 else eps_old[k]
            eps_km = 0.5 * (eps_old[k-1] + eps_old[k]) if k > 0 else eps_old[k]

            beta_kp = 0.5 * (beta[k] + beta[k+1]) if k < beta.shape[0]-1 else beta[k]
            beta_km = 0.5 * (beta[k-1] + beta[k]) if k > 0 else beta[k]

            # Convective energy flux (5/3 terms)
            conv_flux_in = (5/3) * (c1_km - (1/e)*beta_km*I_plus) * eps_km
            conv_flux_out = (5/3) * (c1_kp - (1/e)*beta_kp*I_plus) * eps_kp
            convection = -(conv_flux_out - conv_flux_in) / dz

            # Source/sink terms (c6 term with pressure-like work)
            if k < c6.shape[0]-1:
                dU_dz = (U[k+1] - U[k]) / dz
                # Vectorized division with protection against division by zero
                source = c6[k] - c5[k] * kappa * np.where(eps_old[k] > 0, dU_dz / eps_old[k], 0)
            else:
                source = 0

            # Work term (c4 * dW/de term)
            # You'll need to compute dW/de based on your physics
            work = 0  # Add your work term calculation

            return convection + source + work


        def thomas_algorithm(a, b, c, d, n):
            """
            Solve tridiagonal system using Thomas algorithm (vectorized for multiple systems)

            System: A[k]*x[k-1] + B[k]*x[k] + C[k]*x[k+1] = D[k]

            Parameters:
            -----------
            a : array, shape (n,) or (n, m) - lower diagonal (coefficient of x[k-1])
            b : array, shape (n,) or (n, m) - main diagonal (coefficient of x[k])
            c : array, shape (n,) or (n, m) - upper diagonal (coefficient of x[k+1])
            d : array, shape (n,) or (n, m) - right-hand side
            n : int - system size (number of rows)

            Returns:
            --------
            x : array, shape (n,) or (n, m) - solution

            Notes:
            ------
            If inputs are 2D (n, m), solves m independent tridiagonal systems in parallel
            """
            # Determine if we're solving multiple systems
            shape = d.shape

            # Forward elimination
            c_prime = np.zeros(shape)
            d_prime = np.zeros(shape)

            c_prime[0] = c[0] / b[0]
            d_prime[0] = d[0] / b[0]

            for k in range(1, n):
                denom = b[k] - a[k] * c_prime[k-1]
                c_prime[k] = c[k] / denom
                d_prime[k] = (d[k] - a[k] * d_prime[k-1]) / denom

            # Back substitution
            x = np.zeros(shape)
            x[n-1] = d_prime[n-1]

            for k in range(n-2, -1, -1):
                x[k] = d_prime[k] - c_prime[k] * x[k+1]

            return x
        
        # Solve energy equation for all radial locations at once (vectorized)
        energy_e_plus = solve_electron_energy(
            eps_old=energy_e,
            n_e=n_e_plus,
            E=self.fields.E[:, :, 0],
            B=B,
            j_current=I_plus,
            dt=self.inp.dt,
            dz=self.inp.dx,
            N=self.inp.nx,
            c1=c_1_plus,
            c3=c_3_plus,
            c5=c_5_plus,
            c6=c_6_plus,
            beta_coeff=beta
        )

        ###
        # Compute V_star using cumulative integral along axial direction
        integrand = (c_1_plus - (beta * I_plus / e) - 2/(3*e) * c_3_plus) / (mu_perp * n_e_plus * r * B)
        V_star_plus = np.cumsum(integrand * self.inp.dx, axis=0) + (energy_e - energy_e_c)
        V_plus = V_star_plus + 2/(3*e) * energy_e_plus * np.log(n_e_plus / n_0)
        
        E = -np.diff(V_plus, 1, 0)
        self.fields.populate_E_field(E)
        
        # TODO check
        def f(U: np.ndarray, type):
            """Find k+1/2 (1) or k-1/2 (0) cell interfaces."""
            
            faces = 0.5 * (U[:-1, :] + U[1:, :])
            if type == 1:
                return faces
            elif type == 0:
                return np.pad(faces, ((1, 0), (0, 0)), mode="edge")[0:-1]
    
    
    def check_grid(self):
        U = self.grid
        
        for icomp in range(self.c.NUMQ):
            
            with open(f"output/debug/all_values.txt", "a") as f:
                f.write(f"\n[QN] VALUE FOR ICOMP={icomp}: {U[icomp]}")
            
            mask = np.isnan(U[icomp])
            inf_mask = np.isinf(U[icomp])
            masks = {"nan": mask, "inf": inf_mask}
            
            for mask_name in masks:
                mask = masks[mask_name]
                if mask.any():
                    rows, cols = np.where(mask)
                    rows = list(rows)
                    cols = list(cols)
                    
                    cells = [(rows[i]-2, cols[i]-2) for i in range(len(rows))]

                    with open(f"output/debug/{mask_name}_values.txt", "a") as f:
                        f.write(f"\n[QN] {mask_name} VALUE FOR ICOMP={icomp} AT: {cells}")
                        warnings.warn(f"[QN] {mask_name} VALUE FOR ICOMP={icomp} AT: {cells}", category=RuntimeWarning)
        
        for var in self.var_grids:
            mask = np.isnan(self.var_grids[var])
            inf_mask = np.isinf(self.var_grids[var])
            masks = {"nan": mask, "inf": inf_mask}
            
            for mask_name in masks:
                mask = masks[mask_name]
                if mask.any():
                    rows, cols = np.where(mask)
                    rows = list(rows)
                    cols = list(cols)
                    
                    cells = [(rows[i]-2, cols[i]-2) for i in range(len(rows))]

                    with open(f"output/debug/{mask_name}_values.txt", "a") as f:
                        f.write(f"\n[QN] {mask_name} VALUE FOR VAR={var} AT: {cells}")
                        warnings.warn(f"[QN] {mask_name} VALUE FOR VAR={var} AT: {cells}", category=RuntimeWarning)
            