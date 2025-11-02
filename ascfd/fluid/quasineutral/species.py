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
        
        nu_w = np.zeros_like(self.grid)
        nu_w[0:self.inp.nx/2] = 1e7
                
        beta = np.ones_like(self.grid)
        beta[0:self.inp.nx/2] = 0.1
        
        V_a = self.inp.V_anode # TODO or get from grid?
        V_c = self.inp.V_cathode
        
        ###
        B = self.fields.B[:, :, 1]
        
        # SPECIES PROPERTIES
        n_n_plus = self.simulation.get_species_number_density("n") # get from particle
        n_e_plus = self.simulation.gets_species_number_density("i") # get n_i_plus from particle
        n_e_a_plus = 
        n_e_c_plus = 
        n_0 = # ref density
        
        flux_i_plus = self.simulation.get_species_flux("i")
        flux_e_plus = self
        
        # CALCULATED PROPERTIES
        nu = n_n_plus * k_m + nu_w + (beta/16) * (e*B / m_e)
        mu_perp = (e / m_e) * nu / (nu**2 + (e*B / m_e)**2) # TODO is perp only?
        
        ###
        
        V = 
        V_star = 
        
        energy_e = 
        energy_e_a = 
        energy_e_c = 

        # TERMS
        c_1_plus = flux_i_plus
        c_2_plus = r * B * mu_perp * n_e_plus
        c_3_plus = r * B * mu_perp * n_e_plus * (np.log(n_e_plus / n_0) - 1)
        c_4_plus = n_e_plus
        c_5_plus = n_n_plus * n_e_plus
        c_6_plus = e * mu_perp * n_e_plus * np.diff(V, 1, 0) * (np.diff(V_star, 1, 0) + 2/3 * (np.log(n_e_plus/n_0 - 1) * np.diff(energy_e, 1, 0)))
        
        ###
        I_plus = \
            e * (
                V_a - V_c - \
                2/(3*e) * energy_e_a * np.log(n_e_a_plus / n_0) + \
                2/(3*e) * energy_e_c * np.log(n_e_c_plus / n_0) + \
                np.trapezoid(c_1_plus/(mu_perp * n_e_plus), None, self.inp.dx) - \
                2/(3*e) * np.trapezoid(c_3_plus/(mu_perp * n_e_plus), None, self.inp.dx) * (energy_e_c - energy_e_a)
                ) / np.trapezoid((beta/(mu_perp * n_e_plus)), None, self.inp.dx)
            
        
        ### TODO use implicit instead
        energy_e_plus = \
            self.dt * (
                5/3 * (f(c_1_plus, 1) - 1/e * f(beta, 1) * I_plus) * f(energy_e_plus, 1) - \
                5/3 * (f(c_1_plus, 0) - 1/e * f(beta, 0) * I_plus) * f(energy_e_plus, 0) - \
                10/(9*e) * f(c_2_plus, 1) * f(energy_e, 1) * np.diff()
            )
            
        def solve_electron_energy(eps_old, n_e, E, B, j, dt, dz, N):
            """
            Solve electron energy equation semi-implicitly
            
            Parameters:
            -----------
            eps_old : array, shape (N,) - electron energy at old time [eV]
            n_e : array, shape (N,) - electron density [m^-3]
            E : array, shape (N,) - electric field [V/m]
            B : array, shape (N,) - magnetic field [T]
            j : array, shape (N,) - current density [A/m^2]
            dt : float - timestep [s]
            dz : float - grid spacing [m]
            N : int - number of grid points
            
            Returns:
            --------
            eps_new : array, shape (N,) - electron energy at new time [eV]
            """
            
            e = 1.602e-19  # elementary charge [C]
            m_e = 9.109e-31  # electron mass [kg]
            
            # Initialize arrays for tridiagonal system
            A = np.zeros(N)  # lower diagonal
            B = np.zeros(N)  # main diagonal
            C = np.zeros(N)  # upper diagonal
            D = np.zeros(N)  # right-hand side
            
            # Compute coefficients at cell centers
            c4_old = n_e  # simplified: c4 ~ n_e
            c4_new = n_e  # assume density doesn't change much
            
            # Compute thermal conductivity at cell faces (k+1/2)
            kappa_face = np.zeros(N+1)
            for k in range(1, N):
                # Average epsilon at face
                eps_face = 0.5 * (eps_old[k-1] + eps_old[k])
                
                # Braginskii parallel thermal conductivity
                # kappa_parallel ~ 3.2 * n_e * T_e^(5/2) / (m_e * nu_ei)
                # Simplified here - use appropriate formula for your case
                T_e_face = eps_face * (2.0/3.0)  # Convert energy to temperature [eV]
                
                # Classical conductivity (simplified)
                nu_ei = 2.9e-12 * n_e[k] * 15.0 / T_e_face**1.5  # Coulomb collision freq
                kappa_classical = 3.16e-5 * T_e_face**2.5 / (nu_ei / n_e[k])  # [W/m/eV]
                
                # Anomalous enhancement factor (tuning parameter!)
                alpha_anom = 10.0  # typical for Hall thrusters
                
                # Effective conductivity coefficient
                kappa_face[k] = (10.0/(9.0*e)) * alpha_anom * kappa_classical / dz
            
            # Build tridiagonal system for interior points
            for k in range(1, N-1):
                # Compute explicit source terms
                S_k = compute_sources(k, eps_old, n_e, E, B, j)
                
                # Tridiagonal coefficients
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
            
            # Solve tridiagonal system using Thomas algorithm
            eps_new = thomas_algorithm(A, B, C, D, N)
            
            # Apply physical limiters
            eps_new = np.clip(eps_new, 0.5, 50.0)  # Keep in [0.5, 50] eV range
            
            return eps_new


        def compute_sources(k, eps, n_e, E, B, j):
            """
            Compute explicit source terms for energy equation
            
            Returns source in units of [eV/s]
            """
            e = 1.602e-19
            
            # Joule heating: j·E
            joule_heating = (j[k] * E[k]) / (e * n_e[k])  # [eV/s]
            
            # Ionization energy loss (simplified)
            # Rate = n_e * n_n * <sigma*v>_iz
            # Energy loss = Rate * (epsilon_iz + 3*T_e) where epsilon_iz ~ 12.1 eV for Xe
            n_neutral = 1e18  # [m^-3] - get from your neutral model
            T_e = eps[k] * (2.0/3.0)  # [eV]
            rate_coeff = 1e-14 * np.exp(-12.1/T_e) if T_e > 1.0 else 0.0  # [m^3/s]
            ionization_loss = -n_e[k] * n_neutral * rate_coeff * (12.1 + 3*T_e)  # [eV/m^3/s]
            ionization_loss /= n_e[k]  # [eV/s]
            
            # Convection terms (simplified - add your full terms from eq 21)
            convection = 0.0  # Add your convection terms here
            
            # Total source
            S = joule_heating + ionization_loss + convection
            
            return S


        def thomas_algorithm(a, b, c, d, n):
            """
            Solve tridiagonal system using Thomas algorithm
            
            System: A[k]*x[k-1] + B[k]*x[k] + C[k]*x[k+1] = D[k]
            
            Parameters:
            -----------
            a : array - lower diagonal (coefficient of x[k-1])
            b : array - main diagonal (coefficient of x[k])
            c : array - upper diagonal (coefficient of x[k+1])
            d : array - right-hand side
            n : int - system size
            
            Returns:
            --------
            x : array - solution
            """
            # Forward elimination
            c_prime = np.zeros(n)
            d_prime = np.zeros(n)
            
            c_prime[0] = c[0] / b[0]
            d_prime[0] = d[0] / b[0]
            
            for k in range(1, n):
                denom = b[k] - a[k] * c_prime[k-1]
                c_prime[k] = c[k] / denom
                d_prime[k] = (d[k] - a[k] * d_prime[k-1]) / denom
            
            # Back substitution
            x = np.zeros(n)
            x[n-1] = d_prime[n-1]
            
            for k in range(n-2, -1, -1):
                x[k] = d_prime[k] - c_prime[k] * x[k+1]
            
            return x
        
        energy_e_plus = solve_electron_energy(energy_e, n_e_plus, self.fields.E[:, :, 0], B, I_plus, self.inp.dt, self.inp.dx, self.inp.nx)
            
        # kappa = 10/(9*e) * c_2_plus * np.diff()
        
        ###
        V_star_plus = np.cumulative_sum((c_1_plus - (beta * I_plus / e) - 2/(3*e) * c_3_plus) / (mu_perp * B), axis=0) + (energy_e - energy_e_c)
        V_plus = V_star_plus + 2/(3*e) * energy_e_plus * np.log(n_e_plus / n_0)
        
        E = 
        
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
            