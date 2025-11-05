from ascfd.fluid.species import FluidSpecies
from ascfd.params import SpeciesParams
from ascfd.inputs import Inputs
from ascfd.fields.fields import Fields

import numpy as np
from scipy import linalg
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
        
        # TODO check
        def f(U: np.ndarray, type):
            """Find k+1/2 (1) or k-1/2 (-1) cell interfaces."""
            
            _faces = 0.5 * (U[:-1, :] + U[1:, :])
            faces = np.concatenate((U[[0], :], _faces, U[[-1], :]), axis=0)
            
            if type == +1/2:
                return faces[1:, :]
            elif type == -1/2:
                return faces[:-1, :]
            
        # CONSTANTS
        dx = self.inp.dx
        dt = self.dt
        r_1d = np.linspace(self.inp.ylim[0], self.inp.ylim[1], self.inp.ny, endpoint=False)
        r = r_1d[np.newaxis, :]
        k_m = 2.5e-13
        
        ng = self.inp.ng
        
        e = self.c.e 
        m_e = self.inp.m_e
        
        nu_e = np.ones_like(self.inp.internal_grid) * 1e7
        nu_e[0:self.inp.L_x] = 4e6
        
        nu_w = np.zeros_like(self.inp.internal_grid)
        nu_w[0:self.inp.L_x] = 1e7
                
        beta = np.ones_like(self.inp.internal_grid)
        beta[0:self.inp.L_x] = 0.1
        
        V_a = self.inp.V_anode # TODO or get from grid?
        V_c = self.inp.V_cathode
        
        energy_e_a = 3.0 
        energy_e_c = 3.0
        
        ###
        B = self.fields.B[:, :, 1]
        
        # SPECIES PROPERTIES
        n_n_plus = self.simulation.get_species_number_density("n") # get from particle
        # n_e = self.simulation.get_species_number_density("e")
        # n_e_plus = self.simulation.get_species_number_density("i") # get n_i_plus from particle
        
        n_e = np.ones_like(self.inp.internal_grid) * 0.5e18
        x_cm = np.linspace(0, 2.5, self.inp.L_x)  # 0 to 2.5 cm

        # Combine rising Gaussian and falling exponential
        rise = 3e18 + 2.5e18 * np.exp(-((x_cm - 1.0) / 0.4)**2)  # Gaussian peak at 1 cm
        fall = 0.5e18 + 5e18 * np.exp(-(x_cm - 1.0) / 0.8)       # Exponential decay

        # Take maximum to get the profile shape
        density_profile = np.maximum(rise, fall)
        density_profile = np.clip(density_profile, 0.5e18, 5.5e18)

        n_e[0:self.inp.L_x, :] = density_profile[:, np.newaxis]
        n_e_plus = n_e
        
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
        
        energy_e = self.simulation.get_species_energy("e")

        # TERMS
        c_1_plus = flux_i_plus
        c_2_plus = r * B * mu_perp * n_e_plus
        c_3_plus = r * B * mu_perp * n_e_plus * (np.log(n_e_plus / n_0) - 1)
        c_4 = n_e
        c_4_plus = n_e_plus
        c_5_plus = n_n_plus * n_e_plus

        # c_6: use np.grad for partial x's
        dV_dx = np.gradient(V, self.inp.dx, axis=0)
        dV_star_dx = np.gradient(V_star, self.inp.dx, axis=0)
        deps_dx = np.gradient(energy_e, self.inp.dx, axis=0)

        c_6_plus = e * mu_perp * n_e_plus * dV_dx * (dV_star_dx + (2/3) * (np.log(n_e_plus/n_0) - 1) * deps_dx)
        
        # #####
        
        u_e_plus = mu_perp * (dV_star_dx + 2/(3*e) * (np.log(n_e_plus/n_0) - 1) * deps_dx)
        
        ### DISCHARGE CURRENT (scalar)
        _I_plus = \
            e * (
                V_a - V_c - \
                2/(3*e) * energy_e_a * np.log(n_e_a_plus / n_0) + \
                2/(3*e) * energy_e_c * np.log(n_e_c_plus / n_0) + \
                np.trapz(c_1_plus[:, 20]/(mu_perp[:, 20] * n_e_plus[:, 20]), None, self.inp.dx) - \
                2/(3*e) * np.trapz(c_3_plus[:, 20]/(mu_perp[:, 20] * n_e_plus[:, 20]), None, self.inp.dx) * (energy_e_c - energy_e_a)
                ) / np.trapz((beta[:, 20]/(mu_perp[:, 20] * n_e_plus[:, 20])), None, self.inp.dx)
            
        I_plus = _I_plus # TODO HOW DO THEY DO IT? THEY DON'T MIDLINE AVERAGE?

        ### ######### ########

        ### ENERGY UPDATE!
        W = nu_e * energy_e * np.exp(-20/energy_e)
        dW_deps = nu_e * np.exp(-20/energy_e) * (20/energy_e + 1)
        
        K = self.simulation.get_coeffs("K", energy_e)
        dK_deps = self.simulation.get_coeffs("dK_deps", energy_e)
        # dK_deps = np.gradient(K, dx, axis=0) / np.gradient(energy_e, dx, axis=0) # chain rule - dK/deps = dK/dx * dx/depx
        
        X_plus = c_1_plus - 1/e * beta * I_plus

        # Corrected coefficients - note the factors of e
        A = -5/(6*e) * f(X_plus, -1/2) - \
            10/(9*e**2*dx) * mu_perp * n_e_plus * f(energy_e, -1/2)

        B = (c_4_plus / (dt * e)) + \
            5/(6*e) * (f(X_plus, +1/2) - f(X_plus, -1/2)) + \
            10/(9*e**2*dx) * mu_perp * n_e_plus * (f(energy_e, +1/2) + f(energy_e, -1/2)) - \
            c_5_plus * dK_deps / e - \
            c_4_plus * dW_deps / e

        C = 5/(6*e) * f(X_plus, +1/2) - \
            10/(9*e**2*dx) * mu_perp * n_e_plus * f(energy_e, +1/2)

        D = (1/(dt*e)) * c_4 * energy_e + \
            c_6_plus / e - \
            c_5_plus * K / e + \
            c_5_plus * energy_e * dK_deps / e - \
            c_4_plus * W / e + \
            c_4_plus * energy_e * dW_deps / e
            
        # flatten + pad bcs for matrix
        _A = np.pad(A[1:, 20], 2, mode='constant', constant_values=0)
        _B = np.pad(B[:, 20], 2, mode='constant', constant_values=1)
        _C = np.pad(C[:-1, 20], 2, mode='constant', constant_values=0)
        _D = np.pad(D[:, 20], 2, mode="constant", constant_values=(energy_e_a, energy_e_c))
        
        lower_diagonal = np.diag(_A, k=-1)
        main_diagonal = np.diag(_B)
        upper_diagonal = np.diag(_C, k=1)
        
        a = lower_diagonal + main_diagonal + upper_diagonal
        b = _D / e
        
        _energy_e_plus = linalg.solve(a, b, assume_a='tridiagonal')
        
        energy_e_plus = _energy_e_plus[ng:-ng, np.newaxis] * np.ones((self.inp.nx, self.inp.ny))
        
        ### ###### #######

        ### VOLTAGE UPDATE
        integrand = (c_1_plus - (beta * I_plus / e) - 2/(3*e) * c_3_plus) / (mu_perp * n_e_plus * r * B)
        V_star_plus = np.cumsum(integrand * self.inp.dx, axis=0) + (energy_e - energy_e_c)
        V_plus = V_star_plus + 2/(3*e) * energy_e_plus * np.log(n_e_plus / n_0)
        
        ### ######## ######
        
        ### FINAL UPDATE!
        
        ng = self.inp.ng
        
        E = -np.gradient(V_plus, dx, axis=0)
        self.fields.populate_E_field(E)

        self.grid[self.c.NCOMP, ng:-ng, ng:-ng] = n_e_plus
        self.grid[self.c.UCOMP, ng:-ng, ng:-ng] = u_e_plus
        self.grid[self.c.ECOMP, ng:-ng, ng:-ng] = energy_e_plus
        
        ###
        
        self.bcs.apply_bcs()
        
    
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
            