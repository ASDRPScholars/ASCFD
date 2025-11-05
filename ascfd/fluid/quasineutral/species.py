from ascfd.fluid.species import FluidSpecies
from ascfd.params import SpeciesParams
from ascfd.inputs import Inputs
from ascfd.fields.fields import Fields

import numpy as np
from scipy import linalg
from scipy import integrate
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
        
        energy_e = self.simulation.get_species_energy("e")
        
        energy_e_a = 3
        energy_e_c = 3
        deps_dx = np.gradient(energy_e, self.inp.dx, axis=0)
        
        ###
        B = self.fields.B[:, :, 1]
        
        # SPECIES PROPERTIES
        n_n_plus = self.simulation.get_species_number_density("n") # get from particle
        n_e = self.simulation.get_species_number_density("e")
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
        
        V = self.fields.potential
        V_star = self.fields.potential_star
        # V = np.linspace(V_a, V_c, self.inp.nx)[:, np.newaxis] * np.ones((self.inp.nx, self.inp.ny))
        # V_star = V.copy()

        area = np.pi * (0.050**2 - 0.035**2)
        
        # INTEGRALS
        c_1_plus = flux_i_plus
        c_2_plus = r * B * mu_perp * n_e_plus
        c_3_plus = r * B * mu_perp * n_e_plus * (np.log(n_e_plus / n_0) - 1)
        c_4 = n_e * area
        c_4_plus = n_e_plus * area
        c_5_plus = n_n_plus * n_e_plus * area

        # c_6: use np.grad for partial x's
        dV_dx = np.gradient(V, self.inp.dx, axis=0)
        dV_star_dx = np.gradient(V_star, self.inp.dx, axis=0)

        c_6_plus = e * mu_perp * n_e_plus * dV_dx * (dV_star_dx + (2/3) * (np.log(n_e_plus/n_0) - 1) * deps_dx)
        
        # #####
        
        u_e_plus = mu_perp * (dV_star_dx + 2/(3*e) * (np.log(n_e_plus/n_0) - 1) * deps_dx)
        
        ### DISCHARGE CURRENT (scalar)
        _I_plus = \
            e * (
                V_a - V_c - \
                2/(3*e) * energy_e_a * np.log(n_e_a_plus / n_0) + \
                2/(3*e) * energy_e_c * np.log(n_e_c_plus / n_0) + \
                integrate.trapezoid(c_1_plus[:, 20]/(mu_perp[:, 20] * n_e_plus[:, 20]), None, self.inp.dx) - \
                2/(3*e) * integrate.trapezoid(c_3_plus[:, 20]/(mu_perp[:, 20] * n_e_plus[:, 20]), None, self.inp.dx) * (energy_e_c - energy_e_a)
                ) / integrate.trapezoid((beta[:, 20]/(mu_perp[:, 20] * n_e_plus[:, 20])), None, self.inp.dx)
            
        I_plus = _I_plus # TODO HOW DO THEY DO IT? THEY DON'T MIDLINE AVERAGE?

        ### ######### ########

        ### ENERGY UPDATE!
        W = nu_e * energy_e * np.exp(-20/energy_e)
        dW_deps = nu_e * np.exp(-20/energy_e) * (20/energy_e + 1)
        
        K = self.simulation.get_coeffs("K", energy_e)
        dK_deps = self.simulation.get_coeffs("dK_deps", energy_e)
        # dK_deps = np.gradient(K, dx, axis=0) / np.gradient(energy_e, dx, axis=0) # chain rule - dK/deps = dK/dx * dx/depx
        
        X_plus = c_1_plus - 1/e * beta * I_plus

        A = -5/(6) * f(X_plus, -1/2) - \
            10/(9*e*dx) * mu_perp * n_e_plus * f(energy_e, -1/2)

        B = (c_4_plus / (dt)) + \
            10/(9*e*dx) * mu_perp * n_e_plus * (f(energy_e, +1/2) + f(energy_e, -1/2)) - \
            5/(6) * (f(X_plus, +1/2) - f(X_plus, -1/2)) - \
            c_5_plus * dK_deps - \
            c_4_plus * dW_deps

        C = 5/(6) * f(X_plus, +1/2) - \
            10/(9*e*dx) * mu_perp * n_e_plus * f(energy_e, +1/2)

        D = (1/(dt)) * c_4 * energy_e + \
            c_6_plus - \
            c_5_plus * K + \
            c_5_plus * energy_e * dK_deps - \
            c_4_plus * W + \
            c_4_plus * energy_e * dW_deps
            
        # flatten + pad bcs for matrix
        _A = np.pad(A[1:, 20], 2, mode='constant', constant_values=0)
        _B = np.pad(B[:, 20], 2, mode='constant', constant_values=1)
        _C = np.pad(C[:-1, 20], 2, mode='constant', constant_values=0)
        _D = np.pad(D[:, 20], 2, mode="constant", constant_values=(energy_e_a, energy_e_c))
        
        lower_diagonal = np.diag(_A, k=-1)
        main_diagonal = np.diag(_B)
        upper_diagonal = np.diag(_C, k=1)
        
        a = lower_diagonal + main_diagonal + upper_diagonal
        b = _D
        
        _energy_e_plus = linalg.solve(a, b, assume_a='tridiagonal')
        
        energy_e_plus = _energy_e_plus[ng:-ng, np.newaxis] * np.ones((self.inp.nx, self.inp.ny))
        
        deps_dx_plus = np.gradient(energy_e_plus, self.inp.dx, axis=0)
        
        ### ###### #######

        ### VOLTAGE UPDATE
        deps_dlambda = deps_dx_plus / (r * B)
        integrand = (c_1_plus - (beta * I_plus / e) - 2/(3*e)) / (mu_perp * n_e_plus * r * B) - 2*(np.log(n_e_plus / n_0) - 1)*deps_dlambda
        dVstar_dx = integrand * r * B
        V_star_anode = V_a - (2.0/3.0) * energy_e_a * np.log(n_e_a_plus / n_0)  # All in Volts
        V_star_plus = integrate.cumulative_trapezoid(dVstar_dx, dx=self.inp.dx, axis=0, initial=0) + V_star_anode
        
        V_plus = V_star_plus + 2.0/(3.0) * energy_e_plus * np.log(n_e_plus / n_0)
        
        self.fields.populate_V(V_plus, V_star_plus)
        
        ### ######## ######
        
        ### FINAL UPDATE!
        
        ng = self.inp.ng
        
        E = -np.gradient(V_plus, dx, axis=0)
        self.fields.populate_E(E)

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
            