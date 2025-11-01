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
            