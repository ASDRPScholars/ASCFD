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
        
        print("[UPDATE] THIS IS NUMBER DENSITY", self.grid[self.c.NCOMP])
        
        ng = self.inp.ng
        U = self.grid
        
        e = self.c.e
        q_e = self.params.charge
        m_e = self.params.mass
        n_e = U[self.c.NCOMP, ng:-ng, ng:-ng]
        v_e = self.simulation.get_species_velocity("e")
        T_e = U[self.c.TCOMP, ng:-ng, ng:-ng]
        n_i = self.simulation.get_species_number_density("i")
        n_n = self.simulation.get_species_number_density("n")
        
        j_i_perp = self.simulation.get_species_current_density("i")
       
        k_B = self.c.k_B
        eps_0 = self.c.eps_0
        
        # --- LANDMARK ---
        eps_e = (3/2 * k_B * T_e) + (1/2 * m_e * v_e**2 / 1.60218e-19) # TODO DO WE STILL HAVE TO ADD VELOCITY THOUGH
        
        B = self.fields.B[:, :, 1]
        E_perp = self.fields.E[:, :, 0]
        
        p_e = n_e * k_B * T_e
        grad_p_e_perp = np.gradient(p_e, self.inp.dx, 0)[0]
        
        l_D = np.sqrt((eps_0 * k_B * T_e)/(q_e**2 * n_e)) # debye length - marks eq (2.2)
        
        plasma_param = 4 * np.pi * n_e * l_D**3 
        couloumb_ln = np.log(plasma_param) # coulomb logarithm - ln(lambda_C) = ln(lambda) - robert fitzpatrick 2016 UT notes eq (3.124)
        # TODO: does the above line up with MIT collision notes eq (64) ish?
        Z_star = n_i / n_e # effective charge number only for singly charged ions - mikellides eq (21)
        nu_ei = (n_e * Z_star * q_e**4 * couloumb_ln) / (3 * (2*np.pi)**(3/2) * eps_0**2 * np.sqrt(m_e) * (k_B * T_e)**(3/2))
        
        # nu_en = self.simulation.get_collision_frequency("en")
        # nu_en = 1e7
        # nu_anom = 0 
        # nu_e = nu_en + nu_anom
        
        # --- LANDMARK ---
        N = self.simulation.get_species_number_density("n")
        k_m = 2.5e-13
        nu_w = 1e7
        beta = 0.1 # TODO outside = 1
        
        nu_e = 1e7
        nu = (N * k_m) + nu_w + (beta * e * B / m_e) / 16
        
        K = 2e-14 # TODO add lookup
        
        mu_e = (e / m_e) * nu / (nu**2 + (e * self.fields.B[:, :, 1] / m_e)**2)
        # --- --------- ---
        
        omega_ce = -q_e * np.abs(self.fields.B[:, :, 1]) / m_e # cyclotron frequency - marks eq (2.3)
        hall_param = omega_ce / nu_e # hall parameter - marks eq (2.5), alternatively combine the two to get q_e * B / m_e * nu_e per (pg 50 inline)
        
        eta = (m_e * nu_e) / (q_e**2 * n_e) # eta, total resistivity - mikellides eq (23)
        eta_ei = (m_e * nu_ei) / (q_e**2 * n_e) # eta, electron-ion resistivity - mikellides eq (23)
        
        # mu_e = -q_e/(m_e * nu_e) # total electron mobility - textbook pg 68
        _, right_flux, left_flux, top_flux, bottom_flux = self.flux.getFlux(U, self.inp.nx, self.inp.ny, self.inp.ng, nu_e=nu_e, hall_param=hall_param, omega_ce=omega_ce, mu_e=mu_e)
        
        self.var_grids.update({
            "n_i": n_i,
            "n_n": n_n,
            "p_e": p_e,
            "grad_p_e_perp":grad_p_e_perp,
            "l_D": l_D,
            "plasma_param": plasma_param,
            "couloumb_ln": couloumb_ln,
            # "nu_en": nu_en,
            "nu_e": nu_e,
            "omega_ce": omega_ce,
            "hall_param": hall_param,
            "E_perp": E_perp
        })
        
        for icomp in range(self.c.NUMQ):
            # FIX CONSISTENT GHOSTS IF BOUNDS BECOME A REAL PROBLEM:
            # TODO: ION GET DENSITY FUNCTIONS INCLUDE GHOSTS...
            # TODO: BECAUSE WHEN WE SET GRID, THE GHOST CELLS ACTUALLY GET INCLUDED IN GRID BOUNDS (so tehnically bounds are off by (2*ng)/n in every sim...)
            if icomp == self.c.NCOMP:
                U[icomp, ng:-ng, ng:-ng] = n_i
                self.bcs.apply_bcs()
            
            # elif icomp == self.c.JYCOMP:
            #     (q_e**2 * n_e)/(m_e * nu_e) * (E_perp + np.gradient(p_e, 0, self.inp.dy)/(q_e*n_e)) # parallel electron current - marks eqs (2.29-2.31)
                
            # elif icomp == self.c.JZCOMP:
            #     j_theta = hall_param * j_e_perp # azimuthal electron current - marks eqs (2.29-2.31)
                
            elif icomp == self.c.TCOMP:
                delta = (self.dt / self.inp.dx) * (right_flux[icomp, ng:-ng, ng:-ng] - left_flux[icomp, ng:-ng, ng:-ng]) # + \
                        # (self.dt / self.inp.dy) * (top_flux[icomp, ng:-ng, ng:-ng] - bottom_flux[icomp, ng:-ng, ng:-ng])
                    
                U[icomp, ng:-ng, ng:-ng] -= delta
                self.bcs.apply_bcs()
                
                joule_source = n_e * v_e * e * E_perp
                # neutral_coll_source = n_e * N * K
                # wall_coll_source = nu_e * eps_e * np.exp(-U / eps_e) # TODO WHAT IS U??
                
                U[icomp, ng:-ng, ng:-ng] -= self.dt * (joule_source) #+ neutral_coll_source)
                self.bcs.apply_bcs()
                
            elif icomp == self.c.JXCOMP:
                # --- LANDMARK --- 
                j_perp = ((q_e * n_e * nu_e) / (omega_ce * B)) * (E_perp + (grad_p_e_perp / (q_e * n_e)))
                
                # n_eps_e = n_e * eps_e
                # n_u_e = mu_e * n_e * e * E_perp - mu_e * np.gradient(n_eps_e, self.inp.dx, axis=0)
                # E_perp = eta * (1 + hall_param**2) * (n_u_e * q_e)
                # U[icomp, ng:-ng, ng:-ng] = n_u_e / n_e
                
                print("j_i", np.shape(j_i_perp))
                print("nu_ei", np.shape(nu_ei))
                
                E_perp_new = m_e * nu_e / (q_e**2 * n_e) * (1 + hall_param**2) * j_perp - (grad_p_e_perp / (q_e * n_e)) + (m_e * nu_ei)/(q_e**2 * n_e)*j_i_perp
                
                self.fields.populate_E_field(E_perp_new)
                
                U[icomp, ng:-ng, ng:-ng] = j_perp
                self.bcs.apply_bcs()
                
        self.check_grid()
        
        print("a")
    
    
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
            