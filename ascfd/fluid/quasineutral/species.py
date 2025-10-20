from ascfd.fluid.species import FluidSpecies
from ascfd.params import SpeciesParams
from ascfd.inputs import Inputs
from ascfd.fields.fields import Fields

import numpy as np
import sys
import warnings

class QNFluidSpecies(FluidSpecies):
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields, simulation):
        super().__init__(params, a_inputs, fields, simulation)
        
        self.var_grids = {}
        
        self.E_perp = 4e4 # TODO: better than 0 somehow
        self.check_grid()
        
    # TODO: DOUBLE CHECK WE'RE HANDLING GHOSTS HERE RIGHT CUZ THIS ALWAYS TRIPS ME UP
    
    def update(self):
        """Update electron fluid with ion continuity, Ohm's law momentum, and Euler energy flux."""
        
        print("[UPDATE] THIS IS NUMBER DENSITY", self.grid[self.c.NCOMP])
        
        ng = self.inp.ng
        U = self.grid
        
        q_e = self.params.charge
        m_e = self.params.mass
        n_e = U[self.c.NCOMP, ng:-ng, ng:-ng]
        T_e = U[self.c.TCOMP, ng:-ng, ng:-ng]
        n_i = self.simulation.get_species_number_density("i")
        n_n = self.simulation.get_species_number_density("n")
        
        k_B = self.c.k_B
        eps_0 = self.c.eps_0
        
        p_e = n_e * k_B * T_e
        grad_p_e_perp = np.gradient(p_e, self.inp.dx, 0)[0]
        
        l_D = np.sqrt((eps_0 * k_B * T_e)/(q_e**2 * n_e)) # debye length - marks eq (2.2)
        
        plasma_param = 4 * np.pi * n_e * l_D**3 
        couloumb_ln = np.log(plasma_param) # coulomb logarithm - ln(lambda_C) = ln(lambda) - robert fitzpatrick 2016 UT notes eq (3.124)
        # TODO: does the above line up with MIT collision notes eq (64) ish?
        
        Z_star = n_i / n_e # effective charge number only for singly charged ions - mikellides eq (21)
        
        nu_ei = (n_e * Z_star * q_e**4 * couloumb_ln) / (3 * (2*np.pi)**(3/2) * eps_0**2 * np.sqrt(m_e) * (k_B * T_e)**(3/2))
        nu_en = self.simulation.get_collision_frequency("en")
        nu_anom = 0 # TODO (much later)
        # nu_e = nu_ei + nu_en + nu_anom # TODO: SHOULLD WE STILL USE NU_EI FROM MIKELLIDES
        nu_e = nu_en + nu_anom
        
        omega_ce = q_e * np.abs(self.fields.B[:, :, 1]) / m_e # cyclotron frequency - marks eq (2.3)
        hall_param = omega_ce / nu_e # hall parameter - marks eq (2.5), alternatively combine the two to get q_e * B / m_e * nu_e per (pg 50 inline)
        
        eta = (m_e * nu_e) / (q_e**2 * n_e) # eta, total resistivity - mikellides eq (23)
        eta_ei = (m_e * nu_ei) / (q_e**2 * n_e) # eta, electron-ion resistivity - mikellides eq (23)
        
        mu_e = -q_e/(m_e * nu_e) # total electron mobility - textbook pg 68
        _, right_flux, left_flux, top_flux, bottom_flux = self.flux.getFlux(U, self.inp.nx, self.inp.ny, self.inp.ng, nu_e=nu_e, hall_param=hall_param, omega_ce=omega_ce)
        
        self.var_grids.update({
            "n_i": n_i,
            "n_n": n_n,
            "p_e": p_e,
            "grad_p_e_perp":grad_p_e_perp,
            "l_D": l_D,
            "plasma_param": plasma_param,
            "couloumb_ln": couloumb_ln,
            "nu_en": nu_en,
            "nu_e": nu_e,
            "omega_ce": omega_ce,
            "hall_param": hall_param,
            "E_perp": self.E_perp
        })
        
        for icomp in range(self.c.NUMQ):
            # FIX CONSISTENT GHOSTS IF BOUNDS BECOME A REAL PROBLEM:
            # TODO: ION GET DENSITY FUNCTIONS INCLUDE GHOSTS...
            # TODO: BECAUSE WHEN WE SET GRID, THE GHOST CELLS ACTUALLY GET INCLUDED IN GRID BOUNDS (so tehnically bounds are off by (2*ng)/n in every sim...)
            if icomp == self.c.NCOMP:
                U[icomp, ng:-ng, ng:-ng] = n_i
            
            # elif icomp == self.c.JYCOMP:
            #     (q_e**2 * n_e)/(m_e * nu_e) * (self.E_perp + np.gradient(p_e, 0, self.inp.dy)/(q_e*n_e)) # parallel electron current - marks eqs (2.29-2.31)
                
            # elif icomp == self.c.JZCOMP:
            #     j_theta = hall_param * j_e_perp # azimuthal electron current - marks eqs (2.29-2.31)
                
            elif icomp == self.c.TCOMP:
                delta = (self.dt / self.inp.dx) * (right_flux[icomp, ng:-ng, ng:-ng] - left_flux[icomp, ng:-ng, ng:-ng]) + \
                        (self.dt / self.inp.dy) * (top_flux[icomp, ng:-ng, ng:-ng] - bottom_flux[icomp, ng:-ng, ng:-ng])
                    
                U[icomp, ng:-ng, ng:-ng] -= delta
                
                self.bcs.apply_bcs()
                
                # ACTUAL TODO: add other source terms later
                
                # TODO: complete implement based on mikellides 2012 eq. (25)
                # do necessary conversion between E and n * k_B * T_e?
                # EDIT FLUX DEFINED IN EULER?
                
                # U[icomp, ng:-ng, ng:-ng] -= delta
                # add heat flux source terms
                
            elif icomp == self.c.JXCOMP:
                # j_e = q_e * n_e * u_e
                
                # print("!@! hall_param", hall_param)
                # print("!@! n_e", n_e)
                # print("!@! m_e", m_e)
                # print("!@! nu_e", nu_e)
                # print("!@! self.E_perp", self.E_perp)
                # print("!@! p_e", p_e)
                # print("!@! grad_p_e_perp", np.gradient(p_e, self.inp.dx, 0)[0])
                
                # j_e_perp = (1/(1 + hall_param**2) * (q_e**2 * n_e)/(m_e * nu_e)) * (self.E_perp + grad_p_e_perp/(q_e*n_e)) # perpendicular electron current - marks eqs (2.29-2.31)
                j_e_perp = ((q_e * n_e * nu_e)/(omega_ce * self.fields.B[:, :, 1])) * (self.E_perp + grad_p_e_perp/(q_e*n_e)) # perpendicular electron current - marks eqs (2.29-2.31)
                j_e_perp = (1 / (1+hall_param**2) * (q_e**2 * n_e)/(m_e * nu_e)) * (self.E_perp + grad_p_e_perp/(q_e*n_e))
                j_e_perp_clipped = np.clip(j_e_perp, a_min=None, a_max=1e12)
                j_i_perp = self.simulation.get_species_current_density("i")[ng:-ng, ng:-ng] # TODO completely 1d, for now
                # TODO ^ ALSO SLICING NG 1) HERE AND 2) ON SIMLUATION GET_DENSITY IS KINDA SUS...
            
                j_e_perp = (mu_e / (1 + hall_param**2)) * (-q_e * n_e * self.E_perp + grad_p_e_perp) # textbook eq (7.5-11)
                # j_e_perp_clipped = np.clip(j_e_perp, a_min=None, a_max=1e6)
                # TODO: fix improper dx with ghost cells later, if its a problem
                self.E_perp = eta * (1 + hall_param**2) * j_e_perp # - (grad_p_e_perp/(q_e * n_e)) + eta_ei * j_i_perp # mikellides eqs (24a-24b)
                
                
                U[icomp, ng:-ng, ng:-ng] = j_e_perp
                # U[icomp, ng:-ng, ng:-ng] = 1
                
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
            