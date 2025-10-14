from ascfd.fluid.species import FluidSpecies
from ascfd.params import SpeciesParams
from ascfd.inputs import Inputs
from ascfd.fields.fields import Fields

import numpy as np

class QNFluidSpecies(FluidSpecies):
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields, simulation):
        super().__init__(params, a_inputs, fields, simulation)
        
        self.E_perp = 0 # TODO: IS THIS OK...
        
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
        
        p_e = n_e * k_B * T_e # TODO (ideal gas law?) - CHECK UHHHH - WHY IS THIS SO CLOSE TO ENERGY FORMULA
        
        l_D = np.sqrt((eps_0 * k_B * T_e)/(q_e**2 * n_e)) # debye length - marks eq (2.2)
        
        plasma_param = 4 * np.pi * n_e * l_D**3 
        couloumb_ln = np.log(plasma_param) # coulomb logarithm - ln(lambda_C) = ln(lambda) - robert fitzpatrick 2016 UT notes eq (3.124)
        # TODO: does the above line up with MIT collision notes eq (64) ish?
        
        Z_star = n_i / n_e # effective charge number only for singly charged ions - mikellides eq (21)
        
        nu_ei = (n_e * Z_star * q_e**4 * couloumb_ln) / (3 * (2*np.pi)**(3/2) * eps_0**2 * np.sqrt(m_e) * (k_B * T_e)**(3/2))
        nu_en = self.simulation.get_collision_frequency("en")
        nu_anom = 0 # TODO (much later)
        nu_e = nu_ei + nu_en + nu_anom
        
        omega_ce = q_e * np.abs(self.fields.B[:, :, 1]) / m_e # cyclotron frequency - marks eq (2.3)
        hall_param = omega_ce / nu_e # hall parameter - marks eq (2.5), alternatively combine the two to get q_e * B / m_e * nu_e per (pg 50 inline)
        
        eta = (m_e * nu_e) / (q_e**2 * n_e) # eta, total resistivity - mikellides eq (23)
        eta_ei = (m_e * nu_ei) / (q_e**2 * n_e) # eta, electron-ion resistivity - mikellides eq (23)
        
        _, right_flux, left_flux, top_flux, bottom_flux = self.flux.getFlux(U, self.inp.nx, self.inp.ny, self.inp.ng, nu_e=nu_e, hall_param=hall_param, omega_ce=omega_ce)
        
        for icomp in range(self.c.NUMQ):
            # FIX CONSISTENT GHOSTS IF BOUNDS BECOME A REAL PROBLEM:
            # TODO: ION GET DENSITY FUNCTIONS INCLUDE GHOSTS...
            # TODO: BECAUSE WHEN WE SET GRID, THE GHOST CELLS ACTUALLY GET INCLUDED IN GRID BOUNDS (so tehnically bounds are off by (2*ng)/n in every sim...)
            if icomp == self.c.NCOMP:
                U[icomp, ng:-ng, ng:-ng] = n_i
                print("!CONTINUITY UPDATE!")
                print(n_i)
                print(U[icomp, ng:-ng, ng:-ng])
                
            elif icomp == self.c.JXCOMP:
                # j_e = q_e * n_e * u_e
    
                # j_i = self.simulation.get_species_current_density("i")[ng:-ng, ng:-ng]
                # # j_i = 1e17 * self.inp.q * 20
                # q_e = self.params.charge
                # n_e = U[self.c.NCOMP, ng:-ng, ng:-ng]
                # # n_e = 1e17
                
                # U[icomp, ng:-ng, ng:-ng] = j_i / (q_e * n_e) # ok if no ions OR electrons then we get div by zero yeah
                
                
                
                j_e_perp = (1/(1 + hall_param**2) * (q_e**2 * n_e)/(m_e * nu_e)) * (self.E_perp + np.gradient(p_e, self.inp.dx, 0)[0]/(q_e*n_e)) # perpendicular electron current - marks eqs (2.29-2.31)
                
                j_i_perp = self.simulation.get_species_current_density("i")[ng:-ng, ng:-ng] # TODO completely 1d, for now
                # TODO ^ ALSO SLICING NG 1) HERE AND 2) ON SIMLUATION GET_DENSITY IS KINDA SUS...
            
                # TODO: fix improper dx with ghost cells later, if its a problem
                self.E_perp = eta * (1 + hall_param**2) * j_e_perp - (np.gradient(p_e, self.inp.dx, 0)[0]/(q_e * n_e)) + eta_ei * j_i_perp # mikellides eqs (24a-24b)
                U[icomp, ng:-ng, ng:-ng] = j_e_perp
            
            elif icomp == self.c.JYCOMP:
                (q_e**2 * n_e)/(m_e * nu_e) * (self.E_perp + np.gradient(p_e, 0, self.inp.dy)/(q_e*n_e)) # parallel electron current - marks eqs (2.29-2.31)
                
            elif icomp == self.c.JZCOMP:
                j_theta = hall_param * j_e_perp # azimuthal electron current - marks eqs (2.29-2.31)
                
            elif icomp == self.c.TCOMP:
                delta = (self.dt / self.inp.dx) * (right_flux[icomp, ng:-ng, ng:-ng] - left_flux[icomp, ng:-ng, ng:-ng]) + \
                        (self.dt / self.inp.dy) * (top_flux[icomp, ng:-ng, ng:-ng] - bottom_flux[icomp, ng:-ng, ng:-ng])
                    
                U[icomp, ng:-ng, ng:-ng] -= delta
                
                # ACTUAL TODO: add other source terms later
                
                # TODO: complete implement based on mikellides 2012 eq. (25)
                # do necessary conversion between E and n * k_B * T_e?
                # EDIT FLUX DEFINED IN EULER?
                
                # U[icomp, ng:-ng, ng:-ng] -= delta
                # add heat flux source terms