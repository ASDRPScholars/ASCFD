import numpy as np
from ascfd.fluid.constants import FluidConstants
from ascfd.inputs import Inputs

import sys

class FluidEuler:

    def __init__(self, a_constants : FluidConstants, a_inp: Inputs, a_params, simulation):
        # store constants and system parameters
        self.c = a_constants
        self.inp = a_inp
        self.params = a_params
        self.simluation = simulation

    
    def prim_to_cons(self, a_prim):
        cons = np.zeros_like(a_prim)
              
        if self.inp.e_system == "euler2d":
            cons[self.c.RHOCOMP] = a_prim[self.c.RHOCOMP] # density stays the same
            cons[self.c.MUCOMP] = a_prim[self.c.RHOCOMP] * a_prim[self.c.UCOMP] # momentum components x, y, z
            cons[self.c.MVCOMP] = a_prim[self.c.RHOCOMP] * a_prim[self.c.VCOMP]
            cons[self.c.MWCOMP] = a_prim[self.c.RHOCOMP] * a_prim[self.c.WCOMP] # z-momentum (2.5D)
             # compute total energy including z-velocity
            E = (a_prim[self.c.PCOMP] / ((self.c.gamma - 1) * a_prim[self.c.RHOCOMP]) + 
                 0.5 * (a_prim[self.c.UCOMP]**2 + a_prim[self.c.VCOMP]**2 + 0)) # TODO: a_prim[self.c.WCOMP]**2))
            cons[self.c.ECOMP] = E * a_prim[self.c.RHOCOMP]
            
        elif self.inp.e_system == "quasineutral":
            pass

        elif self.inp.e_system == "mhd2d":
            cons[self.c.RHOCOMP] = a_prim[self.c.RHOCOMP]
            cons[self.c.MUCOMP] = a_prim[self.c.RHOCOMP] * a_prim[self.c.UCOMP]
            cons[self.c.MVCOMP] = a_prim[self.c.RHOCOMP] * a_prim[self.c.VCOMP]
            cons[self.c.BXCOMP] = a_prim[self.c.BXCOMP]
            cons[self.c.BYCOMP] = a_prim[self.c.BYCOMP]
            E = (a_prim[self.c.PCOMP] / ((self.c.gamma - 1) * a_prim[self.c.RHOCOMP]) + 
                 0.5 * (a_prim[self.c.UCOMP]**2 + a_prim[self.c.VCOMP]**2)) + 0.5 * (a_prim[self.c.BXCOMP]**2 + a_prim[self.c.BYCOMP]**2)
            cons[self.c.ECOMP] = E * a_prim[self.c.RHOCOMP]

        else:
            # Raise error for unsupported systems
            raise RuntimeError(f"System not supported: {self.inp.e_system}")    
        
        return cons



    def cons_to_prim(self, a_cons):
        prim = np.zeros_like(a_cons)
            
        if self.inp.e_system == "euler2d":
            # copy density
            prim[self.c.RHOCOMP] = a_cons[self.c.RHOCOMP]
            
            # compute velocity components including z-velocity
            prim[self.c.UCOMP] = a_cons[self.c.MUCOMP] / a_cons[self.c.RHOCOMP]
            prim[self.c.VCOMP] = a_cons[self.c.MVCOMP] / a_cons[self.c.RHOCOMP]
            prim[self.c.WCOMP] = a_cons[self.c.MWCOMP] / a_cons[self.c.RHOCOMP] # z-velocity (2.5D)
            
            # No artificial velocity cap - let physics handle saturation
            
            # compute pressure using kinetic energy including z-component
            kinetic_energy = 0.5 * (prim[self.c.UCOMP]**2 + prim[self.c.VCOMP]**2 + 0) #TODO: prim[self.c.WCOMP]**2)
            
            # Check for negative internal energy (common cause of NaN)
            internal_energy = a_cons[self.c.ECOMP] - a_cons[self.c.RHOCOMP] * kinetic_energy
            if np.any(internal_energy <= 0):
                print(f"!WARNING! Negative internal energy detected!")
                print(f"Total energy: min={np.min(a_cons[self.c.ECOMP]):.3e}")
                print(f"Kinetic energy: max={np.max(a_cons[self.c.RHOCOMP] * kinetic_energy):.3e}")
                print(f"Max z-velocity: {np.max(np.abs(prim[self.c.WCOMP])):.3e}")
                print(f"Negative internal energy cells: {np.sum(internal_energy <= 0)}")
                
                # Floor to small positive value to prevent NaN
                internal_energy = np.maximum(internal_energy, 1e-12 * np.abs(a_cons[self.c.ECOMP]))
            
            prim[self.c.PCOMP] = (self.c.gamma - 1) * internal_energy
            
        elif self.inp.e_system == "quasineutral":
            pass

        elif self.inp.e_system == "mhd2d":
            prim[self.c.RHOCOMP] = a_cons[self.c.RHOCOMP]
            prim[self.c.UCOMP] = a_cons[self.c.MUCOMP] / a_cons[self.c.RHOCOMP]
            prim[self.c.VCOMP] = a_cons[self.c.MVCOMP] / a_cons[self.c.RHOCOMP]
            prim[self.c.BXCOMP] = a_cons[self.c.BXCOMP]
            prim[self.c.BYCOMP] = a_cons[self.c.BYCOMP]
            kinetic_energy = 0.5 * (prim[self.c.UCOMP]**2 + prim[self.c.VCOMP]**2)
            magnetic_energy = 0.5 * (prim[self.c.BXCOMP]**2 + prim[self.c.BYCOMP]**2)
            prim[self.c.PCOMP] = (self.c.gamma - 1) * (
                a_cons[self.c.ECOMP] - a_cons[self.c.RHOCOMP] * kinetic_energy - magnetic_energy
            )

        else:
            raise RuntimeError(f"System not supported: {self.inp.e_system}")
        
        return prim



    def flux(self, a_prim, **kwargs):
        """
        Compute the flux for the Euler equations
        """
        
        flux_x = np.zeros_like(a_prim) # flux in x-direction
        flux_y = np.zeros_like(a_prim) # flux in y-direction

        if self.inp.i_system == "euler2d" and self.params.type == "i":
            # extract primitive variables
            rho = a_prim[self.c.RHOCOMP] # density
            u = a_prim[self.c.UCOMP] # x-velocity
            v = a_prim[self.c.VCOMP] # y-velocity
            p = a_prim[self.c.PCOMP] # pressure
            
            # For 2.5D, w is not transported spatially but contributes to energy
            # TODO: IDK PLEASE WORK
            w = 0 #a_prim[self.c.WCOMP] if hasattr(self.c, 'WCOMP') else 0.0
            
            # compute total energy 
            e = p / ((self.c.gamma - 1) * rho) # internal energy (from ideal gas law)
            E = rho * (e + 0.5 * (u**2 + v**2 + w**2)) # total energy: internal + kinetic (including z)
            
            # flux in x-direction
            flux_x[self.c.RHOCOMP] = rho * u # mass flux
            flux_x[self.c.MUCOMP] = rho * u**2 + p # momentum flux in x 
            flux_x[self.c.MVCOMP] = rho * u * v # momentum flux in y 
            flux_x[self.c.MWCOMP] = 0 #TODO: rho * u * w # z-momentum flux (passively advected)
            flux_x[self.c.ECOMP] = (E + p) * u # energy flux
            
            # flux in y-direction
            flux_y[self.c.RHOCOMP] = rho * v # mass flux
            flux_y[self.c.MUCOMP] = rho * u * v - self.c.g # momentum flux in x 
            flux_y[self.c.MVCOMP] = rho * v**2 + p # momentum flux in y 
            flux_y[self.c.MWCOMP] = 0 #TODO: rho * v * w # z-momentum flux (passively advected)
            flux_y[self.c.ECOMP] = (E + p) * v # energy flux
            
        #TODO: DO PROPERLY
        elif self.inp.e_system == "quasineutral" and self.params.type == "e":
        
            nu_e = kwargs.get('nu_e', None)
            hall_param = kwargs.get('hall_param', None)
            mu_e = kwargs.get('mu_e', None)
            
            ng = self.inp.ng
            
            q_e = self.inp.q
            m_e = self.inp.m_e

            n_e = a_prim[self.c.NCOMP, ng:-ng, ng:-ng]
            
            try:
                v_e = a_prim[self.c.UCOMP, ng:-ng, ng:-ng] # x-velocity
            except:
                v_e = a_prim[self.c.JXCOMP, ng:-ng, ng:-ng] / n_e * q_e
                
            T_e = a_prim[self.c.TCOMP, ng:-ng, ng:-ng]
            k_B = self.inp.k_B
            
            eps_e = 3/2 * k_B * T_e + (1/2 * m_e * v_e**2)
            
            
            p_e = n_e * k_B * T_e # NOTE: CONFIRMED ON TEXTBOOK PG 48 - and guillaume - all are ideal gas
            # u_e = j_e / (q_e * n_e)
            print("mu_e", np.shape(mu_e))
            print("eps_e", np.shape(eps_e))
            print("nu_e", np.shape(nu_e))
            energy_flux = (5/3 * n_e * v_e * eps_e) - (10/9 * mu_e * n_e * eps_e * np.gradient(eps_e, self.inp.dx, axis=0))
            
            temp_flux = 2/(3*k_B*n_e) * energy_flux
            temp_flux = np.pad(temp_flux, ((ng, ng), (ng, ng)), mode="edge")
            
            flux_x[self.c.TCOMP] = temp_flux # thermal energy density (3/2 * n_e * k_B * T_e) equation - rearrange mikellides eq (25), or marks eq (4.8)
            # flux_x[self.c.TCOMP] = 0
            # TODO: flux_y[self.c.ECOMP] = ...
            

        elif self.inp.e_system == "mhd2d":
            rho = a_prim[self.c.RHOCOMP]
            u = a_prim[self.c.UCOMP]
            v = a_prim[self.c.VCOMP]
            p = a_prim[self.c.PCOMP]
            b_x = a_prim[self.c.BXCOMP]
            b_y = a_prim[self.c.BYCOMP]
            
            # Compute total energy 
            e = p / ((self.c.gamma - 1) * rho)
            E = rho * (e + 0.5 * (u**2 + v**2))
            
            # Flux in x-direction
            flux_x[self.c.RHOCOMP] = rho * u
            flux_x[self.c.MUCOMP] = rho * u**2 + p + 0.5 * (b_x**2 + b_y**2) - b_x**2
            flux_x[self.c.MVCOMP] = rho * u * v - b_x * b_y
            flux_x[self.c.ECOMP] = (E + p + 0.5 * (b_x**2 + b_y**2)) * u - b_x * (u * b_x + v * b_y)
            flux_x[self.c.BXCOMP] = 0
            flux_x[self.c.BYCOMP] = u * b_y - v * b_x

            # Flux in y-direction
            flux_y[self.c.RHOCOMP] = rho * v
            flux_y[self.c.MUCOMP] = rho * u * v - b_x * b_y
            flux_y[self.c.MVCOMP] = rho * v**2 + p + 0.5 * (b_x**2 + b_y**2) - b_y**2
            flux_y[self.c.ECOMP] = (E + p + 0.5 * (b_x**2 + b_y**2)) * v - b_y * (u * b_x + v * b_y)
            flux_y[self.c.BXCOMP] = v * b_x - u * b_y
            flux_y[self.c.BYCOMP] = 0

        else:
            # raise error for unsupported systems
            raise RuntimeError(f"System not supported: {self.inp.e_system}")

        return flux_x, flux_y


 
    def get_max_speed(self, a_grid):

        if self.inp.e_system == "euler2d":
            if a_grid.variables == "prim":
                return np.max(a_grid.grid[self.c.UCOMP])
            elif a_grid.variables == "cons":
                return np.max(a_grid.grid[self.c.MUCOMP] / a_grid.grid[self.c.RHOCOMP])
            else:
                print("unsupported")
                exit()
        elif self.inp.e_system == "mhd2d":
            if a_grid.variables == "prim":
                cs = np.sqrt(self.c.gamma * a_grid.grid[self.c.PCOMP] / a_grid.grid[self.c.RHOCOMP])
                mag_pressure = (a_grid.grid[self.c.BXCOMP]**2 + a_grid.grid[self.c.BYCOMP]**2) / a_grid.grid[self.c.RHOCOMP]
                cf = np.sqrt(cs**2 + mag_pressure)  
                speed = np.sqrt(a_grid.grid[self.c.UCOMP]**2 + a_grid.grid[self.c.VCOMP]**2) + cf
                return np.max(speed)
            elif a_grid.variables == "cons":
                u = a_grid.grid[self.c.MUCOMP] / a_grid.grid[self.c.RHOCOMP]
                v = a_grid.grid[self.c.MVCOMP] / a_grid.grid[self.c.RHOCOMP]
                kinetic_energy = 0.5 * (u**2 + v**2)
                mag_pressure = 0.5 * (a_grid.grid[self.c.BXCOMP]**2 + a_grid.grid[self.c.BYCOMP]**2)
                internal_energy = a_grid.grid[self.c.ECOMP] / a_grid.grid[self.c.RHOCOMP] - kinetic_energy - mag_pressure / a_grid.grid[self.c.RHOCOMP]
                p = (self.c.gamma - 1) * a_grid.grid[self.c.RHOCOMP] * internal_energy
                cs = np.sqrt(self.c.gamma * p / a_grid.grid[self.c.RHOCOMP])
                mag_pressure = (a_grid.grid[self.c.BXCOMP]**2 + a_grid.grid[self.c.BYCOMP]**2) / a_grid.grid[self.c.RHOCOMP]
                cf = np.sqrt(cs**2 + mag_pressure)  
                speed = np.sqrt(u**2 + v**2) + cf
                return np.max(speed)
            else:
                print("unsupported")
                exit()
