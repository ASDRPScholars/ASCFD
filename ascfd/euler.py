import numpy as np
from ascfd.constants import Constants

class Euler:

    def __init__(self, a_constants : Constants):
    
        self.c = a_constants

    
    def prim_to_cons(self, a_prim):
        cons = np.zeros_like(a_prim)

        if self.c.system == "euler2D":
            cons[self.c.RHOCOMP] = a_prim[self.c.RHOCOMP]
            cons[self.c.MUCOMP] = a_prim[self.c.RHOCOMP] * a_prim[self.c.UCOMP]
            cons[self.c.MVCOMP] = a_prim[self.c.RHOCOMP] * a_prim[self.c.VCOMP]
            E = (a_prim[self.c.PCOMP] / ((self.c.gamma - 1) * a_prim[self.c.RHOCOMP]) + 
                 0.5 * (a_prim[self.c.UCOMP]**2 + a_prim[self.c.VCOMP]**2))
            cons[self.c.ECOMP] = E * a_prim[self.c.RHOCOMP]

        else:
            raise RuntimeError(f"System not supported: {self.c.system}")    
        
        return cons



    def cons_to_prim(self, a_cons):
        prim = np.zeros_like(a_cons)

        if self.c.system == "euler2D":
            prim[self.c.RHOCOMP] = a_cons[self.c.RHOCOMP]
            prim[self.c.UCOMP] = a_cons[self.c.MUCOMP] / a_cons[self.c.RHOCOMP]
            prim[self.c.VCOMP] = a_cons[self.c.MVCOMP] / a_cons[self.c.RHOCOMP]
            kinetic_energy = 0.5 * (prim[self.c.UCOMP]**2 + prim[self.c.VCOMP]**2)
            prim[self.c.PCOMP] = (self.c.gamma - 1) * (
                a_cons[self.c.ECOMP] - a_cons[self.c.RHOCOMP] * kinetic_energy
            )

        else:
            raise RuntimeError(f"System not supported: {self.c.system}")
        
        return prim



    def flux(self, a_prim):
        """
        Compute the flux for the Euler equations
        """
        
        flux_x = np.zeros_like(a_prim)
        flux_y = np.zeros_like(a_prim)

        if self.c.system == "euler2D":
            rho = a_prim[self.c.RHOCOMP]
            u = a_prim[self.c.UCOMP]
            v = a_prim[self.c.VCOMP]
            p = a_prim[self.c.PCOMP]
            
            # Compute total energy 
            e = p / ((self.c.gamma - 1) * rho)
            E = rho * (e + 0.5 * (u**2 + v**2))
            
            # Flux in x-direction
            flux_x[self.c.RHOCOMP] = rho * u
            flux_x[self.c.MUCOMP] = rho * u**2 + p
            flux_x[self.c.MVCOMP] = rho * u * v
            flux_x[self.c.ECOMP] = (E + p) * u
            
            # Flux in y-direction
            flux_y[self.c.RHOCOMP] = rho * v
            flux_y[self.c.MUCOMP] = rho * u * v
            flux_y[self.c.MVCOMP] = rho * v**2 + p
            flux_y[self.c.ECOMP] = (E + p) * v

        else:
            raise RuntimeError(f"System not supported: {self.c.system}")

        return flux_x, flux_y


 
    def get_max_speed(self, a_grid):
        if a_grid.variables == "prim":
            return np.max(a_grid.grid[self.c.UCOMP])
        elif a_grid.variables == "cons":
            return np.max(a_grid.grid[self.c.MUCOMP] / a_grid.grid[self.c.RHOCOMP])
        else:
            print("unsupported")
            exit()

