import numpy as np
from ascfd.constants import Constants

class Euler:

    def __init__(self, a_constants : Constants):
        # store constants and system parameters
        self.c = a_constants

    
    def prim_to_cons(self, a_prim):
        cons = np.zeros_like(a_prim)

        if self.c.system == "euler2D":
            cons[self.c.RHOCOMP] = a_prim[self.c.RHOCOMP] # density stays the same
            cons[self.c.MUCOMP] = a_prim[self.c.RHOCOMP] * a_prim[self.c.UCOMP] # momentum components x & y
            cons[self.c.MVCOMP] = a_prim[self.c.RHOCOMP] * a_prim[self.c.VCOMP]
             # compute total energy
            E = (a_prim[self.c.PCOMP] / ((self.c.gamma - 1) * a_prim[self.c.RHOCOMP]) + 
                 0.5 * (a_prim[self.c.UCOMP]**2 + a_prim[self.c.VCOMP]**2))
            cons[self.c.ECOMP] = E * a_prim[self.c.RHOCOMP]

        else:
            # Raise error for unsupported systems
            raise RuntimeError(f"System not supported: {self.c.system}")    
        
        return cons



    def cons_to_prim(self, a_cons):
        prim = np.zeros_like(a_cons)

        if self.c.system == "euler2D":
            # copy density
            prim[self.c.RHOCOMP] = a_cons[self.c.RHOCOMP]
            # compute velocity components
            prim[self.c.UCOMP] = a_cons[self.c.MUCOMP] / a_cons[self.c.RHOCOMP]
            prim[self.c.VCOMP] = a_cons[self.c.MVCOMP] / a_cons[self.c.RHOCOMP]
            # compute pressure using kinetic energy
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
        
        flux_x = np.zeros_like(a_prim) # flux in x-direction
        flux_y = np.zeros_like(a_prim) # flux in y-direction

        if self.c.system == "euler2D":
            # extract primitive variables
            rho = a_prim[self.c.RHOCOMP] # density
            u = a_prim[self.c.UCOMP] # x-velocity
            v = a_prim[self.c.VCOMP] # y-velocity
            p = a_prim[self.c.PCOMP] # pressure
            
            # compute total energy 
            e = p / ((self.c.gamma - 1) * rho) # internal energy (from ideal gas law)
            E = rho * (e + 0.5 * (u**2 + v**2)) # total energy: internal + kinetic
            
            # flux in x-direction
            flux_x[self.c.RHOCOMP] = rho * u # mass flux
            flux_x[self.c.MUCOMP] = rho * u**2 + p # momentum flux in x 
            flux_x[self.c.MVCOMP] = rho * u * v # momentum flux in y 
            flux_x[self.c.ECOMP] = (E + p) * u # energy flux
            
            # flux in y-direction
            flux_y[self.c.RHOCOMP] = rho * v # mass flux
            flux_y[self.c.MUCOMP] = rho * u * v - g # momentum flux in x 
            flux_y[self.c.MVCOMP] = rho * v**2 + p # momentum flux in y 
            flux_y[self.c.ECOMP] = (E + p) * v # energy flux

        else:
            # raise error for unsupported systems
            raise RuntimeError(f"System not supported: {self.c.system}")

        return flux_x, flux_y


 
    def get_max_speed(self, a_grid):
        if a_grid.variables == "prim":
            # variables are primitive, max speed is max(u)
            return np.max(a_grid.grid[self.c.UCOMP])
        elif a_grid.variables == "cons":
            # variables are conserved, compute speed as momentum/density
            return np.max(a_grid.grid[self.c.MUCOMP] / a_grid.grid[self.c.RHOCOMP])
        else:
            print("unsupported")
            exit()

def g():
    # definiton of gravity
    g = rho * 9.81
