from ascfd.euler import Euler
# from ascfd.reconstruct import weno5_reconstruction
from ascfd.constants import *
from ascfd.grid import Grid2D
from ascfd.constants import Constants
from ascfd.weno import weno5_plus, weno5_minus
import numpy as np
import sys

class Flux:

    def __init__(self, a_constants: Constants, a_type: str):

        self.type = a_type
        self.c = a_constants

        self.euler = Euler(self.c)


        if self.type == "rusanov":
            self.flux_method = self.rusanov
        elif self.type == "lax_friedrichs":
            self.flux_method = self.lax_friedrichs
        elif self.type == "weno_js":
            self.flux_method = self.weno_js
        else:
            raise RuntimeError(f"Flux method not supported: {self.type}")


    def getFlux(self, a_grid):
        return self.flux_method(a_grid)


    def rusanov(self, a_grid):
        a_grid.assert_variable_type("prim")

        #get density 
        if self.c.system == "euler2D":
            density = a_grid.grid[self.c.RHOCOMP]
        else:
            raise RuntimeError("Density method needs to be implemented.")
            
        a = np.sqrt(self.c.gamma * a_grid.grid[self.c.PCOMP] / density)


        U = a_grid.grid
        consU = self.euler.prim_to_cons(U)

        fx, fy = self.euler.flux(U) #analytical flux

        numFluxX_plus = np.zeros_like(a_grid.grid)
        numFluxX_minus = np.zeros_like(a_grid.grid)
        numFluxY_plus = np.zeros_like(a_grid.grid)
        numFluxY_minus = np.zeros_like(a_grid.grid)

        for i in range(a_grid.Nghost - 1, a_grid.Nx + a_grid.Nghost):
            for j in range(a_grid.Nghost - 1, a_grid.Ny + a_grid.Nghost):
                sMaxX = max(
                    np.abs(a_grid.grid[self.c.UCOMP, i, j]) + a[i, j],
                    np.abs(a_grid.grid[self.c.UCOMP, i+1, j]) + a[i+1, j]
                )
                sMaxY = max(
                    np.abs(a_grid.grid[self.c.VCOMP, i, j]) + a[i, j],
                    np.abs(a_grid.grid[self.c.VCOMP, i, j+1]) + a[i, j+1]
                )



                for icomp in range(self.c.NUMQ):
                    numFluxX_plus[icomp, i, j] = 0.5 * (fx[icomp, i+1, j] + fx[icomp, i, j]) - 0.5 * sMaxX * (consU[icomp, i+1, j] - consU[icomp, i, j])
                    numFluxX_minus[icomp, i, j] = 0.5 * (fx[icomp, i, j] + fx[icomp, i-1, j]) - 0.5 * sMaxX * (consU[icomp, i, j] - consU[icomp, i-1, j])
                    numFluxY_plus[icomp, i, j] = 0.5 * (fy[icomp, i, j+1] + fy[icomp, i, j]) - 0.5 * sMaxY * (consU[icomp, i, j+1] - consU[icomp, i, j])
                    numFluxY_minus[icomp, i, j] = 0.5 * (fy[icomp, i, j] + fy[icomp, i, j-1]) - 0.5 * sMaxY * (consU[icomp, i, j] - consU[icomp, i, j-1])

        return consU, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus
    
    
    def lax_friedrichs(self, a_grid):
        a_grid.assert_variable_type("prim")

        if self.c.system == "euler2D":
            density = a_grid.grid[self.c.RHOCOMP]
        else:
            raise RuntimeError("Density method needs to be implemented.")
        
        a = np.sqrt(self.c.gamma * a_grid.grid[self.c.PCOMP] / density)

        U = a_grid.grid
        consU = self.euler.prim_to_cons(U)

        fx, fy = self.euler.flux(U)  # analytical flux

        numFluxX_plus = np.zeros_like(a_grid.grid)
        numFluxX_minus = np.zeros_like(a_grid.grid)
        numFluxY_plus = np.zeros_like(a_grid.grid)
        numFluxY_minus = np.zeros_like(a_grid.grid)

        # Calculate the maximum wave speed
        max_speed_x = np.max(np.abs(U[self.c.UCOMP]) + a)
        max_speed_y = np.max(np.abs(U[self.c.VCOMP]) + a)

        for i in range(a_grid.Nghost - 1, a_grid.Nx + a_grid.Nghost):
            for j in range(a_grid.Nghost - 1, a_grid.Ny + a_grid.Nghost):
                for icomp in range(self.c.NUMQ):
                    # X-direction fluxes
                    numFluxX_plus[icomp, i, j] = 0.5 * (fx[icomp, i+1, j] + fx[icomp, i, j]) - 0.5 * max_speed_x * (consU[icomp, i+1, j] - consU[icomp, i, j])
                    numFluxX_minus[icomp, i, j] = 0.5 * (fx[icomp, i, j] + fx[icomp, i-1, j]) - 0.5 * max_speed_x * (consU[icomp, i, j] - consU[icomp, i-1, j])
                    
                    # Y-direction fluxes
                    numFluxY_plus[icomp, i, j] = 0.5 * (fy[icomp, i, j+1] + fy[icomp, i, j]) - 0.5 * max_speed_y * (consU[icomp, i, j+1] - consU[icomp, i, j])
                    numFluxY_minus[icomp, i, j] = 0.5 * (fy[icomp, i, j] + fy[icomp, i, j-1]) - 0.5 * max_speed_y * (consU[icomp, i, j] - consU[icomp, i, j-1])

        return consU, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus
    




    def weno_js(self, a_grid):
        a_grid.assert_variable_type("prim")

        # Calculate sound speed
        a = np.sqrt(self.c.gamma * a_grid.grid[self.c.PCOMP] / a_grid.grid[self.c.RHOCOMP])
        
        # Get max speeds in both directions
        max_speed_x = np.max(np.abs(a_grid.grid[self.c.UCOMP]) + a)
        max_speed_y = np.max(np.abs(a_grid.grid[self.c.VCOMP]) + a)

        # Reconstruct primitive variables at cell faces
        primPR_x = weno5_plus(a_grid.grid, a_grid)
        primPL_x = weno5_minus(a_grid.grid, a_grid)
        primPR_y = weno5_plus(a_grid.grid, a_grid, axis=1)  
        primPL_y = weno5_minus(a_grid.grid, a_grid, axis=1)

        # Convert to conservative variables
        consPR_x = self.euler.prim_to_cons(primPR_x)
        consPL_x = self.euler.prim_to_cons(primPL_x)
        consPR_y = self.euler.prim_to_cons(primPR_y)
        consPL_y = self.euler.prim_to_cons(primPL_y)

        # Compute analytical fluxes
        fx_R, fy_R = self.euler.flux(primPR_x)
        fx_L, fy_L = self.euler.flux(primPL_x)

        # Initialize numerical fluxes
        numFluxX_plus = np.zeros_like(a_grid.grid)
        numFluxX_minus = np.zeros_like(a_grid.grid)
        numFluxY_plus = np.zeros_like(a_grid.grid)
        numFluxY_minus = np.zeros_like(a_grid.grid)

        # Compute fluxes for interior points
        for i in range(a_grid.Nghost - 1, a_grid.Nx + a_grid.Nghost):
            for j in range(a_grid.Nghost - 1, a_grid.Ny + a_grid.Nghost):
                for icomp in range(self.c.NUMQ):
                    # X-direction fluxes
                    numFluxX_plus[icomp, i, j] = 0.5 * (fx_R[icomp, i+1, j] + fx_R[icomp, i, j]) - \
                        0.5 * max_speed_x * (consPR_x[icomp, i+1, j] - consPR_x[icomp, i, j])
                    numFluxX_minus[icomp, i, j] = 0.5 * (fx_L[icomp, i, j] + fx_L[icomp, i-1, j]) - \
                        0.5 * max_speed_x * (consPL_x[icomp, i, j] - consPL_x[icomp, i-1, j])
                    
                    # Y-direction fluxes
                    numFluxY_plus[icomp, i, j] = 0.5 * (fy_R[icomp, i, j+1] + fy_R[icomp, i, j]) - \
                        0.5 * max_speed_y * (consPR_y[icomp, i, j+1] - consPR_y[icomp, i, j])
                    numFluxY_minus[icomp, i, j] = 0.5 * (fy_L[icomp, i, j] + fy_L[icomp, i, j-1]) - \
                        0.5 * max_speed_y * (consPL_y[icomp, i, j] - consPL_y[icomp, i, j-1])

        return consPR_x, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus
