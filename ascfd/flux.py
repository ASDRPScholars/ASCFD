from ascfd.euler import Euler
# from ascfd.reconstruct import weno5_reconstruction
from ascfd.constants import *
from ascfd.grid import Grid2D
from ascfd.constants import Constants

import numpy as np
import sys


class Flux:

    def __init__(self, a_constants: Constants, a_type: str):

        self.type = a_type
        self.c = a_constants

        self.euler = Euler(self.c)


        if self.type == "rusanov":
            self.flux_method = self.rusanov
        elif self.type == "rusanov_vectorized":
            self.flux_method = self.rusanov_vectorized
        else:
            raise RuntimeError(f"Flux method not supported: {self.type}")


    def getFlux(self, a_grid, a_Nx, a_Ny, a_Nghost):
        return self.flux_method(a_grid, a_Nx, a_Ny, a_Nghost)


    def rusanov(self, a_grid, a_Nx, a_Ny, a_Nghost):

        #get density 
        if self.c.system == "euler2D":
            density = a_grid[self.c.RHOCOMP]
        else:
            raise RuntimeError("Density method needs to be implemented.")
            
        a = np.sqrt(self.c.gamma * a_grid[self.c.PCOMP] / density)


        U = a_grid
        consU = self.euler.prim_to_cons(U)

        fx, fy = self.euler.flux(U) #analytical flux

        numFluxX_plus = np.zeros_like(a_grid)
        numFluxX_minus = np.zeros_like(a_grid)
        numFluxY_plus = np.zeros_like(a_grid)
        numFluxY_minus = np.zeros_like(a_grid)

        for i in range(a_Nghost - 1, a_Nx + a_Nghost):
            for j in range(a_Nghost - 1, a_Ny + a_Nghost):
                sMaxX = max(
                    np.abs(a_grid[self.c.UCOMP, i, j]) + a[i, j],
                    np.abs(a_grid[self.c.UCOMP, i+1, j]) + a[i+1, j]
                )
                sMaxY = max(
                    np.abs(a_grid[self.c.VCOMP, i, j]) + a[i, j],
                    np.abs(a_grid[self.c.VCOMP, i, j+1]) + a[i, j+1]
                )



                for icomp in range(self.c.NUMQ):
                    numFluxX_plus[icomp, i, j] = 0.5 * (fx[icomp, i+1, j] + fx[icomp, i, j]) - 0.5 * sMaxX * (consU[icomp, i+1, j] - consU[icomp, i, j])
                    numFluxX_minus[icomp, i, j] = 0.5 * (fx[icomp, i, j] + fx[icomp, i-1, j]) - 0.5 * sMaxX * (consU[icomp, i, j] - consU[icomp, i-1, j])
                    numFluxY_plus[icomp, i, j] = 0.5 * (fy[icomp, i, j+1] + fy[icomp, i, j]) - 0.5 * sMaxY * (consU[icomp, i, j+1] - consU[icomp, i, j])
                    numFluxY_minus[icomp, i, j] = 0.5 * (fy[icomp, i, j] + fy[icomp, i, j-1]) - 0.5 * sMaxY * (consU[icomp, i, j] - consU[icomp, i, j-1])

        return consU, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus
    
    def rusanov_vectorized(self, a_grid, a_Nx, a_Ny, a_Nghost):
        #get density 
        if self.c.system == "euler2D":
            density = a_grid[self.c.RHOCOMP]
        else:
            raise RuntimeError("Density method needs to be implemented.")
            
        a = np.sqrt(self.c.gamma * a_grid[self.c.PCOMP] / density)

        U = a_grid
        consU = self.euler.prim_to_cons(U)

        fx, fy = self.euler.flux(U) #analytical flux

        # Define the interior domain indices
        i_start = a_Nghost - 1
        i_end = a_Nx + a_Nghost
        j_start = a_Nghost - 1
        j_end = a_Ny + a_Nghost
        
        # Calculate wave speeds
        u_abs = np.abs(a_grid[self.c.UCOMP, i_start:i_end, j_start:j_end])
        u_abs_right = np.abs(a_grid[self.c.UCOMP, i_start+1:i_end+1, j_start:j_end])
        
        sound_speed = a[i_start:i_end, j_start:j_end]
        sound_speed_right = a[i_start+1:i_end+1, j_start:j_end]
        
        sMaxX = np.maximum(u_abs + sound_speed, u_abs_right + sound_speed_right)
        
        v_abs = np.abs(a_grid[self.c.VCOMP, i_start:i_end, j_start:j_end])
        v_abs_up = np.abs(a_grid[self.c.VCOMP, i_start:i_end, j_start+1:j_end+1])
        
        sound_speed_up = a[i_start:i_end, j_start+1:j_end+1]
        
        sMaxY = np.maximum(v_abs + sound_speed, v_abs_up + sound_speed_up)
        
        # Initialize flux arrays
        numFluxX_plus = np.zeros_like(a_grid)
        numFluxX_minus = np.zeros_like(a_grid)
        numFluxY_plus = np.zeros_like(a_grid)
        numFluxY_minus = np.zeros_like(a_grid)
        
        # Calculate fluxes for all components at once
        for icomp in range(self.c.NUMQ):
            # X-direction fluxes
            numFluxX_plus[icomp, i_start:i_end, j_start:j_end] = 0.5 * (
                fx[icomp, i_start+1:i_end+1, j_start:j_end] + fx[icomp, i_start:i_end, j_start:j_end]
            ) - 0.5 * sMaxX * (
                consU[icomp, i_start+1:i_end+1, j_start:j_end] - consU[icomp, i_start:i_end, j_start:j_end]
            )
            
            numFluxX_minus[icomp, i_start:i_end, j_start:j_end] = 0.5 * (
                fx[icomp, i_start:i_end, j_start:j_end] + fx[icomp, i_start-1:i_end-1, j_start:j_end]
            ) - 0.5 * sMaxX * (
                consU[icomp, i_start:i_end, j_start:j_end] - consU[icomp, i_start-1:i_end-1, j_start:j_end]
            )
            
            # Y-direction fluxes
            numFluxY_plus[icomp, i_start:i_end, j_start:j_end] = 0.5 * (
                fy[icomp, i_start:i_end, j_start+1:j_end+1] + fy[icomp, i_start:i_end, j_start:j_end]
            ) - 0.5 * sMaxY * (
                consU[icomp, i_start:i_end, j_start+1:j_end+1] - consU[icomp, i_start:i_end, j_start:j_end]
            )
            
            numFluxY_minus[icomp, i_start:i_end, j_start:j_end] = 0.5 * (
                fy[icomp, i_start:i_end, j_start:j_end] + fy[icomp, i_start:i_end, j_start-1:j_end-1]
            ) - 0.5 * sMaxY * (
                consU[icomp, i_start:i_end, j_start:j_end] - consU[icomp, i_start:i_end, j_start-1:j_end-1]
            )

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



