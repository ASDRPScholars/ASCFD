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

        self.euler = Euler(self.c) # initialize the euler solver

        # select the flux method based on the type
        if self.type == "rusanov":
            self.flux_method = self.rusanov
        elif self.type == "rusanov_vectorized":
            self.flux_method = self.rusanov_vectorized
        elif self.type == "hllc":
            self.flux_method = self.hllc
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

        fx, fy = self.euler.flux(U) # analytical flux

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

                # compute flux components for each variable
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
            density = a_grid.grid[self.c.RHOCOMP] # extract density
        else:
            raise RuntimeError("Density method needs to be implemented.")
        
        a = np.sqrt(self.c.gamma * a_grid.grid[self.c.PCOMP] / density) # compute sound speed

        U = a_grid.grid  # get primitive variables
        consU = self.euler.prim_to_cons(U) # convert to conserved variables

        fx, fy = self.euler.flux(U) # compute analytical fluxes

        numFluxX_plus = np.zeros_like(a_grid.grid)
        numFluxX_minus = np.zeros_like(a_grid.grid)
        numFluxY_plus = np.zeros_like(a_grid.grid)
        numFluxY_minus = np.zeros_like(a_grid.grid)

        # calculate the maximum wave speed
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

    def hllc(self, a_grid, a_Nx, a_Ny, a_Nghost):
        """
        Calculates the numerical flux using the HLLC approximate Riemann solver.
        """
        if self.c.system != "euler2D":
            raise NotImplementedError("HLLC flux is only implemented for euler2D system.")

        # Get primitive variables (rho, u, v, p)
        rho = a_grid[self.c.RHOCOMP]
        u = a_grid[self.c.UCOMP]
        v = a_grid[self.c.VCOMP]
        p = a_grid[self.c.PCOMP]
        gamma = self.c.gamma

        # Calculate sound speed
        a = np.sqrt(gamma * p / rho)
        
        # Get conservative variables and analytical fluxes
        consU = self.euler.prim_to_cons(a_grid)
        fx, fy = self.euler.flux(a_grid)
        
        # Initialize interface flux arrays 
        # These will store F_{i+1/2} and G_{j+1/2} respectively
        numFluxX = np.zeros_like(a_grid)
        numFluxY = np.zeros_like(a_grid)

        # --- X-direction HLLC Flux --- 
        for i in range(a_Nghost - 1, a_Nx + a_Nghost): # Loop over x-interfaces (i+1/2)
            for j in range(a_Nghost, a_Ny + a_Nghost): # Loop over y cells
                
                # Left state (i, j)
                rhoL = rho[i, j]
                uL = u[i, j]
                vL = v[i, j]
                pL = p[i, j]
                aL = a[i, j]
                HL = (consU[self.c.ECOMP, i, j] + pL) / rhoL # Total enthalpy H = (E+p)/rho
                FL = fx[:, i, j]
                UL = consU[:, i, j]
                
                # Right state (i+1, j)
                rhoR = rho[i+1, j]
                uR = u[i+1, j]
                vR = v[i+1, j]
                pR = p[i+1, j]
                aR = a[i+1, j]
                HR = (consU[self.c.ECOMP, i+1, j] + pR) / rhoR # Total enthalpy H = (E+p)/rho
                FR = fx[:, i+1, j]
                UR = consU[:, i+1, j]
                
                # Estimate wave speeds (Davis estimates)
                SL = min(uL - aL, uR - aR)
                SR = max(uL + aL, uR + aR)
                
                # Estimate contact wave speed S*
                # Avoid division by zero if rhoL(SL-uL) - rhoR(SR-uR) is close to zero
                denominator = rhoL * (SL - uL) - rhoR * (SR - uR)
                if np.abs(denominator) < 1e-10: 
                    Sstar = (uL + uR) / 2.0 # Or some other reasonable estimate 
                else:
                    Sstar = (pR - pL + rhoL * uL * (SL - uL) - rhoR * uR * (SR - uR)) / denominator

                # Calculate HLLC flux F_{i+1/2}
                if SL >= 0:
                    flux_hllc_x = FL
                elif SR <= 0:
                    flux_hllc_x = FR
                elif SL <= 0 and Sstar >= 0:
                    # U*_L state (intermediate state left of contact)
                    UstarL = np.zeros(self.c.NUMQ)
                    factorL = rhoL * (SL - uL) / (SL - Sstar)
                    UstarL[self.c.RHOCOMP] = factorL * 1.0
                    UstarL[self.c.MUCOMP] = factorL * Sstar
                    UstarL[self.c.MVCOMP] = factorL * vL
                    UstarL[self.c.ECOMP] = factorL * (UL[self.c.ECOMP]/rhoL + (Sstar - uL) * (Sstar + pL/(rhoL*(SL - uL))))
                    flux_hllc_x = FL + SL * (UstarL - UL)
                elif Sstar <= 0 and SR >= 0:
                    # U*_R state (intermediate state right of contact)
                    UstarR = np.zeros(self.c.NUMQ)
                    factorR = rhoR * (SR - uR) / (SR - Sstar)
                    UstarR[self.c.RHOCOMP] = factorR * 1.0
                    UstarR[self.c.MUCOMP] = factorR * Sstar
                    UstarR[self.c.MVCOMP] = factorR * vR
                    UstarR[self.c.ECOMP] = factorR * (UR[self.c.ECOMP]/rhoR + (Sstar - uR) * (Sstar + pR/(rhoR*(SR - uR))))
                    flux_hllc_x = FR + SR * (UstarR - UR)
                else: 
                    # Should not happen if SL < SR
                    raise ValueError(f"HLLC condition error: SL={SL}, Sstar={Sstar}, SR={SR} at i={i}, j={j}")
                
                numFluxX[:, i, j] = flux_hllc_x
                
        # --- Y-direction HLLC Flux --- 
        for i in range(a_Nghost, a_Nx + a_Nghost): # Loop over x cells
            for j in range(a_Nghost - 1, a_Ny + a_Nghost): # Loop over y-interfaces (j+1/2)
                
                # Bottom state (i, j) - analogous to Left
                rhoL = rho[i, j]
                uL = u[i, j]
                vL = v[i, j] # Normal velocity for y-flux is v
                pL = p[i, j]
                aL = a[i, j]
                HL = (consU[self.c.ECOMP, i, j] + pL) / rhoL
                GL = fy[:, i, j] # Use y-flux G
                UL = consU[:, i, j]
                
                # Top state (i, j+1) - analogous to Right
                rhoR = rho[i, j+1]
                uR = u[i, j+1]
                vR = v[i, j+1] # Normal velocity for y-flux is v
                pR = p[i, j+1]
                aR = a[i, j+1]
                HR = (consU[self.c.ECOMP, i, j+1] + pR) / rhoR
                GR = fy[:, i, j+1] # Use y-flux G
                UR = consU[:, i, j+1]

                # Estimate wave speeds (using v instead of u)
                SL = min(vL - aL, vR - aR)
                SR = max(vL + aL, vR + aR)
                
                # Estimate contact wave speed S*
                denominator = rhoL * (SL - vL) - rhoR * (SR - vR)
                if np.abs(denominator) < 1e-10:
                    Sstar = (vL + vR) / 2.0
                else:
                    Sstar = (pR - pL + rhoL * vL * (SL - vL) - rhoR * vR * (SR - vR)) / denominator

                # Calculate HLLC flux G_{j+1/2}
                if SL >= 0:
                    flux_hllc_y = GL
                elif SR <= 0:
                    flux_hllc_y = GR
                elif SL <= 0 and Sstar >= 0:
                    # U*_L state (intermediate state left of contact, y-direction)
                    UstarL = np.zeros(self.c.NUMQ)
                    factorL = rhoL * (SL - vL) / (SL - Sstar)
                    UstarL[self.c.RHOCOMP] = factorL * 1.0
                    UstarL[self.c.MUCOMP] = factorL * uL # Tangential velocity u
                    UstarL[self.c.MVCOMP] = factorL * Sstar # Normal velocity S*
                    UstarL[self.c.ECOMP] = factorL * (UL[self.c.ECOMP]/rhoL + (Sstar - vL) * (Sstar + pL/(rhoL*(SL - vL))))
                    flux_hllc_y = GL + SL * (UstarL - UL)
                elif Sstar <= 0 and SR >= 0:
                    # U*_R state (intermediate state right of contact, y-direction)
                    UstarR = np.zeros(self.c.NUMQ)
                    factorR = rhoR * (SR - vR) / (SR - Sstar)
                    UstarR[self.c.RHOCOMP] = factorR * 1.0
                    UstarR[self.c.MUCOMP] = factorR * uR # Tangential velocity u
                    UstarR[self.c.MVCOMP] = factorR * Sstar # Normal velocity S*
                    UstarR[self.c.ECOMP] = factorR * (UR[self.c.ECOMP]/rhoR + (Sstar - vR) * (Sstar + pR/(rhoR*(SR - vR))))
                    flux_hllc_y = GR + SR * (UstarR - UR)
                else:
                    raise ValueError(f"HLLC condition error: SL={SL}, Sstar={Sstar}, SR={SR} at i={i}, j={j}")

                numFluxY[:, i, j] = flux_hllc_y
        

        numFluxX_plus = np.zeros_like(a_grid)
        numFluxX_minus = np.zeros_like(a_grid)
        numFluxY_plus = np.zeros_like(a_grid)
        numFluxY_minus = np.zeros_like(a_grid)
        
        i_int_slice = slice(a_Nghost, a_Nx + a_Nghost) 
        j_int_slice = slice(a_Nghost, a_Ny + a_Nghost)
        
        numFluxX_plus[:, i_int_slice, j_int_slice] = numFluxX[:, a_Nghost:a_Nx + a_Nghost, j_int_slice]
        numFluxX_minus[:, i_int_slice, j_int_slice] = numFluxX[:, a_Nghost-1:a_Nx + a_Nghost-1, j_int_slice]
        numFluxY_plus[:, i_int_slice, j_int_slice] = numFluxY[:, i_int_slice, a_Nghost:a_Ny + a_Nghost]
        numFluxY_minus[:, i_int_slice, j_int_slice] = numFluxY[:, i_int_slice, a_Nghost-1:a_Ny + a_Nghost-1]
        

        return consU, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus



