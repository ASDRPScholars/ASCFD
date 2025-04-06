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
        elif self.type == "hlld":
            self.flux_method = self.actual_hlld
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



    def flux(self, U, pT, u, v, By, Bx_R):
        rho, _, _, _, p, _ = U # TODO: is this the right order?
        
        F = np.zeros(6)
        F[0] = rho * u
        F[1] = rho * u**2 + pT - Bx_R**2
        F[2] = rho * u * v - Bx_R * By
        F[3] = rho * u - Bx_R # * rho * u * w - Bx_R * Bz
        F[4] = (p + 0.5 * (Bx_R**2 + By**2)) * u - Bx_R * (u * Bx_R + v * By)
        F[5] = By * u - Bx_R * v
        
        return F
    
    @staticmethod
    def compute_wave_speeds(u_L, u_R, pT_L, pT_R, rho_L, rho_R, cf_L, cf_R, Bx):
        S_L = min(u_L - cf_L, u_R - cf_R)
        S_R = max(u_L + cf_L, u_R + cf_R)
        
        S_M = ((S_R * rho_R * u_R - S_L * rho_L * u_L + pT_L - pT_R) /
            (S_R * rho_R - S_L * rho_L))

        S_star_L = S_M - np.linalg.norm(Bx) / np.sqrt(rho_L * (S_L - u_L) / (S_L - S_M))
        S_star_R = S_M + np.linalg.norm(Bx) / np.sqrt(rho_R * (S_R - u_R) / (S_R - S_M))
        
        return S_L, S_R, S_M, S_star_L, S_star_R
    
    @staticmethod
    def compute_star_values(rho, u, v, Bx, By, S, S_M):
        """
        Compute the star values of each primitive variable.
        """
        denominator = (rho * (S - u) * (S - S_M) - Bx**2)
        
        rho_star = rho * (S - u) / (S - S_M)
        u_star = u
        v_star = v - (Bx * By * (S_M - u)) / denominator
        
        # TODO: is this true???
        Bx_star = Bx 
        
        By_star = By * ((rho * (S - u)**2 - Bx**2) / denominator)

        # e_star = ((S - u) * e - p_T * u + p_T_star * S_M +
        #             Bx * (np.dot(v, B) - np.dot(v_star, B_star))) / (S - S_M)

        return rho_star, u_star, v_star, Bx_star, By_star
    
    @staticmethod
    def compute_doublestar_values(rho_star_L, rho_star_R, u_star, v_star_L, v_star_R, Bx, By_star_R, By_star_L):
        """
        Compute the double star values of each primitive variable.
        """
        
        sqrt_rho_L = np.sqrt(rho_star_L)
        sqrt_rho_R = np.sqrt(rho_star_R)
        rho_sum = sqrt_rho_L + sqrt_rho_R
        sign_Bx = np.sign(Bx)
        
        rho_doublestar_L = rho_star_L
        rho_doublestar_R = rho_star_R
        u_doublestar_L = u_doublestar_R = u_star
        v_doublestar_L = v_doublestar_R = (sqrt_rho_L * v_star_L + sqrt_rho_R * v_star_R + (By_star_R - By_star_L) * sign_Bx) / rho_sum
        
        #TODO: correct?
        Bx_doublestar_L = Bx_doublestar_R = Bx
        
        By_doublestar_L = By_doublestar_R = (sqrt_rho_L * By_star_R + sqrt_rho_R * By_star_L + sqrt_rho_L * sqrt_rho_R * (v_star_R - v_star_L) * sign_Bx) / rho_sum

        return rho_doublestar_L, rho_doublestar_R, u_doublestar_L, u_doublestar_R, v_doublestar_L, v_doublestar_R, Bx_doublestar_L, Bx_doublestar_R, By_doublestar_L, By_doublestar_R
    
    
    def hlld1D(self, a_grid, direction="x"):
        a_grid.assert_variable_type("prim")

        #get density 
        if self.c.system == "euler2D":
            print("HLLD on Euler is not supported!")
            sys.exit()
        elif self.c.system == "mhd2d":
            density = a_grid.grid[self.c.RHOCOMP]
        else:
            raise RuntimeError("Density method needs to be implemented.")
            
        a = np.sqrt(self.c.gamma * a_grid.grid[self.c.PCOMP] / density)
        
        U = a_grid.grid
        print("shape of U:", np.shape(U))
        print(U)
        consU = self.euler.prim_to_cons(U)
        
        F_hlld_array = np.zeros_like(U)  # Allocate flux array

        for i in range(a_grid.Nghost - 1, a_grid.Nx + a_grid.Nghost):
            for j in range(a_grid.Nghost - 1, a_grid.Ny + a_grid.Nghost):
                
                if direction == "y":
                    U_L = U[:, i, j]
                    U_R = U[:, i, j+1]
                else:
                    U_L = U[:, i, j]
                    U_R = U[:, i+1, j]
                
                rho_L, u_L, v_L, p_L, Bx_L, By_L = U_L
                rho_R, u_R, v_R, p_R, Bx_R, By_R = U_R
                
                print(U_L, U_R)
                
                pT_L = p_L + 0.5 * (Bx_R**2 + By_L**2)
                pT_R = p_R + 0.5 * (Bx_R**2 + By_R**2)
                
                cf_L = np.sqrt((pT_L + Bx_R**2) / rho_L)
                cf_R = np.sqrt((pT_R + Bx_R**2) / rho_R)
                
                S_L, S_R, S_M, S_star_L, S_star_R = self.compute(u_L, u_R, pT_L, pT_R, rho_L, rho_R, cf_L, cf_R, Bx_R)
                
                rho_star_L, u_star_L, v_star_L, Bx_star_L, By_star_L = self.compute_star_values(rho_L, u_L, v_L, Bx_L, By_L, S_L, S_M)
                rho_star_R, u_star_R, v_star_R, Bx_star_R, By_star_R = self.compute_star_values(rho_R, u_R, v_R, Bx_R, By_R, S_R, S_M)
                p_T_star = (rho_star_L * S_star_L * (u_L - u_R) + pT_L - pT_R) / (rho_star_L * (S_star_L - S_M))
                
                rho_doublestar_L, rho_doublestar_R, u_doublestar_L, u_doublestar_R, v_doublestar_L, v_doublestar_R, Bx_doublestar_L, Bx_doublestar_R, By_doublestar_L, By_doublestar_R = self.compute_doublestar_values(rho_star_L, rho_star_R, u_star_L, v_star_L, v_star_R, Bx_star_R, By_star_R, By_star_L)
                
                F_L = self.flux(U_L, pT_L, u_L, v_L, By_L, Bx_R)
                F_R = self.flux(U_R, pT_R, u_R, v_R, By_R, Bx_R)
                
                U_star_L = np.array([rho_star_L, u_star_L, v_star_L, Bx_star_L, By_star_L])
                U_star_R = np.array([rho_star_R, u_star_R, v_star_R, Bx_star_R, By_star_R])
                
                U_doublestar_R = np.array([rho_doublestar_L, u_doublestar_L, v_doublestar_L, Bx_doublestar_L, By_doublestar_L])
                U_doublestar_L = np.array([rho_doublestar_R, u_doublestar_R, v_doublestar_R, Bx_doublestar_R, By_doublestar_R])
                
                if S_L > 0:
                    F_hlld = F_L
                elif S_L <= 0 <= S_star_L:
                    debug_list = [F_L, S_L, U_star_L, U_L, S_star_L, U_doublestar_L, U_star_L]
                    print (f"shape of {debug_list}:")
                    for item in debug_list:
                        print(np.shape(item))
                        
                    F_hlld = F_L + S_L*(U_star_L-U_L) + S_star_L*(U_doublestar_L - U_star_L)
                    
                elif S_star_L <= 0 <= S_M:
                    F_hlld = F_L + S_L*(U_star_L-U_L) + S_star_R*(U_doublestar_R - U_star_R)
                elif S_M <= 0 <= S_star_R:
                    F_hlld = F_R + S_R*(U_star_R - U_R) + S_star_R*(U_doublestar_R - U_star_R)
                elif S_star_R <= 0 <= S_R:
                    F_hlld = F_R + S_R*(U_star_R - U_R)
                elif S_R <= 0:
                    F_hlld = F_R
                else:
                    F_hlld = np.zeros(6)
                    raise ValueError
                F_hlld_array[:, i, j] = F_hlld  # Store in flux array

        return consU, F_hlld_array  # Return entire flux array
    
    # TODO: understand ai code
    def actual_hlld(self, a_grid):
        
        a_grid.assert_variable_type("prim")

        #get density 
        if self.c.system == "euler2D":
            density = a_grid.grid[self.c.RHOCOMP]
        elif self.c.system == "mhd2d":
            density = a_grid.grid[self.c.RHOCOMP]
        else:
            raise RuntimeError("Density method needs to be implemented.")
            
        a = np.sqrt(self.c.gamma * a_grid.grid[self.c.PCOMP] / density)


        U = a_grid.grid
        print(f"actual: {U}")
        consU = self.euler.prim_to_cons(U)
        
        numFluxX_plus = np.zeros_like(a_grid.grid)
        numFluxX_minus = np.zeros_like(a_grid.grid)
        numFluxY_plus = np.zeros_like(a_grid.grid)
        numFluxY_minus = np.zeros_like(a_grid.grid)
        
        # Get the conserved variables and x-direction fluxes
        consU, flux_x = self.hlld1D(a_grid=a_grid, direction="x")
        
        # Get y-direction fluxes
        _, flux_y = self.hlld1D(a_grid=a_grid, direction="y")
        
        for i in range(a_grid.Nghost - 1, a_grid.Nx + a_grid.Nghost):
            for j in range(a_grid.Nghost - 1, a_grid.Ny + a_grid.Nghost):
                for icomp in range(self.c.NUMQ):
                    numFluxX_plus[icomp, i+1, j] = flux_x[icomp, i+1, j]
                    numFluxX_minus[icomp, i, j] = flux_x[icomp, i, j]
                    numFluxY_plus[icomp, i, j+1] = flux_y[icomp, i, j+1]
                    numFluxY_minus[icomp, i, j] = flux_y[icomp, i, j]
                    
        # Return the results as needed
        return consU, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus
