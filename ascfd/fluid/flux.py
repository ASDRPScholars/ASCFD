from ascfd.fluid.euler import FluidEuler
# from ascfd.reconstruct import weno5_reconstruction
from ascfd.fluid.constants import *
from ascfd.fluid.constants import FluidConstants

import numpy as np
import sys


class FluidFlux:

    def __init__(self, a_constants: FluidConstants, a_type: str):

        self.type = a_type
        self.c = a_constants

        self.euler = FluidEuler(self.c) # initialize the euler solver

        # select the flux method based on the type
        if self.type == "rusanov":
            self.flux_method = self.rusanov
        elif self.type == "rusanov_vectorized":
            self.flux_method = self.rusanov_vectorized
        elif self.type == "hllc":
            self.flux_method = self.hllc
        elif self.type == "hlld":
            self.flux_method = self.actual_hlld
        elif self.type == "hlld_new":
            self.flux_method = self.hlld_new
        else:
            raise RuntimeError(f"Flux method not supported: {self.type}")


    def getFlux(self, a_grid, a_Nx, a_Ny, a_Nghost):
        return self.flux_method(a_grid, a_Nx, a_Ny, a_Nghost)


    def rusanov(self, a_grid, a_Nx, a_Ny, a_Nghost):

        #get density 
        if self.c.system == "euler2d":
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
        if self.c.system == "euler2d":
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

        if self.c.system == "euler2d":
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
        if self.c.system != "euler2d":
            raise NotImplementedError("HLLC flux is only implemented for euler2d system.")

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
        
        # print("u", u)
        # print("v", v)
        # print("p", p)
        # print("a", a)
        # print("rho", rho)

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
                    print("[hllc()] SL", SL)
                    print("[hllc()] SL",SR)
                    
                    print("[hllc()] rhoL",rhoL)
                    print("[hllc()] rhoR",rhoR)
                    print("[hllc()] uL",uL)
                    print("[hllc()] uR",uR)

                    print("[hllc()] pL",pL)
                    print("[hllc()] pR",pR)
                    
                    print("[hllc()] aL",aL)
                    print("[hllc()] aR",aR)
                    
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
        if self.c.system == "euler2d":
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
        if self.c.system == "euler2d":
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

    def hlld_new(self, a_grid, a_Nx, a_Ny, a_Nghost):
        """
        Calculates the numerical flux using the HLLD approximate Riemann solver for MHD.
        Based on the formulation by Miyoshi & Kusano (2005).
        """
        if self.c.system != "mhd2d":
            raise NotImplementedError("HLLD flux is currently only implemented for mhd2d system.")
        if self.c.NUMQ != 6:
             raise ValueError("HLLD requires 6 variables (rho, u, v, p, Bx, By) in primitive state.")

        rho = a_grid[self.c.RHOCOMP]; u = a_grid[self.c.UCOMP]; v = a_grid[self.c.VCOMP]
        p = a_grid[self.c.PCOMP]; Bx = a_grid[self.c.BXCOMP]; By = a_grid[self.c.BYCOMP]
        gamma = self.c.gamma
        E = p / (gamma - 1.0) + 0.5 * rho * (u**2 + v**2) + 0.5 * (Bx**2 + By**2)
        
        consU = np.zeros_like(a_grid)
        consU[self.c.RHOCOMP] = rho; consU[self.c.MUCOMP] = rho * u 
        consU[self.c.MVCOMP] = rho * v; consU[self.c.ECOMP] = E
        consU[self.c.BXCOMP] = Bx; consU[self.c.BYCOMP] = By
        
        fx, fy = self._calculate_mhd_fluxes(rho, u, v, p, E, Bx, By, gamma)
        numFluxX = np.zeros_like(a_grid); numFluxY = np.zeros_like(a_grid)

        for i in range(a_Nghost - 1, a_Nx + a_Nghost):
            for j in range(a_Nghost, a_Ny + a_Nghost):
                stateL = {'rho': rho[i, j], 'u': u[i, j], 'v': v[i, j], 'p': p[i, j], 'Bx': Bx[i, j], 'By': By[i, j], 'E': E[i, j]}
                stateR = {'rho': rho[i+1, j], 'u': u[i+1, j], 'v': v[i+1, j], 'p': p[i+1, j], 'Bx': Bx[i+1, j], 'By': By[i+1, j], 'E': E[i+1, j]}
                numFluxX[:, i, j] = self._hlld_1d_flux(stateL, stateR, consU[:, i, j], consU[:, i+1, j], fx[:, i, j], fx[:, i+1, j], gamma, direction='x')

        for i in range(a_Nghost, a_Nx + a_Nghost):
            for j in range(a_Nghost - 1, a_Ny + a_Nghost):
                stateL = {'rho': rho[i, j], 'u': u[i, j], 'v': v[i, j], 'p': p[i, j], 'Bx': Bx[i, j], 'By': By[i, j], 'E': E[i, j]}
                stateR = {'rho': rho[i, j+1], 'u': u[i, j+1], 'v': v[i, j+1], 'p': p[i, j+1], 'Bx': Bx[i, j+1], 'By': By[i, j+1], 'E': E[i, j+1]}
                numFluxY[:, i, j] = self._hlld_1d_flux(stateL, stateR, consU[:, i, j], consU[:, i, j+1], fy[:, i, j], fy[:, i, j+1], gamma, direction='y')

        numFluxX_plus = np.zeros_like(a_grid); numFluxX_minus = np.zeros_like(a_grid)
        numFluxY_plus = np.zeros_like(a_grid); numFluxY_minus = np.zeros_like(a_grid)
        i_int_slice = slice(a_Nghost, a_Nx + a_Nghost); j_int_slice = slice(a_Nghost, a_Ny + a_Nghost)
        numFluxX_plus[:, i_int_slice, j_int_slice] = numFluxX[:, a_Nghost:a_Nx + a_Nghost, j_int_slice]
        numFluxX_minus[:, i_int_slice, j_int_slice] = numFluxX[:, a_Nghost-1:a_Nx + a_Nghost-1, j_int_slice]
        numFluxY_plus[:, i_int_slice, j_int_slice] = numFluxY[:, i_int_slice, a_Nghost:a_Ny + a_Nghost]
        numFluxY_minus[:, i_int_slice, j_int_slice] = numFluxY[:, i_int_slice, a_Nghost-1:a_Ny + a_Nghost-1]
        return consU, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus

    def _calculate_mhd_fluxes(self, rho, u, v, p, E, Bx, By, gamma):
        p_tot = p + 0.5 * (Bx**2 + By**2)
        fx = np.zeros((self.c.NUMQ,) + rho.shape); fy = np.zeros((self.c.NUMQ,) + rho.shape)
        fx[0] = rho * u; fx[1] = rho * u**2 + p_tot - Bx**2; fx[2] = rho * u * v - Bx * By
        fx[3] = (E + p_tot) * u - Bx * (u*Bx + v*By); fx[4] = 0.0; fx[5] = u * By - v * Bx
        fy[0] = rho * v; fy[1] = rho * v * u - By * Bx; fy[2] = rho * v**2 + p_tot - By**2
        fy[3] = (E + p_tot) * v - By * (u*Bx + v*By); fy[4] = v * Bx - u * By; fy[5] = 0.0
        return fx, fy

    def _hlld_1d_flux(self, stateL, stateR, UL, UR, FL, FR, gamma, direction='x'):
        """
        Computes the 1D HLLD flux for a given interface (L/R states).
        Based on Miyoshi & Kusano (2005), Journal of Computational Physics 208, 315-334.
        Handles rotation for y-direction.
        """
        NUMQ = UL.shape[0]
        flux_hlld = np.zeros(NUMQ)
        small_rho = 1e-12 # Avoid division by zero for density
        small_p = 1e-12   # Avoid negative pressure/sound speed issues
        small_num = 1e-14 # General small number for denominators

        # --- Rotate states for y-direction --- 
        if direction == 'y':
            # Swap normal (u, Bx) and tangential (v, By) components
            stateL['u'], stateL['v'] = stateL['v'], stateL['u']
            stateL['Bx'], stateL['By'] = stateL['By'], stateL['Bx']
            stateR['u'], stateR['v'] = stateR['v'], stateR['u']
            stateR['Bx'], stateR['By'] = stateR['By'], stateR['Bx']
            # Rotate conservative state vectors [rho, rhou, rhov, E, Bx, By]
            UL_rot = UL.copy(); UR_rot = UR.copy()
            UL_rot[1], UL_rot[2] = UL[2], UL[1]; UR_rot[1], UR_rot[2] = UR[2], UR[1] # Swap momenta
            UL_rot[4], UL_rot[5] = UL[5], UL[4]; UR_rot[4], UR_rot[5] = UR[5], UR[4] # Swap B fields
            # Rotate analytical flux vectors (G becomes F in rotated frame)
            FL_rot = FL.copy(); FR_rot = FR.copy()
            FL_rot[1], FL_rot[2] = FL[2], FL[1]; FR_rot[1], FR_rot[2] = FR[2], FR[1] # Swap momentum fluxes
            FL_rot[4], FL_rot[5] = FL[5], FL[4]; FR_rot[4], FR_rot[5] = FR[5], FR[4] # Swap B field fluxes
            UL, UR, FL, FR = UL_rot, UR_rot, FL_rot, FR_rot # Use rotated states
            
        # --- Extract states (normal direction is 'u', 'Bx') ---
        # Left state
        rho_L = max(stateL['rho'], small_rho)
        u_L = stateL['u'] # Normal velocity
        v_L = stateL['v'] # Tangential velocity 1
        w_L = 0.0          # Tangential velocity 2 (Placeholder for 3D)
        p_L = max(stateL['p'], small_p)
        Bx_L = stateL['Bx'] # Normal B-field
        By_L = stateL['By'] # Tangential B-field 1
        Bz_L = 0.0          # Tangential B-field 2 (Placeholder for 3D)
        # Right state
        rho_R = max(stateR['rho'], small_rho)
        u_R = stateR['u']
        v_R = stateR['v']
        w_R = 0.0
        p_R = max(stateR['p'], small_p)
        Bx_R = stateR['Bx']
        By_R = stateR['By']
        Bz_R = 0.0

        # Enforce Bx=const across the interface (fundamental to 1D Riemann problem)
        Bx = 0.5 * (Bx_L + Bx_R)
        # if abs(Bx_L - Bx_R) > 1e-9 * (abs(Bx_L) + abs(Bx_R)): # Allow small tolerance
        #      print(f"Warning: Bx_L ({Bx_L}) != Bx_R ({Bx_R}) at interface. Averaging.")
        Bx_L = Bx_R = Bx 

        # Other useful quantities
        ptot_L = p_L + 0.5 * (Bx_L**2 + By_L**2 + Bz_L**2)
        ptot_R = p_R + 0.5 * (Bx_R**2 + By_R**2 + Bz_R**2)
        rho_sqrt_L = np.sqrt(rho_L)
        rho_sqrt_R = np.sqrt(rho_R)
        
        # --- Calculate wave speeds --- 
        # Sound speeds
        a_L = np.sqrt(gamma * p_L / rho_L)
        a_R = np.sqrt(gamma * p_R / rho_R)
        # Fast magnetosonic speeds (Eq. 21 in M&K 2005)
        B_perp_sq_L = By_L**2 + Bz_L**2
        B_perp_sq_R = By_R**2 + Bz_R**2
        cf_L_sq = 0.5 * (a_L**2 + (Bx**2 + B_perp_sq_L)/rho_L + np.sqrt(max( (a_L**2 + (Bx**2 + B_perp_sq_L)/rho_L)**2 - 4*a_L**2*Bx**2/rho_L, 0.0)) )
        cf_L = np.sqrt(cf_L_sq)
        cf_R_sq = 0.5 * (a_R**2 + (Bx**2 + B_perp_sq_R)/rho_R + np.sqrt(max( (a_R**2 + (Bx**2 + B_perp_sq_R)/rho_R)**2 - 4*a_R**2*Bx**2/rho_R, 0.0)) )
        cf_R = np.sqrt(cf_R_sq)
        
        # Estimate signal speeds SL and SR (outermost fast waves)
        SL = min(u_L - cf_L, u_R - cf_R)
        SR = max(u_L + cf_L, u_R + cf_R)

        # --- Determine Flux --- 
        # Check if flow is super-fast magnetosonic
        if SL >= 0:
            flux_hlld = FL
        elif SR <= 0:
            flux_hlld = FR
        else: # Subsonic/Transonic case
            # --- Calculate HLL state (Eq. 30) --- 
            inv_SR_minus_SL = 1.0 / (SR - SL)
            rho_HLL = inv_SR_minus_SL * (SR * UR[0] - SL * UL[0] + FL[0] - FR[0])
            rho_HLL = max(rho_HLL, small_rho) # Ensure positivity
            rhou_HLL = inv_SR_minus_SL * (SR * UR[1] - SL * UL[1] + FL[1] - FR[1])
            rhov_HLL = inv_SR_minus_SL * (SR * UR[2] - SL * UL[2] + FL[2] - FR[2])
            #rhow_HLL = inv_SR_minus_SL * (SR * UR[3] - SL * UL[3] + FL[3] - FR[3]) # If 3D
            E_HLL = inv_SR_minus_SL * (SR * UR[3] - SL * UL[3] + FL[3] - FR[3]) # Adjusted index for 2D MHD E
            # Bx is constant UL[4]
            By_HLL = inv_SR_minus_SL * (SR * UR[5] - SL * UL[5] + FL[5] - FR[5]) # Adjusted index for 2D MHD By
            # Bz_HLL = inv_SR_minus_SL * (SR * UR[6] - SL * UL[6] + FL[6] - FR[6]) # If 3D
            
            # Middle wave speed (contact) S_M = u* (Eq. 31)
            S_M = rhou_HLL / rho_HLL
            
            # Total pressure in star regions p* (Eq. 35)
            ptot_star = inv_SR_minus_SL * ( (SR - u_R)*rho_R*u_L - (SL - u_L)*rho_L*u_R + SR*ptot_R - SL*ptot_L ) # Simplified form from M&K notes
            # ptot_star = inv_SR_minus_SL * (SR*ptot_R - SL*ptot_L + rho_L*u_L*(SL-u_L) - rho_R*u_R*(SR-u_R)) # Original form
            ptot_star = max(ptot_star, small_p) # Ensure positivity

            # --- Calculate Star (*) States --- 
            # Avoid division by zero/negative density by checking speeds vs S_M
            # Left Star State (*L)
            if abs(SL - S_M) < small_num * max(abs(SL), abs(S_M)): # Handle degenerate case SL = SM
                 rho_star_L = rho_HLL # Or some average, density becomes multi-valued
                 # Simplified states or alternative handling needed here. Using HLL state as fallback.
                 UstarL = np.array([rho_HLL, rhou_HLL, rhov_HLL, E_HLL, Bx, By_HLL]) 
            else: 
                rho_star_L = rho_L * (SL - u_L) / (SL - S_M) # Eq. 37
                rho_star_L = max(rho_star_L, small_rho)
                # Denominator for v*, By*, E* (Eq. 41)
                den_L = rho_L * (SL - u_L) * (SL - S_M) - Bx**2
                if abs(den_L) < small_num: den_L = small_num * np.sign(den_L) if den_L != 0 else small_num

                u_star_L = S_M # Normal velocity is S_M
                v_star_L = v_L - Bx * By_L * (S_M - u_L) / den_L # Eq. 38
                w_star_L = w_L #- Bx * Bz_L * (S_M - u_L) / den_L # 3D
                By_star_L = By_L * (rho_L * (SL - u_L)**2 - Bx**2) / den_L # Eq. 39
                # Bz_star_L = Bz_L * (rho_L * (SL - u_L)**2 - Bx**2) / den_L # 3D
                
                # Energy E* (Eq. 40, using dot product form)
                v_dot_B_L = u_L*Bx + v_L*By_L + w_L*Bz_L
                v_star_dot_B_star_L = S_M*Bx + v_star_L*By_star_L #+ w_star_L*Bz_star_L
                E_star_L = ((SL - u_L) * UL[3] - ptot_L * u_L + ptot_star * S_M + Bx * (v_dot_B_L - v_star_dot_B_star_L)) / (SL - S_M)

                UstarL = np.array([rho_star_L, rho_star_L * S_M, rho_star_L * v_star_L, E_star_L, Bx, By_star_L])
            
            # Right Star State (*R)
            if abs(SR - S_M) < small_num * max(abs(SR), abs(S_M)): # Handle degenerate case SR = SM
                rho_star_R = rho_HLL # Fallback
                UstarR = np.array([rho_HLL, rhou_HLL, rhov_HLL, E_HLL, Bx, By_HLL])
            else:
                rho_star_R = rho_R * (SR - u_R) / (SR - S_M) # Eq. 37
                rho_star_R = max(rho_star_R, small_rho)
                # Denominator for v*, By*, E* (Eq. 41)
                den_R = rho_R * (SR - u_R) * (SR - S_M) - Bx**2
                if abs(den_R) < small_num: den_R = small_num * np.sign(den_R) if den_R != 0 else small_num

                u_star_R = S_M # Normal velocity is S_M
                v_star_R = v_R - Bx * By_R * (S_M - u_R) / den_R # Eq. 38
                w_star_R = w_R #- Bx * Bz_R * (S_M - u_R) / den_R # 3D
                By_star_R = By_R * (rho_R * (SR - u_R)**2 - Bx**2) / den_R # Eq. 39
                # Bz_star_R = Bz_R * (rho_R * (SR - u_R)**2 - Bx**2) / den_R # 3D
                
                # Energy E* (Eq. 40)
                v_dot_B_R = u_R*Bx + v_R*By_R + w_R*Bz_R
                v_star_dot_B_star_R = S_M*Bx + v_star_R*By_star_R #+ w_star_R*Bz_star_R
                E_star_R = ((SR - u_R) * UR[3] - ptot_R * u_R + ptot_star * S_M + Bx * (v_dot_B_R - v_star_dot_B_star_R)) / (SR - S_M)
                
                UstarL = np.array([rho_star_L, rho_star_L * S_M, rho_star_L * v_star_L, E_star_L, Bx, By_star_L])
                UstarR = np.array([rho_star_R, rho_star_R * S_M, rho_star_R * v_star_R, E_star_R, Bx, By_star_R])

            # --- Calculate Alfven wave speeds in Star region --- (Eq. 42)
            # S*_L = S_M - |Bx| / sqrt(rho*_L)
            # S*_R = S_M + |Bx| / sqrt(rho*_R)
            sqrt_rho_star_L = np.sqrt(rho_star_L)
            sqrt_rho_star_R = np.sqrt(rho_star_R)
            # Handle Bx = 0 explicitly for Alfven speeds
            if abs(Bx) < small_num:
                S_star_L = S_M
                S_star_R = S_M
            else:
                S_star_L = S_M - abs(Bx) / sqrt_rho_star_L
                S_star_R = S_M + abs(Bx) / sqrt_rho_star_R
            
            # --- Calculate Double-Star (**) State Variables --- 
            # Note: u** = u* = S_M, rho** = rho*, Bx** = Bx
            # Need v**, By**, E** (Eq. 43, 44)
            sign_Bx = np.sign(Bx) if abs(Bx) > small_num else 0.0
            inv_sqrt_rho_sum = 1.0 / (sqrt_rho_star_L + sqrt_rho_star_R)
            
            v_dstar = inv_sqrt_rho_sum * (sqrt_rho_star_L * v_star_L + sqrt_rho_star_R * v_star_R + sign_Bx * (By_star_R - By_star_L))
            # w_dstar = inv_sqrt_rho_sum * (sqrt_rho_star_L * w_star_L + sqrt_rho_star_R * w_star_R + sign_Bx * (Bz_star_R - Bz_star_L)) # 3D
            By_dstar = inv_sqrt_rho_sum * (sqrt_rho_star_L * By_star_R + sqrt_rho_star_R * By_star_L + sign_Bx * sqrt_rho_star_L * sqrt_rho_star_R * (v_star_R - v_star_L))
            # Bz_dstar = inv_sqrt_rho_sum * (sqrt_rho_star_L * Bz_star_R + sqrt_rho_star_R * Bz_star_L + sign_Bx * sqrt_rho_star_L * sqrt_rho_star_R * (w_star_R - w_star_L)) # 3D

            # Energy E** (Eq. 45)
            # E** = E* - sqrt(rho*) sgn(Bx) [ (v* - v**)By* + (w* - w**)Bz* ]
            E_dstar_L = E_star_L - sqrt_rho_star_L * sign_Bx * ((v_star_L - v_dstar) * By_star_L) # + (w_star_L - w_dstar)*Bz_star_L
            E_dstar_R = E_star_R + sqrt_rho_star_R * sign_Bx * ((v_star_R - v_dstar) * By_star_R) # + (w_star_R - w_dstar)*Bz_star_R
            
            # Construct U**_L and U**_R vectors [rho*, rho*u*, rho*v**, E**, Bx, By**]
            # Note: rho, u, Bx are same as U* state, v, By, E are different
            UdstarL = UstarL.copy(); UdstarR = UstarR.copy()
            UdstarL[2] = rho_star_L * v_dstar # rho* v**
            UdstarR[2] = rho_star_R * v_dstar # rho* v**
            UdstarL[3] = E_dstar_L           # E**_L
            UdstarR[3] = E_dstar_R           # E**_R
            UdstarL[5] = By_dstar            # By**
            UdstarR[5] = By_dstar            # By**
            
            # --- Determine flux based on region (Eq. 36 from Miyoshi & Kusano 2005) ---
            if 0 <= SL: # Region L
                flux_hlld = FL
            elif SL < 0 <= S_star_L: # Region *L
                flux_hlld = FL + SL * (UstarL - UL)
            elif S_star_L < 0 <= S_M: # Region **L
                # F**_L = F*_L + S*_L(U**_L - U*_L)
                # F**_L = FL + SL(U*L - UL) + S*_L(U**_L - U*_L)
                flux_hlld = FL + SL * (UstarL - UL) + S_star_L * (UdstarL - UstarL)
            elif S_M < 0 <= S_star_R: # Region **R
                # F**_R = F*_R + S*_R(U**_R - U*_R)
                # F**_R = FR + SR(U*R - UR) + S*_R(U**_R - U*_R)
                flux_hlld = FR + SR * (UstarR - UR) + S_star_R * (UdstarR - UstarR)
            elif S_star_R < 0 <= SR: # Region *R
                flux_hlld = FR + SR * (UstarR - UR)
            elif SR < 0: # Region R
                flux_hlld = FR
            else: # Should not happen
                 # Fallback or raise error
                 print(f"Warning: HLLD flux logic error. Speeds: SL={SL:.2e}, S*L={S_star_L:.2e}, SM={S_M:.2e}, S*R={S_star_R:.2e}, SR={SR:.2e}")
                 raise ValueError("HLLD flux somethign is going wrong :o")

        # --- Rotate flux back for y-direction --- 
        if direction == 'y':
            flux_hlld_rot = flux_hlld.copy()
            flux_hlld_rot[1], flux_hlld_rot[2] = flux_hlld[2], flux_hlld[1] # Swap F_rhou <-> F_rhov
            flux_hlld_rot[4], flux_hlld_rot[5] = flux_hlld[5], flux_hlld[4] # Swap F_Bx <-> F_By
            return flux_hlld_rot
        else:
            return flux_hlld
