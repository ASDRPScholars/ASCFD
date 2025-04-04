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
        elif self.type == "hlld":
            self.flux_method = self.actual_hlld
        else:
            raise RuntimeError(f"Flux method not supported: {self.type}")


    def getFlux(self, a_grid):
        return self.flux_method(a_grid)


    def rusanov(self, a_grid):
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
        print(U)
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
        elif self.c.system == "mhd2d":
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
