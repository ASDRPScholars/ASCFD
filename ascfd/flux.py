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
            self.flux_method = self.hlld
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

    def flux(self, U, pT, u, v, w, By, Bz, Bx):
        rho, _, _, _, p, _, _ = U
        
        F = np.zeros(7)
        F[0] = rho * u
        F[1] = rho * u**2 + pT - Bx**2
        F[2] = rho * u * v - Bx * By
        F[3] = rho * u * w - Bx * Bz
        F[4] = (p + 0.5 * (Bx**2 + By**2 + Bz**2)) * u - Bx * (u * Bx + v * By + w * Bz)
        F[5] = By * u - Bx * v
        F[6] = Bz * u - Bx * w
        
        return F
    
    def hlld(self, a_grid):
        a_grid.assert_variable_type("prim")

        U = a_grid.grid
        consU = self.euler.prim_to_cons(U)

        numFluxX_plus = np.zeros_like(a_grid.grid)
        numFluxX_minus = np.zeros_like(a_grid.grid)
        numFluxY_plus = np.zeros_like(a_grid.grid)
        numFluxY_minus = np.zeros_like(a_grid.grid)

        for i in range(a_grid.Nghost - 1, a_grid.Nx + a_grid.Nghost):
            for j in range(a_grid.Nghost - 1, a_grid.Ny + a_grid.Nghost):
                Bx = U[self.c.B_XCOMP, i, j]
                U_L = U[:, i, j]
                U_R = U[:, i+1, j]
                
                rho_L, u_L, v_L, p_L, Bx_L, By_L = U_L
                rho_R, u_R, v_R, p_R, Bx_R, By_R = U_R
                w_L=0
                w_R=0
                Bz_L=0
                Bz_R=0
                U_L=np.insert(U_L, 3, w_L)
                U_R=np.insert(U_R, 3, w_R)
                
                pT_L = p_L + 0.5 * (Bx**2 + By_L**2 + Bz_L**2)
                pT_R = p_R + 0.5 * (Bx**2 + By_R**2 + Bz_R**2)
                
                cf_L = np.sqrt((pT_L + Bx**2) / rho_L)
                cf_R = np.sqrt((pT_R + Bx**2) / rho_R)
                
                S_L = min(u_L - cf_L, u_R - cf_R)
                S_R = max(u_L + cf_L, u_R + cf_R)
                
                SM = ((S_R * rho_R * u_R - S_L * rho_L * u_L + pT_L - pT_R) /
                      (S_R * rho_R - S_L * rho_L))
                pT_star = ((S_R * rho_R * pT_L - S_L * rho_L * pT_R + rho_R * rho_L * (S_R - u_R) * (S_L - u_L) * (u_R - u_L)) /
                           (S_R * rho_R - S_L * rho_L))
                
                rho_star_L = rho_L * (S_L - u_L) / (S_L - SM)
                rho_star_R = rho_R * (S_R - u_R) / (S_R - SM)

                S_star_L = SM - np.linalg.norm(Bx) / np.sqrt(rho_star_L)
                S_star_R = SM + np.linalg.norm(Bx) / np.sqrt(rho_star_R)
                
                
                sqrt_rho_L = np.sqrt(rho_star_L)
                sqrt_rho_R = np.sqrt(rho_star_R)
                rho_sum = sqrt_rho_L + sqrt_rho_R
                sign_Bx = np.sign(Bx)
                
                v_double_star = (sqrt_rho_L * v_L + sqrt_rho_R * v_R + (By_R - By_L) * sign_Bx) / rho_sum
                w_double_star = (sqrt_rho_L * w_L + sqrt_rho_R * w_R + (Bz_R - Bz_L) * sign_Bx) / rho_sum
                By_double_star = (sqrt_rho_L * By_R + sqrt_rho_R * By_L + sqrt_rho_L * sqrt_rho_R * (v_R - v_L) * sign_Bx) / rho_sum
                Bz_double_star = (sqrt_rho_L * Bz_R + sqrt_rho_R * Bz_L + sqrt_rho_L * sqrt_rho_R * (w_R - w_L) * sign_Bx) / rho_sum
                
                v_star_L = v_L - (Bx * (By_L - By_R)) / (rho_star_L * (S_L - SM))
                w_star_L = w_L - (Bx * (Bz_L - Bz_R)) / (rho_star_L * (S_L - SM))
                By_star_L = By_L
                Bz_star_L = Bz_L
                
                v_star_R = v_R - (Bx * (By_R - By_L)) / (rho_star_R * (S_R - SM))
                w_star_R = w_R - (Bx * (Bz_R - Bz_L)) / (rho_star_R * (S_R - SM))
                By_star_R = By_R
                Bz_star_R = Bz_R

                F_L = self.flux(U_L, pT_L, u_L, v_L, w_L, By_L, Bz_L, Bx)
                F_R = self.flux(U_R, pT_R, u_R, v_R, w_R, By_R, Bz_R, Bx)
                U_double_star_R = np.array([rho_star_R, SM, v_double_star, w_double_star, pT_star, By_double_star, Bz_double_star])
                U_double_star_L = np.array([rho_star_L, SM, v_double_star, w_double_star, pT_star, By_double_star, Bz_double_star])
                U_star_L = np.array([rho_star_L, SM, v_L, w_L, pT_star, By_L, Bz_L])
                U_star_R = np.array([rho_star_R, SM, v_R, w_R, pT_star, By_R, Bz_R])
                
                if S_L > 0:
                    F_hlld = F_L
                elif S_L <= 0 <= S_star_L:
                    F_hlld = F_L + S_L*(U_star_L-U_L) + S_star_L*(U_double_star_L - U_star_L)
                elif S_star_L <= 0 <= SM:
                    F_hlld = F_L + S_L*(U_star_L-U_L) + S_star_R*(U_double_star_R - U_star_R)
                elif SM <= 0 <= S_star_R:
                    F_hlld = F_R + S_R*(U_star_R - U_R) + S_star_R*(U_double_star_R - U_star_R)
                elif S_star_R <= 0 <= S_R:
                    F_hlld = F_R + S_R*(U_star_R - U_R)
                elif S_R <= 0:
                    F_hlld = F_R
                
                for icomp in range(self.c.NUMQ):
                    numFluxX_plus[icomp, i, j] = F_hlld[icomp]
                    numFluxX_minus[icomp, i, j] = F_hlld[icomp]

        return consU, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus
