import numpy as np
from ascfd.constants import Constants


class Euler:

    def __init__(self, a_constants: Constants):
        # store constants and system parameters
        self.c = a_constants

    def prim_to_cons(self, a_prim):
        cons = np.zeros_like(a_prim)

        if self.c.system == "euler2D":
            # density stays the same
            cons[self.c.RHOCOMP] = a_prim[self.c.RHOCOMP]
            cons[self.c.MUCOMP] = a_prim[self.c.RHOCOMP] * a_prim[self.c.UCOMP]  # momentum components x & y
            cons[self.c.MVCOMP] = a_prim[self.c.RHOCOMP] * a_prim[self.c.VCOMP]
            # compute total energy
            
            # for i in range(np.shape(a_prim)[0]):
            #     for j in range(np.shape(a_prim)[1]):
            #         print("rho, u, v", a_prim[self.c.RHOCOMP, i, j], a_prim[self.c.UCOMP, i, j], a_prim[self.c.VCOMP, i, j])
            #         E = np.zeros((104, 104))
            #         E[i, j] = (a_prim[self.c.PCOMP, i, j] / ((self.c.gamma - 1) * a_prim[self.c.RHOCOMP, i, j]) +
            #             0.5 * (a_prim[self.c.UCOMP, i, j]**2 + a_prim[self.c.VCOMP, i, j]**2))

            E = (a_prim[self.c.PCOMP] / ((self.c.gamma - 1) * a_prim[self.c.RHOCOMP]) +
                 0.5 * (a_prim[self.c.UCOMP]**2 + a_prim[self.c.VCOMP]**2))
            cons[self.c.ECOMP] = E * a_prim[self.c.RHOCOMP]

        elif self.c.system == "mhd2d":
            cons[self.c.RHOCOMP] = a_prim[self.c.RHOCOMP]
            cons[self.c.MUCOMP] = a_prim[self.c.RHOCOMP] * a_prim[self.c.UCOMP]
            cons[self.c.MVCOMP] = a_prim[self.c.RHOCOMP] * a_prim[self.c.VCOMP]
            cons[self.c.BXCOMP] = a_prim[self.c.BXCOMP]
            cons[self.c.BYCOMP] = a_prim[self.c.BYCOMP]
            
            # for i in range(np.shape(a_prim)[0]):
            #     for j in range(np.shape(a_prim)[1]):
            #         print("(i, j) is: ", i, j)
            #         print("rho, u, v, Bx, By", a_prim[self.c.RHOCOMP, i, j], a_prim[self.c.UCOMP, i, j], a_prim[self.c.VCOMP, i, j], a_prim[self.c.BXCOMP, i, j], a_prim[self.c.BYCOMP, i, j])
            #         E = np.zeros((104, 104))
            #         E[i, j] = (a_prim[self.c.PCOMP, i, j] / ((self.c.gamma - 1) * a_prim[self.c.RHOCOMP, i, j]) +
            #             0.5 * (a_prim[self.c.UCOMP, i, j]**2 + a_prim[self.c.VCOMP, i, j]**2)) + 0.5 * (a_prim[self.c.BXCOMP, i, j]**2 + a_prim[self.c.BYCOMP, i, j]**2)

            # print(np.shape(E), np.shape(cons))
            
            E = (a_prim[self.c.PCOMP] / ((self.c.gamma - 1) * a_prim[self.c.RHOCOMP]) +
                 0.5 * (a_prim[self.c.UCOMP]**2 + a_prim[self.c.VCOMP]**2)) + 0.5 * (a_prim[self.c.BXCOMP]**2 + a_prim[self.c.BYCOMP]**2)
                        
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
            kinetic_energy = 0.5 * \
                (prim[self.c.UCOMP]**2 + prim[self.c.VCOMP]**2)
            prim[self.c.PCOMP] = (self.c.gamma - 1) * (
                a_cons[self.c.ECOMP] - a_cons[self.c.RHOCOMP] * kinetic_energy
            )

        elif self.c.system == "mhd2d":
            prim[self.c.RHOCOMP] = a_cons[self.c.RHOCOMP]
            prim[self.c.UCOMP] = a_cons[self.c.MUCOMP] / a_cons[self.c.RHOCOMP]
            prim[self.c.VCOMP] = a_cons[self.c.MVCOMP] / a_cons[self.c.RHOCOMP]
            prim[self.c.BXCOMP] = a_cons[self.c.BXCOMP]
            prim[self.c.BYCOMP] = a_cons[self.c.BYCOMP]
            kinetic_energy = 0.5 * \
                (prim[self.c.UCOMP]**2 + prim[self.c.VCOMP]**2)
            magnetic_energy = 0.5 * \
                (prim[self.c.BXCOMP]**2 + prim[self.c.BYCOMP]**2)
            prim[self.c.PCOMP] = (self.c.gamma - 1) * (
                a_cons[self.c.ECOMP] - a_cons[self.c.RHOCOMP] *
                kinetic_energy - magnetic_energy
            )

        else:
            raise RuntimeError(f"System not supported: {self.c.system}")

        return prim

    def flux(self, a_prim):
        """
        Compute the flux for the Euler equations
        """

        flux_x = np.zeros_like(a_prim)  # flux in x-direction
        flux_y = np.zeros_like(a_prim)  # flux in y-direction

        if self.c.system == "euler2D":
            # extract primitive variables
            rho = a_prim[self.c.RHOCOMP]  # density
            u = a_prim[self.c.UCOMP]  # x-velocity
            v = a_prim[self.c.VCOMP]  # y-velocity
            p = a_prim[self.c.PCOMP]  # pressure

            # compute total energy
            # internal energy (from ideal gas law)
            e = p / ((self.c.gamma - 1) * rho)
            # total energy: internal + kinetic
            E = rho * (e + 0.5 * (u**2 + v**2))

            # flux in x-direction
            flux_x[self.c.RHOCOMP] = rho * u  # mass flux
            flux_x[self.c.MUCOMP] = rho * u**2 + p  # momentum flux in x
            flux_x[self.c.MVCOMP] = rho * u * v  # momentum flux in y
            flux_x[self.c.ECOMP] = (E + p) * u  # energy flux

            # flux in y-direction
            flux_y[self.c.RHOCOMP] = rho * v  # mass flux
            # - rho * self.c.g # momentum flux in x
            flux_y[self.c.MUCOMP] = rho * u * v
            flux_y[self.c.MVCOMP] = rho * v**2 + p  # momentum flux in y
            flux_y[self.c.ECOMP] = (E + p) * v  # energy flux

        elif self.c.system == "mhd2d":
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
            flux_x[self.c.MUCOMP] = rho * u**2 + \
                p + 0.5 * (b_x**2 + b_y**2) - b_x**2
            flux_x[self.c.MVCOMP] = rho * u * v - b_x * b_y
            flux_x[self.c.ECOMP] = (
                E + p + 0.5 * (b_x**2 + b_y**2)) * u - b_x * (u * b_x + v * b_y)
            flux_x[self.c.BXCOMP] = 0
            flux_x[self.c.BYCOMP] = u * b_y - v * b_x

            # Flux in y-direction
            flux_y[self.c.RHOCOMP] = rho * v
            flux_y[self.c.MUCOMP] = rho * u * v - b_x * b_y
            flux_y[self.c.MVCOMP] = rho * v**2 + \
                p + 0.5 * (b_x**2 + b_y**2) - b_y**2
            flux_y[self.c.ECOMP] = (
                E + p + 0.5 * (b_x**2 + b_y**2)) * v - b_y * (u * b_x + v * b_y)
            flux_y[self.c.BXCOMP] = v * b_x - u * b_y
            flux_y[self.c.BYCOMP] = 0

        else:
            # raise error for unsupported systems
            raise RuntimeError(f"System not supported: {self.c.system}")

        return flux_x, flux_y

    def get_max_speed(self, a_grid):

        if self.c.system == "euler2D":
            if a_grid.variables == "prim":
                return np.max(a_grid.grid[self.c.UCOMP])
            elif a_grid.variables == "cons":
                return np.max(a_grid.grid[self.c.MUCOMP] / a_grid.grid[self.c.RHOCOMP])
            else:
                print("unsupported")
                exit()
        elif self.c.system == "mhd2d":
            if a_grid.variables == "prim":
                cs = np.sqrt(
                    self.c.gamma * a_grid.grid[self.c.PCOMP] / a_grid.grid[self.c.RHOCOMP])
                mag_pressure = (
                    a_grid.grid[self.c.BXCOMP]**2 + a_grid.grid[self.c.BYCOMP]**2) / a_grid.grid[self.c.RHOCOMP]
                cf = np.sqrt(cs**2 + mag_pressure)
                speed = np.sqrt(
                    a_grid.grid[self.c.UCOMP]**2 + a_grid.grid[self.c.VCOMP]**2) + cf
                return np.max(speed)
            elif a_grid.variables == "cons":
                u = a_grid.grid[self.c.MUCOMP] / a_grid.grid[self.c.RHOCOMP]
                v = a_grid.grid[self.c.MVCOMP] / a_grid.grid[self.c.RHOCOMP]
                kinetic_energy = 0.5 * (u**2 + v**2)
                mag_pressure = 0.5 * \
                    (a_grid.grid[self.c.BXCOMP]**2 +
                     a_grid.grid[self.c.BYCOMP]**2)
                internal_energy = a_grid.grid[self.c.ECOMP] / a_grid.grid[self.c.RHOCOMP] - \
                    kinetic_energy - mag_pressure / a_grid.grid[self.c.RHOCOMP]
                p = (self.c.gamma - 1) * \
                    a_grid.grid[self.c.RHOCOMP] * internal_energy
                cs = np.sqrt(self.c.gamma * p / a_grid.grid[self.c.RHOCOMP])
                mag_pressure = (
                    a_grid.grid[self.c.BXCOMP]**2 + a_grid.grid[self.c.BYCOMP]**2) / a_grid.grid[self.c.RHOCOMP]
                cf = np.sqrt(cs**2 + mag_pressure)
                speed = np.sqrt(u**2 + v**2) + cf
                return np.max(speed)
            else:
                print("unsupported")
                exit()

# def g(rho):
#     # definiton of gravity
#     return rho * 9.81
