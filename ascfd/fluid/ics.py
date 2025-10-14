from ascfd.fluid.constants import FluidConstants
from ascfd.inputs import Inputs
from ascfd.params import SpeciesParams
import numpy as np

class FluidInitialConditions:
    def __init__(self, grid, a_inputs: Inputs, params: SpeciesParams):
        self.c = FluidConstants(a_inputs)
        self.params = params
        self.inp = a_inputs
        self.grid = grid
        
        self.mesh_x, self.mesh_y = np.meshgrid(self.inp.grid_x, self.inp.grid_y)
        
    def apply_ics(self):
        if self.inp.system == "euler2d":
            if hasattr(self.inp, "fluid_ics"):
                if self.inp.fluid_ics == "diagonal_advection":
                    print("applying diag advection")
                    f = self.diagonal_advection_2d
                    
                    new_grid = np.zeros_like(self.grid)
                    
                    for var in range(self.c.NUMQ):
                        print(var)
                        new_grid[var] = f(self.mesh_x, self.mesh_y, var)
                        
                    return new_grid
                
                elif self.inp.fluid_ics == "poisson_validation":
                    print("applying diag advection")
                    f = self.poisson_validation_charge_density
                    
                    new_grid = np.zeros_like(self.grid)
                    
                    for var in range(self.c.NUMQ):
                        print(var)
                        new_grid[var] = f(self.mesh_x, self.mesh_y, var)
                        
                    return new_grid

                elif self.inp.fluid_ics == "static":
                    return self.static()
                
                elif self.inp.fluid_ics == "e_cloud_test":
                    return self.e_cloud_test()
                    
                elif self.inp.fluid_ics == "kelvin_helmholtz":
                    self.grid = self.kelvin_helmholtz_2d()
                elif self.inp.fluid_ics == "double_mach_reflection":
                    self.grid = self.double_mach_reflection_2d()
                elif self.inp.fluid_ics == "riemann_problem":
                    self.grid = self.riemann_2d()

                else:
                    raise RuntimeError("[FLUID] ICS not valid.")
            else:
                return self.static()
            
        elif self.inp.system == "quasineutral":
            if self.inp.fluid_ics == "static":
                return self.qn_static()
           
        elif self.inp.system == "mhd2d":
            if self.inp.fluid_ics == "orszag_tang":
                self.grid = self.orszag_tang_2d()
           
        else:
            raise RuntimeError("[FLUID] ICS not valid.")
        
        
    # TODO: FIX LATER JUST MAKE THEM RETURN GRIDS WE CAN SET EQUAL TO
    # def diagonal_advection_2d(self):
    #     ic_grid = np.zeros_like(self.grid)
        
    #     ic_grid[self.c.RHOCOMP, :, :] = 1
    #     ic_grid[self.c.UCOMP, :, :] = 1
    #     ic_grid[self.c.VCOMP, :, :] = 1
    #     ic_grid[self.c.PCOMP, :, :] = 1
        
    #     return ic_grid
    
    
            
    def diagonal_advection_2d(self, a_x, a_y, a_var, t=0):
        """
        Diagonal advection test case for 2D Euler equations.
        """
        if a_var == 0: #RHOCOMP
            return 1.0 + 0.2 * np.sin(2 * np.pi * (a_x + a_y - t))
        elif a_var == 1: #UCOMP
            return np.ones_like(a_x) * 10
        elif a_var == 2: #VCOMP
            return np.ones_like(a_x) * 10
        elif a_var == 3: #WCOMP (z-velocity)
            return np.zeros_like(a_x)  # Start with zero azimuthal velocity
        elif a_var == 4: #PCOMP
            return np.ones_like(a_x)
        else:
            return 0
        # else:
        #     raise ValueError(f"Unexpected variable: {a_var}")

    def kelvin_helmholtz_2d(a_x, a_y, a_var):
        """
        Kelvin-Helmholtz instability test case for 2D Euler equations.
        """
        if a_var == 0: #RHOCOMP
            return np.ones_like(a_x)
        elif a_var == 1: #UCOMP
            return 0.5 * (np.tanh(20 * a_y) - 1)
        elif a_var == 2: #VCOMP
            return 0.1 * np.sin(2 * np.pi * a_x)
        elif a_var == 3: #PCOMP
            return 2.5 * np.ones_like(a_x)
        else:
            raise ValueError(f"Unexpected variable: {a_var}")

    def double_mach_reflection_2d(a_x, a_y, a_var):
        """
        Double Mach reflection test case for 2D Euler equations.
        """
        x0 = 1/6
        mask = a_x < x0 + a_y / np.sqrt(3)
        
        if a_var == 0: #RHOCOMP
            return np.where(mask, 8.0, 1.4)
        elif a_var == 1: #UCOMP
            return np.where(mask, 8.25 * np.cos(np.pi/6), 0.0)
        elif a_var == 2: #VCOMP
            return np.where(mask, -8.25 * np.sin(np.pi/6), 0.0)
        elif a_var == 3: #PCOMP
            return np.where(mask, 116.5, 1.0)
        else:
            raise ValueError(f"Unexpected variable: {a_var}")

    def riemann_2d(self, a_x, a_y, a_var):
        """
        2D Riemann problem for 2D Euler equations.
        """
        # Define the four states
        rho = np.select([
            (a_x >= 0.5) & (a_y >= 0.5),
            (a_x < 0.5) & (a_y >= 0.5),
            (a_x < 0.5) & (a_y < 0.5),
            (a_x >= 0.5) & (a_y < 0.5)
        ], [1.5, 0.5323, 0.138, 0.5323])
        
        u = np.select([
            (a_x >= 0.5) & (a_y >= 0.5),
            (a_x < 0.5) & (a_y >= 0.5),
            (a_x < 0.5) & (a_y < 0.5),
            (a_x >= 0.5) & (a_y < 0.5)
        ], [0.0, 1.206, 1.206, 0.0])
        
        v = np.select([
            (a_x >= 0.5) & (a_y >= 0.5),
            (a_x < 0.5) & (a_y >= 0.5),
            (a_x < 0.5) & (a_y < 0.5),
            (a_x >= 0.5) & (a_y < 0.5)
        ], [0.0, 0.0, 1.206, 1.206])
        
        p = np.select([
            (a_x >= 0.5) & (a_y >= 0.5),
            (a_x < 0.5) & (a_y >= 0.5),
            (a_x < 0.5) & (a_y < 0.5),
            (a_x >= 0.5) & (a_y < 0.5)
        ], [1.5, 0.3, 0.029, 0.3])

        if a_var == 0: #RHOCOMP
            return rho
        elif a_var == 1: #UCOMP
            return u
        elif a_var == 2: #VCOMP
            return v
        elif a_var == 3: #PCOMP
            return p
        else:
            return 0
        
    def orszag_tang_2d(a_x, a_y, a_var):
        """
        Orszag-Tang vortex test case for 2D MHD.
        """ 
        if a_var == 0:  # RHOCOMP
            return np.ones_like(a_x)
        elif a_var == 1:  # UCOMP
            return -np.sin(2 * np.pi * a_y)
        elif a_var == 2:  # VCOMP
            return np.sin(2 * np.pi * a_x)
        elif a_var == 3:  # PCOMP
            return 5/(12*np.pi)
        elif a_var == 4:  # BX
            return -np.sin(2 * np.pi * a_y)
        elif a_var == 5:  # BY
            return np.sin(4 * np.pi * a_x)
        else:
            raise ValueError(f"Unexpected variable: {a_var}")
    
    def poisson_validation_charge_density(self, a_x, a_y, a_var):
        """
        Poisson validation test: charge density = -sin(x) - cos(y)
        This should produce electric field corresponding to sin(x) + cos(y)
        """
        if a_var == 0:  # Charge density component
            return -np.sin(a_x*10) - np.cos(a_y*10)
        else:
            # Set all other components to zero
            return np.zeros_like(a_x)
    
    # ## real tame
    # def static(self):
    #     ic = np.zeros_like(self.grid)
    #     rho0 = 9e-11                # kg/m³  → n≈1e20 m⁻³
    #     u0 = 0.0                     # m/s
    #     v0 = 0.0
    #     T0 = 1                     # K, choose a reasonable electron/ion temperature
    #     p0 = rho0/self.params.mass * self.c.k_B * T0  # ideal‑gas law: n kT
    #     E0 = p0/(self.c.gamma-1) + 0.5*rho0*(u0**2+v0**2)
        
    #     ic[self.c.RHOCOMP] = rho0
    #     ic[self.c.UCOMP ] = u0  
    #     ic[self.c.VCOMP ] = v0
    #     ic[self.c.PCOMP ] = p0
    #     return ic
    
    ## wip tame
    # def static(self):
    #     ic = np.zeros_like(self.grid)
    #     rho0 = 9.11e-13                # kg/m³  → n≈1e20 m⁻³
    #     u0 = 0.0                     # m/s
    #     v0 = 0.0
    #     T0 = 10                     # K, choose a reasonable electron/ion temperature
    #     p0 = rho0/self.params.mass * self.c.k_B * T0  # ideal‑gas law: n kT
    #     E0 = p0/(self.c.gamma-1) + 0.5*rho0*(u0**2+v0**2)
        
    #     ic[self.c.RHOCOMP] = 9.11e-13
    #     ic[self.c.UCOMP ] = 0  
    #     ic[self.c.VCOMP ] = 0
    #     ic[self.c.PCOMP ] = p0
    #     return ic
    
    #normalized
    # def static(self):
    #     ic = np.zeros_like(self.grid)

    #     # Normalized values (order unity)
    #     rho_norm = 5e-6                  # ρ/ρ_ref = 1
    #     u_norm = 0.0                    # u/v_ref = 0  
    #     v_norm = 0.0                    # v/v_ref = 0
    #     T_norm = 1.0                    # T/T_ref = 1
    #     p_norm = rho_norm * T_norm      # p/p_ref = (ρ/ρ_ref)(T/T_ref)

    #     ic[self.c.RHOCOMP] = rho_norm
    #     ic[self.c.UCOMP] = u_norm
    #     ic[self.c.VCOMP] = v_norm
    #     ic[self.c.WCOMP] = 0.0  # Initialize z-velocity to zero
    #     ic[self.c.PCOMP] = p_norm
    #     return ic
    
    def static(self):
        ic = np.zeros_like(self.grid)

        ic[self.c.RHOCOMP] = self.inp.rho_e / 100
        ic[self.c.UCOMP] = 0
        ic[self.c.VCOMP] = 0
        ic[self.c.WCOMP] = 0.0  # Initialize z-velocity to zero
        ic[self.c.PCOMP] = self.inp.p_e / 100
        
        # print("!NORM! rho_e", self.inp.rho_e)
        # print("!NORM! p_e", self.inp.p_e)
        
        return ic
    
    
    def qn_static(self):
        ic = np.zeros_like(self.grid)
        
        print("[ics] QN STATIC!!")
        
        ic[self.c.NCOMP] = 1e18
        ic[self.c.JXCOMP] = 0
        ic[self.c.JYCOMP] = 0
        ic[self.c.JZCOMP] = 0
        ic[self.c.TCOMP] = 1
        
        print("[ics] INIT WITH:", ic[self.c.NCOMP])
        
        return ic
    
    
    def e_cloud_test(self):
        ic = np.zeros_like(self.grid)

        # Create meshgrid of x-values
        nx, ny = self.inp.nx, self.inp.ny
        ng = self.inp.ng
        dx = self.inp.dx
        x_start = self.inp.xlim[0]

        for i in range(ng, nx + ng):
            x = x_start + (i - ng + 0.5) * dx  # cell center x-position

            for j in range(ng, ny + ng):
                # Gaussian peak at x=0.75 with exponential tail to the right
                center = 0.75
                sigma = 0.05  # you can tweak this
                
                if x <= center:
                    # Gaussian on the left side and at peak
                    gaussian_factor = np.exp(-0.5 * ((x - center) / sigma)**2)
                    rho_val = 0.1 + 9.9 * gaussian_factor  # peaks at 10.0, base at 0.1
                    p_val = 10.0 + 990.0 * gaussian_factor  # peaks at 1000.0, base at 10.0
                else:
                    # Exponential tail on the right side
                    tail_decay = 0.1  # controls how fast the tail decays (you can tweak)
                    exp_factor = np.exp(-(x - center) / tail_decay)
                    rho_val = 1 + 9.9 * exp_factor  # starts at 10.0 at center, decays to 0.1
                    p_val = 10.0 + 990.0 * exp_factor  # starts at 1000.0 at center, decays to 10.0
                
                ic[self.c.RHOCOMP, i, j] = rho_val
                ic[self.c.PCOMP, i, j] = p_val

                ic[self.c.UCOMP, i, j] = 0.0
                ic[self.c.VCOMP, i, j] = 0.0
                ic[self.c.WCOMP, i, j] = 0.0  # Initialize z-velocity to zero

        return ic
    