import matplotlib.pyplot as plt
from ascfd.inputs import Inputs
from ascfd.fields.ics import FieldInitialConditions
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D projection only
# from sympy import sin, cos
# from sympy.abc import x, y

from poissonpy import solvers


class Fields:
    def __init__(self, a_inputs: Inputs):
        self.inp = a_inputs
        
        self.E = np.zeros((self.inp.nx, self.inp.ny, 3))
        self.B = np.zeros((self.inp.nx, self.inp.ny, 3))
        
        self.ics = FieldInitialConditions(self.E, self.B, self.inp)
        
        self.B = self.ics.apply_B_ics()
        self.ics.apply_E_ics()
        
        plt.figure()
        im = plt.imshow(self.B[:, :, 1])
        plt.title("apply_B_ics() magnetic field")
        plt.colorbar(im)
        plt.show()
        
        self.charge_density = np.zeros((self.inp.nx, self.inp.ny))
        self.potential = np.zeros((self.inp.nx, self.inp.ny))

        self.dx = (self.inp.xlim[1] - self.inp.xlim[0]) / (self.inp.nx - 1)
        self.dy = (self.inp.ylim[1] - self.inp.ylim[0]) / (self.inp.ny - 1)
        
        self.eps0 = 8.854e-12  # TODO: CHECK — Permittivity of free space
                
                
    def update_E(self):
        print("UPDATE E")
        self.solve_poisson()
        self._compute_electric_field()
        
        # print("THIS IS CHARGE DENSITY TAKEN IN BY POISSON:", self.charge_density)
        # print("NEW CALCULATED EX IS:", self.E[:, :, 0])
        # print("NEW CALCULATED EY IS:", self.E[:, :, 1])

        # fig, axs = plt.subplots(2, 2, figsize=(15, 15))  # 1 row, 3 columns
        # axs = axs.flatten()

        # # Plot charge density
        # im0 = axs[0].imshow(self.charge_density, cmap="coolwarm")
        # axs[0].set_title("Charge Density")
        # fig.colorbar(im0, ax=axs[0])

        # # Plot Ex
        # im1 = axs[1].imshow(self.E[:, :, 0], cmap="coolwarm")
        # axs[1].set_title("Electric Field Ex")
        # fig.colorbar(im1, ax=axs[1])

        # # Plot Ey
        # im2 = axs[2].imshow(self.E[:, :, 1], cmap="coolwarm")
        # axs[2].set_title("Electric Field Ey")
        # fig.colorbar(im2, ax=axs[2])
        
        # # Plot E
        # im3 = axs[3].imshow(np.sqrt(self.E[:, :, 0]**2 + self.E[:, :, 1]**2), cmap="coolwarm")
        # axs[3].set_title("Electric Field Magnitude")
        # fig.colorbar(im3, ax=axs[3])
        
        # plt.tight_layout()
        # plt.show()
        
        # ### 3D PLOT
        # # Compute E magnitude
        # E_mag = np.sqrt(self.E[5:-5, 5:-5, 0]**2 + self.E[5:-5, 5:-5, 1]**2)

        # # Create meshgrid for X and Y
        # nx, ny = E_mag.shape
        # x = np.arange(nx)
        # y = np.arange(ny)
        # X, Y = np.meshgrid(y, x)  # careful: meshgrid uses (cols, rows) order

        # # Create figure
        # fig = plt.figure(figsize=(10, 8))
        # ax = fig.add_subplot(111, projection='3d')

        # # Plot the surface
        # surf = ax.plot_surface(X, Y, E_mag, cmap='coolwarm')

        # # Add colorbar and labels
        # fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        # ax.set_title("ours")
        # ax.set_xlabel("X")
        # ax.set_ylabel("Y")
        # ax.set_zlabel("|E|")

        # plt.tight_layout()
        # plt.show()
        
        self._clear_calculation_grids()
        
                
    def add_charge_density(self, species_charge_density):
        ng = self.inp.numghosts
        
        if ng != 0:
            self.charge_density += species_charge_density[ng:-ng, ng:-ng]
        
        
    def _clear_calculation_grids(self):
        self.charge_density.fill(0)
        self.potential.fill(0)
    
    
    def solve_poisson(self):
        rhs = - self.charge_density / self.eps0
        
        # Create hall thruster geometry mask
        mask = self._create_hall_thruster_mask(rhs.shape)
        rect = ((self.inp.xlim[0], self.inp.ylim[0]), (self.inp.xlim[1], self.inp.ylim[1]))
        
        # Create boundary segments for different BC types
        boundary_segments = self._create_hall_thruster_boundary_segments(mask, rect)
        
        # Define boundary conditions
        # 1: Anode (300V), 2: Cathode (0V), 3: Walls (0V), 4: Outflow (Neumann)
        boundary_conditions = {
            1: (300.0, "dirichlet"),    # Anode: 300V
            2: (0.0, "dirichlet"),      # Cathode: 0V  
            3: (0.0, "dirichlet"),      # Walls: 0V
            4: (0.0, "neumann_x")       # Outflow: Neumann
        }
        
        solver = solvers.Poisson2DRegion(
            region=mask,
            interior=rhs,
            boundary_conditions=boundary_conditions,
            boundary_segments=boundary_segments,
            rect=rect
        )
        
        self.potential = solver.solve()
    
    
    def _create_hall_thruster_mask(self, shape):
        """Create hall thruster geometry mask with walls at specified locations"""
        ny, nx = shape
        mask = np.ones(shape, dtype=bool)
        
        # Get coordinate mappings
        x_coords = np.linspace(self.inp.xlim[0], self.inp.xlim[1], nx)
        y_coords = np.linspace(self.inp.ylim[0], self.inp.ylim[1], ny)
        
        # Create coordinate grids
        x_grid, y_grid = np.meshgrid(x_coords, y_coords)
        
        # Mask out wall regions: ((0, 0), (0.5, 0.3)) and ((0, 0.6), (0.5, 1))
        # Wall region 1: x ∈ [0, 0.5], y ∈ [0, 0.3]
        wall1_mask = (x_grid >= 0.0) & (x_grid <= 0.5) & (y_grid >= 0.0) & (y_grid <= 0.3)
        
        # Wall region 2: x ∈ [0, 0.5], y ∈ [0.6, 1.0]
        wall2_mask = (x_grid >= 0.0) & (x_grid <= 0.5) & (y_grid >= 0.6) & (y_grid <= 1.0)
        
        # Remove wall regions from computational domain
        mask[wall1_mask] = False
        mask[wall2_mask] = False
        
        return mask
    
    
    def _create_hall_thruster_boundary_segments(self, mask, rect):
        """Create boundary segments for hall thruster with specific BC assignments"""
        from poissonpy.helpers import create_boundary_segments
        from skimage.segmentation import find_boundaries
        
        ny, nx = mask.shape
        x_coords = np.linspace(rect[0][0], rect[1][0], nx)
        y_coords = np.linspace(rect[0][1], rect[1][1], ny)
        
        # Find boundaries
        boundary_mask = find_boundaries(mask, mode="inner")
        boundary_segments = np.zeros_like(mask, dtype=int)
        
        # Get boundary coordinates
        boundary_y, boundary_x = np.where(boundary_mask)
        
        # Map indices to physical coordinates
        x_phys = x_coords[boundary_x]
        y_phys = y_coords[boundary_y]
        
        # Classify boundary segments
        for i, (x, y) in enumerate(zip(x_phys, y_phys)):
            by, bx = boundary_y[i], boundary_x[i]
            
            # Anode: left boundary of channel (x ≈ 0.5, y ∈ [0.3, 0.6])
            if abs(x - 0.5) < 0.05 and 0.3 <= y <= 0.6:
                boundary_segments[by, bx] = 1  # Anode
            
            # Cathode: right boundary (x ≈ 1.0)
            elif abs(x - rect[1][0]) < 0.05:
                boundary_segments[by, bx] = 2  # Cathode
            
            # Walls: boundaries of masked regions
            elif ((0.0 <= x <= 0.5 and (abs(y - 0.0) < 0.05 or abs(y - 0.3) < 0.05)) or
                  (0.0 <= x <= 0.5 and (abs(y - 0.6) < 0.05 or abs(y - 1.0) < 0.05)) or
                  (abs(x - 0.0) < 0.05 and (0.0 <= y <= 0.3 or 0.6 <= y <= 1.0))):
                boundary_segments[by, bx] = 3  # Walls
            
            # Outflow: remaining boundaries (top/bottom of channel)
            else:
                boundary_segments[by, bx] = 4  # Outflow
        
        return boundary_segments
            
    def _compute_electric_field(self):
        """Compute E = -grad(phi) using central differences with ghost cells"""
        ng = self.inp.numghosts
        nx, ny = self.inp.nx_with_ghosts, self.inp.ny_with_ghosts
        
        # Create potential array with ghost cells
        phi_ext = np.zeros((nx, ny))
        phi_ext[ng:-ng, ng:-ng] = self.potential  # Interior domain
        
        # Apply Neumann BCs to ghost cells: ∂φ/∂n = 0
        # This means ghost cells mirror interior values across boundary
        
        # Left boundary ghost cells
        for i in range(ng):
            phi_ext[i, ng:-ng] = phi_ext[2*ng-1-i, ng:-ng]
        
        # Right boundary ghost cells  
        for i in range(ng):
            phi_ext[-1-i, ng:-ng] = phi_ext[-2*ng+i, ng:-ng]
        
        # Bottom boundary ghost cells
        for j in range(ng):
            phi_ext[ng:-ng, j] = phi_ext[ng:-ng, 2*ng-1-j] 
        
        # Top boundary ghost cells
        for j in range(ng):
            phi_ext[ng:-ng, -1-j] = phi_ext[ng:-ng, -2*ng+j]
        
        # Handle corner ghost cells (simple averaging)
        for i in range(ng):
            for j in range(ng):
                # Bottom-left corner
                phi_ext[i, j] = 0.5 * (phi_ext[i, ng] + phi_ext[ng, j])
                # Bottom-right corner  
                phi_ext[i, -1-j] = 0.5 * (phi_ext[i, -1-ng] + phi_ext[ng, -1-j])
                # Top-left corner
                phi_ext[-1-i, j] = 0.5 * (phi_ext[-1-i, ng] + phi_ext[-1-ng, j])
                # Top-right corner
                phi_ext[-1-i, -1-j] = 0.5 * (phi_ext[-1-i, -1-ng] + phi_ext[-1-ng, -1-j])
        
        # Central differences everywhere (interior + boundaries)
        Ex = np.zeros((nx, ny))
        Ey = np.zeros((nx, ny))
        
        # Can now use central differences for all interior points including boundaries
        Ex[1:-1, :] = -(phi_ext[2:, :] - phi_ext[:-2, :]) / (2 * self.dx)
        Ey[:, 1:-1] = -(phi_ext[:, 2:] - phi_ext[:, :-2]) / (2 * self.dy)
        
        # Extract interior domain for storage
        self.E[:, :, 0] = Ex[ng:-ng, ng:-ng]
        self.E[:, :, 1] = Ey[ng:-ng, ng:-ng]
    
    
    def check_E_field(self):
        pass
    