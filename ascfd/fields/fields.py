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
        
        self.charge_density = np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        self.potential = np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))

        self.dx = (self.inp.xlim[1] - self.inp.xlim[0]) / (self.inp.nx - 1)
        self.dy = (self.inp.ylim[1] - self.inp.ylim[0]) / (self.inp.ny - 1)
        
        self.eps0 = 8.854e-12  # TODO: CHECK — Permittivity of free space
                
                
    def update_E(self):
        print("UPDATE E")
        self.solve_poisson()
        self._compute_electric_field()
        
        print("THIS IS CHARGE DENSITY TAKEN IN BY POISSON:", self.charge_density)
        print("NEW CALCULATED EX IS:", self.E[:, :, 0])
        print("NEW CALCULATED EY IS:", self.E[:, :, 1])

        fig, axs = plt.subplots(2, 2, figsize=(15, 15))  # 1 row, 3 columns
        axs = axs.flatten()

        # Plot charge density
        im0 = axs[0].imshow(self.charge_density, cmap="coolwarm")
        axs[0].set_title("Charge Density")
        fig.colorbar(im0, ax=axs[0])

        # Plot Ex
        im1 = axs[1].imshow(self.E[:, :, 0], cmap="coolwarm")
        axs[1].set_title("Electric Field Ex")
        fig.colorbar(im1, ax=axs[1])

        # Plot Ey
        im2 = axs[2].imshow(self.E[:, :, 1], cmap="coolwarm")
        axs[2].set_title("Electric Field Ey")
        fig.colorbar(im2, ax=axs[2])
        
        # Plot E
        im3 = axs[3].imshow(np.sqrt(self.E[:, :, 0]**2 + self.E[:, :, 1]**2), cmap="coolwarm")
        axs[3].set_title("Electric Field Magnitude")
        fig.colorbar(im3, ax=axs[3])
        
        plt.tight_layout()
        plt.show()
        
        ### 3D PLOT
        # Compute E magnitude
        E_mag = np.sqrt(self.E[5:-5, 5:-5, 0]**2 + self.E[5:-5, 5:-5, 1]**2)

        # Create meshgrid for X and Y
        nx, ny = E_mag.shape
        x = np.arange(nx)
        y = np.arange(ny)
        X, Y = np.meshgrid(y, x)  # careful: meshgrid uses (cols, rows) order

        # Create figure
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')

        # Plot the surface
        surf = ax.plot_surface(X, Y, E_mag, cmap='coolwarm')

        # Add colorbar and labels
        fig.colorbar(surf, ax=ax, shrink=0.5, aspect=10)
        ax.set_title("ours")
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("|E|")

        plt.tight_layout()
        plt.show()
        
        self._clear_calculation_grids()
        
                
    def add_charge_density(self, species_charge_density):
        self.charge_density += species_charge_density
        
        
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
        """Create hall thruster geometry mask - True where we solve Poisson equation (fluid domain)"""
        ny, nx = shape
        mask = np.ones(shape, dtype=bool)  # Start with entire domain
        
        # Define wall bounds using integer slicing
        x_end = round(nx * 0.5)
        y1_end = round(ny * 0.4)
        y2_start = round(ny * 0.6)

        # Remove wall regions from solution domain (walls are solid, no solution needed there)
        mask[:y1_end, :x_end] = False    # wall1 (bottom-left)
        mask[y2_start:, :x_end] = False  # wall2 (top-left)
        
        return mask
    
    
    def _create_hall_thruster_boundary_segments(self, mask, rect):
        """Create boundary segments for hall thruster with specific BC assignments"""
        from poissonpy.helpers import create_boundary_segments
        from skimage.segmentation import find_boundaries
        
        ny, nx = mask.shape
        
        # Use same discrete boundaries as mask creation to ensure consistency
        x_end = round(nx * 0.5)
        y1_end = round(ny * 0.4)
        y2_start = round(ny * 0.6)
        
        # Find boundaries
        boundary_mask = find_boundaries(mask, mode="inner")
        boundary_segments = np.zeros_like(mask, dtype=int)
        
        # Get boundary coordinates in index space
        boundary_y, boundary_x = np.where(boundary_mask)
        
        # Classify boundary segments using discrete indices
        for i in range(len(boundary_y)):
            by, bx = boundary_y[i], boundary_x[i]
            
            # Anode: left boundary of channel entrance (x = 0, y ∈ [y1_end, y2_start])
            if bx == 0 and y1_end <= by < y2_start:
                boundary_segments[by, bx] = 1  # Anode
            
            # Cathode: right boundary of entire domain (x = nx-1)
            elif bx == nx - 1:
                boundary_segments[by, bx] = 2  # Cathode
            
            # Walls: interfaces between channel and wall regions
            elif ((bx == x_end - 1 and by < y1_end) or           # bottom wall interface
                  (bx == x_end - 1 and by >= y2_start) or        # top wall interface
                  (by == y1_end and bx < x_end) or               # bottom channel wall
                  (by == y2_start - 1 and bx < x_end)):          # top channel wall
                boundary_segments[by, bx] = 3  # Walls
            
            # Outflow: top and bottom boundaries of main channel
            else:
                boundary_segments[by, bx] = 4  # Outflow
        
        return boundary_segments
            
    def _compute_electric_field(self):
        """Compute E = -grad(phi) using central differences with ghost cells"""
        ng = self.inp.numghosts
        nx, ny = self.inp.nx_with_ghosts, self.inp.ny_with_ghosts
        
        # Central differences only for interior points to avoid boundary artifacts
        Ex = np.zeros((nx, ny))
        Ey = np.zeros((nx, ny))
        
        # Use central differences only for interior points in both directions
        Ex[1:-1, 1:-1] = -(self.potential[2:, 1:-1] - self.potential[:-2, 1:-1]) / (2 * self.dx)
        Ey[1:-1, 1:-1] = -(self.potential[1:-1, 2:] - self.potential[1:-1, :-2]) / (2 * self.dy)
        
        # Extract interior domain for storage
        self.E[:, :, 0] = Ex[ng:-ng, ng:-ng]
        self.E[:, :, 1] = Ey[ng:-ng, ng:-ng]
    
    
    def check_E_field(self):
        pass
    