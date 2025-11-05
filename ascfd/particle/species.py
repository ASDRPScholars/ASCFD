from ascfd.params import SpeciesParams
from ascfd.inputs import Inputs
from ascfd.fields.fields import Fields
from ascfd.particle.constants import ParticleConstants
from ascfd.particle.ics import ParticleInitialConditions
from ascfd.particle.bcs import ParticleBoundaryConditions

import numpy as np

class ParticleSpecies:
    def __init__(self, a_inputs: Inputs, params: SpeciesParams, constants: ParticleConstants, fields: Fields, simulation):
        self.inp = a_inputs
        self.params = params
        self.pc = constants
        
        self.fields = fields
        self.dt = None
        self.simulation = simulation 

        self.particles = np.zeros((self.pc.NUMQ + 1, self.inp.n_particles))
        
        self.ics = ParticleInitialConditions(self, self.inp, self.params, self.pc)
        self.bcs = ParticleBoundaryConditions(self, self.inp, self.params, self.pc)
        
        # self.ics.apply_ics()
        
    def update(self):
        self.push()
        self.ionize()
        self.bcs.apply_bcs()
        
    def push(self):
        """
        Leapfrog (velocity Verlet) pusher for charged particles.

        The leapfrog algorithm is a time-centered, symplectic integrator that's:
        - Second-order accurate: O(dt²)
        - Time-reversible: Can integrate backwards in time
        - Energy-conserving: Bounded energy error even for long integration times
        - Simple and fast: Just two steps, no rotations needed

        Algorithm:
        Velocities are staggered at half-timesteps relative to positions:

        Time:     t^n        t^(n+1/2)      t^(n+1)      t^(n+3/2)
        Position: x^n                       x^(n+1)
        Velocity:          v^(n+1/2)                    v^(n+3/2)

        Steps:
        1. Update velocity: v^(n+3/2) = v^(n+1/2) + a^(n+1)*dt
        2. Update position: x^(n+1) = x^n + v^(n+1/2)*dt

        Note: On the first timestep, velocities need to be initialized at t^(-1/2)
        """
        # Get particle charge and mass
        q = self.params.charge
        m = self.params.mass
        dt = self.dt
        ng = self.inp.ng

        Ex = self.field_to_particles(self.fields.E[:, :, 0])
        Ey = self.field_to_particles(self.fields.E[:, :, 1])

        # Compute acceleration: a = (q/m) * E
        ax = (q / m) * Ex
        ay = (q / m) * Ey

        # Get current velocities (at half-timestep: v^(n+1/2))
        vx = self.particles[self.pc.UCOMP]
        vy = self.particles[self.pc.VCOMP]

        # Update velocities: v^(n+3/2) = v^(n+1/2) + a*dt
        vx_new = vx + ax * dt
        vy_new = vy + ay * dt

        # Update positions: x^(n+1) = x^n + v^(n+1/2)*dt
        # NOTE: We use the OLD velocity (at n+1/2) to update position
        # This maintains the time-centering
        self.particles[self.pc.XCOMP] += vx * dt
        self.particles[self.pc.YCOMP] += vy * dt

        # Store new velocities (now at n+3/2)
        self.particles[self.pc.UCOMP] = vx_new
        self.particles[self.pc.VCOMP] = vy_new
        
    def ionize(self):
        n_e = self.simulation.get_species_number_density("e")
        N = self.simulation.get_species_number_density("n")
        k_iz = self.simulation.get_coeffs("k_iz")
        
        n_iz = n_e * N * k_iz * self.dt
        
        iz_particles = self.ics.seed(weight=n_iz)
        
        if self.params.type == "i":
            self.add_particles(iz_particles)
        
    def add_particles(self, new_particles):
        self.particles = np.hstack([self.particles, new_particles])

    def field_to_particles(self, field: np.ndarray, a_particles=None):
        """
        Interpolate field values from cell-centered grid to particle positions.
        Uses bilinear interpolation for smooth field values.

        Args:
            field: 2D array of shape (nx, ny) without ghost cells

        Returns:
            interpolated_values: 1D array of field values at each particle position
        """

        if a_particles is not None:
            particles = a_particles
        else:
            particles = self.particles
            
        x_pos = particles[self.pc.XCOMP]
        y_pos = particles[self.pc.YCOMP]

        x_grid = (x_pos - self.inp.xlim[0]) / self.inp.dx
        y_grid = (y_pos - self.inp.ylim[0]) / self.inp.dy

        ix = self.ix(x_pos).astype(int)
        iy = self.iy(y_pos).astype(int)

        # Clamp indices to prevent out-of-bounds access along edges
        nx, ny = field.shape
        ix = np.clip(ix, 0, nx - 2)
        iy = np.clip(iy, 0, ny - 2)

        wx = x_grid - ix
        wy = y_grid - iy

        # bilinear interpolation
        interpolated_values = (
            field[ix,     iy    ] * (1 - wx) * (1 - wy) +  # Bottom-left
            field[ix + 1, iy    ] * wx       * (1 - wy) +  # Bottom-right
            field[ix,     iy + 1] * (1 - wx) * wy       +  # Top-left
            field[ix + 1, iy + 1] * wx       * wy           # Top-right
        )

        return interpolated_values

    def ix(self, x_pos):
        """Converts physical x coordinate to grid coordinate."""
        return np.floor((x_pos - self.inp.xlim[0]) / self.inp.dx).astype(int)
    
    def iy(self, y_pos):
        """Converts physical y coordinate to grid coordinate."""
        return np.floor((y_pos - self.inp.ylim[0]) / self.inp.dy).astype(int)
    
    def particles_to_field(self, particle_quantity):
        """
        Deposit particle quantities to grid using bilinear weighting (area weighting).

        This is the inverse operation of field_to_particles. It uses the same bilinear
        interpolation weights to ensure conservation and symmetry.

        For density:    Σ(weights) at cell (i,j) gives particle density contribution
        For momentum:   Σ(weights * velocity) gives momentum density
        For energy:     Σ(weights * 0.5*m*v²) gives energy density

        Args:
            particle_quantity: 1D array of values at each particle position
                             (e.g., particle weights, or weight*velocity, etc.)

        Returns:
            field: 2D array of shape (nx, ny) with deposited quantities (no ghost cells)
        """
        # Initialize empty field
        field = np.zeros((self.inp.nx, self.inp.ny))

        # Get particle positions
        x_pos = self.particles[self.pc.XCOMP]
        y_pos = self.particles[self.pc.YCOMP]

        # Convert to grid coordinates
        x_grid = (x_pos - self.inp.xlim[0]) / self.inp.dx
        y_grid = (y_pos - self.inp.ylim[0]) / self.inp.dy

        # Find base cell indices
        ix = self.ix(x_pos)
        iy = self.iy(y_pos)

        # Calculate interpolation weights (same as field_to_particles)
        wx = x_grid - ix
        wy = y_grid - iy

        # Clip indices to valid range
        ix = np.clip(ix, 0, self.inp.nx - 2)
        iy = np.clip(iy, 0, self.inp.ny - 2)

        # Deposit to the 4 surrounding cells using bilinear weights
        # This is called "area weighting" or "cloud-in-cell" (CIC) scheme

        # Bottom-left cell (ix, iy)
        np.add.at(field, (ix, iy), particle_quantity * (1 - wx) * (1 - wy))

        # Bottom-right cell (ix+1, iy)
        np.add.at(field, (ix + 1, iy), particle_quantity * wx * (1 - wy))

        # Top-left cell (ix, iy+1)
        np.add.at(field, (ix, iy + 1), particle_quantity * (1 - wx) * wy)

        # Top-right cell (ix+1, iy+1)
        np.add.at(field, (ix + 1, iy + 1), particle_quantity * wx * wy)

        return field

    def get_number_density(self):
        """
        Compute particle number density field (particles per unit volume).

        Returns:
            density: 2D array of shape (nx, ny) with number density [1/m²] (or [1/m³] in 3D)
        """
        # Each particle represents a weight (number of physical particles)
        weights = self.particles[self.pc.NUMQ]  # Assuming weight is stored at NUMQ

        # Deposit to grid
        density_integrated = self.particles_to_field(weights)

        # Divide by cell volume to get density
        cell_area = self.inp.dx * self.inp.dy
        density = density_integrated / cell_area

        return density

    def get_velocity(self):
        """
        Compute mean velocity field by depositing momentum and dividing by mass.

        Returns:
            vx_mean, vy_mean, vz_mean: 2D arrays of mean velocity components [m/s]
        """
        m = self.params.mass
        weights = self.particles[self.pc.NUMQ]

        # Get velocities
        vx = self.particles[self.pc.UCOMP]
        vy = self.particles[self.pc.VCOMP]

        # Deposit momentum: mass * velocity * weight
        momentum_vx = self.particles_to_field(m * vx * weights)
        momentum_vy = self.particles_to_field(m * vy * weights)

        # Deposit mass
        mass_deposited = self.particles_to_field(m * weights)

        # Mean velocity = momentum / mass
        # Add small epsilon to avoid division by zero in empty cells
        epsilon = 1e-30
        vx_mean = momentum_vx / (mass_deposited + epsilon)
        vy_mean = momentum_vy / (mass_deposited + epsilon)

        return vx_mean, vy_mean

    def x(self, ix):
        """Converts grid x coordinate to physical coordinate."""
        return (ix+0.5) * self.inp.dx + self.inp.xlim[0]

    def y(self, iy):
        """Converts grid y coordinate to physical coordinate."""
        return (iy+0.5) * self.inp.dy + self.inp.ylim[0]
