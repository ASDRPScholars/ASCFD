from ascfd.particle.constants import ParticleConstants
from ascfd.inputs import Inputs
from ascfd.plasma_refs import PlasmaReferences

import numpy as np

class ParticleBoundaryConditions:
    def __init__(self, a_species, a_inputs: Inputs, params, constants: ParticleConstants):
        self.species = a_species
        self.particles = self.species.particles
        self.inp = a_inputs
        self.params = params
        self.pc = constants
        
    def apply_bcs(self):
        #TODO: readd boundaries that aren't at the bottom corner lol
        # self.apply_inflow_lo()
        self.clean_particles()

    def clean_particles(self):
        """
        Remove particles that have moved outside the domain boundaries.

        Particles are removed if their positions are outside the physical domain or if weight is 0.
        This prevents interpolation errors and handles particle absorption at boundaries.
        """
        # Get physical positions for all particles
        x_pos = self.particles[self.pc.XCOMP]
        y_pos = self.particles[self.pc.YCOMP]

        # Check if particles are within the physical domain bounds
        # Particles exactly on the lower boundary will be removed to prevent accumulation
        valid_mask = (
            (x_pos > self.inp.xlim[0]) & (x_pos < self.inp.xlim[1]) &
            (y_pos > self.inp.ylim[0]) & (y_pos < self.inp.ylim[1]) &
            (self.particles[self.pc.WEIGHT] > 0)
        )

        # remove columns where mask is False
        self.particles = self.particles[:, valid_mask]
        
    def apply_inflow_lo(self):
        """Apply inflow boundary condition with proper n_ppc seeding

        Returns:
            np.ndarray: Array of shape (self.pc.NUMQ+1, n_inject) containing particle data
        """
        WEIGHT = self.pc.NUMQ

        v_n = self.inp.v_n

        # Boundary layer dimensions
        nx = (v_n * self.inp.dt) / self.inp.dx  # cells in x-direction for boundary layer
        ny = self.inp.ny  # full domain in y

        # Number of computational particles to inject
        n_inject = int(self.inp.n_ppc * nx * ny)

        # Weight per particle to match desired flux
        weight = (self.inp.flux_n * self.inp.ylim[1] * self.inp.dt) / n_inject

        # Initialize particle array: (NUMQ+1, n_inject)
        inflow_particles = np.zeros((self.pc.NUMQ + 1, n_inject))

        # Generate random positions for all particles at once
        x_max = self.inp.xlim[1] * (nx / self.inp.nx)
        inflow_particles[self.pc.XCOMP, :] = np.random.uniform(0, x_max, n_inject)
        inflow_particles[self.pc.YCOMP, :] = np.random.uniform(0, self.inp.ylim[1], n_inject)

        # Set velocities and weights
        inflow_particles[self.pc.UCOMP, :] = v_n
        inflow_particles[WEIGHT, :] = weight

        # Optionally add thermal velocities:
        # R1, R2 = np.random.rand(2, n_inject)
        # R3, R4 = np.random.rand(2, n_inject)
        # particle_data[self.pc.UCOMP, :] += v_th * np.sqrt(-np.log(R1)) * np.cos(2 * np.pi * R2)
        # particle_data[self.pc.VCOMP, :] = v_th * np.sqrt(-np.log(R1)) * np.sin(2 * np.pi * R2)
        # particle_data[self.pc.WCOMP, :] = v_th * np.sqrt(-np.log(R3)) * np.cos(2 * np.pi * R4)

        self.species.add_particles(inflow_particles)
    
    
    def remove_particles(self):
        pass