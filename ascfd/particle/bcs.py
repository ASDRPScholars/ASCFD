from ascfd.particle.constants import ParticleConstants
from ascfd.inputs import Inputs
from ascfd.plasma_refs import PlasmaReferences

import numpy as np

class ParticleBoundaryConditions:
    def __init__(self, particle_species, a_inputs: Inputs, params):
        self.particle_species = particle_species
        self.inp = a_inputs
        self.ref = PlasmaReferences()
        self.pc = ParticleConstants()
        self.params = params
        
        
    def apply_bcs(self):
        #TODO: readd boundaries that aren't at the bottom corner lol
        self.apply_inflow_lo()
        # self.remove_particles()

        
    def apply_inflow_lo(self):
        """Apply inflow boundary condition with proper n_ppc seeding"""
        WEIGHT = self.pc.NUMQ
    
        v_n = self.inp.v_n
        
        # Boundary layer dimensions
        nx = (v_n * self.inp.dt) / self.inp.dx  # cells in x-direction for boundary layer
        ny = self.inp.ny  # full domain in y
        
        # Number of computational particles to inject
        n_inject = self.inp.N_ppc * nx * ny
        
        # Weight per particle to match desired flux
        weight = (self.inp.flux_n * self.inp.ylim[1] * self.inp.dt) / n_inject
        
        for p in range(int(n_inject)): 
            particle_data = np.zeros(self.pc.NUMQ + 1)
            
            x_rand = np.random.uniform(0, self.inp.xlim[1] * (nx / self.inp.nx))
            y_rand = np.random.uniform(0, self.inp.ylim[1])
                        
            # R1, R2 = np.random.rand(2)
            # R3, R4 = np.random.rand(2)

            # vx = v_th * np.sqrt(-1 * np.log(R1)) * np.cos(2 * np.pi * R2)
            # vy = v_th * np.sqrt(-1 * np.log(R1)) * np.sin(2 * np.pi * R2)
            # vz = v_th * np.sqrt(-1 * np.log(R3)) * np.cos(2 * np.pi * R4)
            
            particle_data[self.pc.XCOMP] = x_rand
            particle_data[self.pc.YCOMP] = y_rand
            
            particle_data[self.pc.UCOMP] = v_n

            particle_data[WEIGHT] = weight
            
            self.particle_species.add_particle(particle_data)
    
    
    def remove_particles(self):
        pass