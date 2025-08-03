from ascfd.particle.constants import ParticleConstants
from ascfd.inputs import Inputs

import numpy as np

class ParticleBoundaryConditions:
    def __init__(self, particle_species, a_inputs: Inputs, params):
        self.particle_species = particle_species
        self.inp = a_inputs
        self.pc = ParticleConstants()
        self.params = params
        
        
    def apply_bcs(self):
        self.apply_inflow_lo()
        self.remove_particles()
        
        
    def apply_inflow_lo(self):
        """Apply inflow boundary condition with proper n_ppc seeding"""
        WEIGHT = self.pc.NUMQ
        weight = self.params.density * self.inp.dx * self.inp.dy / self.inp.n_ppc
        
        kB = 1
        v_th = np.sqrt(2 * kB * self.params.temperature / self.params.mass)
        
        # For realistic neutral injection, use appropriate flow velocity
        # Scale injection velocity with mass ratio to maintain reasonable flow rates
        mass_ratio_correction = np.sqrt(self.params.mass / 100)  # Compensate for mass change
        neutral_injection_speed = 20 / mass_ratio_correction  # Maintain effective injection rate
        
        # Create n_ppc particles per boundary cell to match target density
        for j in range(self.inp.ny):
            for p in range(self.inp.n_ppc):  # Critical fix: create n_ppc particles per cell
                particle_data = np.zeros(self.pc.NUMQ + 1)
                
                x_offset = np.random.uniform(0.1, 0.5)
                y_offset = np.random.uniform(-0.4999, 0.5)
                            
                R1, R2 = np.random.rand(2)
                R3, R4 = np.random.rand(2)

                vx = v_th * np.sqrt(-1 * np.log(R1)) * np.cos(2 * np.pi * R2)
                vy = v_th * np.sqrt(-1 * np.log(R1)) * np.sin(2 * np.pi * R2)
                vz = v_th * np.sqrt(-1 * np.log(R3)) * np.cos(2 * np.pi * R4)
                
                particle_data[self.pc.XCOMP] = x_offset * self.inp.dx
                particle_data[self.pc.YCOMP] = (j + y_offset - 1) * self.inp.dy
                particle_data[self.pc.UCOMP] = vx + neutral_injection_speed
                particle_data[self.pc.VCOMP] = vy
                particle_data[WEIGHT] = weight
                
                if self.pc.WCOMP < self.pc.NUMQ:
                    particle_data[self.pc.WCOMP] = vz
                    
                # Use efficient particle addition instead of np.hstack
                self.particle_species.add_particle(particle_data)
    
    
    def remove_particles(self):
        pass