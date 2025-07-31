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
        new_particles = []
    
        for j in range(self.inp.ny):
            particle_data = np.zeros(self.pc.NUMQ + 1)
            
            weight = self.params.density * self.inp.dx * self.inp.dy / self.inp.n_ppc
            WEIGHT = self.pc.NUMQ
            
            kB = 1
            v_th = np.sqrt(2 * kB * self.params.temperature / self.params.mass)
            
            x_offset = np.random.uniform(0.1, 0.5)
            y_offset = np.random.uniform(-0.4999, 0.5)
                        
            R1, R2 = np.random.rand(2)
            R3, R4 = np.random.rand(2)

            vx = v_th * np.sqrt(-1 * np.log(R1)) * np.cos(2 * np.pi * R2)
            vy = v_th * np.sqrt(-1 * np.log(R1)) * np.sin(2 * np.pi * R2)
            vz = v_th * np.sqrt(-1 * np.log(R3)) * np.cos(2 * np.pi * R4)
            
            particle_data[self.pc.XCOMP] = x_offset * self.inp.dx
            particle_data[self.pc.YCOMP] = (j + y_offset) * self.inp.dy
            particle_data[self.pc.UCOMP] = vx + 20
            particle_data[self.pc.VCOMP] = vy
            particle_data[WEIGHT] = weight
            
            if self.pc.WCOMP < self.pc.NUMQ:
                particle_data[self.pc.WCOMP] = vz
                
            new_particles.append(particle_data.reshape(-1, 1))
              
        self.particle_species.particles = np.hstack([self.particle_species.particles] + new_particles)  
        # return np.hstack([self.particles] + new_particles)
    
    
    def remove_particles(self):
        pass