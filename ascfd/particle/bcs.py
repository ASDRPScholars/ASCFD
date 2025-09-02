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
        # weight = self.params.density * self.inp.dx * self.inp.dy / self.inp.n_ppc
        
        weight = 1000000
        
        kB = 1
        v_th = np.sqrt(2 * kB * self.params.temperature / self.params.mass)
        
        new_particles = []
        
        # Create n_ppc particles per boundary cell to match target density
        for j in range(self.inp.ny):
            for p in range(self.inp.n_ppc):  # Critical fix: create n_ppc particles per cell
                particle_data = np.zeros(self.pc.NUMQ + 1)
                
                x_rand = np.random.uniform(0, self.inp.xlim[1] / 10)
                y_rand = np.random.uniform(0, self.inp.ylim[1])
                            
                R1, R2 = np.random.rand(2)
                R3, R4 = np.random.rand(2)

                vx = v_th * np.sqrt(-1 * np.log(R1)) * np.cos(2 * np.pi * R2)
                vy = v_th * np.sqrt(-1 * np.log(R1)) * np.sin(2 * np.pi * R2)
                vz = v_th * np.sqrt(-1 * np.log(R3)) * np.cos(2 * np.pi * R4)
                
                particle_data[self.pc.XCOMP] = x_rand
                particle_data[self.pc.YCOMP] = y_rand
                particle_data[self.pc.UCOMP] = 1 #vx + 20
                particle_data[self.pc.VCOMP] = 1 # (vy)
                particle_data[WEIGHT] = weight
                
                if self.pc.WCOMP < self.pc.NUMQ:
                    print("!N! ADDED", particle_data)
                    particle_data[self.pc.WCOMP] = vz
                    
                new_particles.append(particle_data)
                self.particle_species.add_particle(particle_data)
                    
        # Use efficient particle addition instead of np.hstack
        # self.particle_species.add_particles(particle_data)

        print("!!ADD PARTICLE!! FOR", self.params.type, len(new_particles))
    
    
    def remove_particles(self):
        pass