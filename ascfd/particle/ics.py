from ascfd.particle.constants import ParticleConstants
from ascfd.logistics.inputs import Inputs
import numpy as np

class ParticleInitialConditions:
    def __init__(self, particles, a_inputs: Inputs):
        self.pc = ParticleConstants()
        self.inp = a_inputs
        self.particles = particles
        
        
    def apply_ics(self):
        # if self.inp.particle_ics == "random_particles_no_condition":
        self.particles = self.random_particles_no_condition()
        
        # TODO: MAKE IT DIRECTLY MUTATE SPECIES.PARTICLES IDK WHY IT DOESN RN
        return self.random_particles_no_condition()
        print(self.particles[self.pc.XCOMP])
            
            
    def random_particles_no_condition(self):
        ic_particles = np.zeros_like(self.particles)
        
        for n in range (self.inp.n_particles):
            random_x, random_y = np.random.rand(), np.random.rand()
        
            ic_particles[self.pc.XCOMP, n] = random_x
            ic_particles[self.pc.YCOMP, n] = random_y
            
            # TODO: add random maxwellian velocity distribution logic
        
        print(ic_particles[self.pc.XCOMP])
        return ic_particles
    