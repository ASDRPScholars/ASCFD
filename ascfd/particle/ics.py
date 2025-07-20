from ascfd.particle.constants import ParticleConstants
from ascfd.inputs import Inputs
import numpy as np

class ParticleInitialConditions:
    def __init__(self, particles, a_inputs: Inputs):
        self.pc = ParticleConstants()
        self.inp = a_inputs
        self.particles = particles
        
        
    def apply_ics(self):
        
        # TODO: MAKE IT DIRECTLY MUTATE SPECIES.PARTICLES IDK WHY IT DOESN RN
        if self.inp.particle_ics == "origin":
            np.copyto(self.particles, self.origin())
        if self.inp.particle_ics == "random":
            np.copyto(self.particles, self.random())
        if self.inp.particle_ics == "random_left_wall":
            np.copyto(self.particles, self.random_left_wall())
    
        if self.inp.particle_ics == "random_particles_no_condition":
            self.particles = self.random()
            
        print(self.particles[self.pc.XCOMP])
        
    def origin(self):
        ic_particles = np.zeros_like(self.particles)
        
        return ic_particles
            
            
    def random(self):
        ic_particles = np.zeros_like(self.particles)
        
        for n in range (self.inp.n_particles):
            random_x, random_y = np.random.rand(), np.random.rand()
        
            ic_particles[self.pc.XCOMP, n] = random_x
            ic_particles[self.pc.YCOMP, n] = random_y
            
            # TODO: add random maxwellian velocity distribution logic
        
        print(ic_particles[self.pc.XCOMP])
        return ic_particles
    
    
    def random_left_wall(self):
        ic_particles = np.zeros_like(self.particles)

        for n in range (self.inp.n_particles):
            x, y = 0, np.random.rand()
            
            ic_particles[self.pc.XCOMP, n] = x
            ic_particles[self.pc.YCOMP, n] = y
        
        return ic_particles
    