from ascfd.particle.constants import ParticleConstants
from ascfd.particle.ics import ParticleInitialConditions
from ascfd.inputs import Inputs
from ascfd.params import SpeciesParams
from ascfd.fields.fields import Fields

import numpy as np

class ParticleSpecies:
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields):
        self.pc = ParticleConstants()
        self.fields = Fields(a_inputs)
        
        self.inp = a_inputs
        self.params = params
        self.dt = None
        
        self.particles = np.zeros((self.pc.NUMQ, self.inp.n_particles))
        
        self.ics = ParticleInitialConditions(self.particles, self.inp)
        
        self.particles = self.ics.apply_ics()
        print("PARTICLES X POS INIT:", self.particles[self.pc.XCOMP])
    
    
    # TODO: particles.py stuff goes in here
    def update(self):
        
        # this acc doesn't fit very well with the new multispecies structure lol...
        if self.inp.particle_flow_type == "passive":
            pass
        
        elif self.inp.particle_flow_type == "multispecies":
            pass
        
        if self.params.type == "n":
            self.ionize()
        
        charge_density = self.get_charge_density()
        self.fields.update_E(charge_density)
        
    
    def ionize(self):
        for n in range (self.inp.n_particles):
            # make methods to find these values depending on position (could use grid also similar to how particle cell calculation was originally done)
            electron_density = ... 
            neutral_density = ...

            ionization_rate = ... # another method to calculuate

            # statistics now !!!!

            p_ionize = 1 - np.exp(-ionization_rate * self.dt)

            rand = np.random.rand()

            if rand > p_ionize: # quick exit case: we will NOT ionize
                continue
        
            self.particles.
            particles.pop(neutral_particle_index)

            # create new ion in its place with correct values etc...

            # create_particle()
    
    def get_charge_density(self):
        pass
    
    
    def check_particles(self):
        pass
    
    
    