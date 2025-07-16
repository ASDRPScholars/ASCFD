from ascfd.particle.constants import ParticleConstants
from ascfd.particle.ics import ParticleInitialConditions
from ascfd.logistics.inputs import Inputs
from ascfd.species.params import SpeciesParams
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
        charge_density = self.get_charge_density()
        self.fields.update_E(charge_density)
    
    
    def get_charge_density(self):
        pass
    
    
    def check_particles(self):
        pass
    