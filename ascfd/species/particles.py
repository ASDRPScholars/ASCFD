from ascfd.grid import Grid2D
from ascfd.constants import Constants
from ascfd.inputs import Inputs
from ascfd.species.params import SpeciesParams
from ascfd.fields import Fields

import numpy as np

class ParticleSpecies:
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields):
        self.fields = Fields(a_inputs)
        
        self.inp = a_inputs
        self.params = params
        self.dt = None
        
        self.x = np.zeros((self.inp.n_particles, 2))
        self.v = np.zeros((self.inp.n_particles, 2))
        
        
    # TODO: ics_particles.py stuff goes in here
    def apply_ics(self):
        pass
    
    
    # TODO: particles.py stuff goes in here
    def update(self):
        charge_density = self.get_charge_density()
        self.fields.update_E(charge_density)
    
    
    def get_charge_density(self):
        pass
    
    
    def check_particles(self):
        pass
    