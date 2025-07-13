from ascfd.grid import Grid2D
from ascfd.constants import Constants

import numpy as np


class ParticleSpecies:
    def __init__(self, params, n_particles):
        self.params = params
        self.x = np.zeros((n_particles, 2))
        self.v = np.zeros((n_particles, 2))
        
    def update(self):
        pass
    
    def get_charge_density(self):
        pass
    