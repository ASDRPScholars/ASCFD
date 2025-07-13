from ascfd.grid import Grid2D
from ascfd.constants import Constants

import numpy as np


class FluidSpecies:
    def __init__(self, params, a_inputs):
        self.inp = a_inputs
        self.c = Constants(a_inputs)
        self.params = params
        self.grid_data = Grid2D(self.inp.xlim, self.inp.ylim, self.inp.nx, self.inp.ny, self.inp.numghosts, self.c.NUMQ)
        
    def update(self):
        pass
    
    def get_charge_density(self):
        pass
    