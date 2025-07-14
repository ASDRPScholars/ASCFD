from ascfd.constants import Constants
from ascfd.inputs import Inputs
from ascfd.
import numpy as np

class Problem:
    def __init__(self, a_inputs: Inputs):
        self.c = Constants(a_inputs)
        self.inp = a_inputs
        
        match self.inp.ics:
            case "diagonal_advection":
                self.fluid_ic = 
        
    
    def initialize_fluid(self, grid):
        
        for var in range(self.c.NUMQ):
            self.grid[var] = (self.meshX, self.meshY, var)
        
    def initialize_particles():
        
    def initialize_fields():
        
        
    
        