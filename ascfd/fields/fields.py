import ascfd.fluid.ics as ics
import ascfd.inputs as Inputs
import numpy as np

# TODO: write all the logic lol
class Fields:
    def __init__(self, a_inputs: Inputs):
        self.inp = a_inputs
        self.apply_B_ics()
        self.apply_E_ics()
        
        self.E = np.zeros((self.inp.nx + 2 * self.inp.numghosts, self.inp.nx + 2 * self.inp.numghosts))
        self.B = np.zeros((self.inp.nx + 2 * self.inp.numghosts, self.inp.nx + 2 * self.inp.numghosts))
        
    
    # TODO: THIS IS WHERE WE IMPLEMENT THE MAGNETIC FIELD INITIAL CONDITION
    def apply_B_ics(self):
        if self.inp.system == "euler2d":
            if self.inp.E_ics == "hall_thruster_rz":
                self.grid.fill_grid(ics.hall_thruster_rz)
                
                
    def apply_E_ics(self):
        if self.inp.system == "euler2d":
            if self.inp.B_ics == "hall_thruster_rz":
                self.grid.fill_grid(ics.hall_thruster_rz)
                
                
    def update_E(self, charge_density):
        pass
    
    
    def solve_poisson(self):
        pass
    
    
    def check_E_field(self):
        pass
    