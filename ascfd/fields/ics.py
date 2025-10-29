from ascfd.inputs import Inputs
import numpy as np

class FieldInitialConditions:
    def __init__(self, E_field, B_field, a_inputs: Inputs):
        self.E_field = E_field
        self.B_field = B_field
        self.inp = a_inputs
        
    # TODO: THIS IS WHERE WE IMPLEMENT THE MAGNETIC FIELD INITIAL CONDITION
    def apply_B_ics(self):
        if self.inp.e_system in ["euler2d", "quasineutral"]:
            if self.inp.B_ics == "guillaume":
                return np.ones_like(self.B_field) * 0.02
                return self.guillaume()
                print("FROM IC", self.B_field[:, :, 1])
            else:
                raise RuntimeError("[FIELD] ICS not valid.")
                
                
    def apply_E_ics(self):
        return np.ones_like(self.E_field) * 1e4
    
        if self.inp.e_system == "euler2d":
            pass
        
        
        
    def guillaume(self):
        ic_grid = np.zeros_like(self.B_field)
        
        # -> MIT LECTURE
        B_max = self.inp.B_max # Tesla
        x_c = 0.6 * self.inp.xlim[1]
        sigma = 0.05 * self.inp.xlim[1]
        
        x = np.linspace(self.inp.xlim[0], self.inp.xlim[1], self.inp.nx)
    
        gaussian_1d = B_max * np.exp(-((x - x_c) / sigma) ** 2)
        
        ic_grid[:, :, 1] = gaussian_1d[:, np.newaxis]
        
        return ic_grid
    