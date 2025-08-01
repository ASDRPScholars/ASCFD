from ascfd.inputs import Inputs
import numpy as np

class FieldInitialConditions:
    def __init__(self, E_field, B_field, a_inputs: Inputs):
        self.E_field = E_field
        self.B_field = B_field
        self.inp = a_inputs
        
    # TODO: THIS IS WHERE WE IMPLEMENT THE MAGNETIC FIELD INITIAL CONDITION
    def apply_B_ics(self):
        if self.inp.system == "euler2d":
            if self.inp.B_ics == "guillaume":
                return self.guillaume()
                print("FROM IC", self.B_field[:, :, 1])
            else:
                raise RuntimeError("[FIELD] ICS not valid.")
                
                
    def apply_E_ics(self):
        if self.inp.system == "euler2d":
            pass
        
        
    def guillaume(self):
        ic_grid = np.zeros_like(self.B_field)
        
        B_max = 0.02 # 0.237 -> 0.02 per claude
        x_c = 0.5
        sigma = 0.05
        
        x = np.linspace(self.inp.xlim[0], self.inp.xlim[1], self.inp.nx)
        
        # Debug prints
        print(f"x range: {x.min()} to {x.max()}")
        print(f"x_c: {x_c}")
        print(f"sigma: {sigma}")
        print(f"x shape: {x.shape}")
        print(f"ic_grid[:,:,1] shape: {ic_grid[:, :, 1].shape}")
        
        # Calculate the Gaussian
        gaussian_1d = B_max * np.exp(-((x - x_c) / sigma) ** 2)
        print(f"Gaussian 1D max: {gaussian_1d.max()}")
        print(f"Gaussian 1D at x_c: {B_max * np.exp(-((x_c - x_c) / sigma) ** 2)}")
        
        # Broadcast to 2D
        print(np.shape(ic_grid))
        print(np.shape(gaussian_1d))
        ic_grid[:, :, 1] = gaussian_1d[:, np.newaxis]
        
        print(f"ic_grid max after assignment: {ic_grid.max()}")
        
        return ic_grid
    