from ascfd.inputs import Inputs



class Constants:

    def __init__(self, inputs: Inputs):

        # Default ratio of specific heats (gamma)
        self.gamma = 1.4
        self.system = inputs.system

        # Define constants and variables for the Euler 2D system
        if inputs.system == "euler2D":
            
            #primitive variables (Density, X-Velocity, Y-Velocity, Pressure)
            self.RHOCOMP = 0 # Density
            self.UCOMP = 1 # X-component of velocity
            self.VCOMP = 2 # Y-component of velocity
            self.PCOMP = 3 # Pressure
 
             #conserved variables (Density, Momentum-X, Momentum-Y, Energy)
            self.RHOCOMP = 0 # Density
            self.MUCOMP = 1 # X-component of momentum
            self.MVCOMP = 2 # Y-component of momentum
            self.ECOMP = 3 # Total energy

            # number of variables in the system
            self.NUMQ = 4

            self.system = "euler2D"

            self.gamma = inputs.gammas[0] #gammas


            self.variable_names = ["Density", "X-Velocity", "Y-Velocity", "Pressure"]

            self.NS = 1 

        else:
            #error for unsupported systems
            raise RuntimeError(f"system in inputs file is not supported: {inputs.system}")