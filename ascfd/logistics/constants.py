from ascfd.logistics.inputs import Inputs



class Constants:

    def __init__(self, inputs: Inputs):

        # Default ratio of specific heats (gamma)
        self.gamma = 1.4
        self.g = 9.81
        self.system = inputs.system

        # Define constants and variables for the Euler 2D system
        if inputs.system == "euler2d":
            
            #primitive variables (Density, X-Velocity, Y-Velocity, Pressure)
            self.RHOCOMP = 0 # Density
            self.UCOMP = 1 # X-component of velocity
            self.VCOMP = 2 # Y-component of velocity
            self.PCOMP = 3 # Pressure
 
             #conserved variables (Density, Momentum-X, Momentum-Y, Energy)
            self.MUCOMP = 1 # X-component of momentum
            self.MVCOMP = 2 # Y-component of momentum
            self.ECOMP = 3 # Total energy (thermal + kinetic)

            # number of variables in the system
            self.NUMQ = 4

            self.system = "euler2d"

            self.gamma = inputs.gammas[0] #gammas


            self.variable_names = ["Density", "X-Velocity", "Y-Velocity", "Pressure"]

            self.NS = 1 
            
        elif inputs.system == "mhd2d":
            # Primitive variables (Density, u, v, p, Bx, By)
            self.RHOCOMP = 0 # Density
            self.UCOMP = 1   # X-velocity
            self.VCOMP = 2   # Y-velocity
            self.PCOMP = 3   # Pressure
            self.BXCOMP = 4  # X-magnetic field
            self.BYCOMP = 5  # Y-magnetic field

            # Conserved variables (rho, rho*u, rho*v, E, Bx, By)
            self.MUCOMP = 1  # X-momentum
            self.MVCOMP = 2  # Y-momentum
            self.ECOMP = 3   # Total Energy (thermal + kinetic + magnetic)
            
            self.NUMQ = 6 # Number of variables
            
            self.system = "mhd2d"
            
            # Assuming gamma is still relevant for MHD pressure equation
            self.gamma = inputs.gammas[0] # Use first gamma if provided
            
            self.variable_names = ["Density", "X-Velocity", "Y-Velocity", "Pressure", "Bx", "By"]
            
            # Number of species? (Assuming 1 for now for MHD)
            self.NS = 1

        else:
            #error for unsupported systems
            raise RuntimeError(f"system in inputs file is not supported: {inputs.system}")