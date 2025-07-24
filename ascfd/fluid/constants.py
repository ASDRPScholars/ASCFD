from ascfd.inputs import Inputs

class FluidConstants:

    def __init__(self, a_inputs: Inputs, dimensions=2):

        # Default ratio of specific heats (gamma)
        self.gamma = 1.4
        self.g = 9.81
        self.system = a_inputs.system
        
        if a_inputs.system == "hallthruster_rz":
            self.RHOCOMP = 0 # Density
            self.UCOMP = 1 # X-component of velocity
            self.VCOMP = 2 # Y-component of velocity
            self.WCOMP = 3
            self.PCOMP = 4 # Pressure
            
             #conserved variables (Density, Momentum-X, Momentum-Y, Energy)
            self.MUCOMP = 1 # X-component of momentum
            self.MVCOMP = 2 # Y-component of momentum
            self.MWCOMP = 3 # Y-component of momentum
            self.ECOMP = 4 # Total energy (thermal + kinetic)

            # number of variables in the system
            self.NUMQ = 5

            self.system = "hallthruster_rz"

            self.gamma = a_inputs.gammas[0] #gammas

            self.variable_names = ["Density", "X-Velocity", "Y-Velocity", "Z-Velocity", "Pressure"]


        # Define constants and variables for the Euler 2D system
        elif a_inputs.system == "euler2d":
            
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

            self.gamma = a_inputs.gammas[0] #gammas


            self.variable_names = ["Density", "X-Velocity", "Y-Velocity", "Pressure"]

            self.NS = 1 
            
        elif a_inputs.system == "mhd2d":
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
            self.gamma = a_inputs.gammas[0] # Use first gamma if provided
            
            self.variable_names = ["Density", "X-Velocity", "Y-Velocity", "Pressure", "Bx", "By"]
            
            # Number of species? (Assuming 1 for now for MHD)
            self.NS = 1

        else:
            #error for unsupported systems
            raise RuntimeError(f"system in inputs file is not supported: {a_inputs.system}")