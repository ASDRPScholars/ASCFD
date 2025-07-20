from ascfd.inputs import Inputs

class FluidConstants:

    def __init__(self, a_inputs: Inputs):

        # Default ratio of specific heats (gamma)
        self.gamma = 1.4
        self.g = 9.81
        self.system = a_inputs.system

        # Define constants and variables for the Euler 2D system
        if a_inputs.system == "euler2d":
            
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

        elif a_inputs.system == "hall_thruster":
            # Electron fluid variables for quasineutral + quasistatic solver
            self.N_E_COMP = 0      # electron density
            self.E_PAR_COMP = 1    # parallel electric field
            self.E_PERP_COMP = 2   # perpendicular electric field  
            self.J_E_PAR_COMP = 3  # parallel electron current
            self.J_E_PERP_COMP = 4 # perpendicular electron current
            self.T_E_COMP = 5      # electron temperature
            
            self.NUMQ = 6 # Number of electron variables
            
            self.system = "hall_thruster"
            
            self.variable_names = ["Electron_Density", "E_Parallel", "E_Perpendicular", 
                                 "J_e_Parallel", "J_e_Perpendicular", "Electron_Temperature"]
            
            self.NS = 1

        else:
            #error for unsupported systems
            raise RuntimeError(f"system in inputs file is not supported: {a_inputs.system}")