from ascfd.inputs import Inputs
from ascfd.plasma_refs import PlasmaReferences

class FluidConstants:

    def __init__(self, a_inputs: Inputs, dimensions=2):

        # Default ratio of specific heats (gamma)
        self.gamma = 1.4
        self.g = 9.81
        self.system = a_inputs.system
        
        # Initialize plasma normalization for Hall thruster simulations
        # Using electron parameters as reference
        # self.plasma_refs = PlasmaReferences(
        #     n0=1e18,        # 1e18 m⁻³ - typical Hall thruster plasma density
        #     T0=1000.0,      # 1000 K - electron temperature
        #     species_mass=9.1e-31,   # electron mass
        #     species_charge=1.6e-19  # elementary charge
        # )

        # Define constants and variables for the Euler 2D system
        if a_inputs.system == "euler2d":
            
            #primitive variables (Density, X-Velocity, Y-Velocity, Z-Velocity, Pressure)
            self.RHOCOMP = 0 # Density
            self.UCOMP = 1 # X-component of velocity
            self.VCOMP = 2 # Y-component of velocity
            self.WCOMP = 3 # Z-component of velocity (azimuthal/out-of-plane)
            self.PCOMP = 4 # Pressure
 
             #conserved variables (Density, Momentum-X, Momentum-Y, Momentum-Z, Energy)
            self.MUCOMP = 1 # X-component of momentum
            self.MVCOMP = 2 # Y-component of momentum
            self.MWCOMP = 3 # Z-component of momentum (azimuthal/out-of-plane)
            self.ECOMP = 4 # Total energy (thermal + kinetic)

            # number of variables in the system
            self.NUMQ = 5

            self.system = "euler2d"

            self.gamma = a_inputs.gammas[0] #gammas
            
            self.k_B = 8.617e-5
            self.eps_0 = 8.85418782e-12

            self.variable_names = ["Density", "X-Velocity", "Y-Velocity", "Z-Velocity", "Pressure"]

            self.NS = 1 
        
        elif a_inputs.system == "quasineutral":
            
            #primitive variables (Density, X-Velocity, Y-Velocity, Z-Velocity, Pressure)
            self.NCOMP = 0 # Number Density
            self.JXCOMP = 1 # X-component of velocity
            self.JYCOMP = 2 # Y-component of velocity
            self.JZCOMP = 3 # Z-component of velocity (azimuthal/out-of-plane)
            self.TCOMP = 4 # Temperature

            # number of variables in the system
            self.NUMQ = 5

            self.system = "quasineutral"

            self.gamma = a_inputs.gammas[0] #gammas
            
            self.k_B = 8.617e-5
            self.eps_0 = 8.85418782e-12

            self.variable_names = ["Number Density", "X-Velocity", "Y-Velocity", "Z-Velocity", "Temperature"]

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