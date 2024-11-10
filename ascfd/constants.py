from ascfd.inputs import Inputs



class Constants:

    def __init__(self, inputs: Inputs):

        self.gamma = 1.4 # Set the default gamma value (specific heat ratio for ideal gas)
        self.system = inputs.system # Retrieve the system type from inputs
        self.do_gravity = inputs.do_gravity # Flag indicating whether gravity should be considered
        
        if self.do_gravity:
            self.gravity = 9.81  # Define gravity
            self.gravity_vector = [0.0, -self.gravity]  # Gravity acts downward
        else:
            self.gravity = 0.0  # No gravity
            self.gravity_vector = [0.0, 0.0]  # Gravity is zero 
       
        if inputs.system == "euler2D":

            #prim
            self.RHOCOMP = 0 # Index for density
            self.UCOMP = 1 # Index for x-velocity
            self.VCOMP = 2 # Index for y-velocity
            self.PCOMP = 3 # Index for pressure 
 
            #cons
            self.RHOCOMP = 0 # Index for density
            self.MUCOMP = 1 # Index for x-momentum
            self.MVCOMP = 2 # Index for y-momentum
            self.ECOMP = 3 # Index for energy

            # Total number of variables in the system
            self.NUMQ = 4

            self.system = "euler2D"

            # Set the gamma value from the `inputs.gammas`
            self.gamma = inputs.gammas[0]

            # Define the names of the variables
            self.variable_names = ["Density", "X-Velocity", "Y-Velocity", "Pressure"]

            self.NS = 1 

            #Raise an error for invalid inputs
        else:
            raise RuntimeError(f"system in inputs file is not supported: {inputs.system}")