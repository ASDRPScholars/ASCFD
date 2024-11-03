from ascfd.inputs import Inputs



class Constants:

    def __init__(self, inputs: Inputs):

        self.gamma = 1.4
        self.system = inputs.system

        if inputs.system == "euler2D":

            #prim
            self.RHOCOMP = 0
            self.UCOMP = 1
            self.VCOMP = 2
            self.PCOMP = 3
 
            #cons
            self.RHOCOMP = 0
            self.MUCOMP = 1
            self.MVCOMP = 2
            self.ECOMP = 3

            self.NUMQ = 4

            self.system = "euler2D"

            self.gamma = inputs.gammas[0] #gammas


            self.variable_names = ["Density", "X-Velocity", "Y-Velocity", "Pressure"]

            self.NS = 1 

        else:
            raise RuntimeError(f"system in inputs file is not supported: {inputs.system}")