from ascfd.inputs import Inputs

class FieldInitialConditions:
    def __init__(self, a_inputs: Inputs):
        self.inp = a_inputs
        
    # TODO: THIS IS WHERE WE IMPLEMENT THE MAGNETIC FIELD INITIAL CONDITION
    def apply_B_ics(self):
        if self.inp.system == "hallthruster_rz":
            pass
                
                
    def apply_E_ics(self):
        if self.inp.system == "hallthruster_rz":
            pass