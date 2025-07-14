import ascfd.ics as ics

class FieldManager:
    def __init__(self, a_inputs):
        self.inp = a_inputs
        self.apply_ics()
        
    def apply_ics(self):
        if self.inp.system == "euler2d":
            if self.inp.ics == "hall_thruster_rz":
                self.grid_data.fill_grid(ics.hall_thruster_rz)