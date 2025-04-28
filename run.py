from ascfd.inputs import Inputs
from ascfd.simulation import Simulation

inp = Inputs("problems/field_loop.ini")
s = Simulation(inp)
s.run()