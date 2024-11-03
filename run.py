from ascfd.inputs import Inputs
from ascfd.simulation import Simulation

inp = Inputs("problems/diag_advection.ini")
s = Simulation(inp)
s.run()