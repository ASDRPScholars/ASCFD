from ascfd.logistics.inputs import Inputs
from ascfd.simulation import Simulation

inp = Inputs("problems/orszag_tang.ini")
s = Simulation(inp)
s.run()