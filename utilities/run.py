import sys
import os

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)

if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from ascfd.inputs import Inputs
from ascfd.simulation import Simulation

inp = Inputs("problems/orszag_tang.ini")
s = Simulation(inp)
s.run()
