from ascfd.inputs import Inputs
from ascfd.simulation import Simulation
import time


start_time = time.time()


inp = Inputs("problems/timing_diag_advect.ini")
s = Simulation(inp)
s.run()

end_time = time.time()
print(f"Simulation completed in {end_time - start_time:.2f} seconds")


'''
NOTES TIMING:

main/
- 250x250 grid, 2 ghost cells, 20 time steps, 0.5 cfl, RK1
=> Simulation completed in 22.70 seconds

- 1000x1000 grid, 2 ghost cells, 5 time steps, 0.5 cfl, RK1
=> Simulation completed in 86.93 seconds


This is crazy slow. TODO:
- why is moving being made when its off?
- why are we outputting every timestep when its off in the inputs file?
- convert to vectorized, but leave a simpler copy for understanding
- cache....
- then profile and see where the bottlenecks are once the low hanging fruit is done.


'''