from ascfd.logistics.inputs import Inputs
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
- why is movie being made when its off?
    fixed.
- get rid of grid object in flux. this will prevent any automatic python speedups.
- convert to vectorized, but leave a simpler copy for understanding
- cache/compile JIT....
- then profile and see where the bottlenecks are once the low hanging fruit is done.

Just vectorized version:
- 250x250 grid, 2 ghost cells, 20 time steps, 0.5 cfl, RK1, no movie
=> Simulation completed in 6.68 seconds.

- 1000x1000 grid, 2 ghost cells, 5 time steps, 0.5 cfl, RK1, no movie
=> Simulation completed in 30.16 seconds.


JIT Version:

Alright i think i did the low hanging fruit but kind of destroyed the code. Lets retest.
- 250x250 grid, 2 ghost cells, 20 time steps, 0.5 cfl, RK1
=> Simulation completed in 9.56 seconds

- 1000x1000 grid, 2 ghost cells, 5 time steps, 0.5 cfl, RK1, no movie
=> Simulation completed in 32.79 seconds.

The startup is slow but the time stepping is much faster.



'''