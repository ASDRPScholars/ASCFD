
import numpy as np

import time


start_time = time.time()


x = np.linspace(0,1,1000000)


# for i in range(len(x)):
#     x[i] = x[i] + 1


x = x+1

end_time = time.time()
print(f"Simulation completed in {end_time - start_time:.2f} seconds")
