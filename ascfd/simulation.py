from ascfd.grid import Grid2D
from ascfd.constants import Constants
import ascfd.ics as ics
import glob
import matplotlib.animation as animation
import matplotlib.ticker as ticker
import copy

import sys
from ascfd.flux import Flux


from ascfd.euler import Euler

from ascfd.bcs import BoundaryConditions

import numpy as np
import matplotlib.pyplot as plt
import os


class Simulation:
    def __init__(self, a_inputs):
        self.inp = a_inputs
        self.c = Constants(a_inputs)
        self.euler = Euler(self.c)

        self.grid = Grid2D(self.inp.xlim, self.inp.ylim, self.inp.nx,
                           self.inp.ny, self.inp.numghosts, self.c.NUMQ)
        self.bcs = BoundaryConditions(
            self.grid, self.inp.bcs_lo, self.inp.bcs_hi)
        self.flux = Flux(self.c, self.inp.flux)

        self.applyICS()

        self.bcs.apply_bcs()
        self.grid.check_grid(self.c)
        # setup initial time to be the starting time from the inputs file.
        # The starting timestep will always be 0.
        self.t = self.inp.t0
        self.timestepNum = 0

        # -1 is no output. Always output ICs if we are outputting.
        if self.inp.output_freq >= 0:
            self.output()

    def run(self):
        while (self.t < self.inp.t_finish) and self.timestepNum < self.inp.nt:
            print(f"Timestep: {self.timestepNum}, Current time: {self.t}")

            self.bcs.apply_bcs()

            self.grid.assert_variable_type("prim")

            # Determine timestep dt based on CFL condition
            if self.inp.system == "euler2D":
                density = self.grid.grid[self.c.RHOCOMP]
                pressure = self.grid.grid[self.c.PCOMP]
                u = self.grid.grid[self.c.UCOMP]
                v = self.grid.grid[self.c.VCOMP]
                # Ensure pressure and density are positive before sqrt
                pressure = np.maximum(pressure, 1e-12)
                density = np.maximum(density, 1e-12)
                a = np.sqrt(self.c.gamma * pressure / density)  # Sound speed
                max_speed_x = np.max(np.abs(u) + a)
                max_speed_y = np.max(np.abs(v) + a)
                # More robust estimate
                max_speed = max(max_speed_x, max_speed_y)

            elif self.inp.system == "mhd2d":
                density = self.grid.grid[self.c.RHOCOMP]
                pressure = self.grid.grid[self.c.PCOMP]
                u = self.grid.grid[self.c.UCOMP]
                v = self.grid.grid[self.c.VCOMP]
                Bx = self.grid.grid[self.c.BXCOMP]
                By = self.grid.grid[self.c.BYCOMP]

                # Ensure pressure and density are positive
                pressure = np.maximum(pressure, 1e-12)
                density = np.maximum(density, 1e-12)

                a = np.sqrt(self.c.gamma * pressure / density)  # Sound speed
                # Alfven speed squared components
                ca_sq_x = Bx**2 / density
                ca_sq_y = By**2 / density
                ca_sq_tot = ca_sq_x + ca_sq_y

                # Fast magnetosonic speed squared (cf^2)
                # cf^2 = 0.5 * ( (a^2 + ca_tot^2) + sqrt( max( (a^2 + ca_tot^2)^2 - 4*a^2*ca_x^2 , 0.0 ) ) )
                term_under_sqrt = (a**2 + ca_sq_tot)**2 - 4 * a**2 * ca_sq_x
                cf_sq_x = 0.5 * ((a**2 + ca_sq_tot) +
                                 np.sqrt(np.maximum(term_under_sqrt, 0.0)))
                cf_x = np.sqrt(cf_sq_x)

                # Use ca_sq_y for y-direction cf
                term_under_sqrt = (a**2 + ca_sq_tot)**2 - 4 * a**2 * ca_sq_y
                cf_sq_y = 0.5 * ((a**2 + ca_sq_tot) +
                                 np.sqrt(np.maximum(term_under_sqrt, 0.0)))
                cf_y = np.sqrt(cf_sq_y)

                # Max signal speed is max(|u|+cf_x, |v|+cf_y)
                max_signal_x = np.max(np.abs(u) + cf_x)
                max_signal_y = np.max(np.abs(v) + cf_y)
                max_speed = max(max_signal_x, max_signal_y)

            else:
                raise RuntimeError(
                    f"System {self.inp.system} not supported for dt calculation.")

            # Calculate dt, ensuring it doesn't overshoot t_finish
            dt = min(self.inp.cfl * min(self.grid.dx, self.grid.dy) /
                     max_speed, self.inp.t_finish - self.t)
            if dt <= 0:
                raise ValueError(
                    f"Calculated dt is zero or negative ({dt}). Check simulation parameters or state.")

            # EMBEDDED BOUNDARIES — MODULARIZE LATER

            lower_bound = 0
            upper_bound = 1
            step = 0.01

            old_vertices = [(0.4990133642141358, 0.03139525976465669), (0.49605735065723894, 0.06266661678215213), (0.49114362536434436, 0.09369065729286231), (0.48429158056431554, 0.1243449435824274), (0.47552825814757677, 0.1545084971874737), (0.4648882429441257, 0.184062276342339), (0.45241352623300973, 0.21288964578253636), (0.4381533400219318, 0.24087683705085766), (0.42216396275100754, 0.26791339748949833), (0.4045084971874737, 0.29389262614623657), (0.38525662138789457, 0.3187119948743449), (0.3644843137107058, 0.34227355296434436), (0.3422735529643443, 0.3644843137107058), (0.3187119948743448, 0.3852566213878946), (0.2938926261462365, 0.4045084971874737), (0.2679133974894983, 0.42216396275100754), (0.24087683705085758, 0.43815334002193185), (0.21288964578253633, 0.4524135262330098), (0.18406227634233893, 0.46488824294412573), (0.15450849718747373, 0.47552825814757677), (0.12434494358242737, 0.48429158056431554), (0.09369065729286226, 0.49114362536434436), (0.06266661678215213, 0.49605735065723894), (0.03139525976465665, 0.4990133642141358), (-8.040613248383183e-17, 0.5), (-0.0313952597646567, 0.4990133642141358), (-0.06266661678215218, 0.4960573506572389), (-0.09369065729286241, 0.4911436253643443), (-0.12434494358242743, 0.48429158056431554), (-0.15450849718747378, 0.47552825814757677), (-0.184062276342339, 0.4648882429441257), (-0.21288964578253636, 0.45241352623300973), (-0.24087683705085772, 0.43815334002193174), (-0.26791339748949844, 0.4221639627510075), (-0.2938926261462365, 0.4045084971874737), (-0.3187119948743449, 0.3852566213878946), (-0.34227355296434436, 0.3644843137107057), (-0.36448431371070583, 0.34227355296434425), (-0.3852566213878947, 0.31871199487434476), (-0.40450849718747367, 0.2938926261462366), (-0.42216396275100754, 0.26791339748949833), (-0.4381533400219318, 0.2408768370508576), (-0.4524135262330098, 0.21288964578253625), (-0.46488824294412573, 0.18406227634233888), (-0.4755282581475768, 0.15450849718747356), (-0.48429158056431554, 0.12434494358242741), (-0.49114362536434436, 0.09369065729286229), (-0.49605735065723894, 0.06266661678215205), (-0.4990133642141358, 0.03139525976465657), (-0.5, -1.6081226496766366e-16),
                            (-0.4990133642141358, -0.03139525976465667), (-0.49605735065723894, -0.06266661678215214), (-0.4911436253643443, -0.09369065729286238), (-0.48429158056431554, -0.12434494358242751), (-0.47552825814757677, -0.15450849718747386), (-0.4648882429441256, -0.18406227634233915), (-0.45241352623300973, -0.21288964578253633), (-0.43815334002193174, -0.2408768370508577), (-0.4221639627510075, -0.2679133974894984), (-0.4045084971874736, -0.2938926261462367), (-0.3852566213878945, -0.318711994874345), (-0.3644843137107058, -0.34227355296434436), (-0.3422735529643443, -0.36448431371070583), (-0.31871199487434476, -0.3852566213878947), (-0.2938926261462366, -0.40450849718747367), (-0.26791339748949816, -0.42216396275100765), (-0.24087683705085763, -0.4381533400219318), (-0.21288964578253608, -0.4524135262330099), (-0.1840622763423389, -0.46488824294412573), (-0.15450849718747378, -0.47552825814757677), (-0.12434494358242722, -0.4842915805643156), (-0.09369065729286231, -0.49114362536434436), (-0.06266661678215187, -0.49605735065723894), (-0.031395259764656604, -0.4990133642141358), (-9.184850993605148e-17, -0.5), (0.03139525976465686, -0.4990133642141358), (0.06266661678215212, -0.49605735065723894), (0.09369065729286256, -0.4911436253643443), (0.12434494358242747, -0.48429158056431554), (0.15450849718747361, -0.4755282581475768), (0.18406227634233913, -0.4648882429441256), (0.2128896457825363, -0.4524135262330098), (0.24087683705085786, -0.4381533400219317), (0.2679133974894984, -0.4221639627510075), (0.29389262614623685, -0.4045084971874735), (0.318711994874345, -0.3852566213878945), (0.3422735529643443, -0.3644843137107058), (0.36448431371070594, -0.34227355296434414), (0.3852566213878946, -0.3187119948743448), (0.4045084971874739, -0.2938926261462363), (0.42216396275100765, -0.26791339748949816), (0.4381533400219318, -0.24087683705085766), (0.45241352623300984, -0.2128896457825361), (0.46488824294412573, -0.18406227634233893), (0.4755282581475769, -0.1545084971874734), (0.4842915805643156, -0.12434494358242724), (0.49114362536434436, -0.09369065729286234), (0.49605735065723894, -0.0626666167821519), (0.4990133642141358, -0.03139525976465663)]
            vertices = [(0.749506682107068, 0.5156976298823284), (0.7480286753286195, 0.5313333083910761), (0.7455718126821722, 0.5468453286464312), (0.7421457902821578, 0.5621724717912137), (0.7377641290737884, 0.5772542485937369), (0.7324441214720628, 0.5920311381711695), (0.7262067631165049, 0.6064448228912682), (0.7190766700109659, 0.6204384185254288), (0.7110819813755038, 0.6339566987447491), (0.7022542485937369, 0.6469463130731183), (0.6926283106939473, 0.6593559974371724), (0.6822421568553529, 0.6711367764821722), (0.6711367764821722, 0.6822421568553529), (0.6593559974371724, 0.6926283106939473), (0.6469463130731182, 0.7022542485937369), (0.6339566987447491, 0.7110819813755038), (0.6204384185254288, 0.7190766700109659), (0.6064448228912682, 0.7262067631165049), (0.5920311381711695, 0.7324441214720628), (0.5772542485937369, 0.7377641290737884), (0.5621724717912137, 0.7421457902821578), (0.5468453286464311, 0.7455718126821722), (0.5313333083910761, 0.7480286753286195), (0.5156976298823284, 0.749506682107068), (0.49999999999999994, 0.75), (0.48430237011767163, 0.749506682107068), (0.46866669160892394, 0.7480286753286194), (0.4531546713535688, 0.7455718126821722), (0.4378275282087863, 0.7421457902821578), (0.42274575140626314, 0.7377641290737884), (0.4079688618288305, 0.7324441214720628), (0.39355517710873184, 0.7262067631165049), (0.37956158147457114, 0.7190766700109659), (0.36604330125525075, 0.7110819813755037), (0.35305368692688177, 0.7022542485937369), (0.34064400256282756, 0.6926283106939473), (0.3288632235178278, 0.6822421568553528), (0.31775784314464706, 0.6711367764821721), (0.30737168930605263, 0.6593559974371723), (0.29774575140626314, 0.6469463130731183), (0.28891801862449623, 0.6339566987447491), (0.28092332998903413, 0.6204384185254288), (0.2737932368834951, 0.6064448228912681), (0.26755587852793716, 0.5920311381711695), (0.2622358709262116, 0.5772542485937368), (0.25785420971784223, 0.5621724717912137), (0.2544281873178278, 0.5468453286464311), (0.2519713246713805, 0.5313333083910761), (0.2504933178929321, 0.5156976298823283), (0.25, 0.49999999999999994),
                        (0.2504933178929321, 0.4843023701176717), (0.2519713246713805, 0.46866669160892394), (0.25442818731782785, 0.4531546713535688), (0.25785420971784223, 0.43782752820878623), (0.2622358709262116, 0.4227457514062631), (0.26755587852793716, 0.4079688618288304), (0.2737932368834951, 0.39355517710873184), (0.28092332998903413, 0.37956158147457114), (0.2889180186244963, 0.3660433012552508), (0.2977457514062632, 0.35305368692688166), (0.30737168930605274, 0.3406440025628275), (0.3177578431446471, 0.3288632235178278), (0.3288632235178278, 0.31775784314464706), (0.3406440025628276, 0.30737168930605263), (0.35305368692688166, 0.29774575140626314), (0.3660433012552509, 0.2889180186244962), (0.3795615814745712, 0.28092332998903413), (0.39355517710873195, 0.27379323688349505), (0.4079688618288305, 0.26755587852793716), (0.42274575140626314, 0.2622358709262116), (0.4378275282087864, 0.25785420971784223), (0.45315467135356885, 0.2544281873178278), (0.46866669160892405, 0.2519713246713805), (0.4843023701176717, 0.2504933178929321), (0.49999999999999994, 0.25), (0.5156976298823285, 0.2504933178929321), (0.5313333083910761, 0.2519713246713805), (0.5468453286464313, 0.25442818731782785), (0.5621724717912138, 0.25785420971784223), (0.5772542485937369, 0.2622358709262116), (0.5920311381711696, 0.26755587852793716), (0.6064448228912681, 0.2737932368834951), (0.6204384185254289, 0.28092332998903413), (0.6339566987447491, 0.2889180186244963), (0.6469463130731185, 0.29774575140626325), (0.6593559974371725, 0.30737168930605274), (0.6711367764821722, 0.3177578431446471), (0.6822421568553529, 0.32886322351782793), (0.6926283106939473, 0.34064400256282756), (0.702254248593737, 0.3530536869268819), (0.7110819813755038, 0.3660433012552509), (0.7190766700109659, 0.3795615814745712), (0.7262067631165049, 0.39355517710873195), (0.7324441214720628, 0.4079688618288305), (0.7377641290737884, 0.4227457514062633), (0.7421457902821578, 0.4378275282087864), (0.7455718126821722, 0.45315467135356885), (0.7480286753286195, 0.46866669160892405), (0.749506682107068, 0.4843023701176717)]

            # for vertex in old_vertices:
            #     x = (vertex[0] * 0.5) + 0.5
            #     y = (vertex[1] * 0.5) + 0.5
            #     vertices.append((x, y))

            print(vertices)
            print(len(old_vertices), len(vertices))

            def is_inside_polygon(points, point):
                n = len(points)
                inside = False
                p1x, p1y = points[0]
                for i in range(n+1):
                    p2x, p2y = points[i % n]
                    if point[1] > min(p1y, p2y):
                        if point[1] <= max(p1y, p2y):
                            if point[0] <= max(p1x, p2x):
                                if p1y != p2y:
                                    xinters = (point[1] - p1y) * \
                                        (p2x - p1x) / (p2y - p1y) + p1x
                                if p1x == p2x or point[0] <= xinters:
                                    inside = not inside
                    p1x, p1y = p2x, p2y
                return inside

            def fluid_vector_through_time(embedded_boundary_vector, velocity_vector, vertices, lower_bound, upper_bound, step):
                # Calculate normal and parallel velocity vectors
                V_normal, V_parallel = transform_vector(
                    embedded_boundary_vector, velocity_vector)
                # Update the velocity vector and dot product using the previously transformed vectors
                velocity_vector_next = V_parallel
                for t in np.arange(lower_bound, upper_bound, step):
                    point = (velocity_vector[0]*t, velocity_vector[1]*t)
                    # Need to switch to test for outside
                    if not is_inside_polygon(vertices, point):
                        # Calculate normal and parallel velocity vectors
                        velocity_vector_next, V_parallel = transform_vector(
                            embedded_boundary_vector, velocity_vector)
                        # Update velocity_vector_next to encorporate previous point
                        # Update the velocity vector
                        velocity_vector_next = V_parallel
                        V_normal = 0
                return velocity_vector_next

            def transform_vector(embedded_boundary_vector, velocity_vector):
                # Squares embedded boundary
                embedded_boundary_squared = embedded_boundary_vector[0] ** 2 + \
                    embedded_boundary_vector[1] ** 2
                # Calculates the dot product of velocity and embedded boundary vevtor
                dot_product = velocity_vector[0] * embedded_boundary_vector[0] + \
                    velocity_vector[1] * embedded_boundary_vector[1]
                V_parallel = (embedded_boundary_vector[0] * dot_product / embedded_boundary_squared,
                              embedded_boundary_vector[1] * dot_product / embedded_boundary_squared)      # Calculates normal vector
                V_normal = (velocity_vector[0] - V_parallel[0],
                            velocity_vector[1] - V_parallel[1])
                return V_normal, V_parallel

            def find_embedded_boundary_vector(vertices, lower_bound, upper_bound, step):
                grid_size = 50
                x_vals = np.linspace(lower_bound, upper_bound, grid_size)
                y_vals = np.linspace(lower_bound, upper_bound, grid_size)
                # create a grid in which each cell contains the embedded boundary vector for that cell
                vector_map = np.zeros((grid_size, grid_size, 2))
                for i, x in enumerate(x_vals):
                    for j, y in enumerate(y_vals):
                        # finds the point that is the center of the cell
                        cell_center = np.array([x + step/2, y + step/2])
                        # distance to the outside point on the right
                        min_distance1 = float('inf')
                        # distance to the outside point on the left
                        min_distance2 = float('inf')
                        # nearest point outside to the right
                        nearest_point1 = (0, 0)
                        # nearest point outside to the left
                        nearest_point2 = (0, 0)
                        for v in vertices:
                            # find mag. of distance from the vertex to the center of the cell
                            distance = np.linalg.norm(
                                np.array(v) - cell_center)
                            # split into v[0]>x and v[0]<x
                            if (v[0] > x):
                                # check is it's outside the cell and less than previous distance
                                if distance > step / 2 and distance < min_distance1:
                                    min_distance1 = distance
                                    nearest_point1 = np.array(v)
                            if (v[0] < x):
                                # check is it's outside the cell and less than previous distance
                                if distance > step / 2 and distance < min_distance2:
                                    min_distance2 = distance
                                    nearest_point2 = np.array(v)
                        # vector from center of cell to nearest point
                        embedded_boundary_vector = nearest_point1 - nearest_point2
                        vector_map[i, j] = embedded_boundary_vector
                return vector_map

            vector_map = find_embedded_boundary_vector(
                vertices, lower_bound, upper_bound, step)

            if self.inp.timeStepper == "RK1":
                # returns numerical flux and conservative variables at interface
                self.grid.assert_variable_type("prim")
                consU, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus = self.flux.getFlux(
                    self.grid.grid, self.grid.Nx, self.grid.Ny, self.grid.Nghost)

                # Start with the current conservative variables
                U_new = np.copy(consU)
                primU = self.euler.cons_to_prim(U_new)

                # FLUID UPDATE
                for i in range(self.grid.Nghost, self.grid.Nx + self.grid.Nghost):
                    for j in range(self.grid.Nghost, self.grid.Ny + self.grid.Nghost):
                        # TODO: change hard coded
                        point = np.array([0.01*i, 0.01*j])
                        inside_polygon = is_inside_polygon(vertices, point)
                        for icomp in range(self.c.NUMQ):
                            if inside_polygon:
                                velocity_vector = np.array(
                                    [primU[1, i, j], primU[2, i, j]])
                                if icomp == 1:
                                    U_new[1, i, j] = U_new[0, i, j] * fluid_vector_through_time(
                                        vector_map[i][j], velocity_vector, vertices, lower_bound, upper_bound, step)[0]
                                if icomp == 2:
                                    U_new[2, i, j] = U_new[0, i, j] * fluid_vector_through_time(
                                        vector_map[i][j], velocity_vector, vertices, lower_bound, upper_bound, step)[1]
                                else:
                                    U_new[icomp, i, j] = 1
                            else:
                                U_new[icomp, i, j] = consU[icomp, i, j] - (
                                    (dt / self.grid.dx) * (numFluxX_plus[icomp, i, j] - numFluxX_minus[icomp, i, j]) +
                                    (dt / self.grid.dy) *
                                    (numFluxY_plus[icomp, i, j] -
                                     numFluxY_minus[icomp, i, j])
                                )

                # Then, update flow field based on the embedded boundary.

                # take a step in particles

            else:
                raise RuntimeError("Timestepping method not supported.")

            # self.grid.plot()
            # Update the grid with the new conservative variables
            # self.grid.set(U_new)
            self.grid.grid = self.euler.cons_to_prim(U_new)
            self.grid.variables = "prim"

            # Convert back to primitive variables
            # self.grid.transform(self.euler.cons_to_prim, "prim")

            self.bcs.apply_bcs()

            # assert np.all(np.isfinite(self.grid.grid)), f"Invalid values in grid at timestep {self.timestepNum}"
            # assert np.all(self.grid.grid[self.c.PCOMP] > 0), f"Negative pressure detected at timestep {self.timestepNum}"

            self.timestepNum += 1
            self.t += dt

            # always output the last timestep.
            if (self.timestepNum % self.inp.output_freq == 0) or (self.timestepNum == self.inp.nt-1):
                self.output()

            # DEBUG
            # self.grid.plot()
            self.grid.check_grid(self.c)

        if self.inp.make_movie:
            self.generate_movie()

        print("SUCCESS!")
        return self.grid

    def plot(self):
        if not os.path.exists(self.inp.output_dir):
            os.makedirs(self.inp.output_dir)

        if self.inp.system == "euler2D":
            fig, axs = plt.subplots(3, 1, figsize=(10, 15))
            axs[0].scatter(
                self.grid.x, self.grid.grid[self.c.RHOCOMP, :], c="black")
            axs[0].set_ylabel("Density")

            axs[1].scatter(
                self.grid.x, self.grid.grid[self.c.UCOMP, :],  c="black")
            axs[1].set_ylabel("Velocity")

            axs[2].scatter(
                self.grid.x, self.grid.grid[self.c.PCOMP, :],  c="black")
            axs[2].set_ylabel("Pressure")

        elif self.inp.system == "mhd2d":
            fig, axs = plt.subplots(3, 1, figsize=(10, 15))
            axs[0].scatter(
                self.grid.x, self.grid.grid[self.c.RHOCOMP, :], c="black")
            axs[0].set_ylabel("Density")

            axs[1].scatter(
                self.grid.x, self.grid.grid[self.c.UCOMP, :],  c="black")
            axs[1].set_ylabel("Velocity")

            axs[2].scatter(
                self.grid.x, self.grid.grid[self.c.PCOMP, :],  c="black")
            axs[2].set_ylabel("Pressure")

            axs[3].scatter(
                self.grid.x, self.grid.grid[self.c.B_XCOMP, :],  c="black")
            axs[3].set_ylabel("Magnetic Field")

        axs[0].set_title(f"Time: {self.t:.4f}")
        plt.savefig(
            f"{self.inp.output_dir}/plot_dt{str(self.timestepNum).zfill(6)}")
        plt.close()

    def applyICS(self):

        if self.inp.system == "euler2D":
            if self.inp.ics == "diagonal_advection":
                self.grid.fill_grid(ics.diagonal_advection_2d)
            elif self.inp.ics == "kelvin_helmholtz":
                self.grid.fill_grid(ics.kelvin_helmholtz_2d)
            elif self.inp.ics == "double_mach_reflection":
                self.grid.fill_grid(ics.double_mach_reflection_2d)
            elif self.inp.ics == "riemann_problem":
                self.grid.fill_grid(ics.riemann_2d)
            else:
                raise RuntimeError("[FLUID] ICS not valid.")

        elif self.inp.system == "mhd2d":
            if self.inp.ics == "orszag_tang":
                self.grid.fill_grid(ics.orszag_tang_2d)

        else:
            raise RuntimeError("[FLUID] ICS not valid.")

    def applyParticles(self):
        """Particle Setup"""
        # I assume this should function similar to apply ics?
        # check if self.inp.particle_ic = ...
        pass

    def output(self):
        # Ensure the base output directory exists
        os.makedirs(self.inp.output_dir, exist_ok=True)

        # Ensure the frames subdirectory exists
        frames_dir = os.path.join(self.inp.output_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

        # File naming convention: output_timestepNum.txt
        output_filename = os.path.join(
            frames_dir, f"output_{str(self.timestepNum).zfill(6)}.txt")
        output_plotname = os.path.join(
            frames_dir, f"output_{str(self.timestepNum).zfill(6)}.png")

        with open(output_filename, 'w') as f:
            # Write header
            f.write(f"# Time: {self.t:.4f}\n")
            f.write("# x, y, density, x-velocity, y-velocity, pressure\n")

            for i in range(self.grid.Nghost, self.grid.Nx - self.grid.Nghost):
                for j in range(self.grid.Nghost, self.grid.Ny - self.grid.Nghost):
                    x = self.grid.x[i]
                    y = self.grid.y[j]
                    components = [self.grid.grid[q, i, j]
                                  for q in range(self.c.NUMQ)]
                    f.write(f"{x:.12f}, {y:.12f}, " +
                            ", ".join(f"{comp:.8f}" for comp in components) + "\n")

        if self.inp.system == "euler2D":
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        elif self.inp.system == "mhd2d":
            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs = axs.ravel()  # Flatten the array to index by i

        for i in range(self.c.NUMQ):
            # Exclude ghost cells from the plot
            plot_data = self.grid.grid[i, self.grid.Nghost:-
                                       self.grid.Nghost, self.grid.Nghost:-self.grid.Nghost].T
            extent = [self.grid.x[self.grid.Nghost], self.grid.x[-self.grid.Nghost-1],
                      self.grid.y[self.grid.Nghost], self.grid.y[-self.grid.Nghost-1]]

            im = axs[i].imshow(plot_data, origin='lower', extent=extent)
            plt.colorbar(im, ax=axs[i])
            axs[i].set_title(self.c.variable_names[i])
            axs[i].set_xlabel('x')
            axs[i].set_ylabel('y')

        fig.suptitle(f"Time: {self.t:.4f}, Timestep: {self.timestepNum}")
        plt.tight_layout()
        fig.savefig(output_plotname)
        plt.close()

    def generate_movie(self):
        # Create a directory for the frames if it doesn't exist
        frames_dir = os.path.join(self.inp.output_dir, "frames")
        # if not os.path.exists(frames_dir):
        #     os.makedirs(frames_dir)

        # List all the output files and sort them
        # output_files = sorted(glob.glob(os.path.join(self.inp.output_dir, "output_*.png")))

        # movie_filename = os.path.join(self.inp.output_dir, "simulation_movie.mp4")

        ffmpeg_command = f"ffmpeg -y -framerate 24 -i {frames_dir}/output_%06d.png -c:v libx264 -pix_fmt yuv420p {self.inp.output_dir}/movie.mp4"
        os.system(ffmpeg_command)

        # if self.c.NUMQ == 3:
        #     fig, axs = plt.subplots(3, 1, figsize=(10, 15))
        # elif self.c.NUMQ == 4:
        #     fig, axs = plt.subplots(2,2, figsize=(10, 15))
        # else:
        #     print("self.c.NUMQ: ", self.c.NUMQ)
        #     raise RuntimeError("System not implemented for movie.")

        # axs = axs.ravel()  # Flatten the array to index by i

        # def update_plot(file):
        #     data = np.loadtxt(file, delimiter=',', skiprows=2)
        #     x = data[:, 0]

        #     #data is indexed by
        #     # data[:,i] where i = 0 for x, i = 1 for icomp1, i=2 for icomp2

        #     with open(file, 'r') as f:
        #         lines = f.readlines()
        #         time_line = lines[0]
        #         time = float(time_line.split(':')[1].strip())

        #     timestep = int(file.split('_')[-1].split('.')[0])

        #     for i in range(self.c.NUMQ):
        #         axs[i].clear()

        #         axs[i].scatter(x, data[:,i+1], c="black")
        #         axs[i].set_ylabel(self.c.variable_names[i])

        #     axs[0].set_title(f"Time: {time:.4f}, Timestep: {timestep}")

        # # Create an animation by updating the plot for each output file
        # ani = animation.FuncAnimation(fig, update_plot, frames=output_files, repeat=False)

        # # Save the animation as a movie file using ffmpeg
        # movie_filename = os.path.join(self.inp.output_dir, "simulation_movie.mp4")
        # ani.save(movie_filename, writer='ffmpeg', fps=10)

        # print(f"Movie saved as {movie_filename}")
