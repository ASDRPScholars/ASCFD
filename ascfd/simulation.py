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
from matplotlib import cm
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
            self.grid, self.inp.bcs_lo, self.inp.bcs_hi, self.c)
        self.flux = Flux(self.c, self.inp.flux)

        self.apply_ics()

        self.bcs.apply_bcs()
        self.grid.check_grid(self.c)
        # setup initial time to be the starting time from the inputs file.
        # The starting timestep will always be 0.
        self.t = self.inp.t0
        self.timestepNum = 0
        
        self.highlight_near_polygon = np.ones_like(self.grid.grid, dtype=bool)
        # self.highlight_inside_polygon = []

        # -1 is no output. Always output ICs if we are outputting.
        if self.inp.output_freq >= 0:
            self.output()

    def run(self):
        
        # def is_inside_polygon(polygon_points, point):
        #     """
        #     Determine whether a point is inside a polygon using a horizontal ray-casting algorithm. 

        #     i.e. If we draw a horizontal, right-facing ray from a point and it intersects with the polygon an ODD number of times
        #     (an odd number of edges), the point is inside the polygon. If that right-facing raw intersects an EVEN number of edges,
        #     then the point is OUTSIDE the polygon. Draw it for yourself and see!

        #     Instead of counting even and odd, we'll just alternate between inside = True and inside = False.
        #     """

        #     n = len(polygon_points)
        #     inside = False

        #     px, py = point
        #     n = len(polygon_points)
            
        #     for i in range(n):
        #         # we'll use these two points to draw a line/edge of the polygon
        #         p1x, p1y = polygon_points[i]
        #         p2x, p2y = polygon_points[(i + 1) % n]

        #         # check if the point's y-level crosses this edge
        #         if (p1y > py) != (p2y > py) and p1y != p2y:
        #             # find the x where this edge intersects y = py
        #             xint = (py - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
        #             if px < xint:
        #                 inside = not inside

        #     return inside
        
        def is_near_polygon(polygon_points, i_start, i_end, j_start, j_end):
            
            # TODO: RENAME VARIABLES FROM "INSIDE" -> "NEAR"
            # TODO: MAKE GRID_SIZE A PARAMETER
            
            grid_size = 100
            
            inside_polygon_array = np.zeros((i_end-i_start+2, j_end-j_start+2), dtype=bool)
            near_polygon_array = np.zeros((i_end-i_start+2, j_end-j_start+2), dtype=bool)
            near_polygon_points_array = np.zeros((i_end-i_start+2, j_end-j_start+2), dtype=object)
            
            n = len(polygon_points)

            for i in range(i_start, i_end):
                for j in range(j_start, j_end):
                    
                    testing_inside = True
                    inside = False
                    near = False
                    near_polygon_points = []
                    
                    px, py = i*self.grid.dx, j*self.grid.dy
            
                    debug = 0
                    inside_indices = []
            
                    print("--POINT--", (i, j))
                    
                    for (tx, ty) in [(px, py), (px+0.01, py), (px-0.01, py), (px, py+0.01), (px, py-0.01)]:
                        print("TESTING POINT", (tx*100, ty*100))
                        point_inside = False
                        # print("ITERATING OVER TESTING POITNS")
                        
                        for k in range(n):
                            # we'll use these two points to draw a line/edge of the polygon
                            p1x, p1y = polygon_points[k]
                            p2x, p2y = polygon_points[(k + 1) % n]
                            
                            # print("ITERATING OVER POLYGON")
                            
                            # print(p1y, p2y, tx, ty)
                            # print((p1y > ty) != (p2y > ty))
                            # print(p1y != p2y)

                            # check if the point's y-level crosses this edge
                            if ((p1y > ty) != (p2y > ty)) and p1y != p2y:
                                # print("CHECKING Y-LEVEL")
                                # find the x where this edge intersects y = py
                                xint = (ty - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                                # print("CHECKING X-INT", tx, xint)
                                if tx < xint:
                                    point_inside = not point_inside
                                    # print("point_inside changed to", point_inside)
                                    # print("intersect!", debug)
                            
                        # test the point itself first - and if its inside, then break and just return inside = true and near = false
                        if testing_inside:
                            # print("testing_inside", testing_inside)
                            if point_inside:
                                inside = True
                                # print("found an inside point!", inside)
                                break
                            # if the point itself is NOT inside, THEN test neighboring points:
                            else:
                                testing_inside = False
                        
                        if point_inside:
                            near = True
                            near_polygon_points.append((round(tx*100), round(ty*100)))
                            inside_indices.append(debug)
                            
                        debug += 1
                            
                    # print("INSIDE POINTS + BOOL", near_polygon_points, near)
                    # print("INSIDE INDICES", inside_indices)
                    
                    inside_polygon_array[i, j] = inside
                    near_polygon_array[i, j] = near
                    near_polygon_points_array[i, j] = near_polygon_points
                    
            # if True in inside_polygon_array:
            #     raise ValueError("INSIDE HAS A TRUE")
            # if True in near_polygon_array:
            #     raise ValueError("NEAR HAS A TRUE")
                
            return inside_polygon_array, near_polygon_array, near_polygon_points_array
            
        
        # TODO: find the negative sign
        def fluid_vector_through_time(embedded_boundary_vector, velocity_vector, vertices, dt):
            # TODO: no logic necessary to double-check if its going to head into the embedded boundary?
            
            _, V_parallel = decompose_vector(embedded_boundary_vector, velocity_vector)
            return V_parallel


        def decompose_vector(embedded_boundary_vector, velocity_vector):
            norm_squared = np.dot(embedded_boundary_vector, embedded_boundary_vector)
            if norm_squared == 0:
                return velocity_vector, (0, 0)  # or return velocity unchanged
            dot_product = np.dot(velocity_vector, embedded_boundary_vector)
            V_parallel = (dot_product / norm_squared) * embedded_boundary_vector
            V_normal = velocity_vector - V_parallel
            return V_normal, V_parallel

        def find_embedded_boundary_vector(vertices, lower_bound, upper_bound, step):
            grid_size = 100
            x_vals = np.linspace(lower_bound, upper_bound, grid_size)
            y_vals = np.linspace(lower_bound, upper_bound, grid_size)
            vector_map = np.zeros((grid_size, grid_size, 2))

            for i, x in enumerate(x_vals):
                for j, y in enumerate(y_vals):
                    cell_center = np.array([x + step / 2, y + step / 2])
                    min_dist = float('inf')
                    tangent = np.array([1, 0])  # default

                    for k in range(len(vertices)):
                        v1 = np.array(vertices[k])
                        v2 = np.array(vertices[(k + 1) % len(vertices)])
                        edge_center = 0.5 * (v1 + v2)
                        dist = np.linalg.norm(cell_center - edge_center)
                        if dist < min_dist:
                            min_dist = dist
                            tangent = v2 - v1

                    vector_map[i, j] = tangent
            return vector_map

        vertices = [(0.55, 0.5), (0.5487510413195065, 0.524958354161707), (0.5450166444603104, 0.5496673326987653), (0.5388341222814015, 0.5738800516653348), (0.5302652485007213, 0.5973545855771626), (0.5193956404725932, 0.6198563846510508), (0.5063339037274196, 0.6411606183487588), (0.4912105468211221, 0.6610544218094228), (0.47417667733679136, 0.6793390227248807), (0.45540249206766614, 0.6958317274068708), (0.43507557646703493, 0.7103677462019741), (0.41339903035639436, 0.7228018400153589), (0.3905894386191684, 0.7330097714918066), (0.36687470715614684, 0.7408895463542983), (0.34249178572506017, 0.7463624324971151), (0.31768430041692564, 0.7493737466510136), (0.2927001194246777, 0.7498934007603762), (0.2677888764261187, 0.7479162026131172), (0.24319947632672811, 0.7434619077195488), (0.219177608284124, 0.7365750219218535), (0.1959632908632143, 0.7273243567064204), (0.1737884738500355, 0.7158023416622183), (0.15287472068616342, 0.7021241009548974), (0.1334309946800438, 0.68642630304418), (0.11565157111468849, 0.6688657951377877), (0.09971409611326643, 0.6496180360259889), (0.08577781165776305, 0.6288753429553658), (0.0739819644957346, 0.6068449700584573), (0.06444441483283536, 0.5837470375389759), (0.0572604587126023, 0.5598123323034953), (0.05250187584988858, 0.5352800020149665), (0.05021621243168012, 0.5103951656083223), (0.05042630605131174, 0.4854064641431046), (0.053130057522783825, 0.4605635764641875), (0.058300451855134855, 0.43611472449329175), (0.06588582817730107, 0.41230419307759464), (0.07581039591646344, 0.38936988917628645), (0.0879749920723982, 0.3675409647728763), (0.10225807202139614, 0.34703552726431985), (0.11851692394996532, 0.3280584602040062), (0.13658909478409734, 0.31079937617301767), (0.15629401336668305, 0.29543072223389716), (0.1774347946648253, 0.28210605689660284), (0.19980020698000633, 0.2709585158126362), (0.22316678250539515, 0.26209948152762097), (0.24730105014230508, 0.25561747058372575), (0.27196186826623625, 0.2515772490916339), (0.29690283413427715, 0.2500191856089748), (0.32187474585986137, 0.25095884779103983), (0.3466280923556435, 0.25438684684391677), (0.37091554636580615, 0.26026893133421525), (0.3944944356782446, 0.2685463294180667), (0.4171291678250937, 0.27913633606996135), (0.43859358404478965, 0.2919331394440243), (0.45867321898565794, 0.30680887811100266), (0.4771674435728144, 0.3236149186074014), (0.4938914696275618, 0.3421833405319189), (0.5086781962097893, 0.36232861435058966), (0.5213798792353292, 0.38384945514655966), (0.5318696076860084, 0.40653083379243976), (0.5400425716625912, 0.43014612545026726), (0.5458171096106459, 0.4544593739319747), (0.5491355242558043, 0.47922764929562434)]
        
        lower_bound = 0
        upper_bound = 1
        step = self.grid.dx
        
        i_start, i_end = self.grid.Nghost, self.grid.Nx + self.grid.Nghost
        j_start, j_end = self.grid.Nghost, self.grid.Ny + self.grid.Nghost
        
        vector_map = find_embedded_boundary_vector(
            vertices, lower_bound, upper_bound, step)
        
        inside_polygon, near_polygon, near_polygon_points = is_near_polygon(vertices, i_start, i_end, j_start, j_end)
        
        # TIME LOOP
        while (self.t < self.inp.t_finish) and self.timestepNum < self.inp.nt:
            print(f"Timestep: {self.timestepNum}, Current time: {self.t}")

            self.bcs.apply_bcs()

            self.grid.assert_variable_type("prim")

            # Store primitive variables at the start of the step for Powell terms
            primU_n = np.copy(self.grid.grid)

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

            if self.inp.timeStepper == "RK1":
                # returns numerical flux and conservative variables at interface
                self.grid.assert_variable_type("prim")
                
                consU = self.euler.prim_to_cons(self.grid.grid)
                
                # consU, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus = self.flux.getFlux(
                #     self.grid.grid, self.grid.Nx, self.grid.Ny, self.grid.Nghost)

                # Start with the current conservative variables
                U_new = np.copy(consU)
                # print("TYPE OF U_new!!!!", type(U_new))
                primU = self.euler.cons_to_prim(U_new)
                
                print(self.grid.grid.shape)
                grid_shape_2d = self.grid.grid.shape[1:]
                print(grid_shape_2d)
                
                # TODO: move everything into here eventually (including near points) so we don't have to recalculate them each timestep
                # near_polygon = np.zeros(grid_shape_2d, dtype=bool)
                # inside_polygon = np.zeros(grid_shape_2d, dtype=bool)

                # FLUID UPDATE
                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        
                        point = (self.grid.dx * i, self.grid.dy * j)
                        px, py = point
                        # print("POINT:", i, j)
                        # print("CONVERTED POINT: ", point)
                        
                        velocity_vector = np.array([
                                    primU[self.c.UCOMP, i, j],
                                    primU[self.c.VCOMP, i, j]
                                ])
                        
                        # future_point = point + velocity_vector * dt

                        for icomp in range(self.c.NUMQ):
                            near_is_zero = False
                            outside_is_zero = False
                            
                            if inside_polygon[i, j] == True:
                                print("inside polygon")
                                # Optionally: self.highlight_inside_polygon.append((i*0.01, j*0.01))
                                break

                            # near_polygon[i, j], near_polygon_points = is_near_polygon(vertices, point)
                            # print("NEW NEAR POLYGON PROPER VALUE:",  is_near_polygon(vertices, point))
                            # print("NEW NEAR POLYGON ARRAY: ", near_polygon[i, j])

                            if near_polygon[i, j] == True:
                                print("HIII NEAR POLYGON")
                                self.highlight_near_polygon[icomp, i, j] = False
                                print("UPDATED NEAR POLYGON HIGHLIGHT")

                                fluid_vec = fluid_vector_through_time(
                                    vector_map[i][j],
                                    velocity_vector,
                                    vertices,
                                    dt
                                )

                                if icomp == self.c.MUCOMP:
                                    print("UPDATING MU ON", (px, py))
                                    U_new[icomp, i, j] = U_new[self.c.RHOCOMP, i, j] * fluid_vec[0]

                                elif icomp == self.c.MVCOMP:
                                    print("UPDATING MV ON", (px, py))
                                    U_new[icomp, i, j] = U_new[self.c.RHOCOMP, i, j] * fluid_vec[1]
                                    print(U_new[icomp, i, j])
                                    if U_new[icomp, i, j] <= 0:
                                        print("IS Y-VELOCITY POSITIVE, ZERO, OR NEGATIVE?", 
                                            U_new[icomp, i, j] > 0, 
                                            U_new[icomp, i, j] == 0, 
                                            U_new[icomp, i, j] < 0)
                                        print("WHAT IS THE INITIAL Y-VELOCITY VECTOR?", velocity_vector[1])
                                        print("IS FLUID VEC POSITIVE, ZERO, OR NEGATIVE?", 
                                            fluid_vec[1] > 0, 
                                            fluid_vec[1] == 0, 
                                            fluid_vec[1] < 0)

                                elif icomp == self.c.RHOCOMP:
                                    for (tx, ty) in near_polygon_points[i, j]:
                                        print("UPDATING RHO ON", (tx, ty))
                                        U_new[icomp, tx, ty] = U_new[icomp, i, j]
                                        print("NEW RHO DIFFERENCE =", U_new[icomp, tx, ty] - U_new[icomp, i, j])

                                elif icomp == self.c.ECOMP:
                                    print("LENGTH OF INSIDE POINTS", len(near_polygon_points))
                                    for (tx, ty) in near_polygon_points[i, j]:
                                        print("UPDATING E ON", (tx, ty))
                                        U_new[icomp, tx, ty] = U_new[icomp, i, j]
                                        print("NEW E DIFFERENCE =", U_new[icomp, tx, ty] - U_new[icomp, i, j])

                                # Skip finite volume update if near_polygon and velocity component
                
                # TODO: is it better for this to use consU since we're directly modifying U_new?
                _, numFluxX_plus, numFluxX_minus, numFluxY_plus, numFluxY_minus = self.flux.getFlux(
                    self.grid.grid, self.grid.Nx, self.grid.Ny, self.grid.Nghost)

                for i in range(i_start, i_end):
                    for j in range(j_start, j_end):
                        for icomp in range(self.c.NUMQ):
                            if icomp in (self.c.MUCOMP, self.c.MVCOMP) and near_polygon[i, j] == True:
                                continue
                            elif inside_polygon[i, j] == True:
                                continue
                            else:
                                # Fallback: apply finite volume update if not inside/near
                                delta = (
                                    (dt / self.grid.dx) * (numFluxX_plus[icomp, i, j] - numFluxX_minus[icomp, i, j]) +
                                    (dt / self.grid.dy) * (numFluxY_plus[icomp, i, j] - numFluxY_minus[icomp, i, j])
                                )
                                updated_value = consU[icomp, i, j] - delta

                                # # Apply floor where needed
                                # floor_values = {self.c.MUCOMP: 0.01, self.c.MVCOMP: 0.01}
                                U_new[icomp, i, j] = updated_value
                            
                                    
                print("printed", self.highlight_near_polygon)
                    
                # if not near_is_zero:
                #     print("**NEAR** HAS ALL ZERO VELOCITIES")
                # if not outside_is_zero:
                #     print("**OUTSIDE** HAS ALL ZERO VELOCITIES")

                # Powell divergence cleaning for MHD
                if self.inp.system == "mhd2d":
                    # Calculate div(B) using central differences on consU
                    divB = np.zeros_like(consU[0])
                    # Need to calculate divB over the domain where U_new is updated + 1 layer for central diff
                    # However, we only apply the source term within the main update loop domain.
                    # Note: Using consU which contains Bx, By directly.
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            # Central difference requires i-1, i+1, j-1, j+1
                            # Ensure indices are within the bounds where consU is valid (including ghosts)
                            divB_x = (
                                consU[self.c.BXCOMP, i + 1, j] - consU[self.c.BXCOMP, i - 1, j]) / (2.0 * self.grid.dx)
                            divB_y = (
                                consU[self.c.BYCOMP, i, j + 1] - consU[self.c.BYCOMP, i, j - 1]) / (2.0 * self.grid.dy)
                            divB[i, j] = divB_x + divB_y

                    # Calculate Powell source terms using primU_n and consU
                    powell_source = calculate_powell_source(
                        consU, primU_n, divB, self.c)

                    # Apply Powell source terms to U_new
                    for i in range(i_start, i_end):
                        for j in range(j_start, j_end):
                            for icomp in range(self.c.NUMQ):
                                U_new[icomp, i, j] += dt * \
                                    powell_source[icomp, i, j]

                # TODO: embedded boundaries

                # TODO: particle step

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

    # TODO: why does this unused plot function exist?
    
    # def plot(self):
    #     if not os.path.exists(self.inp.output_dir):
    #         os.makedirs(self.inp.output_dir)

    #     if self.inp.system == "euler2D":
    #         fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    #         axs[0].scatter(
    #             self.grid.x, self.grid.grid[self.c.RHOCOMP, :], c="black")
    #         axs[0].set_ylabel("Density")

    #         axs[1].scatter(
    #             self.grid.x, self.grid.grid[self.c.UCOMP, :],  c="black")
    #         axs[1].set_ylabel("Velocity")

    #         axs[2].scatter(
    #             self.grid.x, self.grid.grid[self.c.PCOMP, :],  c="black")
    #         axs[2].set_ylabel("Pressure")

    #     elif self.inp.system == "mhd2d":
    #         fig, axs = plt.subplots(3, 1, figsize=(10, 15))
    #         axs[0].scatter(
    #             self.grid.x, self.grid.grid[self.c.RHOCOMP, :], c="black")
    #         axs[0].set_ylabel("Density")

    #         axs[1].scatter(
    #             self.grid.x, self.grid.grid[self.c.UCOMP, :],  c="black")
    #         axs[1].set_ylabel("Velocity")

    #         axs[2].scatter(
    #             self.grid.x, self.grid.grid[self.c.PCOMP, :],  c="black")
    #         axs[2].set_ylabel("Pressure")

    #         axs[3].scatter(
    #             self.grid.x, self.grid.grid[self.c.B_XCOMP, :],  c="black")
    #         axs[3].set_ylabel("Magnetic Field")

    #     axs[0].set_title(f"Time: {self.t:.4f}")
    #     plt.savefig(
    #         f"{self.inp.output_dir}/plot_dt{str(self.timestepNum).zfill(6)}")
    #     plt.close()

    def apply_ics(self):
        if self.inp.system == "euler2D":
            if self.inp.ics == "diagonal_advection":
                print("diagt??")
                self.grid.fill_grid(ics.diagonal_advection_2d)
            elif self.inp.ics == "kelvin_helmholtz":
                self.grid.fill_grid(ics.kelvin_helmholtz_2d)
            elif self.inp.ics == "double_mach_reflection":
                self.grid.fill_grid(ics.double_mach_reflection_2d)
            elif self.inp.ics == "riemann_problem":
                self.grid.fill_grid(ics.riemann_2d)
            elif self.inp.ics == "static":
                print("static!")
                self.grid.fill_grid(ics.static_2d)
            else:
                raise RuntimeError("[FLUID] ICS not valid.")

        elif self.inp.system == "mhd2d":
            if self.inp.ics == "orszag_tang":
                self.grid.fill_grid(ics.orszag_tang_2d)
            elif self.inp.ics == "field_loop":
                self.grid.fill_grid(ics.field_loop_2d)
            elif self.inp.ics == "rotor":
                print("before filling grid")
                self.grid.fill_grid(ics.rotor_2d)
                print("after filling grid")

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
        
        data_dir = os.path.join(self.inp.output_dir, "raw data")
        os.makedirs(data_dir, exist_ok=True)

        # File naming convention: output_timestepNum.txt
        output_filename = os.path.join(
            data_dir, f"output_{str(self.timestepNum).zfill(6)}.txt")
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
                    components = [self.grid.grid[q, i, j] for q in range(self.c.NUMQ)]
                    f.write(f"{x:.12f}, {y:.12f}, " + ", ".join(f"{comp:.8f}" for comp in components) + "\n")

        if self.inp.system == "euler2D":
            fig, axs = plt.subplots(2, 2, figsize=(15, 15))
        elif self.inp.system == "mhd2d":
            fig, axs = plt.subplots(2, 3, figsize=(18, 12))
        axs = axs.ravel()  # Flatten the array to index by i

        for q in range(self.c.NUMQ):
            # Exclude ghost cells from the plot
            plot_data = self.grid.grid[q, self.grid.Nghost:-self.grid.Nghost, self.grid.Nghost:-self.grid.Nghost].T # TODO: why transpose?
            extent = [self.grid.x[self.grid.Nghost], self.grid.x[-self.grid.Nghost-1],
                      self.grid.y[self.grid.Nghost], self.grid.y[-self.grid.Nghost-1]]

            im = axs[q].imshow(plot_data, origin='lower', extent=extent, cmap='magma')
            print("shape", plot_data.shape)
            
            plt.colorbar(im, ax=axs[q])
            
            # for i, j in self.highlight_near_polygon:
            #     axs[q].plot(i-(self.grid.Nghost*self.grid.dx), j-(self.grid.Nghost*self.grid.dy), marker='s', color='red', markersize=3)
                
            # for i, j in self.highlight_inside_polygon:
            #     axs[q].plot(i-self.grid.Nghost, j-self.grid.Nghost, marker='s', color='blue', markersize=2)
                
            # highlight_layer = np.zeros((self.grid.shape, 4))  # RGBA image
            # highlight_layer[self.highlight_near_polygon] = [1, 0, 0, 0.6]  # Red with alpha
            
            # lots of cool tricks to make a transparent mask from here: https://stackoverflow.com/questions/10127284/overlay-imshow-plots-in-matplotlib
            transparent = cm.get_cmap("binary")
            alphas = np.linspace(0.5, 0, transparent.N+3)
            transparent._init()
            transparent._lut[:,-1] = alphas
            
            highlight_data = (self.highlight_near_polygon[q, self.grid.Nghost:-self.grid.Nghost, self.grid.Nghost:-self.grid.Nghost]).T # TODO: why transpose?
            axs[q].imshow(highlight_data, origin='lower', extent=extent, alpha=0.5, cmap=transparent)
            
            axs[q].set_title(self.c.variable_names[q])
            axs[q].set_xlabel('x')
            axs[q].set_ylabel('y')

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

# Define a helper function to calculate the Powell source term outside the main loop for clarity


def calculate_powell_source(consU, primU, divB, c):
    """Calculates the Powell et al. (1999) source terms."""
    Bx = consU[c.BXCOMP]
    By = consU[c.BYCOMP]
    u = primU[c.UCOMP]
    v = primU[c.VCOMP]

    powell_source = np.zeros_like(consU)
    # S_rho = 0
    powell_source[c.MUCOMP] = -Bx * divB
    powell_source[c.MVCOMP] = -By * divB
    # S_MWCOMP = 0 in 2D
    powell_source[c.ECOMP] = -(u * Bx + v * By) * divB
    powell_source[c.BXCOMP] = -u * divB
    powell_source[c.BYCOMP] = -v * divB
    # S_BZCOMP = 0 in 2D
    return powell_source
