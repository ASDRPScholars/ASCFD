from ascfd.grid import Grid2D
from ascfd.constants import Constants
import numpy as np

class EmbeddedBoundaries:
    
    def __init__(self, grid: Grid2D, a_constants: Constants):
        self.grid = grid
        self.c = a_constants
        self.highlight_near_polygon = np.ones_like(self.grid.grid, dtype=bool)
    
    def find_inside_near_points(self, polygon_points, i_start, i_end, j_start, j_end): 
        
        print(polygon_points, i_start, i_end, j_start, j_end)
        
        inside_polygon_array = np.zeros((i_end-i_start+2, j_end-j_start+2), dtype=bool)
        near_polygon_array = np.zeros((i_end-i_start+2, j_end-j_start+2), dtype=bool)
        near_polygon_points_array = np.zeros((i_end-i_start+2, j_end-j_start+2), dtype=object)
        
        n = len(polygon_points)

        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                # print("looping", i, j)
                testing_inside = True
                inside = False
                near = False
                near_polygon_points = []
                
                px, py = i*self.grid.dx, j*self.grid.dy
                
                for (tx, ty) in [(px, py), (px+self.grid.dx, py), (px-self.grid.dx, py), (px, py+self.grid.dy), (px, py-self.grid.dy)]:
                    point_inside = False
       
                    for k in range(n):
                        # we'll use these two points to draw a line/edge of the polygon
                        p1x, p1y = polygon_points[k]
                        p2x, p2y = polygon_points[(k + 1) % n]

                        # check if the point's y-level crosses this edge
                        if ((p1y > ty) != (p2y > ty)) and p1y != p2y:

                            # find the x where this edge intersects y = py
                            xint = (ty - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                            # print("CHECKING X-INT", tx, xint)
                            if tx < xint:
                                point_inside = not point_inside
                        
                    # test the point itself first - and if its inside, then break and just return inside = true and near = false
                    if testing_inside:
                        if point_inside:
                            inside = True
                            break
                        # if the point itself is NOT inside, THEN test neighboring points:
                        else:
                            testing_inside = False
                    
                    if point_inside:
                        near = True
                        near_polygon_points.append((round(tx*100), round(ty*100)))
                
                inside_polygon_array[i, j] = inside
                near_polygon_array[i, j] = near
                near_polygon_points_array[i, j] = near_polygon_points
                
                # print("results", inside_polygon_array[i, j], near_polygon_array[i, j], near_polygon_points_array[i, j])
                
        # print(inside_polygon_array, near_polygon_array, near_polygon_points_array)
            
        return inside_polygon_array, near_polygon_array, near_polygon_points_array
        
    
    def fluid_vector_through_time(self, embedded_boundary_vector, velocity_vector):
        _, V_parallel = self.decompose_vector(embedded_boundary_vector, velocity_vector)
        return V_parallel

    @staticmethod
    def decompose_vector(embedded_boundary_vector, velocity_vector):
        norm_squared = np.dot(embedded_boundary_vector, embedded_boundary_vector)
        if norm_squared == 0:
            return velocity_vector, (0, 0)  # or return velocity unchanged
        dot_product = np.dot(velocity_vector, embedded_boundary_vector)
        V_parallel = (dot_product / norm_squared) * embedded_boundary_vector
        V_normal = velocity_vector - V_parallel
        return V_normal, V_parallel

    # TODO: check that x_int and dx implementation behaves properly
    def find_embedded_boundary_vector(self, vertices):
        vector_map = np.zeros((self.grid.Nx, self.grid.Ny, 2))

        for i, x in enumerate(self.grid.x_int):
            for j, y in enumerate(self.grid.y_int):
                cell_center = np.array([x + self.grid.dx / 2, y + self.grid.dy / 2])
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
            
    lower_bound = 0
    upper_bound = 1
    
    def apply_embedded_boundary_conditions(self, U_new, primU, system, vertices, inside_polygon, near_polygon, near_polygon_points, i_start, i_end, j_start, j_end):
        # Compute vector map once outside the loop
        vector_map = self.find_embedded_boundary_vector(vertices)
        
        for i in range(i_start, i_end):
            for j in range(j_start, j_end):
                if inside_polygon[i, j]:
                    continue
                    
                if near_polygon[i, j]:
                    velocity_vector = np.array([primU[self.c.UCOMP, i, j], primU[self.c.VCOMP, i, j]])
                    fluid_vec = self.fluid_vector_through_time(
                        vector_map[i, j],
                        velocity_vector
                    )
                    
                    # Update momentum components
                    U_new[self.c.MUCOMP, i, j] = U_new[self.c.RHOCOMP, i, j] * fluid_vec[0]
                    U_new[self.c.MVCOMP, i, j] = U_new[self.c.RHOCOMP, i, j] * fluid_vec[1]
                    
                    # Update density and energy for near points
                    if system == "euler2D":
                        for (tx, ty) in near_polygon_points[i, j]:
                            U_new[self.c.RHOCOMP, tx, ty] = U_new[self.c.RHOCOMP, i, j]
                            U_new[self.c.ECOMP, tx, ty] = U_new[self.c.ECOMP, i, j]
                    
                    elif system == "mhd2d":
                        for (tx, ty) in near_polygon_points[i, j]:
                            U_new[self.c.RHOCOMP, tx, ty] = U_new[self.c.RHOCOMP, i, j]
                            U_new[self.c.ECOMP, tx, ty] = U_new[self.c.ECOMP, i, j]
    
                            U_new[self.c.BXCOMP, tx, ty] = U_new[self.c.BXCOMP, i, j]
                            U_new[self.c.BYCOMP, tx, ty] = U_new[self.c.BYCOMP, i, j]
                            
                    else:
                        raise AssertionError("Unknown variable system passed into embedded boundaries")
                    
                    # Update highlight array
                    self.highlight_near_polygon[:, i, j] = False
                    
        print("finished applying eb's!")
        
        return U_new
                                

