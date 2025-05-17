import numpy as np
import matplotlib.pyplot as plt


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


# need to change this to muliple different vectors - make it 2d
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
            nearest_point1 = (0, 0)  # nearest point outside to the right
            nearest_point2 = (0, 0)  # nearest point outside to the left
            for v in vertices:
                # find mag. of distance from the vertex to the center of the cell
                distance = np.linalg.norm(np.array(v) - cell_center)
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


def generate_heat_map(vertices, velocity_vector, lower_bound, upper_bound, step, nx, ny, nu, nt, sigma, c):
    u = np.ones((nx, ny, 2))
    u[:, :, 0] = velocity_vector[0]
    u[:, :, 1] = velocity_vector[1]
    un = u.copy()
    dx = 2/(nx-1)
    dy = 2/(ny-1)
    dt = sigma * dx * dy / nu
    CFL = c * dt / dx
    u[int(.5 / dy):int(1 / dy + 1), int(.5 / dx):int(1 / dx + 1)] = 2
    if CFL > 1:
        print("Warning: CFL condition is not met.")
    grid_size = 50
    x_vals = np.linspace(lower_bound, upper_bound, nx)
    y_vals = np.linspace(lower_bound, upper_bound, ny)
    heat_map = np.zeros((nx, ny))
    vector_map = find_embedded_boundary_vector(
        vertices, lower_bound, upper_bound, step)
    for n in range(nt):
        for j in range(0, nx):
            for i in range(0, ny):
                if is_inside_polygon(vertices, (x_vals[i], y_vals[j])):
                    un[i, j] = fluid_vector_through_time(
                        vector_map[i][j], un[i, j], vertices, lower_bound, upper_bound, step)
                else:
                    un[i, j] = u[i, j] - CFL * \
                        (u[i, j] - u[i-1, j]) - CFL * (u[i, j] - u[i, j-1])
                heat_map[j, i] = np.sqrt(un[i, j][0]**2+un[i, j][1]**2)
        u = un.copy()
    plt.imshow(heat_map, extent=[lower_bound, upper_bound, lower_bound,
               upper_bound], origin='lower', cmap='hot', vmin=0.0, vmax=2.0)
    plt.colorbar(label='velocity magnitude')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('heat map')
    plt.show()


lower_bound = -0.6
upper_bound = 0.6
step = 0.01
# grid_size = nx = ny to work
nx = 50
ny = 50
nu = 0.05
nt = 1
sigma = .2
c = 0.3
velocity_vector = (1, 0)
vertices = "[(0.4990133642141358, 0.03139525976465669), (0.49605735065723894, 0.06266661678215213), (0.49114362536434436, 0.09369065729286231), (0.48429158056431554, 0.1243449435824274), (0.47552825814757677, 0.1545084971874737), (0.4648882429441257, 0.184062276342339), (0.45241352623300973, 0.21288964578253636), (0.4381533400219318, 0.24087683705085766), (0.42216396275100754, 0.26791339748949833), (0.4045084971874737, 0.29389262614623657), (0.38525662138789457, 0.3187119948743449), (0.3644843137107058, 0.34227355296434436), (0.3422735529643443, 0.3644843137107058), (0.3187119948743448, 0.3852566213878946), (0.2938926261462365, 0.4045084971874737), (0.2679133974894983, 0.42216396275100754), (0.24087683705085758, 0.43815334002193185), (0.21288964578253633, 0.4524135262330098), (0.18406227634233893, 0.46488824294412573), (0.15450849718747373, 0.47552825814757677), (0.12434494358242737, 0.48429158056431554), (0.09369065729286226, 0.49114362536434436), (0.06266661678215213, 0.49605735065723894), (0.03139525976465665, 0.4990133642141358), (-8.040613248383183e-17, 0.5), (-0.0313952597646567, 0.4990133642141358), (-0.06266661678215218, 0.4960573506572389), (-0.09369065729286241, 0.4911436253643443), (-0.12434494358242743, 0.48429158056431554), (-0.15450849718747378, 0.47552825814757677), (-0.184062276342339, 0.4648882429441257), (-0.21288964578253636, 0.45241352623300973), (-0.24087683705085772, 0.43815334002193174), (-0.26791339748949844, 0.4221639627510075), (-0.2938926261462365, 0.4045084971874737), (-0.3187119948743449, 0.3852566213878946), (-0.34227355296434436, 0.3644843137107057), (-0.36448431371070583, 0.34227355296434425), (-0.3852566213878947, 0.31871199487434476), (-0.40450849718747367, 0.2938926261462366), (-0.42216396275100754, 0.26791339748949833), (-0.4381533400219318, 0.2408768370508576), (-0.4524135262330098, 0.21288964578253625), (-0.46488824294412573, 0.18406227634233888), (-0.4755282581475768, 0.15450849718747356), (-0.48429158056431554, 0.12434494358242741), (-0.49114362536434436, 0.09369065729286229), (-0.49605735065723894, 0.06266661678215205), (-0.4990133642141358, 0.03139525976465657), (-0.5, -1.6081226496766366e-16), (-0.4990133642141358, -0.03139525976465667), (-0.49605735065723894, -0.06266661678215214), (-0.4911436253643443, -0.09369065729286238), (-0.48429158056431554, -0.12434494358242751), (-0.47552825814757677, -0.15450849718747386), (-0.4648882429441256, -0.18406227634233915), (-0.45241352623300973, -0.21288964578253633), (-0.43815334002193174, -0.2408768370508577), (-0.4221639627510075, -0.2679133974894984), (-0.4045084971874736, -0.2938926261462367), (-0.3852566213878945, -0.318711994874345), (-0.3644843137107058, -0.34227355296434436), (-0.3422735529643443, -0.36448431371070583), (-0.31871199487434476, -0.3852566213878947), (-0.2938926261462366, -0.40450849718747367), (-0.26791339748949816, -0.42216396275100765), (-0.24087683705085763, -0.4381533400219318), (-0.21288964578253608, -0.4524135262330099), (-0.1840622763423389, -0.46488824294412573), (-0.15450849718747378, -0.47552825814757677), (-0.12434494358242722, -0.4842915805643156), (-0.09369065729286231, -0.49114362536434436), (-0.06266661678215187, -0.49605735065723894), (-0.031395259764656604, -0.4990133642141358), (-9.184850993605148e-17, -0.5), (0.03139525976465686, -0.4990133642141358), (0.06266661678215212, -0.49605735065723894), (0.09369065729286256, -0.4911436253643443), (0.12434494358242747, -0.48429158056431554), (0.15450849718747361, -0.4755282581475768), (0.18406227634233913, -0.4648882429441256), (0.2128896457825363, -0.4524135262330098), (0.24087683705085786, -0.4381533400219317), (0.2679133974894984, -0.4221639627510075), (0.29389262614623685, -0.4045084971874735), (0.318711994874345, -0.3852566213878945), (0.3422735529643443, -0.3644843137107058), (0.36448431371070594, -0.34227355296434414), (0.3852566213878946, -0.3187119948743448), (0.4045084971874739, -0.2938926261462363), (0.42216396275100765, -0.26791339748949816), (0.4381533400219318, -0.24087683705085766), (0.45241352623300984, -0.2128896457825361), (0.46488824294412573, -0.18406227634233893), (0.4755282581475769, -0.1545084971874734), (0.4842915805643156, -0.12434494358242724), (0.49114362536434436, -0.09369065729286234), (0.49605735065723894, -0.0626666167821519), (0.4990133642141358, -0.03139525976465663)]"
vertices = eval(vertices)

generate_heat_map(vertices, velocity_vector, lower_bound,
                  upper_bound, step, nx, ny, nu, nt, sigma, c)
