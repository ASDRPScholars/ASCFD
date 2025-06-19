import numpy as np

def interpolate_edge(p1, p2, num_points):
    return [tuple(p1 + (p2 - p1) * t) for t in np.linspace(0, 1, num_points, endpoint=False)]

def generate_wall(corners, points_per_edge):
    wall = []
    for i in range(len(corners)):
        p1 = np.array(corners[i])
        p2 = np.array(corners[(i + 1) % len(corners)])
        wall.extend(interpolate_edge(p1, p2, points_per_edge))
    return wall

top_wall_corners = [(-0.1, 0.6), (0.4, 0.6), (0.4, 1.1), (-0.1, 1.1)]
bottom_wall_corners = [(-0.1, -0.1), (0.4, -0.1), (0.4, 0.4), (-0.1, 0.4)]

top_wall = generate_wall(top_wall_corners, 50)
bottom_wall = generate_wall(bottom_wall_corners, 50)

# Optional: print the number of points and preview
print(f"Top wall: {len(top_wall)} points")
print(f"Bottom wall: {len(bottom_wall)} points")

print(top_wall)
print(bottom_wall)