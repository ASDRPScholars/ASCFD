airfoil_vertices = []

with open('random/airfoil.dat', 'r') as file:
    lines = file.readlines()

    # Skip the first line (airfoil name)
    for line in lines[1:]:
        # Split the line by whitespace and convert to floats
        parts = line.strip().split()
        if len(parts) == 2:
            x, y = map(float, parts)
            airfoil_vertices.append((x+0.5, y+1))

print(airfoil_vertices)