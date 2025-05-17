import math

a = 0.5
b = 0.5
r = 0.25

# The lower this value the higher quality the circle is with more points generated
stepSize = 0.1

# Generated vertices
positions = []

t = 0
while t < 2 * math.pi:
    positions.append((r * math.cos(t) + a, r * math.sin(t) + b))
    t += stepSize

print(positions)
