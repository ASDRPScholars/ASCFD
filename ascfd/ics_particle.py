from ascfd.constants import *
import numpy as np
from matplotlib.path import Path


def checkPolygon(point, points):
    path = Path(points)
    return path.contains_point(point)


def random_particles_no_condition(numParticles):
    particles = []

    while len(particles) < numParticles:
        randomX, randomY = np.random.rand(), np.random.rand()
        particles.append([randomX, randomY])
        
    return particles

# dont use this yet
def random_particles_with_condition(numParticles, points):
    particles = []

    while len(particles) < numParticles:
        randomX, randomY = np.random.rand(), np.random.rand()

        if not checkPolygon([randomX, randomY], points):
            particles.append([randomX, randomY])
        
    return particles
