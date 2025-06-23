from ascfd.constants import *
import numpy as np


def random_particles_no_condition(numParticles):
    particles = []

    while len(particles) < numParticles:
        randomX, randomY = np.random.rand(), np.random.rand()
        particles.append([randomX, randomY])
        
    return np.array(particles)

# dont use this yet
def random_particles_with_condition(numParticles, conditionFunction):
    particles = []

    while len(particles) < numParticles:
        randomX, randomY = np.random.rand(), np.random.rand()

        if conditionFunction(randomX, randomY):
            particles.append([randomX, randomY])
