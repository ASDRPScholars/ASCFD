import numpy as np

DEFAULT_MASS = 2 # I just randomly picked these values
DEFAULT_CHARGE = 3

def create_particle(particles, position, type, charge, mass):

    new_particle = {
        '_position': position, # tuple
        '_type': type,
        '_charge': charge,
        '_mass': mass,
    }

    particles.append(new_particle)


def ionize_particles(particles, ionizationRates, dt):
    neutral_particles = (particles['_type'] == "Neutral")
    neutral_particle_indices = np.where(neutral_particles)[0]

    for neutral_particle_index in neutral_particle_indices:
        neutral_particle = particles[neutral_particle_index]

        particle_position = neutral_particle['_position']
    
        # make methods to find these values depending on position (could use grid also similar to how particle cell calculation was originally done)
        electron_density = ... 
        neutral_density = ...

        ionization_rate = ... # another method to calculuate

        # statistics now !!!!

        p_ionize = 1 - np.exp(-ionization_rate * dt)

        rand = np.random.rand()

        if rand > p_ionize: # quick exit case: we will NOT ionize
            continue
    
        particles.pop(neutral_particle_index)

        # create new ion in its place with correct values etc...

        # create_particle()


        
        



