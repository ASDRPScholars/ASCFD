import numpy as np
#constants
# WHAT TO DEFAULT TO?
DEFAULT_MASS = 2 # Ions
DEFAULT_CHARGE = 3 #ions
ELECTRON_MASS = 1
ELECTRON_CHARGE = -1

def create_particle(particles, position, type, charge, mass):

    new_particle = {
        '_position': position, # tuple
        '_type': type,
        '_charge': charge,
        '_mass': mass,
    }

    particles.append(new_particle)

def get_local_density(particles, position, type, size=1):
    count =0
    for p in particles:
        if p['_type'] == type:
            dist = np.linalg.norm(np.array(p['_position']))
            if dist < size:
                count+=1
    volume=(size**3 if len(position) == 3 else size**2)
    return count / volume if volume > 0 else 0
def calc_ionization_rate(electron_density, neutral_density, coefficient=1e-13):
    return coefficient*electron_density*neutral_density
def ionize_particles(particles, ionizationRates, dt):
    neutral_particles = (particles['_type'] == "Neutral")
    neutral_particle_indices = np.where(neutral_particles)[0]

    for neutral_particle_index in neutral_particle_indices:
        neutral_particle = particles[neutral_particle_index]

        position = neutral_particle['_position']
    
        # make methods to find these values depending on position (could use grid also similar to how particle cell calculation was originally done)
        electron_density = get_local_density(particles, position, "Electron") 
        neutral_density = get_local_density(particles, position, "Neutral")
        ionization_rate = calc_ionization_rate(electron_density, neutral_density)

        # Monte Carlo ayy

        p_ionize = 1 - np.exp(-ionization_rate * dt)

        rand = np.random.rand()

        if rand > p_ionize: # quick exit case: we will NOT ionize
            continue
            # should this + what follows be an else statement?
            # where id neutral_particle defined
        particles.pop(neutral_particle_index)

        # create new ion in its place with correct values etc...
        # defaulted the charge and mass to 1?
        create_particle(particles, position, "Ion", DEFAULT_CHARGE, DEFAULT_MASS)
        create_particle(particles, position, "Electron", ELECTRON_CHARGE, ELECTRON_MASS)

def update_velocity(particle, charge, electric_field, magnetic_field, dt):
