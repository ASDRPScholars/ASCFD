from ascfd.particle.constants import ParticleConstants
from ascfd.inputs import Inputs
import numpy as np

class ParticleInitialConditions:
    def __init__(self, particles, a_inputs, species_params: Inputs):
        self.pc = ParticleConstants()
        self.inp = a_inputs
        self.particles = particles
        self.params = species_params
        
        
    def apply_ics(self):
        
        if self.inp.particle_ics == "origin":
            self.particles[:] = self.origin()
        if self.inp.particle_ics == "random":
            self.particles[:] = self.random()
        if self.inp.particle_ics == "random_left_wall":
            self.particles[:] = self.random_left_wall()
    
        if self.inp.particle_ics == "random_particles_no_condition":
            self.particles[:] = self.random()
            
        print(self.particles[self.pc.XCOMP])
        
    def origin(self):
        ic_particles = np.zeros_like(self.particles)
        
        return ic_particles
            
            
    def random(self):
        print("initializing random!")
        ic_particles = np.zeros_like(self.particles)
        
        # Scale drift velocity to account for realistic mass ratio
        mass_ratio_correction = np.sqrt(self.params.mass / 100)  # Compensate for mass change
        drift_velocity = 20 / mass_ratio_correction  # Maintain effective flow rate
    
        for n in range(self.inp.n_particles):
            random_x, random_y = np.random.rand() * (self.inp.xlim[1] - self.inp.xlim[0]), np.random.rand() * (self.inp.ylim[1] - self.inp.ylim[0])
            vx, vy, vz = self.sample_maxwellian_velocity()

            ic_particles[self.pc.XCOMP, n] = random_x
            ic_particles[self.pc.YCOMP, n] = random_y
            ic_particles[self.pc.UCOMP, n] = vx + drift_velocity
            ic_particles[self.pc.VCOMP, n] = vy
            if self.pc.WCOMP < self.pc.NUMQ:
                ic_particles[self.pc.WCOMP, n] = vz

        return ic_particles
    
    
    def random_left_wall(self):
        ic_particles = np.zeros_like(self.particles)

        for n in range(self.inp.n_particles):
            x, y = 0.0, np.random.rand()
            vx, vy, vz = self.sample_maxwellian_velocity()

            ic_particles[self.pc.XCOMP, n] = x
            ic_particles[self.pc.YCOMP, n] = y
            ic_particles[self.pc.UCOMP, n] = vx
            ic_particles[self.pc.VCOMP, n] = vy
            if self.pc.WCOMP < self.pc.NUMQ:
                ic_particles[self.pc.WCOMP, n] = vz

        return ic_particles
    
    def sample_maxwellian_velocity(self):
        # kB = 1.380649e-23
        kB = 1
        v_th = np.sqrt(2 * kB * self.params.temperature / self.params.mass)
        # print("!$! IC!", self.params.temperature)
        # print(self.params.mass)
        # print(v_th)

        if self.params.temperature <= 0:
            return 0.0, 0.0, 0.0
    
        R1, R2 = np.random.rand(2)
        R3, R4 = np.random.rand(2)

        vx = v_th * np.sqrt(-1 * np.log(R1)) * np.cos(2 * np.pi * R2)
        vy = v_th * np.sqrt(-1 * np.log(R1)) * np.sin(2 * np.pi * R2)
        vz = v_th * np.sqrt(-1 * np.log(R3)) * np.cos(2 * np.pi * R4)   

        # print(vx, vy, vz)

        return vx, vy, vz