from ascfd.particle.constants import ParticleConstants
from ascfd.inputs import Inputs
from ascfd.plasma_refs import PlasmaReferences
import numpy as np

class ParticleInitialConditions:
    def __init__(self, a_inputs, species_params: Inputs):
        self.pc = ParticleConstants()
        self.inp = a_inputs
        self.ref = PlasmaReferences()
        self.params = species_params
        
        
    def apply_ics(self, weight=None):
        
        if self.inp.particle_ics == "origin":
            self.particles[:] = self.origin()
        if self.inp.particle_ics == "random":
            ic_particles = self.random(weight)
            print("!IC! IC PARTICLES", ic_particles)
            self.particles[:] = ic_particles
            print("!IC! SELF.PARTICLES", self.particles)
        if self.inp.particle_ics == "random_left_wall":
            self.particles[:] = self.random_left_wall()
    
        if self.inp.particle_ics == "random_particles_no_condition":
            self.particles[:] = self.random()
            
        print(self.particles[self.pc.XCOMP])
        
    def origin(self):
        ic_particles = np.ones_like(self.particles) * 0.5
        
        return ic_particles
            
    def random(self, weight=None):
        if self.params.type == "n":
            weight *= 3.7666e6
            
        ic_particles = np.zeros((self.pc.NUMQ+1, (self.inp.nx * self.inp.ny)))
        
        dx = self.inp.dx
        dy = self.inp.dy
    
        for i in range(self.inp.nx):
            for j in range(self.inp.ny):
                for np in range(self.inp.n_ppc):
                    random_x, random_y = np.random.uniform(i*dx, (i+1)*dx), np.random.uniform(j*dx, (j+1)*dy)
                    vx, vy, vz = self.sample_maxwellian_velocity()

                    n = i+j 
                    
                    ic_particles[self.pc.XCOMP, n] = random_x
                    ic_particles[self.pc.YCOMP, n] = random_y
                    ic_particles[self.pc.UCOMP, n] = vx
                    ic_particles[self.pc.VCOMP, n] = vy
                    
                    if weight is not None:
                        ic_particles[self.pc.WEIGHT, n] = weight[i, j]
                        
                    elif self.params == "n":
                        ic_particles[self.pc.WEIGHT, n] = self.inp.n_n / (self.inp.nx * self.inp.ny)
                        
                    elif self.params == "i":
                        ic_particles[self.pc.WEIGHT, n] = self.inp.n_i / (self.inp.nx * self.inp.ny)
            
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
        kB = 1.380649e-23
        # kB = 1
        v_th = np.sqrt(2 * kB * self.params.temperature / self.params.mass)
        # print("!$! IC!", self.params.temperature)
        # print(self.params.mass)
        # print(v_th)

        if self.params.temperature <= 0:
            return 0.0, 0.0, 0.0
    
        R1, R2 = np.random.rand(2)
        R3, R4 = np.random.rand(2)

        # TODO: WHY MULTIPLIER WHERES THE MISSING LINK
        vx = v_th * np.sqrt(-1 * np.log(R1)) * np.cos(2 * np.pi * R2) / self.ref.v
        vy = v_th * np.sqrt(-1 * np.log(R1)) * np.sin(2 * np.pi * R2) / self.ref.v
        vz = v_th * np.sqrt(-1 * np.log(R3)) * np.cos(2 * np.pi * R4) / self.ref.v  

        # print(vx, vy, vz)

        return vx, vy, vz