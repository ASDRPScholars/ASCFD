from ascfd.particle.constants import ParticleConstants
from ascfd.inputs import Inputs
from ascfd.plasma_refs import PlasmaReferences

import numpy as np

class ParticleInitialConditions:
    def __init__(self, species, a_inputs, species_params: Inputs, constants):
        self.species = species
        self.inp = a_inputs
        self.params = species_params
        self.pc = constants
        
    def apply_ics(self, weight=None):
      
        if ic is not None:
            ic = ic
        else:
            ic = self.inp.i_ics
            
        if ic == "origin":
            self.species.particles[:] = self.origin()
        elif ic == "random":
            self.species.particles[:] = self.random()
        elif ic == "random_left_wall":
            self.species.particles[:] = self.random_left_wall()
        else:
            assert ValueError(f"[PARTICLE ICS] system [{ic}] not supported!")

        
    def origin(self):
        ic_particles = np.ones_like(self.particles) * 0.5
        
        return ic_particles
            
    def random(self):
        
        dx = self.inp.dx
        dy = self.inp.dy
        
        if self.params.type == "i":
            weight = self.inp.n_i * dx * dy
        elif self.params.type == "n":
            weight = self.inp.n_n * dx * dy
            
        ic_particles = np.zeros_like(self.species.particles)
    
        for i in range(self.inp.nx):
            for j in range(self.inp.ny):
                for n_p in range(self.inp.n_ppc):
                    random_x, random_y = np.random.uniform(i*dx, (i+1)*dx) + self.inp.xlim[0], np.random.uniform(j*dx, (j+1)*dy) + self.inp.ylim[0]
                    vx, vy, vz = self.sample_maxwellian_velocity()

                    n = (i * self.inp.ny) + j 
                    
                    ic_particles[self.pc.XCOMP, n] = random_x
                    ic_particles[self.pc.YCOMP, n] = random_y
                    ic_particles[self.pc.UCOMP, n] = vx
                    ic_particles[self.pc.VCOMP, n] = vy
                    
                    if weight is not None:
                        ic_particles[self.pc.WEIGHT, n] = weight
                        
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
        k_B = self.pc.k_B

        v_th = np.sqrt(2 * k_B * self.params.temperature / self.params.mass)

        if self.params.temperature <= 0:
            return 0.0, 0.0, 0.0
    
        R1, R2 = np.random.rand(2)
        R3, R4 = np.random.rand(2)

        # TODO: WHY MULTIPLIER WHERES THE MISSING LINK
        vx = v_th * np.sqrt(-1 * np.log(R1)) * np.cos(2 * np.pi * R2) 
        vy = v_th * np.sqrt(-1 * np.log(R1)) * np.sin(2 * np.pi * R2) 
        vz = v_th * np.sqrt(-1 * np.log(R3)) * np.cos(2 * np.pi * R4) 

        # print(vx, vy, vz)

        return vx, vy, vz