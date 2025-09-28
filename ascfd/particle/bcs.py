from ascfd.particle.constants import ParticleConstants
from ascfd.inputs import Inputs
from ascfd.plasma_refs import PlasmaReferences

import numpy as np

class ParticleBoundaryConditions:
    def __init__(self, particle_species, a_inputs: Inputs, params):
        self.particle_species = particle_species
        self.inp = a_inputs
        self.ref = PlasmaReferences()
        self.pc = ParticleConstants()
        self.params = params
        
        
    def apply_bcs(self):
        #TODO: readd boundaries that aren't at the bottom corner lol
        pass
        self.apply_inflow_lo()
        # self.remove_particles()
        
        
    def apply_inflow_lo(self):
        """Apply inflow boundary condition with proper n_ppc seeding"""
        WEIGHT = self.pc.NUMQ
        
        weight = self.inp.n_n * self.inp.v_n * self.inp.ylim[1] * self.inp.dt / self.inp.n_ppc
        # weight = 1000000
        
        #TODO: kB = 1.380649e-23
        kB = 1
        v_th = np.sqrt(2 * kB * self.params.temperature / self.params.mass)
        
        new_particles = []
        
        n_inject = self.inp.n_flow_rate * self.inp.dt / self.inp.m_n
        n_p_inject = n_inject / weight
        
        # Create n_ppc particles per boundary cell to match target density
        for j in range(self.inp.ny):
            for p in range(int(n_p_inject)):  # Critical fix: create n_ppc particles per cell
                particle_data = np.zeros(self.pc.NUMQ + 1)
                
                x_rand = np.random.uniform(0, self.inp.xlim[1] / 10)
                y_rand = np.random.uniform(0, self.inp.ylim[1])
                            
                R1, R2 = np.random.rand(2)
                R3, R4 = np.random.rand(2)

                vx = v_th * np.sqrt(-1 * np.log(R1)) * np.cos(2 * np.pi * R2)
                vy = v_th * np.sqrt(-1 * np.log(R1)) * np.sin(2 * np.pi * R2)
                vz = v_th * np.sqrt(-1 * np.log(R3)) * np.cos(2 * np.pi * R4)
                
                particle_data[self.pc.XCOMP] = x_rand
                
                print("!N! NEW X IS", x_rand, "STORED AS", particle_data[self.pc.XCOMP])
                particle_data[self.pc.YCOMP] = y_rand
                
                # TODO: WHY MULTIPLIER WHERES THE MISSING LINK
                particle_data[self.pc.UCOMP] = 150 / self.ref.v
                particle_data[self.pc.VCOMP] = 0 # vy + drift
                particle_data[WEIGHT] = weight
                
                if self.pc.WCOMP < self.pc.NUMQ:
                    particle_data[self.pc.WCOMP] = vz
                    
                # Use efficient particle addition instead of np.hstack
                print("!N! ABOUT TO ADD PARTICLE WITH X, Y=", particle_data[self.pc.XCOMP], particle_data[self.pc.YCOMP])
                self.particle_species.add_particle(particle_data)
                print("!N! AFTER ADD_PARTICLE, ALL STORED X, Y:", self.particle_species.particles[self.pc.XCOMP, :10], self.particle_species.particles[self.pc.YCOMP, :10])

        print("!!ADD PARTICLE!! FOR", self.params.type, len(new_particles))
    
    
    def remove_particles(self):
        pass