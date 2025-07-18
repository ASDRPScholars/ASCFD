from ascfd.particle.constants import ParticleConstants
from ascfd.particle.ics import ParticleInitialConditions
from ascfd.inputs import Inputs
from ascfd.params import SpeciesParams
from ascfd.fields.fields import Fields

import numpy as np

class ParticleSpecies:
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields):
        self.pc = ParticleConstants()
        self.fields = fields
        
        self.inp = a_inputs
        self.params = params
        self.dt = None
        
        self.particles = np.zeros((self.pc.NUMQ, self.inp.n_particles))
        
        self.ics = ParticleInitialConditions(self.particles, self.inp)
        
        self.particles = self.ics.apply_ics()
        print("PARTICLES X POS INIT:", self.particles[self.pc.XCOMP])
    
    
    # TODO: particles.py stuff goes in here
    def update(self):

        self.V = np.array(self.particles[self.pc.UCOMP], self.particles[self.pc.VCOMP], self.particles[self.pc.WCOMP])
        
        # this acc doesn't fit very well with the new multispecies structure lol...
        if self.inp.particle_flow_type == "passive":
            pass
        
        elif self.inp.particle_flow_type == "multispecies":
            for n in range (self.inp.n_particles):
                
                # TODO: XCOMP -> i, YCOMP -> j interpolation logic
                i = 0
                j = 0
                
                if self.params.type == "n":
                    self.ionize()
                # TODO: add other source terms too...
                
            self.calculate_mom_source_terms(i, j, n)      
            self.apply_mom_source_terms()
        
        charge_density = self.get_charge_density()
        
        # ELECTRIC FIELD UPDATE
        self.fields.add_charge_density(charge_density)
        self.fields.update_E()
        
        
    def calculate_mom_source_terms(self, i, j, n):
        s_lorentz = (self.params.charge / self.params.mass) * (self.fields.E[i, j] + np.cross(self.V[n], self.fields.B[i, j])) # q/m * (E + VxB)
        self.V[n] += s_lorentz
        
        
    def apply_mom_source_terms(self):
        
        u = self.V[0]
        v = self.V[1]
        w = self.V[2]
        
        self.particles[self.pc.UCOMP] = u
        self.particles[self.pc.VCOMP] = v
        self.particles[self.pc.WCOMP] = w
        
        
    def update_energy(self):
        pass
    
    
    def ionize(self):
        for n in range (self.inp.n_particles):
            # make methods to find these values depending on position (could use grid also similar to how particle cell calculation was originally done)
            electron_density = ... 
            neutral_density = ...

            ionization_rate = ... # another method to calculuate

            # statistics now !!!!

            p_ionize = 1 - np.exp(-ionization_rate * self.dt)

            rand = np.random.rand()

            if rand > p_ionize: # quick exit case: we will NOT ionize
                continue
        
            # self.particles.
            # particles.pop(neutral_particle_index)

            # create new ion in its place with correct values etc...

            # create_particle()
    
    def get_charge_density(self):
        pass
    
    
    def check_particles(self):
        pass
    
    
    