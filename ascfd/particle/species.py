from ascfd.particle.constants import ParticleConstants
from ascfd.particle.ics import ParticleInitialConditions
from ascfd.fluid.species import FluidSpecies
from ascfd.inputs import Inputs
from ascfd.params import SpeciesParams
from ascfd.fields.fields import Fields

import numpy as np
from scipy.spatial import cKDTree

class ParticleSpecies:
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields, electrons: FluidSpecies, neutrals: "ParticleSpecies"):
        self.pc = ParticleConstants()
        self.fields = fields
        self.electrons = electrons
        self.neutrals = neutrals
        
        self.inp = a_inputs
        self.params = params
        self.dt = None
        
        self.particles = np.zeros((self.pc.NUMQ, self.inp.n_particles))
        
        self.ics = ParticleInitialConditions(self.particles, self.inp)
        
        self.ics.apply_ics()
        
        print("PARTICLES X POS INIT:", self.particles[self.pc.XCOMP])
    
    
    # TODO: particles.py stuff goes in here
    def update(self):

        self.V = np.array([
                    self.particles[self.pc.UCOMP], 
                    self.particles[self.pc.VCOMP], 
                    self.particles[self.pc.WCOMP]
                    ])
        
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
        
        
    # TODO: PLACEHOLDER ADD PARTICLE (before we migrate to lists idk)
    def add_particle(self):
        pass
        
        
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
            electrons_density = self.electrons.get_number_density()
            neutrals_density = self.neutrals.get_number_density()
            
            c = 1e-13

            ionization_rate = c * electrons_density * neutrals_density # another method to calculuate

            # statistics now !!!!

            p_ionize = 1 - np.exp(-ionization_rate * self.dt)

            rand = np.random.rand()

            if rand > p_ionize: # quick exit case: we will NOT ionize
                continue
            
            # TODO: PLACEHOLDER
            self.add_particle()
        
            # self.particles.
            # particles.pop(neutral_particle_index)

            # create new ion in its place with correct values etc...

            # create_particle()
    
    def get_charge_density(self):
        charge_density = self.params.charge * self.get_number_density()
        return charge_density
    
    
    #this might be a bit faster but idk
    def get_number_density(self):
        positions = self.particles[self.c.XCOMP:self.c.YCOMP]
        tree = cKDTree(positions)
        
        #TODO: adjust local radius as necessary
        radius = 0.01
        
        counts = tree.query_ball_point(positions, r=radius, return_length=True)
        area = np.pi * (radius ** 2)
        
        number_density = counts / area
        return number_density
    

    def get_number_density_linear_search(self):
        count = 0
        radius = 0.01
        
        for n in range(self.inp.n_particles):
            dist = np.linalg.norm(self.particles[self.c.XCOMP:self.c.YCOMP, n])
            if dist < radius:
                count += 1
                
        area = np.pi * (radius ** 2)
        
        number_density = count / area
        return number_density
    
    
    def check_particles(self):
        pass
    
    
    