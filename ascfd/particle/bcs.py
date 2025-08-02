from ascfd.particle.constants import ParticleConstants
from ascfd.inputs import Inputs

import numpy as np

class ParticleBoundaryConditions:
    def __init__(self, particle_species, a_inputs: Inputs, params):
        self.particle_species = particle_species
        self.inp = a_inputs
        self.pc = ParticleConstants()
        self.params = params
        
        
    def apply_bcs(self):
        self.apply_inflow_lo()
        self.remove_particles()
        
        
    def apply_inflow_lo(self):
        """Apply inflow boundary condition with proper n_ppc seeding"""
        # Create n_ppc particles per boundary cell to match target density
        WEIGHT = self.pc.NUMQ
        weight = self.params.density * self.inp.dx * self.inp.dy / self.inp.n_ppc
        
        kB = 1
        v_th = np.sqrt(2 * kB * self.params.temperature / self.params.mass)
        
        # Loop over each y-cell and create n_ppc particles per cell
        for j in range(self.inp.ny):
            for p in range(self.inp.n_ppc):  # Create n_ppc particles per cell!
                particle_data = np.zeros(self.pc.NUMQ + 1)
                
                # Distribute particles randomly within the cell
                x_offset = np.random.uniform(0.1, 0.9)  # Within left boundary cell
                y_offset = np.random.uniform(-0.49, 0.49)  # Within y-cell
                
                # Maxwell-Boltzmann velocity distribution for each particle
                R1, R2 = np.random.rand(2)
                R3, R4 = np.random.rand(2)

                vx = v_th * np.sqrt(-1 * np.log(R1)) * np.cos(2 * np.pi * R2)
                vy = v_th * np.sqrt(-1 * np.log(R1)) * np.sin(2 * np.pi * R2)
                vz = v_th * np.sqrt(-1 * np.log(R3)) * np.cos(2 * np.pi * R4)
                
                # Position and velocity assignment
                particle_data[self.pc.XCOMP] = x_offset * self.inp.dx
                particle_data[self.pc.YCOMP] = (j + y_offset) * self.inp.dy
                particle_data[self.pc.UCOMP] = vx + 20  # Bulk flow velocity
                particle_data[self.pc.VCOMP] = vy
                particle_data[WEIGHT] = weight
                
                if self.pc.WCOMP < self.pc.NUMQ:
                    particle_data[self.pc.WCOMP] = vz
                    
                # Add each particle
                self.particle_species.add_particle(particle_data)
    
    
    def remove_particles(self):
        pass