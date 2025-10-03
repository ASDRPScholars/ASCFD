from ascfd.fluid.species import FluidSpecies
from ascfd.params import SpeciesParams
from ascfd.inputs import Inputs
from ascfd.fields.fields import Fields

class QNFluidSpecies(FluidSpecies):
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields, simulation):
        super().__init__(params, a_inputs, fields, simulation)
        
    # TODO: DOUBLE CHECK WE'RE HANDLING GHOSTS HERE RIGHT CUZ THIS ALWAYS TRIPS ME UP
    
    def update(self):
        """Update electron fluid with ion continuity, Ohm's law momentum, and Euler energy flux. (vectorized)"""
        
        ng = self.inp.ng
        U = self.grid
        _, right_flux, left_flux, top_flux, bottom_flux = self.flux.getFlux(U, self.inp.nx, self.inp.ny, self.inp.ng)
        
        for icomp in range(self.c.NUMQ):
            # TODO: make sure in the WHOLE CODE that ONLY grids have ghost cells - everything else that DOESN'T PHYSICALLY NEED THEM should NOT have them
            if icomp == self.c.NCOMP:
                U[icomp, ng:-ng, ng:-ng] = self.simulation.get_species_number_density("i")[ng:-ng, ng:-ng]
                
            elif icomp == self.c.UCOMP:
                # j_e = q_e * n_e * u_e
                
                # TODO: implement actual ion current
                j_i = self.simulation.get_species_number_density("i")[ng:-ng, ng:-ng] * self.inp.q * 20
                q_e = self.params.charge
                n_e = U[self.c.NCOMP, ng:-ng, ng:-ng]
                
                U[icomp, ng:-ng, ng:-ng] = -j_i / (q_e * n_e)
                
            elif icomp == self.c.TCOMP:
                delta = (self.dt / self.inp.dx) * (right_flux[icomp, ng:-ng, ng:-ng] - left_flux[icomp, ng:-ng, ng:-ng]) + \
                        (self.dt / self.inp.dy) * (top_flux[icomp, ng:-ng, ng:-ng] - bottom_flux[icomp, ng:-ng, ng:-ng])
                    
                U[icomp, ng:-ng, ng:-ng] -= delta
                
                # ACTUAL TODO: add other source terms later
                
                # TODO: complete implement based on mikellides 2012 eq. (25)
                # do necessary conversion between E and n * k_B * T_e?
                # EDIT FLUX DEFINED IN EULER?
                
                # U[icomp, ng:-ng, ng:-ng] -= delta
                # add heat flux source terms