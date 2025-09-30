from ascfd.fluid.species import FluidSpecies
from ascfd.params import SpeciesParams
from ascfd.inputs import Inputs
from ascfd.fields.fields import Fields

class QNFluidSpecies(FluidSpecies):
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields, simulation):
        super().__init__(params, a_inputs, fields, simulation)
        
    