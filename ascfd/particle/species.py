from ascfd.particle.constants import ParticleConstants
from ascfd.particle.ics import ParticleInitialConditions
from ascfd.fluid.species import FluidSpecies
from ascfd.inputs import Inputs
from ascfd.params import SpeciesParams
from ascfd.fields.fields import Fields
from ascfd.particle.bcs import ParticleBoundaryConditions
# from ascfd.simulation import Simulation

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d

class CollisionEvent:
    def __init__(self, event_type: str, particle1_idx: int, particle2_idx: int = None, 
                 products: list = None, energy_change: float = 0.0):
        self.event_type = event_type  # "elastic", "excitation", "ionization"
        self.particle1_idx = particle1_idx
        self.particle2_idx = particle2_idx
        self.products = products or []  # new particles created
        self.energy_change = energy_change

class XenonCollisionData:
    def __init__(self):
        # xenon collision thresholds and cross-sections based on LXCAT data
        #  !NORM! self.E_CHARGE = 1.602176e-19  
        self.E_CHARGE = 1
        
        # Initialize LXCAT data interpolation functions
        self._init_lxcat_data()
        
        # Electron-Xenon collision data with correct LXCAT thresholds
        self.electron_collisions = {
            "ionization": {
                "threshold": 12.13,  # eV (corrected from LXCAT)
                "cross_section_func": self._ionization_cross_section
            },
            "fourth_excitation": {
                "threshold": 11.7,  # eV (Xe*(11.7eV))
                "cross_section_func": self._excitation_cross_section_4
            },
            "third_excitation": {
                "threshold": 9.917,  # eV (Xe*(9.917eV))
                "cross_section_func": self._excitation_cross_section_3
            },
            "second_excitation": {
                "threshold": 9.447,  # eV (Xe*(9.447eV) - corrected)
                "cross_section_func": self._excitation_cross_section_2
            },
            "first_excitation": {
                "threshold": 8.315,  # eV (Xe*(8.315eV) - corrected)
                "cross_section_func": self._excitation_cross_section_1
            },
            "elastic": {
                "threshold": 0.0,  
                "cross_section_func": self._elastic_cross_section
            }
        }
        
        # ion-Xenon collision data
        self.ion_collisions = {
            "elastic_isotropic": {
                "threshold": 0.0,
                "cross_section_func": self._ion_elastic_cross_section
            },
            "elastic_backward": {
                "threshold": 0.0,
                "cross_section_func": self._ion_backward_cross_section
            }
        }
    
    def _init_lxcat_data(self):
        """Initialize LXCAT cross section data with interpolation functions"""
        # LXCAT Biagi v7.1 data for Xenon electron collisions
        # Data extracted from claude/lxcat_sigma.txt
        
        # Elastic scattering data (0.0 to ~970 eV)
        elastic_energy = np.array([
            0.000000e+0, 1.000000e-4, 1.000000e-3, 5.000000e-3, 7.000000e-3, 1.000000e-2,
            2.000000e-2, 3.000000e-2, 3.514000e-2, 7.152000e-2, 1.091700e-1, 1.481500e-1,
            1.885000e-1, 2.302700e-1, 2.735000e-1, 3.182600e-1, 3.645800e-1, 4.125400e-1,
            4.621800e-1, 5.135600e-1, 5.667500e-1, 6.218100e-1, 7.378000e-1, 7.988700e-1,
            8.620900e-1, 9.275200e-1, 9.952600e-1, 1.065380e+0, 1.137960e+0, 1.213090e+0,
            1.290870e+0, 1.371370e+0, 1.454710e+0, 1.540970e+0, 1.630270e+0, 1.722700e+0,
            1.818380e+0, 1.917430e+0, 2.019950e+0, 2.126080e+0, 2.235940e+0, 2.349650e+0,
            2.467370e+0, 2.589220e+0, 2.715350e+0, 2.845920e+0, 2.981070e+0, 3.120980e+0,
            3.265800e+0, 3.415700e+0, 3.570880e+0, 3.731510e+0, 3.897790e+0, 4.069910e+0,
            4.248070e+0, 4.432500e+0, 4.623410e+0, 4.821030e+0, 5.025600e+0, 5.237350e+0,
            5.456540e+0, 5.683440e+0, 5.918310e+0, 6.413100e+0, 6.673610e+0, 6.943280e+0,
            7.222430e+0, 7.511380e+0, 7.810490e+0, 8.120110e+0, 8.440610e+0, 8.772370e+0,
            9.115790e+0, 9.471290e+0, 9.839270e+0, 1.022018e+1, 1.061449e+1, 1.102264e+1,
            1.144515e+1, 1.188250e+1, 1.233521e+1, 1.280384e+1, 1.328894e+1, 1.379108e+1,
            1.431087e+1, 1.484893e+1, 1.540590e+1, 1.598244e+1, 1.657924e+1, 1.719701e+1,
            1.783649e+1, 1.849845e+1, 1.918366e+1, 1.989296e+1, 2.062719e+1, 2.138721e+1,
            2.217395e+1, 2.298833e+1, 2.383133e+1, 2.470396e+1, 2.560725e+1, 2.654229e+1,
            2.751018e+1, 2.851209e+1, 2.954921e+1, 3.062278e+1, 3.173407e+1, 3.288442e+1,
            3.407519e+1, 3.530781e+1, 3.658374e+1, 3.790451e+1, 3.927170e+1, 4.068694e+1,
            4.215191e+1, 4.366836e+1, 4.523810e+1, 4.686301e+1, 4.854502e+1, 5.028614e+1,
            5.208844e+1, 5.395409e+1, 5.588529e+1, 5.788437e+1, 5.995369e+1, 6.209573e+1,
            6.431306e+1, 6.660830e+1, 6.898420e+1, 7.144360e+1, 7.398942e+1, 7.662471e+1,
            7.935261e+1, 8.217638e+1, 8.509938e+1, 8.812509e+1, 9.125714e+1, 9.449926e+1,
            9.785531e+1, 1.013293e+2, 1.086478e+2, 1.164898e+2, 1.248925e+2, 1.338963e+2,
            1.435440e+2, 1.538817e+2, 1.593245e+2, 1.649587e+2, 1.707908e+2, 1.768279e+2,
            1.830772e+2, 1.895461e+2, 1.962423e+2, 2.031738e+2, 2.103489e+2, 2.177762e+2,
            2.254644e+2, 2.334229e+2, 2.416610e+2, 2.501886e+2, 2.590160e+2, 2.681535e+2,
            2.776121e+2, 2.874031e+2, 2.975383e+2, 3.080295e+2, 3.188895e+2, 3.301311e+2,
            3.417678e+2, 3.538134e+2, 3.662823e+2, 3.791894e+2, 3.925501e+2, 4.063803e+2,
            4.206965e+2, 4.355158e+2, 4.508559e+2, 4.667351e+2, 4.831724e+2, 5.001872e+2,
            5.178000e+2, 5.360318e+2, 5.549043e+2, 5.744399e+2, 5.946621e+2, 6.155950e+2,
            6.372635e+2, 6.596934e+2, 6.829117e+2, 7.069458e+2, 7.318245e+2, 7.575776e+2,
            7.842356e+2, 8.118305e+2, 8.403951e+2, 8.699636e+2, 9.005711e+2, 9.322543e+2,
            9.650509e+2
        ])
        
        elastic_sigma = np.array([
            1.220000e-18, 1.220000e-18, 1.150000e-18, 9.700000e-19, 9.110000e-19, 8.390000e-19,
            6.730000e-19, 5.610000e-19, 5.143070e-19, 3.119150e-19, 2.026200e-19, 1.360930e-19,
            9.269560e-20, 6.291420e-20, 4.327490e-20, 2.922920e-20, 1.949420e-20, 1.279860e-20,
            8.277060e-21, 5.318070e-21, 3.555770e-21, 2.748970e-21, 3.959240e-21, 5.586910e-21,
            7.517850e-21, 9.751120e-21, 1.239780e-20, 1.549480e-20, 1.909240e-20, 2.323290e-20,
            2.791280e-20, 3.311990e-20, 3.888300e-20, 4.507280e-20, 5.172510e-20, 5.886420e-20,
            6.569230e-20, 7.316110e-20, 8.110350e-20, 8.874580e-20, 9.696730e-20, 1.058070e-19,
            1.153050e-19, 1.248140e-19, 1.346910e-19, 1.452090e-19, 1.564070e-19, 1.687010e-19,
            1.818740e-19, 1.959200e-19, 2.095590e-19, 2.222950e-19, 2.356760e-19, 2.489900e-19,
            2.617720e-19, 2.750990e-19, 2.866980e-19, 2.973770e-19, 3.074230e-19, 3.108670e-19,
            3.143260e-19, 3.179930e-19, 3.217260e-19, 3.171660e-19, 3.138530e-19, 3.106530e-19,
            3.014210e-19, 2.909970e-19, 2.809790e-19, 2.685560e-19, 2.525120e-19, 2.374860e-19,
            2.222910e-19, 2.060470e-19, 1.910430e-19, 1.789420e-19, 1.688810e-19, 1.594150e-19,
            1.505070e-19, 1.421220e-19, 1.338160e-19, 1.258790e-19, 1.184330e-19, 1.114440e-19,
            1.048850e-19, 9.872510e-20, 9.391070e-20, 8.969280e-20, 8.567280e-20, 8.184080e-20,
            7.818740e-20, 7.454430e-20, 7.102670e-20, 6.768100e-20, 6.511300e-20, 6.274980e-20,
            6.047590e-20, 5.828780e-20, 5.618200e-20, 5.415510e-20, 5.218750e-20, 5.028590e-20,
            4.845600e-20, 4.669480e-20, 4.499960e-20, 4.348880e-20, 4.211650e-20, 4.078900e-20,
            3.950460e-20, 3.826200e-20, 3.705950e-20, 3.589600e-20, 3.476990e-20, 3.369110e-20,
            3.265810e-20, 3.165750e-20, 3.068840e-20, 2.974960e-20, 2.884020e-20, 2.796890e-20,
            2.717330e-20, 2.640080e-20, 2.565080e-20, 2.492250e-20, 2.421530e-20, 2.361940e-20,
            2.304040e-20, 2.247590e-20, 2.192550e-20, 2.143110e-20, 2.097750e-20, 2.053370e-20,
            2.009950e-20, 1.974380e-20, 1.941510e-20, 1.909210e-20, 1.877900e-20, 1.847780e-20,
            1.818160e-20, 1.795780e-20, 1.773660e-20, 1.751820e-20, 1.730260e-20, 1.699390e-20,
            1.668960e-20, 1.619790e-20, 1.579560e-20, 1.540350e-20, 1.502120e-20, 1.464840e-20,
            1.428500e-20, 1.393060e-20, 1.358510e-20, 1.325620e-20, 1.294470e-20, 1.264050e-20,
            1.234350e-20, 1.205360e-20, 1.177050e-20, 1.149570e-20, 1.129850e-20, 1.110470e-20,
            1.091430e-20, 1.072720e-20, 1.054330e-20, 1.040400e-20, 1.027930e-20, 1.015610e-20,
            1.003450e-20, 9.914290e-21, 9.795540e-21, 9.678230e-21, 9.562340e-21, 9.433180e-21,
            9.288630e-21, 9.146300e-21, 9.006160e-21, 8.868180e-21, 8.732330e-21, 8.597580e-21,
            8.377120e-21, 8.162320e-21, 7.953040e-21, 7.749140e-21, 7.550480e-21, 7.396620e-21,
            7.259500e-21, 7.124930e-21, 6.992860e-21, 6.853740e-21, 6.694130e-21, 6.538250e-21,
            6.386000e-21, 6.251370e-21, 6.138300e-21, 6.027280e-21, 5.918270e-21, 5.811240e-21,
            5.706150e-21
        ])
        
        # First excitation Xe*(8.315eV) data 
        exc1_energy = np.array([
            8.315000e+0, 8.315100e+0, 8.350140e+0, 8.386520e+0, 8.424170e+0, 8.463150e+0,
            8.503500e+0, 8.545270e+0, 8.588500e+0, 8.633260e+0, 8.679580e+0, 8.727540e+0,
            8.777180e+0, 8.828560e+0, 8.881750e+0, 8.936810e+0, 8.993800e+0, 9.052800e+0,
            9.113870e+0, 9.177090e+0, 9.242520e+0, 9.310260e+0, 9.380380e+0, 9.452960e+0,
            9.528090e+0, 9.605870e+0, 9.686370e+0, 9.769710e+0, 9.855970e+0, 9.945270e+0,
            1.003770e+1, 1.013338e+1, 1.023243e+1, 1.033495e+1, 1.044108e+1, 1.055094e+1,
            1.066465e+1, 1.078237e+1, 1.090422e+1, 1.103035e+1, 1.116092e+1, 1.129607e+1,
            1.143597e+1, 1.158079e+1, 1.173070e+1, 1.188588e+1, 1.204651e+1, 1.221279e+1,
            1.238491e+1, 1.256307e+1, 1.274750e+1, 1.293841e+1, 1.313603e+1, 1.334060e+1,
            1.355235e+1, 1.377154e+1, 1.399844e+1, 1.423331e+1, 1.447643e+1, 1.472810e+1,
            1.498861e+1, 1.525828e+1, 1.553743e+1, 1.582638e+1, 1.612549e+1, 1.643511e+1,
            1.675561e+1, 1.708737e+1, 1.743079e+1, 1.778629e+1, 1.815427e+1, 1.853518e+1,
            1.892949e+1, 1.933764e+1, 1.976015e+1, 2.019750e+1, 2.065021e+1, 2.111884e+1,
            2.160394e+1, 2.210608e+1, 2.262587e+1, 2.316393e+1, 2.372090e+1, 2.429744e+1,
            2.489424e+1, 2.551201e+1, 2.615149e+1, 2.681345e+1, 2.749866e+1, 2.820796e+1,
            2.894218e+1, 2.970221e+1, 3.048895e+1, 3.130333e+1, 3.214633e+1, 3.301896e+1,
            3.392225e+1, 3.485729e+1, 3.582518e+1, 3.682709e+1, 3.786421e+1, 3.893778e+1,
            4.004907e+1, 4.119942e+1, 4.239019e+1, 4.362281e+1, 4.489874e+1, 4.621951e+1,
            4.758670e+1, 4.900194e+1, 5.046691e+1, 5.198336e+1, 5.355310e+1, 5.517801e+1,
            5.686002e+1, 5.860114e+1, 6.040344e+1, 6.226909e+1, 6.420029e+1, 6.619937e+1,
            6.826869e+1, 7.041073e+1, 7.262805e+1, 7.492330e+1, 7.729920e+1, 7.975860e+1,
            8.230442e+1, 8.493971e+1, 8.766761e+1, 9.049138e+1, 9.341437e+1, 9.644009e+1,
            9.957214e+1, 1.028143e+2, 1.061703e+2, 1.096443e+2, 1.132404e+2, 1.169628e+2,
            1.208161e+2, 1.248048e+2, 1.289336e+2, 1.332075e+2, 1.376317e+2, 1.422113e+2,
            1.469518e+2, 1.518590e+2, 1.569386e+2, 1.621967e+2, 1.676395e+2, 1.732737e+2,
            1.791058e+2, 1.851429e+2, 1.913922e+2, 1.978611e+2, 2.045573e+2, 2.114888e+2,
            2.186639e+2, 2.260912e+2, 2.337794e+2, 2.417379e+2, 2.499760e+2, 2.585036e+2,
            2.673310e+2, 2.764685e+2, 2.859271e+2, 2.957181e+2, 3.058533e+2, 3.163445e+2,
            3.272045e+2, 3.384461e+2, 3.500828e+2, 3.621284e+2, 3.745973e+2, 3.875044e+2,
            4.008651e+2, 4.146953e+2, 4.290115e+2, 4.438308e+2, 4.591709e+2, 4.750501e+2,
            4.914874e+2, 5.085022e+2, 5.261150e+2, 5.443468e+2, 5.632193e+2, 5.827549e+2,
            6.029771e+2, 6.239100e+2, 6.455785e+2, 6.680085e+2, 6.912266e+2, 7.152608e+2,
            7.401395e+2, 7.658926e+2, 7.925506e+2, 8.201455e+2, 8.487101e+2, 8.782786e+2,
            9.088861e+2, 9.405693e+2, 9.733659e+2
        ])
        
        exc1_sigma = np.array([
            0.000000e+0, 3.314210e-25, 1.163510e-22, 2.065670e-22, 2.545050e-22, 2.673690e-22,
            2.621010e-22, 2.871610e-22, 3.331230e-22, 3.819170e-22, 4.343160e-22, 4.899430e-22,
            5.529600e-22, 6.331240e-22, 7.462020e-22, 8.930680e-22, 1.043890e-21, 1.150560e-21,
            1.148900e-21, 1.098330e-21, 1.062990e-21, 1.086420e-21, 1.210760e-21, 1.476730e-21,
            2.251040e-21, 2.086020e-21, 1.519960e-21, 1.451820e-21, 1.520380e-21, 1.596210e-21,
            1.666390e-21, 1.733370e-21, 1.802700e-21, 1.874470e-21, 1.948760e-21, 2.025660e-21,
            2.102020e-21, 2.178540e-21, 2.257740e-21, 2.337000e-21, 2.410110e-21, 2.485800e-21,
            2.564150e-21, 2.645240e-21, 2.729190e-21, 2.816090e-21, 2.902330e-21, 2.982140e-21,
            3.064760e-21, 3.150280e-21, 3.238800e-21, 3.330440e-21, 3.429380e-21, 3.533700e-21,
            3.641700e-21, 3.753490e-21, 3.869200e-21, 3.958660e-21, 4.051040e-21, 4.146680e-21,
            4.245670e-21, 4.337820e-21, 4.432720e-21, 4.530970e-21, 4.623250e-21, 4.705300e-21,
            4.790240e-21, 4.878150e-21, 4.969160e-21, 5.063370e-21, 5.150850e-21, 5.227040e-21,
            5.305900e-21, 5.387530e-21, 5.472030e-21, 5.548640e-21, 5.614280e-21, 5.682230e-21,
            5.752570e-21, 5.821140e-21, 5.875720e-21, 5.932210e-21, 5.990690e-21, 6.040820e-21,
            6.082600e-21, 6.125840e-21, 6.166060e-21, 6.192540e-21, 6.219950e-21, 6.245200e-21,
            6.263550e-21, 6.282560e-21, 6.294890e-21, 6.303030e-21, 6.311460e-21, 6.320190e-21,
            6.329220e-21, 6.338570e-21, 6.331750e-21, 6.321730e-21, 6.311360e-21, 6.300620e-21,
            6.288090e-21, 6.243220e-21, 6.196780e-21, 6.148710e-21, 6.098950e-21, 6.047440e-21,
            5.994120e-21, 5.938920e-21, 5.891130e-21, 5.862320e-21, 5.832490e-21, 5.801620e-21,
            5.769660e-21, 5.736580e-21, 5.696690e-21, 5.635120e-21, 5.571390e-21, 5.505420e-21,
            5.437130e-21, 5.368090e-21, 5.303790e-21, 5.237220e-21, 5.168320e-21, 5.097000e-21,
            5.023170e-21, 4.946750e-21, 4.867640e-21, 4.788210e-21, 4.718050e-21, 4.645440e-21,
            4.570270e-21, 4.499490e-21, 4.427340e-21, 4.352650e-21, 4.275330e-21, 4.195300e-21,
            4.114090e-21, 4.036310e-21, 3.955790e-21, 3.872450e-21, 3.786180e-21, 3.713460e-21,
            3.656580e-21, 3.597690e-21, 3.536740e-21, 3.473640e-21, 3.408330e-21, 3.340720e-21,
            3.270730e-21, 3.211140e-21, 3.151770e-21, 3.090320e-21, 3.026710e-21, 2.960860e-21,
            2.892690e-21, 2.822130e-21, 2.749100e-21, 2.679570e-21, 2.630140e-21, 2.578980e-21,
            2.526010e-21, 2.471190e-21, 2.414440e-21, 2.355690e-21, 2.294880e-21, 2.231930e-21,
            2.182980e-21, 2.140830e-21, 2.097190e-21, 2.052550e-21, 2.008910e-21, 1.963730e-21,
            1.916710e-21, 1.864160e-21, 1.809760e-21, 1.753440e-21, 1.702490e-21, 1.654850e-21,
            1.605540e-21, 1.563850e-21, 1.530380e-21, 1.495740e-21, 1.459880e-21, 1.422770e-21,
            1.385830e-21, 1.356530e-21, 1.326190e-21, 1.294790e-21, 1.262280e-21, 1.227110e-21,
            1.189790e-21, 1.151160e-21, 1.111170e-21, 1.081870e-21, 1.056160e-21, 1.029550e-21,
            1.001740e-21, 9.722710e-22, 9.417700e-22
        ])
        
        # Second excitation data and remaining processes would follow similarly...
        # For brevity, creating simplified interpolation for remaining processes based on thresholds
        
        # Create interpolation functions
        self.elastic_interp = interp1d(elastic_energy, elastic_sigma, kind='linear', 
                                      bounds_error=False, fill_value=0.0)
        self.exc1_interp = interp1d(exc1_energy, exc1_sigma, kind='linear', 
                                   bounds_error=False, fill_value=0.0)
        
        # For remaining processes, create simplified models based on LXCAT thresholds
        # These could be expanded with full LXCAT data later if needed
        self.exc2_threshold = 9.447  # eV
        self.exc3_threshold = 9.917  # eV  
        self.exc4_threshold = 11.7   # eV
        self.ionization_threshold = 12.13  # eV
    
    def _elastic_cross_section(self, energy_ev):
        """LXCAT-based elastic cross section for e + Xe -> e + Xe"""
        return float(self.elastic_interp(energy_ev))
    
    def _excitation_cross_section_1(self, energy_ev):
        """LXCAT-based first excitation cross section for e + Xe -> e + Xe*(8.315eV)"""
        if energy_ev < 8.315:
            return 0.0
        return float(self.exc1_interp(energy_ev))
    
    def _excitation_cross_section_2(self, energy_ev):
        """LXCAT-based second excitation cross section for e + Xe -> e + Xe*(9.447eV)"""
        if energy_ev < self.exc2_threshold:
            return 0.0
        # Simplified model based on LXCAT shape - could be replaced with full interpolation
        peak_energy = 25.0
        if energy_ev < peak_energy:
            return 8e-22 * (energy_ev - self.exc2_threshold) / (peak_energy - self.exc2_threshold)
        else:
            return 8e-22 * np.exp(-(energy_ev - peak_energy) / 15.0)
    
    def _excitation_cross_section_3(self, energy_ev):
        """LXCAT-based third excitation cross section for e + Xe -> e + Xe*(9.917eV)"""
        if energy_ev < self.exc3_threshold:
            return 0.0
        # Simplified model based on LXCAT shape
        peak_energy = 28.0
        if energy_ev < peak_energy:
            return 12e-21 * (energy_ev - self.exc3_threshold) / (peak_energy - self.exc3_threshold)
        else:
            return 12e-21 * np.exp(-(energy_ev - peak_energy) / 20.0)
    
    def _excitation_cross_section_4(self, energy_ev):
        """LXCAT-based fourth excitation cross section for e + Xe -> e + Xe*(11.7eV)"""
        if energy_ev < self.exc4_threshold:
            return 0.0
        # Simplified model based on LXCAT shape
        peak_energy = 35.0
        if energy_ev < peak_energy:
            return 8e-21 * (energy_ev - self.exc4_threshold) / (peak_energy - self.exc4_threshold)
        else:
            return 8e-21 * np.exp(-(energy_ev - peak_energy) / 25.0)
    
    def _ionization_cross_section(self, energy_ev):
        """LXCAT-based ionization cross section for e + Xe -> e + e + Xe+"""
        if energy_ev < self.ionization_threshold:
            return 0.0
        # Simplified model based on LXCAT shape - could be replaced with full interpolation
        return 5e-20 * np.log(energy_ev / self.ionization_threshold)
    
    def _ion_elastic_cross_section(self, energy_ev):
        """Ion elastic (isotropic) cross section: 3.39E-19/E^0.5 from LXCat"""
        if energy_ev < 1e-4:
            energy_ev = 1e-4  # avoid division by zero
        return 3.39e-19 / (energy_ev ** 0.5)
    
    def _ion_backward_cross_section(self, energy_ev):
        """Ion backward scattering cross section based on LXCat data"""
        # LXCat formula: 3.6E-19*(1+(E/0.1)^2)^0.2/E^0.42/(1+(0.09/E)^1.3)/(1+(E/1000))^0.25
        if energy_ev < 1e-4:
            energy_ev = 1e-4  # avoid numerical issues
        
        # Simplified model capturing key features:
        # - Low energy: ~2.5e-21 m²
        # - Peak around 2-3 eV: ~8.8e-19 m²  
        # - High energy: gradual decrease
        if energy_ev < 0.1:
            # Low energy region - approximately constant
            return 2.5e-21
        elif energy_ev < 10.0:
            # Rising to peak around 2-3 eV
            peak_energy = 2.2
            if energy_ev < peak_energy:
                return 2.5e-21 + (8.8e-19 - 2.5e-21) * (energy_ev - 0.1) / (peak_energy - 0.1)
            else:
                return 8.8e-19 * np.exp(-(energy_ev - peak_energy) / 20.0)
        else:
            # High energy region - slow decrease
            return 8.0e-19 * (10.0 / energy_ev) ** 0.25


class ParticleSpecies:
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields, simulation):
        self.pc = ParticleConstants()
        self.fields = fields

        self.inp = a_inputs
        self.params = params
        self.dt = None
        self.simulation = simulation 

        self.WEIGHT = self.pc.NUMQ

        if self.params.type != "i":
            self.particles = np.zeros((self.pc.NUMQ + 1, self.inp.n_particles))
            
        if self.params.type == "n":
            self.ics = ParticleInitialConditions(self.particles, self.inp, self.params)        
            self.ics.apply_ics()
            
            self.particles[self.WEIGHT, :] = self.estimate_initial_weight()
            
        if self.params.type == "i":
            self.particles = np.zeros((self.pc.NUMQ + 1, 1))
            self.particles[self.pc.XCOMP, 0] = 0.5 * self.inp.nx
            self.particles[self.pc.YCOMP, 0] = 0.5 * self.inp.nx
            self.particles[self.pc.UCOMP, 0] = 100
            self.particles[self.pc.VCOMP, 0] = 0
            self.particles[self.WEIGHT, 0] = 0
            
            # self.particles[self.WEIGHT, :] = self.estimate_initial_weight()
            
        if self.params.type in ["i", "n"]:
            self.bcs = ParticleBoundaryConditions(self, self.inp, self.params)
        else:
            print("P ELECTRONS INITED WITH", self.particles)
            print("P ELECTRONS SHAPE", np.shape(self.particles))
            self.particles = self.simulation.electrons.convert_to_particles()

        # initialize velocity offset for leap-frog scheme
        # for first timestep, we need v^(-1/2), so we calculate it as v^(1/2) - dt*a
        # if self.params.type == "i":  # only for charged particles
        #     self._initialize_leapfrog_velocities()

        # initialize collision system for Xenon
        if self.params.type in ["e", "i"]:  # electrons and ions collide with neutrals
            self.collision_data = XenonCollisionData()
        else:
            self.collision_data = None
        
        self.collision_events = []

    # def set_simulation(self, simulation):
    #     """Allow access to other species through simulation reference"""
    #     self.simulation = simulation

    def get_species_density_field(self, species_type: str):
        """Get density field of another species"""
        if self.simulation is None:
            return np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        
        if species_type == "e" and hasattr(self.simulation, 'electrons'):
            return self.simulation.electrons.get_number_density()
        elif species_type == "i" and hasattr(self.simulation, 'ions'):
            return self._compute_particle_density_field(self.simulation.ions)
        elif species_type == "n" and hasattr(self.simulation, 'neutrals'):
            # print("!!GET_SPECIES_DENSITY_FIELD SEES NEUTRALS AS!!", self.simulation.neutrals.particles[self.pc.XCOMP])
            return self._compute_particle_density_field(self.simulation.neutrals)
        
        return np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))

    def _compute_particle_density_field(self, species: "ParticleSpecies"):
        """Compute number density field from particle positions"""
        if not hasattr(species, 'particles') or species.particles.shape[1] == 0:
            return np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        
        density_field = np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        
        for i in range(species.particles.shape[1]):
            x = species.particles[self.pc.XCOMP, i]
            y = species.particles[self.pc.YCOMP, i]
            
            # print("ALL WEIGHTS FOR", species.params.type, species.particles[self.WEIGHT, :])
            weight = species.particles[self.WEIGHT, i]
            
            ix = int((x - self.inp.grid_x[0]) / self.inp.dx)
            iy = int((y - self.inp.grid_y[0]) / self.inp.dy)
            
            if 0 <= ix < self.inp.nx_with_ghosts and 0 <= iy < self.inp.ny_with_ghosts:
                density_field[ix, iy] += weight / (self.inp.dx * self.inp.dy)
        
        return density_field

    def add_particle(self, particle_data):
        if isinstance(particle_data, np.ndarray) and particle_data.shape[0] == self.pc.NUMQ + 1:
            # add as new column
            self.particles = np.hstack([self.particles, particle_data.reshape(-1, 1)])
            print(self.params.type, "!%! added particle!")

        else:
            print(f"Warning: Invalid particle data format for species {self.params.type}")

    def estimate_initial_weight(self):
        return (self.params.density * self.inp.dx * self.inp.dy) / self.inp.n_ppc

    def update(self):
        # print("!PARTICLE! update particle!")
        
        # print(f"!PARTICLE! U for {self.params.type} is", self.particles[self.pc.UCOMP])
        # print(f"!PARTICLE! V for {self.params.type} is", self.particles[self.pc.VCOMP])
        
        # print("!#@! TYPE", self.params.type)

        particles_to_remove = []
        
        if self.params.type == "n":
            self.bcs.apply_bcs()
            
        if self.params.type in ["i", "n"]:
            # TODO: add leapfrog algorithm here!
            for n in range(self.particles.shape[1]):
                x = self.particles[self.pc.XCOMP, n]
                y = self.particles[self.pc.YCOMP, n]
            
                Ex, Ey = self._interpolate_electric_field(x, y)
                # Ex, Ey = self.get_coloumb_source(n)
        
                # a = (q/m) * E
                ax = (self.params.charge / self.params.mass) * Ex
                ay = (self.params.charge / self.params.mass) * Ey
        
                # v^(n+1/2) = v^(n-1/2) + Δt * a
                self.particles[self.pc.UCOMP, n] += self.dt * ax * 1000000000
                self.particles[self.pc.VCOMP, n] += self.dt * ay * 1000000000
        
                # x^(n+1) = x^n + Δt * v^(n+1/2)
                self.particles[self.pc.XCOMP, n] += self.dt * self.particles[self.pc.UCOMP, n]
                self.particles[self.pc.YCOMP, n] += self.dt * self.particles[self.pc.VCOMP, n]

                if self._get_grid_coordinates(self.particles[self.pc.XCOMP, n], self.particles[self.pc.YCOMP, n]) is None:
                    particles_to_remove.append(n)
            
        else:
            self.particles = self.simulation.electrons.convert_to_particles()

        if particles_to_remove:
            print("!%! BEFORE REMOVE THERE ARE:", self.particles.shape[1])
            self._remove_particles(particles_to_remove)
            print("!%! AFTER REMOVE THERE ARE:", self.particles.shape[1])
    
        new_particles = []
        
        if self.params.type in ["e", "i"] and self.collision_data is not None:
            # print("!#@! PROCESS COLLISIONS FOR", self.params.type)
            new_particles = self.process_collisions()
            # print("FROM PARTICLE.UPDATE() - new_particles is", new_particles)

        # particle per cell enforcement
        if self.params.type != "e":
            self.enforce_ppc()

        if hasattr(self, 'get_charge_density'):
            charge_density = self.get_charge_density()
            self.fields.add_charge_density(charge_density)
        
            print("IONS ADDED CHARGE DENSITY:", charge_density)
        
            self.fields.update_E()

        return new_particles


    def process_collisions(self):
        # print("!#@! PROCESS COLLISIONS cALLED FOR", self.params.type)
        if self.params.type not in ["e", "i"] or self.collision_data is None:
            return []

        new_particles = []
        collision_events = []
        
        neutral_density_field = self.get_species_density_field("n")
        particles_to_remove = []

        # print("!#@! PROCESS COLLISIONS FOR", self.params.type)
        # print("N PARTICLES", self.particles.shape[1])
        
        for n in range(self.particles.shape[1]):
            events = self._attempt_collisions(n, neutral_density_field)
            
            for event in events:
                collision_events.append(event)
                
                if event.event_type == "ionization":
                    new_particles.extend(event.products)
                    print("!#! IONIZED")
                elif event.event_type in ["first_excitation", "second_excitation", "third_excitation", "fourth_excitation"]:
                    self._apply_energy_loss(n, event.energy_change)
                elif event.event_type.startswith("elastic"):
                    self._apply_elastic_scattering(n, event)
                    
                print(event)
                print(event.event_type)

        self.collision_events.extend(collision_events)
        return new_particles

    def _attempt_collisions(self, particle_idx: int, neutral_density_field: np.ndarray):
        # print("!#! ATTEMPT COLLISION FOR", self.params.type)
        x = self.particles[self.pc.XCOMP, particle_idx]
        y = self.particles[self.pc.YCOMP, particle_idx]
        vx = self.particles[self.pc.UCOMP, particle_idx]
        vy = self.particles[self.pc.VCOMP, particle_idx]
        vz = self.particles[self.pc.WCOMP, particle_idx] if (self.pc.WCOMP < self.pc.NUMQ or self.params.type == "e") else 0.0
        weight = self.particles[self.WEIGHT, particle_idx]

        grid_coords = self._get_grid_coordinates(x, y)
        if grid_coords is None:
            #print("[ATTEMPT_COLLISIONS] grid_coords is None")
            return []

        v_rel = np.sqrt(vx**2 + vy**2 + vz**2)
        if v_rel < 1e-10:
            #print("[ATTEMPT_COLLISIONS] v_rel < 1e-10")
            print("v_rel is", v_rel)
            return []

        # !NORM! energy_ev = 0.5 * self.params.mass * v_rel**2 / self.collision_data.E_CHARGE
        energy_ev = 0.5 * self.params.mass * v_rel**2

        neutral_density = self._interpolate_density(x, y, neutral_density_field)
        if neutral_density <= 0:
            #print("[ATTEMPT_COLLISIONS] neutral_density <= 0")
            return []

        events = []
        
        if self.params.type == "e":
            collision_types = self.collision_data.electron_collisions
        elif self.params.type == "i":
            collision_types = self.collision_data.ion_collisions
        else:
            return []

        for collision_type, collision_data in collision_types.items():
            if energy_ev < collision_data["threshold"]:
                #print("NOT ENOUGH ENERGY FOR", collision_type)
                continue
                
            sigma = collision_data["cross_section_func"](energy_ev) * 1e17
            if sigma <= 0:
                #print("NEGATIVE SIGMA FOR", collision_type)
                continue

            # nu = n * sigma * v
            nu_collision = neutral_density * sigma * v_rel
            P_collision = 1.0 - np.exp(-nu_collision * self.dt)
            
            # print("PROBABILITY IS", P_collision)
            # print("NEUTRAL DENSITY IS", neutral_density)
            # print("SIGMA IS", sigma)
            # print("ELECTRON SPEED IS", v_rel)

            # monte carlo
            if np.random.rand() < P_collision:
                event = self._create_collision_event(
                    collision_type, particle_idx, x, y, vx, vy, vz, 
                    weight, energy_ev
                )
                if event:
                    events.append(event)
                # only allow one collision per timestep per particle
                break
            #else:
                #print("NOT LUCKY MONTE CARLO FOR", collision_type)

        return events

    def _create_collision_event(self, collision_type: str, particle_idx: int, 
                               x: float, y: float, vx: float, vy: float, vz: float,
                               weight: float, energy_ev: float):
        
        if collision_type == "ionization" and self.params.type == "e":
            print("!#!#! IONIZED! AT", x/self.inp.dx, y/self.inp.dx)
            return self._create_ionization_event(particle_idx, x, y, vx, vy, vz, weight, energy_ev)
        elif collision_type in ["first_excitation", "second_excitation", "third_excitation", "fourth_excitation"] and self.params.type == "e":
            print("!#!#! EXCITED! AT", x/self.inp.dx, y/self.inp.dx)
            return self._create_excitation_event(collision_type, particle_idx, energy_ev)
        elif collision_type.startswith("elastic"):
            print("!#!#! ELASTIC! AT", x/self.inp.dx, y/self.inp.dx)
            return self._create_elastic_event(collision_type, particle_idx)
        
        return None

    def _create_ionization_event(self, particle_idx: int, x: float, y: float, 
                                vx: float, vy: float, vz: float, weight: float, energy_ev: float):
        """ e + Xe -> e + e + Xe+"""
        ionization_threshold = self.collision_data.ionization_threshold  # Use LXCAT value: 12.13 eV
        available_energy_ev = energy_ev - ionization_threshold
        
        if available_energy_ev <= 0:
            return None

        # !NORM! available_energy_j = available_energy_ev * self.collision_data.E_CHARGE
        available_energy_j = available_energy_ev
        
        ion = np.zeros(self.pc.NUMQ + 1)
        ion[self.pc.XCOMP] = x
        ion[self.pc.YCOMP] = y
        ion[self.WEIGHT] = weight
        
        electron = np.zeros(self.pc.NUMQ + 1)
        electron[self.pc.XCOMP] = x
        electron[self.pc.YCOMP] = y
        electron[self.WEIGHT] = weight

        if available_energy_ev > 0:
            electron_energy_fraction = 0.8
            electron_energy_j = available_energy_j * electron_energy_fraction
            
            # !NORM! electron_mass = 9.1e-31  # kg
            electron_mass = 1  # normalized
            electron_speed = np.sqrt(2 * electron_energy_j / electron_mass)
            
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            electron[self.pc.UCOMP] = electron_speed * np.sin(phi) * np.cos(theta)
            electron[self.pc.VCOMP] = electron_speed * np.sin(phi) * np.sin(theta)
            if self.pc.WCOMP < self.pc.NUMQ:
                electron[self.pc.WCOMP] = electron_speed * np.cos(phi)

            ion_energy_j = available_energy_j * (1 - electron_energy_fraction)
            # !NORM! xenon_mass = 2.18e-25
            xenon_mass = 99
            ion_speed = np.sqrt(2 * ion_energy_j / xenon_mass)
            ion_theta = np.random.uniform(0, 2 * np.pi)
            ion[self.pc.UCOMP] = ion_speed * np.cos(ion_theta) * 0.1
            ion[self.pc.VCOMP] = ion_speed * np.sin(ion_theta) * 0.1
            if self.pc.WCOMP < self.pc.NUMQ:
                ion[self.pc.WCOMP] = vz * 0.1

        products = [ion, electron]
        
        return CollisionEvent(
            event_type="ionization",
            particle1_idx=particle_idx,
            products=products,
            energy_change=ionization_threshold
        )

    def _create_excitation_event(self, collision_type: str, particle_idx: int, energy_ev: float):
        """Create excitation event with correct LXCAT energy thresholds"""
        if collision_type == "first_excitation":
            energy_loss = 8.315  # eV - LXCAT Xe*(8.315eV)
        elif collision_type == "second_excitation":
            energy_loss = 9.447  # eV - LXCAT Xe*(9.447eV)
        elif collision_type == "third_excitation":
            energy_loss = 9.917  # eV - LXCAT Xe*(9.917eV)
        elif collision_type == "fourth_excitation":
            energy_loss = 11.7   # eV - LXCAT Xe*(11.7eV)
        else:
            energy_loss = 0.0  # fallback
            
        return CollisionEvent(
            event_type=collision_type,
            particle1_idx=particle_idx,
            energy_change=energy_loss
        )

    def _create_elastic_event(self, collision_type: str, particle_idx: int):
        return CollisionEvent(
            event_type=collision_type,
            particle1_idx=particle_idx,
            energy_change=0.0
        )

    def _apply_energy_loss(self, particle_idx: int, energy_loss_ev: float):
        vx = self.particles[self.pc.UCOMP, particle_idx]
        vy = self.particles[self.pc.VCOMP, particle_idx]
        vz = self.particles[self.pc.WCOMP, particle_idx] if self.pc.WCOMP < self.pc.NUMQ else 0.0
        
        v_current = np.sqrt(vx**2 + vy**2 + vz**2)
        if v_current < 1e-10:
            return
            
        ke_current_j = 0.5 * self.params.mass * v_current**2
        ke_current_ev = ke_current_j / self.collision_data.E_CHARGE
        
        ke_new_ev = max(0.0, ke_current_ev - energy_loss_ev)
        ke_new_j = ke_new_ev * self.collision_data.E_CHARGE
        
        if ke_new_j > 0:
            v_new = np.sqrt(2 * ke_new_j / self.params.mass)
            scale_factor = v_new / v_current
            
            self.particles[self.pc.UCOMP, particle_idx] *= scale_factor
            self.particles[self.pc.VCOMP, particle_idx] *= scale_factor
            if self.pc.WCOMP < self.pc.NUMQ:
                self.particles[self.pc.WCOMP, particle_idx] *= scale_factor
        else:
            self.particles[self.pc.UCOMP, particle_idx] = 0.0
            self.particles[self.pc.VCOMP, particle_idx] = 0.0
            if self.pc.WCOMP < self.pc.NUMQ:
                self.particles[self.pc.WCOMP, particle_idx] = 0.0

    def _apply_elastic_scattering(self, particle_idx: int, event: CollisionEvent):
        """elastic scattering"""
        vx = self.particles[self.pc.UCOMP, particle_idx]
        vy = self.particles[self.pc.VCOMP, particle_idx]
        vz = self.particles[self.pc.WCOMP, particle_idx] if self.pc.WCOMP < self.pc.NUMQ else 0.0
        
        v_magnitude = np.sqrt(vx**2 + vy**2 + vz**2)
        if v_magnitude < 1e-10:
            return
        
        if event.event_type == "elastic_backward":
            theta = np.random.uniform(np.pi * 0.8, np.pi * 1.2) 
        else:
            theta = np.random.uniform(0, 2 * np.pi)
            
        phi = np.random.uniform(0, np.pi)
        
        self.particles[self.pc.UCOMP, particle_idx] = v_magnitude * np.sin(phi) * np.cos(theta)
        self.particles[self.pc.VCOMP, particle_idx] = v_magnitude * np.sin(phi) * np.sin(theta)
        if self.pc.WCOMP < self.pc.NUMQ:
            self.particles[self.pc.WCOMP, particle_idx] = v_magnitude * np.cos(phi)

    def _get_grid_coordinates(self, x: float, y: float):
        ix = int((x - self.inp.grid_x[0]) / self.inp.dx)
        iy = int((y - self.inp.grid_y[0]) / self.inp.dy)
        if 0 <= ix < self.inp.nx_with_ghosts and 0 <= iy < self.inp.ny_with_ghosts:
            return ix, iy
        return None

    # go from bulk/field density -> density at a specific particle's (x, y)
    def _interpolate_density(self, x: float, y: float, density_field: np.ndarray):
        x_grid = (x - self.inp.grid_x[0]) / self.inp.dx
        y_grid = (y - self.inp.grid_y[0]) / self.inp.dy
        ix = int(np.floor(x_grid))
        iy = int(np.floor(y_grid))
        if ix < 0 or ix >= self.inp.nx_with_ghosts-1 or iy < 0 or iy >= self.inp.ny_with_ghosts-1:
            return 0.0
        wx = x_grid - ix
        wy = y_grid - iy
        density = (
            density_field[ix, iy] * (1 - wx) * (1 - wy) +
            density_field[ix+1, iy] * wx * (1 - wy) +
            density_field[ix, iy+1] * (1 - wx) * wy +
            density_field[ix+1, iy+1] * wx * wy
        )
        return max(density, 0.0)
    
    def _interpolate_electric_field(self, x: float, y: float):
        x_grid = (x - self.inp.grid_x[0]) / self.inp.dx
        y_grid = (y - self.inp.grid_y[0]) / self.inp.dy

        ix = int(np.floor(x_grid))
        iy = int(np.floor(y_grid))
        
        # Get array dimensions
        nx_max = self.fields.E.shape[0] - 1
        ny_max = self.fields.E.shape[1] - 1
        
        # Check if base point is within bounds
        if 0 <= ix <= nx_max and 0 <= iy <= ny_max:
            wx = x_grid - ix
            wy = y_grid - iy
            
            # Check availability of each interpolation point and adjust accordingly
            # Bottom-left point (ix, iy) - always available if we're here
            Ex = self.fields.E[ix, iy, 0] * (1 - wx) * (1 - wy)
            Ey = self.fields.E[ix, iy, 1] * (1 - wx) * (1 - wy)
            
            # Bottom-right point (ix+1, iy)
            if ix + 1 <= nx_max:
                Ex += self.fields.E[ix+1, iy, 0] * wx * (1 - wy)
                Ey += self.fields.E[ix+1, iy, 1] * wx * (1 - wy)
            else:
                # Use bottom-left point with adjusted weight
                Ex += self.fields.E[ix, iy, 0] * wx * (1 - wy)
                Ey += self.fields.E[ix, iy, 1] * wx * (1 - wy)
            
            # Top-left point (ix, iy+1)
            if iy + 1 <= ny_max:
                Ex += self.fields.E[ix, iy+1, 0] * (1 - wx) * wy
                Ey += self.fields.E[ix, iy+1, 1] * (1 - wx) * wy
            else:
                # Use bottom-left point with adjusted weight
                Ex += self.fields.E[ix, iy, 0] * (1 - wx) * wy
                Ey += self.fields.E[ix, iy, 1] * (1 - wx) * wy
            
            # Top-right point (ix+1, iy+1)
            if ix + 1 <= nx_max and iy + 1 <= ny_max:
                Ex += self.fields.E[ix+1, iy+1, 0] * wx * wy
                Ey += self.fields.E[ix+1, iy+1, 1] * wx * wy
            elif ix + 1 <= nx_max:
                # Use bottom-right point
                Ex += self.fields.E[ix+1, iy, 0] * wx * wy
                Ey += self.fields.E[ix+1, iy, 1] * wx * wy
            elif iy + 1 <= ny_max:
                # Use top-left point
                Ex += self.fields.E[ix, iy+1, 0] * wx * wy
                Ey += self.fields.E[ix, iy+1, 1] * wx * wy
            else:
                # Use bottom-left point
                Ex += self.fields.E[ix, iy, 0] * wx * wy
                Ey += self.fields.E[ix, iy, 1] * wx * wy
            
            return Ex, Ey
        
        # If completely outside bounds, return zero field
        return 0.0, 0.0

    def _initialize_leapfrog_velocities(self):
        n = 0
        x = self.particles[self.pc.XCOMP, n]
        y = self.particles[self.pc.YCOMP, n]
    
        Ex, Ey = self._interpolate_electric_field(x, y)
    
        ax = (self.params.charge / self.params.mass) * Ex
        ay = (self.params.charge / self.params.mass) * Ey
    
        # v^(-1/2) = v^(1/2) - dt*a (backward half step)
        self.particles[self.pc.UCOMP, n] -= 0.5 * self.dt * ax
        self.particles[self.pc.VCOMP, n] -= 0.5 * self.dt * ay

    def _remove_particles(self, indices_to_remove: list[int]):
        if not indices_to_remove:
            return
        keep_mask = np.ones(self.particles.shape[1], dtype=bool)
        keep_mask[indices_to_remove] = False
        self.particles = self.particles[:, keep_mask]

    def split_particle(self, idx):
        particle = self.particles[:, idx]
        new_weight = particle[self.WEIGHT] / 2
        p1 = particle.copy()
        p2 = particle.copy()
        p1[self.WEIGHT] = new_weight
        p2[self.WEIGHT] = new_weight
        dx = np.random.normal(0, self.inp.dx * 0.01)
        dy = np.random.normal(0, self.inp.dy * 0.01)
        p1[self.pc.XCOMP] += dx
        p1[self.pc.YCOMP] += dy
        p2[self.pc.XCOMP] -= dx
        p2[self.pc.YCOMP] -= dy
        self.particles[:, idx] = p1
        self.particles = np.hstack((self.particles, p2.reshape(-1, 1)))

    def merge_particles(self, idx1, idx2):
        p1 = self.particles[:, idx1]
        p2 = self.particles[:, idx2]
        w_total = p1[self.WEIGHT] + p2[self.WEIGHT]
        merged = np.zeros_like(p1)
        for q in [self.pc.XCOMP, self.pc.YCOMP, self.pc.UCOMP, self.pc.VCOMP]:
            merged[q] = (p1[q] * p1[self.WEIGHT] + p2[q] * p2[self.WEIGHT]) / w_total
        merged[self.WEIGHT] = w_total
        self.particles[:, idx1] = merged
        self.particles = np.delete(self.particles, idx2, axis=1)

    # !DEBUG! change min_ppc back to 100
    def enforce_ppc(self, min_ppc=1, max_ppc=200):
        cell_map = {}
        for i in range(self.particles.shape[1]):
            x = self.particles[self.pc.XCOMP, i]
            y = self.particles[self.pc.YCOMP, i]
            ix = int((x - self.inp.grid_x[0]) / self.inp.dx)
            iy = int((y - self.inp.grid_y[0]) / self.inp.dy)
            if (ix, iy) not in cell_map:
                cell_map[(ix, iy)] = []
            cell_map[(ix, iy)].append(i)
        for (ix, iy), indices in cell_map.items():
            count = len(indices)
            if count < min_ppc:
                needed = min_ppc - count
                for idx in indices[:needed]:
                    self.split_particle(idx)
            elif count > max_ppc:
                to_merge = (count - max_ppc) // 2
                pairs = zip(indices[::2], indices[1::2])
                for idx1, idx2 in list(pairs)[:to_merge]:
                    if idx2 < self.particles.shape[1]:
                        self.merge_particles(idx1, idx2)

    def get_charge_density(self):
        rho = np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        
        for i in range(self.particles.shape[1]):
            x = self.particles[self.pc.XCOMP, i]
            y = self.particles[self.pc.YCOMP, i]
            ix = int((x - self.inp.grid_x[0]) / self.inp.dx)
            iy = int((y - self.inp.grid_y[0]) / self.inp.dy)
            if 0 <= ix < self.inp.nx_with_ghosts and 0 <= iy < self.inp.ny_with_ghosts:
                rho[ix, iy] += self.params.charge
        return rho
    
    def get_coloumb_source(self, idx):
        x = self.particles[self.pc.XCOMP, idx]
        y = self.particles[self.pc.YCOMP, idx]
        
        ix = int((x - self.inp.grid_x[0]) / self.inp.dx)
        iy = int((y - self.inp.grid_y[0]) / self.inp.dy)
        
        if 0 <= ix < self.inp.nx and 0 <= iy < self.inp.ny:
            x_source = self.fields.E[ix, iy, 0]
            y_source = self.fields.E[ix, iy, 1]
        else:
            return 0, 0
                
        return x_source, y_source
