import numpy as np

class PlasmaReferences:
    """
    Plasma normalization reference scales for Hall thruster simulation.
    All equations are solved in normalized units, then converted back for output.
    """
    
    def __init__(self):
        self.m = 2.18e-25
        self.q = 1.602e-19
        self.L = 0.001
        self.v = 1
        self.dt = None
    
    # def __init__(self, n0=1e18, T0=1000.0, species_mass=9.1e-31, species_charge=1.6e-19):
    #     # Physical parameters
    #     self.n0 = n0                    # Reference density [m⁻³]
    #     self.T0 = T0                    # Reference temperature [K]  
    #     self.m = species_mass           # Species mass [kg]
    #     self.q = abs(species_charge)    # Species charge magnitude [C]
        
    #     # Physical constants
    #     self.k_B = 1.38e-23            # Boltzmann constant [J/K]
    #     self.eps_0 = 8.85e-12          # Permittivity [F/m]
        
    #     # Derived reference scales
    #     # Length scale: Debye length
    #     self.L_ref = np.sqrt(self.eps_0 * self.k_B * self.T0 / (self.n0 * self.q**2))
        
    #     # Time scale: Plasma frequency  
    #     self.omega_p = np.sqrt(self.n0 * self.q**2 / (self.eps_0 * self.m))
    #     self.t_ref = 1.0 / self.omega_p
        
    #     # Velocity scale: Thermal velocity
    #     self.v_ref = np.sqrt(self.k_B * self.T0 / self.m)
        
    #     # Mass density scale
    #     self.rho_ref = self.n0 * self.m
        
    #     # Pressure scale  
    #     self.p_ref = self.n0 * self.k_B * self.T0
        
    #     # Electric field scale
    #     self.E_ref = self.m * self.v_ref * self.omega_p / self.q
        
    #     # Potential scale
    #     self.phi_ref = self.E_ref * self.L_ref
        
    #     # Charge density scale
    #     self.charge_ref = self.q * self.n0
        
    #     print(f"Plasma References Initialized:")
    #     print(f"  Debye length: {self.L_ref*1e6:.2f} μm")
    #     print(f"  Plasma period: {self.t_ref*1e9:.2f} ns") 
    #     print(f"  Thermal velocity: {self.v_ref/1000:.1f} km/s")
    #     print(f"  Reference E-field: {self.E_ref:.2e} V/m")
        
    # def normalize_length(self, L_physical):
    #     """Convert physical length [m] to normalized length [L_ref]"""
    #     return L_physical / self.L_ref
        
    # def denormalize_length(self, L_normalized):
    #     """Convert normalized length [L_ref] to physical length [m]"""
    #     return L_normalized * self.L_ref
        
    # def normalize_time(self, t_physical):
    #     """Convert physical time [s] to normalized time [t_ref]"""
    #     return t_physical / self.t_ref
        
    # def denormalize_time(self, t_normalized):
    #     """Convert normalized time [t_ref] to physical time [s]"""
    #     return t_normalized * self.t_ref
        
    # def normalize_velocity(self, v_physical):
    #     """Convert physical velocity [m/s] to normalized velocity [v_ref]"""
    #     return v_physical / self.v_ref
        
    # def denormalize_velocity(self, v_normalized):
    #     """Convert normalized velocity [v_ref] to physical velocity [m/s]"""
    #     return v_normalized * self.v_ref
        
    # def normalize_density(self, rho_physical):
    #     """Convert physical density [kg/m³] to normalized density [rho_ref]"""
    #     return rho_physical / self.rho_ref
        
    # def denormalize_density(self, rho_normalized):
    #     """Convert normalized density [rho_ref] to physical density [kg/m³]"""
    #     return rho_normalized * self.rho_ref
        
    # def normalize_pressure(self, p_physical):
    #     """Convert physical pressure [Pa] to normalized pressure [p_ref]"""
    #     return p_physical / self.p_ref
        
    # def denormalize_pressure(self, p_normalized):
    #     """Convert normalized pressure [p_ref] to physical pressure [Pa]"""
    #     return p_normalized * self.p_ref
        
    # def normalize_efield(self, E_physical):
    #     """Convert physical E-field [V/m] to normalized E-field [E_ref]"""
    #     return E_physical / self.E_ref
        
    # def denormalize_efield(self, E_normalized):
    #     """Convert normalized E-field [E_ref] to physical E-field [V/m]"""
    #     return E_normalized * self.E_ref
        
    # def normalize_potential(self, phi_physical):
    #     """Convert physical potential [V] to normalized potential [phi_ref]"""
    #     return phi_physical / self.phi_ref
        
    # def denormalize_potential(self, phi_normalized):
    #     """Convert normalized potential [phi_ref] to physical potential [V]"""
    #     return phi_normalized * self.phi_ref
        
    # def normalize_charge_density(self, charge_physical):
    #     """Convert physical charge density [C/m³] to normalized charge density [charge_ref]"""
    #     return charge_physical / self.charge_ref
        
    # def denormalize_charge_density(self, charge_normalized):
    #     """Convert normalized charge density [charge_ref] to physical charge density [C/m³]"""
    #     return charge_normalized * self.charge_ref