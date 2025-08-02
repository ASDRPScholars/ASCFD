import numpy as np
from scipy.interpolate import interp1d

class LXCATParser:
    def __init__(self):
        self.cross_sections = {}
    
    def parse_lxcat_file(self, filename, process_name):
        """
        Parse LXCAT cross section file and create interpolation function
        
        Args:
            filename: Path to LXCAT text file
            process_name: Name to store this cross section under (e.g., 'elastic', 'exc1')
        """
        energies = []
        cross_sections = []
        
        with open(filename, 'r') as f:
            lines = f.readlines()
        
        # Find the data table (between dashes)
        in_table = False
        for line in lines:
            line = line.strip()
            
            # Start of table
            if line.startswith('-----') and len(line) >= 5:
                in_table = True
                continue
            
            # End of table
            if in_table and line.startswith('-----') and len(line) >= 5:
                break
            
            # Parse data lines
            if in_table and line:
                try:
                    parts = line.split()
                    if len(parts) >= 2:
                        energy = float(parts[0])
                        cross_section = float(parts[1])
                        energies.append(energy)
                        cross_sections.append(cross_section)
                except ValueError:
                    continue
        
        # Convert to numpy arrays
        energies = np.array(energies)
        cross_sections = np.array(cross_sections)
        
        # Create interpolation function
        # Use linear interpolation in log-log space for better behavior
        # Handle zero cross sections by adding small offset
        min_cs = 1e-30  # Small value to avoid log(0)
        safe_cs = np.maximum(cross_sections, min_cs)
        
        # Create interpolator
        interp_func = interp1d(
            energies, 
            safe_cs, 
            kind='linear',
            bounds_error=False,
            fill_value=(safe_cs[0], safe_cs[-1])  # Extrapolate with edge values
        )
        
        # Store the interpolation function
        self.cross_sections[process_name] = interp_func
        
        print(f"Loaded {process_name}: {len(energies)} data points from {energies[0]:.3e} to {energies[-1]:.3e} eV")
    
    def get_cross_section(self, process_name, energy_ev):
        """
        Get interpolated cross section for given process and energy
        
        Args:
            process_name: Name of the process (e.g., 'elastic', 'exc1')
            energy_ev: Energy in eV (can be scalar or array)
        
        Returns:
            Cross section in m² (same shape as energy_ev)
        """
        if process_name not in self.cross_sections:
            raise ValueError(f"Process '{process_name}' not found. Available: {list(self.cross_sections.keys())}")
        
        return self.cross_sections[process_name](energy_ev)


class XenonCollisionData:
    def __init__(self, lxcat_files=None):
        """Initialize with LXCAT data files"""
        self.parser = LXCATParser()
        
        # Thresholds (in eV)
        self.exc1_threshold = 8.315
        self.exc2_threshold = 9.447
        self.exc3_threshold = 9.917
        self.exc4_threshold = 11.7
        self.ionization_threshold = 12.13
        
        # Load LXCAT files if provided
        if lxcat_files:
            for process_name, filename in lxcat_files.items():
                self.parser.parse_lxcat_file(filename, process_name)
        
        # Initialize collision dictionaries
        self.electron_collisions = self._init_electron_collisions()
        self.ion_collisions = self._init_ion_collisions()
    
    def load_lxcat_file(self, filename, process_name):
        """Load a single LXCAT file"""
        self.parser.parse_lxcat_file(filename, process_name)
        # Reinitialize collision dictionaries to use new data
        self.electron_collisions = self._init_electron_collisions()
        self.ion_collisions = self._init_ion_collisions()
    
    def _init_electron_collisions(self):
        """Initialize electron collision dictionary"""
        return {
            "elastic": {
                "threshold": 0.0,
                "cross_section_func": self._elastic_cross_section,
                "energy_loss": 0.0
            },
            "excitation_1": {
                "threshold": self.exc1_threshold,
                "cross_section_func": self._excitation_cross_section_1,
                "energy_loss": self.exc1_threshold
            },
            "excitation_2": {
                "threshold": self.exc2_threshold,
                "cross_section_func": self._excitation_cross_section_2,
                "energy_loss": self.exc2_threshold
            },
            "excitation_3": {
                "threshold": self.exc3_threshold,
                "cross_section_func": self._excitation_cross_section_3,
                "energy_loss": self.exc3_threshold
            },
            "excitation_4": {
                "threshold": self.exc4_threshold,
                "cross_section_func": self._excitation_cross_section_4,
                "energy_loss": self.exc4_threshold
            },
            "ionization": {
                "threshold": self.ionization_threshold,
                "cross_section_func": self._ionization_cross_section,
                "energy_loss": self.ionization_threshold
            }
        }
    
    def _init_ion_collisions(self):
        """Initialize ion collision dictionary"""
        return {
            "elastic": {
                "threshold": 0.0,
                "cross_section_func": self._ion_elastic_cross_section,
                "energy_loss": 0.0
            },
            "backward": {
                "threshold": 0.0,
                "cross_section_func": self._ion_backward_cross_section,
                "energy_loss": 0.0
            }
        }
    
    def load_lxcat_file(self, filename, process_name):
        """Load a single LXCAT file"""
        self.parser.parse_lxcat_file(filename, process_name)
    
    def _elastic_cross_section(self, energy_ev):
        """Elastic cross section - uses LXCAT data if available, otherwise analytical"""
        if 'elastic' in self.parser.cross_sections:
            return float(self.parser.get_cross_section('elastic', energy_ev))
    
    def _excitation_cross_section_1(self, energy_ev):
        """First excitation cross section - uses LXCAT data if available"""
        if energy_ev < self.exc1_threshold:
            return 0.0
        
        if 'exc1' in self.parser.cross_sections:
            return float(self.parser.get_cross_section('exc1', energy_ev))
    
    def _excitation_cross_section_2(self, energy_ev):
        """Second excitation cross section"""
        if energy_ev < self.exc2_threshold:
            return 0.0
        
        if 'exc2' in self.parser.cross_sections:
            return float(self.parser.get_cross_section('exc2', energy_ev))
    
    def _excitation_cross_section_3(self, energy_ev):
        """Third excitation cross section"""
        if energy_ev < self.exc3_threshold:
            return 0.0
        
        if 'exc3' in self.parser.cross_sections:
            return float(self.parser.get_cross_section('exc3', energy_ev))
    
    def _excitation_cross_section_4(self, energy_ev):
        """Fourth excitation cross section"""
        if energy_ev < self.exc4_threshold:
            return 0.0
        
        if 'exc4' in self.parser.cross_sections:
            return float(self.parser.get_cross_section('exc4', energy_ev))
    
    def _ionization_cross_section(self, energy_ev):
        """Ionization cross section"""
        if energy_ev < self.ionization_threshold:
            return 0.0
        
        if 'ionization' in self.parser.cross_sections:
            return float(self.parser.get_cross_section('ionization', energy_ev))
        
    def _ion_elastic_cross_section(self, energy_ev):
        """Ion elastic cross section"""
        if 'ion_elastic' in self.parser.cross_sections:
            return float(self.parser.get_cross_section('ion_elastic', energy_ev))
    
    def _ion_backward_cross_section(self, energy_ev):
        """Ion backward scattering cross section"""
        if 'ion_backward' in self.parser.cross_sections:
            return float(self.parser.get_cross_section('ion_backward', energy_ev))
        