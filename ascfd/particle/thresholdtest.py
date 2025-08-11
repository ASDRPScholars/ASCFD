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
        """
        Initialize with LXCAT data files
        
        Args:
            lxcat_files: Dictionary mapping process names to file paths
                        e.g., {'elastic': 'elastic.txt', 'exc1': 'exc1.txt'}
        """
        self.parser = LXCATParser()
        
        # Thresholds
        self.exc1_threshold = 8.315  # eV
        self.exc2_threshold = 9.447  # eV
        self.exc3_threshold = 9.917  # eV  
        self.exc4_threshold = 11.7   # eV
        self.ionization_threshold = 12.13  # eV
        
        # Load LXCAT files if provided
        if lxcat_files:
            for process_name, filename in lxcat_files.items():
                self.parser.parse_lxcat_file(filename, process_name)
    
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

def test_and_plot_cross_sections():
    """Test function that plots all cross sections vs energy"""
    import matplotlib.pyplot as plt
    
    xe = XenonCollisionData({
    'elastic': 'ascfd/particle/lxcat/elastic.txt',
    'exc1': 'ascfd/particle/lxcat/exc1.txt',
    'exc2': 'ascfd/particle/lxcat/exc2.txt',
    'exc3': 'ascfd/particle/lxcat/exc3.txt',
    'exc4': 'ascfd/particle/lxcat/exc4.txt',
    'ionization': 'ascfd/particle/lxcat/ionization.txt'
})
    
    # Energy range from 0.001 eV to 1000 eV (log scale)
    energies = np.logspace(-3, 3, 1000)  # 0.001 to 1000 eV
    
    # Calculate all cross sections
    elastic = [xe._elastic_cross_section(E) for E in energies]
    exc1 = [xe._excitation_cross_section_1(E) for E in energies]
    exc2 = [xe._excitation_cross_section_2(E) for E in energies]
    exc3 = [xe._excitation_cross_section_3(E) for E in energies]
    exc4 = [xe._excitation_cross_section_4(E) for E in energies]
    ionization = [xe._ionization_cross_section(E) for E in energies]
    
    # Create the plot
    plt.figure(figsize=(12, 8))
    
    plt.loglog(energies, elastic, 'b-', linewidth=2, label='Elastic')
    plt.loglog(energies, exc1, 'r-', linewidth=2, label='Exc1 (8.315 eV)')
    plt.loglog(energies, exc2, 'g-', linewidth=2, label='Exc2 (9.447 eV)')
    plt.loglog(energies, exc3, 'm-', linewidth=2, label='Exc3 (9.917 eV)')
    plt.loglog(energies, exc4, 'c-', linewidth=2, label='Exc4 (11.7 eV)')
    plt.loglog(energies, ionization, 'orange', linewidth=2, label='Ionization (12.13 eV)')
    
    plt.xlabel('Energy (eV)', fontsize=12)
    plt.ylabel('Cross Section (m²)', fontsize=12)
    plt.title('Xenon Electron Collision Cross Sections', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=10)
    
    # Set reasonable y-axis limits
    plt.ylim(1e-23, 1e-18)
    plt.xlim(0.01, 1000)
    
    plt.tight_layout()
    plt.show()
    
    # Also print some key values
    print("\nKey cross section values:")
    print("Energy (eV)  | Elastic (m²)    | Exc1 (m²)       | Ionization (m²)")
    print("-" * 70)
    test_energies = [0.01, 0.1, 0.62, 1.0, 5.0, 10.0, 20.0, 50.0, 100.0]
    for E in test_energies:
        elastic_val = xe._elastic_cross_section(E)
        exc1_val = xe._excitation_cross_section_1(E)
        ion_val = xe._ionization_cross_section(E)
        print(f"{E:8.2f}     | {elastic_val:.3e}   | {exc1_val:.3e}     | {ion_val:.3e}")

# Example usage and testing
if __name__ == "__main__":
    test_and_plot_cross_sections()