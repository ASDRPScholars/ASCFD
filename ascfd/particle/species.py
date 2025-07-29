from ascfd.particle.constants import ParticleConstants
from ascfd.particle.ics import ParticleInitialConditions
from ascfd.fluid.species import FluidSpecies
from ascfd.inputs import Inputs
from ascfd.params import SpeciesParams
from ascfd.fields.fields import Fields

import numpy as np
from scipy.spatial import cKDTree

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
        # xenon collision thresholds and cross-sections based on the table
        self.E_CHARGE = 1.602176e-19  
        
        # Electron-Xenon collision data
        self.electron_collisions = {
            "elastic": {
                "threshold": 0.0,  
                "cross_section_func": self._elastic_cross_section
            },
            "first_excitation": {
                "threshold": 19.82,  # eV
                "cross_section_func": self._excitation_cross_section_1
            },
            "second_excitation": {
                "threshold": 20.61,  # eV  
                "cross_section_func": self._excitation_cross_section_2
            },
            "ionization": {
                "threshold": 24.59,  # eV
                "cross_section_func": self._ionization_cross_section
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
    
    def _elastic_cross_section(self, energy_ev):
        # 
        if energy_ev < 0.1:
            return 5e-16  # cm^2 at very low energy
        return 3e-16 * (1 + 1/np.sqrt(energy_ev))  # rough approx
    
    def _excitation_cross_section_1(self, energy_ev):
        if energy_ev < 19.82:
            return 0.0
        peak_energy = 25.0
        if energy_ev < peak_energy:
            return 2e-16 * (energy_ev - 19.82) / (peak_energy - 19.82)
        else:
            return 2e-16 * np.exp(-(energy_ev - peak_energy) / 10.0)
    
    def _excitation_cross_section_2(self, energy_ev):
        if energy_ev < 20.61:
            return 0.0
        peak_energy = 26.0
        if energy_ev < peak_energy:
            return 1e-16 * (energy_ev - 20.61) / (peak_energy - 20.61)
        else:
            return 1e-16 * np.exp(-(energy_ev - peak_energy) / 10.0)
    
    def _ionization_cross_section(self, energy_ev):
        if energy_ev < 24.59:
            return 0.0
        return 1.5e-16 * np.log(energy_ev / 24.59) if energy_ev > 24.59 else 0.0
    
    def _ion_elastic_cross_section(self, energy_ev):
        return 1e-15  
    
    def _ion_backward_cross_section(self, energy_ev):
        return 5e-16

class ParticleSpecies:
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields):
        self.pc = ParticleConstants()
        self.fields = fields

        self.inp = a_inputs
        self.params = params
        self.dt = None
        self.simulation = None 

        self.particles = np.zeros((self.pc.NUMQ + 1, self.inp.n_particles))
        self.WEIGHT = self.pc.NUMQ

        self.particles[self.WEIGHT, :] = self.estimate_initial_weight()
        self.ics = ParticleInitialConditions(self.particles, self.inp, self.params)
        
        self.ics.apply_ics()

        # initialize collision system for Xenon
        if params.type in ["e", "i"]:  # electrons and ions collide with neutrals
            self.collision_data = XenonCollisionData()
        else:
            self.collision_data = None
        
        self.collision_events = []

    def set_simulation(self, simulation):
        """Allow access to other species through simulation reference"""
        self.simulation = simulation

    def get_species_density_field(self, species_type: str):
        """Get density field of another species"""
        if self.simulation is None:
            return np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        
        if species_type == "e" and hasattr(self.simulation, 'electrons'):
            return self._compute_particle_density_field(self.simulation.electrons)
        elif species_type == "i" and hasattr(self.simulation, 'ions'):
            return self._compute_particle_density_field(self.simulation.ions)
        elif species_type == "n" and hasattr(self.simulation, 'neutrals'):
            return self._compute_particle_density_field(self.simulation.neutrals)
        
        return np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))

    def _compute_particle_density_field(self, species):
        """Compute number density field from particle positions"""
        if not hasattr(species, 'particles') or species.particles.shape[1] == 0:
            return np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        
        density_field = np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        
        for i in range(species.particles.shape[1]):
            x = species.particles[self.pc.XCOMP, i]
            y = species.particles[self.pc.YCOMP, i]
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
        else:
            print(f"Warning: Invalid particle data format for species {self.params.type}")

    def estimate_initial_weight(self):
        Vc = self.inp.dx * self.inp.dy
        target_ppc = 100  # N_PPC
        return (self.params.density * Vc) / target_ppc

    def update(self):
        print("!PARTICLE! update particle!")
        new_particles = []

        if self.params.type in ["e", "i"] and self.collision_data is not None:
            new_particles = self.process_collisions()

        # particle per cell enforcement
        self.enforce_ppc()

        # update electric field with current charge distribution
        if hasattr(self, 'get_charge_density'):
            charge_density = self.get_charge_density()
            self.fields.add_charge_density(charge_density)
            
            print("IONS ADDED CHARGE DENSITY:", charge_density)
            
            # self.fields.update_E

        return new_particles

    def process_collisions(self):
        if self.params.type not in ["e", "i"] or self.collision_data is None:
            return []

        new_particles = []
        collision_events = []
        
        neutral_density_field = self.get_species_density_field("n")
        
        particles_to_remove = []

        for n in range(self.particles.shape[1]):
            events = self._attempt_collisions(n, neutral_density_field)
            
            for event in events:
                collision_events.append(event)
                
                if event.event_type == "ionization":
                    new_particles.extend(event.products)
                elif event.event_type in ["first_excitation", "second_excitation"]:
                    self._apply_energy_loss(n, event.energy_change)
                elif event.event_type.startswith("elastic"):
                    self._apply_elastic_scattering(n, event)

        if particles_to_remove:
            self._remove_particles(particles_to_remove)

        self.collision_events.extend(collision_events)
        return new_particles

    def _attempt_collisions(self, particle_idx: int, neutral_density_field: np.ndarray):
        x = self.particles[self.pc.XCOMP, particle_idx]
        y = self.particles[self.pc.YCOMP, particle_idx]
        vx = self.particles[self.pc.UCOMP, particle_idx]
        vy = self.particles[self.pc.VCOMP, particle_idx]
        vz = self.particles[self.pc.WCOMP, particle_idx] if self.pc.WCOMP < self.pc.NUMQ else 0.0
        weight = self.particles[self.WEIGHT, particle_idx]

        grid_coords = self._get_grid_coordinates(x, y)
        if grid_coords is None:
            return []

        v_rel = np.sqrt(vx**2 + vy**2 + vz**2)
        if v_rel < 1e-10:
            return []

        energy_ev = 0.5 * self.params.mass * v_rel**2 / self.collision_data.E_CHARGE

        neutral_density = self._interpolate_density(x, y, neutral_density_field)
        if neutral_density <= 0:
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
                continue
                
            sigma = collision_data["cross_section_func"](energy_ev)
            if sigma <= 0:
                continue

            # nu = n * sigma * v
            nu_collision = neutral_density * sigma * v_rel
            P_collision = 1.0 - np.exp(-nu_collision * self.dt)

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

        return events

    def _create_collision_event(self, collision_type: str, particle_idx: int, 
                               x: float, y: float, vx: float, vy: float, vz: float,
                               weight: float, energy_ev: float):
        
        if collision_type == "ionization" and self.params.type == "e":
            return self._create_ionization_event(particle_idx, x, y, vx, vy, vz, weight, energy_ev)
        elif collision_type in ["first_excitation", "second_excitation"] and self.params.type == "e":
            return self._create_excitation_event(collision_type, particle_idx, energy_ev)
        elif collision_type.startswith("elastic"):
            return self._create_elastic_event(collision_type, particle_idx)
        
        return None

    def _create_ionization_event(self, particle_idx: int, x: float, y: float, 
                                vx: float, vy: float, vz: float, weight: float, energy_ev: float):
        """ e + Xe -> e + e + Xe+"""
        ionization_threshold = 24.59  # eV
        available_energy_ev = energy_ev - ionization_threshold
        
        if available_energy_ev <= 0:
            return None

        available_energy_j = available_energy_ev * self.collision_data.E_CHARGE
        
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
            electron_mass = 9.1e-31  # kg
            electron_speed = np.sqrt(2 * electron_energy_j / electron_mass)
            
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            electron[self.pc.UCOMP] = electron_speed * np.sin(phi) * np.cos(theta)
            electron[self.pc.VCOMP] = electron_speed * np.sin(phi) * np.sin(theta)
            if self.pc.WCOMP < self.pc.NUMQ:
                electron[self.pc.WCOMP] = electron_speed * np.cos(phi)

            ion_energy_j = available_energy_j * (1 - electron_energy_fraction)
            xenon_mass = 2.18e-25
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
        if collision_type == "first_excitation":
            energy_loss = 19.82  # eV
        else:  # second_excitation
            energy_loss = 20.61  # eV
            
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

    # !DEBUG! change back to 100
    def enforce_ppc(self, min_ppc=10, max_ppc=200):
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
        rho = np.zeros_like(self.fields.E[0])
        for i in range(self.particles.shape[1]):
            x = self.particles[self.pc.XCOMP, i]
            y = self.particles[self.pc.YCOMP, i]
            ix = int((x - self.inp.grid_x[0]) / self.inp.dx)
            iy = int((y - self.inp.grid_y[0]) / self.inp.dy)
            if 0 <= ix < self.inp.nx_with_ghosts and 0 <= iy < self.inp.ny_with_ghosts:
                rho[ix, iy] += self.params.charge
        return rho
