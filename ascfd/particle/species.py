from ascfd.particle.constants import ParticleConstants
from ascfd.particle.ics import ParticleInitialConditions
from ascfd.fluid.species import FluidSpecies
from ascfd.inputs import Inputs
from ascfd.params import SpeciesParams
from ascfd.fields.fields import Fields
from ascfd.particle.bcs import ParticleBoundaryConditions
from ascfd.cross_sections.xe import XenonCollisionData
# from ascfd.simulation import Simulation

import numpy as np
from scipy.spatial import cKDTree
from scipy.interpolate import interp1d
from scipy import integrate, special
from collections import deque

class CollisionEvent:
    def __init__(self, event_type: str, particle1_idx: int, particle2_idx: int = None, 
                 products: list = None, energy_change: float = 0.0):
        self.event_type = event_type  # "elastic", "excitation", "ionization"
        self.particle1_idx = particle1_idx
        self.particle2_idx = particle2_idx
        self.products = products or []  # new particles created
        self.energy_change = energy_change

class ParticleSpecies:
    def __init__(self, pc: ParticleConstants, params: SpeciesParams, a_inputs: Inputs, fields: Fields, collision_data: XenonCollisionData, simulation):
        self.pc = pc
        self.fields = fields

        self.inp = a_inputs
        self.params = params
        self.dt = None
        self.simulation = simulation 

        self.WEIGHT = self.pc.NUMQ
        
        self.num_collisions = np.zeros((self.inp.nx, self.inp.ny))
        self.cross_section_grid = np.zeros((self.inp.nx, self.inp.ny))
        self.sigma_temp_storage = [[[] for _ in range(self.inp.ny)] for _ in range(self.inp.nx)]
        self.sigma = np.zeros((self.inp.nx, self.inp.ny))

        self.collision_data = collision_data
        self.collision_count = np.zeros(3)
        
        # Track ionization positions for spatial analysis
        self.ionization_positions_x = []
        
        # Cache for active particles to reduce recomputation
        self._active_indices_cache = None
        self._cache_valid = False
        
        # Pre-allocate particle arrays with 3x initial capacity for growth
        initial_capacity = max(self.inp.n_particles * 3, 1000) # TODO: TEMP IC - if self.params.type != "i" else 1000
        self.capacity = initial_capacity
        
        # Determine number of components based on species type
        # Electrons may have 3D velocity (NUMQ + 1), ions may not
        if self.params.type == "e":
            num_components = self.pc.NUMQ + 1  # Include 3D velocity
        else:
            num_components = self.pc.NUMQ + 1  # Standard components
        
        self.particles = np.zeros((num_components, self.capacity))
        print(f"Initialized {self.params.type} with {num_components} components, capacity {self.capacity}")
        
        # Active particle management
        self.active_count = 0
        self.free_slots = deque()
        self.is_active = np.zeros(self.capacity, dtype=bool)
        
        # Spatial sorting counter for cache optimization
        self.sort_counter = 0
        self.sort_frequency = 50  # Sort every 50 timesteps

        # TODO: TEMP IC - if self.params.type != "i":
        # Initialize active particles
        self.active_count = min(self.inp.n_particles, self.capacity)
        self.is_active[:self.active_count] = True
        print(f"Initialized {self.active_count} active {self.params.type} particles")
        
        # TODO: TEMP IC -
        # if self.params.type in ["n", "i"]:
        #     # Create temporary array for ICs - ensure correct shape
        #     temp_particles = self.particles[:, :self.active_count].copy()
        #     self.ics = ParticleInitialConditions(temp_particles, self.inp, self.params)        
        #     # self.ics.apply_ics()
            
        #     # Copy back the initialized data
        #     self.particles[:, :self.active_count] = temp_particles
        #     weight = self.estimate_initial_weight()
        #     if not (np.isnan(weight) or np.isinf(weight)):
        #         self.particles[self.WEIGHT, :self.active_count] = weight
        #     else:
        #         print(f"ERROR: Invalid initial weight for {self.params.type}: {weight}")
        #         self.particles[self.WEIGHT, :self.active_count] = 1.0  # Default weight
            
        # TODO: VERY MUCH SCUFFED TEMP IC - ALSO FIGURE OUT THIS LOGIC
        # elif self.params.type == "i":
            
        #     temp_particles = self.particles[:, :self.active_count].copy()
        #     self.ics = ParticleInitialConditions(temp_particles, self.inp, self.params)        
        #     self.ics.apply_ics()
            
        #     # Copy back the initialized data
        #     self.particles[:, :self.active_count] = temp_particles
        #     weight = self.estimate_initial_weight()
            
        #     if not (np.isnan(weight) or np.isinf(weight)):
        #         self.particles[self.WEIGHT, :self.active_count] = weight
                
        #     pass
            
        #     self.active_count = 1
        #     self.is_active[0] = True
            
        #     # Initialize ion with valid position and velocity
        #     init_x = 0.5 * self.inp.nx
        #     init_y = 0.5 * self.inp.ny  # Fix: was using nx for both x and y
            
        #     # Validate initial values
        #     if np.isnan(init_x) or np.isnan(init_y) or np.isinf(init_x) or np.isinf(init_y):
        #         print(f"ERROR: Invalid initial position for ion: x={init_x}, y={init_y}")
        #         init_x, init_y = 1.0, 1.0  # Safe fallback
            
        #     self.particles[self.pc.XCOMP, 0] = init_x
        #     self.particles[self.pc.YCOMP, 0] = init_y
        #     self.particles[self.pc.UCOMP, 0] = 100
        #     self.particles[self.pc.VCOMP, 0] = 0
        #     self.particles[self.WEIGHT, 0] = 1.0  # Non-zero weight
            
        if self.params.type in ["i", "n"]:
            self.bcs = ParticleBoundaryConditions(self, self.inp, self.params)
        else:
            print("P ELECTRONS INITED WITH", self.active_count, "particles")
            print("P ELECTRONS CAPACITY", self.capacity)
            self.particles = self.simulation.electrons.convert_to_particles()
            self.active_count = self.particles.shape[1]
            self.is_active[:self.active_count] = True

        # initialize velocity offset for leap-frog scheme
        # for first timestep, we need v^(-1/2), so we calculate it as v^(1/2) - dt*a
        # if self.params.type == "i":  # only for charged particles
        #     self._initialize_leapfrog_velocities()
        
        self.collision_events = []

    def update(self):
        
        print(f"!START! BEGINNING UPDATE: first active particles = {self.particles[self.pc.XCOMP, self._get_active_indices()[:3]] if self._get_active_indices() else 'NONE'}")
        
        self.num_collisions.fill(0)
        self.ionization_positions_x.clear()
        # Invalidate active particle cache at start of update
        self._cache_valid = False
        
        particles_to_remove = []
        
        if self.params.type == "n":
            self.bcs.apply_bcs()
            
        if self.params.type in ["i", "n"]:
            # Vectorized particle update for better performance
            active_indices = self._get_active_indices()
            
            if len(active_indices) > 0:
                # Extract positions for all active particles at once
                x_positions = self.particles[self.pc.XCOMP, active_indices]
                y_positions = self.particles[self.pc.YCOMP, active_indices]
                
                # Batch interpolate electric fields
                Ex_values, Ey_values = self._interpolate_electric_field_batch(x_positions, y_positions)
                
                # Vectorized acceleration calculation using pre-computed ratio
                # TODO: !GOODENOUGH! 
                ax_values = self.params.charge * Ex_values * self.dt # * 50000
                ay_values = self.params.charge * Ey_values * self.dt # * 50000
                
                # Check for invalid fields/accelerations
                invalid_mask = (np.isnan(Ex_values) | np.isnan(Ey_values) | 
                               np.isinf(Ex_values) | np.isinf(Ey_values) |
                               np.isnan(ax_values) | np.isnan(ay_values) |
                               np.isinf(ax_values) | np.isinf(ay_values))
                
                if np.any(invalid_mask):
                    invalid_indices = np.array(active_indices)[invalid_mask]
                    particles_to_remove.extend(invalid_indices.tolist())
                    print(f"WARNING: Removing {np.sum(invalid_mask)} particles with invalid fields")
                
                # Update velocities for valid particles
                valid_mask = ~invalid_mask
                valid_indices = np.array(active_indices)[valid_mask]
                
                if len(valid_indices) > 0:
                    # Vectorized velocity update
                    self.particles[self.pc.UCOMP, valid_indices] += ax_values[valid_mask]
                    self.particles[self.pc.VCOMP, valid_indices] += ay_values[valid_mask]
                    
                    # Check for near-zero velocities
                    low_vel_mask = ((self.particles[self.pc.UCOMP, valid_indices] <= 1e-1) &
                                   (self.particles[self.pc.VCOMP, valid_indices] <= 1e-1))
                    if np.any(low_vel_mask):
                        low_vel_indices = valid_indices[low_vel_mask]
                        particles_to_remove.extend(low_vel_indices.tolist())
                    
                    # Vectorized position update for remaining particles
                    good_vel_mask = ~low_vel_mask
                    good_indices = valid_indices[good_vel_mask]
                    
                    if True: # len(good_indices) > 0
                        
                        #TODO: add back good_indices mask
                        #TODO: FIGURE OUT WHERE THE MISSING LINK IS IN NORMALIZATION
                        self.particles[self.pc.XCOMP] += (self.dt * self.particles[self.pc.UCOMP])
                        
                        # TODO: ADD MULTIPLIER HERE AND SUDDENLY NEUTRAL DENSITY WORKS? (grid bound issue for sure... it's not seeing small grid bounds?) - also if you turn it off then u can see REAL particle axial advection (but sparse)
                        self.particles[self.pc.YCOMP] += (self.dt * self.particles[self.pc.VCOMP])
                            
                        # Check for particles that moved outside domain
                        print(f"!B! BEFORE BOUNDARY CHECK: x_positions = {self.particles[self.pc.XCOMP, good_indices[:5]]}")
                        for idx in good_indices:
                            x_pos = self.particles[self.pc.XCOMP, idx]
                            y_pos = self.particles[self.pc.YCOMP, idx]
                            grid_coords = self._get_grid_coordinates(x_pos, y_pos)
                            if grid_coords is None:
                                print(f"!REMOVE! Particle {idx} at ({x_pos:.6f}, {y_pos:.6f}) flagged for removal - outside domain")
                                particles_to_remove.append(idx)
                            else:
                                print(f"!KEEP! Particle {idx} at ({x_pos:.6f}, {y_pos:.6f}) -> grid {grid_coords} - keeping")
                            break  # Only debug first particle to avoid spam
                        print(f"!B! AFTER BOUNDARY CHECK: x_positions = {self.particles[self.pc.XCOMP, good_indices[:5]]}")
            

        if particles_to_remove:
            active_before = np.sum(self.is_active)
            # print("!%! BEFORE REMOVE THERE ARE:", active_before)
            active_indices_before_remove = self._get_active_indices()
            # print("!%! BEFORE REMOVE X POSITIONS:", self.particles[self.pc.XCOMP, active_indices_before_remove[:5]] if active_indices_before_remove else "NO ACTIVE PARTICLES")
            self._remove_particles(particles_to_remove)
            # print("!%! REMOVING THESE PARTICLES:")
            # for idx in particles_to_remove:
            #     print(f"({self.particles[self.pc.XCOMP, idx]}, {self.particles[self.pc.YCOMP, idx]})")
            active_after = np.sum(self.is_active)
            # print(f"!%! AFTER REMOVE {self.params.type} THERE ARE:", active_after)
            active_indices_after_remove = self._get_active_indices()
            # print("!%! AFTER REMOVE X POSITIONS:", self.particles[self.pc.XCOMP, active_indices_after_remove[:5]] if active_indices_after_remove else "NO ACTIVE PARTICLES")
    
        new_particles = []
        
        if self.params.type in ["e", "i"] and self.collision_data is not None and self.inp.collision_type == "dsmc":
            # TODO: READD COLLISIONS
            new_particles = self.process_collisions()
            
            if new_particles:
                self.add_particles(new_particles)
                
        elif self.params.type in ["i"] and self.inp.collision_type == "rate_coeffs":
            n_e = self.simulation.get_species_number_density("e")
            n_n = self.simulation.get_species_number_density("n")
            k_iz = self.simulation.get_coeffs("k_iz")
            
            # TODO toggle: ion_rate = n_e * n_n * k_iz # LANDMARK short paper
            print("a")
        
        # TODO turn on?
        if self.params.type != "e":
            pass
            # print(f"!PPC_BEFORE! BEFORE PPC: first active particles = {self.particles[self.pc.XCOMP, self._get_active_indices()[:3]] if self._get_active_indices() else 'NONE'}")
            # self.enforce_ppc()
            # print(f"!PPC_AFTER! AFTER PPC: first active particles = {self.particles[self.pc.XCOMP, self._get_active_indices()[:3]] if self._get_active_indices() else 'NONE'}")

        if hasattr(self, 'get_charge_density'):
            charge_density = self.get_charge_density()
            self.fields.add_charge_density(charge_density)
        
            # print("IONS ADDED CHARGE DENSITY:", charge_density)
        
        if self.inp.e_system != "quasineutral":
            self.fields.update_E()
        
        # Perform periodic spatial sorting for cache locality (do this at end of update)
        self.sort_counter += 1
        if self.sort_counter >= self.sort_frequency:
            print(f"!SORT_BEFORE! BEFORE SPATIAL SORT: first active particles = {self.particles[self.pc.XCOMP, self._get_active_indices()[:3]] if self._get_active_indices() else 'NONE'}")
            self._sort_particles_spatially()
            print(f"!SORT_AFTER! AFTER SPATIAL SORT: first active particles = {self.particles[self.pc.XCOMP, self._get_active_indices()[:3]] if self._get_active_indices() else 'NONE'}")
            self.sort_counter = 0

        print("--##-- COLLISION EVENTS FOR", self.params.type)
        print("ionization", self.collision_count[0])
        print("excitations", self.collision_count[1])
        print("elastic", self.collision_count[2])

        self.collision_count.fill(0)

        print(f"!END! END UPDATE: first active particles = {self.particles[self.pc.XCOMP, self._get_active_indices()[:3]] if self._get_active_indices() else 'NONE'}")

        return new_particles

    def compute_particle_density_field(self):
        """Compute number density field from particle positions"""
        if not hasattr(self, 'particles') or not hasattr(self, 'is_active'):
            return np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        
        density_field = np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        
        active_indices = self._get_active_indices()
        
        if len(active_indices) > 0:
            # Vectorized density computation
            x_positions = self.particles[self.pc.XCOMP, active_indices]
            y_positions = self.particles[self.pc.YCOMP, active_indices]
            weights = self.particles[self.WEIGHT, active_indices]
            
            # Vectorized grid index calculation
            ix_values = ((x_positions - self.inp.grid_x[0]) / self.inp.dx).astype(int)
            iy_values = ((y_positions - self.inp.grid_y[0]) / self.inp.dy).astype(int)
            
            # Bounds check
            valid_mask = ((ix_values >= 0) & (ix_values < self.inp.nx_with_ghosts) & 
                         (iy_values >= 0) & (iy_values < self.inp.ny_with_ghosts))
            
            if np.any(valid_mask):
                ix_valid = ix_values[valid_mask]
                iy_valid = iy_values[valid_mask]
                weights_valid = weights[valid_mask]
                
                # Use np.add.at for efficient accumulation
                np.add.at(density_field, (ix_valid, iy_valid), weights_valid / (self.inp.dx * self.inp.dy))
        
        return density_field
    
    
    def get_x_velocity_field(self):
        """Compute x-velocity field from particle velocities, defaults to 0 where no particles"""
        velocity_field = np.zeros((self.inp.nx, self.inp.ny))

        active_indices = self._get_active_indices()
        if len(active_indices) == 0:
            return velocity_field

        # Extract particle data
        x_positions = self.particles[self.pc.XCOMP, active_indices]
        y_positions = self.particles[self.pc.YCOMP, active_indices]
        x_vel = self.particles[self.pc.UCOMP, active_indices]

        # Grid indices
        ix_values = ((x_positions - self.inp.grid_x[0]) / self.inp.dx).astype(int)
        iy_values = ((y_positions - self.inp.grid_y[0]) / self.inp.dy).astype(int)

        # Bounds check
        valid_mask = ((ix_values >= 0) & (ix_values < self.inp.nx) &
                     (iy_values >= 0) & (iy_values < self.inp.ny))

        if not np.any(valid_mask):
            return velocity_field

        ix_valid = ix_values[valid_mask]
        iy_valid = iy_values[valid_mask]
        x_vel_valid = x_vel[valid_mask]

        # Accumulate velocities and counts
        count_field = np.zeros_like(velocity_field)
        np.add.at(velocity_field, (ix_valid, iy_valid), x_vel_valid)
        np.add.at(count_field, (ix_valid, iy_valid), 1)

        # Average where particles exist (elsewhere remains 0)
        nonzero_mask = count_field > 0
        velocity_field[nonzero_mask] /= count_field[nonzero_mask]

        return velocity_field


    def _get_free_slot(self):
        """Get a free slot index for new particle"""
        # First, try to use a recycled slot
        if self.free_slots:
            slot = self.free_slots.popleft()
            # Ensure slot is valid
            if slot >= self.particles.shape[1]:
                print(f"ERROR: free slot {slot} >= array size {self.particles.shape[1]}")
                return None
            return slot
        
        # Next, try to use a new slot within current array bounds
        current_array_size = self.particles.shape[1]
        if self.active_count < current_array_size:
            slot = self.active_count
            self.active_count += 1
            return slot
        
        # Need to expand capacity - array is full
        if current_array_size >= self.capacity:
            old_size = self.particles.shape[1]
            self._expand_capacity()
            new_size = self.particles.shape[1]
            print(f"Expanded array from {old_size} to {new_size} (capacity now {self.capacity})")
            
            # After expansion, active_count should be valid
            slot = self.active_count
            self.active_count += 1
            return slot
        else:
            print(f"ERROR: active_count {self.active_count} >= array_size {current_array_size} but capacity {self.capacity} > array_size")
            return None
    
    def _expand_capacity(self):
        """Double the particle array capacity when needed"""
        old_capacity = self.capacity
        old_shape = self.particles.shape  # (components, particles)
        old_is_active_size = len(self.is_active)
        self.capacity *= 2
        
        print(f"Expanding {self.params.type}: old_capacity={old_capacity}, old_shape={old_shape}, old_is_active_size={old_is_active_size}, new_capacity={self.capacity}")
        
        # Use actual array dimensions, not assumed NUMQ + 1
        old_components = old_shape[0]  # Number of components (varies by species)
        old_particles = old_shape[1]   # Number of particles
        
        # Expand particle array - preserve exact component count
        new_particles = np.zeros((old_components, self.capacity))
        copy_size = min(old_particles, old_particles)  # This is just old_particles, but keeping pattern
        new_particles[:, :copy_size] = self.particles[:, :copy_size]
        
        # Verify no NaN values were copied
        if np.any(np.isnan(new_particles[:, :copy_size])) or np.any(np.isinf(new_particles[:, :copy_size])):
            print(f"ERROR: NaN/inf values detected during {self.params.type} array expansion!")
            # Clean up any NaN/inf values
            nan_mask = np.isnan(new_particles) | np.isinf(new_particles)
            new_particles[nan_mask] = 0.0
        
        self.particles = new_particles
        
        # Expand is_active array - copy exactly what exists
        new_is_active = np.zeros(self.capacity, dtype=bool)
        copy_active_size = min(old_is_active_size, len(self.is_active), old_particles)
        new_is_active[:copy_active_size] = self.is_active[:copy_active_size]
        self.is_active = new_is_active
        
        print(f"Expanded {self.params.type}: {old_components} components, copied {copy_size} particles and {copy_active_size} active flags. New shape: {self.particles.shape}")
        
        # Verify expansion worked
        if self.particles.shape[1] != self.capacity:
            print(f"ERROR: Array expansion failed! Expected size {self.capacity}, got {self.particles.shape[1]}")
        if len(self.is_active) != self.capacity:
            print(f"ERROR: is_active expansion failed! Expected size {self.capacity}, got {len(self.is_active)}")
    
    def add_particle(self, particle_data):
        """Efficiently add particle using slot-based management"""
        if isinstance(particle_data, np.ndarray):
            expected_components = self.particles.shape[0]
            if particle_data.shape[0] != expected_components:
                print(f"Warning: Component mismatch for {self.params.type}: got {particle_data.shape[0]}, expected {expected_components}")
                return
                
            slot = self._get_free_slot()
            if slot is None or slot >= self.particles.shape[1]:
                print(f"ERROR: Cannot add particle, invalid slot {slot}")
                return
                
            self.particles[:, slot] = particle_data.flatten()
            self.is_active[slot] = True
            # Invalidate cache when adding particles
            self._cache_valid = False
            # print(self.params.type, "!%! added particle to slot", slot, "with x=", self.particles[self.pc.XCOMP, slot], "y=", self.particles[self.pc.YCOMP, slot])
        else:
            print(f"Warning: Invalid particle data format for species {self.params.type}")
    
    def add_particles(self, particle_list):
        """Efficiently add multiple particles"""
        # if not particle_list:
        #     return
            
        expected_components = self.particles.shape[0]
        added_count = 0
        
        for particle_data in particle_list:
            if isinstance(particle_data, np.ndarray) and particle_data.shape[0] == expected_components:
                slot = self._get_free_slot()
                if slot is None or slot >= self.particles.shape[1]:
                    print(f"ERROR: Cannot add particle in batch, invalid slot {slot}")
                    continue
                self.particles[:, slot] = particle_data.flatten()
                self.is_active[slot] = True
                added_count += 1
            else:
                print(f"Warning: Skipping particle with wrong component count: got {particle_data.shape[0] if hasattr(particle_data, 'shape') else 'invalid'}, expected {expected_components}")
        
        # Invalidate cache only once after adding all particles
        if added_count > 0:
            self._cache_valid = False

    def estimate_initial_weight(self):
        return (self.inp.n_n * self.inp.dx * self.inp.dy) / self.inp.N_ppc

    def process_collisions(self):
        # print("!#@! PROCESS COLLISIONS cALLED FOR", self.params.type)
        if self.params.type not in ["e", "i"] or self.collision_data is None:
            return []

        new_particles = []
        collision_events = []
        
        m_e = self.inp.m_e
        k_B = self.inp.k_B
        n_e = self.simulation.get_species_number_density("e")
        v_e = self.simulation.get_species_velocity("e")
        T_e = self.simulation.get_species_temperature("e")
        n_n = self.simulation.get_species_number_density("n")
        particles_to_remove = []

        # print("!#@! PROCESS COLLISIONS FOR", self.params.type)
        active_count = np.sum(self.is_active)
        # print("N ACTIVE PARTICLES", active_count)
        
        if self.inp.e_system == "euler2d":
            # Use optimized active particle retrieval
            active_indices = self._get_active_indices()
            if len(active_indices) < 100:  # Only print for small numbers to avoid spam
                print(f"!ACTIVE! Processing {len(active_indices)} active particles")

            for n in active_indices:
                events = self._attempt_collisions(n, n_n)
                
                for event in events:
                    collision_events.append(event)
                    if event.event_type == "ionization":
                        new_particles.extend(event.products)
                        # print("!#! IONIZED")
                    elif event.event_type in ["first_excitation", "second_excitation", "third_excitation", "fourth_excitation"]:
                        self._apply_energy_loss(n, event.energy_change)
                    elif event.event_type.startswith("elastic"):
                        self._apply_elastic_scattering(n, event)

            self.collision_events.extend(collision_events)
            
        elif self.inp.e_system == "quasineutral":
            return # TODO: IDK DO WE NEED TO IMPLEMENT FIFE ANYMORE
            
            # def I_theta(theta, beta2):
            #     # analytic J1 piece
            #     J1 = theta * np.exp(-1.0/theta) - special.exp1(1.0/theta)   # exp1 is E1
            #     part1 = np.log(1.25 * beta2) * J1

            #     # define integrand for J2
            #     def integrand(u):
            #         return np.exp(-u/theta) * (1.0 - 1.0/u) * np.log(u)

            #     # numeric J2 via adaptive quad (good for infinite tail / exponential decay)
            #     J2, err = integrate.quad(integrand, 1.0, np.inf, epsabs=1e-10, epsrel=1e-10, limit=200)

            #     return part1 + J2
            
            # f_e = n_e * (m_e / (2*np.pi * k_B * T_e))**(3/2) * np.exp(-(m_e*v_e**2)/(2*k_B*T_e))
            
            # f = np.exp
            # I_theta = integrate.quad()
            # ion_production_rate = quad()
            
            epsilon_i = 12.13 # eV
            
            beta_1 = 1.00 # darwin ionization model constants for xenon - from Fife 1999 Table 2.3
            beta_2 = 0.80 #
            Q = 4.13e-13 #
            
            def integrand(u, theta):
                return np.exp(-(u/theta)) * (1-(1/u)) * np.log(1.25 * beta_2 * u)
            
            for i in range (0, self.inp.nx):
                for j in range (0, self.inp.ny):
                    theta = k_B * T_e[i, j] / epsilon_i # TODO: will we have to for loop over T_e...
                    I_theta = integrate.quad(integrand, 1, np.inf, args=(theta,))
            
                    zeta = (Q * beta_1 * I_theta[i, j]) / theta**(3/2)
                    ion_rate = zeta * n_e[i, j] * n_n[i, j]
        
        return new_particles


    def _attempt_collisions(self, particle_idx: int, neutral_density_field: np.ndarray):
        # print("!#! ATTEMPT COLLISION FOR", self.params.type)
        
        # Safety check for valid particle index
        if not (0 <= particle_idx < self.capacity and self.is_active[particle_idx]):
            return []
            
        x = self.particles[self.pc.XCOMP, particle_idx]
        y = self.particles[self.pc.YCOMP, particle_idx]
        vx = self.particles[self.pc.UCOMP, particle_idx]
        vy = self.particles[self.pc.VCOMP, particle_idx]
        vz = self.particles[self.pc.WCOMP, particle_idx] if (self.pc.WCOMP < self.pc.NUMQ or self.params.type == "e") else 0.0
        weight = self.particles[self.WEIGHT, particle_idx]

        grid_coords = self._get_grid_coordinates(x, y)
        if grid_coords is None:
            print("[ATTEMPT_COLLISIONS] grid_coords is None")
            return []
        
        if self.params.type == "e":
            # !GOODENOUGH!
            v_rel = np.sqrt(vx**2 + vy**2 + vz**2)
            # TODO: do we need to add actual relative velocity for ion-neutral?
        else:
            v_rel = np.sqrt(vx**2 + vy**2 + vz**2)

        if v_rel < 1e-10:
            print("[ATTEMPT_COLLISIONS] v_rel < 1e-10")
            print("v_rel is", v_rel)
            return []

        # !NORM! 
        energy_ev = 0.5 * self.params.mass * v_rel**2 / abs(self.params.charge)
        # energy_ev = 0.5 * self.params.mass * v_rel**2

        neutral_density = self._interpolate_density(x, y, neutral_density_field)
        if neutral_density <= 0:
            print("[ATTEMPT_COLLISIONS] neutral_density <= 0")
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
                print("[ATTEMPT_COLLISIONS] NOT ENOUGH ENERGY FOR", collision_type, energy_ev)
                continue
                
            # !GOODENOUGH!
            sigma = collision_data["cross_section_func"](energy_ev)
                        
            if sigma <= 0:
                #print("NEGATIVE SIGMA FOR", collision_type)
                continue

            # Deposit sigma onto grid for cross section tracking

            # nu = n * sigma * v
            # TODO: DO WE NEED TO MULTIPLY THIS BY WEIGHT OR SOMETHING?? PROBABILITY IS IN 1e-6 RANGE WITHOUT (WHICH IS CORRECT, BUT LOW)
            nu_collision = neutral_density * sigma * v_rel # * 1e5
            P_collision = 1.0 - np.exp(-nu_collision * self.inp.dt)
            
            i, j = self._get_grid_coordinates(x, y)
            if 0 <= i < self.inp.nx and 0 <= j < self.inp.ny:
                self.sigma_temp_storage[i][j-1].append(P_collision)
                
            print("--[ATTEMPT_COLLISIONS]-- for", collision_type)
            print("PROBABILITY IS", P_collision)
            print("NEUTRAL DENSITY IS", neutral_density)
            print("SIGMA IS", sigma)
            print("ELECTRON SPEED IS", v_rel)
            print("ELECTRON ENERGY IS", energy_ev)
            # print("WEIGHT IS", weight)

            # monte carlo
            if np.random.rand() < P_collision:
                event = self._create_collision_event(
                    collision_type, particle_idx, x, y, vx, vy, vz, 
                    weight, energy_ev
                )
                if event:
                    events.append(event)
                    i, j = self._get_grid_coordinates(x, y)
                    # if i < self.inp.nx and j < self.inp.ny:
                    #     self.num_collisions[i, j] += 1

                    if event.event_type == "ionization":
                        self.collision_count[0] += 1
                        # Store X position of ionization event
                        x_pos = x
                        self.ionization_positions_x.append(x_pos)
                    elif event.event_type.endswith("excitation"):
                        self.collision_count[1] += 1
                        self.ionization_positions_x.append(x_pos)
                    elif event.event_type.startswith("elastic"):
                        self.collision_count[2] += 1

                # only allow one collision per timestep per particle
                break
            #else:
                #print("NOT LUCKY MONTE CARLO FOR", collision_type)

        return events

    def _create_collision_event(self, collision_type: str, particle_idx: int, 
                               x: float, y: float, vx: float, vy: float, vz: float,
                               weight: float, energy_ev: float):
        
        if collision_type == "ionization" and self.params.type == "e":
            # print("!#!#! IONIZED! AT", x/self.inp.dx, y/self.inp.dx)
            return self._create_ionization_event(particle_idx, x, y, vx, vy, vz, weight, energy_ev)
        elif collision_type in ["first_excitation", "second_excitation", "third_excitation", "fourth_excitation"] and self.params.type == "e":
        #     print("!#!#! EXCITED! AT", x/self.inp.dx, y/self.inp.dx)
            return self._create_excitation_event(collision_type, particle_idx, energy_ev)
        elif collision_type.startswith("elastic"):
            # print("!#!#! ELASTIC! AT", x/self.inp.dx, y/self.inp.dx)
            return self._create_elastic_event(collision_type, particle_idx)
        
        return None

    def _create_ionization_event(self, particle_idx: int, x: float, y: float, 
                                vx: float, vy: float, vz: float, weight: float, energy_ev: float):
        """ e + Xe -> e + e + Xe+"""
        # Validate input parameters
        if (np.isnan(x) or np.isnan(y) or np.isnan(energy_ev) or np.isnan(weight) or
            np.isinf(x) or np.isinf(y) or np.isinf(energy_ev) or np.isinf(weight)):
            print(f"ERROR: Invalid ionization event parameters: x={x}, y={y}, energy={energy_ev}, weight={weight}")
            return None
            
        ionization_threshold = self.collision_data.ionization_threshold  # Use LXCAT value: 12.13 eV
        available_energy_ev = energy_ev - ionization_threshold
        
        if available_energy_ev <= 0:
            return None

        # !NORM! available_energy_j = available_energy_ev * self.collision_data.E_CHARGE
        available_energy_j = available_energy_ev
        
        # Determine component count based on species type
        ion_components = self.pc.NUMQ + 1  # Standard components for ions
        electron_components = self.pc.NUMQ + 1  # May be different for electrons
        
        ion = np.zeros(ion_components)
        ion[self.pc.XCOMP] = x
        ion[self.pc.YCOMP] = y
        ion[self.WEIGHT] = weight
        
        electron = np.zeros(electron_components)
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
        
        # Validate created particles before returning
        for i, product in enumerate(products):
            if np.any(np.isnan(product)) or np.any(np.isinf(product)):
                print(f"ERROR: Invalid product particle {i} in ionization event: {product}")
                return None
        
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
        # Safety check for valid particle index
        if not (0 <= particle_idx < self.capacity and self.is_active[particle_idx]):
            return
            
        vx = self.particles[self.pc.UCOMP, particle_idx]
        vy = self.particles[self.pc.VCOMP, particle_idx]
        vz = self.particles[self.pc.WCOMP, particle_idx] if self.pc.WCOMP < self.pc.NUMQ else 0.0
        
        v_current = np.sqrt(vx**2 + vy**2 + vz**2)
        if v_current < 1e-10:
            return
            
        ke_current_j = 0.5 * self.params.mass * v_current**2
        ke_current_ev = ke_current_j # / self.collision_data.E_CHARGE
        
        ke_new_ev = max(0.0, ke_current_ev - energy_loss_ev)
        ke_new_j = ke_new_ev # * self.collision_data.E_CHARGE
        
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
        # Safety check for valid particle index
        if not (0 <= particle_idx < self.capacity and self.is_active[particle_idx]):
            return
            
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

    def _get_active_indices(self):
        """Get cached active particle indices to avoid recomputation"""
        if not self._cache_valid:
            safe_capacity = min(self.capacity, self.particles.shape[1]) #TODO: , len(self.is_active))
            active_indices = []
            for i in range(safe_capacity):
                if (self.is_active[i] and 
                    not np.isnan(self.particles[self.pc.XCOMP, i]) and 
                    # TODO: not np.isnan(self.particles[self.pc.YCOMP, i]) and
                    not np.isinf(self.particles[self.pc.XCOMP, i]) and 
                    not np.isinf(self.particles[self.pc.YCOMP, i])):
                    active_indices.append(i)
                elif self.is_active[i]:  # Active but invalid data
                    print(f"WARNING: Deactivating {self.params.type} particle {i} with invalid position")
                    self.is_active[i] = False
                    self.free_slots.append(i)
                    self.particles[:, i] = 0.0
            
            self._active_indices_cache = active_indices
            self._cache_valid = True
        
        return self._active_indices_cache
    
    def _interpolate_electric_field_batch(self, x_positions, y_positions):
        """Batch interpolate electric fields for multiple particles"""
        if len(x_positions) == 0:
            return np.array([]), np.array([])
        
        # Vectorized grid coordinate calculation
        x_grid = (x_positions - self.inp.grid_x[0]) / self.inp.dx
        y_grid = (y_positions - self.inp.grid_y[0]) / self.inp.dy
        
        ix = np.floor(x_grid).astype(int)
        iy = np.floor(y_grid).astype(int)
        
        # Interpolation weights
        wx = x_grid - ix
        wy = y_grid - iy
        
        # Get array bounds
        nx_max = self.fields.E.shape[0] - 1
        ny_max = self.fields.E.shape[1] - 1
        
        # Initialize output arrays
        Ex_values = np.zeros_like(x_positions)
        Ey_values = np.zeros_like(y_positions)
        
        # Vectorized bounds checking and interpolation
        valid_mask = ((ix >= 0) & (ix <= nx_max) & (iy >= 0) & (iy <= ny_max))
        
        if np.any(valid_mask):
            ix_valid = ix[valid_mask]
            iy_valid = iy[valid_mask]
            wx_valid = wx[valid_mask]
            wy_valid = wy[valid_mask]
            
            # Bilinear interpolation for valid points
            Ex_interp = (self.fields.E[ix_valid, iy_valid, 0] * (1 - wx_valid) * (1 - wy_valid))
            Ey_interp = (self.fields.E[ix_valid, iy_valid, 1] * (1 - wx_valid) * (1 - wy_valid))
            
            # Add contributions from neighboring points where they exist
            right_valid = (ix_valid + 1 <= nx_max)
            if np.any(right_valid):
                Ex_interp[right_valid] += (self.fields.E[ix_valid[right_valid] + 1, iy_valid[right_valid], 0] * 
                                         wx_valid[right_valid] * (1 - wy_valid[right_valid]))
                Ey_interp[right_valid] += (self.fields.E[ix_valid[right_valid] + 1, iy_valid[right_valid], 1] * 
                                         wx_valid[right_valid] * (1 - wy_valid[right_valid]))
            
            top_valid = (iy_valid + 1 <= ny_max)
            if np.any(top_valid):
                Ex_interp[top_valid] += (self.fields.E[ix_valid[top_valid], iy_valid[top_valid] + 1, 0] * 
                                       (1 - wx_valid[top_valid]) * wy_valid[top_valid])
                Ey_interp[top_valid] += (self.fields.E[ix_valid[top_valid], iy_valid[top_valid] + 1, 1] * 
                                       (1 - wx_valid[top_valid]) * wy_valid[top_valid])
            
            corner_valid = right_valid & top_valid
            if np.any(corner_valid):
                Ex_interp[corner_valid] += (self.fields.E[ix_valid[corner_valid] + 1, iy_valid[corner_valid] + 1, 0] * 
                                          wx_valid[corner_valid] * wy_valid[corner_valid])
                Ey_interp[corner_valid] += (self.fields.E[ix_valid[corner_valid] + 1, iy_valid[corner_valid] + 1, 1] * 
                                          wx_valid[corner_valid] * wy_valid[corner_valid])
            
            Ex_values[valid_mask] = Ex_interp
            Ey_values[valid_mask] = Ey_interp
        
        return Ex_values, Ey_values
    
    def _get_grid_coordinates(self, x: float, y: float):
        # Validate inputs first
        if np.isnan(x) or np.isnan(y) or np.isinf(x) or np.isinf(y):
            return None
            
        
        ix = int((x) / self.inp.dx)
        iy = int((y) / self.inp.dy)
        
        # if ix and iy:
        #     return ix, iy
    
        # Proper boundary checking to prevent edge accumulation
        if self.inp.ng <= ix <= self.inp.nx + self.inp.ng and self.inp.ng <= iy <= self.inp.ny + self.inp.ng - 2:
            return ix, iy
        return None

    def update_cross_section_grid(self):
        """Average sigma values from temporary storage and update cross_section_grid"""
        for i in range(self.inp.nx):
            for j in range(self.inp.ny):
                if len(self.sigma_temp_storage[i][j]) > 0:
                    # self.cross_section_grid[i, j] = np.mean(self.sigma_temp_storage[i][j])
                    self.cross_section_grid[i, j] = np.sum(self.sigma_temp_storage[i][j])
                    # Clear the temporary storage after averaging
                    self.sigma_temp_storage[i][j].clear()

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
        """Efficiently remove particles using slot recycling"""
        if not indices_to_remove:
            return
        
        # Invalidate cache when removing particles
        self._cache_valid = False
        
        # Mark slots as inactive and add to free list
        for idx in indices_to_remove:
            if 0 <= idx < self.capacity and self.is_active[idx]:
                self.is_active[idx] = False
                self.free_slots.append(idx)
                # Clear particle data for cleanliness
                self.particles[:, idx] = 0.0

    def split_particle(self, idx):
        """Split particle using efficient slot management"""
        if not (0 <= idx < self.capacity and self.is_active[idx]):
            return
            
        particle = self.particles[:, idx].copy()
        new_weight = particle[self.WEIGHT] / 2
        
        # Modify original particle
        dx = np.random.normal(0, self.inp.dx * 0.01)
        dy = np.random.normal(0, self.inp.dy * 0.01)
        self.particles[self.WEIGHT, idx] = new_weight
        self.particles[self.pc.XCOMP, idx] += dx
        self.particles[self.pc.YCOMP, idx] += dy
        
        # Create second particle in new slot
        new_slot = self._get_free_slot()
        self.particles[:, new_slot] = particle
        self.particles[self.WEIGHT, new_slot] = new_weight
        self.particles[self.pc.XCOMP, new_slot] -= dx
        self.particles[self.pc.YCOMP, new_slot] -= dy
        self.is_active[new_slot] = True

    def merge_particles(self, idx1, idx2):
        """Merge particles using efficient slot management"""
        if not (0 <= idx1 < self.capacity and self.is_active[idx1] and 
                0 <= idx2 < self.capacity and self.is_active[idx2]):
            return
            
        p1 = self.particles[:, idx1]
        p2 = self.particles[:, idx2]
        w_total = p1[self.WEIGHT] + p2[self.WEIGHT]
        
        if w_total == 0:
            return
            
        # Merge into first particle
        for q in [self.pc.XCOMP, self.pc.YCOMP, self.pc.UCOMP, self.pc.VCOMP]:
            self.particles[q, idx1] = (p1[q] * p1[self.WEIGHT] + p2[q] * p2[self.WEIGHT]) / w_total
        self.particles[self.WEIGHT, idx1] = w_total
        
        # Remove second particle
        self.is_active[idx2] = False
        self.free_slots.append(idx2)
        self.particles[:, idx2] = 0.0

    # !DEBUG! change min_ppc back to 100
    def enforce_ppc(self, min_ppc=1, max_ppc=200):
        """Enforce particles per cell using efficient slot-based system"""
        cell_map = {}
        
        # Use optimized active particle retrieval
        active_indices = self._get_active_indices()
        
        if len(active_indices) > 0:
            # Vectorized cell mapping
            x_positions = self.particles[self.pc.XCOMP, active_indices]
            y_positions = self.particles[self.pc.YCOMP, active_indices]
            
            ix_values = ((x_positions - self.inp.grid_x[0]) / self.inp.dx).astype(int)
            iy_values = ((y_positions - self.inp.grid_y[0]) / self.inp.dy).astype(int)
            
            # Build cell map efficiently
            for i, (ix, iy) in enumerate(zip(ix_values, iy_values)):
                cell_key = (ix, iy)
                if cell_key not in cell_map:
                    cell_map[cell_key] = []
                cell_map[cell_key].append(active_indices[i])
            
        for (ix, iy), indices in cell_map.items():
            count = len(indices)
            if count < min_ppc:
                needed = min_ppc - count
                for idx in indices[:needed]:
                    if self.is_active[idx]:  # Double-check active status
                        self.split_particle(idx)
            elif count > max_ppc:
                to_merge = (count - max_ppc) // 2
                pairs = zip(indices[::2], indices[1::2])
                for idx1, idx2 in list(pairs)[:to_merge]:
                    if (self.is_active[idx1] and self.is_active[idx2]):
                        self.merge_particles(idx1, idx2)

    def get_charge_density(self):
        """Compute charge density field from active particles only"""
        rho = np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))
        
        # Use optimized active particle retrieval
        active_indices = self._get_active_indices()
        
        if len(active_indices) > 0:
            # Vectorized charge density computation
            x_positions = self.particles[self.pc.XCOMP, active_indices]
            y_positions = self.particles[self.pc.YCOMP, active_indices] 
            
            # Vectorized grid index calculation
            ix_values = ((x_positions - self.inp.grid_x[0]) / self.inp.dx).astype(int)
            iy_values = ((y_positions - self.inp.grid_y[0]) / self.inp.dy).astype(int)
            
            # Bounds check
            valid_mask = ((ix_values >= 0) & (ix_values < self.inp.nx_with_ghosts) & 
                         (iy_values >= 0) & (iy_values < self.inp.ny_with_ghosts))
            
            if np.any(valid_mask):
                ix_valid = ix_values[valid_mask]
                iy_valid = iy_values[valid_mask]
                
                # Use np.add.at for efficient accumulation
                np.add.at(rho, (ix_valid, iy_valid), self.params.charge)
        
        return rho
    
    def _sort_particles_spatially(self):
        """Sort active particles by spatial position for better cache locality"""
        # Use optimized active particle retrieval
        active_indices = self._get_active_indices()
                
        if len(active_indices) < 2:
            return
        
        # Vectorized position extraction
        x_positions = self.particles[self.pc.XCOMP, active_indices]
        y_positions = self.particles[self.pc.YCOMP, active_indices]
        
        # Create sorting indices based on spatial coordinates
        # Z-order sorting for better 2D locality
        sort_indices = np.lexsort((y_positions, x_positions))
        sorted_active_indices = np.array(active_indices)[sort_indices]
        
        # Create new particle array with sorted order
        temp_particles = np.zeros_like(self.particles)
        temp_is_active = np.zeros_like(self.is_active)
        
        # Vectorized copy of sorted particles
        temp_particles[:, :len(sorted_active_indices)] = self.particles[:, sorted_active_indices]
        temp_is_active[:len(sorted_active_indices)] = True
        
        # Update arrays
        self.particles = temp_particles
        self.is_active = temp_is_active
        
        # Update free slots list and invalidate cache
        self.free_slots = deque(range(len(active_indices), self.capacity))
        self._cache_valid = False
        
        print(f"Sorted {len(active_indices)} {self.params.type} particles spatially")
    
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
