from ascfd.particle.constants import ParticleConstants
from ascfd.particle.ics import ParticleInitialConditions
from ascfd.inputs import Inputs
from ascfd.params import SpeciesParams
from ascfd.fields.fields import Fields

import numpy as np

class IonizationEvent:
    neutral_idx: int
    ion_particle: np.ndarray
    electron_particle: np.ndarray
    energy_deposited: float

class CrossSectionData:
    def __init__(self, filename: str = "Xe_e_ionization.txt"):
        self.energies = None
        self.cross_sections = None
        self.ionization_threshold = 12.13  # eV for Xenon
        self.load_data(filename)

    def load_data(self, filename):
        data = np.loadtxt(filename, comments="#")
        self.energies = data[:, 0]
        self.cross_sections = data[:, 1]

    def get_cross_section(self, energy_ev: float) -> float:
        if energy_ev < self.ionization_threshold:
            return 0.0
        return float(np.interp(energy_ev, self.energies, self.cross_sections))

class ParticleSpecies:
    def __init__(self, params: SpeciesParams, a_inputs: Inputs, fields: Fields):
        self.pc = ParticleConstants()
        self.fields = fields
        self.inp = a_inputs
        self.params = params
        self.dt = None

        self.particles = np.zeros((self.pc.NUMQ + 1, self.inp.n_particles))
        self.WEIGHT = self.pc.NUMQ

        self.particles[self.WEIGHT, :] = self.estimate_initial_weight()
        self.ics = ParticleInitialConditions(self.particles, self.inp, self.params)
        self.particles = self.ics.apply_ics()

        self.cross_section_data = CrossSectionData() if params.type == "n" else None
        self.ionization_events = []

    def estimate_initial_weight(self):
        Vc = self.inp.dx * self.inp.dy
        target_ppc = 100  # N_PPC
        return (self.params.density * Vc) / target_ppc

    def update(self):
        new_ions, new_electrons = [], []

        # only neutrals ionize
        if self.params.type == "n":
            new_ions, new_electrons = self.ionize()

        # particle per cell enforcement
        self.enforce_ppc()

        # update electric field with current charge distribution
        charge_density = self.get_charge_density()
        self.fields.update_E(charge_density)

        return new_ions + new_electrons

    def ionize(self):
        if self.params.type != "n":
            return [], []

        new_ions = []
        new_electrons = []
        ionization_events = []
        electron_density_field = self._compute_electron_density_field()

        for n in range(self.particles.shape[1]):
            event = self._attempt_ionization(n, electron_density_field)
            if event is not None:
                ionization_events.append(event)
                new_ions.append(event.ion_particle)
                new_electrons.append(event.electron_particle)

        if ionization_events:
            self._remove_ionized_neutrals([event.neutral_idx for event in ionization_events])
            self.ionization_events.extend(ionization_events)

        return new_ions, new_electrons

    def _compute_electron_density_field(self):
        # placeholder - override with actual species lookup if needed
        return np.zeros((self.inp.nx_with_ghosts, self.inp.ny_with_ghosts))

    def _attempt_ionization(self, particle_idx: int, electron_density_field: np.ndarray):
        x = self.particles[self.pc.XCOMP, particle_idx]
        y = self.particles[self.pc.YCOMP, particle_idx]
        vx = self.particles[self.pc.UCOMP, particle_idx]
        vy = self.particles[self.pc.VCOMP, particle_idx]
        vz = self.particles[self.pc.WCOMP, particle_idx] if self.pc.WCOMP < self.pc.NUMQ else 0.0
        weight = self.particles[self.WEIGHT, particle_idx]

        grid_coords = self._get_grid_coordinates(x, y)
        if grid_coords is None:
            return None

        ix, iy = grid_coords
        v_rel = np.sqrt(vx**2 + vy**2 + vz**2)
        if v_rel < 1e-10:
            return None

        energy_ev = 0.5 * self.params.mass * v_rel**2 / self.pc.E_CHARGE
        sigma = self.cross_section_data.get_cross_section(energy_ev)
        if sigma <= 0:
            return None

        electron_density = self._interpolate_density(x, y, electron_density_field)
        if electron_density <= 0:
            return None

        # Villafana 2021 (pg 60): nu = ne * sigma * v
        nu_ionization = electron_density * sigma * v_rel
        P_ionization = 1.0 - np.exp(-nu_ionization * self.dt)

        # monte carlo sampling
        if np.random.rand() >= P_ionization:
            return None

        ion_particle, electron_particle, energy_cost = self._create_ionization_products(
            x, y, vx, vy, vz, weight, energy_ev
        )

        return IonizationEvent(
            neutral_idx=particle_idx,
            ion_particle=ion_particle,
            electron_particle=electron_particle,
            energy_deposited=energy_cost
        )

    def _get_grid_coordinates(self, x: float, y: float):
        ix = int((x - self.inp.grid_x[0]) / self.inp.dx)
        iy = int((y - self.inp.grid_y[0]) / self.inp.dy)
        if 0 <= ix < self.inp.nx_with_ghosts and 0 <= iy < self.inp.ny_with_ghosts:
            return ix, iy
        return None

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

    def _create_ionization_products(self, x: float, y: float, vx: float, vy: float, vz: float, weight: float, energy_ev: float):
        ionization_energy = self.cross_section_data.ionization_threshold
        available_energy_ev = energy_ev - ionization_energy
        available_energy_j = available_energy_ev * self.pc.E_CHARGE

        ion = np.zeros(self.pc.NUMQ + 1)
        ion[self.pc.XCOMP] = x
        ion[self.pc.YCOMP] = y
        ion[self.WEIGHT] = weight

        electron = np.zeros(self.pc.NUMQ + 1)
        electron[self.pc.XCOMP] = x
        electron[self.pc.YCOMP] = y
        electron[self.WEIGHT] = weight

        if available_energy_ev > 0:
            electron_energy_fraction = 0.8  # most excess energy goes to electron
            electron_energy_j = available_energy_j * electron_energy_fraction
            electron_speed = np.sqrt(2 * electron_energy_j / self.pc.ELECTRON_MASS)
            theta = np.random.uniform(0, 2 * np.pi)
            phi = np.random.uniform(0, np.pi)
            electron[self.pc.UCOMP] = electron_speed * np.sin(phi) * np.cos(theta)
            electron[self.pc.VCOMP] = electron_speed * np.sin(phi) * np.sin(theta)
            if self.pc.WCOMP < self.pc.NUMQ:
                electron[self.pc.WCOMP] = electron_speed * np.cos(phi)

            ion_energy_j = available_energy_j * (1 - electron_energy_fraction)
            ion_speed = np.sqrt(2 * ion_energy_j / self.params.mass)
            ion_theta = np.random.uniform(0, 2 * np.pi)
            ion[self.pc.UCOMP] = vx + ion_speed * np.cos(ion_theta) * 0.1
            ion[self.pc.VCOMP] = vy + ion_speed * np.sin(ion_theta) * 0.1
            if self.pc.WCOMP < self.pc.NUMQ:
                ion[self.pc.WCOMP] = vz * 0.9
        else:
            # not enough energy
            ion[self.pc.UCOMP] = vx
            ion[self.pc.VCOMP] = vy
            electron[self.pc.UCOMP] = 0.0
            electron[self.pc.VCOMP] = 0.0
            if self.pc.WCOMP < self.pc.NUMQ:
                ion[self.pc.WCOMP] = vz
                electron[self.pc.WCOMP] = 0.0

        return ion, electron, ionization_energy

    def _remove_ionized_neutrals(self, indices_to_remove: list[int]):
        if not indices_to_remove:
            return
        keep_mask = np.ones(self.particles.shape[1], dtype=bool)
        keep_mask[indices_to_remove] = False
        self.particles = self.particles[:, keep_mask]

    def split_particle(self, idx):
        # duplicate particle with half weight and slight offset
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
        # weighted average merge
        p1 = self.particles[:, idx1]
        p2 = self.particles[:, idx2]
        w_total = p1[self.WEIGHT] + p2[self.WEIGHT]
        merged = np.zeros_like(p1)
        for q in [self.pc.XCOMP, self.pc.YCOMP, self.pc.UCOMP, self.pc.VCOMP]:
            merged[q] = (p1[q] * p1[self.WEIGHT] + p2[q] * p2[self.WEIGHT]) / w_total
        merged[self.WEIGHT] = w_total
        self.particles[:, idx1] = merged
        self.particles = np.delete(self.particles, idx2, axis=1)

    def enforce_ppc(self, min_ppc=100, max_ppc=200):
        # maintain consistent PPC in each grid cell
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
        # deposit particle charge to grid
        rho = np.zeros_like(self.fields.E[0])
        for i in range(self.particles.shape[1]):
            x = self.particles[self.pc.XCOMP, i]
            y = self.particles[self.pc.YCOMP, i]
            ix = int((x - self.inp.grid_x[0]) / self.inp.dx)
            iy = int((y - self.inp.grid_y[0]) / self.inp.dy)
            if 0 <= ix < self.inp.nx_with_ghosts and 0 <= iy < self.inp.ny_with_ghosts:
                rho[ix, iy] += self.params.charge
        return rho