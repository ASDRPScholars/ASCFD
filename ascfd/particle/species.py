from ascfd.particle.constants import ParticleConstants
from ascfd.particle.ics import ParticleInitialConditions
from ascfd.inputs import Inputs
from ascfd.params import SpeciesParams
from ascfd.fields.fields import Fields

import numpy as np

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

        self.ics = ParticleInitialConditions(self.particles, self.inp)

        self.particles = self.ics.apply_ics()
        
        print("PARTICLES X POS INIT:", self.particles[self.pc.XCOMP])

    def estimate_initial_weight(self):
        Vc = self.inp.dx * self.inp.dy  
        target_ppc = 100  # N_PPC
        return (self.params.density * Vc) / target_ppc

    def update(self):
        new_ions = []

        if self.inp.particle_flow_type == "passive":
            pass

        elif self.inp.particle_flow_type == "multispecies":
            self.V = np.zeros((self.particles.shape[1], 3))  # Shape (n_particles, 3)
            self.V[:, 0] = self.particles[self.pc.UCOMP]
            self.V[:, 1] = self.particles[self.pc.VCOMP]
            self.V[:, 2] = self.particles[self.pc.WCOMP] if self.pc.WCOMP < self.pc.NUMQ else 0.0

            for n in range(self.particles.shape[1]):
                x = self.particles[self.pc.XCOMP, n]
                y = self.particles[self.pc.YCOMP, n]

                ix = int((x - self.inp.grid_x[0]) / self.inp.dx)
                iy = int((y - self.inp.grid_y[0]) / self.inp.dy)

                if 0 <= ix < self.inp.nx_with_ghosts and 0 <= iy < self.inp.ny_with_ghosts:
                    self.calculate_mom_source_terms(ix, iy, n)

            self.apply_mom_source_terms()

            
            self.particles[self.pc.XCOMP] += self.particles[self.pc.UCOMP] * self.dt
            self.particles[self.pc.YCOMP] += self.particles[self.pc.VCOMP] * self.dt
            if self.pc.WCOMP < self.pc.NUMQ:
                self.particles[self.pc.ZCOMP] += self.particles[self.pc.WCOMP] * self.dt

        
        if self.params.type == "n":
            new_ions = self.ionize()

        
        charge_density = self.get_charge_density()
        self.fields.update_E(charge_density)

        self.enforce_ppc()

        return new_ions

    def calculate_mom_source_terms(self, i, j, n):
        Ex = self.fields.E[0][i, j]
        Ey = self.fields.E[1][i, j]
        Bx = self.fields.B[0][i, j]
        By = self.fields.B[1][i, j]
        Bz = self.fields.B[2][i, j]

        E = np.array([Ex, Ey, 0.0])
        B = np.array([Bx, By, Bz])
        V = self.V[n]

        lorentz = (self.params.charge / self.params.mass) * (E + np.cross(V, B))
        self.V[n] += lorentz * self.dt

    def apply_mom_source_terms(self):
        self.particles[self.pc.UCOMP] = self.V[:, 0]
        self.particles[self.pc.VCOMP] = self.V[:, 1]
        if self.pc.WCOMP < self.pc.NUMQ:
            self.particles[self.pc.WCOMP] = self.V[:, 2]

    def ionize(self):
        new_ions = []
        for n in range(self.particles.shape[1]):
            x = self.particles[self.pc.XCOMP, n]
            y = self.particles[self.pc.YCOMP, n]
            ix = int((x - self.inp.grid_x[0]) / self.inp.dx)
            iy = int((y - self.inp.grid_y[0]) / self.inp.dy)

            if not (0 <= ix < self.inp.nx_with_ghosts and 0 <= iy < self.inp.ny_with_ghosts):
                continue

            # Dummy model (you can replace this with fluid-based lookup later)
            electron_density = 1e18
            neutral_density = 1e20
            ionization_rate = 1e6

            p_ionize = 1 - np.exp(-ionization_rate * self.dt)
            if np.random.rand() > p_ionize:
                continue

            ion = np.zeros(self.pc.NUMQ)
            ion[self.pc.XCOMP] = x
            ion[self.pc.YCOMP] = y
            ion[self.pc.UCOMP] = self.particles[self.pc.UCOMP, n]
            ion[self.pc.VCOMP] = self.particles[self.pc.VCOMP, n]

            new_ions.append(ion)
            self.particles[:, n] = np.nan

        self.particles = self.particles[:, ~np.isnan(self.particles[self.pc.XCOMP])]
        return new_ions

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

    def check_particles(self):
        num_nans = np.sum(np.isnan(self.particles))
        num_out_of_bounds = np.sum(
            (self.particles[self.pc.XCOMP] < self.inp.grid_x[0]) |
            (self.particles[self.pc.XCOMP] > self.inp.grid_x[-1]) |
            (self.particles[self.pc.YCOMP] < self.inp.grid_y[0]) |
            (self.particles[self.pc.YCOMP] > self.inp.grid_y[-1])
        )
        print(f"[Check] Nans: {num_nans}, Out of bounds: {num_out_of_bounds}, Total: {self.particles.shape[1]}")

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

    def enforce_ppc(self, min_ppc=100, max_ppc=200):
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

            # split
            if count < min_ppc:
                needed = min_ppc - count
                for idx in indices[:needed]:
                    self.split_particle(idx)

            # merge
            elif count > max_ppc:
                to_merge = (count - max_ppc) // 2
                pairs = zip(indices[::2], indices[1::2])
                for idx1, idx2 in list(pairs)[:to_merge]:
                    if idx2 < self.particles.shape[1]:  # safety check
                        self.merge_particles(idx1, idx2)