import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FuncAnimation, FFMpegWriter


class Cell:
    def __init__(self, cell_id, initial_center, parameters):
        self.id = cell_id
        self.parameters = parameters
        self.rho1 = parameters.get('CYTOPLASM_DENSITY', 1.2)
        self.rho2 = parameters.get('NUCLEUS_DENSITY', 1.4)

        n_nodes = parameters['N_NODES']
        angles = np.linspace(0, 2 * np.pi, n_nodes, endpoint=False)
        ring = np.column_stack((np.cos(angles), np.sin(angles)))
        center = np.asarray(initial_center, dtype=float)

        self.m = center + parameters['MEMBRANE_RADIUS'] * ring
        self.n = center + parameters['NUCLEUS_RADIUS'] * ring
        self.original_x_vectors = self.n - self.m
        self.original_y_vectors = self.m - np.roll(self.m, 1, axis=0)

        self.previous_forces_m = np.zeros_like(self.m)
        self.previous_forces_n = np.zeros_like(self.n)

        self.bonded = np.zeros(n_nodes, dtype=bool)
        self.restart = np.zeros(n_nodes, dtype=bool)

        self.center_of_mass = self.calculate_center_of_mass()
        self.original_hat_x_vectors = self.n - self.center_of_mass
        self.original_hat_y_vectors = self.n - np.roll(self.n, 1, axis=0)
        self.reach = 0.0

    def get_membrane_positions(self):
        return self.m

    def get_nucleus_positions(self):
        return self.n

    @staticmethod
    def _polygon_moments(points, weight):
        x, y = points[:, 0], points[:, 1]
        x_next, y_next = np.roll(x, -1), np.roll(y, -1)
        dx = x - x_next
        mx = weight / 6 * np.sum(dx * ((y + 2 * y_next) * x_next + (2 * y + y_next) * x))
        my = weight / 6 * np.sum(dx * (y**2 + y * y_next + y_next**2))
        area = weight / 2 * np.sum((y + y_next) * dx)
        return mx, my, area

    def calculate_center_of_mass(self):
        mx1, my1, a1 = self._polygon_moments(self.m, self.rho1)
        mx2, my2, a2 = self._polygon_moments(self.n, self.rho2 - self.rho1)
        area = a1 + a2

        if abs(area) < 1e-10:
            return self.m.mean(axis=0)
        return np.array([(mx1 + mx2) / area, (my1 + my2) / area])


class ChemicalSignalModule:
    def __init__(self, parameters):
        self.params = parameters
        self.sources = []
        self.seeded = set()

        self.n = parameters['GRID_RES']
        self.radius = parameters['DOMAIN_RADIUS']
        self.k = 2 * self.radius / (self.n - 1)

        xs = np.linspace(-self.radius, self.radius, self.n)
        gx, gy = np.meshgrid(xs, xs)
        dist = np.hypot(gx, gy)

        if parameters.get('CIRCULAR_DOMAIN', True):
            self.inside = dist <= self.radius
            wall_dist = self.radius - dist
        else:
            self.inside = np.ones((self.n, self.n), dtype=bool)
            wall_dist = np.minimum(np.minimum(gx + self.radius, self.radius - gx),
                                   np.minimum(gy + self.radius, self.radius - gy))

        width = parameters.get('EDGE_WIDTH', 0.0)
        drop = parameters.get('EDGE_BETA_DROP', 0.0)
        if width > 0 and drop > 0:
            ramp = np.clip((width - wall_dist) / width, 0.0, 1.0)
        else:
            ramp = np.zeros_like(wall_dist)
        self.beta_field = parameters['GL_R'] - drop * ramp

        if parameters.get('ISOTROPIC_LAPLACIAN', True):
            near_w, diag_w = 4.0, 1.0
            self.lap_scale = 1.0 / (6.0 * self.k**2)
            self.stability_limit = 0.375
        else:
            near_w, diag_w = 1.0, 0.0
            self.lap_scale = 1.0 / self.k**2
            self.stability_limit = 0.25

        padded = np.pad(self.inside, 1, constant_values=False)
        self.nearest = {}
        self.stencil = []
        for dj in (-1, 0, 1):
            for di in (-1, 0, 1):
                if dj == 0 and di == 0:
                    continue
                nbr = padded[1 + dj:1 + dj + self.n, 1 + di:1 + di + self.n]
                if dj == 0 or di == 0:
                    self.nearest[(dj, di)] = nbr
                    weight = near_w
                else:
                    weight = diag_w
                if weight > 0:
                    self.stencil.append((weight, nbr.astype(float), dj, di))

        self.C = np.zeros((self.n, self.n))
        self.W = np.zeros((self.n, self.n))
        self._gradient_cache = None
        self._dt_cache = None

        h = parameters['MAX_TIME'] / parameters['N_TIMESTEPS']
        lam = max(parameters['MU_C'], parameters['MU_W']) * h / self.k**2
        if lam >= self.stability_limit:
            raise ValueError(f"Unstable scheme: mu*h/k^2 = {lam:.3f} must be < {self.stability_limit}")

    def add_source(self, center, start_time=0.0):
        self.sources.append((np.asarray(center, dtype=float), float(start_time)))

    def seed_pending(self, t):
        for idx, (center, start_time) in enumerate(self.sources):
            if idx in self.seeded or t < start_time:
                continue
            self._seed(center)
            self.seeded.add(idx)
            self._gradient_cache = None
            self._dt_cache = None

    def _seed(self, center):
        rho = np.sqrt(-self.params['GL_ETA'] / self.params['GL_A'])
        i0 = int(np.floor((center[0] + self.radius) / self.k))
        j0 = int(np.floor((center[1] + self.radius) / self.k))
        sites = [(j0, i0), (j0, i0 + 1), (j0 + 1, i0 + 1), (j0 + 1, i0)]

        for q, (j, i) in enumerate(sites):
            phase = q * np.pi / 2
            self.C[j, i] = rho * np.cos(phase)
            self.W[j, i] = rho * np.sin(phase)

    def _laplacian(self, F):
        P = np.pad(F, 1)
        n = self.n
        out = np.zeros_like(F)
        for weight, nbr, dj, di in self.stencil:
            out += weight * nbr * (P[1 + dj:1 + dj + n, 1 + di:1 + di + n] - F)
        return out * self.lap_scale * self.inside

    def _compute_rates(self):
        p = self.params
        C, W = self.C, self.W
        mag = C**2 + W**2

        reaction_c = p['GL_ETA'] * C - self.beta_field * W + mag * (p['GL_A'] * C - p['GL_B'] * W)
        reaction_w = self.beta_field * C + p['GL_ETA'] * W + mag * (p['GL_A'] * W + p['GL_B'] * C)

        dC_dt = (reaction_c + p['MU_C'] * self._laplacian(C)) * self.inside
        dW_dt = (reaction_w + p['MU_W'] * self._laplacian(W)) * self.inside
        return dC_dt, dW_dt

    def step(self, h):
        dC_dt, dW_dt = self._compute_rates()
        self.C = self.C + h * dC_dt
        self.W = self.W + h * dW_dt
        self._gradient_cache = None
        self._dt_cache = None

    def _gradient_fields(self):
        if self._gradient_cache is None:
            C, n = self.C, self.n
            P = np.pad(C, 1)

            def neighbour(dj, di):
                return np.where(self.nearest[(dj, di)], P[1 + dj:1 + dj + n, 1 + di:1 + di + n], C)

            gx = (neighbour(0, 1) - neighbour(0, -1)) / (2 * self.k) * self.inside
            gy = (neighbour(1, 0) - neighbour(-1, 0)) / (2 * self.k) * self.inside
            self._gradient_cache = (gx, gy)
        return self._gradient_cache

    def _time_derivative_field(self):
        if self._dt_cache is None:
            dC_dt, _ = self._compute_rates()
            self._dt_cache = dC_dt
        return self._dt_cache

    def _sample(self, F, points):
        u = np.clip((points[:, 0] + self.radius) / self.k, 0, self.n - 1.000001)
        v = np.clip((points[:, 1] + self.radius) / self.k, 0, self.n - 1.000001)
        i = u.astype(int)
        j = v.astype(int)
        fu = u - i
        fv = v - j
        return ((1 - fv) * ((1 - fu) * F[j, i] + fu * F[j, i + 1])
                + fv * ((1 - fu) * F[j + 1, i] + fu * F[j + 1, i + 1]))

    def calculate_gradient(self, positions):
        gx, gy = self._gradient_fields()
        grad = np.column_stack((self._sample(gx, positions), self._sample(gy, positions)))

        if self.params.get('DIRECTIONAL_SENSING', True):
            dC_dt = self._sample(self._time_derivative_field(), positions)
            rising_phase = (np.sign(dC_dt) > 0)[:, None]
            grad = np.where(rising_phase, grad, 0.0)

        return grad

    def get_field(self):
        return np.ma.masked_where(~self.inside, self.C)


class CollisionModule:
    def __init__(self, parameters):
        self.params = parameters
        self.groups = []
        self.group_of = {}

    @staticmethod
    def points_in_fan(points, center, ring):
        v2 = ring - center
        v3 = np.roll(ring, -1, axis=0) - center
        p = points[:, None, :] - center

        det = v2[:, 0] * v3[:, 1] - v2[:, 1] * v3[:, 0]

        with np.errstate(divide='ignore', invalid='ignore'):
            u = (p[:, :, 0] * v3[None, :, 1] - p[:, :, 1] * v3[None, :, 0]) / det
            v = (v2[None, :, 0] * p[:, :, 1] - v2[None, :, 1] * p[:, :, 0]) / det
            inside = (u > 0) & (v > 0) & (u + v < 1)

        return inside.any(axis=1), inside.argmax(axis=1)

    def detect_and_handle(self, all_cells):
        centers = np.array([c.center_of_mass for c in all_cells])

        for cell, center in zip(all_cells, centers):
            cell.reach = np.max(np.linalg.norm(cell.m - center, axis=1))

        gaps = np.linalg.norm(centers[:, None, :] - centers[None, :, :], axis=2)
        reaches = np.array([c.reach for c in all_cells])

        close = np.triu(gaps < reaches[:, None] + reaches[None, :], k=1)

        for i, j in np.argwhere(close):
            self._check_one_way(all_cells[i], all_cells[j])
            self._check_one_way(all_cells[j], all_cells[i])

    def _check_one_way(self, cell_a, cell_b):
        offsets = cell_a.m - cell_b.center_of_mass
        dists_sq = np.einsum('ij,ij->i', offsets, offsets)

        near = np.flatnonzero(
            ~cell_a.bonded & (dists_sq < cell_b.reach**2)
        )

        if near.size == 0:
            return

        inside, triangle = self.points_in_fan(cell_a.m[near], cell_b.center_of_mass, cell_b.m)
        n_a = len(cell_a.m)
        n_b = len(cell_b.m)

        for node_a, tri in zip(near[inside], triangle[inside]):
            if cell_a.bonded[node_a]:
                continue

            candidate_b = self._nearest_free_node(cell_a.m[node_a], cell_b, tri)

            if candidate_b is None:
                continue

            self._bond(cell_a, node_a, cell_b, candidate_b)

            for offset_a in (-1, 1):
                nbr_a = (node_a + offset_a) % n_a

                if cell_a.bonded[nbr_a]:
                    continue

                for offset_b in (-1, 0, 1):
                    nbr_b = (candidate_b + offset_b) % n_b

                    if cell_b.bonded[nbr_b]:
                        continue

                    if np.linalg.norm(cell_a.m[nbr_a] - cell_b.m[nbr_b]) < 0.1:
                        self._bond(cell_a, nbr_a, cell_b, nbr_b)
                        break

    def _nearest_free_node(self, point, cell_b, tri):
        n_b = len(cell_b.m)
        window = self.params.get('BOND_SEARCH_WINDOW', 40)

        idx = (tri + np.arange(-window, window + 2)) % n_b
        idx = idx[~cell_b.bonded[idx]]

        if idx.size == 0:
            return None

        d = np.linalg.norm(cell_b.m[idx] - point, axis=1)
        return int(idx[np.argmin(d)])

    def _bond(self, cell_a, node_a, cell_b, node_b):
        key_a = (cell_a.id, node_a)
        key_b = (cell_b.id, node_b)

        if key_a in self.group_of or key_b in self.group_of:
            return
        contact = 0.5 * (cell_a.m[node_a] + cell_b.m[node_b])

        group = [(cell_a, node_a), (cell_b, node_b)]
        self.groups.append(group)
        self.group_of[key_a] = group
        self.group_of[key_b] = group

        cell_a.bonded[node_a] = True
        cell_b.bonded[node_b] = True

        for cell, node in group:
            cell.m[node] = contact.copy()
            cell.restart[node] = True


class Simulation:
    def __init__(self, parameters):
        self.params = parameters
        self.all_cells = []
        self.chemical_module = ChemicalSignalModule(parameters)
        self.collision_module = CollisionModule(parameters)
        self.time_step = 0
        self.h = parameters['MAX_TIME'] / parameters['N_TIMESTEPS']

        stiffness = max(
            8 * parameters['DELTA'],
            parameters['ALPHA_N'] + 4 * parameters.get('DELTA_N', parameters['DELTA']) + parameters['ALPHA'],
        )
        if self.h * stiffness > 0.8:
            raise ValueError(
                f"Unstable cell step: h*stiffness = {self.h * stiffness:.2f} must be < 0.8 "
                f"(use h < {0.8 / stiffness:.4f})"
            )

    def initialize_simulation(self):
        p = self.params
        signal_centers = p['SIGNAL_CENTERS']
        signal_start_times = p['SIGNAL_START_TIMES']
        outer_centers = p['OUTER_CENTERS']

        for center, start_time in zip(signal_centers, signal_start_times):
            self.chemical_module.add_source(center, start_time)

        self.chemical_module.seed_pending(0.0)

        for k, center in enumerate(outer_centers):
            self.all_cells.append(Cell(k, center, p))

        print(f"{p['CASE_NAME']} initialized: {len(outer_centers)} cells.")

    def run_one_step(self):
        current_time = self.time_step * self.h
        self.chemical_module.seed_pending(current_time)

        bonds_before = len(self.collision_module.groups)
        self.collision_module.detect_and_handle(self.all_cells)
        bonds_after = len(self.collision_module.groups)

        if bonds_after > bonds_before:
            print(f"t={current_time:.3f}: {bonds_after - bonds_before} new bonded pairs")

        if self.time_step == 0:
            self._step_heuns_method(current_time)
        else:
            self._step_ab2_method(current_time)

        self.chemical_module.step(self.h)
        self.time_step += 1

    def _calculate_forces_at_state(self, cell, m, n, center_of_mass, t):
        p = self.params

        effective_beta = (p['BETA'] * p['REACTION_DAMPING_K'] ** np.count_nonzero(cell.bonded))

        gradient = self.chemical_module.calculate_gradient(m)
        force_chemotaxis = effective_beta * gradient
        force_chemotaxis[cell.bonded] = 0.0

        force_internal = (p['ALPHA'] * (n - m - cell.original_x_vectors))

        spring_prev = m - np.roll(m, 1, axis=0)
        spring_next = np.roll(m, -1, axis=0) - m
        y_next = np.roll(cell.original_y_vectors, -1, axis=0)

        force_surface = p['DELTA'] * (cell.original_y_vectors - spring_prev - y_next + spring_next)

        total_m = (force_chemotaxis + force_internal + force_surface)
        force_nuc_center = p['ALPHA_N'] * (center_of_mass - n + cell.original_hat_x_vectors)
        force_reaction = -force_internal

        nuc_spring_prev = n - np.roll(n, 1, axis=0)
        nuc_spring_next = np.roll(n, -1, axis=0) - n
        hat_y_next = np.roll(cell.original_hat_y_vectors, -1, axis=0)

        force_nuc_surface = p.get('DELTA_N', p['DELTA']) * (cell.original_hat_y_vectors - nuc_spring_prev - hat_y_next + nuc_spring_next)
        total_n = (force_nuc_center + force_reaction + force_nuc_surface)

        return total_m, total_n

    def _calculate_current_forces(self, cell, t):
        return self._calculate_forces_at_state(cell, cell.m, cell.n, cell.center_of_mass, t)

    def _apply_adhesion_forces(self, forces_m):
        for group in self.collision_module.groups:
            total_force = np.zeros(2)

            for cell, node in group:
                total_force += forces_m[cell.id, node]

            for cell, node in group:
                forces_m[cell.id, node] = total_force

    def _heun_second_stage(self, k1_m, k1_n, t):
        h = self.h

        predicted_states = []
        for cell in self.all_cells:
            pred_m = cell.m + h * k1_m[cell.id]
            pred_n = cell.n + h * k1_n[cell.id]
            cell_temp = Cell(cell.id, [0, 0], cell.parameters)
            cell_temp.m = pred_m.copy()
            cell_temp.n = pred_n.copy()
            cell_temp.rho1 = cell.rho1
            cell_temp.rho2 = cell.rho2
            pred_com = cell_temp.calculate_center_of_mass()
            predicted_states.append((pred_m, pred_n, pred_com))

        k2_forces = [
            self._calculate_forces_at_state(cell, pred_m, pred_n, pred_com, t + h)
            for cell, (pred_m, pred_n, pred_com) in zip(self.all_cells, predicted_states)
        ]
        k2_m = np.stack([f[0] for f in k2_forces])
        k2_n = np.stack([f[1] for f in k2_forces])
        self._apply_adhesion_forces(k2_m)

        return k2_m, k2_n

    def _step_heuns_method(self, t):
        h = self.h

        k1_forces = [self._calculate_current_forces(cell, t) for cell in self.all_cells]
        k1_m = np.stack([f[0] for f in k1_forces])
        k1_n = np.stack([f[1] for f in k1_forces])
        self._apply_adhesion_forces(k1_m)

        k2_m, k2_n = self._heun_second_stage(k1_m, k1_n, t)

        for cell in self.all_cells:
            f1_m, f2_m = k1_m[cell.id], k2_m[cell.id]
            f1_n, f2_n = k1_n[cell.id], k2_n[cell.id]

            cell.m += 0.5 * h * (f1_m + f2_m)
            cell.n += 0.5 * h * (f1_n + f2_n)

            cell.previous_forces_m = f1_m.copy()
            cell.previous_forces_n = f1_n.copy()

            cell.restart[:] = False
            cell.center_of_mass = cell.calculate_center_of_mass()

    def _step_ab2_method(self, t):
        h = self.h

        current_forces = [self._calculate_current_forces(cell, t) for cell in self.all_cells]
        fn_m = np.stack([f[0] for f in current_forces])
        fn_n = np.stack([f[1] for f in current_forces])
        self._apply_adhesion_forces(fn_m)

        if any(cell.restart.any() for cell in self.all_cells):
            k2_m, k2_n = self._heun_second_stage(fn_m, fn_n, t)
        else:
            k2_m = k2_n = None

        for cell in self.all_cells:
            f_curr_m = fn_m[cell.id]
            f_curr_n = fn_n[cell.id]
            f_prev_m = cell.previous_forces_m
            f_prev_n = cell.previous_forces_n

            new_m = cell.m + (h / 2.0) * (3.0 * f_curr_m - f_prev_m)
            new_n = cell.n + (h / 2.0) * (3.0 * f_curr_n - f_prev_n)

            if cell.restart.any():
                r = cell.restart
                new_m[r] = cell.m[r] + (h / 2.0) * (f_curr_m[r] + k2_m[cell.id][r])
                new_n[r] = cell.n[r] + (h / 2.0) * (f_curr_n[r] + k2_n[cell.id][r])

            cell.restart[:] = False

            cell.previous_forces_m = f_curr_m.copy()
            cell.previous_forces_n = f_curr_n.copy()

            cell.m = new_m
            cell.n = new_n
            cell.center_of_mass = cell.calculate_center_of_mass()



def default_params(max_time=50.0, frames=500, h=0.002):
    speed_factor = max(1, round(max_time / h / frames))
    n_timesteps = frames * speed_factor
    domain_radius = 10.0
    n_outer_cells = 8
    outer_ring_radius = 6.0

    ring_angles = np.linspace(0, 2 * np.pi, n_outer_cells, endpoint=False)
    outer_centers = [(outer_ring_radius * np.cos(a), outer_ring_radius * np.sin(a)) for a in ring_angles]

    return {
        'CASE_NAME': 'Case 1 (Spiral Wave)',
        'N_NODES': 300,
        'MEMBRANE_RADIUS': 0.7,
        'NUCLEUS_RADIUS': 0.175,
        'ALPHA': 0.5,
        'DELTA': 30,
        'ALPHA_N': 100,
        'BETA': 2.5,
        'REACTION_DAMPING_K': 0.2 ** (1 / 60.0),
        'MAX_TIME': max_time,
        'N_TIMESTEPS': n_timesteps,
        'ANIMATION_SPEED_FACTOR': speed_factor,
        'FRAMES': frames,

        'DOMAIN_RADIUS': domain_radius,
        'GRID_RES': 200,
        'CIRCULAR_DOMAIN': True,
        'ISOTROPIC_LAPLACIAN': True,
        'GL_ETA': 2.0,
        'GL_R': 1.00,
        'GL_A': -2.0,
        'GL_B': 2.0,
        'MU_C': 0.1,
        'MU_W': 0.1,
        'EDGE_WIDTH': 0.0,
        'EDGE_BETA_DROP': 0.0,
        'DIRECTIONAL_SENSING': True,

        'SIGNAL_CENTERS': [(0.0, 0.0)],
        'SIGNAL_START_TIMES': [0.0],
        'OUTER_CENTERS': outer_centers,

        'RANDOM_SEED': None,
    }


def render(sim, sim_params, filename):
    domain_radius = sim_params['DOMAIN_RADIUS']
    frames = sim_params['FRAMES']
    half = domain_radius + sim.chemical_module.k / 2

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_facecolor('#001122')
    ax.set_aspect('equal', 'box')
    ax.set_xlim(-domain_radius, domain_radius)
    ax.set_ylim(-domain_radius, domain_radius)

    spiral_cmap = LinearSegmentedColormap.from_list('dilao', ['#1010ff', '#3a0a50', '#ff1010'])
    limit_cycle_radius = np.sqrt(-sim_params['GL_ETA'] / sim_params['GL_A'])

    gradient_image = ax.imshow(
        sim.chemical_module.get_field(),
        extent=[-half, half, -half, half],
        origin='lower',
        cmap=spiral_cmap,
        vmin=-limit_cycle_radius,
        vmax=limit_cycle_radius,
        zorder=1,
        alpha=0.8
    )
    cell_plots = []

    for cell in sim.all_cells:
        mem_pos = cell.get_membrane_positions()
        nuc_pos = cell.get_nucleus_positions()

        mem_fill = ax.fill(mem_pos[:, 0], mem_pos[:, 1], color='white', alpha=1, zorder=2)[0]
        nuc_fill = ax.fill(nuc_pos[:, 0], nuc_pos[:, 1], color='white', alpha=1, zorder=3)[0]
        mem_plot, = ax.plot(mem_pos[:, 0], mem_pos[:, 1], '-', color='black', zorder=4)
        nuc_plot, = ax.plot(nuc_pos[:, 0], nuc_pos[:, 1], '-', color='black', zorder=4)

        cell_plots.append({
            'mem': mem_plot,
            'nuc': nuc_plot,
            'mem_fill': mem_fill,
            'nuc_fill': nuc_fill
        })

    time_text = ax.text(0.02, 0.95, '', transform=ax.transAxes, color='white')

    def update_animation(frame):
        target_step = frame * sim_params['ANIMATION_SPEED_FACTOR']
        while sim.time_step < target_step:
            sim.run_one_step()

        current_time = sim.time_step * sim.h
        gradient_image.set_data(sim.chemical_module.get_field())

        for i, cell in enumerate(sim.all_cells):
            mem_pos = cell.get_membrane_positions()
            nuc_pos = cell.get_nucleus_positions()

            cell_plots[i]['mem'].set_data(
                np.append(mem_pos[:, 0], mem_pos[0, 0]),
                np.append(mem_pos[:, 1], mem_pos[0, 1])
            )

            cell_plots[i]['nuc'].set_data(
                np.append(nuc_pos[:, 0], nuc_pos[0, 0]),
                np.append(nuc_pos[:, 1], nuc_pos[0, 1])
            )

            cell_plots[i]['mem_fill'].set_xy(mem_pos)
            cell_plots[i]['nuc_fill'].set_xy(nuc_pos)

        time_text.set_text(f'Time: {min(current_time, sim_params["MAX_TIME"]):.2f} hrs')

        return (
            [p['mem'] for p in cell_plots] +
            [p['nuc'] for p in cell_plots] +
            [p['mem_fill'] for p in cell_plots] +
            [p['nuc_fill'] for p in cell_plots] +
            [time_text, gradient_image]
        )

    ani = FuncAnimation(fig, update_animation, frames=frames + 1, interval=50, blit=True, repeat=False)
    writer = FFMpegWriter(fps=20, metadata=dict(artist='Me'), bitrate=1800)
    ani.save(filename, writer=writer)
    print("Done!")


if __name__ == "__main__":
    sim_params = default_params()
    sim = Simulation(sim_params)
    sim.initialize_simulation()
    render(sim, sim_params, "spiral1_1 (2).mp4")