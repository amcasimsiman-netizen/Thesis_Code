import numpy as np

from spiral1 import ChemicalSignalModule, Simulation, default_params, render


class TargetSignalModule(ChemicalSignalModule):
    def _seed(self, center):
        rho = np.sqrt(-self.params['GL_ETA'] / self.params['GL_A'])
        xs = np.linspace(-self.radius, self.radius, self.n)
        gx, gy = np.meshgrid(xs, xs)
        disc = np.hypot(gx - center[0], gy - center[1]) <= self.params['SEED_RADIUS']

        self.C[disc] = rho
        self.W[disc] = 0.0


if __name__ == "__main__":
    sim_params = default_params()
    sim_params['CASE_NAME'] = 'Case 2 (Target Wave)'
    sim_params['SEED_RADIUS'] = 0.2

    sim = Simulation(sim_params)
    sim.chemical_module = TargetSignalModule(sim_params)
    sim.initialize_simulation()
    render(sim, sim_params, "target1_1.mp4")