import sys
import time
import numpy as np
from numba import njit, prange
from mpi4py import MPI
import visualizer3d

# Constante gravitationnelle en unités [ly^3 / (M_sun * an^2)]
G = 1.560339e-13


def generate_star_color(mass: float) -> tuple[int, int, int]:
    if mass > 5.0:
        return (150, 180, 255)
    if mass > 2.0:
        return (255, 255, 255)
    if mass >= 1.0:
        return (255, 255, 200)
    return (255, 150, 100)


@njit(parallel=True)
def update_stars_in_grid(cell_start_indices, body_indices,
                         cell_masses, cell_com_positions,
                         masses, positions, grid_min,
                         cell_size, n_cells):
    n_bodies = positions.shape[0]
    cell_start_indices.fill(-1)

    cell_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        cell_counts[morse_idx] += 1

    running_index = 0
    for i in range(len(cell_counts)):
        cell_start_indices[i] = running_index
        running_index += cell_counts[i]
    cell_start_indices[len(cell_counts)] = running_index

    current_counts = np.zeros(shape=(np.prod(n_cells),), dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0
        morse_idx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        index_in_cell = cell_start_indices[morse_idx] + current_counts[morse_idx]
        body_indices[index_in_cell] = ibody
        current_counts[morse_idx] += 1

    for i in prange(len(cell_counts)):
        cell_mass = 0.0
        com_position = np.zeros(3, dtype=np.float32)
        start_idx = cell_start_indices[i]
        end_idx = cell_start_indices[i + 1]
        for j in range(start_idx, end_idx):
            ibody = body_indices[j]
            m = masses[ibody]
            cell_mass += m
            com_position += positions[ibody] * m
        if cell_mass > 0.0:
            com_position /= cell_mass
        cell_masses[i] = cell_mass
        cell_com_positions[i] = com_position


@njit(parallel=True)
def compute_acceleration(positions, masses,
                         cell_start_indices, body_indices,
                         cell_masses, cell_com_positions,
                         grid_min, cell_size, n_cells):
    n_bodies = positions.shape[0]
    a = np.zeros_like(positions)

    for ibody in prange(n_bodies):
        pos = positions[ibody]
        cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)
        for i in range(3):
            if cell_idx[i] >= n_cells[i]:
                cell_idx[i] = n_cells[i] - 1
            elif cell_idx[i] < 0:
                cell_idx[i] = 0

        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    morse_idx = ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]
                    if (abs(ix - cell_idx[0]) > 2) or (abs(iy - cell_idx[1]) > 2) or (abs(iz - cell_idx[2]) > 2):
                        cell_com = cell_com_positions[morse_idx]
                        cell_mass = cell_masses[morse_idx]
                        if cell_mass > 0.0:
                            direction = cell_com - pos
                            distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
                            if distance > 1.0e-10:
                                inv_dist3 = 1.0 / (distance ** 3)
                                a[ibody, :] += G * direction[:] * inv_dist3 * cell_mass
                    else:
                        start_idx = cell_start_indices[morse_idx]
                        end_idx = cell_start_indices[morse_idx + 1]
                        for j in range(start_idx, end_idx):
                            jbody = body_indices[j]
                            if jbody != ibody:
                                direction = positions[jbody] - pos
                                distance = np.sqrt(direction[0] ** 2 + direction[1] ** 2 + direction[2] ** 2)
                                if distance > 1.0e-10:
                                    inv_dist3 = 1.0 / (distance ** 3)
                                    a[ibody, :] += G * direction[:] * inv_dist3 * masses[jbody]
    return a


class SpatialGrid:
    def __init__(self, positions: np.ndarray, nb_cells_per_dim: tuple[int, int, int]):
        self.min_bounds = np.min(positions, axis=0) - 1.0e-6
        self.max_bounds = np.max(positions, axis=0) + 1.0e-6
        self.n_cells = np.array(nb_cells_per_dim)
        self.cell_size = (self.max_bounds - self.min_bounds) / self.n_cells
        self.cell_start_indices = np.full(np.prod(self.n_cells) + 1, -1, dtype=np.int64)
        self.body_indices = np.empty(shape=(positions.shape[0],), dtype=np.int64)
        self.cell_masses = np.zeros(shape=(np.prod(self.n_cells),), dtype=np.float32)
        self.cell_com_positions = np.zeros(shape=(np.prod(self.n_cells), 3), dtype=np.float32)

    def update(self, positions: np.ndarray, masses: np.ndarray):
        update_stars_in_grid(self.cell_start_indices, self.body_indices,
                             self.cell_masses, self.cell_com_positions,
                             masses, positions, self.min_bounds,
                             self.cell_size, self.n_cells)


class NBodySystem:
    def __init__(self, positions, velocities, masses, ncells_per_dir=(20, 20, 1)):
        self.positions = positions.copy()
        self.velocities = velocities.copy()
        self.masses = masses.copy()
        self.grid = SpatialGrid(self.positions, ncells_per_dir)
        self.grid.update(self.positions, self.masses)

        self.total_compute_time = 0.0
        self.n_steps = 0

    def update_positions(self, dt):
        t0 = time.perf_counter()

        a = compute_acceleration(self.positions, self.masses,
                                 self.grid.cell_start_indices, self.grid.body_indices,
                                 self.grid.cell_masses, self.grid.cell_com_positions,
                                 self.grid.min_bounds, self.grid.cell_size, self.grid.n_cells)

        self.positions += self.velocities * dt + 0.5 * a * dt * dt
        self.grid.update(self.positions, self.masses)

        a_new = compute_acceleration(self.positions, self.masses,
                                     self.grid.cell_start_indices, self.grid.body_indices,
                                     self.grid.cell_masses, self.grid.cell_com_positions,
                                     self.grid.min_bounds, self.grid.cell_size, self.grid.n_cells)

        self.velocities += 0.5 * (a + a_new) * dt

        t1 = time.perf_counter()
        self.total_compute_time += (t1 - t0)
        self.n_steps += 1


def load_data(filename):
    positions = []
    velocities = []
    masses = []
    box = np.array([[-1.0e-6, -1.0e-6, -1.0e-6], [1.0e-6, 1.0e-6, 1.0e-6]], dtype=np.float64)

    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            data = line.split()
            m = float(data[0])
            p = [float(data[1]), float(data[2]), float(data[3])]
            v = [float(data[4]), float(data[5]), float(data[6])]
            masses.append(m)
            positions.append(p)
            velocities.append(v)
            for i in range(3):
                box[0][i] = min(box[0][i], p[i] - 1.0e-6)
                box[1][i] = max(box[1][i], p[i] + 1.0e-6)

    masses = np.array(masses, dtype=np.float32)
    positions = np.array(positions, dtype=np.float32)
    velocities = np.array(velocities, dtype=np.float32)
    colors = np.array([generate_star_color(m) for m in masses], dtype=np.float32)
    max_mass = float(np.max(masses))
    return masses, positions, velocities, colors, max_mass, box


def run_rank0_display(comm, positions0, masses, colors, max_mass, box, dt):
    intensity = np.clip(masses / max_mass, 0.5, 1.0)
    bounds = [[box[0][0], box[1][0]], [box[0][1], box[1][1]], [box[0][2], box[1][2]]]
    visu = visualizer3d.Visualizer3D(positions0, colors, intensity, bounds)

    def updater(_dt):
        # Rank 0 receives the new positions from rank 1 at each frame.
        return comm.recv(source=1, tag=11)

    visu.run(updater=updater, dt=dt)

    # Signal rank 1 to stop when display window is closed.
    comm.send("stop", dest=1, tag=99)


def run_rank1_compute(comm, positions, velocities, masses, n_cells_per_dir, dt):
    system = NBodySystem(positions, velocities, masses, ncells_per_dir=n_cells_per_dir)

    # Warm-up for Numba JIT
    system.update_positions(dt)

    while True:
        system.update_positions(dt)
        # Send full positions to rank 0 for rendering.
        comm.send(system.positions.copy(), dest=0, tag=11)

        if comm.iprobe(source=0, tag=99):
            _ = comm.recv(source=0, tag=99)
            break

    if system.n_steps > 0:
        avg = system.total_compute_time / system.n_steps
        print(f"[rank 1] avg compute time per step: {avg * 1000:.3f} ms over {system.n_steps} steps")


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    if size != 2:
        if rank == 0:
            print("Use exactly 2 MPI processes: rank 0 (display), rank 1 (compute).")
        return

    filename = "data/galaxy_1000"
    dt = 0.001
    n_cells_per_dir = (20, 20, 1)

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        dt = float(sys.argv[2])
    if len(sys.argv) > 5:
        n_cells_per_dir = (int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]))

    if rank == 0:
        masses, positions, velocities, colors, max_mass, box = load_data(filename)
    else:
        masses = None
        positions = None
        velocities = None
        colors = None
        max_mass = None
        box = None

    masses = comm.bcast(masses, root=0)
    positions = comm.bcast(positions, root=0)
    velocities = comm.bcast(velocities, root=0)
    colors = comm.bcast(colors, root=0)
    max_mass = comm.bcast(max_mass, root=0)
    box = comm.bcast(box, root=0)

    if rank == 0:
        print(f"[rank 0] display process ready for {filename}")
        run_rank0_display(comm, positions, masses, colors, max_mass, box, dt)
    else:
        print("[rank 1] compute process running")
        run_rank1_compute(comm, positions, velocities, masses, n_cells_per_dir, dt)


if __name__ == "__main__":
    main()
