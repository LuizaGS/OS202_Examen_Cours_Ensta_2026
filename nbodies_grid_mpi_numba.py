import sys
import time
import numpy as np
from mpi4py import MPI
from numba import njit, prange

# Constante gravitationnelle en unités [ly^3 / (M_sun * an^2)]
G = 1.560339e-13


# Version parallèle locale : cette fonction utilise Numba avec parallel=True.
@njit(parallel=True)
def update_stars_in_grid(cell_start_indices, body_indices,
                         cell_masses, cell_com_positions,
                         masses, positions, grid_min,
                         cell_size, n_cells):
    n_bodies = positions.shape[0]
    n_tot_cells = n_cells[0] * n_cells[1] * n_cells[2]

    cell_start_indices.fill(-1)
    cell_counts = np.zeros(n_tot_cells, dtype=np.int64)

    # Boucle conservée séquentielle : plusieurs corps peuvent écrire dans la même cellule.
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for d in range(3):
            if cell_idx[d] < 0:
                cell_idx[d] = 0
            elif cell_idx[d] >= n_cells[d]:
                cell_idx[d] = n_cells[d] - 1
        midx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        cell_counts[midx] += 1

    # Boucle conservée séquentielle : somme préfixe avec dépendance entre itérations.
    run = 0
    for i in range(n_tot_cells):
        cell_start_indices[i] = run
        run += cell_counts[i]
    cell_start_indices[n_tot_cells] = run

    # Boucle conservée séquentielle : dépendance sur current_counts.
    current_counts = np.zeros(n_tot_cells, dtype=np.int64)
    for ibody in range(n_bodies):
        cell_idx = np.floor((positions[ibody] - grid_min) / cell_size).astype(np.int64)
        for d in range(3):
            if cell_idx[d] < 0:
                cell_idx[d] = 0
            elif cell_idx[d] >= n_cells[d]:
                cell_idx[d] = n_cells[d] - 1
        midx = cell_idx[0] + cell_idx[1] * n_cells[0] + cell_idx[2] * n_cells[0] * n_cells[1]
        at = cell_start_indices[midx] + current_counts[midx]
        body_indices[at] = ibody
        current_counts[midx] += 1

    # Partie parallélisée : chaque cellule est traitée indépendamment.
    for i in prange(n_tot_cells):
        m_tot = 0.0
        com = np.zeros(3, dtype=np.float32)
        start = cell_start_indices[i]
        end = cell_start_indices[i + 1]
        for j in range(start, end):
            ibody = body_indices[j]
            m = masses[ibody]
            m_tot += m
            com += positions[ibody] * m
        if m_tot > 0.0:
            com /= m_tot
        cell_masses[i] = m_tot
        cell_com_positions[i] = com


# Version MPI : calcul de l'accélération uniquement pour les corps du processus courant.
@njit(parallel=True)
def compute_acceleration_for_indices(owned_indices,
                                     positions, masses,
                                     cell_start_indices, body_indices,
                                     cell_masses, cell_com_positions,
                                     grid_min, cell_size, n_cells):
    a = np.zeros((owned_indices.shape[0], 3), dtype=np.float32)

    # Partie parallélisée localement : chaque corps local calcule son accélération indépendamment.
    for ii in prange(owned_indices.shape[0]):
        ibody = owned_indices[ii]
        pos = positions[ibody]
        cell_idx = np.floor((pos - grid_min) / cell_size).astype(np.int64)
        for d in range(3):
            if cell_idx[d] < 0:
                cell_idx[d] = 0
            elif cell_idx[d] >= n_cells[d]:
                cell_idx[d] = n_cells[d] - 1

        for ix in range(n_cells[0]):
            for iy in range(n_cells[1]):
                for iz in range(n_cells[2]):
                    midx = ix + iy * n_cells[0] + iz * n_cells[0] * n_cells[1]
                    far_cell = (abs(ix - cell_idx[0]) > 2) or (abs(iy - cell_idx[1]) > 2) or (abs(iz - cell_idx[2]) > 2)
                    if far_cell:
                        cm = cell_com_positions[midx]
                        m = cell_masses[midx]
                        if m > 0.0:
                            dvec = cm - pos
                            dist = np.sqrt(dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2])
                            if dist > 1.0e-10:
                                inv_dist3 = 1.0 / (dist ** 3)
                                a[ii, :] += G * dvec[:] * inv_dist3 * m
                    else:
                        start = cell_start_indices[midx]
                        end = cell_start_indices[midx + 1]
                        for j in range(start, end):
                            jbody = body_indices[j]
                            if jbody != ibody:
                                dvec = positions[jbody] - pos
                                dist = np.sqrt(dvec[0] * dvec[0] + dvec[1] * dvec[1] + dvec[2] * dvec[2])
                                if dist > 1.0e-10:
                                    inv_dist3 = 1.0 / (dist ** 3)
                                    a[ii, :] += G * dvec[:] * inv_dist3 * masses[jbody]
    return a


def load_data(filename):
    masses = []
    positions = []
    velocities = []
    with open(filename, "r", encoding="utf-8") as f:
        for line in f:
            vals = line.split()
            masses.append(float(vals[0]))
            positions.append([float(vals[1]), float(vals[2]), float(vals[3])])
            velocities.append([float(vals[4]), float(vals[5]), float(vals[6])])
    return (
        np.array(masses, dtype=np.float32),
        np.array(positions, dtype=np.float32),
        np.array(velocities, dtype=np.float32),
    )


# Répartition MPI des corps selon leur position dans la grille.
def body_owner_ranks(positions, grid_min, cell_size, n_cells, n_ranks):
    cell_x = np.floor((positions[:, 0] - grid_min[0]) / cell_size[0]).astype(np.int64)
    cell_x = np.clip(cell_x, 0, n_cells[0] - 1)
    owners = (cell_x * n_ranks) // n_cells[0]
    owners = np.clip(owners, 0, n_ranks - 1)
    return owners


# Synchronisation MPI des positions et vitesses après les mises à jour locales.
def sync_updates_object(comm, owned_idx, owned_pos, owned_vel, global_pos, global_vel):
    packed = comm.allgather((owned_idx, owned_pos, owned_vel))
    for idx, pos, vel in packed:
        if idx.size > 0:
            global_pos[idx] = pos
            global_vel[idx] = vel


def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    filename = "data/galaxy_1000"
    dt = 0.001
    n_cells = np.array((20, 20, 1), dtype=np.int64)
    n_steps = 200

    if len(sys.argv) > 1:
        filename = sys.argv[1]
    if len(sys.argv) > 2:
        dt = float(sys.argv[2])
    if len(sys.argv) > 5:
        n_cells = np.array((int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5])), dtype=np.int64)
    if len(sys.argv) > 6:
        n_steps = int(sys.argv[6])

    if rank == 0:
        masses, positions, velocities = load_data(filename)
        grid_min = np.min(positions, axis=0) - 1.0e-6
        grid_max = np.max(positions, axis=0) + 1.0e-6
    else:
        masses = None
        positions = None
        velocities = None
        grid_min = None
        grid_max = None

    masses = comm.bcast(masses, root=0)
    positions = comm.bcast(positions, root=0)
    velocities = comm.bcast(velocities, root=0)
    grid_min = comm.bcast(grid_min, root=0)
    grid_max = comm.bcast(grid_max, root=0)

    cell_size = (grid_max - grid_min) / n_cells

    n_bodies = positions.shape[0]
    n_tot_cells = int(n_cells[0] * n_cells[1] * n_cells[2])
    cell_start_indices = np.full(n_tot_cells + 1, -1, dtype=np.int64)
    body_indices = np.empty(n_bodies, dtype=np.int64)
    cell_masses = np.zeros(n_tot_cells, dtype=np.float32)
    cell_com_positions = np.zeros((n_tot_cells, 3), dtype=np.float32)

    # Warm-up Numba avant la mesure pour éviter d'inclure le coût de compilation JIT dans le benchmark.
    update_stars_in_grid(cell_start_indices, body_indices,
                         cell_masses, cell_com_positions,
                         masses, positions, grid_min,
                         cell_size, n_cells)
    owners = body_owner_ranks(positions, grid_min, cell_size, n_cells, size)
    owned_indices = np.where(owners == rank)[0].astype(np.int64)
    _ = compute_acceleration_for_indices(owned_indices,
                                         positions, masses,
                                         cell_start_indices, body_indices,
                                         cell_masses, cell_com_positions,
                                         grid_min, cell_size, n_cells)
    comm.Barrier()

    t0 = MPI.Wtime()
    for _step in range(n_steps):
        update_stars_in_grid(cell_start_indices, body_indices,
                             cell_masses, cell_com_positions,
                             masses, positions, grid_min,
                             cell_size, n_cells)

        owners = body_owner_ranks(positions, grid_min, cell_size, n_cells, size)
        owned_indices = np.where(owners == rank)[0].astype(np.int64)

        # Les "cellules fantômes" sont implicitement gérées via la grille globale reconstruite.
        # Chaque rang met à jour uniquement ses étoiles locales (owned_indices).
        a_local = compute_acceleration_for_indices(owned_indices,
                                                   positions, masses,
                                                   cell_start_indices, body_indices,
                                                   cell_masses, cell_com_positions,
                                                   grid_min, cell_size, n_cells)

        if owned_indices.size > 0:
            positions[owned_indices] += velocities[owned_indices] * dt + 0.5 * a_local * dt * dt

        sync_updates_object(comm,
                            owned_indices,
                            positions[owned_indices],
                            velocities[owned_indices],
                            positions,
                            velocities)

        update_stars_in_grid(cell_start_indices, body_indices,
                             cell_masses, cell_com_positions,
                             masses, positions, grid_min,
                             cell_size, n_cells)

        a_new_local = compute_acceleration_for_indices(owned_indices,
                                                       positions, masses,
                                                       cell_start_indices, body_indices,
                                                       cell_masses, cell_com_positions,
                                                       grid_min, cell_size, n_cells)

        if owned_indices.size > 0:
            velocities[owned_indices] += 0.5 * (a_local + a_new_local) * dt

        sync_updates_object(comm,
                            owned_indices,
                            positions[owned_indices],
                            velocities[owned_indices],
                            positions,
                            velocities)

    t1 = MPI.Wtime()
    local_elapsed = t1 - t0
    # Le temps parallèle retenu est celui du processus le plus lent.
    elapsed = comm.reduce(local_elapsed, op=MPI.MAX, root=0)

    if rank == 0:
        avg_step = elapsed / n_steps
        print(f"[MPI] file={filename} n_bodies={n_bodies} n_steps={n_steps} procs={size}")
        print(f"[MPI] grid=({n_cells[0]}, {n_cells[1]}, {n_cells[2]}) dt={dt}")
        print(f"[MPI] total_time={elapsed:.6f} s avg_step={avg_step*1000:.3f} ms")


if __name__ == "__main__":
    main()