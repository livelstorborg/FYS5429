import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from src.pinn import train_pinn
from src.pde import fem_solve, create_grid, make_c_fn
from src.plotting import plot_2d_snapshots


# ==================================================================
#                         Parameters
# ==================================================================
T      = 1.0
c0     = 1.0
Nx_fem = 100
Ny_fem = 100
Nx_plt = 50
Ny_plt = 50

snap_times = [0.25, 0.5, 1.0 / np.sqrt(2), T]

FIG_DIR  = Path(__file__).parent.parent / "figs" / "2d_var"
DATA_DIR = Path(__file__).parent.parent / "data" / "2d_var"
FIG_DIR.mkdir(parents=True, exist_ok=True)
DATA_DIR.mkdir(parents=True, exist_ok=True)


# ==================================================================
#               Metamaterial / Variable-c Setup
# ==================================================================
eps   = 0.4
delta = 0.03   # ~3× dx at Nx=100; sharp enough to localise the particle, resolvable by FEM
Omega = 2 * np.pi * 1.0
s_fn  = lambda t: eps**2 * jnp.sin(Omega * t)

c_max        = c0 / jnp.sqrt(0.1)
z_particles  = [(0.5, 0.5)]
c_fn_2d      = make_c_fn(z_particles, c0=c0, eps=eps, delta=delta, s_fn=s_fn)


# ==================================================================
#                     FEM solve (fine grid)
# ==================================================================
x_fem, y_fem, t_fem, dx_fem, dy_fem, dt_fem = create_grid(
    Nx=Nx_fem, Ny=Ny_fem, T=T, c=c_max, cfl=0.8, dim=2
)

X_fem, Y_fem = jnp.meshgrid(x_fem, y_fem, indexing="ij")
u0_2d = jnp.sin(jnp.pi * X_fem) * jnp.sin(jnp.pi * Y_fem)

u_fem = fem_solve(x_fem, t_fem, dx_fem, dt_fem, c=c_fn_2d, u0=u0_2d, y=y_fem, dy=dy_fem, dim=2)


# ==================================================================
#               PINN: adam (8000 steps) → lbfgs (5000 steps)
# ==================================================================
# Build interface points: ring of 64 points on the particle boundary
_theta = np.linspace(0, 2 * np.pi, 64, endpoint=False)
interface_pts = np.stack([
    np.clip(z_particles[0][0] + eps * np.cos(_theta), 0.0, 1.0),
    np.clip(z_particles[0][1] + eps * np.sin(_theta), 0.0, 1.0),
], axis=1)  # (64, 2)

_, model, losses = train_pinn(
    (128, 128, 128, 128),
    dim=2,
    activation=jnn.gelu,
    adam_steps=8000,
    lbfgs_steps=5000,
    N_int=5000,
    N_ic=500,
    T=T,
    c=c_fn_2d,
    lambda_ic=100.0,
    lr=1e-3,
    seed=0,
    log_every=500,
    fourier_freqs=[1, 2, 4, 8],
    interface_points=interface_pts,
    bias_frac=0.3,
    bias_std=3 * delta,
)


# ==================================================================
#          Evaluate both solutions on a shared plot grid
# ==================================================================
x_plt = jnp.linspace(0.0, 1.0, Nx_plt)
y_plt = jnp.linspace(0.0, 1.0, Ny_plt)

# Subsample FEM onto plot grid
ix = np.round(np.linspace(0, len(x_fem) - 1, Nx_plt)).astype(int)
iy = np.round(np.linspace(0, len(y_fem) - 1, Ny_plt)).astype(int)

pinn_snaps = []
fem_snaps  = []

for t_val in snap_times:
    # PINN prediction
    X2, Y2 = jnp.meshgrid(x_plt, y_plt, indexing="ij")
    T2  = jnp.full_like(X2, t_val)
    xyt = jnp.stack([X2.ravel(), Y2.ravel(), T2.ravel()], axis=1)
    u_pred = model(xyt).squeeze().reshape(Nx_plt, Ny_plt)
    pinn_snaps.append(np.array(u_pred))

    # FEM subsampled
    it = int(jnp.argmin(jnp.abs(t_fem - t_val)))
    fem_snaps.append(np.array(u_fem)[it][np.ix_(ix, iy)])

u_pinn_arr = np.stack(pinn_snaps, axis=0)   # (4, Nx_plt, Ny_plt)
u_fem_arr  = np.stack(fem_snaps,  axis=0)
error_arr  = np.abs(u_pinn_arr - u_fem_arr)

snap_idx = [0, 1, 2, 3]


# ==================================================================
#                          Save data
# ==================================================================
np.savez(
    DATA_DIR / "snapshots.npz",
    u_pinn    = u_pinn_arr,
    u_fem     = u_fem_arr,
    error     = error_arr,
    x_plt     = np.array(x_plt),
    y_plt     = np.array(y_plt),
    snap_times= np.array(snap_times),
)
np.savetxt(DATA_DIR / "losses.csv", np.array(losses), delimiter=",", header="loss")
print(f"Data saved to {DATA_DIR}")


# ==================================================================
#                        Snapshot plots
# ==================================================================
plot_2d_snapshots(
    u_fem_arr, t_indices=snap_idx, t_labels=snap_times,
    title=r"FEM Solution — 2D Variable-$c$ Wave",
    cmap="inferno", show=True, savefig=True,
    filepath=str(FIG_DIR / "fem_solution_2d_var.pdf"),
)

plot_2d_snapshots(
    u_pinn_arr, t_indices=snap_idx, t_labels=snap_times,
    title=r"PINN Solution — 2D Variable-$c$ Wave",
    cmap="inferno", show=True, savefig=True,
    filepath=str(FIG_DIR / "pinn_solution_2d_var.pdf"),
)

plot_2d_snapshots(
    error_arr, t_indices=snap_idx, t_labels=snap_times,
    title=r"Absolute Error $|$PINN $-$ FEM$|$ — 2D Variable-$c$ Wave",
    cmap="inferno", show=True, savefig=True,
    filepath=str(FIG_DIR / "error_2d_var.pdf"),
)
