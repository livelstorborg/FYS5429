import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.nn as jnn
import numpy as np

from src.pde import fd_solve, fem_solve, u_exact, create_grid
from src.pinn import train_pinn
from src.experiment import absolute_error, relative_error
from src.plotting import (
    plot_2d_snapshots,
)

# Parameters
Nx = 100
Ny = 100
T = 1.0
c = 1.0
cfl = 1.0

# time points to evaluate
t1_eval = 0.25
t2_eval = 0.5
t3_eval = 1.0 / np.sqrt(2)
t4_eval = T


x, y, t, dx, dy, dt = create_grid(Nx=Nx, Ny=Ny, T=T, c=c, cfl=cfl, dim=2)

u_fd = fd_solve(x, t, dx, dt, y=y, dy=dy, c=c, dim=2)
u_fem = fem_solve(x, t, dx, dt, y=y, dy=dy, c=c, dim=2)
u_exact = u_exact(x, t, y=y, c=c, dim=2)

idx_t1 = jnp.argmin(jnp.abs(t - t1_eval))
idx_t2 = jnp.argmin(jnp.abs(t - t2_eval))
idx_t3 = jnp.argmin(jnp.abs(t - t3_eval))
idx_t4 = jnp.argmin(jnp.abs(t - t4_eval))

t_indices = [idx_t1, idx_t2, idx_t3, idx_t4]
t_labels = [t1_eval, t2_eval, t3_eval, t4_eval]


fd_error = np.abs(np.array(u_fd) - np.array(u_exact))
fem_error = np.abs(np.array(u_fem) - np.array(u_exact))

plot_2d_snapshots(
    fd_error, t_indices, t_labels,
    title=r"$\mathbf{2D}$ Wave Equation — Absolute Error (FD)",
    cmap="inferno",
    savefig=True,
    filepath=str(Path(__file__).parent.parent / "figs" / "schemes" / "fd_error_heatmaps_2d_const.pdf"),
)
plot_2d_snapshots(
    fem_error, t_indices, t_labels,
    title=r"$\mathbf{2D}$ Wave Equation — Absolute Error (FEM)",
    cmap="inferno",
    savefig=True,
    filepath=str(Path(__file__).parent.parent / "figs" / "schemes" / "fem_error_heatmaps_2d_const.pdf"),
)
