import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import matplotlib.pyplot as plt

from src.pde import fd_solve, fem_solve, u_exact, create_grid
from src.pinn import train_pinn
from src.experiment import absolute_error, relative_error
from src.plotting import (
    plot_solution_at_t,
    plot_error_at_t,
    plot_3d_surface,
    subplot_3d_surfaces,
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


def plot_error_heatmaps_2d(u_scheme, u_exact, t_indices, t_labels, label, savepath):
    n = len(t_indices)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 5), constrained_layout=True)
    fig.suptitle(
        rf"$\mathbf{{2D}}$ Wave Equation — Absolute Error ({label})",
        fontsize=16,
        fontweight="bold",
        y=0.92,
    )

    for col, (idx, t_val) in enumerate(zip(t_indices, t_labels)):
        ax = axes[col]
        err = np.abs(np.array(u_scheme[idx]) - np.array(u_exact[idx]))
        im = ax.imshow(
            err.T, origin="lower", extent=[0, 1, 0, 1], cmap="viridis", aspect="equal"
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_title(rf"$\mathbf{{t={t_val:.2f}}}$", fontsize=11)

    plt.savefig(savepath, dpi=300, bbox_inches="tight")
    plt.show()


plot_error_heatmaps_2d(
    u_fd, u_exact, t_indices, t_labels, "FD", "figs/fd_error_heatmaps_2d_const.pdf"
)
plot_error_heatmaps_2d(
    u_fem, u_exact, t_indices, t_labels, "FEM", "figs/fem_error_heatmaps_2d_const.pdf"
)
