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

t0_eval = 0
t1_eval = 0.25
t2_eval = 0.5
t3_eval = 1. / np.sqrt(2) 
t4_eval = T





x2, y2, t2, dx2, dy2, dt2 = create_grid(Nx=10, Ny=10, T=T, c=c, cfl=cfl, dim=2)

u_fd_2d  = fd_solve(x2, t2, dx2, dt2, y=y2, dy=dy2, c=c, dim=2)
u_fem_2d = fem_solve(x2, t2, dx2, dt2, y=y2, dy=dy2, c=c, dim=2)
u_ex_2d  = u_exact(x2, t2, y=y2, c=c, dim=2)

idx_t1_2d = jnp.argmin(jnp.abs(t2 - t1_eval))
idx_t2_2d = jnp.argmin(jnp.abs(t2 - t2_eval))
idx_t3_2d = jnp.argmin(jnp.abs(t2 - t3_eval))

t_indices = [idx_t1_2d, idx_t2_2d, idx_t3_2d]
t_labels  = [t1_eval, t2_eval, t3_eval]

# --- Solution heatmaps at each time point ---
fig, axes = plt.subplots(3, 3, figsize=(15, 12))
fig.suptitle(r"$\mathbf{2D}$ Wave Equation — Solutions", fontsize=18, fontweight="bold")

for col, (idx, t_val) in enumerate(zip(t_indices, t_labels)):
    data = [u_ex_2d[idx], u_fd_2d[idx], u_fem_2d[idx]]
    row_titles = ["Analytical", "FD", "FEM"]
    vmin = float(jnp.min(u_ex_2d[idx]))
    vmax = float(jnp.max(u_ex_2d[idx]))

    for row, (U, title) in enumerate(zip(data, row_titles)):
        ax = axes[row, col]
        im = ax.imshow(np.array(U).T, origin="lower", extent=[0,1,0,1],
                       vmin=vmin, vmax=vmax, cmap="viridis", aspect="equal")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(rf"{title}, $t={t_val:.2f}$", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("../figs/solution_2D.pdf", dpi=300, bbox_inches="tight")
plt.show()

# --- Error heatmaps at each time point ---
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle(r"$\mathbf{2D}$ Wave Equation — Absolute Error", fontsize=18, fontweight="bold")

for col, (idx, t_val) in enumerate(zip(t_indices, t_labels)):
    schemes = [
        (u_fd_2d[idx],  "FD"),
        (u_fem_2d[idx], "FEM"),
    ]
    for row, (U, label) in enumerate(schemes):
        ax = axes[row, col]
        err = np.abs(np.array(U) - np.array(u_ex_2d[idx]))
        im = ax.imshow(err.T, origin="lower", extent=[0,1,0,1],
                       cmap="plasma", aspect="equal")
        plt.colorbar(im, ax=ax)
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)
        ax.set_title(rf"|{label} - Analytical|, $t={t_val:.2f}$", fontsize=12, fontweight="bold")

plt.tight_layout()
plt.savefig("../figs/error_2D.pdf", dpi=300, bbox_inches="tight")
plt.show()