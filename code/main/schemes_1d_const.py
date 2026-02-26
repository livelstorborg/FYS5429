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

t1_eval = 0.25
t2_eval = 0.5
t3_eval = T

# ============================================
#           Constant c = 1.0
# ============================================

# --------- 1D ---------
x, t, dx, dt = create_grid(Nx=Nx, T=T, c=c, cfl=cfl, dim=1)

u_fd = fd_solve(x, t, dx, dt, c=c, dim=1)
u_fem = fem_solve(x, t, dx, dt, c=c, dim=1)
u_ex = u_exact(x, t, c=c, dim=1)

idx_t1 = jnp.argmin(jnp.abs(t - t1_eval))
idx_t2 = jnp.argmin(jnp.abs(t - t2_eval))
idx_t3 = jnp.argmin(jnp.abs(t - t3_eval))

fd_t1 = {
    "grid": x,
    "u_num": u_fd[idx_t1, :],
    "u_true": u_ex[idx_t1, :],
    "dx": dx,
    "t": float(t[idx_t1]),
    "dim": 1
}
fd_t2 = {
    "grid": x,
    "u_num": u_fd[idx_t2, :],
    "u_true": u_ex[idx_t2, :],
    "dx": dx,
    "t": float(t[idx_t2]),
    "dim": 1
}

fd_t3 = {
    "grid": x,
    "u_num": u_fd[idx_t3, :],
    "u_true": u_ex[idx_t3, :],
    "dx": dx,
    "t": float(t[idx_t3]),
    "dim": 1
}

fem_t1 = {
    "grid": x,
    "u_num": u_fem[idx_t1, :],
    "u_true": u_ex[idx_t1, :],
    "dx": dx,
    "t": float(t[idx_t1]),
    "dim": 1
}
fem_t2 = {
    "grid": x,
    "u_num": u_fem[idx_t2, :],
    "u_true": u_ex[idx_t2, :],
    "dx": dx,
    "t": float(t[idx_t2]),
    "dim": 1
}

fem_t3 = {
    "grid": x,
    "u_num": u_fem[idx_t3, :],
    "u_true": u_ex[idx_t3, :],
    "dx": dx,
    "t": float(t[idx_t3]),
    "dim": 1
}   


plot_error_at_t(
    grid=x,
    u_num=[
        {"data": [fd_t1["u_num"],  fd_t2["u_num"], fd_t3["u_num"]],  "label": "FD"},
        {"data": [fem_t1["u_num"], fem_t2["u_num"], fem_t3["u_num"]], "label": "FEM"},
    ],
    u_true=[fd_t1["u_true"], fd_t2["u_true"], fd_t3["u_true"]],
    dx=fd_t1["dx"],
    t=[fd_t1["t"], fd_t2["t"], fd_t3["t"]],
    dim=1,
    filepath="../figs/error_1D.pdf",
)





# ---------- Solution surfaces ----------
fig_exact = plot_3d_surface(
    x,
    t,
    u_ex,
    elev=20,
    azim=45,
    title="Analytical Solution (1D)",
    savefig=True,
    save_path="../figs/exact_solution_1D.pdf",
)

fig_fd = plot_3d_surface(
    x,
    t,
    u_fd,
    elev=20,
    azim=45,
    title="Finite Difference Solution (1D)",
    savefig=True,
    save_path="../figs/fd_solution_1D.pdf",
)

fig_fem = plot_3d_surface(
    x,
    t,
    u_fem,
    elev=20,
    azim=45,
    title="Finite Element Solution (1D)",
    savefig=True,
    save_path="../figs/fem_solution_1D.pdf",
)

subplot_fig = subplot_3d_surfaces(
    figures=[
        {'x': x, 't': t, 'U': u_ex},
        {'x': x, 't': t, 'U': u_fd},
        {'x': x, 't': t, 'U': u_fem},
    ],
    titles=["Analytical", "Finite Difference", "Finite Element"],
    elev=20,
    azims=[45, 45, 45],
    cmap="viridis",
    colorbar_label="u(x, t)",
    suptitle="Wave Equation Solutions",
    savefig=True,
    save_path="../figs/comparison_subplot.pdf",
    show=True,
)