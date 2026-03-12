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


x, t, dx, dt = create_grid(Nx=Nx, T=T, c=c, cfl=cfl, dim=1)

u_fd = fd_solve(x, t, dx, dt, c=c, dim=1)
u_fem = fem_solve(x, t, dx, dt, c=c, dim=1)
u_ex = u_exact(x, t, c=c, dim=1)


# ==================================================================
#                       Surface solution plots
# ==================================================================
fig_exact = plot_3d_surface(
    x,
    t,
    u_ex,
    elev=20,
    azim=45,
    title="Analytical Solution (1D)",
)

fig_fd = plot_3d_surface(
    x,
    t,
    u_fd,
    elev=20,
    azim=45,
    title="Finite Difference Solution (1D)",
)

fig_fem = plot_3d_surface(
    x,
    t,
    u_fem,
    elev=20,
    azim=45,
    title="Finite Element Solution (1D)",
)

subplot_fig = subplot_3d_surfaces(
    figures=[
        {"x": x, "t": t, "U": u_ex},
        {"x": x, "t": t, "U": u_fd},
        {"x": x, "t": t, "U": u_fem},
    ],
    titles=["Analytical", "Finite Difference", "Finite Element"],
    elev=20,
    azims=[45, 45, 45],
    cmap="viridis",
    colorbar_label="u(x, t)",
    suptitle="Wave Equation Solutions",
    show=True,
)


# ==================================================================
#                         Surface error plots
# ==================================================================

u_fd_error = np.abs(u_fd - u_ex)
u_fem_error = np.abs(u_fem - u_ex)

fig_fd_error = plot_3d_surface(
    x,
    t,
    u_fd_error,
    elev=20,
    azim=45,
    title="Finite Difference Error (1D)",
    show=True,
    savefig=True,
    filepath="figs/fd_error_surface_1d_const.pdf",
)

fig_fem_error = plot_3d_surface(
    x,
    t,
    u_fem_error,
    elev=20,
    azim=45,
    title="Finite Element Error (1D)",
    show=True,
    savefig=True,
    filepath="figs/fem_error_surface_1d_const.pdf",
)
