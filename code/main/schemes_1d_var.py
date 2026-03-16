import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import matplotlib.pyplot as plt

from src.pde import fd_solve, fem_solve, create_grid
from src.pinn import train_pinn
from src.experiment import absolute_error, relative_error
from src.plotting import (
    plot_3d_surface,
    subplot_3d_surfaces,
)

# --- Metamaterial setup ---
# n²(x,t) = 1 + eta(x,t),  eta = s(t)/eps²  inside particles D_i = eps*B + z_i
# c_fn(x,t) = c0 / n(x,t)  (effective wave speed)
# ── Parameters — fixed across 1D, 2D, FD, FEM, PINN ──────────────
c0 = 1.0
eps = 0.2  # particle size ε << 1
delta = 0.004  # interface width δ << ε
Omega = 2 * np.pi * 4.0  # modulation frequency Ω
s = lambda t: eps**2 * jnp.sin(Omega * t)  # critical scaling
z_particles = [0.5]  # single particle at centre


def _smooth_indicator(x, z_i):
    dist = jnp.sqrt((x - z_i) ** 2 + 1e-30)
    return 0.5 * (1.0 + jnp.tanh((eps / 2 - dist) / delta))


def n_sq(x, t):
    eta = (s(t) / eps**2) * sum(_smooth_indicator(x, z) for z in z_particles)
    return jnp.maximum(1.0 + eta, 0.1)  # clamp to avoid n²≤ 0


def c_fn(x_arr, t_scalar):
    return c0 / jnp.sqrt(n_sq(x_arr, t_scalar))


c_max = c0 / jnp.sqrt(0.1)  # max c_eff when n² hits the clamp (= 0.1)

# Parameters
Nx = 500
T = 1.0
cfl = 0.8

x, t, dx, dt = create_grid(Nx=Nx, T=T, c=c_max, cfl=cfl, dim=1)

u0 = jnp.sin(jnp.pi * x)
u_fd = fd_solve(x, t, dx, dt, c=c_fn, u0=u0, dim=1)
u_fem = fem_solve(x, t, dx, dt, c=c_fn, u0=u0, dim=1)

# Reference: same problem with constant c for comparison
u_const = fd_solve(x, t, dx, dt, c=c0, dim=1)


# ==================================================================
#                       Surface solution plots
# ==================================================================
fig_const = plot_3d_surface(
    x,
    t,
    u_const,
    elev=20,
    azim=45,
    title=r"Constant $c$ Solution (1D)",
)

fig_fd = plot_3d_surface(
    x,
    t,
    u_fd,
    elev=20,
    azim=45,
    title=r"Finite Difference Solution (1D, metamaterial)",
)

fig_fem = plot_3d_surface(
    x,
    t,
    u_fem,
    elev=20,
    azim=45,
    title=r"Finite Element Solution (1D, metamaterial)",
)

subplot_fig = subplot_3d_surfaces(
    figures=[
        {"x": x, "t": t, "U": u_const},
        {"x": x, "t": t, "U": u_fd},
        {"x": x, "t": t, "U": u_fem},
    ],
    titles=[r"Constant $c$", "FD (metamaterial)", "FEM (metamaterial)"],
    elev=20,
    azims=[45, 45, 45],
    cmap="viridis",
    colorbar_label="u(x, t)",
    suptitle=r"Wave Equation Solutions — metamaterial",
    show=True,
    savefig=True,
    filepath="figs/schemes/solution_surface_comparison_1d.pdf",
)
