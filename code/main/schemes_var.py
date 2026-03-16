import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np

from src.pde import fem_solve, create_grid, make_c_fn
from src.experiment import absolute_error, relative_error
from src.plotting import (
    plot_3d_surface,
    subplot_3d_surfaces,
    plot_2d_snapshots,
)

# ===================== Metamaterial setup =====================
# n²(x,t) = 1 + eta(x,t),  eta = s(t)/eps²  inside particles D_i = eps*B + z_i
# c_fn(x,t) = c0 / n(x,t)  (effective wave speed)

c0 = 1.0
eps = 0.4     # particle size ε << 1
delta = 0.004  # interface width δ << ε
Omega = 2 * np.pi * 4.0  # modulation frequency Ω
s = lambda t: eps**2 * jnp.sin(Omega * t)

c_max = c0 / jnp.sqrt(0.1)  # worst-case wave speed (n²_min = 0.1)


# ==================================================================
#                          1D experiment
# ==================================================================
z_particles_1d = [(0.5,)]  # single particle at x = 0.5

c_fn_1d = make_c_fn(z_particles_1d, c0=c0, eps=eps, delta=delta, s_fn=s)

Nx = 500
T = 1.0
cfl = 0.8

x, t, dx, dt = create_grid(Nx=Nx, T=T, c=c_max, cfl=cfl, dim=1)

u0_1d = jnp.sin(jnp.pi * x)
u_fem = fem_solve(x, t, dx, dt, c=c_fn_1d, u0=u0_1d, dim=1)
u_const = fem_solve(x, t, dx, dt, c=c0, dim=1)

# Surface solution plots
fig_const = plot_3d_surface(x, t, u_const, elev=20, azim=45)
fig_fem = plot_3d_surface(x, t, u_fem, elev=20, azim=45)

subplot_3d_surfaces(
    figures=[
        {"x": x, "t": t, "U": u_const},
        {"x": x, "t": t, "U": u_fem},
    ],
    titles=[r"FEM $c_{constant}$", "FEM $c(x, t)$"],
    elev=20,
    azims=[45, 45],
    cmap="inferno",
    colorbar_label="u(x, t)",
    suptitle=r"1D Wave Equation",
    show=True,
    savefig=True,
    filepath="figs/schemes/solution_surface_comparison_1d.pdf",
)


# ==================================================================
#                          2D experiment
# ==================================================================
z_particles_2d = [(0.5, 0.5)]  # single particle at (x, y) = (0.5, 0.5)

c_fn_2d = make_c_fn(z_particles_2d, c0=c0, eps=eps, delta=delta, s_fn=s)

Nx2 = 100
Ny2 = 100
x2, y2, t2, dx2, dy2, dt2 = create_grid(Nx=Nx2, Ny=Ny2, T=T, c=c_max, cfl=cfl, dim=2)

X2, Y2 = jnp.meshgrid(x2, y2, indexing="ij")
u0_2d = jnp.sin(jnp.pi * X2) * jnp.sin(jnp.pi * Y2)

u_fem_2d = fem_solve(x2, t2, dx2, dt2, c=c_fn_2d, u0=u0_2d, y=y2, dy=dy2, dim=2)
u_const_2d = fem_solve(x2, t2, dx2, dt2, c=c0, u0=u0_2d, y=y2, dy=dy2, dim=2)

# Snapshot times: t = 0.25, 0.5, 1/sqrt(2), 1.0
snap_times = [0.25, 0.5, 1.0 / np.sqrt(2), 1.0]
snap_indices = [int(jnp.argmin(jnp.abs(t2 - ts))) for ts in snap_times]


plot_2d_snapshots(
    u_fem_2d,
    t_indices=snap_indices,
    t_labels=snap_times,
    title=r"FEM $c(x,y,t)$",
    cmap="inferno",
    show=True,
    savefig=True,
    filepath="figs/schemes/solution_snapshots_2d_var.pdf",
)

# 3D x-t surface plots at fixed y=0.5 (slice through the particle centre).
# u_*_2d has shape (Nt+1, Nx+1, Ny+1), so [:, :, iy] gives (Nt+1, Nx+1) —
# exactly the shape subplot_3d_surfaces expects, no transpose needed.
iy = int(jnp.argmin(jnp.abs(y2 - 0.5)))

subplot_3d_surfaces(
    figures=[
        {"x": x2, "t": t2, "U": u_const_2d[:, :, iy]},
        {"x": x2, "t": t2, "U": u_fem_2d[:, :, iy]},
    ],
    titles=[r"FEM $c_{constant}$", r"FEM $c(x,y,t)$"],
    elev=20,
    azims=[45, 45],
    cmap="inferno",
    xlabel="x",
    ylabel="t",
    colorbar_label=r"$u(x,\, y{=}0.5,\, t)$",
    suptitle=r"2D Wave Equation — $x$-$t$ slice at $y=0.5$",
    show=True,
    savefig=True,
    filepath="figs/schemes/solution_surface_2d_xt.pdf",
)
