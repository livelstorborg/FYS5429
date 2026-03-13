import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
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

# --- Metamaterial setup ---
# n²(x,y,t) = 1 + eta(x,y,t),  eta = s(t)/eps²  inside particles D_i = eps*B + z_i
# c_fn(x,y,t) = c0 / n(x,y,t)  (effective wave speed)
# ── Parameters — fixed across 1D, 2D, FD, FEM, PINN ──────────────
c0 = 1.0
eps = 0.1  # particle size ε << 1
delta = 0.004  # interface width δ << ε
Omega = 2 * np.pi * 2.0  # modulation frequency Ω
s = lambda t: eps**2 * jnp.sin(Omega * t)  # critical scaling

# 2D particle centres (x_i, y_i) on a 3×3 grid in [0,1]²
z_particles_2d = [
    (0.25, 0.25),
    (0.50, 0.25),
    (0.75, 0.25),
    (0.25, 0.50),
    (0.50, 0.50),
    (0.75, 0.50),
    (0.25, 0.75),
    (0.50, 0.75),
    (0.75, 0.75),
]


def _smooth_indicator_2d(x, y, z_i):
    """Smooth radial indicator for a 2D disc of radius eps centred at z_i=(zx, zy)."""
    zx, zy = z_i
    dist = jnp.sqrt((x - zx) ** 2 + (y - zy) ** 2 + 1e-30)
    return 0.5 * (1.0 + jnp.tanh((eps / 2 - dist) / delta))


def n_sq(x, y, t):
    """Effective refractive index squared n²(x,y,t)."""
    eta = (s(t) / eps**2) * sum(_smooth_indicator_2d(x, y, z) for z in z_particles_2d)
    return jnp.maximum(1.0 + eta, 0.1)  # clamp to avoid n² ≤ 0


def c_fn(x_arr, y_arr, t_scalar):
    """Spatially varying wave speed c(x,y,t) = c0 / n(x,y,t)."""
    return c0 / jnp.sqrt(n_sq(x_arr, y_arr, t_scalar))


c_max = c0  # max c_eff = c0 outside particles where n = 1

# ── Grid ──────────────────────────────────────────────────────────
Nx = 200  # reduced from 500: 2D memory scales as Nx*Ny*Nt
Ny = 200
T = 1.0
cfl = 0.8

x, y, t, dx, dy, dt = create_grid(Nx=Nx, Ny=Ny, T=T, c=c_max, cfl=cfl, dim=2)

# ── Initial condition: sin(π x)·sin(π y) on [0,1]² ───────────────
XX, YY = jnp.meshgrid(x, y, indexing="ij")
u0 = jnp.sin(jnp.pi * XX) * jnp.sin(jnp.pi * YY)

# ── Solvers ───────────────────────────────────────────────────────
u_fd = fd_solve(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, c=c_fn, u0=u0, dim=2)
u_fem = fem_solve(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, c=c_fn, u0=u0, dim=2)

# Reference: constant-c solve for comparison
u_const = fd_solve(x=x, y=y, t=t, dx=dx, dy=dy, dt=dt, c=c_max, u0=u0, dim=2)

# ── Time snapshots ────────────────────────────────────────────────
t1_eval = 0.25
t2_eval = 0.50
t3_eval = 1.0 / np.sqrt(2)
t4_eval = T

t_evals = [t1_eval, t2_eval, t3_eval, t4_eval]
t_indices = [int(np.argmin(np.abs(t - te))) for te in t_evals]

# ==================================================================
#               Surface / slice plots at each snapshot
# ==================================================================
for te, ti in zip(t_evals, t_indices):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    for ax, U, title in zip(
        axes,
        [u_const, u_fd, u_fem],
        [r"Constant $c$", "FD (metamaterial)", "FEM (metamaterial)"],
    ):
        im = ax.contourf(x, y, U[ti].T, levels=40, cmap="viridis")
        fig.colorbar(im, ax=ax, label="u(x,y)")
        ax.set_title(f"{title}\nt = {te:.3f}")
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.suptitle(r"Wave Equation Solutions — 2D metamaterial", fontsize=13)
    fig.tight_layout()
    plt.savefig(f"figs/schemes/solution_2d_t{te:.3f}.pdf")
    plt.show()

# ==================================================================
#               Error plots (FD vs FEM) at each snapshot
# ==================================================================
for te, ti in zip(t_evals, t_indices):
    err = np.abs(u_fd[ti] - u_fem[ti])
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.contourf(x, y, err.T, levels=40, cmap="hot_r")
    fig.colorbar(im, ax=ax, label="|FD − FEM|")
    ax.set_title(f"|FD − FEM| at t = {te:.3f}")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    plt.savefig(f"figs/schemes/error_2d_fd_fem_t{te:.3f}.pdf")
    plt.show()
