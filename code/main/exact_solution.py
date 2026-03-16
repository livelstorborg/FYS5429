import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
import numpy as np

from src.pde import u_exact, create_grid
from src.plotting import plot_3d_surface, plot_2d_snapshots


# ----- Parameters -----
Nx = 100
Ny = 100
T = 1.0
c = 1.0
cfl = 1.0

# ---------- 1D -----------
x, t, dx, dt = create_grid(Nx=Nx, T=T, c=c, cfl=cfl, dim=1)
u_ex = u_exact(x, t, c=c, dim=1)

fig_exact = plot_3d_surface(
    x,
    t,
    u_ex,
    elev=20,
    azim=45,
    cmap="inferno",
    title="Analytical Solution (1D)",
    savefig=True,
    filepath=str(Path(__file__).parent.parent / "figs" / "exact_solution_1d.pdf"),
)

# ---------- 2D ----------
x2, y2, t2, dx2, dy2, dt2 = create_grid(Nx=Nx, Ny=Ny, T=T, c=c, cfl=cfl, dim=2)
u_ex_2d = u_exact(x2, t2, y=y2, c=c, dim=2)  # (Nt, Nx, Ny)

t_snap_vals = [0.25, 0.5, 1.0 / np.sqrt(2), 1.0]
t_snap_idx = [int(jnp.argmin(jnp.abs(t2 - tv))) for tv in t_snap_vals]

plot_2d_snapshots(
    u_ex_2d, t_snap_idx, t_snap_vals,
    title=r"$\mathbf{2D}$ Wave Equation — Analytical Solution",
    cmap="inferno",
    savefig=True,
    filepath=str(Path(__file__).parent.parent / "figs" / "exact_solution_2d.pdf"),
)
