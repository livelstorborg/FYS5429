import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

from src.pde import fem_solve, u_exact, create_grid, make_c_fn
from src.plotting import (
    plot_2d_snapshots,
    plot_3d_surface,
    subplot_3d_surfaces,
)

# ==================================================================
#                       Shared Configuration
# ==================================================================

T = 1.0
c0 = 1.0

# Grid sizes
Nx_1d = 500   
Nx_2d = 100   
Ny_2d = 100


snap_times = [0.25, 0.5, 1.0 / np.sqrt(2), T]

# Output directories
FIG_1D = Path(__file__).parent.parent / "figs" / "FEM" / "1D"
FIG_2D = Path(__file__).parent.parent / "figs" / "FEM" / "2D"

# ==================================================================
#                   Metamaterial / Variable-c Setup
# ==================================================================
# n²(x,t) = 1 + eta(x,t),  eta = s(t)/eps²  inside particles D_i = eps*B + z_i
# c_fn(x,t) = c0 / n(x,t)  (effective wave speed)

eps = 0.4      # particle size ε << 1
delta = 0.004  # interface width δ << ε
Omega = 2 * np.pi * 4.0  # modulation frequency Ω
s_fn = lambda t: eps**2 * jnp.sin(Omega * t)

c_max = c0 / jnp.sqrt(0.1)  # worst-case wave speed (n²_min = 0.1)

# 1D: single particle at x = 0.5
z_particles_1d = [(0.5,)]
c_fn_1d = make_c_fn(z_particles_1d, c0=c0, eps=eps, delta=delta, s_fn=s_fn)

# 2D: single particle at (x, y) = (0.5, 0.5)
z_particles_2d = [(0.5, 0.5)]
c_fn_2d = make_c_fn(z_particles_2d, c0=c0, eps=eps, delta=delta, s_fn=s_fn)



# ==================================================================
#                  1D — Constant Wave Equation
# ==================================================================
cfl_const = 1.0
x1, t1, dx1, dt1 = create_grid(Nx=Nx_1d, T=T, c=c0, cfl=cfl_const, dim=1)

u_fem_const  = fem_solve(x1, t1, dx1, dt1, c=c0, dim=1)
u_ex_const   = u_exact(x1, t1, c=c0, dim=1)
u_fem_const_error = np.abs(u_fem_const - u_ex_const)

# Solution surface comparison
subplot_3d_surfaces(
    figures=[
        {"x": x1, "t": t1, "U": u_ex_const},
        {"x": x1, "t": t1, "U": u_fem_const},
        {"x": x1, "t": t1, "U": u_fem_const_error},
    ],
    titles=["Analytical", "FEM", "Absolute Error"],
    elev=20,
    azims=[45, 45, 45],
    cmap="inferno",
    colorbar_label="u(x, t)",
    suptitle=r"1D Wave Equation with $\mathbf{c_{constant}}$",
    show=True,
    savefig=True,
    filepath=str(FIG_1D / "solution_surface_comparison_1d_const.pdf"),
)




# ==================================================================
#                  1D — Variable Wave Equation
# ==================================================================
cfl_var = 0.8
x1v, t1v, dx1v, dt1v = create_grid(Nx=Nx_1d, T=T, c=c_max, cfl=cfl_var, dim=1)

u0_1d = jnp.sin(jnp.pi * x1v)
u_fem_var   = fem_solve(x1v, t1v, dx1v, dt1v, c=c_fn_1d, u0=u0_1d, dim=1)
u_fem_homog = fem_solve(x1v, t1v, dx1v, dt1v, c=c0, dim=1)

# Solution surface comparison: homogeneous vs variable
subplot_3d_surfaces(
    figures=[
        {"x": x1v, "t": t1v, "U": u_fem_homog},
        {"x": x1v, "t": t1v, "U": u_fem_var},
    ],
    titles=[r"$\mathbf{c_{constant}}$", r"$\mathbf{c(x,t)}$"],
    elev=20,
    azims=[45, 45],
    cmap="inferno",
    colorbar_label="u(x, t)",
    suptitle=r"FEM Solution for 1D Wave Equation",
    show=True,
    savefig=True,
    filepath=str(FIG_1D / "solution_surface_comparison_1d_var.pdf"),
)




# ==================================================================
#                  2D — Constant Wave Equation
# ==================================================================
x2c, y2c, t2c, dx2c, dy2c, dt2c = create_grid(
    Nx=Nx_2d, Ny=Ny_2d, T=T, c=c0, cfl=cfl_const, dim=2
)

u_fem_2d_const  = fem_solve(x2c, t2c, dx2c, dt2c, y=y2c, dy=dy2c, c=c0, dim=2)
u_ex_2d_const   = u_exact(x2c, t2c, y=y2c, c=c0, dim=2)

snap_idx_const = [int(jnp.argmin(jnp.abs(t2c - ts))) for ts in snap_times]

# Absolute error heatmaps
fem_error_2d_const = np.abs(np.array(u_fem_2d_const) - np.array(u_ex_2d_const))
plot_2d_snapshots(
    fem_error_2d_const,
    t_indices=snap_idx_const,
    t_labels=snap_times,
    title=r"2D Absolute Error for FEM with $\mathbf{c_{constant}}$",
    cmap="inferno",
    savefig=True,
    filepath=str(FIG_2D / "fem_error_heatmaps_2d_const.pdf"),
)




# ==================================================================
#                  2D — Variable Wave Equation
# ==================================================================
x2v, y2v, t2v, dx2v, dy2v, dt2v = create_grid(
    Nx=Nx_2d, Ny=Ny_2d, T=T, c=c_max, cfl=cfl_var, dim=2
)

X2v, Y2v = jnp.meshgrid(x2v, y2v, indexing="ij")
u0_2d = jnp.sin(jnp.pi * X2v) * jnp.sin(jnp.pi * Y2v)

u_fem_2d_var   = fem_solve(x2v, t2v, dx2v, dt2v, c=c_fn_2d, u0=u0_2d, y=y2v, dy=dy2v, dim=2)
u_fem_2d_homog = fem_solve(x2v, t2v, dx2v, dt2v, c=c0,      u0=u0_2d, y=y2v, dy=dy2v, dim=2)

snap_idx_var = [int(jnp.argmin(jnp.abs(t2v - ts))) for ts in snap_times]

# Variable-c solution snapshots
plot_2d_snapshots(
    u_fem_2d_var,
    t_indices=snap_idx_var,
    t_labels=snap_times,
    title=r"FEM Solution for 2D Wave Equation with $\mathbf{c(x,y,t)}$",
    cmap="inferno",
    show=True,
    savefig=True,
    filepath=str(FIG_2D / "solution_snapshots_2d_var.pdf"),
)

# Optional: 3D x-t surface at fixed y=0.5 (slice through particle centre).
# u_*_2d has shape (Nt+1, Nx+1, Ny+1); [:, :, iy] → (Nt+1, Nx+1).
# iy = int(jnp.argmin(jnp.abs(y2v - 0.5)))
# subplot_3d_surfaces(
#     figures=[
#         {"x": x2v, "t": t2v, "U": u_fem_2d_homog[:, :, iy]},
#         {"x": x2v, "t": t2v, "U": u_fem_2d_var[:, :, iy]},
#     ],
#     titles=[r"FEM $\mathbf{c_{constant}}$", r"FEM $\mathbf{c(x,y,t)}$"],
#     elev=20, azims=[45, 45],
#     cmap="inferno", xlabel="x", ylabel="t",
#     colorbar_label=r"$u(x,\, y{=}0.5,\, t)$",
#     suptitle=r"2D Wave Equation — $x$-$t$ slice at $y=0.5$",
#     show=True, savefig=True,
#     filepath=str(FIG_2D / "solution_surface_2d_xt.pdf"),
# )