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






# ============================================

# --------- 2D ---------
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



# --------- 2D ---------

# # ===== FD vs Analytical Nx1=10 -> dx=0.1=====
# fd_t1_2D_Nx1 = test_explicit_scheme_at_t(Nx=Nx1, Ny=Ny1, T=T, t_eval=t1, dim=2)
# fd_t2_2D_Nx1 = test_explicit_scheme_at_t(Nx=Nx1, Ny=Ny1, T=T, t_eval=t2, dim=2)
# fd_t3_2D_Nx1 = test_explicit_scheme_at_t(Nx=Nx1, Ny=Ny1, T=T, t_eval=t3, dim=2)

# # ===== FD vs Analytical Nx2=100 -> dx=0.01 =====
# fd_t1_2D_Nx2 = test_explicit_scheme_at_t(Nx=Nx2, Ny=Ny2, T=T, t_eval=t1, dim=2)
# fd_t2_2D_Nx2 = test_explicit_scheme_at_t(Nx=Nx2, Ny=Ny2, T=T, t_eval=t2, dim=2)
# fd_t3_2D_Nx2 = test_explicit_scheme_at_t(Nx=Nx2, Ny=Ny2, T=T, t_eval=t3, dim=2)

# # ===== Solutions for Nx1 =====
# plot_solution_at_t(
#     grid=[fd_t1_2D_Nx1["grid"], fd_t2_2D_Nx1["grid"], fd_t3_2D_Nx1["grid"]],
#     u_num=[fd_t1_2D_Nx1["u_num"], fd_t2_2D_Nx1["u_num"], fd_t3_2D_Nx1["u_num"]],
#     u_true=[fd_t1_2D_Nx1["u_true"], fd_t2_2D_Nx1["u_true"], fd_t3_2D_Nx1["u_true"]],
#     dx=fd_t1_2D_Nx1["dx"],
#     t=[fd_t1_2D_Nx1["t"], fd_t2_2D_Nx1["t"], fd_t3_2D_Nx1["t"]],
#     dim=fd_t1_2D_Nx1["dim"],
#     filepath=f"figs/solution_2D_Nx{Nx1}_dx{fd_t1_2D_Nx1['dx']:.3f}.pdf",
# )

# # ===== Solutions for Nx2 =====
# plot_solution_at_t(
#     grid=[fd_t1_2D_Nx2["grid"], fd_t2_2D_Nx2["grid"], fd_t3_2D_Nx2["grid"]],
#     u_num=[fd_t1_2D_Nx2["u_num"], fd_t2_2D_Nx2["u_num"], fd_t3_2D_Nx2["u_num"]],
#     u_true=[fd_t1_2D_Nx2["u_true"], fd_t2_2D_Nx2["u_true"], fd_t3_2D_Nx2["u_true"]],
#     dx=fd_t1_2D_Nx2["dx"],
#     t=[fd_t1_2D_Nx2["t"], fd_t2_2D_Nx2["t"], fd_t3_2D_Nx2["t"]],
#     dim=fd_t1_2D_Nx2["dim"],
#     filepath=f"figs/solution_2D_Nx{Nx2}_dx{fd_t1_2D_Nx2['dx']:.3f}.pdf",
# )

# error_data_2D = [
#     # Nx1 data
#     {
#         'grid': fd_t1_2D_Nx1["grid"],
#         'error': [
#             absolute_error(fd_t1_2D_Nx1["u_num"], fd_t1_2D_Nx1["u_true"]),
#             absolute_error(fd_t2_2D_Nx1["u_num"], fd_t2_2D_Nx1["u_true"]),
#             absolute_error(fd_t3_2D_Nx1["u_num"], fd_t3_2D_Nx1["u_true"])
#         ],
#         'dx': fd_t1_2D_Nx1["dx"]
#     },
#     # Nx2 data
#     {
#         'grid': fd_t1_2D_Nx2["grid"],
#         'error': [
#             absolute_error(fd_t1_2D_Nx2["u_num"], fd_t1_2D_Nx2["u_true"]),
#             absolute_error(fd_t2_2D_Nx2["u_num"], fd_t2_2D_Nx2["u_true"]),
#             absolute_error(fd_t3_2D_Nx2["u_num"], fd_t3_2D_Nx2["u_true"])
#         ],
#         'dx': fd_t1_2D_Nx2["dx"]
#     }
# ]

# plot_scheme_error_at_t(
#     grid=error_data_2D,
#     error=None,
#     dx=None,
#     t=[t1, t2, t3],
#     dim=2,
#     title="Absolute Error (2D)",
#     filepath="figs/abs_error_multi_dx_2D.pdf",
#     log_scale=True,
# )

# ============================================
#           Space dependent c = n(x)
# ============================================



# # --------- 1D ---------
# # ===== FD vs Analytical Nx1=10 -> dx=0.1=====
# fd_t1_1D_Nx1 = test_explicit_scheme_at_t(Nx=Nx1, T=T, c=C, t_eval=t1, dim=1)
# fd_t2_1D_Nx1 = test_explicit_scheme_at_t(Nx=Nx1, T=T, c=C, t_eval=t2, dim=1)
# fd_t3_1D_Nx1 = test_explicit_scheme_at_t(Nx=Nx1, T=T, c=C, t_eval=t3, dim=1)

# # ===== FD vs Analytical Nx2=100 -> dx=0.01 =====
# fd_t1_1D_Nx2 = test_explicit_scheme_at_t(Nx=Nx2, T=T, t_eval=t1, dim=1)
# fd_t2_1D_Nx2 = test_explicit_scheme_at_t(Nx=Nx2, T=T, t_eval=t2, dim=1)
# fd_t3_1D_Nx2 = test_explicit_scheme_at_t(Nx=Nx2, T=T, t_eval=t3, dim=1)

# # ===== Solutions for Nx1 =====
# plot_solution_at_t(
#     grid=[fd_t1_1D_Nx1["grid"], fd_t2_1D_Nx1["grid"], fd_t3_1D_Nx1["grid"]],
#     u_num=[fd_t1_1D_Nx1["u_num"], fd_t2_1D_Nx1["u_num"], fd_t3_1D_Nx1["u_num"]],
#     u_true=[fd_t1_1D_Nx1["u_true"], fd_t2_1D_Nx1["u_true"], fd_t3_1D_Nx1["u_true"]],
#     dx=fd_t1_1D_Nx1["dx"],
#     t=[fd_t1_1D_Nx1["t"], fd_t2_1D_Nx1["t"], fd_t3_1D_Nx1["t"]],
#     dim=fd_t1_1D_Nx1["dim"],
#     filepath=f"figs/solution_1D_Nx{Nx1}_dx{fd_t1_1D_Nx1['dx']:.3f}.pdf",
# )

# # ===== Solutions for Nx2 =====
# plot_solution_at_t(
#     grid=[fd_t1_1D_Nx2["grid"], fd_t2_1D_Nx2["grid"], fd_t3_1D_Nx2["grid"]],
#     u_num=[fd_t1_1D_Nx2["u_num"], fd_t2_1D_Nx2["u_num"], fd_t3_1D_Nx2["u_num"]],
#     u_true=[fd_t1_1D_Nx2["u_true"], fd_t2_1D_Nx2["u_true"], fd_t3_1D_Nx2["u_true"]],
#     dx=fd_t1_1D_Nx2["dx"],
#     t=[fd_t1_1D_Nx2["t"], fd_t2_1D_Nx2["t"], fd_t3_1D_Nx2["t"]],
#     dim=fd_t1_1D_Nx2["dim"],
#     filepath=f"figs/solution_1D_Nx{Nx2}_dx{fd_t1_1D_Nx2['dx']:.3f}.pdf",
# )