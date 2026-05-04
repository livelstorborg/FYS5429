import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

import pandas as pd

from src.pinn import train_pinn
from src.pde import fem_solve, create_grid, make_c_fn
from src.plotting import subplot_3d_surfaces


# ==================================================================
#                         Parameters
# ==================================================================
T      = 1.0
c0     = 1.0
Nx_fem = 1000    # FEM spatial resolution (dx=0.001, needs ~3 cells across delta=0.004)
Nx_plt = 100     # plot grid resolution
Nt_plt = 100


# ==================================================================
#               Metamaterial / Variable-c Setup
# ==================================================================
# n²(x,t) = 1 + η(x,t),  η = s(t)/ε²  inside particle D = εB + z
# c(x,t)  = c0 / n(x,t)

eps   = 0.4                         # particle radius
delta = 0.004                       # interface width
Omega = 2 * np.pi * 4.0             # modulation frequency
s_fn  = lambda t: eps**2 * jnp.sin(Omega * t)

c_max       = c0 / jnp.sqrt(0.1)   # worst-case wave speed (n²_min = 0.1)
z_particles = [(0.5,)]              # single particle at x = 0.5
c_fn_1d     = make_c_fn(z_particles, c0=c0, eps=eps, delta=delta, s_fn=s_fn)


DATA_DIR = Path(__file__).parent.parent / "data" / "1d_var"
DATA_DIR.mkdir(parents=True, exist_ok=True)

use_pre_computed = False
solution_path = DATA_DIR / "solution.csv"
losses_path   = DATA_DIR / "losses.csv"

if use_pre_computed and solution_path.exists() and losses_path.exists():
    print(f"Loading precomputed results from {DATA_DIR}")
    df_sol = pd.read_csv(solution_path)
    x_plt      = df_sol["x"].unique()
    t_plt      = df_sol["t"].unique()
    u_fem_plt  = df_sol.pivot(index="t", columns="x", values="u_fem").values
    u_pinn_plt = df_sol.pivot(index="t", columns="x", values="u_pinn").values
    error_plt  = df_sol.pivot(index="t", columns="x", values="error").values
    losses     = pd.read_csv(losses_path)["loss"].values
else:
    # ==================================================================
    #                     FEM solve (fine grid)
    # ==================================================================
    x_fem, t_fem, dx_fem, dt_fem = create_grid(Nx=Nx_fem, T=T, c=c_max, cfl=0.8, dim=1)

    u0_1d = jnp.sin(jnp.pi * x_fem)
    u_fem = fem_solve(x_fem, t_fem, dx_fem, dt_fem, c=c_fn_1d, u0=u0_1d, dim=1)

    # ==================================================================
    #               PINN: adam (5000 steps) → lbfgs (3000 steps)
    # ==================================================================
    # Particle boundaries: center ± eps for each particle
    interface_pts = [z[0] - eps for z in z_particles] + [z[0] + eps for z in z_particles]

    model, losses, _ = train_pinn(
        (128, 128, 128, 128),
        dim=1,
        activation=jnn.gelu,
        optimizer="lbfgs",
        steps=2000,
        adam_warmup_steps=3000,
        N_int=3000,
        N_ic=300,
        T=T,
        c=c_fn_1d,
        lambda_ic=100.0,
        lr=1e-3,
        seed=0,
        log_every=500,
        interface_points=interface_pts,
        bias_frac=0.3,
        bias_std=3 * delta,
        fourier_freqs=[1, 2, 4, 8],
    )

    # ==================================================================
    #          Evaluate both solutions on a shared plot grid
    # ==================================================================
    x_plt = jnp.linspace(0.0, 1.0, Nx_plt)
    t_plt = jnp.linspace(0.0, T,   Nt_plt)

    ix = np.round(np.linspace(0, len(x_fem) - 1, Nx_plt)).astype(int)
    it = np.round(np.linspace(0, len(t_fem) - 1, Nt_plt)).astype(int)
    u_fem_plt = np.array(u_fem)[np.ix_(it, ix)]

    X, T_grid = jnp.meshgrid(x_plt, t_plt)
    xt = jnp.stack([X.ravel(), T_grid.ravel()], axis=1)
    u_pinn_plt = model(xt).squeeze().reshape(Nt_plt, Nx_plt)

    error_plt = np.abs(np.array(u_pinn_plt) - u_fem_plt)

    # ==================================================================
    #                          Save data
    # ==================================================================
    X_grid, T_grid_save = np.meshgrid(np.array(x_plt), np.array(t_plt))
    df_sol = pd.DataFrame({
        "x":     X_grid.ravel(),
        "t":     T_grid_save.ravel(),
        "u_fem": u_fem_plt.ravel(),
        "u_pinn": np.array(u_pinn_plt).ravel(),
        "error": error_plt.ravel(),
    })
    df_sol.to_csv(solution_path, index=False)

    df_losses = pd.DataFrame({"step": np.arange(len(losses)), "loss": np.array(losses)})
    df_losses.to_csv(losses_path, index=False)

    print(f"Saved results to {DATA_DIR}")


# ==================================================================
#                        Surface plots
# ==================================================================
subplot_3d_surfaces(
    figures=[
        {"x": np.array(x_plt), "t": np.array(t_plt), "U": u_fem_plt},
        {"x": np.array(x_plt), "t": np.array(t_plt), "U": np.array(u_pinn_plt)},
        {"x": np.array(x_plt), "t": np.array(t_plt), "U": error_plt},
    ],
    titles=["FEM", "PINN", "Absolute Error"],
    elev=20,
    azims=[45, 45, 45],
    cmap="inferno",
    colorbar_label="u(x, t)",
    suptitle="1D Variable-c Wave Equation",
    show=True,
    savefig=True,
    filepath=str(
        Path(__file__).parent.parent / "figs" / "1d_var" / "solution_surface_1d_var.pdf"
    ),
)
