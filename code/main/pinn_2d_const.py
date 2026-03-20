import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.nn as jnn
import jax.numpy as jnp

from src.experiment import run_architecture_sweep_2d
from src.pde import create_grid, fem_solve, u_exact
from src.plotting import plot_2d_snapshots
from src.pinn import train_pinn


# ======================================================================
#     2D architecture sweep: adam (2000 steps) → lbfgs (3000 steps)
# ======================================================================
hidden_widths = [64, 128, 256]
num_hidden_layers = [3, 4, 5]
activation_fns = {
    "tanh": jnn.tanh,
    "GeLU": jnn.gelu,
    "SiLU": jnn.swish,
}

results = run_architecture_sweep_2d(
    hidden_widths=hidden_widths,
    num_hidden_layers=num_hidden_layers,
    activation_fns=activation_fns,
    T=1.0,
    adam_steps=2000,
    lbfgs_steps=3000,
    N_int=1000,
    N_ic=100,
    lambda_ic=100.0,
    lr=1e-3,
    seeds=(0, 7, 103, 42),
    Nx_eval=50,
    Ny_eval=50,
    Nt_eval=50,
    save_to_csv=False,
    use_pre_computed=True,
    data_dir=str(Path(__file__).parent.parent / "data" / "2d_const"),
)


# =====================================================
#          Printing results from sweep in tables
# =====================================================
print("\nArchitecture sweep results (2D constant c):")
print(results.to_string(index=False))



# ======================================================================
#         Plot for best model 
#         4 layers, 128 nodes and GeLU activation
# ======================================================================
T = 1.0
c0 = 1.0

# Grid sizes 
Nx = 100   
Ny = 100
Nt = 50

x, y, t, dx, dy, dt = create_grid(Nx, Ny, T=T, dim=2)
u_ex = u_exact(x, t, y=y, c=c0, dim=2)

snap_times = [0.25, 0.5, 1/jnp.sqrt(2), 1]
snap_idx_const = [int(jnp.argmin(jnp.abs(t - ts))) for ts in snap_times]

_, model, _ = train_pinn(
    (128, 128, 128, 128),
    dim=2,
    activation=jnn.gelu,
    adam_steps=2000,
    lbfgs_steps=3000,
    N_int=3000,
    N_ic=300,
    T=T,
    c=c0,
    lambda_ic=100.0,
    lr=1e-3,
    seed=0,
    log_every=500,
)

# Evaluate model on the 2D grid at each snapshot time
X_plot, Y_plot = jnp.meshgrid(x, y, indexing="ij")
XY_flat = jnp.stack([X_plot.ravel(), Y_plot.ravel()], axis=1)

u_pinn_snaps = []
u_error_snaps = []
for idx in snap_idx_const:
    ti = float(t[idx])
    T_col = jnp.full((XY_flat.shape[0], 1), ti)
    xyt = jnp.concatenate([XY_flat, T_col], axis=1)
    u_pred = model(xyt).reshape(len(x), len(y))
    u_pinn_snaps.append(u_pred)
    u_error_snaps.append(jnp.abs(u_pred - u_ex[idx]))

u_pinn_snaps = jnp.stack(u_pinn_snaps)   # (4, Nx+1, Ny+1)
u_error_snaps = jnp.stack(u_error_snaps) # (4, Nx+1, Ny+1)
snap_idx_plot = list(range(len(snap_idx_const)))  # [0, 1, 2, 3]

# ---------- Solutions plots -----------
plot_2d_snapshots(
    u_pinn_snaps,
    t_indices=snap_idx_plot,
    t_labels=snap_times,
    title=r"Pinn Solution for 2D Wave Equation with $\mathbf{c_{constant}}$",
    cmap="inferno",
    savefig=True,
    filepath=str("figs/" + "2d_const/" +  "pinn_solution_2d_const.pdf"),
)



# ---------- Error plots -----------

plot_2d_snapshots(
    u_error_snaps,
    t_indices=snap_idx_plot,
    t_labels=snap_times,
    title=r"2D Absolute Error for PINN with $\mathbf{c_{constant}}$",
    cmap="inferno",
    savefig=True,
    filepath=str("figs/" + "2d_const/" +  "pinn_error_heatmaps_2d_const.pdf"),
)