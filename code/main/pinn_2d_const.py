import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax
import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from src.pde import fd_solve, fem_solve, u_exact, u_exact_1d, create_grid
from src.pinn import train_pinn
from src.experiment import absolute_error, relative_error, run_architecture_sweep
from src.plotting import (
    plot_3d_surface,
    subplot_3d_surfaces,
    plot_loss,
    plot_loss_components,
    plot_loss_comparison,
    print_optimizer_comparison_tables,
    plot_heatmap_width_depth,
    plot_2d_snapshots,
)




# =============================================================
#            Full architecture sweep for all optimizers,
#            activation functions, widths and depths
# =============================================================
opt_names = ["adam", "adamw"]
hidden_widths = [64, 128, 256]
num_hidden_layers = [3, 4, 5]
activation_fns = {
    "tanh": jnn.tanh,
    "GeLU": jnn.gelu,
    "SiLU": jnn.swish,
}

for opt in opt_names:
    run_architecture_sweep(
        hidden_widths=hidden_widths,
        num_hidden_layers=num_hidden_layers,
        activation_fns=activation_fns,
        T=1.0,
        n_windows=1,
        steps_per_window=5000,
        N_int=1000,
        N_ic=100,
        lambda_ic=100.0,
        lr=1e-3,
        grad_clip=1.0,
        dim=2,
        norm="L2",
        seeds=(0, 7, 103, 42),
        Nx_eval=50,
        Nt_eval=50,
        optimizer=opt,
        lr_schedule="cosine",
        save_to_csv=False,
        use_pre_computed=True,
        data_dir=str(Path(__file__).parent.parent / "data" / "2d_const" / opt),
    )





# =====================================================
#          Printing results from sweep in tables 
# =====================================================
print_optimizer_comparison_tables(
    Path(__file__).parent.parent / "data" / "2d_const"
)





# =============================================================
#   Loss plots: adam vs adamw, GeLU, 4 layers, 256 nodes
# =============================================================
# all_losses = {}
# all_loss_comps = {}

# for opt in ["adam", "adamw"]:
#     _, losses, loss_comps = train_pinn(
#         layers=[3, 256, 256, 256, 256, 1],
#         activations=[jnn.gelu] * 4,
#         steps_per_window=5000,
#         n_windows=1,
#         N_int=1000,
#         N_ic=100,
#         T=1.0,
#         c=1.0,
#         lambda_ic=100.0,
#         lr=1e-3,
#         grad_clip=1.0,
#         dim=2,
#         seed=0,
#         optimizer=opt,
#         lr_schedule="cosine",
#     )
#     all_losses[opt] = losses
#     all_loss_comps[opt] = loss_comps

#     plot_loss_components(
#         loss_comps,
#         show=True,
#         savefig=True,
#         filepath=str(Path(__file__).parent.parent / "figs" / "2d_const" / f"training_loss_components_{opt}.pdf"),
#     )

# plot_loss_comparison(
#     losses_list=[all_losses["adam"], all_losses["adamw"]],
#     labels=["Adam", "AdamW"],
#     title="2D Wave Eq - 4 Layers & 256 Nodes (GeLU)",
#     show=True,
#     savefig=True,
#     filepath=str(Path(__file__).parent.parent / "figs" / "2d_const" / "training_loss_comparison.pdf"),
# )


# =============================================================
#          Single model evaluation (solution + error plots)
#          Best model: AdamW, 4 layers, 256 nodes, GeLU
# =============================================================
Nx_plot = 50
Ny_plot = 50
Nt_plot = 200
c = 1.0
T = 1.0

x_plot = jnp.linspace(0, 1, Nx_plot)
y_plot = jnp.linspace(0, 1, Ny_plot)
t_plot = jnp.linspace(0, T, Nt_plot)

model, _, _ = train_pinn(
    layers=[3, 256, 256, 256, 256, 1],
    activations=[jnn.gelu] * 4,
    steps_per_window=5000,
    n_windows=1,
    N_int=1000,
    N_ic=100,
    T=T,
    c=c,
    lambda_ic=100.0,
    lr=1e-3,
    grad_clip=1.0,
    dim=2,
    seed=0,
    optimizer="adamw",
    lr_schedule="cosine",
)

# Evaluate PINN on grid: u_pinn shape (Nt, Nx, Ny)
X, Y = jnp.meshgrid(x_plot, y_plot, indexing="ij")
XY = jnp.stack([X.ravel(), Y.ravel()], axis=1)

u_pinn_list = []
for ti in t_plot:
    T_grid = jnp.full((XY.shape[0], 1), ti)
    xyt = jnp.concatenate([XY, T_grid], axis=1)
    u_pinn_list.append(model(xyt).squeeze().reshape(Nx_plot, Ny_plot))
u_pinn = jnp.stack(u_pinn_list, axis=0)  # (Nt, Nx, Ny)

# Analytical solution
u_exact_grid = jnp.stack(
    [jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y) * jnp.cos(jnp.sqrt(2.0) * jnp.pi * c * ti)
     for ti in t_plot],
    axis=0,
)  # (Nt, Nx, Ny)

# Time snapshots
t_snap_vals = [0.25, 0.5, 1.0 / float(jnp.sqrt(2)), 1.0]
t_snap_idx = [int(jnp.argmin(jnp.abs(t_plot - tv))) for tv in t_snap_vals]

# ================= Solution plot =================
plot_2d_snapshots(
    u_pinn, t_snap_idx, t_snap_vals,
    title=r"$\mathbf{2D}$ Wave Equation — PINN Solution",
    cmap="inferno",
    savefig=True,
    filepath=str(Path(__file__).parent.parent / "figs" / "2d_const" / "pinn_solution_snapshots_adamw.pdf"),
)

# ================= Error plot =================
error = jnp.abs(u_pinn - u_exact_grid)
plot_2d_snapshots(
    error, t_snap_idx, t_snap_vals,
    title=r"$\mathbf{2D}$ Wave Equation — PINN Absolute Error",
    cmap="inferno",
    savefig=True,
    filepath=str(Path(__file__).parent.parent / "figs" / "2d_const" / "pinn_error_snapshots_adamw.pdf"),
)

