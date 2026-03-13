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
    plot_solution_at_t,
    plot_scheme_errors_at_t,
    plot_3d_surface,
    subplot_3d_surfaces,
    plot_loss,
    plot_loss_components,
    print_optimizer_comparison_tables,
    plot_heatmap_width_depth,
)


# =============================================================
#            Full archecture sweep for all optimizers, 
#            activation functinos, widths and depths
# =============================================================
opt_names = ["adam", "adamw", "lbfgs"]
hidden_widths = [32, 64, 128]
num_hidden_layers = [2, 3, 4]
activation_fns = {
    "tanh": jnn.tanh,
    "sine": jnp.sin,
    "GeLU": jnn.gelu,
    "SiLU": jnn.swish,
    "ReLU": jnn.relu,
}

for opt in opt_names:
    run_architecture_sweep(
        hidden_widths=hidden_widths,
        num_hidden_layers=num_hidden_layers,
        activation_fns=activation_fns,
        T=1.0,
        n_windows=1,
        steps_per_window=2000,
        N_int=500,
        N_ic=100,
        lambda_ic=100.0,
        lr=1e-3,
        grad_clip=1.0,
        dim=1,
        norm="L2",
        seeds=(0, 42, 7, 103, 73),
        Nx_eval=50,
        Nt_eval=50,
        optimizer=opt,
        save_to_csv=True,
        use_pre_computed=False,
        data_dir=str(Path(__file__).parent.parent / "data" / "1d_const" / opt),
    )


# =====================================================
#          Heatmaps for width vs depth for sweep 
# =====================================================
for opt in opt_names:
    df = run_architecture_sweep(
        hidden_widths=hidden_widths,
        num_hidden_layers=num_hidden_layers,
        activation_fns=activation_fns,
        use_pre_computed=True,
        data_dir=str(Path(__file__).parent.parent / "data" / "1d_const" / opt),
    )

    for act in ["tanh", "sine", "GeLU", "SiLU", "ReLU"]:
        plot_heatmap_width_depth(
            df,
            activation=act,
            show=True,
            savefig=True,
            filepath=str(Path(__file__).parent.parent / "figs" / "1d_const" / f"heatmap_{opt}_{act}.pdf"),
        )


# =====================================================
#          Printing results from sweep in tables 
# =====================================================
print_optimizer_comparison_tables(
    Path(__file__).parent.parent / "data" / "1d_const"
)








# =====================================================
#            Loss for single model
#            will use this for the best model (maybe)
# =====================================================
"""
Experiment for testing, using windows=1 for the simple 1d constant case 
    - windows=1 gives no spikes when switching windows 
    - for time-marchin to be reasonable, we need to train sufficiently long, so that the error isnt accumulating
    - when doing a sweep we cant let each model train that long, and time-marching isnt the way to go
"""


# # --- Parameters ---
# c           = 1.0
# L           = 1.0
# T_final     = 1.0
# n_windows   = 1
# steps_per_window = 10000
# N_int       = 2000
# N_ic        = 200
# lambda_ic   = 100.0
# lr          = 1e-3
# seed        = 0

# # --- Train ---
# model, losses, loss_comps = train_pinn(
#     dim=1,
#     layers=[2, 64, 64, 64, 1],
#     activations=[jax.nn.gelu] * 3,
#     steps=n_windows * steps_per_window,
#     n_windows=n_windows,
#     steps_per_window=steps_per_window,
#     N_int=N_int,
#     N_ic=N_ic,
#     T=T_final,
#     L=L,
#     c=c,
#     lambda_ic=lambda_ic,
#     lr=lr,
#     seed=seed,
#     grad_clip=1.0,
# )

# # --- 3D surface plots ---
# Nx_plot, Nt_plot = 100, 100
# x_plot = jnp.linspace(0.0, L, Nx_plot)
# t_plot = jnp.linspace(0.0, T_final, Nt_plot)

# # Evaluate PINN on grid: U shape (Nt, Nx)
# X_grid, T_grid = jnp.meshgrid(x_plot, t_plot)         
# inp = jnp.stack([X_grid.ravel(), T_grid.ravel()], axis=1)
# U_pinn = model(inp).reshape(Nt_plot, Nx_plot)

# U_exact = u_exact_1d(x_plot, t_plot, c=c)
# U_abs_error = jnp.abs(U_pinn - U_exact)
# U_rel_error = U_abs_error / (jnp.abs(U_exact) + 1e-8)

# subplot_fig = subplot_3d_surfaces(
#     figures=[
#         {"x": x_plot, "t": t_plot, "U": U_exact},
#         {"x": x_plot, "t": t_plot, "U": U_pinn},
#         {"x": x_plot, "t": t_plot, "U": U_abs_error},
#         {"x": x_plot, "t": t_plot, "U": U_rel_error},
#     ],
#     titles=["Analytical", "PINN", "Absolute Error", "Relative Error"],
#     elev=20,
#     azims=[45, 45, 45, 45],
#     cmap="viridis",
#     colorbar_label="u(x, t)",
#     suptitle="Wave Equation Solutions — 1D PINN",
#     show=True,
#     savefig=True,
#     filepath=str(Path(__file__).parent.parent / "figs" / "1d_const" / "solution_surface_pinn_1d.pdf"),
# )



# # --- Training loss plots ---
# plot_loss(
#     losses,
#     show=True,
#     savefig=True,
#     filepath=str(Path(__file__).parent.parent / "figs" / "1d_const" / "training_loss_pinn_1d_const.pdf"),
# )
# plot_loss_components(
#     loss_comps,
#     show=True,
#     savefig=True,
#     filepath=str(Path(__file__).parent.parent / "figs" / "1d_const" / "training_loss_components_pinn_1d_const.pdf"),
# )

