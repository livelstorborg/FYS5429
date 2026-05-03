import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt

# from src.experiment import run_architecture_sweep, run_k_loss_sweep, run_c_loss_sweep
from src.plotting import (
    # print_optimizer_comparison_tables,
    # plot_optimizer_comparison,
    # plot_loss_curve_multiple_k,
    # plot_loss_curve_multiple_c,
    plot_ck_norm_comparison,
)
from src.pinn import train_pinn, pack_params

# # =============================================================
# #            Full architecture sweep for all optimizers,
# #            activation functions, widths and depths
# # =============================================================
# hidden_widths = [32, 64, 128]
# num_hidden_layers = [2, 3, 4]
# activation_fns = {
#     "tanh": jnn.tanh,
#     "sine": jnp.sin,
#     "GeLU": jnn.gelu,
#     "SiLU": jnn.swish,
#     "ReLU": jnn.relu,
# }
#
# run_architecture_sweep(
#     hidden_widths=hidden_widths,
#     num_hidden_layers=num_hidden_layers,
#     activation_fns=activation_fns,
#     T=1.0,
#     adam_steps=5000,
#     lbfgs_steps=500,
#     N_int=1000,
#     N_ic=100,
#     lambda_ic=100.0,
#     lr=1e-3,
#     seeds=(0, 7, 103, 42),
#     Nx_eval=50,
#     Nt_eval=50,
#     save_to_csv=False,
#     use_pre_computed=True,
#     data_dir=str(Path(__file__).parent.parent / "data" / "1d_const"),
# )
#
#
# # ---------- Printing results from sweep in tables ----------
# print_optimizer_comparison_tables(
#     Path(__file__).parent.parent / "data" / "1d_const"
# )


# # =====================================================
# #          Loss curves for best model
# #          4 layers & 128 nodes, GeLU
# #          with timing for each optimizer
# # =====================================================
#
# import time
#
# t0 = time.perf_counter()
# model_adam, losses_adam, _ = train_pinn(
#     (128, 128, 128, 128),
#     dim=1,
#     activation=jnn.gelu,
#     optimizer="adam",
#     steps=5000,
#     N_int=1000,
#     N_ic=100,
#     T=1.0,
#     lambda_ic=100.0,
#     lr=1e-3,
#     seed=0,
#     log_every=500,
# )
# print(f"Adam training time: {time.perf_counter() - t0:.1f}s")
#
# t0 = time.perf_counter()
# model_lbfgs, losses_lbfgs_cold, _ = train_pinn(
#     (128, 128, 128, 128),
#     dim=1,
#     activation=jnn.gelu,
#     optimizer="lbfgs",
#     steps=5000,
#     adam_warmup_steps=0,
#     N_int=1000,
#     N_ic=100,
#     T=1.0,
#     lambda_ic=100.0,
#     lr=1e-3,
#     seed=0,
#     log_every=500,
# )
# print(f"L-BFGS training time: {time.perf_counter() - t0:.1f}s")
#
# t0 = time.perf_counter()
# model_lbfgs, losses_lbfgs_warm, _ = train_pinn(
#     (128, 128, 128, 128),
#     dim=1,
#     activation=jnn.gelu,
#     optimizer="lbfgs",
#     steps=3000,
#     adam_warmup_steps=2000,
#     N_int=1000,
#     N_ic=100,
#     T=1.0,
#     lambda_ic=100.0,
#     lr=1e-3,
#     seed=0,
#     init_params=pack_params(model_adam),
#     log_every=100,
# )
# print(f"Adam + L-BFGS training time: {time.perf_counter() - t0:.1f}s")
#
# plot_optimizer_comparison(
#     losses_adam=losses_adam,
#     losses_lbfgs_warm=losses_lbfgs_warm,
#     losses_lbfgs_cold=losses_lbfgs_cold,
#     show=True,
#     savefig=True,
#     fig_dir=str(Path(__file__).parent.parent / "figs" / "1d_const"),
# )


DATA_DIR = Path(__file__).parent.parent / "data" / "1d_const_losses"
FIG_DIR = Path(__file__).parent.parent / "figs" / "1d_const_losses"

# # =============================================================
# #            Loss for different k values (L2)
# # =============================================================
# k_vals = [1., 2., 3., 4., 5.]
#
# _, loss_curves_k = run_k_loss_sweep(k_vals=k_vals,
#                                     architecture=(128, 128, 128, 128),
#                                     activation=jnn.gelu,
#                                     lbfgs_steps=1000,
#                                     adam_warmup_steps=2000,
#                                     N_int=2000,
#                                     N_ic=200,
#                                     lr=1e-3,
#                                     lambda_ic=100.0,
#                                     seed=0,
#                                     log_every=100,
# )
#
# plot_loss_curve_multiple_k(
#     k_vals,
#     loss_curves_k,
#     savefig=True,
#     fig_dir=str(FIG_DIR),
# )
#
#
# # =============================================================
# #           Loss for different c values (L2)
# # =============================================================
# c_vals = [1., 2., 3., 4., 5.]
#
# _, loss_curves_c = run_c_loss_sweep(c_vals=c_vals,
#                                     architecture=(128, 128, 128, 128),
#                                     activation=jnn.gelu,
#                                     lbfgs_steps=1000,
#                                     adam_warmup_steps=2000,
#                                     N_int=2000,
#                                     N_ic=200,
#                                     lr=1e-3,
#                                     lambda_ic=100.0,
#                                     seed=0,
#                                     log_every=100,
# )
#
# plot_loss_curve_multiple_c(
#     c_vals,
#     loss_curves_c,
#     savefig=True,
#     fig_dir=str(FIG_DIR),
# )


# =============================================================
#    Norm comparison (L2 / H1 / H2) for (c=1, k=5) and (c=5, k=1)
#    GeLU, 4 layers, 128 nodes
# =============================================================
norms = ["L2", "H1", "H2"]
_ck_train_kwargs = dict(
    architecture=(128, 128, 128, 128),
    activation=jnn.gelu,
    lbfgs_steps=3000,
    adam_warmup_steps=6000,
    N_int=2000,
    N_ic=200,
    lr=1e-3,
    lambda_ic=100.0,
    seed=0,
    log_every=100,
)

curves_c1k5 = {}
for norm in norms:
    _, losses, _ = train_pinn(
        _ck_train_kwargs["architecture"],
        dim=1,
        activation=_ck_train_kwargs["activation"],
        optimizer="lbfgs",
        steps=_ck_train_kwargs["lbfgs_steps"],
        adam_warmup_steps=_ck_train_kwargs["adam_warmup_steps"],
        N_int=_ck_train_kwargs["N_int"],
        N_ic=_ck_train_kwargs["N_ic"],
        T=1.0,
        lambda_ic=_ck_train_kwargs["lambda_ic"],
        lr=_ck_train_kwargs["lr"],
        seed=_ck_train_kwargs["seed"],
        log_every=_ck_train_kwargs["log_every"],
        c=1.0,
        k=5,
        norm=norm,
    )
    curves_c1k5[norm] = losses

curves_c5k1 = {}
for norm in norms:
    _, losses, _ = train_pinn(
        _ck_train_kwargs["architecture"],
        dim=1,
        activation=_ck_train_kwargs["activation"],
        optimizer="lbfgs",
        steps=_ck_train_kwargs["lbfgs_steps"],
        adam_warmup_steps=_ck_train_kwargs["adam_warmup_steps"],
        N_int=_ck_train_kwargs["N_int"],
        N_ic=_ck_train_kwargs["N_ic"],
        T=1.0,
        lambda_ic=_ck_train_kwargs["lambda_ic"],
        lr=_ck_train_kwargs["lr"],
        seed=_ck_train_kwargs["seed"],
        log_every=_ck_train_kwargs["log_every"],
        c=5.0,
        k=1,
        norm=norm,
    )
    curves_c5k1[norm] = losses

plot_ck_norm_comparison(
    curves_c1k5,
    curves_c5k1,
    savefig=True,
    fig_dir=str(FIG_DIR),
)
