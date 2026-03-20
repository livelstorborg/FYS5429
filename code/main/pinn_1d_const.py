import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.nn as jnn
import jax.numpy as jnp
import matplotlib.pyplot as plt
import time

from src.experiment import run_architecture_sweep
from src.plotting import print_optimizer_comparison_tables, plot_optimizer_comparison
from src.pinn import train_pinn, pack_params

# =============================================================
#            Full architecture sweep for all optimizers,
#            activation functions, widths and depths
# =============================================================
hidden_widths = [32, 64, 128]
num_hidden_layers = [2, 3, 4]
activation_fns = {
    "tanh": jnn.tanh,
    "sine": jnp.sin,
    "GeLU": jnn.gelu,
    "SiLU": jnn.swish,
    "ReLU": jnn.relu,
}

run_architecture_sweep(
    hidden_widths=hidden_widths,
    num_hidden_layers=num_hidden_layers,
    activation_fns=activation_fns,
    T=1.0,
    adam_steps=5000,
    lbfgs_steps=500,
    N_int=1000,
    N_ic=100,
    lambda_ic=100.0,
    lr=1e-3,
    seeds=(0, 7, 103, 42),
    Nx_eval=50,
    Nt_eval=50,
    save_to_csv=False,
    use_pre_computed=True,
    data_dir=str(Path(__file__).parent.parent / "data" / "1d_const"),
)


# ---------- Printing results from sweep in tables ----------
print_optimizer_comparison_tables(
    Path(__file__).parent.parent / "data" / "1d_const"
)





# =====================================================
#          Loss curves for best model
#          4 layers & 128 nodes, GeLU
#          with timing for each optimizer
# =====================================================

t0 = time.perf_counter()
model_adam, losses_adam, _ = train_pinn(
    (128, 128, 128, 128),
    dim=1,
    activation=jnn.gelu,
    optimizer="adam",
    steps=5000,
    N_int=1000,
    N_ic=100,
    T=1.0,
    lambda_ic=100.0,
    lr=1e-3,
    seed=0,
    log_every=500,
)
print(f"Adam training time: {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
model_lbfgs, losses_lbfgs_cold, _ = train_pinn(
    (128, 128, 128, 128),
    dim=1,
    activation=jnn.gelu,
    optimizer="lbfgs",
    steps=5000,
    adam_warmup_steps=0,
    N_int=1000,
    N_ic=100,
    T=1.0,
    lambda_ic=100.0,
    lr=1e-3,
    seed=0,
    log_every=500,
)
print(f"L-BFGS training time: {time.perf_counter() - t0:.1f}s")

t0 = time.perf_counter()
model_lbfgs, losses_lbfgs_warm, _ = train_pinn(
    (128, 128, 128, 128),
    dim=1,
    activation=jnn.gelu,
    optimizer="lbfgs",
    steps=3000,
    adam_warmup_steps=2000,
    N_int=1000,
    N_ic=100,
    T=1.0,
    lambda_ic=100.0,
    lr=1e-3,
    seed=0,
    init_params=pack_params(model_adam),
    log_every=100,
)
print(f"Adam + L-BFGS training time: {time.perf_counter() - t0:.1f}s")

plot_optimizer_comparison(
    losses_adam=losses_adam,
    losses_lbfgs_warm=losses_lbfgs_warm,
    losses_lbfgs_cold=losses_lbfgs_cold,
    show=True,
    savefig=True,
    fig_dir=str(Path(__file__).parent.parent / "figs" / "1d_const"),
)