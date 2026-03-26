import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.nn as jnn
import jax.numpy as jnp

from src.experiment import run_activation_loss_sweep, run_k_loss_sweep, run_c_loss_sweep
from src.plotting import (
    plot_activation_loss_sweep,
    plot_average_max_conditional_number_table,
    plot_loss_curve_multiple_k,
    plot_loss_curve_multiple_c,
)


activation_fns = {
    "tanh": jnn.tanh,
    "sine": jnp.sin,
    "GeLU": jnn.gelu,
    "SiLU": jnn.swish,
    "softmax": jnn.softmax,
}

norms = ["L2", "H1", "H2"]

DATA_DIR = Path(__file__).parent.parent / "data" / "1d_const_losses"
FIG_DIR = Path(__file__).parent.parent / "figs" / "1d_const_losses"


# =============================================================
#       Loss sweep for all activation functions and norms
# =============================================================

results = run_activation_loss_sweep(
    activation_fns=activation_fns,
    norms=norms,
    architecture=(128, 128, 128, 128),
    T=1.0,
    adam_steps=6000,
    lbfgs_steps=4000,
    N_int=3000,
    N_ic=300,
    lr=1e-3,
    lambda_ic=100.0,
    seed=0,
    log_every=100,
    k=1.0,
    early_stopping=True,
    patience=100,
    min_delta=1e-6,
    save_to_csv=False,
    use_pre_computed=True,
    data_dir=str(DATA_DIR),
)

# ---------- Plotting L2, H1, H2 losses for all activation functions ----------
plot_activation_loss_sweep(
    activation_fns=activation_fns,
    data_dir=str(DATA_DIR),
    fig_dir=str(FIG_DIR),
)


# ---------- Printing average max conditional number table ----------
plot_average_max_conditional_number_table(
    activation_fns=activation_fns,
    norms=norms,
    data_dir=str(DATA_DIR)
)


# =============================================================
#            Loss for different k values (L2)
# =============================================================
k_vals = [1., 2., 3., 4., 5.]

_, loss_curves_k = run_k_loss_sweep(k_vals=k_vals,
                                    architecture=(128, 128, 128, 128),
                                    activation=jnn.gelu,
                                    lbfgs_steps=1000,
                                    adam_warmup_steps=2000,
                                    N_int=2000,
                                    N_ic=200,
                                    lr=1e-3,
                                    lambda_ic=100.0,
                                    seed=0,
                                    log_every=100,
)

plot_loss_curve_multiple_k(
    k_vals,
    loss_curves_k,
    savefig=True,
    fig_dir=str(FIG_DIR),
)


# =============================================================
#           Loss for different c values (L2)
# =============================================================
c_vals = [1., 2., 3., 4., 5.]

_, loss_curves_c = run_c_loss_sweep(c_vals=c_vals,
                                    architecture=(128, 128, 128, 128),
                                    activation=jnn.gelu,
                                    lbfgs_steps=1000,
                                    adam_warmup_steps=2000,
                                    N_int=2000,
                                    N_ic=200,
                                    lr=1e-3,
                                    lambda_ic=100.0,
                                    seed=0,
                                    log_every=100,
)

plot_loss_curve_multiple_c(
    c_vals,
    loss_curves_c,
    savefig=True,
    fig_dir=str(FIG_DIR),
)
