import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.nn as jnn
import jax.numpy as jnp

from src.experiment import run_architecture_sweep_2d


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
    save_to_csv=True,
    use_pre_computed=False,
    data_dir=str(Path(__file__).parent.parent / "data" / "2d_const"),
)


# =====================================================
#          Printing results from sweep in tables
# =====================================================
print(results.to_string(index=False))
