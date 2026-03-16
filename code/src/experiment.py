import os
import time
import pandas as pd
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

try:
    from src.pde import u_exact
    from src.pinn import train_wave_pinn, pack_params
    from src.utils import compute_error_metrics_1d, compute_error_metrics_2d
except ModuleNotFoundError:
    from pde import u_exact
    from pinn import train_wave_pinn, pack_params
    from utils import compute_error_metrics_1d, compute_error_metrics_2d


# ---------- Utils (might delete) ----------
def absolute_error(u_num, u_true):
    return np.abs(u_num - u_true)


def relative_error(u_num, u_true, eps=1e-8):
    return np.abs(u_num - u_true) / (np.abs(u_true) + eps)


# ---------- Width/depth sweep ----------
def run_architecture_sweep(
    hidden_widths,
    num_hidden_layers,
    activation_fns,
    *,
    T=1.0,
    N_int=1000,
    N_ic=100,
    lambda_ic=100.0,
    lr=1e-3,
    c=1.0,
    adam_steps=5000,
    lbfgs_steps=500,
    seeds=(0, 7, 103, 42),
    Nx_eval=50,
    Nt_eval=50,
    save_to_csv=False,
    use_pre_computed=False,
    data_dir="data",
):
    """
    Architecture sweep over hidden widths, depths, and activation functions.

    For each architecture and seed: trains Adam, then warm-starts L-BFGS from
    Adam's converged params.  Both optimizers' results are saved together.

    Returns a dict {"adam": DataFrame, "lbfgs": DataFrame}, each with columns:
        activation, hidden_layers, width, L2_rel_mean, Linf_mean
    """
    if use_pre_computed:
        print(f"Loading pre-computed results from {data_dir} ...")
        act_names = list(activation_fns.keys())
        return {
            opt: load_sweep_results_from_csv(
                data_dir=os.path.join(data_dir, opt), activation_fns=act_names
            )
            for opt in ("adam", "lbfgs")
        }

    print(f"========== ARCHITECTURE SWEEP ==========\n adam({adam_steps} steps) → lbfgs({lbfgs_steps} steps)")
    if save_to_csv:
        for opt in ("adam", "lbfgs"):
            os.makedirs(os.path.join(data_dir, opt), exist_ok=True)

    x_eval = jnp.linspace(0.0, 1.0, Nx_eval)
    t_eval = jnp.linspace(0.0, T, Nt_eval)

    all_results = {"adam": [], "lbfgs": []}
    total_start = time.time()

    for act_name, act_fn in activation_fns.items():
        print(f"\n---------- {act_name} ----------")
        act_start = time.time()

        for depth in num_hidden_layers:
            for width in hidden_widths:
                widths = tuple([width] * depth)
                print(f"  Layers={depth}, Width={width}")

                arch_rows = {"adam": [], "lbfgs": []}
                L2_all    = {"adam": [], "lbfgs": []}
                Linf_all  = {"adam": [], "lbfgs": []}

                for seed in seeds:
                    print(f"SEED={seed}\n")

                    # --- Adam ---
                    model_adam, _, _ = train_wave_pinn(
                        widths=widths,
                        activation=act_fn,
                        optimizer="adam",
                        steps=adam_steps,
                        N_int=N_int,
                        N_ic=N_ic,
                        T=T,
                        c=c,
                        lambda_ic=lambda_ic,
                        lr=lr,
                        seed=seed,
                        adam_warmup_steps=0,
                        log_every=adam_steps,
                    )
                    L2, Linf, _, _, _ = compute_error_metrics_1d(
                        model_adam, x=x_eval, t=t_eval, c=c
                    )
                    print(f"Adam: Relative L2={float(L2):.3e}  Linf={float(Linf):.3e}")
                    L2_all["adam"].append(L2)
                    Linf_all["adam"].append(Linf)
                    if save_to_csv:
                        arch_rows["adam"].append({
                            "activation": act_name, "hidden_layers": depth,
                            "width": width, "seed": seed,
                            "L2_rel": float(L2), "Linf": float(Linf),
                        })

                    # --- L-BFGS warm-started from Adam ---
                    model_lbfgs, _, _ = train_wave_pinn(
                        widths=widths,
                        activation=act_fn,
                        optimizer="lbfgs",
                        steps=lbfgs_steps,
                        N_int=N_int,
                        N_ic=N_ic,
                        T=T,
                        c=c,
                        lambda_ic=lambda_ic,
                        lr=lr,
                        seed=seed,
                        adam_warmup_steps=0,
                        init_params=pack_params(model_adam),
                        log_every=lbfgs_steps,
                    )
                    L2, Linf, _, _, _ = compute_error_metrics_1d(
                        model_lbfgs, x=x_eval, t=t_eval, c=c
                    )
                    print(f"L-BFGS: Relative L2={float(L2):.3e}  Linf={float(Linf):.3e}")
                    L2_all["lbfgs"].append(L2)
                    Linf_all["lbfgs"].append(Linf)
                    if save_to_csv:
                        arch_rows["lbfgs"].append({
                            "activation": act_name, "hidden_layers": depth,
                            "width": width, "seed": seed,
                            "L2_rel": float(L2), "Linf": float(Linf),
                        })

                if save_to_csv:
                    for opt in ("adam", "lbfgs"):
                        fname = os.path.join(
                            data_dir, opt, f"{act_name}_L{depth}_N{width}.csv"
                        )
                        pd.DataFrame(arch_rows[opt]).to_csv(fname, index=False)
                        print(f"    saved {fname}")

                for opt in ("adam", "lbfgs"):
                    all_results[opt].append({
                        "activation":    act_name,
                        "hidden_layers": depth,
                        "width":         width,
                        "L2_rel_mean":   float(np.mean(L2_all[opt])),
                        "Linf_mean":     float(np.mean(Linf_all[opt])),
                    })

        print(f"  {act_name} done in {time.time()-act_start:.1f}s  "
              f"(total {time.time()-total_start:.1f}s)\n")

    return {opt: pd.DataFrame(rows) for opt, rows in all_results.items()}


# ---------- Learningrate sweep ----------
def run_learning_rate_sweep():
    pass


def load_sweep_results_from_csv(data_dir="../data", activation_fns=None):
    """
    Load sweep results from CSV files in data_dir.

    Parameters:
    -----------
    data_dir : str
        Directory containing the CSV files
    activation_fns : list[str] or None
        List of activation function names to load.
        If None, loads all CSV files matching pattern 'sweep_*.csv'

    Returns:
    --------
    pd.DataFrame
        Aggregated results (means over seeds), same format as
        run_architecture_sweep returns.
    """

    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found")

    # Collect all matching CSV files: {act_name}_L{L}_N{W}.csv
    all_files = [f for f in os.listdir(data_dir) if f.endswith(".csv")]
    if activation_fns is not None:
        all_files = [f for f in all_files if any(f.startswith(act + "_") for act in activation_fns)]

    if not all_files:
        raise FileNotFoundError(f"No sweep CSV files found in '{data_dir}'")

    # Load and concatenate all dataframes
    dfs = []
    for csv_file in sorted(all_files):
        filepath = os.path.join(data_dir, csv_file)
        df = pd.read_csv(filepath)
        dfs.append(df)
        print(f"Loaded {filepath}")

    if not dfs:
        raise FileNotFoundError("No valid CSV files could be loaded")

    # Concatenate all data
    all_data = pd.concat(dfs, ignore_index=True)

    # Aggregate by taking mean over seeds
    aggregated = all_data.groupby(
        ["activation", "hidden_layers", "width"], as_index=False
    ).agg({"L2_rel": "mean", "Linf": "mean"})

    # Rename columns to match original format
    aggregated = aggregated.rename(
        columns={"L2_rel": "L2_rel_mean", "Linf": "Linf_mean"}
    )

    print(f"\nLoaded {len(all_data)} individual results")
    print(f"Aggregated to {len(aggregated)} configurations")

    return aggregated
