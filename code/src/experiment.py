import os
import time
from pathlib import Path

import pandas as pd
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

try:
    from src.pde import u_exact
    from src.pinn import train_pinn, pack_params
    from src.utils import compute_error_metrics_1d, compute_error_metrics_2d, weight_condition_numbers
except ModuleNotFoundError:
    from pde import u_exact
    from pinn import train_pinn, pack_params
    from utils import compute_error_metrics_1d, compute_error_metrics_2d, weight_condition_numbers


# ---------- Utils ) ----------
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
                    model_adam, _, _ = train_pinn(
                        widths,
                        dim=1,
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
                    model_lbfgs, _, _ = train_pinn(
                        widths,
                        dim=1,
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


# ---------- 2D architecture sweep (adam → lbfgs only) ----------
def run_architecture_sweep_2d(
    hidden_widths,
    num_hidden_layers,
    activation_fns,
    *,
    T=1.0,
    N_int=2000,
    N_ic=200,
    lambda_ic=100.0,
    lr=1e-3,
    c=1.0,
    adam_steps=2000,
    lbfgs_steps=3000,
    seeds=(0, 7, 103, 42),
    Nx_eval=25,
    Ny_eval=25,
    Nt_eval=25,
    save_to_csv=False,
    use_pre_computed=False,
    data_dir="data",
):
    """
    Architecture sweep for the 2D wave PINN using Adam→L-BFGS as one combined optimizer.

    For each architecture and seed: trains Adam for `adam_steps`, then
    warm-starts L-BFGS for `lbfgs_steps`. Only the final model is saved.

    Returns a DataFrame with columns: activation, hidden_layers, width, L2_rel_mean, Linf_mean.
    """
    if use_pre_computed:
        print(f"Loading pre-computed results from {data_dir} ...")
        act_names = list(activation_fns.keys())
        return load_sweep_results_from_csv(data_dir=data_dir, activation_fns=act_names)

    print(f"========== 2D ARCHITECTURE SWEEP ==========\n adam({adam_steps} steps) → lbfgs({lbfgs_steps} steps)")
    if save_to_csv:
        os.makedirs(data_dir, exist_ok=True)

    x_eval = jnp.linspace(0.0, 1.0, Nx_eval)
    y_eval = jnp.linspace(0.0, 1.0, Ny_eval)
    t_eval = jnp.linspace(0.0, T, Nt_eval)

    all_results = []
    total_start = time.time()

    for act_name, act_fn in activation_fns.items():
        print(f"\n---------- {act_name} ----------")
        act_start = time.time()

        for depth in num_hidden_layers:
            for width in hidden_widths:
                widths = tuple([width] * depth)
                print(f"  Layers={depth}, Width={width}")

                arch_rows = []
                L2_all    = []
                Linf_all  = []

                for seed in seeds:
                    print(f"    SEED={seed}")

                    _, model_final, _ = train_pinn(
                        widths,
                        dim=2,
                        activation=act_fn,
                        adam_steps=adam_steps,
                        lbfgs_steps=lbfgs_steps,
                        N_int=N_int,
                        N_ic=N_ic,
                        T=T,
                        c=c,
                        lambda_ic=lambda_ic,
                        lr=lr,
                        seed=seed,
                        log_every=max(adam_steps, lbfgs_steps),
                    )

                    L2, Linf, _, _, _ = compute_error_metrics_2d(
                        model_final, x=x_eval, y=y_eval, t=t_eval, c=c
                    )
                    print(f"    L2={float(L2):.3e}  Linf={float(Linf):.3e}")
                    L2_all.append(L2)
                    Linf_all.append(Linf)
                    if save_to_csv:
                        arch_rows.append({
                            "activation": act_name, "hidden_layers": depth,
                            "width": width, "seed": seed,
                            "L2_rel": float(L2), "Linf": float(Linf),
                        })

                if save_to_csv:
                    fname = os.path.join(data_dir, f"{act_name}_L{depth}_N{width}.csv")
                    pd.DataFrame(arch_rows).to_csv(fname, index=False)
                    print(f"    saved {fname}")

                all_results.append({
                    "activation":    act_name,
                    "hidden_layers": depth,
                    "width":         width,
                    "L2_rel_mean":   float(np.mean(L2_all)),
                    "Linf_mean":     float(np.mean(Linf_all)),
                })

        print(f"  {act_name} done in {time.time()-act_start:.1f}s  "
              f"(total {time.time()-total_start:.1f}s)\n")

    return pd.DataFrame(all_results)



# ----------- Activation function sweep (1D, constant losses) -----------
def run_activation_loss_sweep(
        activation_fns,
        norms,
        architecture=(128, 128, 128, 128),
        T=1.0,
        adam_steps=2000,
        lbfgs_steps=1000,
        N_int=1000,
        N_ic=100,
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
        data_dir=str(Path(__file__).parent.parent / "data" / "1d_const_losses"),
):

    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for act_name, act_fn in activation_fns.items():
        print(f"\n=== Activation: {act_name} ===")
        csv_path = data_dir / f"{act_name}.csv"

        if use_pre_computed and csv_path.exists():
            print(f"  Found cached CSV, loading ...")
            df = pd.read_csv(csv_path)
            curves = {
                "L2": df["L2_loss"].tolist(),
                "H1": df["H1_loss"].tolist(),
                "H2": df["H2_loss"].tolist(),
            }
            cond_numbers = {}
            for n in norms:
                col = f"max_cond_{n}"
                if col in df.columns:
                    cond_numbers[n] = df[col].iloc[0]
                    print(f"  max weight cond [{n}]: {cond_numbers[n]:.2e}")
        else:
            curves = {}
            cond_numbers = {}

            for n in norms:
                print(f"  norm={n} ...")
                model, losses, _ = train_pinn(
                    architecture,
                    dim=1,
                    activation=act_fn,
                    optimizer="lbfgs",
                    steps=lbfgs_steps,
                    adam_warmup_steps=adam_steps,
                    N_int=N_int,
                    N_ic=N_ic,
                    T=T,
                    lambda_ic=lambda_ic,
                    lr=lr,
                    seed=seed,
                    log_every=log_every,
                    k=k,
                    norm=n,
                    early_stopping=early_stopping,
                    patience=patience,
                    min_delta=min_delta,
                )
                curves[n] = losses
                conds = weight_condition_numbers(model)
                cond_numbers[n] = max(conds)
                print(f"    final loss = {losses[-1]:.3e}")
                print(f"    weight cond numbers: {[f'{c:.2e}' for c in conds]}  (max={cond_numbers[n]:.2e})")

            if save_to_csv:
                min_len = min(len(curves[n]) for n in norms)
                df = pd.DataFrame({
                    "step": np.arange(min_len),
                    "L2_loss": np.array(curves["L2"])[:min_len],
                    "H1_loss": np.array(curves["H1"])[:min_len],
                    "H2_loss": np.array(curves["H2"])[:min_len],
                    "max_cond_L2": cond_numbers["L2"],
                    "max_cond_H1": cond_numbers["H1"],
                    "max_cond_H2": cond_numbers["H2"],
                })
                df.to_csv(csv_path, index=False)
                print(f"  Saved CSV: {act_name}.csv")

        all_results[act_name] = {"curves": curves, "cond_numbers": cond_numbers}

    return all_results


# ----------- k sweep (1D, constant, L2 loss) -----------
# ----------- c sweep (1D, constant, L2 loss) -----------
def run_c_loss_sweep(
        c_vals,
        architecture=(64, 64, 64),
        activation=jnn.gelu,
        lbfgs_steps=1500,
        adam_warmup_steps=1000,
        N_int=1000,
        N_ic=100,
        T=1.0,
        lambda_ic=100.0,
        lr=1e-3,
        seed=0,
        log_every=500,
):
    """
    Train a PINN for each c in c_vals and return loss curves as a DataFrame.

    Returns
    -------
    pd.DataFrame with columns: c, step, loss
    list of loss curves, one per c
    """
    rows = []
    loss_curves = []
    for c in c_vals:
        print(f"\n--- c={int(c)} ---")
        _, losses, _ = train_pinn(
            architecture,
            dim=1,
            activation=activation,
            optimizer="lbfgs",
            steps=lbfgs_steps,
            adam_warmup_steps=adam_warmup_steps,
            N_int=N_int,
            N_ic=N_ic,
            T=T,
            lambda_ic=lambda_ic,
            lr=lr,
            seed=seed,
            log_every=log_every,
            c=c,
        )
        loss_curves.append(losses)
        print(f"  final loss = {losses[-1]:.3e}")
        for step, loss in enumerate(losses):
            rows.append({"c": c, "step": step, "loss": loss})

    return pd.DataFrame(rows), loss_curves


# ----------- k sweep (1D, constant, L2 loss) -----------
def run_k_loss_sweep(
        k_vals,
        architecture=(64, 64, 64),
        activation=jnn.gelu,
        lbfgs_steps=1500,
        adam_warmup_steps=1000,
        N_int=1000,
        N_ic=100,
        T=1.0,
        lambda_ic=100.0,
        lr=1e-3,
        seed=0,
        log_every=500,
):
    """
    Train a PINN for each k in k_vals and return loss curves as a DataFrame.

    Returns
    -------
    pd.DataFrame with columns: k, step, loss
    """
    rows = []
    loss_curves = []
    for k in k_vals:
        print(f"\n--- k={int(k)} ---")
        _, losses, _ = train_pinn(
            architecture,
            dim=1,
            activation=activation,
            optimizer="lbfgs",
            steps=lbfgs_steps,
            adam_warmup_steps=adam_warmup_steps,
            N_int=N_int,
            N_ic=N_ic,
            T=T,
            lambda_ic=lambda_ic,
            lr=lr,
            seed=seed,
            log_every=log_every,
            k=k,
        )
        loss_curves.append(losses)
        print(f"  final loss = {losses[-1]:.3e}")
        for step, loss in enumerate(losses):
            rows.append({"k": k, "step": step, "loss": loss})

    return pd.DataFrame(rows), loss_curves


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
