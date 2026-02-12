import os
import time
import pandas as pd
import jax
import jax.nn as jnn
import jax.numpy as jnp
import numpy as np

from src.pde import fd_solve, u_exact
from src.pinn import train_pinn, compute_error_metrics


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
    T=0.5,
    steps=5000,
    N_int=1000,
    lr=1e-3,
    seeds=(0,),
    Nx_eval=100,
    Ny_eval=None,
    Nt_eval=100,
    optimizer='adam',
    save_to_csv=False,
    use_pre_computed=False,
    data_dir="data",
):
    """
    Run a general architecture sweep over:
      - hidden_widths:        list[int]
      - num_hidden_layers:    list[int]
      - activation_fns:       dict[str, callable]

    Optional keyword arguments allow further control.
    If use_pre_computed=True, loads results from existing CSV files.
    If save_to_csv=True (and use_pre_computed=False), saves individual
    seed results to CSV files (one per architecture: activation_L{layers}_N{width}.csv) 
    in data_dir.

    Returns:
      A DataFrame with aggregated results (means over seeds).
    """

    if use_pre_computed:
        print("Loading pre-computed results from CSV files...")
        act_names = list(activation_fns.keys())
        return load_sweep_results_from_csv(data_dir=data_dir, activation_fns=act_names)

    print("Running architecture sweep...")

    if save_to_csv:
        os.makedirs(data_dir, exist_ok=True)

    all_results = []
    total_start_time = time.time()

    for act_name, act_fn in activation_fns.items():
        print(f"---------- {act_name} ----------")
        act_start_time = time.time()

        for L in num_hidden_layers:
            for W in hidden_widths:
                print(f"Layers: {L}, Nodes: {W}")
                layers = [2] + [W] * L + [1]
                activations = [act_fn] * (len(layers) - 2)

                L2_all = []
                Linf_all = []
                arch_results = []  # Store results for this specific architecture

                for seed in seeds:
                    print(f"SEED = {seed}")
                    model, losses, loss_components = train_pinn(
                        layers=layers,
                        activations=activations,
                        steps=steps,
                        N_int=N_int,
                        T=T,
                        lr=lr,
                        seed=seed,
                        optimizer=optimizer,
                    )

                    # Create evaluation grids
                    x_eval = jnp.linspace(0, 1, Nx_eval)
                    t_eval = jnp.linspace(0, T, Nt_eval)
                    y_eval = jnp.linspace(0, 1, Ny_eval) if Ny_eval else None
                    
                    # Determine dimension (1D or 2D only)
                    dim_eval = 2 if Ny_eval else 1

                    L2, Linf = compute_error_metrics(
                        model,
                        x=x_eval,
                        y=y_eval,
                        z=None,
                        t=t_eval,
                        dim=dim_eval,
                    )
                    L2_all.append(L2)
                    Linf_all.append(Linf)

                    if save_to_csv:
                        arch_results.append(
                            {
                                "activation": act_name,
                                "hidden_layers": L,
                                "width": W,
                                "seed": seed,
                                "L2_rel": float(L2),
                                "Linf": float(Linf),
                            }
                        )

                # Save CSV for this specific architecture
                if save_to_csv and arch_results:
                    df_arch = pd.DataFrame(arch_results)
                    filename = os.path.join(data_dir, f"{act_name}_L{L}_N{W}.csv")
                    df_arch.to_csv(filename, index=False)
                    print(f"Saved to {filename}")

                result = {
                    "activation": act_name,
                    "hidden_layers": L,
                    "width": W,
                    "L2_rel_mean": float(sum(L2_all) / len(L2_all)),
                    "Linf_mean": float(sum(Linf_all) / len(Linf_all)),
                }

                print(
                    f"L2_rel={result['L2_rel_mean']:.3e}, "
                    f"Linf={result['Linf_mean']:.3e}"
                )

                all_results.append(result)
                print("\n")

        act_elapsed = time.time() - act_start_time
        total_elapsed = time.time() - total_start_time
        print(f"    {act_name} completed in {act_elapsed:.1f}s")
        print(f"    Total time elapsed: {total_elapsed:.1f}s")
        print("\n")

    results_df = pd.DataFrame(all_results)
    return results_df






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

    # Determine which files to load
    if activation_fns is None:
        # Load all sweep CSV files
        csv_files = [
            f
            for f in os.listdir(data_dir)
            if f.startswith("sweep_") and f.endswith(".csv")
        ]
    else:
        csv_files = [f"sweep_{act_name}.csv" for act_name in activation_fns]

    if not csv_files:
        raise FileNotFoundError(f"No sweep CSV files found in '{data_dir}'")

    # Load and concatenate all dataframes
    dfs = []
    for csv_file in csv_files:
        filepath = os.path.join(data_dir, csv_file)
        if os.path.exists(filepath):
            df = pd.read_csv(filepath)
            dfs.append(df)
            print(f"Loaded {filepath}")
        else:
            print(f"Warning: {filepath} not found, skipping")

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
