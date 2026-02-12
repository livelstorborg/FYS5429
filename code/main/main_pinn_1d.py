import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
import jax.nn as jnn
import numpy as np
import pandas as pd

from src.pde import fd_solve, fem_solve, u_exact, create_grid
from src.pinn import train_pinn
from src.experiment import absolute_error, relative_error, run_architecture_sweep
from src.plotting import (
    plot_solution_at_t,
    plot_scheme_error_at_t,
    plot_3d_surface,
    subplot_3d_surfaces,
)


def print_optimizer_comparison_tables(data_folder):
    """
    Read CSV files from optimizer subfolders and print comparison tables.
    One table per optimizer with columns: Activation, Layers, Nodes, L2-error, Linf-error
    
    Parameters:
    -----------
    data_folder : str or Path
        Path to the data folder containing optimizer subdirectories
        (e.g., '../data/1d_constant' which contains 'adam/', 'adamw/', 'lbfgs/')
    """
    data_path = Path(data_folder)
    
    if not data_path.exists():
        print(f"Error: Folder {data_path} does not exist!")
        return
    
    # Get all optimizer subdirectories
    optimizer_dirs = [d for d in data_path.iterdir() if d.is_dir()]
    
    if not optimizer_dirs:
        print(f"No optimizer subdirectories found in {data_path}")
        return
    
    optimizer_dirs = sorted(optimizer_dirs)
    
    print(f"\n{'='*80}")
    print(f"Results from: {data_path}")
    print(f"{'='*80}\n")
    
    # Process each optimizer
    for opt_dir in optimizer_dirs:
        optimizer_name = opt_dir.name
        csv_files = list(opt_dir.glob("*.csv"))
        
        if not csv_files:
            print(f"No CSV files found for optimizer: {optimizer_name}")
            continue
        
        print(f"\n{'─'*80}")
        print(f"Optimizer: {optimizer_name.upper()}")
        print(f"{'─'*80}")
        
        # Collect data for this optimizer
        table_data = []
        
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                
                # Parse filename: {activation}_{layers}_{nodes}.csv
                # Example: GeLU_L1_N32.csv -> activation=GeLU, layers=1, nodes=32
                filename = csv_file.stem
                parts = filename.split('_')
                
                if len(parts) >= 3:
                    activation = parts[0]
                    layers = parts[1].replace('L', '')  # Remove 'L' prefix
                    nodes = parts[2].replace('N', '')   # Remove 'N' prefix
                    
                    # Calculate mean errors across seeds
                    l2_error = df['L2_rel'].mean()
                    linf_error = df['Linf'].mean()
                    
                    table_data.append({
                        'Activation': activation,
                        'Layers': int(layers),
                        'Nodes': int(nodes),
                        'L2-error': f"{l2_error:.6f}",
                        'Linf-error': f"{linf_error:.6f}"
                    })
                    
            except Exception as e:
                print(f"Warning: Could not read {csv_file}: {e}")
        
        if table_data:
            # Create DataFrame and sort by Activation, then Layers, then Nodes
            result_df = pd.DataFrame(table_data)
            result_df = result_df.sort_values(['Activation', 'Layers', 'Nodes'])
            
            # Print with separators between different activation functions
            print(f"{'Activation':<12} {'Layers':<8} {'Nodes':<8} {'L2-error':<12} {'Linf-error':<12}")
            prev_activation = None
            for _, row in result_df.iterrows():
                if prev_activation is not None and row['Activation'] != prev_activation:
                    print()
                    print("-" * 60)
                    print()
                print(f"{row['Activation']:<12} {row['Layers']:<8} {row['Nodes']:<8} {row['L2-error']:<12} {row['Linf-error']:<12}")
                prev_activation = row['Activation']
        else:
            print("No data available for this optimizer")
    
    print(f"\n{'='*80}\n")


Nx = 100
T = 2.0
c = 1.0
cfl = 0.4


x, t, dx, dt = create_grid(Nx=Nx, T=T, c=c, cfl=cfl, dim=1)

u_exact = u_exact(x, t, c=c, dim=1)
u_fd = fd_solve(x, t, dx, dt, c=c, dim=1)
u_fem = fem_solve(x, t, dx, dt, c=c, dim=1)

u_pinn, losses, loss_components = train_pinn(
    steps=5000,
    layers=[2, 32, 32, 32, 1],
    activations=[jnn.silu, jnn.silu, jnn.silu],
    N_int=100,
    T=T,
    c=c,
    dim=1,
    lambda_ic=10.0,
)


X_mesh, T_mesh = np.meshgrid(x, t)
xt_test = jnp.column_stack([X_mesh.ravel(), T_mesh.ravel()])
u_pinn_vals = u_pinn(xt_test).reshape(X_mesh.shape)


# fig_exact = plot_3d_surface(
#     x,
#     t,
#     u_exact,
#     elev=20,
#     azim=45,
#     title="Analytical Solution (1D)",
#     savefig=False,
#     show=False,
# )

# fig_fd = plot_3d_surface(
#     x,
#     t,
#     u_fd,
#     elev=20,
#     azim=45,
#     title="Finite Difference Solution (1D)",
#     savefig=False,
#     show=False,
# )

# fig_fem = plot_3d_surface(
#     x,
#     t,
#     u_fem,
#     elev=20,
#     azim=45,
#     title="Finite Element Solution (1D)",
#     savefig=False,
#     show=False,
# )

# fig_pinn = plot_3d_surface(
#     x,
#     t,
#     u_pinn_vals,
#     elev=20,
#     azim=45,
#     title="PINN Solution (1D)",
#     savefig=False,
#     show=False,
# )

# subplot_fig = subplot_3d_surfaces(
#     figures=[
#         {'x': x, 't': t, 'U': u_exact},
#         {'x': x, 't': t, 'U': u_fd},
#         {'x': x, 't': t, 'U': u_fem},
#         {'x': x, 't': t, 'U': u_pinn_vals},
#     ],
#     titles=["Analytical", "Finite Difference", "Finite Element", "PINN (SiLU)"],
#     elev=20,
#     azims=[45, 45, 45, 45],
#     cmap="viridis",
#     colorbar_label="u(x, t)",
#     suptitle="Wave Equation Solutions",
#     savefig=False,
#     save_path="../figs/analytical_fd_pinn_silu_subplot.pdf",
#     show=True,
# )

# subplot_error = subplot_3d_surfaces(
#     figures=[
#         {'x': x, 't': t, 'U': np.abs(u_fd - u_exact)},
#         {'x': x, 't': t, 'U': np.abs(u_fem - u_exact)},
#         {'x': x, 't': t, 'U': np.abs(u_pinn_vals - u_exact)},
#     ],
#     titles=["Finite Difference", "Finite Element", "PINN (SiLU)"],
#     elev=20,
#     azims=[45, 45, 45],
#     cmap="viridis",
#     colorbar_label="u(x, t)",
#     suptitle="Error",
#     savefig=False,
#     save_path="../figs/comparison_subplot.pdf",
#     show=True,
# )


# opt_names = ['adam', 'adamw', 'lbfgs']
opt_names = ['lbfgs']

for opt in opt_names:
    run_architecture_sweep(
        hidden_widths=[64],
        num_hidden_layers=[2],
        activation_fns={
            # 'tanh': jnn.tanh,
            # 'sine': jnp.sin,
            # 'GeLU': jnn.gelu,
            # 'SiLU': jnn.swish,
            'ReLU': jnn.relu,
        },
        T=1.0,
        steps=1000,
        N_int=1000,
        lr=1e-3,
        seeds=(0,),
        Nx_eval=100,
        Ny_eval=None,
        Nt_eval=100,
        optimizer=opt,
        save_to_csv=True,
        use_pre_computed=False,
        data_dir=f"../data/1d_constant/{opt}",
    )


# Example usage: Print comparison tables for all optimizers in 1d_constant folder
print_optimizer_comparison_tables("../data/1d_constant")
