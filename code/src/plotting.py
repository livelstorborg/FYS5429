import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import flax.nnx as nnx




# =============================================================================
#                       Single 3D surface
# =============================================================================
def plot_3d_surface(
    x,
    t,
    U,
    title="",
    elev=30,
    azim=-135,
    cmap="viridis",
    xlabel="x",
    ylabel="t",
    colorbar_label="u(x, t)",
    show=False,
    savefig=False,
    filepath=None,
):
    """
    Create a single 3D surface plot.
    
    Parameters:
    -----------
    x : array
        Spatial coordinates
    t : array
        Time coordinates
    U : 2D array (Nt x Nx)
        Solution values
    title : str
        Plot title
    elev : float
        Elevation angle for viewing
    azim : float
        Azimuth angle for viewing
    cmap : str
        Colormap name
    colorbar_label : str
        Label for the colorbar
    show : bool
        Whether to display the plot
    savefig : bool
        Whether to save the figure
    filepath : str or None
        Path to save the figure (used if savefig=True)
    
    Returns:
    --------
    fig : matplotlib figure
        The figure object
    """
    fig = plt.figure(figsize=(12, 10))
    ax = fig.add_subplot(111, projection="3d")

    X, T = np.meshgrid(x, t)
    surf = ax.plot_surface(X, T, U, cmap=cmap, edgecolor="none", alpha=0.8)

    # Labels
    ax.set_xlabel(xlabel, fontsize=16, labelpad=12)
    ax.set_ylabel(ylabel, fontsize=16, labelpad=12)
    ax.set_zlabel(colorbar_label, fontsize=16, labelpad=12)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=5)

    # Rotate z-axis labels
    for label in ax.zaxis.get_ticklabels():
        label.set_rotation(45)
        label.set_fontsize(16)

    ax.view_init(elev=elev, azim=azim)

    # Tick font sizes
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.tick_params(axis="z", which="major", labelsize=16)

    # Colorbar
    cbar = fig.colorbar(
        surf,
        ax=ax,
        aspect=25,
        shrink=0.75,
    )
    cbar.set_label(colorbar_label, fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=16)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    
    # Save if requested
    if savefig and filepath:
        fig.savefig(filepath, dpi=300, bbox_inches="tight", pad_inches=0.1)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)  # <-- VIKTIG: Lukk figuren hvis den ikke skal vises
    
    return fig


# =============================================================================
#                       Subplots of 3D surfaces
# =============================================================================
def subplot_3d_surfaces(
    figures,
    titles=None,
    elev=20,
    azims=None,
    cmap="viridis",
    xlabel="x",
    ylabel="t",
    colorbar_label="Error",
    suptitle="3D Surface Plots",
    title_fontsize=18,
    show=False,
    savefig=False,
    filepath=None,
):
    """
    Create a row of 3D surface plots from existing figure data.
    
    Parameters:
    -----------
    figures : list of dicts
        Each dict should contain: {'x': x, 't': t, 'U': U}
    titles : list of str or None
        Titles for each subplot
    elev : float
        Elevation angle for all subplots
    azims : list of float or None
        Azimuth angles for each subplot (defaults to [0, 45, 90, 135])
    cmap : str
        Colormap name
    colorbar_label : str
        Label for the shared colorbar
    suptitle : str
        Overall figure title
    show : bool
        Whether to display the plot
    savefig : bool
        Whether to save the figure
    save_path : str or None
        Path to save the figure
    
    Returns:
    --------
    fig : matplotlib figure
        The figure object
    """
    n_plots = len(figures)
    
    if azims is None:
        azims = [45 * i for i in range(n_plots)]
    
    if titles is None:
        titles = [""] * n_plots
    
    fig, axes = plt.subplots(
        1,
        n_plots,
        figsize=(8 * n_plots, 7),
        subplot_kw={"projection": "3d"},
        gridspec_kw={"wspace": 0.1},
    )
    
    # Handle single subplot case
    if n_plots == 1:
        axes = [axes]

    for idx, (ax, fig_data, az, title) in enumerate(zip(axes, figures, azims, titles)):
        x = fig_data['x']
        t = fig_data['t']
        U = fig_data['U']

        X, T = np.meshgrid(x, t)
        surf = ax.plot_surface(X, T, U, cmap=cmap, edgecolor="none", alpha=0.9)
        ax.view_init(elev=elev, azim=az)
        ax.grid(alpha=0.3)

        if title:
            ax.set_title(title, fontsize=title_fontsize, y=0.98, fontweight="bold")

        # Labels
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)

        # Z-axis handling
        ax.tick_params(axis="z", labelsize=10)
        for label in ax.get_zticklabels():
            label.set_rotation(45)
            label.set_fontsize(10)

        # Set tick sizes on x/t axis
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

        # Per-subplot colorbar
        cbar = fig.colorbar(surf, ax=ax, shrink=0.5, aspect=20, pad=0.1)
        cbar.set_label(colorbar_label, fontsize=12)
        cbar.ax.tick_params(labelsize=10)

    fig.suptitle(suptitle, fontsize=22, fontweight="bold", y=0.95)

    # Save if requested
    if savefig and filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=300, bbox_inches="tight", pad_inches=0.1)
    
    # Show if requested
    if show:
        plt.show()
    






# =============================================================================
#                       Loss curve
# =============================================================================
def plot_loss(losses, show=True, savefig=False, filepath=None):
    fig, ax = plt.subplots(figsize=(8,5))
    ax.semilogy(losses, linewidth=2)
    ax.set_xlabel("Iterations", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_title(r"$\mathbf{Training\ Loss}$", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.grid(True)
    plt.tight_layout()
    if savefig and filepath:
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# =============================================================================
#                 Comparing loss across models, with smoothed curves
# =============================================================================
def plot_loss_comparison(losses_list, labels, title="Training Loss Comparison", show=True, savefig=False, filepath=None, smooth_window=50):
    """
    Plot and compare training losses for multiple models.

    Each model gets one color with two lines:
      - low-alpha raw loss
      - solid smoothed loss (running average)

    Parameters
    ----------
    losses_list  : list of array-like, one per model
    labels       : list of str, one label per model
    smooth_window: int, window size for running average
    show         : bool
    savefig      : bool
    filepath     : str, save path if savefig=True
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["black", "red"]

    smoothed_data = []
    for i, (losses, label) in enumerate(zip(losses_list, labels)):
        color = colors[i % len(colors)]
        losses_np = np.array(losses)
        steps = np.arange(len(losses_np))

        ax.semilogy(steps, losses_np, color=color, alpha=0.4, linewidth=4)

        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(losses_np, kernel, mode="valid")
        offset = smooth_window - 1
        smoothed_data.append((steps[offset:], smoothed, color, label))

    for steps_s, smoothed, color, label in smoothed_data:
        ax.semilogy(steps_s, smoothed, color=color, label=label, linewidth=2)

    ax.set_xlabel("Iterations", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_title(title, fontsize=18, fontweight="bold")
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True)
    plt.tight_layout()

    if savefig and filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# =============================================================================
#                       Loss componens curves
# =============================================================================
def plot_loss_components(loss_comps, show=True, savefig=False, filepath=None):
    fig, ax = plt.subplots(figsize=(8,5))
    for key, label in [("pde", "PDE"), ("ic_u", "IC u"), ("ic_ut", "IC ∂u/∂t"), ("sobolev", "Sobolev")]:
        if key in loss_comps and any(v > 0 for v in loss_comps[key]):
            ax.semilogy(loss_comps[key], label=label, linewidth=2)
    ax.set_xlabel("Iterations", fontsize=16)
    ax.set_ylabel("Loss component", fontsize=16)
    ax.set_title(r"$\mathbf{Loss\ Components}$", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True)
    plt.tight_layout()
    if savefig and filepath:
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# =============================================================================
#                       Heatmap of L2 error across width/depth
# =============================================================================
def plot_heatmap_width_depth(df, activation, show=True, savefig=False, filepath=None):
    data = df[df["activation"] == activation]

    if data.empty:
        raise ValueError(f"No entries found for activation '{activation}'.")

    widths = sorted(data["width"].unique())
    depths = sorted(data["hidden_layers"].unique())

    M = np.zeros((len(depths), len(widths)))

    for _, row in data.iterrows():
        i = depths.index(row["hidden_layers"])
        j = widths.index(row["width"])
        M[i, j] = row["L2_rel_mean"]

    fig, ax = plt.subplots(figsize=(7, 5))
    im = ax.imshow(M, cmap="viridis", origin="lower")

    ax.set_xticks(range(len(widths)))
    ax.set_xticklabels(widths, fontsize=16)  # Added fontsize

    ax.set_yticks(range(len(depths)))
    ax.set_yticklabels(depths, fontsize=16)  # Added fontsize

    ax.set_xlabel("Width", fontsize=16)
    ax.set_ylabel("Depth", fontsize=16)
    ax.set_title(
        rf"Relative $\mathbf{{L^2}}$ Error ({activation})",
        fontsize=18,
        fontweight="bold",
    )

    # ----------------------------
    # Add numbers inside each cell
    # ----------------------------
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            ax.text(
                j,
                i,
                f"{M[i, j]:.5f}",
                ha="center",
                va="center",
                color="white",
                fontsize=12,
                alpha=1,
            )

    # --------------------
    # Make grid lines faint (draw FIRST so rectangle is on top)
    # --------------------
    ax.set_xticks(np.arange(-0.5, len(widths), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(depths), 1), minor=True)
    ax.grid(which="minor", color=(1, 1, 1, 0.3), linewidth=0.3)
    ax.grid(which="major", alpha=0)  # hide old major grid

    # -----------------------------------
    # Draw red rectangle around min value (with high zorder to be on top)
    # -----------------------------------
    min_row, min_col = np.unravel_index(np.argmin(M), M.shape)
    rect = plt.Rectangle(
        (min_col - 0.5, min_row - 0.5),
        1,
        1,
        facecolor="none",
        edgecolor="red",
        linewidth=3,
        zorder=10,
    )
    ax.add_patch(rect)

    cbar = fig.colorbar(im, ax=ax, label=r"$L^2$ error")
    cbar.ax.tick_params(labelsize=16)
    cbar.set_label(r"$L^2$ error", fontsize=16)

    # ----------------------------
    # Format colorbar in scientific notation
    # ----------------------------
    cbar.formatter.set_powerlimits((0, 0))  # Force scientific notation
    cbar.update_ticks()

    fig.tight_layout()

    if savefig and filepath:
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()

    return fig


# =============================================================================
#                       Printing table of results from CSV files
# =============================================================================
def print_optimizer_comparison_tables(data_folder):
    """
    Read CSV files from optimizer subfolders and print comparison tables.
    One table per optimizer with columns: Activation, Layers, Nodes, L2-error, Linf-error

    Parameters:
    -----------
    data_folder : str or Path
        Path to the data folder containing optimizer subdirectories
        (e.g., 'data/1d_constant' which contains 'adam/', 'adamw/', 'lbfgs/')
    """
    data_path = Path(data_folder)

    if not data_path.exists():
        print(f"Error: Folder {data_path} does not exist!")
        return

    optimizer_dirs = sorted([d for d in data_path.iterdir() if d.is_dir()])

    if not optimizer_dirs:
        print(f"No optimizer subdirectories found in {data_path}")
        return

    print(f"\n{'=' * 80}")
    print(f"Results from: {data_path}")
    print(f"{'=' * 80}\n")

    for opt_dir in optimizer_dirs:
        optimizer_name = opt_dir.name
        csv_files = list(opt_dir.glob("*.csv"))

        if not csv_files:
            print(f"No CSV files found for optimizer: {optimizer_name}")
            continue

        print(f"\n{'─' * 80}")
        print(f"Optimizer: {optimizer_name.upper()}")
        print(f"{'─' * 80}")

        table_data = []
        for csv_file in sorted(csv_files):
            try:
                df = pd.read_csv(csv_file)
                parts = csv_file.stem.split("_")
                if len(parts) >= 3:
                    table_data.append({
                        "Activation": parts[0],
                        "Layers": int(parts[1].replace("L", "")),
                        "Nodes": int(parts[2].replace("N", "")),
                        "L2-error": f"{df['L2_rel'].mean():.6f}",
                        "Linf-error": f"{df['Linf'].mean():.6f}",
                    })
            except Exception as e:
                print(f"Warning: Could not read {csv_file}: {e}")

        if table_data:
            result_df = pd.DataFrame(table_data).sort_values(["Activation", "Layers", "Nodes"])
            print(f"{'Activation':<12} {'Layers':<8} {'Nodes':<8} {'L2-error':<12} {'Linf-error':<12}")
            prev_activation = None
            for _, row in result_df.iterrows():
                if prev_activation is not None and row["Activation"] != prev_activation:
                    print("\n" + "-" * 60 + "\n")
                print(f"{row['Activation']:<12} {row['Layers']:<8} {row['Nodes']:<8} {row['L2-error']:<12} {row['Linf-error']:<12}")
                prev_activation = row["Activation"]

            best = result_df.loc[result_df["L2-error"].astype(float).idxmin()]
            print(f"\n  Best: {best['Activation']}, Layers={best['Layers']}, Nodes={best['Nodes']}, L2={best['L2-error']}")
        else:
            print("No data available for this optimizer")

    print(f"\n{'=' * 80}\n")




# =============================================================================
#                       Plotting 2D snapshots at specific time points
# =============================================================================
def plot_2d_snapshots(field, t_indices, t_labels, title, cmap="viridis",
                      show=True, savefig=False, filepath=None):
    """
    Plot 2D snapshots of a field (solution, absolute error, etc.) at given time indices.

    Parameters
    ----------
    field       : array-like, shape (Nt, Nx, Ny)
    t_indices   : list of int, time indices to plot
    t_labels    : list of float, corresponding time values (for titles)
    title       : str, overall figure title
    cmap        : str, colormap
    show        : bool
    savefig     : bool
    filepath    : str, save path if savefig=True
    """
    n = len(t_indices)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 6), constrained_layout=True)
    if n == 1:
        axes = [axes]

    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.93)

    for ax, idx, t_val in zip(axes, t_indices, t_labels):
        snapshot = np.array(field[idx])
        im = ax.imshow(
            snapshot.T, origin="lower", extent=[0, 1, 0, 1],
            cmap=cmap, aspect="equal"
        )
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.01)
        ax.set_xlabel("x", fontsize=11)
        ax.set_ylabel("y", fontsize=11)
        ax.set_title(rf"$\mathbf{{t={t_val:.2f}}}$", fontsize=18)

    if savefig and filepath is not None:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    plt.close()





def plot_optimizer_comparison(
    *,
    losses_adam,
    losses_lbfgs_warm,
    losses_lbfgs_cold,
    smooth_window=50,
    show=True,
    savefig=False,
    fig_dir=None,
):
    import numpy as np

    lbfgs_warm_plot = [losses_adam[-1]] + list(losses_lbfgs_warm)

    losses_list = [losses_adam, losses_lbfgs_cold, lbfgs_warm_plot]
    labels = ["Adam", "L-BFGS", "Adam + L-BFGS"]
    colors = ["black", "darkgreen", "red"]

    fig, ax = plt.subplots(figsize=(8, 5))

    smoothed_data = []
    for losses, label, color in zip(losses_list, labels, colors):
        losses_np = np.array(losses)
        steps = np.arange(len(losses_np))
        ax.semilogy(steps, losses_np, color=color, alpha=0.4, linewidth=5)
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(losses_np, kernel, mode="valid")
        offset = smooth_window - 1
        smoothed_data.append((steps[offset:], smoothed, color, label))

    for steps_s, smoothed, color, label in smoothed_data:
        ax.semilogy(steps_s, smoothed, color=color, label=label, linewidth=2)

    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_title(r"$\mathbf{Optimizer \ Comparison}$", fontsize=18)
    ax.tick_params(axis="both", labelsize=16)
    ax.legend(fontsize=16)
    ax.grid(True)
    plt.tight_layout()

    if savefig and fig_dir:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(fig_dir) / "optimizers_loss.pdf", bbox_inches="tight")
    if show:
        plt.show()


# =============================================================================
#                   Norm comparison (L2 / H1 / H2 loss curves)
# =============================================================================
def plot_norm_comparison(act_name, loss_L2, loss_H1, loss_H2, savefig=True, filepath=None, show=False):
    fig, ax = plt.subplots(figsize=(8, 5))

    for losses, label, color in zip(
        [loss_L2, loss_H1, loss_H2],
        ["L2", "H1", "H2"],
        ["black", "red", "steelblue"],
    ):
        losses_np = np.array(losses)
        steps = np.arange(len(losses_np))
        ax.semilogy(steps, losses_np, color=color, label=label, linewidth=2)

    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_title(act_name, fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True)
    plt.tight_layout()

    if savefig and filepath:
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_activation_loss_sweep(
        activation_fns,
        data_dir,
        fig_dir,
        show=False,
        savefig=True,
):
    """
    Read CSVs produced by run_activation_loss_sweep and save one loss-curve PDF
    per activation function.

    Parameters
    ----------
    activation_fns : dict[str, callable]
        Keys are used to find the corresponding CSV (e.g. "GeLU" -> GeLU.csv).
    data_dir : str or Path
    fig_dir  : str or Path
    """
    data_dir = Path(data_dir)
    fig_dir = Path(fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)

    for act_name in activation_fns:
        csv_path = data_dir / f"{act_name}.csv"
        if not csv_path.exists():
            print(f"  No CSV for {act_name}, skipping.")
            continue
        df = pd.read_csv(csv_path)
        curves = {
            "L2": df["L2_loss"].tolist(),
            "H1": df["H1_loss"].tolist(),
            "H2": df["H2_loss"].tolist(),
        }
        filepath = fig_dir / f"loss_vs_step_{act_name}.pdf"
        plot_norm_comparison(
            act_name,
            curves["L2"], curves["H1"], curves["H2"],
            savefig=savefig,
            filepath=str(filepath),
            show=show,
        )
        print(f"  Saved figure: loss_vs_step_{act_name}.pdf")


# =============================================================================
#              Max condition number table (per activation / norm)
# =============================================================================
def plot_average_max_conditional_number_table(activation_fns, norms, data_dir):
    data_dir = Path(data_dir)

    col_w = 14
    header = f"{'Activation':<14}" + "".join(f"{'max_cond_' + n:>{col_w}}" for n in norms)
    print(header)
    print("-" * len(header))

    for act_name in activation_fns:
        csv_path = data_dir / f"{act_name}.csv"
        if not csv_path.exists():
            print(f"{act_name:<14}" + "".join(f"{'N/A':>{col_w}}" for _ in norms))
            continue
        df = pd.read_csv(csv_path)
        row = f"{act_name:<14}"
        for n in norms:
            col = f"max_cond_{n}"
            val = df[col].mean() if col in df.columns else float("nan")
            row += f"{val:>{col_w}.2e}"
        print(row)


# =============================================================================
#                   Loss curves for multiple k values
# =============================================================================
def plot_loss_curve_multiple_k(k_vals, loss_curves, smooth_window=50, savefig=False, fig_dir=None):
    """
    One loss curve per k on a single figure.

    Parameters
    ----------
    k_vals      : list of k values
    loss_curves : list of loss histories, one per k
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(k_vals)))

    for losses, k_val, color in zip(loss_curves, k_vals, colors):
        losses_np = np.array(losses)
        steps = np.arange(len(losses_np))
        ax.semilogy(steps, losses_np, color=color, alpha=0.3, linewidth=3)
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(losses_np, kernel, mode="valid")
        ax.semilogy(steps[smooth_window - 1:], smoothed,
                    color=color, linewidth=2, label=f"k={int(k_val)}")

    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_title(r"$\mathbf{Training\ Loss\ per\ k}$", fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True)
    plt.tight_layout()

    if savefig and fig_dir:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(fig_dir) / "loss_curves_per_k.pdf", bbox_inches="tight")
    plt.show()


def plot_loss_curve_multiple_c(c_vals, loss_curves, smooth_window=50, savefig=False, fig_dir=None):
    """
    One loss curve per c on a single figure.

    Parameters
    ----------
    c_vals      : list of c values
    loss_curves : list of loss histories, one per c
    """
    fig, ax = plt.subplots(figsize=(9, 6))
    colors = plt.cm.viridis(np.linspace(0, 0.85, len(c_vals)))

    for losses, c_val, color in zip(loss_curves, c_vals, colors):
        losses_np = np.array(losses)
        steps = np.arange(len(losses_np))
        ax.semilogy(steps, losses_np, color=color, alpha=0.3, linewidth=3)
        kernel = np.ones(smooth_window) / smooth_window
        smoothed = np.convolve(losses_np, kernel, mode="valid")
        ax.semilogy(steps[smooth_window - 1:], smoothed,
                    color=color, linewidth=2, label=f"c={int(c_val)}")

    ax.set_xlabel("Step", fontsize=16)
    ax.set_ylabel("Loss", fontsize=16)
    ax.set_title(r"$\mathbf{Training\ Loss\ per\ c}$", fontsize=18, fontweight="bold")
    ax.tick_params(labelsize=14)
    ax.legend(fontsize=14)
    ax.grid(True)
    plt.tight_layout()

    if savefig and fig_dir:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(fig_dir) / "loss_curves_per_c.pdf", bbox_inches="tight")
    plt.show()


def plot_ck_norm_comparison(
    curves_c1k5,
    curves_c5k1,
    smooth_window=50,
    show=True,
    savefig=False,
    fig_dir=None,
):
    """
    Two-panel subplot comparing L2/H1/H2 loss curves for (c=1,k=5) and (c=5,k=1).

    Parameters
    ----------
    curves_c1k5 : dict with keys "L2", "H1", "H2" — loss lists for c=1, k=5
    curves_c5k1 : dict with keys "L2", "H1", "H2" — loss lists for c=5, k=1
    """
    norm_colors = {"L2": "black", "H1": "red", "H2": "green"}
    panels = [
        (curves_c1k5, r"$c=1,\ k=5$"),
        (curves_c5k1, r"$c=5,\ k=1$"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=False)

    for ax, (curves, title) in zip(axes, panels):
        smoothed_data = []
        for norm, color in norm_colors.items():
            losses_np = np.array(curves[norm])
            steps = np.arange(len(losses_np))
            ax.semilogy(steps, losses_np, color=color, alpha=0.4, linewidth=5)
            kernel = np.ones(smooth_window) / smooth_window
            smoothed = np.convolve(losses_np, kernel, mode="valid")
            offset = smooth_window - 1
            smoothed_data.append((steps[offset:], smoothed, color, norm))

        for steps_s, smoothed, color, label in smoothed_data:
            ax.semilogy(steps_s, smoothed, color=color, label=label, linewidth=2)

        ax.set_xlabel("Step", fontsize=16)
        ax.set_ylabel("Loss", fontsize=16)
        ax.set_title(title, fontsize=18)
        ax.tick_params(axis="both", labelsize=14)
        ax.legend(fontsize=14)
        ax.grid(True)

    plt.tight_layout()

    if savefig and fig_dir:
        Path(fig_dir).mkdir(parents=True, exist_ok=True)
        fig.savefig(Path(fig_dir) / "norm_comparison_ck.pdf", bbox_inches="tight")
    if show:
        plt.show()