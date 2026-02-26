import matplotlib.pyplot as plt
import os
import jax.numpy as jnp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# -----------------------------------------------------------------------------
# Part b: finite-difference comparisons
# -----------------------------------------------------------------------------

def plot_solution_at_t(
    *,
    grid,
    u_num,
    u_true,
    dx,
    t,
    dim,
    filepath,
):
    colors = ["blue", "green", "orange", "purple"]
    num_t = len(t)

    fig, axes = plt.subplots(1, num_t, figsize=(6 * num_t, 5))
    fig.suptitle(
        rf"$\mathbf{{{dim}D}}$ Wave Equation, $\mathbf{{\Delta x = {dx:.3f}}}$",
        fontsize=20,
        fontweight="bold",
        y=0.98
    )

    if num_t == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        grid_t = grid[idx] if isinstance(grid, list) else grid
        u_true_t = u_true[idx] if isinstance(u_true, list) else u_true

        ax.plot(grid_t, u_true_t, label="Analytical", color="red", linewidth=5, alpha=0.5)
        for i, scheme in enumerate(u_num):
            ax.plot(grid_t, scheme["data"][idx], ":", label=scheme["label"],
                    color=colors[i % len(colors)], linewidth=3)

        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xlabel("x", fontsize=16)
        ax.set_ylabel("u(x, t)", fontsize=16)
        ax.set_title(rf"$t = {float(t[idx]):.3f}$", fontsize=16, fontweight="bold")
        ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()


def plot_error_at_t(
    *,
    grid,
    u_num,
    u_true,
    dx,
    t,
    dim,
    filepath,
):
    colors = ["blue", "green", "orange", "purple"]
    num_t = len(t)

    fig, axes = plt.subplots(1, num_t, figsize=(6 * num_t, 5))
    fig.suptitle(
        rf"$\mathbf{{{dim}D}}$ Wave Equation, $\mathbf{{\Delta x = {dx:.3f}}}$",
        fontsize=20,
        fontweight="bold",
        y=0.98
    )

    if num_t == 1:
        axes = [axes]

    for idx, ax in enumerate(axes):
        grid_t = grid[idx] if isinstance(grid, list) else grid
        u_true_t = u_true[idx] if isinstance(u_true, list) else u_true

        for i, scheme in enumerate(u_num):
            err = np.abs(scheme["data"][idx] - u_true_t)
            ax.plot(grid_t, err, label=f"{scheme['label']}",
                    color=colors[i % len(colors)], linewidth=2)

        ax.set_yscale("log")
        ax.grid(alpha=0.3)
        ax.tick_params(axis='both', labelsize=16)
        ax.set_xlabel("x", fontsize=16)
        ax.set_ylabel("Absolute Error", fontsize=16)
        ax.set_title(rf"$t = {float(t[idx]):.3f}$", fontsize=16, fontweight="bold")
        ax.legend(fontsize=14)

    plt.tight_layout()
    plt.savefig(filepath, dpi=300, bbox_inches="tight")
    plt.show()




# -----------------------------------------------------------------------------
# Part b: FD scheme error curves
# -----------------------------------------------------------------------------
def plot_scheme_errors_at_t(error_list, t, title, filepath):
    plt.figure(figsize=(8, 5))

    for err in error_list:
        x = err["x"]
        dx = err["dx"]
        # Determine which error to plot based on which time is closer
        if abs(err["t1"] - t) < abs(err["t2"] - t):
            error = np.abs(err["t_error"])
        else:
            error = np.abs(err["t2_error"])
        plt.plot(x, error, label=rf"$\Delta x = {dx:.2f}$", linewidth=2)

    plt.grid(alpha=0.3)
    plt.xlabel("x", fontsize=16)
    plt.ylabel("Error", fontsize=16)
    plt.title(title, fontsize=18, fontweight="bold")
    plt.legend(fontsize=16)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    plt.yscale("log")
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.show()





# -----------------------------------------------------------------------------
# Part c: 3D surfaces for solutions/errors
# -----------------------------------------------------------------------------
def plot_3d_surface(
    x, 
    t, 
    U, 
    title="", 
    elev=30, 
    azim=-135,
    cmap="viridis",
    colorbar_label="u(x, t)",
    show=False,
    savefig=False,
    save_path=None
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
    save_path : str or None
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
    ax.set_xlabel("x", fontsize=16, labelpad=12)
    ax.set_ylabel("t", fontsize=16, labelpad=12)
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
    if savefig and save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    
    # Show if requested
    if show:
        plt.show()
    else:
        plt.close(fig)  # <-- VIKTIG: Lukk figuren hvis den ikke skal vises
    
    return fig



def subplot_3d_surfaces(
    figures,
    titles=None,
    elev=20,
    azims=None,
    cmap="viridis",
    colorbar_label="Error",
    suptitle="3D Surface Plots",
    show=False,
    savefig=False,
    save_path=None,
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
        figsize=(6.25 * n_plots, 6),
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
            ax.set_title(title, fontsize=14, pad=10)

        # Labels
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("t", fontsize=12)

        # Z-axis handling
        if idx == 0:
            # First subplot: show z-ticks + numbers + rotate them
            ax.tick_params(axis="z", labelsize=10)
            for label in ax.get_zticklabels():
                label.set_rotation(45)
                label.set_fontsize(10)
        else:
            # Other subplots: keep ticks but remove numbers
            zticks = ax.get_zticks()
            ax.set_zticks(zticks)
            ax.set_zticklabels(["" ] * len(zticks))

        # Set tick sizes on x/t axis
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

    # Shared colorbar
    cbar = fig.colorbar(surf, ax=axes, shrink=0.5, aspect=20, pad=0.02)
    cbar.set_label(colorbar_label, fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(suptitle, fontsize=22, fontweight="bold", y=0.98)
    
    # Save if requested
    if savefig and save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
    
    # Show if requested
    if show:
        plt.show()
    







def plot_training_loss(losses):
    losses_np = jnp.asarray(losses)

    steps = np.arange(len(losses_np))
    plt.figure(figsize=(6, 4))
    plt.semilogy(steps, losses_np)
    plt.xlabel("Training step")
    plt.ylabel("Loss (log scale)")
    plt.title(f"PINN training loss — {len(losses_np)} steps")
    plt.grid(alpha=0.3)
    plt.show()
    return losses_np


# -----------------------------------------------------------------------------
# Part d: architecture sweep heatmaps
# -----------------------------------------------------------------------------
def plot_heatmap_width_depth(df, activation, show=True):
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

    if show:
        plt.show()

    return fig


# -----------------------------------------------------------------------------
# Part d: run all heatmaps
# -----------------------------------------------------------------------------
def plot_all_heatmaps(df, save_dir="figs", show=False):
    os.makedirs(save_dir, exist_ok=True)
    activations = df["activation"].unique()

    for act in activations:
        print(f"Creating heatmap for activation: {act}")
        fig = plot_heatmap_width_depth(df, activation=act, show=show)

        filename = f"heatmap_activation_{act}.pdf"
        filepath = os.path.join(save_dir, filename)

        fig.savefig(filepath, dpi=300, bbox_inches="tight")
        plt.close(fig)

        print(f"Saved: {filepath}")

