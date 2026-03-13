import matplotlib.pyplot as plt
import os
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from pathlib import Path
from mpl_toolkits.mplot3d import Axes3D
import flax.nnx as nnx


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
    savefig=False,
    filepath=None,
    show=False,
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
    if savefig and filepath:
        plt.savefig(filepath, dpi=300, bbox_inches="tight")
    if show:
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
    if savefig and filepath:
        fig.savefig(filepath, dpi=300, bbox_inches="tight", pad_inches=0.1)
    
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
    if savefig and filepath:
        fig.savefig(filepath, dpi=300, bbox_inches="tight", pad_inches=0.1)
    
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


def plot_loss(losses, show=True, savefig=False, filepath=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.semilogy(losses)
    ax.set_xlabel("Step (across windows)")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True)
    plt.tight_layout()
    if savefig and filepath:
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


def plot_loss_components(loss_comps, show=True, savefig=False, filepath=None):
    fig, ax = plt.subplots(figsize=(6, 4))
    for key, label in [("pde", "PDE"), ("ic_ut", "IC ∂u/∂t"), ("sobolev", "Sobolev")]:
        if key in loss_comps and any(v > 0 for v in loss_comps[key]):
            ax.semilogy(loss_comps[key], label=label)
    ax.set_xlabel("Step (across windows)")
    ax.set_ylabel("Loss component")
    ax.set_title("Loss Components")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    if savefig and filepath:
        fig.savefig(filepath, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
    return fig


# -----------------------------------------------------------------------------
# Part d: architecture sweep heatmaps
# -----------------------------------------------------------------------------
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


# -----------------------------------------------------------------------------
# Part d: run all heatmaps
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# PINN error reports
# -----------------------------------------------------------------------------
def error_report_1d_wave(
    model,
    u_exact_fn,
    times,
    *,
    c=1.0,
    L=1.0,
    Nx=201,
    make_snapshot=True,
    snapshot_metric="relL2",
    logy=True,
    title_prefix="",
):
    times = jnp.asarray(times, dtype=float)
    x = jnp.linspace(0.0, L, Nx)

    @nnx.jit
    def eval_at_time(t):
        t = jnp.asarray(t)
        xt = jnp.stack([x, jnp.full_like(x, t)], axis=1)

        u_pred = model(xt).squeeze()
        u_true = u_exact_fn(x, jnp.array([t]), c=c)[0]  # (Nx,)

        err = u_pred - u_true
        relL2 = jnp.sqrt(jnp.mean(err**2)) / (jnp.sqrt(jnp.mean(u_true**2)) + 1e-12)
        Linf = jnp.max(jnp.abs(err))
        mae = jnp.mean(jnp.abs(err))
        rmse = jnp.sqrt(jnp.mean(err**2))

        # H1 semi-norm: error in du/dx
        def u_pred_scalar(xt_single):
            return model(xt_single[None]).squeeze()

        dudx_pred = jax.vmap(jax.jacfwd(u_pred_scalar))(xt)[:, 0]  # (Nx,)
        dudx_true = jnp.gradient(u_true, x)
        err_dx = dudx_pred - dudx_true
        h1 = jnp.sqrt(jnp.mean(err**2) + jnp.mean(err_dx**2))

        return relL2, Linf, mae, rmse, h1, u_pred, u_true, err

    relL2_list, Linf_list, mae_list, rmse_list, h1_list = [], [], [], [], []
    snaps = {}

    for t in times:
        relL2, Linf, mae, rmse, h1, u_pred, u_true, err = eval_at_time(t)
        relL2_list.append(float(relL2))
        Linf_list.append(float(Linf))
        mae_list.append(float(mae))
        rmse_list.append(float(rmse))
        h1_list.append(float(h1))

        if make_snapshot:
            snaps[float(t)] = {
                "u_pred": jnp.array(u_pred),
                "u_true": jnp.array(u_true),
                "err": jnp.array(err),
            }

    relL2_arr = jnp.array(relL2_list)
    Linf_arr = jnp.array(Linf_list)
    mae_arr = jnp.array(mae_list)
    rmse_arr = jnp.array(rmse_list)
    h1_arr = jnp.array(h1_list)

    worst_rel_idx = int(jnp.argmax(relL2_arr))
    worst_inf_idx = int(jnp.argmax(Linf_arr))

    summary = {
        "relL2_mean": float(relL2_arr.mean()),
        "relL2_median": float(jnp.median(relL2_arr)),
        "relL2_max": float(relL2_arr[worst_rel_idx]),
        "relL2_max_time": float(times[worst_rel_idx]),
        "Linf_mean": float(Linf_arr.mean()),
        "Linf_max": float(Linf_arr[worst_inf_idx]),
        "Linf_max_time": float(times[worst_inf_idx]),
        "mae_mean": float(mae_arr.mean()),
        "rmse_mean": float(rmse_arr.mean()),
        "h1_mean": float(h1_arr.mean()),
    }

    plt.figure(figsize=(8, 4.8))
    plt.plot(times, relL2_arr, marker="o", label="Relative L2")
    plt.plot(times, Linf_arr, marker="s", label="L∞")
    plt.plot(times, mae_arr, marker="^", label="MAE")
    plt.plot(times, rmse_arr, marker="D", label="RMSE")
    plt.plot(times, h1_arr, marker="v", label="H1")
    plt.xlabel("Time t")
    plt.ylabel("Error")
    ttl = "Error metrics vs time"
    if title_prefix:
        ttl = f"{title_prefix} — {ttl}"
    plt.title(ttl)
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if make_snapshot and len(snaps) > 0:
        if snapshot_metric == "Linf":
            t_snap = float(times[worst_inf_idx])
            snap_label = "worst L∞"
        else:
            t_snap = float(times[worst_rel_idx])
            snap_label = "worst rel L2"

        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(x, snaps[t_snap]["u_true"], label="u_true")
        plt.plot(x, snaps[t_snap]["u_pred"], "--", label="u_pred")
        plt.title(f"Solution at t={t_snap:.4f} ({snap_label})")
        plt.xlabel("x")
        plt.legend()
        plt.grid(True, alpha=0.3)

        plt.subplot(1, 2, 2)
        plt.plot(x, jnp.abs(snaps[t_snap]["err"]))
        plt.title(f"|error| at t={t_snap:.4f}")
        plt.xlabel("x")
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

    print("Accuracy summary:")
    print(
        f"  relL2: mean={summary['relL2_mean']:.3e}, median={summary['relL2_median']:.3e}, "
        f"max={summary['relL2_max']:.3e} at t={summary['relL2_max_time']:.4f}"
    )
    print(
        f"  Linf : mean={summary['Linf_mean']:.3e}, max={summary['Linf_max']:.3e} at t={summary['Linf_max_time']:.4f}"
    )
    print(f"  MAE  : mean={summary['mae_mean']:.3e}")
    print(f"  RMSE : mean={summary['rmse_mean']:.3e}")
    print(f"  H1   : mean={summary['h1_mean']:.3e}")

    return {
        "times": times,
        "relL2": relL2_arr,
        "Linf": Linf_arr,
        "mae": mae_arr,
        "rmse": rmse_arr,
        "h1": h1_arr,
        "summary": summary,
    }


def error_report_2d_wave(
    model,
    u_exact_fn,
    times,
    *,
    c=1.0,
    L=1.0,
    Nx=81,
    Ny=81,
    make_snapshot=True,
    snapshot_metric="relL2",
    logy=True,
    title_prefix="",
):
    times = jnp.asarray(times, dtype=float)

    x = jnp.linspace(0.0, L, Nx)
    y = jnp.linspace(0.0, L, Ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    XY = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    @nnx.jit
    def eval_at_time(t):
        t = jnp.asarray(t)
        T = jnp.full((XY.shape[0], 1), t)
        xyt = jnp.concatenate([XY, T], axis=1)

        u_pred = model(xyt).reshape(Nx, Ny)
        u_true = u_exact_fn(X, Y, t, c=c)

        err = u_pred - u_true
        relL2 = jnp.sqrt(jnp.mean(err**2)) / (jnp.sqrt(jnp.mean(u_true**2)) + 1e-12)
        Linf = jnp.max(jnp.abs(err))
        mae = jnp.mean(jnp.abs(err))
        rmse = jnp.sqrt(jnp.mean(err**2))

        # H1 semi-norm: error in du/dx and du/dy
        def u_pred_scalar(xyt_single):
            return model(xyt_single[None]).squeeze()

        jac = jax.vmap(jax.jacfwd(u_pred_scalar))(xyt)  # (Nx*Ny, 3)
        dudx_pred = jac[:, 0].reshape(Nx, Ny)
        dudy_pred = jac[:, 1].reshape(Nx, Ny)
        dudx_true = jnp.gradient(u_true, x, axis=0)
        dudy_true = jnp.gradient(u_true, y, axis=1)
        err_dx = dudx_pred - dudx_true
        err_dy = dudy_pred - dudy_true
        h1 = jnp.sqrt(jnp.mean(err**2) + jnp.mean(err_dx**2) + jnp.mean(err_dy**2))

        return relL2, Linf, mae, rmse, h1, u_pred, u_true, err

    relL2_list, Linf_list, mae_list, rmse_list, h1_list = [], [], [], [], []
    snaps = {}

    for t in times:
        relL2, Linf, mae, rmse, h1, u_pred, u_true, err = eval_at_time(t)
        relL2_list.append(float(relL2))
        Linf_list.append(float(Linf))
        mae_list.append(float(mae))
        rmse_list.append(float(rmse))
        h1_list.append(float(h1))

        if make_snapshot:
            snaps[float(t)] = {
                "u_pred": jnp.array(u_pred),
                "u_true": jnp.array(u_true),
                "err": jnp.array(err),
            }

    relL2_arr = jnp.array(relL2_list)
    Linf_arr = jnp.array(Linf_list)
    mae_arr = jnp.array(mae_list)
    rmse_arr = jnp.array(rmse_list)
    h1_arr = jnp.array(h1_list)

    worst_rel_idx = int(jnp.argmax(relL2_arr))
    worst_inf_idx = int(jnp.argmax(Linf_arr))

    summary = {
        "relL2_mean": float(relL2_arr.mean()),
        "relL2_median": float(jnp.median(relL2_arr)),
        "relL2_max": float(relL2_arr[worst_rel_idx]),
        "relL2_max_time": float(times[worst_rel_idx]),
        "Linf_mean": float(Linf_arr.mean()),
        "Linf_max": float(Linf_arr[worst_inf_idx]),
        "Linf_max_time": float(times[worst_inf_idx]),
        "mae_mean": float(mae_arr.mean()),
        "rmse_mean": float(rmse_arr.mean()),
        "h1_mean": float(h1_arr.mean()),
    }

    plt.figure(figsize=(8, 4.8))
    plt.plot(times, relL2_arr, marker="o", label="Relative L2")
    plt.plot(times, Linf_arr, marker="s", label="L∞")
    plt.plot(times, mae_arr, marker="^", label="MAE")
    plt.plot(times, rmse_arr, marker="D", label="RMSE")
    plt.plot(times, h1_arr, marker="v", label="H1")
    plt.xlabel("Time t")
    plt.ylabel("Error")
    ttl = "Error metrics vs time"
    if title_prefix:
        ttl = f"{title_prefix} — {ttl}"
    plt.title(ttl)
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if make_snapshot and len(snaps) > 0:
        if snapshot_metric == "Linf":
            t_snap = float(times[worst_inf_idx])
            snap_label = "worst L∞"
        else:
            t_snap = float(times[worst_rel_idx])
            snap_label = "worst rel L2"

        err = snaps[t_snap]["err"]
        abs_err = jnp.abs(err)

        plt.figure(figsize=(15, 4.5))

        plt.subplot(1, 3, 1)
        plt.imshow(snaps[t_snap]["u_true"], origin="lower", aspect="auto")
        plt.title(f"u_true at t={t_snap:.4f}")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(snaps[t_snap]["u_pred"], origin="lower", aspect="auto")
        plt.title(f"u_pred at t={t_snap:.4f}")
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow(abs_err, origin="lower", aspect="auto")
        plt.title(f"|error| at t={t_snap:.4f} ({snap_label})")
        plt.colorbar()

        plt.tight_layout()
        plt.show()

    print("Accuracy summary:")
    print(
        f"  relL2: mean={summary['relL2_mean']:.3e}, median={summary['relL2_median']:.3e}, "
        f"max={summary['relL2_max']:.3e} at t={summary['relL2_max_time']:.4f}"
    )
    print(
        f"  Linf : mean={summary['Linf_mean']:.3e}, max={summary['Linf_max']:.3e} at t={summary['Linf_max_time']:.4f}"
    )
    print(f"  MAE  : mean={summary['mae_mean']:.3e}")
    print(f"  RMSE : mean={summary['rmse_mean']:.3e}")
    print(f"  H1   : mean={summary['h1_mean']:.3e}")

    return {
        "times": times,
        "relL2": relL2_arr,
        "Linf": Linf_arr,
        "mae": mae_arr,
        "rmse": rmse_arr,
        "h1": h1_arr,
        "summary": summary,
    }


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

