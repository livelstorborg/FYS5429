import matplotlib.pyplot as plt
import os
import jax.numpy as jnp
import numpy as np
from mpl_toolkits.mplot3d import Axes3D


# -----------------------------------------------------------------------------
# Part b: finite-difference comparisons
# -----------------------------------------------------------------------------
def plot_solution(x, u_num, u_true, title="", filepath="../figs/plot.pdf"):
    plt.plot(x, u_true, label="Analytical", color="red", linewidth=5, alpha=0.5)
    plt.plot(x, u_num, ":", label="Numerical", color="blue", linewidth=3)
    plt.grid(alpha=0.3)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.xlabel("x", fontsize=16)
    plt.ylabel("u(x, t)", fontsize=16)
    plt.title(title, fontsize=18, fontweight="bold")
    plt.legend(fontsize=16)
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.tight_layout()
    plt.show()


# -----------------------------------------------------------------------------
# Part b: FD scheme error curves
# -----------------------------------------------------------------------------
def plot_scheme_errors_t1(error_list, title, filepath):
    plt.figure(figsize=(8, 5))

    for err in error_list:
        x = err["x"]
        dx = err["dx"]
        plt.plot(x, err["t1_error"], label=rf"$\Delta x = {dx:.2f}$", linewidth=2)

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
# Part b: FD scheme error curves
# -----------------------------------------------------------------------------
def plot_scheme_errors_t2(error_list, title, filepath):
    plt.figure(figsize=(8, 5))

    for err in error_list:
        x = err["x"]
        dx = err["dx"]
        plt.plot(x, err["t2_error"], label=rf"$\Delta x = {dx:.2f}$", linewidth=2)

    plt.grid(alpha=0.3)
    plt.xlabel("x", fontsize=16)
    plt.ylabel("Error", fontsize=16)
    plt.title(title, fontsize=18, fontweight="bold")
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.yscale("log")
    plt.legend(fontsize=16)
    plt.tight_layout()
    plt.savefig(filepath, bbox_inches="tight", dpi=300)
    plt.show()


# -----------------------------------------------------------------------------
# Part c: 3D surfaces for solutions/errors
# -----------------------------------------------------------------------------
def plot_3d_surface(x, t, U, title="", elev=30, azim=-135, save_path=None):
    fig = plt.figure(figsize=(12, 10), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")

    X, T = np.meshgrid(x, t)
    surf = ax.plot_surface(X, T, U, cmap="plasma", edgecolor="none", alpha=0.8)

    # Labels
    ax.set_xlabel("x", fontsize=16, labelpad=12)
    ax.set_ylabel("t", fontsize=16, labelpad=12)
    ax.set_zlabel("u(x, t)", fontsize=16, labelpad=12)
    ax.set_title(title, fontsize=18, fontweight="bold", pad=5)

    for label in ax.zaxis.get_ticklabels():
        label.set_rotation(45)
        label.set_fontsize(16)

    ax.view_init(elev=elev, azim=azim)

    # Tick font sizes
    ax.tick_params(axis="both", which="major", labelsize=16)
    ax.tick_params(axis="z", which="major", labelsize=16)

    # Bigger colorbar
    cbar = fig.colorbar(
        surf,
        ax=ax,
        aspect=25,
        shrink=0.75,
    )
    cbar.set_label("u(x, t)", fontsize=16, labelpad=10)
    cbar.ax.tick_params(labelsize=16)
    cbar.formatter.set_powerlimits((0, 0))
    cbar.update_ticks()
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)

    return fig


def subplot_3d_surface(
    x,
    t,
    surfaces,
    elev=20,
    azims=None,
    save_path="code/figs/3d_subplot.pdf",
    title="3d surface plots",
):
    """
    Create a 1x4 row of 3D surface plots with a shared colorbar.
    surfaces: list of 4 (Nt x Nx) matrices.
    azims: list of 4 azimuths.
    """
    if azims is None:
        azims = [0, 45, 90, 135]

    fig, axes = plt.subplots(
        1,
        4,
        figsize=(25, 6),
        subplot_kw={"projection": "3d"},
        gridspec_kw={"wspace": 0.1},
    )

    X, T = np.meshgrid(x, t)

    for idx, (ax, U, az) in enumerate(zip(axes, surfaces, azims)):
        surf = ax.plot_surface(X, T, U, cmap="viridis", edgecolor="none", alpha=0.9)
        ax.view_init(elev=elev, azim=az)
        ax.grid(alpha=0.3)

        xticks = ax.get_xticks()
        yticks = ax.get_yticks()

        # -------------------------
        # 2. LABELS
        # -------------------------
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("t", fontsize=12)

        # -------------------------
        # 3. Z-axis (special handling)
        # -------------------------
        if idx == 0:
            # First subplot: show z-ticks + numbers + rotate them
            ax.tick_params(axis="z", labelsize=10)

            # Rotate z-axis tick labels
            for label in ax.get_zticklabels():
                label.set_rotation(45)
                label.set_fontsize(10)

        else:
            # Other subplots: keep ticks but remove numbers
            ax.set_zticklabels([""] * len(ax.get_zticks()))

        # -------------------------
        # 4. Set tick sizes on x/t axis
        # -------------------------
        ax.tick_params(axis="x", labelsize=10)
        ax.tick_params(axis="y", labelsize=10)

    # Shared colorbar
    cbar = fig.colorbar(surf, ax=axes.ravel().tolist(), shrink=0.5, aspect=20, pad=0.02)
    cbar.set_label("Error", fontsize=12)
    cbar.ax.tick_params(labelsize=10)

    fig.suptitle(title, fontsize=22, fontweight="bold", y=0.85)
    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight", pad_inches=0.1)
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
