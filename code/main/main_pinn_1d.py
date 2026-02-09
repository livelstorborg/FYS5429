import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import jax.numpy as jnp
import jax.nn as jnn
import numpy as np

from src.pde import fd_solve, u_exact, create_grid
from src.pinn import train_pinn
from src.experiment import absolute_error, relative_error, run_architecture_sweep
from src.plotting import (
    plot_solution_at_t,
    plot_scheme_error_at_t,
    plot_3d_surface,
    subplot_3d_surfaces,
)


Nx = 100
T = 3.0
c = 1.0
cfl = 0.4


x, t, dx, dt = create_grid(Nx=Nx, T=T, c=c, cfl=cfl, dim=1)

u_exact = u_exact(x, t, c=c, dim=1)
u_fd = fd_solve(x, t, dx, dt, c=c, dim=1)
u_pinn, losses, loss_components = train_pinn(
    steps=5000,
    layers=[2, 32, 32, 32, 1],
    activations=[jnn.silu, jnn.silu, jnn.silu],
    N_int=100,
    T=T,
    c=c,
    dim=1,
)


X_mesh, T_mesh = np.meshgrid(x, t)
xt_test = jnp.column_stack([X_mesh.ravel(), T_mesh.ravel()])
u_pinn_vals = u_pinn(xt_test).reshape(X_mesh.shape)


fig_exact = plot_3d_surface(
    x,
    t,
    u_exact,
    elev=20,
    azim=45,
    title="Analytical Solution (1D)",
    savefig=False,
    show=False,
)

fig_fd = plot_3d_surface(
    x,
    t,
    u_fd,
    elev=20,
    azim=45,
    title="Finite Difference Solution (1D)",
    savefig=False,
    show=False,
)

fig_pinn = plot_3d_surface(
    x,
    t,
    u_pinn_vals,
    elev=20,
    azim=45,
    title="PINN Solution (1D)",
    savefig=False,
    show=False,
)

subplot_fig = subplot_3d_surfaces(
    figures=[
        {'x': x, 't': t, 'U': u_exact},
        {'x': x, 't': t, 'U': u_fd},
        {'x': x, 't': t, 'U': u_pinn_vals},
    ],
    titles=["Analytical", "Finite Difference", "PINN (SiLU)"],
    elev=20,
    azims=[45, 45, 45],
    cmap="viridis",
    colorbar_label="u(x, t)",
    suptitle="Wave Equation Solutions",
    savefig=True,
    save_path="../figs/analytical_fd_pinn_silu_subplot.pdf",
    show=True,
)



run_architecture_sweep(
    hidden_widths=[32, 64],
    num_hidden_layers=[1, 2],
    activation_fns={
        'tanh': jnn.tanh,
        'sine': jnp.sin,
        # 'GeLU': jnn.gelu,
        # 'SiLU': jnn.swish,
        # 'ReLU': jnn.relu,
    },
    T=1.0,
    steps=1000,
    N_int=1000,
    lr=1e-3,
    seeds=(0,),
    Nx_eval=100,
    Ny_eval=None,
    Nt_eval=100,
    save_to_csv=True,
    use_pre_computed=False,
    data_dir="../data/1d_constant/",
)
