import jax.numpy as jnp
import jax.nn as jnn
import numpy as np

from src.pde import fd_solve, u_exact
from src.pinn import train_pinn
from src.experiment import test_explicit_scheme_at_t, absolute_error, relative_error
from src.plotting import plot_3d_surface, subplot_3d_surface


# Parameters for 1D wave equation
Nx = 100
T = 3
c = 1.0
cfl = 0.4
dim = 1

# Finite Difference scheme
u_fd, x, t = fd_solve(Nx=Nx, T=T, c=c, cfl=cfl, dim=dim)
u_true = u_exact(x, t=t, c=c, dim=dim)

# Train PINN with SiLU Activation
model_silu, losses_silu, loss_components_silu = train_pinn(
    steps=1000,
    layers=[2, 32, 32, 32, 1],
    activations=[jnn.silu, jnn.silu, jnn.silu],
    N_int=1000,
    T=T,
    c=c,
    dim=dim,
)

# Evaluate PINN on grid
X, T_grid = jnp.meshgrid(x, t)
xt = jnp.stack([X.ravel(), T_grid.ravel()], axis=1)
u_nn_silu = model_silu(xt).squeeze().reshape(len(t), len(x))

# Error surfaces
error_fd = np.abs(u_fd - u_true)
error_nn_silu = np.abs(u_nn_silu - u_true)

# Plotting
rotations = [45, 135, 225, 315]

fd_surfaces = [error_fd for _ in rotations]
pinn_surfaces_silu = [error_nn_silu for _ in rotations]

# Surface plots of solutions
plot_3d_surface(
    x,
    t,
    u_true,
    elev=20,
    azim=45,
    save_path="figs/exact_solution.pdf",
    title="Analytical Solution",
)

subplot_3d_surface(
    x,
    t,
    fd_surfaces,
    elev=20,
    azims=rotations,
    save_path="figs/fd_error_surfaces.pdf",
    title="Absolute Error of Finite Difference Scheme",
)

subplot_3d_surface(
    x,
    t,
    pinn_surfaces_silu,
    elev=20,
    azims=rotations,
    save_path="figs/pinn_error_surfaces_silu.pdf",
    title="Absolute Error of PINN Solution (SiLU)",
)