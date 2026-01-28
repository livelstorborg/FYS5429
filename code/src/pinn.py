import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import matplotlib.pyplot as plt

from src.pde import u_exact


# -----------------------------------------------------------------------------
# Models
# -----------------------------------------------------------------------------
class MLP(nnx.Module):
    """Standard MLP used inside the hard-BC trial solution."""

    def __init__(self, layers, activations, key):
        assert len(activations) == len(layers) - 2, (
            "Length of activations must be number of layers minus 2 "
            "(one activation per hidden layer)."
        )

        self.rngs = nnx.Rngs(params=key)
        self.layers = nnx.List(
            [
                nnx.Linear(layers[i], layers[i + 1], rngs=self.rngs)
                for i in range(len(layers) - 1)
            ]
        )

        self.activations = activations

    def __call__(self, x):
        for linear, act in zip(self.layers[:-1], self.activations):
            x = act(linear(x))
        return self.layers[-1](x)


class MLP_HardBC(nnx.Module):
    """
    MLP with hard boundary conditions.

    Trial solution: u(x,t) = (1-t)*sin(πx) + t*x*(1-x)*N(x,t)

    This automatically satisfies:
    - BC: u(0,t) = 0, u(1,t) = 0
    - IC: u(x,0) = sin(πx)
    """

    def __init__(self, layers, activations, key):
        self.network = MLP(layers, activations, key)

    def __call__(self, xt):
        """
        xt: (N, 2) array where xt[:, 0] = x, xt[:, 1] = t
        """
        x = xt[:, 0:1]
        t = xt[:, 1:2]

        # Get unconstrained network output
        N = self.network(xt)

        # Apply hard BC: u = (1-t)*sin(πx) + t*x*(1-x)*N
        u = (1.0 - t) * jnp.sin(jnp.pi * x) + t * x * (1.0 - x) * N

        return u


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
def pde_residual(model, xt, nu):
    def u_single(z):
        return model(z[None, :])[0, 0]

    jac = jax.vmap(jax.grad(u_single))(xt)
    u_x = jac[:, 0]
    u_t = jac[:, 1]

    def u_x_single(z):
        return jax.grad(u_single)(z)[0]

    u_xx = jax.vmap(jax.grad(u_x_single))(xt)[:, 0]

    return (u_t - nu * u_xx) ** 2


def loss_fn(model, x_int, t_int, nu: float = 1.0):
    """
    Loss for hard BC: only PDE residual needed!
    BCs and IC are automatically satisfied.
    """
    xt_int = jnp.concatenate([x_int, t_int], axis=1)
    loss_pde = pde_residual(model, xt_int, nu).mean()
    return loss_pde


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def sample_points(
    key,
    N_int=512,
    T=0.5,
    L=1.0,
):
    """
    Sample interior points only; boundary/initial conditions are enforced hard.
    """
    k1, k2 = jax.random.split(key, 2)

    x_int = jax.random.uniform(k1, (N_int, 1), minval=0.0, maxval=L)
    t_int = jax.random.uniform(k2, (N_int, 1), minval=0.0, maxval=T)

    return x_int, t_int


def train_pinn(
    layers=[2, 64, 64, 1],
    activations=None,
    steps=5000,
    N_int=1000,
    T=0.5,
    L=1.0,
    nu=1.0,
    lr=5e-4,
    seed=0,
):
    if activations is None:
        activations = [jax.nn.tanh] * (len(layers) - 2)

    main_key = jax.random.PRNGKey(seed)
    key_model, key_loop = jax.random.split(main_key)

    model = MLP_HardBC(layers, activations, key=key_model)

    schedule = optax.exponential_decay(
        init_value=lr,
        transition_steps=1000,
        decay_rate=0.95,
    )
    opt = nnx.Optimizer(model, optax.adam(schedule), wrt=nnx.Param)

    losses = []
    key = key_loop

    @jax.jit
    def train_step_hard(model, opt, key):
        """JIT-compiled step for hard BC"""
        key, subkey = jax.random.split(key)
        x_int, t_int = sample_points(subkey, N_int, T=T, L=L)

        def loss_func(m):
            return loss_fn(m, x_int, t_int, nu=nu)

        loss, grads = nnx.value_and_grad(loss_func)(model)
        opt.update(model, grads)
        return model, opt, key, loss

    for step in range(steps):
        model, opt, key, loss = train_step_hard(model, opt, key)
        losses.append(loss)

        if step % 500 == 0 or step == steps - 1:
            print(f"[step {step:4d}] loss = {float(loss):.3e}")

    return model, jnp.array(losses)


# -----------------------------------------------------------------------------
# Evaluation / utilities (PINN-focused)
# -----------------------------------------------------------------------------
def compare_nn_and_exact(model, Nx=200, Nt=100, T=1.0, return_only=False):
    x = jnp.linspace(0, 1, Nx + 1)
    t = jnp.linspace(0, T, Nt + 1)

    X, Tt = jnp.meshgrid(x, t)

    xt = jnp.stack([X.ravel(), Tt.ravel()], axis=1)
    u_pred_flat = model(xt)  # Shape: (N, 1)

    print(f"🔍 Evaluation debug:")
    print(f"  u_pred_flat shape: {u_pred_flat.shape}")
    print(
        f"  u_pred_flat range: [{float(u_pred_flat.min()):.6f}, {float(u_pred_flat.max()):.6f}]"
    )

    u_pred_flat = u_pred_flat.squeeze()  # Remove last dimension: (N,)
    u_pred = u_pred_flat.reshape(Nt + 1, Nx + 1)  # Shape: (Nt+1, Nx+1)

    print(f"  u_pred shape after reshape: {u_pred.shape}")
    print(f"  u_pred range: [{float(u_pred.min()):.6f}, {float(u_pred.max()):.6f}]")

    u_true = u_exact(x, t)
    print(f"  u_true shape: {u_true.shape}")
    print(f"  u_true range: [{float(u_true.min()):.6f}, {float(u_true.max()):.6f}]")

    error = jnp.abs(u_pred - u_true)

    if return_only:
        return u_pred, u_true, x, t

    plt.figure(figsize=(7, 5))
    plt.imshow(
        u_pred.T, extent=[0, 1, 0, T], origin="lower", aspect="auto"
    )  # Transpose for imshow
    plt.colorbar()
    plt.title("PINN prediction $u_\\theta(x,t)$")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.imshow(
        u_true.T, extent=[0, 1, 0, T], origin="lower", aspect="auto"
    )  # Transpose for imshow
    plt.colorbar()
    plt.title("Analytical solution $u(x,t)$")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.imshow(
        error.T, extent=[0, 1, 0, T], origin="lower", aspect="auto"
    )  # Transpose for imshow
    plt.colorbar()
    plt.title("Absolute error $|u_\\theta - u|$")
    plt.xlabel("x")
    plt.ylabel("t")
    plt.show()

    plt.figure(figsize=(7, 5))
    plt.plot(x, u_pred[:, -1], label="PINN")
    plt.plot(x, u_true[:, -1], "--", label="Exact")
    plt.title(f"Solution at t = {T}")
    plt.xlabel("x")
    plt.ylabel("u(x,T)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.show()

    return X, Tt, u_pred, u_true, error


def evaluate_pinn(model, x, t):
    """
    Evaluate trained PINN model on full spatio-temporal grid.
    x: (Nx,) array
    t: (Nt,) array
    Returns u_nn of shape (Nt, Nx)
    """
    X, Tgrid = jnp.meshgrid(x, t)
    xt = jnp.stack([X.flatten(), Tgrid.flatten()], axis=1)

    @jax.vmap
    def predict_single(z):
        return model(z[jnp.newaxis, :])[0]

    u_flat = predict_single(xt)
    return u_flat.reshape(len(t), len(x))


def compute_error_metrics(model, Nx=100, Nt=100, T=0.5):
    x = jnp.linspace(0.0, 1.0, Nx + 1)
    t = jnp.linspace(0.0, T, Nt + 1)
    X, Tt = jnp.meshgrid(x, t)

    xt = jnp.stack([X.ravel(), Tt.ravel()], axis=1)

    u_pred = model(xt).reshape(Nt + 1, Nx + 1)
    u_true = u_exact(x, t)

    error = u_pred - u_true
    abs_error = jnp.abs(error)

    num = jnp.sqrt(jnp.mean(error**2))
    den = jnp.sqrt(jnp.mean(u_true**2))
    L2_rel = num / den

    Linf = jnp.max(abs_error)

    return float(L2_rel), float(Linf)
