import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import matplotlib.pyplot as plt

from .pde import u_exact


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
    MLP with hard boundary conditions for WAVE equation.

    Trial solution: u(x,t) = sin(πx) * N(x,t)

    This automatically satisfies:
    - BC: u(0,t) = 0, u(1,t) = 0
    
    Note: Initial conditions are enforced through loss function
    """

    def __init__(self, layers, activations, key):
        self.network = MLP(layers, activations, key)

    def __call__(self, xt):
        """
        xt: (N, 2) array where xt[:, 0] = x, xt[:, 1] = t
        """
        x = xt[:, 0:1]

        # Get unconstrained network output
        N = self.network(xt)

        # Apply hard BC: u = sin(πx)*N(x,t)
        # This guarantees u(0,t) = u(1,t) = 0
        u = jnp.sin(jnp.pi * x) * N

        return u


# -----------------------------------------------------------------------------
# Loss
# -----------------------------------------------------------------------------
def pde_residual(model, xyzt, c, dim):
    """
    Compute PDE residual for wave equation in 1D or 2D.
    
    Wave equation:
    - 1D: ∂²u/∂t² - c²∂²u/∂x² = 0
    - 2D: ∂²u/∂t² - c²(∂²u/∂x² + ∂²u/∂y²) = 0
    
    Parameters:
    -----------
    model : neural network
    xyzt : (N, dim+1) array
        - 1D: [x, t]
        - 2D: [x, y, t]
    c : float
        Wave speed
    dim : int (1 or 2)
        Spatial dimension
    
    Returns:
    --------
    residual² : (N,) array of squared residuals
    """
    def u_single(z):
        """u evaluated at single point"""
        return model(z[None, :])[0, 0]

    # Compute Hessian (all second derivatives)
    def hessian_single(z):
        """Compute Hessian [∂²u/∂z_i∂z_j] at point z"""
        return jax.hessian(u_single)(z)
    
    # Vectorize over batch
    hess = jax.vmap(hessian_single)(xyzt)  # Shape: (N, dim+1, dim+1)
    
    # Extract second derivatives based on dimension
    if dim == 1:
        # xyzt = [x, t]
        u_xx = hess[:, 0, 0]  # ∂²u/∂x²
        u_tt = hess[:, 1, 1]  # ∂²u/∂t²
        laplacian = u_xx
        
    elif dim == 2:
        # xyzt = [x, y, t]
        u_xx = hess[:, 0, 0]  # ∂²u/∂x²
        u_yy = hess[:, 1, 1]  # ∂²u/∂y²
        u_tt = hess[:, 2, 2]  # ∂²u/∂t²
        laplacian = u_xx + u_yy
        
    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")

    # Wave equation residual: ∂²u/∂t² - c²∇²u
    residual = u_tt - c**2 * laplacian
    
    return residual ** 2


def loss_fn(model, x_int, t_int, x_ic, t_ic, c=1.0, lambda_ic=10.0, dim=1):
    """
    Loss for wave equation PINN with hard BC.
    
    Components:
    1. PDE residual in interior
    2. Initial condition u(x,0) = sin(πx) (soft constraint)
    3. Initial velocity ∂u/∂t(x,0) = 0 (soft constraint)
    
    Hard BC u(0,t) = u(1,t) = 0 is enforced by architecture.
    """
    # 1. PDE residual
    xt_int = jnp.concatenate([x_int, t_int], axis=1)
    loss_pde = pde_residual(model, xt_int, c, dim).mean()

    # 2. Initial condition: u(x,0) = sin(πx)
    xt_ic = jnp.concatenate([x_ic, t_ic], axis=1)
    u_ic_pred = model(xt_ic)
    u_ic_true = jnp.sin(jnp.pi * x_ic)
    loss_ic_u = jnp.mean((u_ic_pred - u_ic_true) ** 2)

    # 3. Initial velocity: ∂u/∂t(x,0) = 0
    def u_single(z):
        return model(z[None, :])[0, 0]
    
    u_t_ic = jax.vmap(lambda z: jax.grad(u_single)(z)[1])(xt_ic)
    loss_ic_ut = jnp.mean(u_t_ic ** 2)

    # Total loss
    loss_total = loss_pde + lambda_ic * (loss_ic_u + loss_ic_ut)
    
    return loss_total, {
        'pde': loss_pde,
        'ic_u': loss_ic_u,
        'ic_ut': loss_ic_ut,
        'total': loss_total
    }


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def sample_points_wave(key, N_int=512, N_ic=100, T=2.0, L=1.0):
    """
    Sample points for wave equation:
    - Interior points (x,t) for PDE residual
    - Initial condition points (x,0) for IC
    """
    k1, k2, k3 = jax.random.split(key, 3)

    # Interior points
    x_int = jax.random.uniform(k1, (N_int, 1), minval=0.0, maxval=L)
    t_int = jax.random.uniform(k2, (N_int, 1), minval=0.0, maxval=T)

    # Initial condition points (t=0)
    x_ic = jax.random.uniform(k3, (N_ic, 1), minval=0.0, maxval=L)
    t_ic = jnp.zeros((N_ic, 1))

    return x_int, t_int, x_ic, t_ic


def train_pinn(
    layers=[2, 64, 64, 64, 1],
    activations=None,
    steps=10000,
    N_int=1000,
    N_ic=100,
    T=2.0,
    L=1.0,
    c=1.0,
    lambda_ic=10.0,
    lr=1e-3,
    seed=0,
    dim=1,
):
    """
    Train PINN for wave equation in 1D or 2D.
    """
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
    loss_components = {'pde': [], 'ic_u': [], 'ic_ut': []}
    key = key_loop

    @jax.jit
    def train_step(model, opt, key):
        """JIT-compiled training step"""
        key, subkey = jax.random.split(key)
        x_int, t_int, x_ic, t_ic = sample_points_wave(
            subkey, N_int, N_ic, T=T, L=L
        )

        def loss_func(m):
            loss_val, components = loss_fn(
                m, x_int, t_int, x_ic, t_ic, c=c, lambda_ic=lambda_ic, dim=dim
            )
            return loss_val, components

        (loss, components), grads = nnx.value_and_grad(
            loss_func, has_aux=True
        )(model)
        opt.update(model, grads)
        
        return model, opt, key, loss, components

    print(f"Training PINN for {dim}D wave equation (c={c}, T={T})...")
    print(f"Architecture: {layers}")
    print(f"Steps: {steps}, N_int: {N_int}, N_ic: {N_ic}")
    print("-" * 60)

    for step in range(steps):
        model, opt, key, loss, components = train_step(model, opt, key)
        
        losses.append(float(loss))
        for k in ['pde', 'ic_u', 'ic_ut']:
            loss_components[k].append(float(components[k]))

        if step % 500 == 0 or step == steps - 1:
            print(
                f"[{step:5d}] loss={float(loss):.3e} | "
                f"pde={float(components['pde']):.3e} | "
                f"ic_u={float(components['ic_u']):.3e} | "
                f"ic_ut={float(components['ic_ut']):.3e}"
            )

    return model, jnp.array(losses), loss_components




def compute_error_metrics(model, x, y=None, z=None, t=None, c=1.0, dim=1):
    """
    Compute L² relative and L∞ errors for PINN in 1D or 2D.
    
    Parameters:
    -----------
    model : PINN model
        Trained neural network
    x : array
        x-coordinates
    y : array, optional
        y-coordinates (required for dim=2)
    z : array, optional
        Not used (kept for compatibility)
    t : array
        time coordinates
    c : float
        wave speed
    dim : int (1 or 2)
        Spatial dimension
    
    Returns:
    --------
    L2_rel : float
        Relative L² error
    Linf : float
        L∞ error
    """

    if dim == 1:
        u_true = u_exact(x, t=t, c=c, dim=1)
        
        X, T = jnp.meshgrid(x, t)
        xt = jnp.stack([X.ravel(), T.ravel()], axis=1)
        u_pred = model(xt).squeeze().reshape(len(t), len(x))
        
    elif dim == 2:
        if y is None:
            raise ValueError("y must be provided for 2D")
        
        u_true = u_exact(x, y=y, t=t, c=c, dim=2)
        
        u_pred_list = []
        for ti in t:
            X, Y = jnp.meshgrid(x, y, indexing='ij')
            T_grid = jnp.full_like(X, ti)
            xyt = jnp.stack([X.ravel(), Y.ravel(), T_grid.ravel()], axis=1)
            u_pred_t = model(xyt).squeeze().reshape(len(x), len(y))
            u_pred_list.append(u_pred_t)
        u_pred = jnp.stack(u_pred_list, axis=0)
    
    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")
    
    # Compute errors
    error = u_pred - u_true
    
    # L² relative error
    L2_rel = jnp.sqrt(jnp.mean(error**2)) / jnp.sqrt(jnp.mean(u_true**2))
    
    # L∞ error
    Linf = jnp.max(jnp.abs(error))
    
    return float(L2_rel), float(Linf)


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 60)
    print("Training PINN for 1D Wave Equation")
    print("=" * 60)
    
    model, losses, loss_comps = train_pinn(
        layers=[2, 64, 64, 64, 1],
        steps=5000,
        N_int=1000,
        N_ic=100,
        T=2.0,
        c=1.0,
        lambda_ic=10.0,
        lr=1e-3,
        seed=42,
    )
    
    # Plot training loss
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.semilogy(losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.semilogy(loss_comps['pde'], label='PDE')
    plt.semilogy(loss_comps['ic_u'], label='IC u')
    plt.semilogy(loss_comps['ic_ut'], label='IC ∂u/∂t')
    plt.xlabel("Step")
    plt.ylabel("Loss component")
    plt.title("Loss Components")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('pinn_wave_training.png', dpi=150)
    plt.show()
    
    # Compare with exact solution
    compare_nn_and_exact(model, c=1.0, Nx=200, Nt=100, T=2.0)
    
    # Compute errors
    L2_rel, Linf = compute_error_metrics(model, c=1.0, Nx=100, Nt=100, T=2.0)
    print(f"\n{'='*60}")
    print(f"Error metrics:")
    print(f"  L2 relative error: {L2_rel:.6e}")
    print(f"  L∞ error: {Linf:.6e}")
    print(f"{'='*60}")