"""
pinn_wave_nnx.py

PINN for the 1D wave equation using Flax NNX + optax (Adam / AdamW / L-BFGS).

PDE:   u_tt = c^2 * u_xx,   x in [0,1], t in [0,T]
BCs:   u(0,t) = u(1,t) = 0  (hard, via sin(pi*x) factor)
ICs:   u(x,0) = sin(pi*x),  u_t(x,0) = 0  (soft penalty)
Exact: u(x,t) = sin(pi*x) * cos(c*pi*t)

Design follows the Poisson-solver pattern in 1D_poisson_nnx.py:
  - Flax NNX module whose nnx.Param leaves are the weight pytree
  - pack_params / apply_params for explicit param manipulation
  - forward_params for a pure-JAX forward pass (needed for nested autodiff)
  - optax.lbfgs() with value_and_grad_from_state (no time windows)
"""

import jax
jax.config.update("jax_enable_x64", True)   # L-BFGS requires float64

import flax.nnx as nnx
import jax.numpy as jnp
import optax


# =============================================================================
# Model
# =============================================================================

class MLP(nnx.Module):
    """
    MLP for the 1D wave PINN.

    Weights/biases are stored as nnx.Param (the JAX-pytree leaves).
    `activation` is a plain Python attribute — treated as a static compile-time
    constant by jax.jit (it is part of the pytree treedef, not a leaf).

    __call__ applies the hard BC/IC ansatz:
        u(x,t) = sin(pi*x) + t * sin(pi*x) * N(x,t)
    which guarantees u(0,t)=u(1,t)=0 and u(x,0)=sin(pi*x) exactly.
    Input xt: (N, 2)  →  Output: (N, 1).
    """

    def __init__(self, widths: tuple, *, key, activation=jax.nn.tanh):
        keys = jax.random.split(key, len(widths) + 1)
        dims = [2] + list(widths) + [1]
        self.ws = nnx.List()
        self.bs = nnx.List()
        self.activation = activation          # plain attr, static under jax.jit
        for k, din, dout in zip(keys, dims[:-1], dims[1:]):
            wk, bk = jax.random.split(k)
            W = jax.random.normal(wk, (din, dout)) * jnp.sqrt(2.0 / din)
            b = jnp.zeros((dout,), dtype=W.dtype)
            self.ws.append(nnx.Param(W))
            self.bs.append(nnx.Param(b))

    def __call__(self, xt):
        """Hard-BC/IC forward pass — used for evaluation and error metrics."""
        x = xt[:, 0]
        t = xt[:, 1]
        z = xt
        act = self.activation
        for W, b in zip(self.ws[:-1], self.bs[:-1]):
            z = act(z @ W + b)
        N = (z @ self.ws[-1] + self.bs[-1])[:, 0]        # (N,)
        u = jnp.sin(jnp.pi * x) + t * jnp.sin(jnp.pi * x) * N
        return u[:, None]                                  # (N, 1)


# =============================================================================
# Parameter helpers  (same API as 1D_poisson_nnx.py)
# =============================================================================

def pack_params(model):
    ws = tuple(jnp.asarray(W.value) for W in model.ws)
    bs = tuple(jnp.asarray(b.value) for b in model.bs)
    return (ws, bs)


def apply_params(model, params):
    ws, bs = params
    for Wi, bi, Wp, bp in zip(model.ws, model.bs, ws, bs):
        Wi.value = Wp
        bi.value = bp


# =============================================================================
# Pure-JAX forward pass  (used inside loss / autodiff)
# =============================================================================

def forward_params(params, xt2d, activation):
    """Raw network output — no hard BC.  xt2d: (N, 2) → (N, 1)."""
    ws, bs = params
    z = xt2d
    for W, b in zip(ws[:-1], bs[:-1]):
        z = activation(z @ W + b)
    return z @ ws[-1] + bs[-1]


def u_hat_params(params, xt2d, activation):
    """
    Additive hard-IC/BC ansatz:
        u(x,t) = sin(pi*x) + t * sin(pi*x) * N(x,t)
    Guarantees exactly:
        u(0,t) = u(1,t) = 0   (spatial BC, via sin factor)
        u(x,0) = sin(pi*x)    (displacement IC, additive u0 term)
    Only the velocity IC u_t(x,0)=0 needs a soft penalty.
    xt2d: (N, 2) → (N,)
    """
    x = xt2d[:, 0]
    t = xt2d[:, 1]
    N = forward_params(params, xt2d, activation)[:, 0]
    return jnp.sin(jnp.pi * x) + t * jnp.sin(jnp.pi * x) * N


def u_scalar_params(params, x, t, activation):
    """Scalar u at a single (x, t) point — entry point for jax.grad."""
    xt = jnp.stack([x, t])[None]           # (1, 2)
    return u_hat_params(params, xt, activation)[0]


# =============================================================================
# Derivatives  (scalar, for use inside vmap)
# =============================================================================

def u_t_params(params, x, t, activation):
    """du/dt at (x, t)."""
    return jax.grad(
        lambda t_: u_scalar_params(params, x, t_, activation)
    )(t)


def u_tt_params(params, x, t, activation):
    """d²u/dt² at (x, t)."""
    return jax.grad(
        jax.grad(lambda t_: u_scalar_params(params, x, t_, activation))
    )(t)


def u_xx_params(params, x, t, activation):
    """d²u/dx² at (x, t)."""
    return jax.grad(
        jax.grad(lambda x_: u_scalar_params(params, x_, t, activation))
    )(x)


# =============================================================================
# Training steps  (JIT-compiled; `opt`, `activation`, and `norm` are static)
# =============================================================================

def _train_step_adam(model, opt_state, x_int, t_int, x_ic, c, lambda_ic, opt,
                     norm, lambda_sob, lambda_sob2):
    try:
        from src.loss import loss_scalar
    except ModuleNotFoundError:
        from loss import loss_scalar
    params = pack_params(model)
    activation = model.activation
    loss, grads = jax.value_and_grad(
        lambda p: loss_scalar(
            p, x_int, t_int, x_ic, c, lambda_ic, activation,
            norm=norm, lambda_sob=lambda_sob, lambda_sob2=lambda_sob2,
        )
    )(params)
    updates, opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    apply_params(model, new_params)
    return model, opt_state, loss


train_step_adam = jax.jit(
    _train_step_adam, static_argnames=["opt", "norm"]
)


def _train_step_lbfgs(model, opt_state, x_int, t_int, x_ic, c, lambda_ic, opt,
                      norm, lambda_sob, lambda_sob2):
    try:
        from src.loss import loss_scalar
    except ModuleNotFoundError:
        from loss import loss_scalar
    params = pack_params(model)
    activation = model.activation

    def loss_fn_wrapped(p):
        return loss_scalar(
            p, x_int, t_int, x_ic, c, lambda_ic, activation,
            norm=norm, lambda_sob=lambda_sob, lambda_sob2=lambda_sob2,
        )

    value_and_grad_fn = optax.value_and_grad_from_state(loss_fn_wrapped)
    value, grads = value_and_grad_fn(params, state=opt_state)
    updates, opt_state = opt.update(
        grads,
        opt_state,
        params,
        value=value,
        grad=grads,
        value_fn=loss_fn_wrapped,
    )
    new_params = optax.apply_updates(params, updates)
    apply_params(model, new_params)
    return model, opt_state, value


train_step_lbfgs = jax.jit(
    _train_step_lbfgs, static_argnames=["opt", "norm"]
)


# =============================================================================
# Main training function
# =============================================================================

def train_wave_pinn(
    widths,
    *,
    activation=jax.nn.tanh,
    optimizer="adam",
    steps=5000,
    adam_warmup_steps=1000,
    init_params=None,
    N_int=1000,
    N_ic=100,
    T=1.0,
    c=1.0,
    lambda_ic=100.0,
    lr=1e-3,
    grad_clip=1.0,
    seed=0,
    log_every=500,
    norm="L2",
    lambda_sob=1.0,
    lambda_sob2=1.0,
):
    """
    Train a PINN for the 1D wave equation without time windows.

    Parameters
    ----------
    widths            : tuple of ints, hidden layer widths, e.g. (64, 64)
    activation        : JAX activation callable (static — one version compiled per fn)
    optimizer         : "adam" | "adamw" | "lbfgs"
    steps             : total optimisation steps (L-BFGS steps only, if warm-started)
    adam_warmup_steps : Adam steps run before L-BFGS (only used when optimizer="lbfgs", ignored if init_params is set)
    init_params       : optional (ws, bs) pytree — if given, model is initialised from these weights instead of random init
    N_int             : number of interior collocation points per step
    N_ic              : number of IC collocation points per step
    T                 : final time
    c                 : wave speed
    lambda_ic         : IC loss weight
    lr                : learning rate (used for Adam/AdamW and the L-BFGS warm-up)
    grad_clip         : global gradient norm clip for Adam/AdamW (set <=0 to disable)
    seed              : random seed
    log_every         : print/record interval
    norm              : "L2" | "H1" | "H2" — PDE loss norm (see loss_nnx.py)
    lambda_sob        : weight for ‖∇r‖² term (H1, H2)
    lambda_sob2       : weight for ‖H_r‖²_F term (H2 only)

    Returns
    -------
    model           : trained MLP  (call model(xt) for predictions)
    losses          : list[float], total loss at every step (warmup + L-BFGS combined)
    loss_components : list[dict],  component losses every log_every steps
    """
    key = jax.random.PRNGKey(seed)
    key, k_model = jax.random.split(key)

    model = MLP(widths, key=k_model, activation=activation)
    if init_params is not None:
        apply_params(model, init_params)

    c_arr           = jnp.array(c)
    lambda_ic_arr   = jnp.array(lambda_ic)
    lambda_sob_arr  = jnp.array(lambda_sob)
    lambda_sob2_arr = jnp.array(lambda_sob2)

    opt_lower = optimizer.lower()
    is_lbfgs  = opt_lower == "lbfgs"

    losses: list[float] = []
    loss_components: list[dict] = []

    # -------------------------------------------------------------------------
    # Adam warm-up  (always runs when optimizer="lbfgs" and adam_warmup_steps>0)
    # -------------------------------------------------------------------------
    if is_lbfgs and adam_warmup_steps > 0:
        warmup_schedule = optax.exponential_decay(lr, transition_steps=1000, decay_rate=0.95)
        warmup_base = optax.adam(warmup_schedule)
        if grad_clip is not None and grad_clip > 0:
            warmup_opt = optax.chain(optax.clip_by_global_norm(grad_clip), warmup_base)
        else:
            warmup_opt = warmup_base

        warmup_state = warmup_opt.init(pack_params(model))
        print(f"  Warm-starting with {adam_warmup_steps} Adam steps ...")

        for i in range(adam_warmup_steps):
            key, k_x, k_t, k_ic = jax.random.split(key, 4)
            x_int = jax.random.uniform(k_x,  (N_int,), minval=0.0, maxval=1.0)
            t_int = jax.random.uniform(k_t,  (N_int,), minval=0.0, maxval=T)
            x_ic  = jax.random.uniform(k_ic, (N_ic,),  minval=0.0, maxval=1.0)
            model, warmup_state, L = train_step_adam(
                model, warmup_state, x_int, t_int, x_ic,
                c_arr, lambda_ic_arr, warmup_opt,
                norm, lambda_sob_arr, lambda_sob2_arr,
            )
            losses.append(float(L))
            if (i + 1) % log_every == 0:
                print(f"  [warmup] step {i+1:5d} | loss={float(L):.3e}")

    # -------------------------------------------------------------------------
    # Main optimizer
    # -------------------------------------------------------------------------
    if opt_lower == "adam":
        schedule = optax.exponential_decay(lr, transition_steps=1000, decay_rate=0.95)
        base_opt = optax.adam(schedule)
    elif opt_lower == "adamw":
        schedule = optax.exponential_decay(lr, transition_steps=1000, decay_rate=0.95)
        base_opt = optax.adamw(schedule)
    elif opt_lower == "lbfgs":
        base_opt = optax.lbfgs()
    else:
        raise ValueError(f"Unknown optimizer {optimizer!r}. Use 'adam', 'adamw', or 'lbfgs'.")

    if not is_lbfgs and grad_clip is not None and grad_clip > 0:
        opt = optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
    else:
        opt = base_opt

    # L-BFGS requires fixed collocation points (consistent loss for Hessian approx)
    if is_lbfgs:
        key, k_x, k_t, k_ic = jax.random.split(key, 4)
        x_int_fixed = jax.random.uniform(k_x,  (N_int,), minval=0.0, maxval=1.0)
        t_int_fixed = jax.random.uniform(k_t,  (N_int,), minval=0.0, maxval=T)
        x_ic_fixed  = jax.random.uniform(k_ic, (N_ic,),  minval=0.0, maxval=1.0)

    opt_state = opt.init(pack_params(model))

    for i in range(steps):
        if is_lbfgs:
            model, opt_state, L = train_step_lbfgs(
                model, opt_state,
                x_int_fixed, t_int_fixed, x_ic_fixed,
                c_arr, lambda_ic_arr, opt,
                norm, lambda_sob_arr, lambda_sob2_arr,
            )
        else:
            key, k_x, k_t, k_ic = jax.random.split(key, 4)
            x_int = jax.random.uniform(k_x,  (N_int,), minval=0.0, maxval=1.0)
            t_int = jax.random.uniform(k_t,  (N_int,), minval=0.0, maxval=T)
            x_ic  = jax.random.uniform(k_ic, (N_ic,),  minval=0.0, maxval=1.0)
            model, opt_state, L = train_step_adam(
                model, opt_state, x_int, t_int, x_ic,
                c_arr, lambda_ic_arr, opt,
                norm, lambda_sob_arr, lambda_sob2_arr,
            )

        losses.append(float(L))

        if (i + 1) % log_every == 0:
            try:
                from src.loss import loss_fn
            except ModuleNotFoundError:
                from loss import loss_fn
            eval_x_int = x_int_fixed if is_lbfgs else x_int
            eval_t_int = t_int_fixed if is_lbfgs else t_int
            eval_x_ic  = x_ic_fixed  if is_lbfgs else x_ic
            params = pack_params(model)
            _, comps = loss_fn(
                params, eval_x_int, eval_t_int, eval_x_ic,
                c_arr, lambda_ic_arr, activation,
                norm=norm, lambda_sob=lambda_sob_arr, lambda_sob2=lambda_sob2_arr,
            )
            comps_f = {k: float(v) for k, v in comps.items()}
            loss_components.append(comps_f)
            log_line = (
                f"step {i+1:5d}: loss={float(L):.3e}"
                f" | pde={comps_f['pde']:.3e}"
                f" | ic_ut={comps_f['ic_ut']:.3e}"
            )
            if "sobolev" in comps_f:
                log_line += f" | sob={comps_f['sobolev']:.3e}"
            if "sobolev2" in comps_f:
                log_line += f" | sob2={comps_f['sobolev2']:.3e}"
            print(log_line)

    return model, losses, loss_components
