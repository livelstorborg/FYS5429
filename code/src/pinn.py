"""
pinn_wave_nnx.py

PINN for the 1D wave equation using Flax NNX + optax (Adam / AdamW / L-BFGS).

PDE:   u_tt = c^2 * u_xx,   x in [0,1], t in [0,T]
BCs:   u(0,t) = u(1,t) = 0  (hard, via sin(k*pi*x) factor)
ICs:   u(x,0) = sin(k*pi*x),  u_t(x,0) = 0  (soft penalty)
Exact: u(x,t) = sin(k*pi*x) * cos(c*k*pi*t)

The `k` parameter sets the wavenumber in the trial function.  Defaults to k=1.

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
    `activation` and `k` are plain Python attributes — treated as static
    compile-time constants by jax.jit.

    __call__ applies the hard BC/IC ansatz:
        u(x,t) = sin(k*pi*x) + t * sin(k*pi*x) * N(x,t)
    which guarantees u(0,t)=u(1,t)=0 and u(x,0)=sin(k*pi*x) exactly.
    Input xt: (N, 2)  →  Output: (N, 1).
    """

    def __init__(self, widths: tuple, *, key, activation=jax.nn.tanh, k=1, fourier_freqs=None):
        # fourier_freqs: list of frequencies for Fourier feature embedding of (x, t).
        self.fourier_freqs = tuple(fourier_freqs) if fourier_freqs is not None else None
        in_dim = 2 if fourier_freqs is None else 2 + 4 * len(fourier_freqs)

        keys = jax.random.split(key, len(widths) + 1)
        dims = [in_dim] + list(widths) + [1]
        self.ws = nnx.List()
        self.bs = nnx.List()
        self.activation = activation          # plain attr, static under jax.jit
        self.k = k                            # plain attr, static under jax.jit
        for k_, din, dout in zip(keys, dims[:-1], dims[1:]):
            wk, bk = jax.random.split(k_)
            W = jax.random.normal(wk, (din, dout)) * jnp.sqrt(2.0 / din)
            b = jnp.zeros((dout,), dtype=W.dtype)
            self.ws.append(nnx.Param(W))
            self.bs.append(nnx.Param(b))

    def _encode(self, xt):
        if self.fourier_freqs is None:
            return xt
        freqs = jnp.array(self.fourier_freqs)   # (F,)
        x = xt[:, 0:1]                          # (N, 1)
        t = xt[:, 1:2]                          # (N, 1)
        args_x = 2 * jnp.pi * freqs * x        # (N, F)
        args_t = 2 * jnp.pi * freqs * t        # (N, F)
        return jnp.concatenate(
            [xt, jnp.cos(args_x), jnp.sin(args_x), jnp.cos(args_t), jnp.sin(args_t)],
            axis=1,
        )

    def __call__(self, xt):
        """Hard-BC/IC forward pass — used for evaluation and error metrics."""
        x = xt[:, 0]
        t = xt[:, 1]
        z = self._encode(xt)
        act = self.activation
        for W, b in zip(self.ws[:-1], self.bs[:-1]):
            z = act(z @ W + b)
        N = (z @ self.ws[-1] + self.bs[-1])[:, 0]        # (N,)
        sx = jnp.sin(self.k * jnp.pi * x)
        u = sx + t * sx * N
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


def u_hat_params(params, xt2d, activation, k):
    """
    Additive hard-IC/BC ansatz:
        u(x,t) = sin(k*pi*x) + t * sin(k*pi*x) * N(x,t)
    Guarantees exactly:
        u(0,t) = u(1,t) = 0   (spatial BC, via sin factor)
        u(x,0) = sin(k*pi*x)  (displacement IC, additive u0 term)
    Only the velocity IC u_t(x,0)=0 needs a soft penalty.
    xt2d: (N, 2) → (N,)
    """
    x = xt2d[:, 0]
    t = xt2d[:, 1]
    N = forward_params(params, xt2d, activation)[:, 0]
    sx = jnp.sin(k * jnp.pi * x)
    return sx + t * sx * N


def u_scalar_params(params, x, t, activation, k):
    """Scalar u at a single (x, t) point — entry point for jax.grad."""
    xt = jnp.stack([x, t])[None]           # (1, 2)
    return u_hat_params(params, xt, activation, k)[0]


# =============================================================================
# Derivatives  (scalar, for use inside vmap)
# =============================================================================

def u_t_params(params, x, t, activation, k):
    """du/dt at (x, t)."""
    return jax.grad(
        lambda t_: u_scalar_params(params, x, t_, activation, k)
    )(t)


def u_tt_params(params, x, t, activation, k):
    """d²u/dt² at (x, t)."""
    return jax.grad(
        jax.grad(lambda t_: u_scalar_params(params, x, t_, activation, k))
    )(t)


def u_xx_params(params, x, t, activation, k):
    """d²u/dx² at (x, t)."""
    return jax.grad(
        jax.grad(lambda x_: u_scalar_params(params, x_, t, activation, k))
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
    k = model.k
    loss, grads = jax.value_and_grad(
        lambda p: loss_scalar(
            p, x_int, t_int, x_ic, c, lambda_ic, activation,
            norm=norm, lambda_sob=lambda_sob, lambda_sob2=lambda_sob2, k=k,
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
    k = model.k

    def loss_fn_wrapped(p):
        return loss_scalar(
            p, x_int, t_int, x_ic, c, lambda_ic, activation,
            norm=norm, lambda_sob=lambda_sob, lambda_sob2=lambda_sob2, k=k,
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
# 2D wave PINN  (u_tt = c²(u_xx + u_yy),  (x,y) ∈ [0,1]², t ∈ [0,T])
# =============================================================================

class MLP2d(nnx.Module):
    """
    MLP for the 2D wave PINN.

    Hard-BC/IC ansatz:
        u(x,y,t) = sin(k·πx)sin(k·πy) + t·sin(k·πx)sin(k·πy)·N(x,y,t)
    Guarantees u=0 on all four walls and u(x,y,0)=sin(k·πx)sin(k·πy) exactly.
    Input xyt: (N, 3) → Output: (N, 1).

    fourier_freqs: optional list of frequencies for Fourier feature embedding
                  of (x, y, t).  Input dim becomes 3 + 6·len(fourier_freqs).
    """

    def __init__(self, widths: tuple, *, key, activation=jax.nn.tanh, k=1, fourier_freqs=None):
        self.fourier_freqs = tuple(fourier_freqs) if fourier_freqs is not None else None
        in_dim = 3 if fourier_freqs is None else 3 + 6 * len(fourier_freqs)
        keys = jax.random.split(key, len(widths) + 1)
        dims = [in_dim] + list(widths) + [1]
        self.ws = nnx.List()
        self.bs = nnx.List()
        self.activation = activation
        self.k = k
        for k_, din, dout in zip(keys, dims[:-1], dims[1:]):
            wk, bk = jax.random.split(k_)
            W = jax.random.normal(wk, (din, dout)) * jnp.sqrt(2.0 / din)
            b = jnp.zeros((dout,), dtype=W.dtype)
            self.ws.append(nnx.Param(W))
            self.bs.append(nnx.Param(b))

    def _encode(self, xyt):
        if self.fourier_freqs is None:
            return xyt
        freqs = jnp.array(self.fourier_freqs)   # (F,)
        x = xyt[:, 0:1]
        y = xyt[:, 1:2]
        t = xyt[:, 2:3]
        args_x = 2 * jnp.pi * freqs * x
        args_y = 2 * jnp.pi * freqs * y
        args_t = 2 * jnp.pi * freqs * t
        return jnp.concatenate(
            [xyt,
             jnp.cos(args_x), jnp.sin(args_x),
             jnp.cos(args_y), jnp.sin(args_y),
             jnp.cos(args_t), jnp.sin(args_t)],
            axis=1,
        )

    def __call__(self, xyt):
        x = xyt[:, 0]
        y = xyt[:, 1]
        t = xyt[:, 2]
        z = self._encode(xyt)
        act = self.activation
        for W, b in zip(self.ws[:-1], self.bs[:-1]):
            z = act(z @ W + b)
        N = (z @ self.ws[-1] + self.bs[-1])[:, 0]
        spatial = jnp.sin(self.k * jnp.pi * x) * jnp.sin(self.k * jnp.pi * y)
        u = spatial + t * spatial * N
        return u[:, None]   # (N, 1)


def _train_step_adam_2d(model, opt_state, x_int, y_int, t_int, x_ic, y_ic,
                        c, lambda_ic, opt):
    try:
        from src.loss import loss_scalar_2d
    except ModuleNotFoundError:
        from loss import loss_scalar_2d
    params = pack_params(model)
    activation = model.activation
    k = model.k
    fourier_freqs = model.fourier_freqs
    loss, grads = jax.value_and_grad(
        lambda p: loss_scalar_2d(p, x_int, y_int, t_int, x_ic, y_ic, c, lambda_ic, activation, k)
    )(params)
    updates, new_opt_state = opt.update(grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    apply_params(model, new_params)
    return model, new_opt_state, loss


train_step_adam_2d = jax.jit(_train_step_adam_2d, static_argnames=["opt"])


def _train_step_lbfgs_2d(model, opt_state, x_int, y_int, t_int, x_ic, y_ic,
                         c, lambda_ic, opt):
    try:
        from src.loss import loss_scalar_2d
    except ModuleNotFoundError:
        from loss import loss_scalar_2d
    params = pack_params(model)
    activation = model.activation
    k = model.k

    def loss_fn_wrapped(p):
        return loss_scalar_2d(p, x_int, y_int, t_int, x_ic, y_ic, c, lambda_ic, activation, k)

    value_and_grad_fn = optax.value_and_grad_from_state(loss_fn_wrapped)
    value, grads = value_and_grad_fn(params, state=opt_state)
    updates, new_opt_state = opt.update(
        grads, opt_state, params,
        value=value, grad=grads, value_fn=loss_fn_wrapped,
    )
    new_params = optax.apply_updates(params, updates)
    apply_params(model, new_params)
    return model, new_opt_state, value


train_step_lbfgs_2d = jax.jit(_train_step_lbfgs_2d, static_argnames=["opt"])


# =============================================================================
# Schedule helper
# =============================================================================

def _make_schedule(lr, steps, kind):
    """
    Build a learning-rate schedule.

    kind : "cosine"      — cosine decay from lr to 0 over `steps`
           "exponential" — exponential decay, ×0.95 every 1000 steps
    """
    if kind == "cosine":
        return optax.cosine_decay_schedule(lr, decay_steps=steps)
    elif kind == "exponential":
        return optax.exponential_decay(lr, transition_steps=1000, decay_rate=0.95)
    else:
        raise ValueError(f"lr_schedule must be 'cosine' or 'exponential', got {kind!r}")


# =============================================================================
# Main training function
# =============================================================================

def _train_pinn_1d(
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
    lr_schedule="cosine",
    k=1,
    early_stopping=False,
    patience=100,
    min_delta=1e-6,
    interface_points=None,
    bias_frac=0.3,
    bias_std=0.012,
    fourier_freqs=None,
):
    """
    Train a PINN for the 1D wave equation without time windows.

    Parameters
    ----------
    widths            : tuple of ints, hidden layer widths, e.g. (64, 64)
    activation        : JAX activation callable (static — one version compiled per fn)
    optimizer         : "adam" | "adamw" | "lbfgs"
    steps             : total optimisation steps (L-BFGS steps only, if warm-started)
    adam_warmup_steps : Adam steps run before L-BFGS (only used when optimizer="lbfgs")
    init_params       : optional (ws, bs) pytree — if given, model is initialised from these
    N_int             : number of interior collocation points per step
    N_ic              : number of IC collocation points per step
    T                 : final time
    c                 : wave speed — scalar for constant-c, callable c(x,t) for variable-c
    lambda_ic         : IC loss weight
    lr                : peak learning rate
    grad_clip         : global gradient norm clip for Adam/AdamW (set <=0 to disable)
    seed              : random seed
    log_every         : print/record interval
    norm              : "L2" | "H1" | "H2" — PDE loss norm (constant-c only)
    lambda_sob        : weight for ‖∇r‖² term (H1, H2)
    lambda_sob2       : weight for ‖H_r‖²_F term (H2 only)
    lr_schedule       : "cosine" | "exponential"
    k                 : wavenumber in trial function ansatz (default 1)
    early_stopping    : stop early if relative loss change over `patience` steps < min_delta
    patience          : number of steps to look back for early stopping (default 100)
    min_delta         : minimum relative change threshold (default 1e-6)
    interface_points  : list/array of x locations to cluster collocation points around
    bias_frac         : fraction of N_int points placed near interface_points (default 0.3)
    bias_std          : std dev of Gaussian bias around each interface point (default 0.012)
    fourier_freqs     : list of frequencies for Fourier feature encoding, e.g. [1,2,4,8]

    Returns
    -------
    model           : trained MLP  (call model(xt) for predictions)
    losses          : list[float], total loss at every step
    loss_components : list[dict],  component losses every log_every steps
    """
    key = jax.random.PRNGKey(seed)
    key, k_model = jax.random.split(key)

    model = MLP(widths, key=k_model, activation=activation, k=k, fourier_freqs=fourier_freqs)
    if init_params is not None:
        apply_params(model, init_params)

    lambda_ic_arr   = jnp.array(lambda_ic)
    lambda_sob_arr  = jnp.array(lambda_sob)
    lambda_sob2_arr = jnp.array(lambda_sob2)

    is_var_c = callable(c)
    c_arr    = c if is_var_c else jnp.array(c)

    # Biased sampler: clusters `bias_frac` of points near interface boundaries.
    _ipts = jnp.array(interface_points) if interface_points is not None else None

    def _sample_x(key, n):
        if _ipts is None or bias_frac <= 0.0:
            return jax.random.uniform(key, (n,), minval=0.0, maxval=1.0)
        n_biased  = int(n * bias_frac)
        n_uniform = n - n_biased
        key, k_u, k_b, k_w = jax.random.split(key, 4)
        x_uniform = jax.random.uniform(k_u, (n_uniform,), minval=0.0, maxval=1.0)
        which    = jax.random.randint(k_w, (n_biased,), 0, _ipts.shape[0])
        x_biased = _ipts[which] + jax.random.normal(k_b, (n_biased,)) * bias_std
        x_biased = jnp.clip(x_biased, 0.0, 1.0)
        return jnp.concatenate([x_uniform, x_biased])

    opt_lower = optimizer.lower()
    is_lbfgs  = opt_lower == "lbfgs"

    losses: list[float] = []
    loss_components: list[dict] = []

    # -------------------------------------------------------------------------
    # For variable-c, build closure-based JIT steps so c_fn never touches JIT
    # as a traced argument (Python callables are not valid JAX pytree leaves).
    # -------------------------------------------------------------------------
    if is_var_c:
        try:
            from src.loss import loss_scalar_var_1d
        except ModuleNotFoundError:
            from loss import loss_scalar_var_1d

        def _make_var_adam_step(opt_):
            @jax.jit
            def _step(params, opt_state, x_int, t_int, x_ic):
                loss, grads = jax.value_and_grad(
                    lambda p: loss_scalar_var_1d(
                        p, x_int, t_int, x_ic, c_arr, lambda_ic_arr, activation, k, fourier_freqs
                    )
                )(params)
                updates, new_state = opt_.update(grads, opt_state, params)
                return optax.apply_updates(params, updates), new_state, loss
            return _step

        def _make_var_lbfgs_step(opt_, x_int_f, t_int_f, x_ic_f):
            @jax.jit
            def _step(params, opt_state):
                def loss_fn_w(p):
                    return loss_scalar_var_1d(
                        p, x_int_f, t_int_f, x_ic_f, c_arr, lambda_ic_arr, activation, k, fourier_freqs
                    )
                value_and_grad_fn = optax.value_and_grad_from_state(loss_fn_w)
                value, grads = value_and_grad_fn(params, state=opt_state)
                updates, new_state = opt_.update(
                    grads, opt_state, params,
                    value=value, grad=grads, value_fn=loss_fn_w,
                )
                return optax.apply_updates(params, updates), new_state, value
            return _step

        def _run_var_adam(n_steps, opt_, tag):
            nonlocal key
            step_fn = _make_var_adam_step(opt_)
            state = opt_.init(pack_params(model))
            for i in range(n_steps):
                key, k_x, k_t, k_ic = jax.random.split(key, 4)
                x_int = _sample_x(k_x, N_int)
                t_int = jax.random.uniform(k_t,  (N_int,), minval=0.0, maxval=T)
                x_ic  = jax.random.uniform(k_ic, (N_ic,),  minval=0.0, maxval=1.0)
                new_params, state, L = step_fn(pack_params(model), state, x_int, t_int, x_ic)
                apply_params(model, new_params)
                losses.append(float(L))
                if (i + 1) % log_every == 0:
                    print(f"  [{tag}] step {i+1:5d} | loss={float(L):.3e}")
            return state

        def _run_var_lbfgs(n_steps, opt_):
            nonlocal key
            key, k_x, k_t, k_ic = jax.random.split(key, 4)
            x_f  = _sample_x(k_x, N_int)
            t_f  = jax.random.uniform(k_t,  (N_int,), minval=0.0, maxval=T)
            ic_f = jax.random.uniform(k_ic, (N_ic,),  minval=0.0, maxval=1.0)
            step_fn = _make_var_lbfgs_step(opt_, x_f, t_f, ic_f)
            state = opt_.init(pack_params(model))
            for i in range(n_steps):
                new_params, state, L = step_fn(pack_params(model), state)
                apply_params(model, new_params)
                losses.append(float(L))
                if (i + 1) % log_every == 0:
                    print(f"  [L-BFGS] step {i+1:5d} | loss={float(L):.3e}")

    # -------------------------------------------------------------------------
    # Adam warm-up  (always runs when optimizer="lbfgs" and adam_warmup_steps>0)
    # -------------------------------------------------------------------------
    if is_lbfgs and adam_warmup_steps > 0:
        warmup_base = optax.adam(_make_schedule(lr, adam_warmup_steps, lr_schedule))
        if grad_clip is not None and grad_clip > 0:
            warmup_opt = optax.chain(optax.clip_by_global_norm(grad_clip), warmup_base)
        else:
            warmup_opt = warmup_base

        print(f"  Warm-starting with {adam_warmup_steps} Adam steps ...")

        if is_var_c:
            _run_var_adam(adam_warmup_steps, warmup_opt, "warmup")
        else:
            warmup_state = warmup_opt.init(pack_params(model))
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
                if early_stopping and len(losses) > patience:
                    rel_change = abs(losses[-1] - losses[-patience - 1]) / (abs(losses[-patience - 1]) + 1e-30)
                    if rel_change < min_delta:
                        print(f"  [warmup] Early stopping at step {i+1} (rel_change={rel_change:.2e})")
                        break

    # -------------------------------------------------------------------------
    # Main optimizer
    # -------------------------------------------------------------------------
    if opt_lower == "adam":
        base_opt = optax.adam(_make_schedule(lr, steps, lr_schedule))
    elif opt_lower == "adamw":
        base_opt = optax.adamw(_make_schedule(lr, steps, lr_schedule))
    elif opt_lower == "lbfgs":
        base_opt = optax.lbfgs()
    else:
        raise ValueError(f"Unknown optimizer {optimizer!r}. Use 'adam', 'adamw', or 'lbfgs'.")

    if not is_lbfgs and grad_clip is not None and grad_clip > 0:
        opt = optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
    else:
        opt = base_opt

    if is_var_c:
        if is_lbfgs:
            _run_var_lbfgs(steps, opt)
        else:
            _run_var_adam(steps, opt, optimizer)
        return model, losses, loss_components

    # -------------------------------------------------------------------------
    # Constant-c path (original code)
    # -------------------------------------------------------------------------
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

        if early_stopping and len(losses) > patience:
            rel_change = abs(losses[-1] - losses[-patience - 1]) / (abs(losses[-patience - 1]) + 1e-30)
            if rel_change < min_delta:
                print(f"  [{optimizer}] Early stopping at step {i+1} (rel_change={rel_change:.2e})")
                break

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
                norm=norm, lambda_sob=lambda_sob_arr, lambda_sob2=lambda_sob2_arr, k=k,
            )
            comps_f = {k_: float(v) for k_, v in comps.items()}
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


def _train_pinn_2d(
    widths,
    *,
    activation=jax.nn.tanh,
    adam_steps=2000,
    lbfgs_steps=3000,
    init_params=None,
    N_int=2000,
    N_ic=200,
    T=1.0,
    c=1.0,
    lambda_ic=100.0,
    lr=1e-3,
    grad_clip=1.0,
    seed=0,
    log_every=500,
    lr_schedule="cosine",
    k=1,
    fourier_freqs=None,
    interface_points=None,
    bias_frac=0.3,
    bias_std=0.012,
):
    """
    Train a PINN for the 2D wave equation u_tt = c²(u_xx + u_yy)
    or variable-c conservative form u_tt = ∇·(c²∇u).

    Runs `adam_steps` of Adam, then warm-starts L-BFGS from Adam's parameters
    and runs `lbfgs_steps` of L-BFGS.

    Parameters
    ----------
    widths           : tuple of ints, hidden layer widths
    c                : wave speed — scalar for constant-c, callable c(x,y,t) for variable-c
    fourier_freqs    : list of frequencies for Fourier feature encoding, e.g. [1,2,4,8]
    interface_points : (M, 2) array of (x, y) locations to bias collocation points toward
    bias_frac        : fraction of N_int points placed near interface_points (default 0.3)
    bias_std         : std dev of Gaussian bias around each interface point (default 0.012)

    Returns
    -------
    model_adam   : MLP2d after Adam phase
    model_lbfgs  : MLP2d after L-BFGS phase
    losses       : list[float], total loss at every step (Adam + L-BFGS)
    """
    key = jax.random.PRNGKey(seed)
    key, k_model = jax.random.split(key)

    model = MLP2d(widths, key=k_model, activation=activation, k=k, fourier_freqs=fourier_freqs)
    if init_params is not None:
        apply_params(model, init_params)

    is_var_c = callable(c)
    c_arr         = c if is_var_c else jnp.array(c)
    lambda_ic_arr = jnp.array(lambda_ic)
    losses: list[float] = []

    # -------------------------------------------------------------------------
    # Biased collocation sampler — clusters points near interface_points in (x,y)
    # -------------------------------------------------------------------------
    _ipts = jnp.array(interface_points) if interface_points is not None else None   # (M, 2)

    def _sample_xy(key, n):
        """Sample (x, y) pairs with optional bias toward interface_points."""
        if _ipts is None or bias_frac <= 0.0:
            kx, ky = jax.random.split(key)
            return jax.random.uniform(kx, (n,)), jax.random.uniform(ky, (n,))
        n_biased  = int(n * bias_frac)
        n_uniform = n - n_biased
        key, k_ux, k_uy, k_bx, k_by, k_w = jax.random.split(key, 6)
        x_uniform = jax.random.uniform(k_ux, (n_uniform,))
        y_uniform = jax.random.uniform(k_uy, (n_uniform,))
        which     = jax.random.randint(k_w, (n_biased,), 0, _ipts.shape[0])
        centers   = _ipts[which]                              # (n_biased, 2)
        noise     = jax.random.normal(jnp.stack([k_bx, k_by]), (2, n_biased)) * bias_std
        x_biased  = jnp.clip(centers[:, 0] + noise[0], 0.0, 1.0)
        y_biased  = jnp.clip(centers[:, 1] + noise[1], 0.0, 1.0)
        return (
            jnp.concatenate([x_uniform, x_biased]),
            jnp.concatenate([y_uniform, y_biased]),
        )

    # -------------------------------------------------------------------------
    # Variable-c: build JIT closures so c_fn never becomes a traced argument
    # -------------------------------------------------------------------------
    if is_var_c:
        try:
            from src.loss import loss_scalar_var_2d
        except ModuleNotFoundError:
            from loss import loss_scalar_var_2d

        def _make_var_adam_step_2d(opt_):
            @jax.jit
            def _step(params, opt_state, x_int, y_int, t_int, x_ic, y_ic):
                loss, grads = jax.value_and_grad(
                    lambda p: loss_scalar_var_2d(
                        p, x_int, y_int, t_int, x_ic, y_ic,
                        c_arr, lambda_ic_arr, activation, k, fourier_freqs
                    )
                )(params)
                updates, new_state = opt_.update(grads, opt_state, params)
                return optax.apply_updates(params, updates), new_state, loss
            return _step

        def _make_var_lbfgs_step_2d(opt_, x_f, y_f, t_f, x_ic_f, y_ic_f):
            @jax.jit
            def _step(params, opt_state):
                def loss_fn_w(p):
                    return loss_scalar_var_2d(
                        p, x_f, y_f, t_f, x_ic_f, y_ic_f,
                        c_arr, lambda_ic_arr, activation, k, fourier_freqs
                    )
                value_and_grad_fn = optax.value_and_grad_from_state(loss_fn_w)
                value, grads = value_and_grad_fn(params, state=opt_state)
                updates, new_state = opt_.update(
                    grads, opt_state, params,
                    value=value, grad=grads, value_fn=loss_fn_w,
                )
                return optax.apply_updates(params, updates), new_state, value
            return _step

    # -------------------------------------------------------------------------
    # Adam phase
    # -------------------------------------------------------------------------
    base_opt = optax.adam(_make_schedule(lr, adam_steps, lr_schedule))
    if grad_clip is not None and grad_clip > 0:
        adam_opt = optax.chain(optax.clip_by_global_norm(grad_clip), base_opt)
    else:
        adam_opt = base_opt

    adam_state = adam_opt.init(pack_params(model))

    if is_var_c:
        adam_step_fn = _make_var_adam_step_2d(adam_opt)

    for i in range(adam_steps):
        key, k_xy, k_t, k_ix, k_iy = jax.random.split(key, 5)
        x_int, y_int = _sample_xy(k_xy, N_int)
        t_int = jax.random.uniform(k_t,  (N_int,), maxval=T)
        x_ic  = jax.random.uniform(k_ix, (N_ic,))
        y_ic  = jax.random.uniform(k_iy, (N_ic,))

        if is_var_c:
            new_params, adam_state, L = adam_step_fn(
                pack_params(model), adam_state, x_int, y_int, t_int, x_ic, y_ic
            )
            apply_params(model, new_params)
        else:
            model, adam_state, L = train_step_adam_2d(
                model, adam_state, x_int, y_int, t_int, x_ic, y_ic,
                c_arr, lambda_ic_arr, adam_opt,
            )
        losses.append(float(L))
        if (i + 1) % log_every == 0:
            print(f"  [Adam]   step {i+1:5d} | loss={float(L):.3e}")

    # snapshot after Adam
    model_adam = MLP2d(widths, key=jax.random.PRNGKey(0), activation=activation, k=k,
                       fourier_freqs=fourier_freqs)
    apply_params(model_adam, pack_params(model))

    # -------------------------------------------------------------------------
    # L-BFGS phase  (fixed collocation points)
    # -------------------------------------------------------------------------
    lbfgs_opt = optax.lbfgs()
    key, k_xy, k_t, k_ix, k_iy = jax.random.split(key, 5)
    x_int_fixed, y_int_fixed = _sample_xy(k_xy, N_int)
    t_int_fixed = jax.random.uniform(k_t,  (N_int,), maxval=T)
    x_ic_fixed  = jax.random.uniform(k_ix, (N_ic,))
    y_ic_fixed  = jax.random.uniform(k_iy, (N_ic,))

    lbfgs_state = lbfgs_opt.init(pack_params(model))

    if is_var_c:
        lbfgs_step_fn = _make_var_lbfgs_step_2d(
            lbfgs_opt, x_int_fixed, y_int_fixed, t_int_fixed, x_ic_fixed, y_ic_fixed
        )

    for i in range(lbfgs_steps):
        if is_var_c:
            new_params, lbfgs_state, L = lbfgs_step_fn(pack_params(model), lbfgs_state)
            apply_params(model, new_params)
        else:
            model, lbfgs_state, L = train_step_lbfgs_2d(
                model, lbfgs_state,
                x_int_fixed, y_int_fixed, t_int_fixed, x_ic_fixed, y_ic_fixed,
                c_arr, lambda_ic_arr, lbfgs_opt,
            )
        losses.append(float(L))
        if (i + 1) % log_every == 0:
            print(f"  [L-BFGS] step {i+1:5d} | loss={float(L):.3e}")

    return model_adam, model, losses


# =============================================================================
# Unified public entry point
# =============================================================================

def train_pinn(widths, *, dim=1, **kwargs):
    """
    Train a wave-equation PINN.

    Parameters
    ----------
    widths : tuple[int]
        Hidden layer widths.
    dim : int
        Spatial dimension — 1 or 2.
    k : int or float
        Wavenumber in the trial function ansatz.  The hard IC is
        u(x,0) = sin(k·πx) in 1D, sin(k·πx)sin(k·πy) in 2D.
        Defaults to 1.
    fourier_freqs : list[float] or None
        Frequencies for Fourier feature encoding, e.g. [1,2,4,8].
        Applies to both constant-c and variable-c paths.
    interface_points : array-like or None
        Locations to bias collocation points toward.
        1D: 1D array of x positions.
        2D: (M,2) array of (x,y) positions.
    bias_frac : float
        Fraction of interior points clustered near interface_points.
    bias_std : float
        Std dev of Gaussian noise around each interface point.
    **kwargs
        Forwarded to the underlying training function:
          dim=1 → _train_pinn_1d   (supports optimizer, steps, norm, …)
          dim=2 → _train_pinn_2d   (adam_steps, lbfgs_steps)

    Returns
    -------
    dim=1 : (model, losses, loss_components)
    dim=2 : (model_adam, model_lbfgs, losses)
    """
    if dim == 1:
        return _train_pinn_1d(widths, **kwargs)
    elif dim == 2:
        return _train_pinn_2d(widths, **kwargs)
    else:
        raise ValueError(f"dim must be 1 or 2, got {dim!r}")



