"""
loss.py

Loss functions for the params-based NNX wave PINN (pinn.py).

Works with the (params, activation) pytree API, which is required for nested
JAX autodiff (grad inside the residual, then grad of the total loss for the
outer optimizer).

Constant-c norms (1D and 2D):
  "L2"  — MSE on PDE residual  r = u_tt - c²u_xx
  "H1"  — L2 + lambda_sob  * MSE on ‖∇_xt r‖²
  "H2"  — H1 + lambda_sob2 * MSE on ‖H_xt r‖²_F

Variable-c (1D and 2D):
  Conservative-form residual: u_tt = ∇·(c²∇u)
  Fourier feature encoding optional via `fourier_freqs`.
  IC soft penalty omitted (displacement IC is hard-enforced via the ansatz).

The `k` parameter controls the wavenumber in the trial function ansatz:
  1D:  u(x,t)   = sin(k·πx) + t·sin(k·πx)·N(x,t)
  2D:  u(x,y,t) = sin(k·πx)sin(k·πy) + t·sin(k·πx)sin(k·πy)·N(x,y,t)
Defaults to k=1 (the fundamental mode).
"""

import jax
import jax.numpy as jnp


# =============================================================================
# Forward pass and derivative helpers  (self-contained — no circular import)
# These mirror the functions in pinn_wave_nnx.py but live here so loss_nnx.py
# has no dependency on pinn_wave_nnx.py.
# =============================================================================

def _encode_fourier(xt2d, fourier_freqs):
    if fourier_freqs is None:
        return xt2d
    freqs = jnp.array(fourier_freqs)        # (F,)
    x = xt2d[:, 0:1]                        # (N, 1)
    t = xt2d[:, 1:2]                        # (N, 1)
    args_x = 2 * jnp.pi * freqs * x        # (N, F)
    args_t = 2 * jnp.pi * freqs * t        # (N, F)
    return jnp.concatenate(
        [xt2d, jnp.cos(args_x), jnp.sin(args_x), jnp.cos(args_t), jnp.sin(args_t)],
        axis=1,
    )


def _forward_params(params, xt2d, activation, fourier_freqs=None):
    ws, bs = params
    z = _encode_fourier(xt2d, fourier_freqs)
    for W, b in zip(ws[:-1], bs[:-1]):
        z = activation(z @ W + b)
    return z @ ws[-1] + bs[-1]


def _u_hat_params(params, xt2d, activation, k, fourier_freqs=None):
    """Hard-BC/IC ansatz: u(x,t) = sin(k·πx) + t·sin(k·πx)·N(x,t)."""
    x = xt2d[:, 0]
    t = xt2d[:, 1]
    N = _forward_params(params, xt2d, activation, fourier_freqs)[:, 0]
    sx = jnp.sin(k * jnp.pi * x)
    return sx + t * sx * N


def _u_scalar(params, x, t, activation, k, fourier_freqs=None):
    xt = jnp.stack([x, t])[None]
    return _u_hat_params(params, xt, activation, k, fourier_freqs)[0]


def _u_t(params, x, t, activation, k, fourier_freqs=None):
    return jax.grad(lambda t_: _u_scalar(params, x, t_, activation, k, fourier_freqs))(t)


def _u_tt(params, x, t, activation, k, fourier_freqs=None):
    return jax.grad(jax.grad(lambda t_: _u_scalar(params, x, t_, activation, k, fourier_freqs)))(t)


def _u_xx(params, x, t, activation, k, fourier_freqs=None):
    return jax.grad(jax.grad(lambda x_: _u_scalar(params, x_, t, activation, k, fourier_freqs)))(x)


def _u_x(params, x, t, activation, k, fourier_freqs=None):
    return jax.grad(lambda x_: _u_scalar(params, x_, t, activation, k, fourier_freqs))(x)


# =============================================================================
# Scalar residual
# =============================================================================

def r_scalar_params(params, x, t, c, activation, k):
    """Scalar PDE residual r = u_tt - c² u_xx at a single (x, t) point."""
    utt = _u_tt(params, x, t, activation, k)
    uxx = _u_xx(params, x, t, activation, k)
    return utt - c**2 * uxx


# =============================================================================
# PDE residual terms — one function per norm
# =============================================================================

def pde_terms_l2(params, x_int, t_int, c, activation, k):
    """
    L2: mean(r²).

    Returns (loss_pde, extra_dict).  extra_dict is empty for L2.
    """
    def r_single(xi, ti):
        return r_scalar_params(params, xi, ti, c, activation, k)

    r = jax.vmap(r_single)(x_int, t_int)
    return jnp.mean(r**2), {}


def pde_terms_h1(params, x_int, t_int, c, activation, k):
    """
    H1 (Sobolev): mean(r²) plus mean(‖∇_xt r‖²).

    ∇_xt r = (∂r/∂x, ∂r/∂t) — 3rd-order derivatives of u.

    Returns (loss_pde, {"sobolev": loss_sob}).
    """
    def r_and_grad_single(xi, ti):
        r = r_scalar_params(params, xi, ti, c, activation, k)
        dr_dx, dr_dt = jax.grad(r_scalar_params, argnums=(1, 2))(
            params, xi, ti, c, activation, k
        )
        return r**2, dr_dx**2 + dr_dt**2

    r_sq, grad_sq = jax.vmap(r_and_grad_single)(x_int, t_int)
    return r_sq.mean(), {"sobolev": grad_sq.mean()}


def pde_terms_h2(params, x_int, t_int, c, activation, k):
    """
    H2: mean(r²) plus mean(‖∇_xt r‖²) plus mean(‖H_xt r‖²_F).

    H_xt r is the 2×2 Hessian of r w.r.t. (x, t) — 4th-order derivatives of u.
    Frobenius norm: ‖H‖²_F = r_xx² + 2*r_xt² + r_tt²

    Returns (loss_pde, {"sobolev": loss_sob, "sobolev2": loss_sob2}).
    """
    def r_of_xt(xt, params, c, activation, k):
        return r_scalar_params(params, xt[0], xt[1], c, activation, k)

    def terms_single(xi, ti):
        xt = jnp.stack([xi, ti])
        r = r_of_xt(xt, params, c, activation, k)

        grad_r = jax.grad(r_of_xt)(xt, params, c, activation, k)          # (2,)
        hess_r = jax.hessian(r_of_xt)(xt, params, c, activation, k)       # (2, 2)

        return r**2, jnp.sum(grad_r**2), jnp.sum(hess_r**2)

    r_sq, grad_sq, hess_sq = jax.vmap(terms_single)(x_int, t_int)
    return r_sq.mean(), {
        "sobolev":  grad_sq.mean(),
        "sobolev2": hess_sq.mean(),
    }


# =============================================================================
# Unified loss function
# =============================================================================

def loss_fn(
    params,
    x_int,
    t_int,
    x_ic,
    c,
    lambda_ic,
    activation,
    *,
    norm="L2",
    lambda_sob=1.0,
    lambda_sob2=1.0,
    k=1.0,
):
    """
    Total PINN loss with selectable norm.

    Parameters
    ----------
    params      : (ws, bs) parameter pytree from pack_params
    x_int       : (N_int,) interior x collocation points
    t_int       : (N_int,) interior t collocation points
    x_ic        : (N_ic,)  IC x collocation points (evaluated at t=0)
    c           : wave speed (scalar JAX array)
    lambda_ic   : IC loss weight (scalar JAX array)
    activation  : activation function (static — one compiled version per fn)
    norm        : "L2" | "H1" | "H2"
    lambda_sob  : weight for ‖∇r‖² term (H1 and H2)
    lambda_sob2 : weight for ‖H_r‖²_F term (H2 only)
    k           : wavenumber in trial function ansatz (default 1)

    Returns
    -------
    (total_loss, components_dict)
    components_dict always has keys "pde" and "ic_ut".
    H1 adds "sobolev"; H2 adds "sobolev" and "sobolev2".
    """
    t0 = jnp.zeros(())

    # --- PDE term ---
    if norm == "L2":
        loss_pde, extra = pde_terms_l2(params, x_int, t_int, c, activation, k)
    elif norm == "H1":
        loss_pde, extra = pde_terms_h1(params, x_int, t_int, c, activation, k)
    elif norm == "H2":
        loss_pde, extra = pde_terms_h2(params, x_int, t_int, c, activation, k)
    else:
        raise ValueError(f"norm must be 'L2', 'H1', or 'H2', got {norm!r}")

    # --- IC velocity term: u_t(x, 0) = 0 ---
    u_t_ic = jax.vmap(lambda xi: _u_t(params, xi, t0, activation, k))(x_ic)
    loss_ic_ut = jnp.mean(u_t_ic**2)

    # --- Total ---
    loss_total = loss_pde + lambda_ic * loss_ic_ut
    if "sobolev" in extra:
        loss_total = loss_total + lambda_sob * extra["sobolev"]
    if "sobolev2" in extra:
        loss_total = loss_total + lambda_sob2 * extra["sobolev2"]

    components = {"pde": loss_pde, "ic_ut": loss_ic_ut, **extra}
    return loss_total, components


def loss_scalar(
    params,
    x_int,
    t_int,
    x_ic,
    c,
    lambda_ic,
    activation,
    *,
    norm="L2",
    lambda_sob=1.0,
    lambda_sob2=1.0,
    k=1.0,
):
    """Scalar wrapper — pass to jax.grad or optax's value_and_grad_from_state."""
    loss, _ = loss_fn(
        params, x_int, t_int, x_ic, c, lambda_ic, activation,
        norm=norm, lambda_sob=lambda_sob, lambda_sob2=lambda_sob2, k=k,
    )
    return loss


# =============================================================================
# 1D variable-c loss  (u_tt = ∂_x(c(x,t)²·∂_x u), conservative form)
# c_fn must be callable: c_fn(x_scalar, t_scalar) -> scalar
# c_fn must be captured in the JIT closure, not passed as a traced argument.
# =============================================================================

def r_scalar_params_var_1d(params, x, t, c_fn, activation, k, fourier_freqs=None):
    """PDE residual r = u_tt - ∂_x(c²∂_x u) at a single (x,t) point (conservative form)."""
    utt = _u_tt(params, x, t, activation, k, fourier_freqs)
    uxx = _u_xx(params, x, t, activation, k, fourier_freqs)
    ux  = _u_x( params, x, t, activation, k, fourier_freqs)
    c2     = c_fn(x, t) ** 2
    dc2_dx = jax.grad(lambda x_: c_fn(x_, t) ** 2)(x)
    return utt - c2 * uxx - dc2_dx * ux


def loss_scalar_var_1d(params, x_int, t_int, x_ic, c_fn, lambda_ic, activation, k=1.0, fourier_freqs=None):
    """
    Scalar loss for 1D variable-c PINN (conservative form).

    Parameters
    ----------
    params       : (ws, bs) parameter pytree
    x_int        : (N_int,) interior x collocation points
    t_int        : (N_int,) interior t collocation points
    x_ic         : (N_ic,)  IC x collocation points (t=0)
    c_fn         : callable c_fn(x, t) -> scalar wave speed
    lambda_ic    : IC velocity loss weight
    activation   : activation function
    k            : wavenumber in ansatz
    fourier_freqs: list of frequencies for Fourier feature encoding, or None
    """
    def r_single(xi, ti):
        return r_scalar_params_var_1d(params, xi, ti, c_fn, activation, k, fourier_freqs)

    loss_pde = jnp.mean(jax.vmap(r_single)(x_int, t_int) ** 2)

    t0 = jnp.zeros(())
    u_t_ic = jax.vmap(lambda xi: _u_t(params, xi, t0, activation, k, fourier_freqs))(x_ic)
    loss_ic_ut = jnp.mean(u_t_ic ** 2)

    return loss_pde + lambda_ic * loss_ic_ut


# =============================================================================
# 2D wave equation loss  (u_tt = c² (u_xx + u_yy))
# Hard-BC/IC ansatz: u(x,y,t) = sin(k·πx)sin(k·πy) + t·sin(k·πx)sin(k·πy)·N(x,y,t)
# =============================================================================

def _encode_fourier_2d(xyt, fourier_freqs):
    """Fourier feature encoding for 3-input (x, y, t) coordinates."""
    if fourier_freqs is None:
        return xyt
    freqs = jnp.array(fourier_freqs)       # (F,)
    x = xyt[:, 0:1]                        # (N, 1)
    y = xyt[:, 1:2]                        # (N, 1)
    t = xyt[:, 2:3]                        # (N, 1)
    args_x = 2 * jnp.pi * freqs * x       # (N, F)
    args_y = 2 * jnp.pi * freqs * y       # (N, F)
    args_t = 2 * jnp.pi * freqs * t       # (N, F)
    return jnp.concatenate(
        [xyt,
         jnp.cos(args_x), jnp.sin(args_x),
         jnp.cos(args_y), jnp.sin(args_y),
         jnp.cos(args_t), jnp.sin(args_t)],
        axis=1,
    )


def _forward_params_2d(params, xyt, activation, fourier_freqs=None):
    ws, bs = params
    z = _encode_fourier_2d(xyt, fourier_freqs)
    for W, b in zip(ws[:-1], bs[:-1]):
        z = activation(z @ W + b)
    return z @ ws[-1] + bs[-1]


def _u_hat_params_2d(params, xyt, activation, k, fourier_freqs=None):
    """
    2D hard-BC/IC ansatz:
        u(x,y,t) = sin(k·πx)sin(k·πy) + t·sin(k·πx)sin(k·πy)·N(x,y,t)
    Guarantees u=0 on all four walls and u(x,y,0)=sin(k·πx)sin(k·πy).
    xyt: (N, 3) → (N,)
    """
    x = xyt[:, 0]
    y = xyt[:, 1]
    t = xyt[:, 2]
    N = _forward_params_2d(params, xyt, activation, fourier_freqs)[:, 0]
    spatial = jnp.sin(k * jnp.pi * x) * jnp.sin(k * jnp.pi * y)
    return spatial + t * spatial * N


def _u_scalar_2d(params, x, y, t, activation, k, fourier_freqs=None):
    """Scalar u at a single (x, y, t) point — entry point for jax.grad."""
    xyt = jnp.stack([x, y, t])[None]
    return _u_hat_params_2d(params, xyt, activation, k, fourier_freqs)[0]


def _u_t_2d(params, x, y, t, activation, k, fourier_freqs=None):
    return jax.grad(lambda t_: _u_scalar_2d(params, x, y, t_, activation, k, fourier_freqs))(t)


def r_scalar_params_2d(params, x, y, t, c, activation, k):
    """PDE residual r = u_tt - c²(u_xx + u_yy) at a single (x,y,t) point."""
    utt = jax.grad(jax.grad(lambda t_: _u_scalar_2d(params, x, y, t_, activation, k)))(t)
    uxx = jax.grad(jax.grad(lambda x_: _u_scalar_2d(params, x_, y, t, activation, k)))(x)
    uyy = jax.grad(jax.grad(lambda y_: _u_scalar_2d(params, x, y_, t, activation, k)))(y)
    return utt - c**2 * (uxx + uyy)


def loss_fn_2d(params, x_int, y_int, t_int, x_ic, y_ic, c, lambda_ic, activation, k=1.0):
    """
    2D PINN loss: MSE PDE residual + lambda_ic * MSE velocity IC.

    Returns (total_loss, {"pde": ..., "ic_ut": ...}).
    """
    def r_single(xi, yi, ti):
        return r_scalar_params_2d(params, xi, yi, ti, c, activation, k)

    r = jax.vmap(r_single)(x_int, y_int, t_int)
    loss_pde = jnp.mean(r**2)

    t0 = jnp.zeros(())
    u_t_ic = jax.vmap(lambda xi, yi: _u_t_2d(params, xi, yi, t0, activation, k))(x_ic, y_ic)
    loss_ic_ut = jnp.mean(u_t_ic**2)

    total = loss_pde + lambda_ic * loss_ic_ut
    return total, {"pde": loss_pde, "ic_ut": loss_ic_ut}


def loss_scalar_2d(params, x_int, y_int, t_int, x_ic, y_ic, c, lambda_ic, activation, k=1.0):
    """Scalar wrapper for 2D loss — pass to jax.grad or optax."""
    loss, _ = loss_fn_2d(params, x_int, y_int, t_int, x_ic, y_ic, c, lambda_ic, activation, k)
    return loss


# =============================================================================
# 2D variable-c loss  (u_tt = ∇·(c(x,y,t)²·∇u), conservative form)
# c_fn must be callable: c_fn(x_scalar, y_scalar, t_scalar) -> scalar
# c_fn must be captured in the JIT closure, not passed as a traced argument.
# =============================================================================

def r_scalar_params_var_2d(params, x, y, t, c_fn, activation, k, fourier_freqs=None):
    """
    PDE residual r = u_tt - ∇·(c²∇u) at a single (x,y,t) point (conservative form).

    ∇·(c²∇u) = c²(u_xx + u_yy) + (∂_x c²)u_x + (∂_y c²)u_y
    """
    utt = jax.grad(jax.grad(lambda t_: _u_scalar_2d(params, x, y, t_, activation, k, fourier_freqs)))(t)
    uxx = jax.grad(jax.grad(lambda x_: _u_scalar_2d(params, x_, y, t, activation, k, fourier_freqs)))(x)
    uyy = jax.grad(jax.grad(lambda y_: _u_scalar_2d(params, x, y_, t, activation, k, fourier_freqs)))(y)
    ux  = jax.grad(lambda x_: _u_scalar_2d(params, x_, y, t, activation, k, fourier_freqs))(x)
    uy  = jax.grad(lambda y_: _u_scalar_2d(params, x, y_, t, activation, k, fourier_freqs))(y)

    c2_fn  = lambda x_, y_: c_fn(x_, y_, t) ** 2
    dc2_dx = jax.grad(c2_fn, argnums=0)(x, y)
    dc2_dy = jax.grad(c2_fn, argnums=1)(x, y)
    c2     = c_fn(x, y, t) ** 2

    return utt - c2 * (uxx + uyy) - dc2_dx * ux - dc2_dy * uy


def loss_scalar_var_2d(params, x_int, y_int, t_int, x_ic, y_ic, c_fn, lambda_ic, activation, k=1.0, fourier_freqs=None):
    """
    Scalar loss for 2D variable-c PINN (conservative form).

    Parameters
    ----------
    params       : (ws, bs) parameter pytree
    x_int        : (N_int,) interior x collocation points
    y_int        : (N_int,) interior y collocation points
    t_int        : (N_int,) interior t collocation points
    x_ic         : (N_ic,)  IC x collocation points (t=0)
    y_ic         : (N_ic,)  IC y collocation points (t=0)
    c_fn         : callable c_fn(x, y, t) -> scalar wave speed
    lambda_ic    : IC velocity loss weight
    activation   : activation function
    k            : wavenumber in ansatz
    fourier_freqs: list of frequencies for Fourier feature encoding, or None
    """
    def r_single(xi, yi, ti):
        return r_scalar_params_var_2d(params, xi, yi, ti, c_fn, activation, k, fourier_freqs)

    loss_pde = jnp.mean(jax.vmap(r_single)(x_int, y_int, t_int) ** 2)

    t0 = jnp.zeros(())
    u_t_ic = jax.vmap(
        lambda xi, yi: _u_t_2d(params, xi, yi, t0, activation, k, fourier_freqs)
    )(x_ic, y_ic)
    loss_ic_ut = jnp.mean(u_t_ic ** 2)

    return loss_pde + lambda_ic * loss_ic_ut
