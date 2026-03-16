"""
loss_nnx.py

Structured loss functions for the params-based NNX wave PINN (pinn_wave_nnx.py).

Mirrors loss.py in structure but works with the (params, activation) pytree API,
which is required for nested JAX autodiff (grad inside the residual, then grad of
the total loss for the outer optimizer).

Supported norms:
  "L2"  — MSE on PDE residual  r = u_tt - c²u_xx
  "H1"  — L2 + lambda_sob  * MSE on ‖∇_xt r‖²   (gradient of residual w.r.t. inputs)
  "H2"  — H1 + lambda_sob2 * MSE on ‖H_xt r‖²_F  (Frobenius norm of Hessian of r)

IC term (u_t(x,0)=0 soft penalty) is identical across all norms, because the
displacement IC is enforced hard via the ansatz in pinn_wave_nnx.py.
"""

import jax
import jax.numpy as jnp


# =============================================================================
# Forward pass and derivative helpers  (self-contained — no circular import)
# These mirror the functions in pinn_wave_nnx.py but live here so loss_nnx.py
# has no dependency on pinn_wave_nnx.py.
# =============================================================================

def _forward_params(params, xt2d, activation):
    ws, bs = params
    z = xt2d
    for W, b in zip(ws[:-1], bs[:-1]):
        z = activation(z @ W + b)
    return z @ ws[-1] + bs[-1]


def _u_hat_params(params, xt2d, activation):
    """Hard-BC/IC ansatz: u(x,t) = sin(πx) + t·sin(πx)·N(x,t)."""
    x = xt2d[:, 0]
    t = xt2d[:, 1]
    N = _forward_params(params, xt2d, activation)[:, 0]
    return jnp.sin(jnp.pi * x) + t * jnp.sin(jnp.pi * x) * N


def _u_scalar(params, x, t, activation):
    xt = jnp.stack([x, t])[None]
    return _u_hat_params(params, xt, activation)[0]


def _u_t(params, x, t, activation):
    return jax.grad(lambda t_: _u_scalar(params, x, t_, activation))(t)


def _u_tt(params, x, t, activation):
    return jax.grad(jax.grad(lambda t_: _u_scalar(params, x, t_, activation)))(t)


def _u_xx(params, x, t, activation):
    return jax.grad(jax.grad(lambda x_: _u_scalar(params, x_, t, activation)))(x)


# =============================================================================
# Scalar residual
# =============================================================================

def r_scalar_params(params, x, t, c, activation):
    """Scalar PDE residual r = u_tt - c² u_xx at a single (x, t) point."""
    utt = _u_tt(params, x, t, activation)
    uxx = _u_xx(params, x, t, activation)
    return utt - c**2 * uxx


# =============================================================================
# PDE residual terms — one function per norm
# =============================================================================

def pde_terms_l2(params, x_int, t_int, c, activation):
    """
    L2: mean(r²).

    Returns (loss_pde, extra_dict).  extra_dict is empty for L2.
    """
    def r_single(xi, ti):
        return r_scalar_params(params, xi, ti, c, activation)

    r = jax.vmap(r_single)(x_int, t_int)
    return jnp.mean(r**2), {}


def pde_terms_h1(params, x_int, t_int, c, activation):
    """
    H1 (Sobolev): mean(r²) plus mean(‖∇_xt r‖²).

    ∇_xt r = (∂r/∂x, ∂r/∂t) — 3rd-order derivatives of u.

    Returns (loss_pde, {"sobolev": loss_sob}).
    """
    def r_and_grad_single(xi, ti):
        r = r_scalar_params(params, xi, ti, c, activation)
        dr_dx, dr_dt = jax.grad(r_scalar_params, argnums=(1, 2))(
            params, xi, ti, c, activation
        )
        return r**2, dr_dx**2 + dr_dt**2

    r_sq, grad_sq = jax.vmap(r_and_grad_single)(x_int, t_int)
    return r_sq.mean(), {"sobolev": grad_sq.mean()}


def pde_terms_h2(params, x_int, t_int, c, activation):
    """
    H2: mean(r²) plus mean(‖∇_xt r‖²) plus mean(‖H_xt r‖²_F).

    H_xt r is the 2×2 Hessian of r w.r.t. (x, t) — 4th-order derivatives of u.
    Frobenius norm: ‖H‖²_F = r_xx² + 2*r_xt² + r_tt²

    Returns (loss_pde, {"sobolev": loss_sob, "sobolev2": loss_sob2}).
    """
    def r_of_xt(xt, params, c, activation):
        return r_scalar_params(params, xt[0], xt[1], c, activation)

    def terms_single(xi, ti):
        xt = jnp.stack([xi, ti])
        r = r_of_xt(xt, params, c, activation)

        grad_r = jax.grad(r_of_xt)(xt, params, c, activation)          # (2,)
        hess_r = jax.hessian(r_of_xt)(xt, params, c, activation)       # (2, 2)

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

    Returns
    -------
    (total_loss, components_dict)
    components_dict always has keys "pde" and "ic_ut".
    H1 adds "sobolev"; H2 adds "sobolev" and "sobolev2".
    """
    t0 = jnp.zeros(())

    # --- PDE term ---
    if norm == "L2":
        loss_pde, extra = pde_terms_l2(params, x_int, t_int, c, activation)
    elif norm == "H1":
        loss_pde, extra = pde_terms_h1(params, x_int, t_int, c, activation)
    elif norm == "H2":
        loss_pde, extra = pde_terms_h2(params, x_int, t_int, c, activation)
    else:
        raise ValueError(f"norm must be 'L2', 'H1', or 'H2', got {norm!r}")

    # --- IC velocity term: u_t(x, 0) = 0 ---
    u_t_ic = jax.vmap(lambda xi: _u_t(params, xi, t0, activation))(x_ic)
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
):
    """Scalar wrapper — pass to jax.grad or optax's value_and_grad_from_state."""
    loss, _ = loss_fn(
        params, x_int, t_int, x_ic, c, lambda_ic, activation,
        norm=norm, lambda_sob=lambda_sob, lambda_sob2=lambda_sob2,
    )
    return loss
