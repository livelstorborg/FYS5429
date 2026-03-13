import jax
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# PDE residuals
# -----------------------------------------------------------------------------
def pde_residual(model, xyzt, c, dim):
    """
    Compute PDE residual for wave equation in 1D or 2D.

    Wave equation:
    - 1D: u_tt - c^2 u_xx = 0
    - 2D: u_tt - c^2 (u_xx + u_yy) = 0

    Returns squared residual per point: (N,)
    """

    def u_single(z):
        return model(z[None, :])[0, 0]

    def hessian_single(z):
        return jax.hessian(u_single)(z)

    hess = jax.vmap(hessian_single)(xyzt)  # (N, dim+1, dim+1)

    if dim == 1:
        u_xx = hess[:, 0, 0]
        u_tt = hess[:, 1, 1]
        lap = u_xx
    elif dim == 2:
        u_xx = hess[:, 0, 0]
        u_yy = hess[:, 1, 1]
        u_tt = hess[:, 2, 2]
        lap = u_xx + u_yy
    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")

    r = u_tt - c**2 * lap
    return r**2


def u_t_batch(model, xyt, dim):
    """Batch u_t for dim=1 or dim=2, where time is last coordinate."""

    def u_single(z):
        return model(z[None, :])[0, 0]

    def u_t_single(z):
        g = jax.grad(u_single)(z)
        return g[dim]  # time index: 1 for dim=1, 2 for dim=2

    return jax.vmap(u_t_single)(xyt)


def pde_residual_sobolev(model, xyzt, c, dim):
    """
    Compute squared PDE residual and squared gradient of residual (Sobolev/H1 terms).

    For each collocation point returns (r², ‖∇r‖²) where r is the wave-equation
    residual and ∇r is its gradient w.r.t. all inputs.

    Returns two arrays of shape (N,): (r_sq, sob_sq).
    """

    def u_single(z):
        return model(z[None, :])[0, 0]

    def r_single(z):
        H = jax.hessian(u_single)(z)
        if dim == 1:
            return H[1, 1] - c**2 * H[0, 0]           # u_tt - c²u_xx
        else:
            return H[2, 2] - c**2 * (H[0, 0] + H[1, 1])  # u_tt - c²(u_xx+u_yy)

    def sob_single(z):
        r = r_single(z)
        grad_r = jax.grad(r_single)(z)
        return r**2, jnp.sum(grad_r**2)

    return jax.vmap(sob_single)(xyzt)  # (r_sq, sob_sq), each shape (N,)


# -----------------------------------------------------------------------------
# Loss functions
# -----------------------------------------------------------------------------
def loss_l2(
    model,
    x_int,
    t_int,
    x_ic,
    t_ic,
    *,
    c=1.0,
    lambda_ic=10.0,
    dim=1,
    y_int=None,
    y_ic=None,
    prev_model=None,
    u0_fn=None,
    v0_fn=None,
    t0=None,
):
    """Standard L2 (MSE) loss for the wave-equation PINN."""

    if dim == 1:
        xt_int = jnp.concatenate([x_int, t_int], axis=1)
        loss_pde = pde_residual(model, xt_int, c, dim).mean()

        xt_ic = jnp.concatenate([x_ic, t_ic], axis=1)

        if prev_model is None:
            ut_true = v0_fn(x_ic).squeeze() if v0_fn is not None else jnp.zeros(x_ic.shape[0])
        else:
            ut_true = jax.lax.stop_gradient(u_t_batch(prev_model, xt_ic, dim=1))

        u_t_ic = u_t_batch(model, xt_ic, dim=1)
        loss_ic_ut = jnp.mean((u_t_ic - ut_true) ** 2)

        loss_total = loss_pde + lambda_ic * loss_ic_ut
        return loss_total, {
            "pde": loss_pde,
            "ic_u": 0.0,
            "ic_ut": loss_ic_ut,
            "total": loss_total,
        }

    elif dim == 2:
        if y_int is None or y_ic is None:
            raise ValueError("For dim=2 you must provide y_int and y_ic.")
        if t0 is None:
            raise ValueError("For dim=2 you must provide t0 (window start time).")

        xyt_int = jnp.concatenate([x_int, y_int, t_int], axis=1)
        xyt_ic = jnp.concatenate([x_ic, y_ic, t_ic], axis=1)

        loss_pde = pde_residual(model, xyt_int, c, dim=2).mean()

        if prev_model is None:
            if u0_fn is None or v0_fn is None:
                raise ValueError(
                    "For dim=2 with prev_model=None you must provide u0_fn and v0_fn."
                )
            u_true = u0_fn(x_ic, y_ic).squeeze()
            ut_true = v0_fn(x_ic, y_ic).squeeze()
        else:
            u_true = prev_model(xyt_ic).squeeze()
            ut_true = u_t_batch(prev_model, xyt_ic, dim=2)
            u_true = jax.lax.stop_gradient(u_true)
            ut_true = jax.lax.stop_gradient(ut_true)

        u_pred = model(xyt_ic).squeeze()
        ut_pred = u_t_batch(model, xyt_ic, dim=2)

        loss_ic_u = jnp.mean((u_pred - u_true) ** 2)
        loss_ic_ut = jnp.mean((ut_pred - ut_true) ** 2)

        loss_total = loss_pde + lambda_ic * (loss_ic_u + loss_ic_ut)
        return loss_total, {
            "pde": loss_pde,
            "ic_u": loss_ic_u,
            "ic_ut": loss_ic_ut,
            "total": loss_total,
        }

    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")


def loss_sobolev(
    model,
    x_int,
    t_int,
    x_ic,
    t_ic,
    *,
    c=1.0,
    lambda_ic=10.0,
    dim=1,
    lambda_sob=1.0,
    y_int=None,
    y_ic=None,
    prev_model=None,
    u0_fn=None,
    v0_fn=None,
    t0=None,
):
    """
    Sobolev (H1) loss for the wave-equation PINN.

    Penalises both r² (standard L2 PDE residual) and ‖∇r‖² (gradient of residual
    w.r.t. inputs), weighted by lambda_sob.
    """

    if dim == 1:
        xt_int = jnp.concatenate([x_int, t_int], axis=1)
        r_sq, sob_sq = pde_residual_sobolev(model, xt_int, c, dim)
        loss_pde = r_sq.mean()
        loss_sob = sob_sq.mean()

        xt_ic = jnp.concatenate([x_ic, t_ic], axis=1)

        if prev_model is None:
            ut_true = v0_fn(x_ic).squeeze() if v0_fn is not None else jnp.zeros(x_ic.shape[0])
        else:
            ut_true = jax.lax.stop_gradient(u_t_batch(prev_model, xt_ic, dim=1))

        u_t_ic = u_t_batch(model, xt_ic, dim=1)
        loss_ic_ut = jnp.mean((u_t_ic - ut_true) ** 2)

        loss_total = loss_pde + lambda_ic * loss_ic_ut + lambda_sob * loss_sob
        return loss_total, {
            "pde": loss_pde,
            "ic_u": 0.0,
            "ic_ut": loss_ic_ut,
            "sobolev": loss_sob,
            "total": loss_total,
        }

    elif dim == 2:
        if y_int is None or y_ic is None:
            raise ValueError("For dim=2 you must provide y_int and y_ic.")
        if t0 is None:
            raise ValueError("For dim=2 you must provide t0 (window start time).")

        xyt_int = jnp.concatenate([x_int, y_int, t_int], axis=1)
        xyt_ic = jnp.concatenate([x_ic, y_ic, t_ic], axis=1)

        r_sq, sob_sq = pde_residual_sobolev(model, xyt_int, c, dim)
        loss_pde = r_sq.mean()
        loss_sob = sob_sq.mean()

        if prev_model is None:
            if u0_fn is None or v0_fn is None:
                raise ValueError(
                    "For dim=2 with prev_model=None you must provide u0_fn and v0_fn."
                )
            u_true = u0_fn(x_ic, y_ic).squeeze()
            ut_true = v0_fn(x_ic, y_ic).squeeze()
        else:
            u_true = prev_model(xyt_ic).squeeze()
            ut_true = u_t_batch(prev_model, xyt_ic, dim=2)
            u_true = jax.lax.stop_gradient(u_true)
            ut_true = jax.lax.stop_gradient(ut_true)

        u_pred = model(xyt_ic).squeeze()
        ut_pred = u_t_batch(model, xyt_ic, dim=2)

        loss_ic_u = jnp.mean((u_pred - u_true) ** 2)
        loss_ic_ut = jnp.mean((ut_pred - ut_true) ** 2)

        loss_total = loss_pde + lambda_ic * (loss_ic_u + loss_ic_ut) + lambda_sob * loss_sob
        return loss_total, {
            "pde": loss_pde,
            "ic_u": loss_ic_u,
            "ic_ut": loss_ic_ut,
            "sobolev": loss_sob,
            "total": loss_total,
        }

    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")


def loss_fn(
    model,
    x_int,
    t_int,
    x_ic,
    t_ic,
    *,
    c=1.0,
    lambda_ic=10.0,
    dim=1,
    norm="L2",
    lambda_sob=1.0,
    y_int=None,
    y_ic=None,
    prev_model=None,
    u0_fn=None,
    v0_fn=None,
    t0=None,
):
    """
    Unified loss dispatcher.

    norm="L2"     — standard MSE on PDE residual (default, backward compatible)
    norm="Sobolev" — MSE on residual + lambda_sob * MSE on gradient of residual (H1)

    dim=1:
      - PDE residual over interior (x,t)
      - IC: u(x,0)=sin(pi x) (soft)
      - IC: u_t(x,0)=0 (soft)

    dim=2 (time-marching flavor):
      - PDE residual over interior (x,y,t) in window
      - IC at t=t0:
          if prev_model is None: match (u0_fn, v0_fn)
          else: match (prev_model, u_t(prev_model)) with stop_gradient

    Inputs for dim=2:
      - y_int, y_ic required
      - t_ic should be filled with t0 (shape (N_ic,1))
      - prev_model optional
      - u0_fn, v0_fn required if prev_model is None
      - t0 required (float)
    """
    if norm == "L2":
        return loss_l2(
            model, x_int, t_int, x_ic, t_ic,
            c=c, lambda_ic=lambda_ic, dim=dim,
            y_int=y_int, y_ic=y_ic, prev_model=prev_model,
            u0_fn=u0_fn, v0_fn=v0_fn, t0=t0,
        )
    elif norm == "Sobolev":
        return loss_sobolev(
            model, x_int, t_int, x_ic, t_ic,
            c=c, lambda_ic=lambda_ic, dim=dim, lambda_sob=lambda_sob,
            y_int=y_int, y_ic=y_ic, prev_model=prev_model,
            u0_fn=u0_fn, v0_fn=v0_fn, t0=t0,
        )
    else:
        raise ValueError(f"norm must be 'L2' or 'Sobolev', got {norm!r}")
