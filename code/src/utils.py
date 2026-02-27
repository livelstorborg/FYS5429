import jax
import jax.numpy as jnp
from jax import vmap, jacfwd

try:
    from src.pde import u_exact_1d, u_exact_2d
except ModuleNotFoundError:
    from pde import u_exact_1d, u_exact_2d


# -----------------------------------------------------------------------------
# Error metrics
# -----------------------------------------------------------------------------
def compute_error_metrics_1d(model, x, t, c=1.0):
    u_true = u_exact_1d(x, t=t, c=c)

    X, T = jnp.meshgrid(x, t)
    xt = jnp.stack([X.ravel(), T.ravel()], axis=1)
    u_pred = model(xt).squeeze().reshape(len(t), len(x))

    err = u_pred - u_true
    relL2 = jnp.sqrt(jnp.mean(err**2)) / (jnp.sqrt(jnp.mean(u_true**2)) + 1e-12)
    Linf = jnp.max(jnp.abs(err))
    mae = jnp.mean(jnp.abs(err))
    rmse = jnp.sqrt(jnp.mean(err**2))

    # H1 semi-norm: error in du/dx
    def u_pred_scalar(xt_single):
        return model(xt_single[None]).squeeze()

    dudx_pred = vmap(jacfwd(u_pred_scalar))(xt)[:, 0]  # x-component
    dudx_pred = dudx_pred.reshape(len(t), len(x))
    dudx_true = jnp.gradient(u_true, x, axis=1)

    err_grad = dudx_pred - dudx_true
    h1 = jnp.sqrt(jnp.mean(err**2) + jnp.mean(err_grad**2))

    return float(relL2), float(Linf), float(mae), float(rmse), float(h1)


def compute_error_metrics_2d(model, x, y, t, c=1.0):
    u_pred_list = []
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    XY = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    for ti in t:
        T_grid = jnp.full((XY.shape[0], 1), ti)
        xyt = jnp.concatenate([XY, T_grid], axis=1)
        u_pred_t = model(xyt).squeeze().reshape(len(x), len(y))
        u_pred_list.append(u_pred_t)
    u_pred = jnp.stack(u_pred_list, axis=0)  # (Nt, Nx, Ny)

    u_true_list = [u_exact_2d(X, Y, ti, c=c) for ti in t]
    u_true = jnp.stack(u_true_list, axis=0)

    err = u_pred - u_true
    relL2 = jnp.sqrt(jnp.mean(err**2)) / (jnp.sqrt(jnp.mean(u_true**2)) + 1e-12)
    Linf = jnp.max(jnp.abs(err))
    mae = jnp.mean(jnp.abs(err))
    rmse = jnp.sqrt(jnp.mean(err**2))

    # H1 semi-norm: error in du/dx and du/dy
    def u_pred_scalar(xyt_single):
        return model(xyt_single[None]).squeeze()

    all_xyt = []
    for ti in t:
        T_grid = jnp.full((XY.shape[0], 1), ti)
        xyt = jnp.concatenate([XY, T_grid], axis=1)
        all_xyt.append(xyt)
    all_xyt = jnp.concatenate(all_xyt, axis=0)  # (Nt*Nx*Ny, 3)

    jac = vmap(jacfwd(u_pred_scalar))(all_xyt)  # (Nt*Nx*Ny, 3)
    dudx_pred = jac[:, 0].reshape(len(t), len(x), len(y))
    dudy_pred = jac[:, 1].reshape(len(t), len(x), len(y))

    dudx_true = jnp.gradient(u_true, x, axis=1)
    dudy_true = jnp.gradient(u_true, y, axis=2)

    err_dx = dudx_pred - dudx_true
    err_dy = dudy_pred - dudy_true
    h1 = jnp.sqrt(jnp.mean(err**2) + jnp.mean(err_dx**2) + jnp.mean(err_dy**2))

    return float(relL2), float(Linf), float(mae), float(rmse), float(h1)
