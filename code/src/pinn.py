import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import matplotlib.pyplot as plt

# If you have a separate PDE module, keep this import.
# Otherwise, you can comment it out and use the u_exact defined below.
# from .pde import u_exact


# -----------------------------------------------------------------------------
# Exact solutions / ICs (can be replaced by your own in .pde)
# -----------------------------------------------------------------------------
def u0_fn_1d(x):
    # u(x,0) = sin(pi x)
    return jnp.sin(jnp.pi * x)


def v0_fn_1d(x):
    # u_t(x,0) = 0
    return jnp.zeros_like(x)


def u_exact_1d(x, t, c=1.0):
    x = jnp.asarray(x)
    t = jnp.asarray(t)
    return jnp.sin(jnp.pi * x[None, :]) * jnp.cos(c * jnp.pi * t[:, None])


def u0_fn_2d(x, y):
    # u(x,y,0) = sin(pi x) sin(pi y)
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)


def v0_fn_2d(x, y):
    # u_t(x,y,0) = 0
    return jnp.zeros_like(x)


def u_exact_2d(x, y, t, c=1.0):
    # Exact for 2D wave with u0=sin(pi x)sin(pi y), v0=0, BC u=0 on boundary via sine factors
    omega = c * jnp.pi * jnp.sqrt(2.0)
    return jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y) * jnp.cos(omega * t)


def u_exact_dispatch(x, y=None, t=None, c=1.0, dim=1):
    if dim == 1:
        return u_exact_1d(x, t=t, c=c)
    elif dim == 2:
        if y is None:
            raise ValueError("y must be provided for dim=2")
        return u_exact_2d(x, y, t=t, c=c)
    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")


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


class PINN_HardBC(nnx.Module):
    """
    Single model that supports dim=1 or dim=2, with hard BC and (optionally) hard IC.

    dim=1 trial solution (hard BC only):
        u(x,t) = sin(pi x) * N(x,t)

    dim=2 time-marching style (hard IC via additive form + hard boundary via sine factor):
        u(x,y,t) = u0(x,y) + t * sin(pi x) sin(pi y) * N(x,y,t)

    Notes:
      - For dim=1, IC is enforced in the loss (soft).
      - For dim=2 (time-marching), IC at the window start is enforced in the loss, but
        the architecture also makes u(x,y,t0) close to prev window by loss; the additive
        structure makes it easy to keep u(x,y,0)=u0(x,y) if you start at t0=0.
    """

    def __init__(self, layers, activations, key, *, dim=1, u0_fn=None):
        self.network = MLP(layers, activations, key)
        self.dim = dim
        self.u0_fn = u0_fn  # used for dim=2 additive ansatz (initial displacement)

        if self.dim == 2 and self.u0_fn is None:
            raise ValueError("For dim=2, you must provide u0_fn(x,y).")

    def __call__(self, inp):
        if self.dim == 1:
            # inp: (N,2) = [x,t]
            x = inp[:, 0:1]
            N = self.network(inp)
            return jnp.sin(jnp.pi * x) * N

        elif self.dim == 2:
            # inp: (N,3) = [x,y,t]
            x = inp[:, 0:1]
            y = inp[:, 1:2]
            t = inp[:, 2:3]

            S = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
            N = self.network(inp)
            u0 = self.u0_fn(x, y)

            # Additive ansatz (common in time-marching setups)
            return u0 + t * S * N

        else:
            raise ValueError(f"dim must be 1 or 2, got {self.dim}")


# -----------------------------------------------------------------------------
# Loss / residuals
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
    # 2D extras (for time marching / window IC)
    y_int=None,
    y_ic=None,
    prev_model=None,
    u0_fn=None,
    v0_fn=None,
    t0=None,
):
    """
    Unified loss.

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

    if dim == 1:
        xt_int = jnp.concatenate([x_int, t_int], axis=1)
        loss_pde = pde_residual(model, xt_int, c, dim).mean()

        xt_ic = jnp.concatenate([x_ic, t_ic], axis=1)
        u_ic_pred = model(xt_ic)
        u_ic_true = jnp.sin(jnp.pi * x_ic)
        loss_ic_u = jnp.mean((u_ic_pred - u_ic_true) ** 2)

        u_t_ic = u_t_batch(model, xt_ic, dim=1)
        loss_ic_ut = jnp.mean(u_t_ic**2)

        loss_total = loss_pde + lambda_ic * (loss_ic_u + loss_ic_ut)
        return loss_total, {
            "pde": loss_pde,
            "ic_u": loss_ic_u,
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


# -----------------------------------------------------------------------------
# Sampling
# -----------------------------------------------------------------------------
def sample_points_wave_1d(key, N_int=512, N_ic=100, T=2.0, L=1.0):
    k1, k2, k3 = jax.random.split(key, 3)
    x_int = jax.random.uniform(k1, (N_int, 1), minval=0.0, maxval=L)
    t_int = jax.random.uniform(k2, (N_int, 1), minval=0.0, maxval=T)

    x_ic = jax.random.uniform(k3, (N_ic, 1), minval=0.0, maxval=L)
    t_ic = jnp.zeros((N_ic, 1))
    return x_int, t_int, x_ic, t_ic


def sample_points_wave_2d_window(key, N_int, N_ic, t0, t1, L=1.0):
    k1, k2, k3, k4, k5 = jax.random.split(key, 5)

    x_int = jax.random.uniform(k1, (N_int, 1), minval=0.0, maxval=L)
    y_int = jax.random.uniform(k2, (N_int, 1), minval=0.0, maxval=L)
    t_int = jax.random.uniform(k3, (N_int, 1), minval=t0, maxval=t1)

    x_ic = jax.random.uniform(k4, (N_ic, 1), minval=0.0, maxval=L)
    y_ic = jax.random.uniform(k5, (N_ic, 1), minval=0.0, maxval=L)
    t_ic = jnp.full((N_ic, 1), t0)

    return x_int, y_int, t_int, x_ic, y_ic, t_ic


# -----------------------------------------------------------------------------
# Training
# -----------------------------------------------------------------------------
def train_pinn(
    layers=None,
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
    optimizer="adam",
    grad_clip=1.0,
    # 2D time-marching extras
    n_windows=5,
    steps_per_window=None,
    u0_fn=None,
    v0_fn=None,
):
    """
    Train PINN for wave equation in 1D or 2D.

    dim=1:
      - trains on full time [0,T] with soft IC

    dim=2:
      - trains with time-marching windows (like your file #2)
      - uses n_windows windows across [0,T]
      - trains each window for steps_per_window (defaults to steps//n_windows)
      - IC matching uses (u0_fn,v0_fn) for first window, then prev_model for later
    """

    if layers is None:
        layers = [2, 64, 64, 64, 1] if dim == 1 else [3, 64, 64, 64, 1]

    if activations is None:
        activations = [jax.nn.tanh] * (len(layers) - 2)

    main_key = jax.random.PRNGKey(seed)
    key_model, key_loop = jax.random.split(main_key)

    if dim == 1:
        model = PINN_HardBC(layers, activations, key=key_model, dim=1)
        key = key_loop

        # Optional L-BFGS branch kept from file #1
        if optimizer.lower() == "lbfgs":
            from jaxopt import LBFGS

            x_int, t_int, x_ic, t_ic = sample_points_wave_1d(key, N_int, N_ic, T=T, L=L)

            def loss_for_lbfgs(model_state):
                temp_model = nnx.clone(model)
                nnx.update(temp_model, model_state)

                loss_val, _ = loss_fn(
                    temp_model,
                    x_int,
                    t_int,
                    x_ic,
                    t_ic,
                    c=c,
                    lambda_ic=lambda_ic,
                    dim=1,
                )
                is_bad = jnp.isnan(loss_val) | jnp.isinf(loss_val)
                loss_val = jnp.where(is_bad, 1e6, loss_val)
                loss_val = jnp.clip(loss_val, 0.0, 1e6)
                return loss_val

            params_init = nnx.state(model, nnx.Param)
            solver = LBFGS(
                fun=loss_for_lbfgs,
                maxiter=steps,
                tol=1e-6,
                linesearch="backtracking",
                maxls=15,
            )

            print(f"Training PINN for 1D wave equation (c={c}, T={T})...")
            print("Optimizer: L-BFGS (JAXopt)")
            print(f"Architecture: {layers}")
            print(f"Max iterations: {steps}")
            print("-" * 60)

            result = solver.run(params_init)
            nnx.update(model, result.params)

            final_loss, comps = loss_fn(
                model, x_int, t_int, x_ic, t_ic, c=c, lambda_ic=lambda_ic, dim=1
            )

            print(f"[L-BFGS] Converged in {result.state.iter_num} iterations")
            print(f"Final loss: {float(final_loss):.3e}")
            print(
                f"  pde={float(comps['pde']):.3e} | "
                f"ic_u={float(comps['ic_u']):.3e} | "
                f"ic_ut={float(comps['ic_ut']):.3e}"
            )

            losses = jnp.array([float(final_loss)])
            loss_components = {
                "pde": [float(comps["pde"])],
                "ic_u": [float(comps["ic_u"])],
                "ic_ut": [float(comps["ic_ut"])],
            }
            return model, losses, loss_components

        # First-order optimizers (same style as file #1)
        schedule = optax.exponential_decay(
            init_value=lr,
            transition_steps=1000,
            decay_rate=0.95,
        )

        if not hasattr(optax, optimizer):
            raise ValueError(
                f"Unknown optimizer: {optimizer}. "
                f"Available: adam, adamw, sgd, rmsprop, nadam"
            )

        base_optimizer = getattr(optax, optimizer)(schedule)

        if grad_clip is not None and grad_clip > 0:
            tx = optax.chain(
                optax.clip_by_global_norm(grad_clip),
                base_optimizer,
            )
            opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
            print(f"Gradient clipping enabled: max_norm={grad_clip}")
        else:
            opt = nnx.Optimizer(model, base_optimizer, wrt=nnx.Param)

        losses = []
        loss_components = {"pde": [], "ic_u": [], "ic_ut": []}

        @jax.jit
        def train_step(model, opt, key):
            key, subkey = jax.random.split(key)
            x_int, t_int, x_ic, t_ic = sample_points_wave_1d(
                subkey, N_int, N_ic, T=T, L=L
            )

            def loss_func(m):
                return loss_fn(
                    m, x_int, t_int, x_ic, t_ic, c=c, lambda_ic=lambda_ic, dim=1
                )

            (loss, comps), grads = nnx.value_and_grad(loss_func, has_aux=True)(model)
            opt.update(model, grads)
            return model, opt, key, loss, comps

        print(f"Training PINN for 1D wave equation (c={c}, T={T})...")
        print(f"Optimizer: {optimizer}")
        print(f"Architecture: {layers}")
        print(f"Steps: {steps}, N_int: {N_int}, N_ic: {N_ic}")
        print("-" * 60)

        for step in range(steps):
            model, opt, key, loss, comps = train_step(model, opt, key)

            losses.append(float(loss))
            for k in ["pde", "ic_u", "ic_ut"]:
                loss_components[k].append(float(comps[k]))

            if step % 500 == 0 or step == steps - 1:
                print(
                    f"[{step:5d}] loss={float(loss):.3e} | "
                    f"pde={float(comps['pde']):.3e} | "
                    f"ic_u={float(comps['ic_u']):.3e} | "
                    f"ic_ut={float(comps['ic_ut']):.3e}"
                )

        return model, jnp.array(losses), loss_components

    # ---------------------------
    # dim=2: time-marching training
    # ---------------------------
    if u0_fn is None:
        u0_fn = u0_fn_2d
    if v0_fn is None:
        v0_fn = v0_fn_2d

    model = PINN_HardBC(layers, activations, key=key_model, dim=2, u0_fn=u0_fn)
    key = key_loop

    if steps_per_window is None:
        steps_per_window = max(1, steps // max(1, n_windows))

    dt = T / n_windows
    prev_model = None
    histories = []

    # Optimizer for 2D windows (keep same “feel” as file #1, but per-window like file #2)
    tx = optax.chain(
        optax.clip_by_global_norm(grad_clip if grad_clip is not None else 1e9),
        optax.adam(lr),
    )

    @nnx.jit
    def train_step_window(model, opt, key, t0, t1):
        key, sub = jax.random.split(key)
        x_int, y_int, t_int, x_ic, y_ic, t_ic = sample_points_wave_2d_window(
            sub, N_int, N_ic, t0, t1, L=L
        )

        def loss_func(m):
            return loss_fn(
                m,
                x_int,
                t_int,
                x_ic,
                t_ic,
                c=c,
                lambda_ic=lambda_ic,
                dim=2,
                y_int=y_int,
                y_ic=y_ic,
                prev_model=prev_model,
                u0_fn=u0_fn,
                v0_fn=v0_fn,
                t0=t0,
            )

        (loss, comps), grads = nnx.value_and_grad(loss_func, has_aux=True)(model)
        opt.update(model, grads)
        return model, opt, key, loss, comps

    print(f"Training PINN for 2D wave equation (c={c}, T={T})...")
    print("Optimizer: adam (time-marching)")
    print(f"Architecture: {layers}")
    print(f"Windows: {n_windows}, steps_per_window: {steps_per_window}")
    print(f"N_int: {N_int}, N_ic: {N_ic}, lambda_ic: {lambda_ic}, lr: {lr}")
    print("-" * 60)

    for k in range(n_windows):
        t0 = k * dt
        t1 = (k + 1) * dt

        opt = nnx.Optimizer(model, tx, wrt=nnx.Param)
        hist = {"loss": [], "pde": [], "ic_u": [], "ic_ut": []}

        for s in range(steps_per_window):
            model, opt, key, loss, comps = train_step_window(model, opt, key, t0, t1)

            hist["loss"].append(float(loss))
            hist["pde"].append(float(comps["pde"]))
            hist["ic_u"].append(float(comps["ic_u"]))
            hist["ic_ut"].append(float(comps["ic_ut"]))

            log_every = max(1, steps_per_window // 10)
            if s % log_every == 0 or s == steps_per_window - 1:
                print(
                    f"[window {t0:.3f}->{t1:.3f}] step {s:5d} | "
                    f"loss={float(loss):.3e} "
                    f"pde={float(comps['pde']):.3e} "
                    f"ic_u={float(comps['ic_u']):.3e} "
                    f"ic_ut={float(comps['ic_ut']):.3e}"
                )

        histories.append(hist)
        prev_model = nnx.clone(model)

    # Return the same signature as 1D: model, losses, loss_components
    # For 2D, "losses" will be concatenated loss history across windows.
    all_losses = jnp.array([v for h in histories for v in h["loss"]])
    loss_components = {
        "pde": [v for h in histories for v in h["pde"]],
        "ic_u": [v for h in histories for v in h["ic_u"]],
        "ic_ut": [v for h in histories for v in h["ic_ut"]],
        "histories": histories,  # keep per-window too
    }
    return model, all_losses, loss_components


# -----------------------------------------------------------------------------
# Error metrics / reports
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
    return float(relL2), float(Linf), float(mae), float(rmse)


def compute_error_metrics_2d(model, x, y, t, c=1.0):
    u_true = u_exact_2d(
        x[:, None], y[None, :], t=t, c=c
    )  # broadcast-friendly if needed

    # Build predictions across time
    u_pred_list = []
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    XY = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    for ti in t:
        T_grid = jnp.full((XY.shape[0], 1), ti)
        xyt = jnp.concatenate([XY, T_grid], axis=1)
        u_pred_t = model(xyt).squeeze().reshape(len(x), len(y))
        u_pred_list.append(u_pred_t)
    u_pred = jnp.stack(u_pred_list, axis=0)  # (Nt,Nx,Ny)

    # Build u_true with matching shape
    # u_exact_2d expects arrays shaped like X,Y and scalar t. We'll loop to match u_pred.
    u_true_list = [u_exact_2d(X, Y, ti, c=c) for ti in t]
    u_true = jnp.stack(u_true_list, axis=0)

    err = u_pred - u_true
    relL2 = jnp.sqrt(jnp.mean(err**2)) / (jnp.sqrt(jnp.mean(u_true**2)) + 1e-12)
    Linf = jnp.max(jnp.abs(err))
    mae = jnp.mean(jnp.abs(err))
    rmse = jnp.sqrt(jnp.mean(err**2))
    return float(relL2), float(Linf), float(mae), float(rmse)


def error_report_2d_wave(
    model,
    u_exact_fn,
    times,
    *,
    c=1.0,
    L=1.0,
    Nx=81,
    Ny=81,
    make_snapshot=True,
    snapshot_metric="relL2",  # "relL2" or "Linf"
    logy=True,
    title_prefix="",
):
    times = jnp.asarray(times, dtype=float)

    x = jnp.linspace(0.0, L, Nx)
    y = jnp.linspace(0.0, L, Ny)
    X, Y = jnp.meshgrid(x, y, indexing="ij")
    XY = jnp.stack([X.ravel(), Y.ravel()], axis=1)

    @nnx.jit
    def eval_at_time(t):
        t = jnp.asarray(t)
        T = jnp.full((XY.shape[0], 1), t)
        xyt = jnp.concatenate([XY, T], axis=1)

        u_pred = model(xyt).reshape(Nx, Ny)
        u_true = u_exact_fn(X, Y, t, c=c)

        err = u_pred - u_true
        relL2 = jnp.sqrt(jnp.mean(err**2)) / (jnp.sqrt(jnp.mean(u_true**2)) + 1e-12)
        Linf = jnp.max(jnp.abs(err))
        mae = jnp.mean(jnp.abs(err))
        rmse = jnp.sqrt(jnp.mean(err**2))
        return relL2, Linf, mae, rmse, u_pred, u_true, err

    relL2_list, Linf_list, mae_list, rmse_list = [], [], [], []
    snaps = {}

    for t in times:
        relL2, Linf, mae, rmse, u_pred, u_true, err = eval_at_time(t)
        relL2_list.append(float(relL2))
        Linf_list.append(float(Linf))
        mae_list.append(float(mae))
        rmse_list.append(float(rmse))

        if make_snapshot:
            snaps[float(t)] = {
                "u_pred": jnp.array(u_pred),
                "u_true": jnp.array(u_true),
                "err": jnp.array(err),
            }

    relL2_arr = jnp.array(relL2_list)
    Linf_arr = jnp.array(Linf_list)
    mae_arr = jnp.array(mae_list)
    rmse_arr = jnp.array(rmse_list)

    worst_rel_idx = int(jnp.argmax(relL2_arr))
    worst_inf_idx = int(jnp.argmax(Linf_arr))

    summary = {
        "relL2_mean": float(relL2_arr.mean()),
        "relL2_median": float(jnp.median(relL2_arr)),
        "relL2_max": float(relL2_arr[worst_rel_idx]),
        "relL2_max_time": float(times[worst_rel_idx]),
        "Linf_mean": float(Linf_arr.mean()),
        "Linf_max": float(Linf_arr[worst_inf_idx]),
        "Linf_max_time": float(times[worst_inf_idx]),
        "mae_mean": float(mae_arr.mean()),
        "rmse_mean": float(rmse_arr.mean()),
    }

    plt.figure(figsize=(8, 4.8))
    plt.plot(times, relL2_arr, marker="o", label="Relative L2")
    plt.plot(times, Linf_arr, marker="s", label="L∞")
    plt.plot(times, mae_arr, marker="^", label="MAE")
    plt.plot(times, rmse_arr, marker="D", label="RMSE")
    plt.xlabel("Time t")
    plt.ylabel("Error")
    ttl = "Error metrics vs time"
    if title_prefix:
        ttl = f"{title_prefix} — {ttl}"
    plt.title(ttl)
    if logy:
        plt.yscale("log")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.show()

    if make_snapshot and len(snaps) > 0:
        if snapshot_metric == "Linf":
            t_snap = float(times[worst_inf_idx])
            snap_label = "worst L∞"
        else:
            t_snap = float(times[worst_rel_idx])
            snap_label = "worst rel L2"

        err = snaps[t_snap]["err"]
        abs_err = jnp.abs(err)

        plt.figure(figsize=(15, 4.5))

        plt.subplot(1, 3, 1)
        plt.imshow(snaps[t_snap]["u_true"], origin="lower", aspect="auto")
        plt.title(f"u_true at t={t_snap:.4f}")
        plt.colorbar()

        plt.subplot(1, 3, 2)
        plt.imshow(snaps[t_snap]["u_pred"], origin="lower", aspect="auto")
        plt.title(f"u_pred at t={t_snap:.4f}")
        plt.colorbar()

        plt.subplot(1, 3, 3)
        plt.imshow(abs_err, origin="lower", aspect="auto")
        plt.title(f"|error| at t={t_snap:.4f} ({snap_label})")
        plt.colorbar()

        plt.tight_layout()
        plt.show()

    print("Accuracy summary:")
    print(
        f"  relL2: mean={summary['relL2_mean']:.3e}, median={summary['relL2_median']:.3e}, "
        f"max={summary['relL2_max']:.3e} at t={summary['relL2_max_time']:.4f}"
    )
    print(
        f"  Linf : mean={summary['Linf_mean']:.3e}, max={summary['Linf_max']:.3e} at t={summary['Linf_max_time']:.4f}"
    )
    print(f"  MAE  : mean={summary['mae_mean']:.3e}")
    print(f"  RMSE : mean={summary['rmse_mean']:.3e}")

    return {
        "times": times,
        "relL2": relL2_arr,
        "Linf": Linf_arr,
        "mae": mae_arr,
        "rmse": rmse_arr,
        "summary": summary,
    }


# -----------------------------------------------------------------------------
# Example usage
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    # -------------------
    # Choose dimension here
    # -------------------
    dim = 1  # set to 1 or 2

    c = 1.0
    L = 1.0
    T_final = 0.5 if dim == 2 else 2.0

    if dim == 1:
        model, losses, loss_comps = train_pinn(
            dim=1,
            layers=[2, 64, 64, 64, 1],
            activations=[jax.nn.tanh] * 3,
            steps=5000,
            N_int=1000,
            N_ic=200,
            T=T_final,
            L=L,
            c=c,
            lambda_ic=10.0,
            lr=1e-3,
            seed=42,
            optimizer="adam",
            grad_clip=1.0,
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
        plt.semilogy(loss_comps["pde"], label="PDE")
        plt.semilogy(loss_comps["ic_u"], label="IC u")
        plt.semilogy(loss_comps["ic_ut"], label="IC ∂u/∂t")
        plt.xlabel("Step")
        plt.ylabel("Loss component")
        plt.title("Loss Components")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Errors on grid
        x = jnp.linspace(0.0, 1.0, 201)
        t = jnp.linspace(0.0, T_final, 51)
        relL2, Linf, mae, rmse = compute_error_metrics_1d(model, x, t, c=c)
        print(
            f"1D errors: relL2={relL2:.3e}, Linf={Linf:.3e}, MAE={mae:.3e}, RMSE={rmse:.3e}"
        )

    else:
        # 2D time-marching settings
        n_windows = 5
        steps_per_window = 4000

        model, losses, loss_comps = train_pinn(
            dim=2,
            layers=[3, 64, 64, 64, 1],
            activations=[jax.nn.relu] * 3,
            steps=n_windows
            * steps_per_window,  # used only to default steps_per_window if None
            steps_per_window=steps_per_window,
            n_windows=n_windows,
            N_int=2000,
            N_ic=200,
            T=T_final,
            L=L,
            c=c,
            lambda_ic=100.0,
            lr=1e-5,
            seed=0,
            grad_clip=1.0,
            u0_fn=u0_fn_2d,
            v0_fn=v0_fn_2d,
        )

        # Plot losses (flattened across windows)
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.semilogy(losses)
        plt.xlabel("Step (across windows)")
        plt.ylabel("Loss")
        plt.title("Training Loss (2D time-marching)")
        plt.grid(True)

        plt.subplot(1, 2, 2)
        plt.semilogy(loss_comps["pde"], label="PDE")
        plt.semilogy(loss_comps["ic_u"], label="IC u")
        plt.semilogy(loss_comps["ic_ut"], label="IC ∂u/∂t")
        plt.xlabel("Step (across windows)")
        plt.ylabel("Loss component")
        plt.title("Loss Components")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # Error report at window boundaries (like file #2)
        times = jnp.linspace(0.0, T_final, n_windows + 1)
        report = error_report_2d_wave(
            model,
            u_exact_fn=u_exact_2d,
            times=times,
            c=c,
            L=L,
            Nx=61,
            Ny=61,
            logy=False,
            make_snapshot=True,
            snapshot_metric="relL2",
            title_prefix="Time-marching PINN (merged)",
        )
