import jax
import jax.numpy as jnp
import flax.nnx as nnx
import optax
import matplotlib.pyplot as plt

try:
    from src.pde import u0_fn_2d, v0_fn_2d
    from src.loss import loss_fn, u_t_batch
except ModuleNotFoundError:
    from pde import u0_fn_2d, v0_fn_2d
    from loss import loss_fn, u_t_batch



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
        self.u_ic_fn = u0_fn  # IC function for current window; updated each window
        self.t0 = 0.0         # window start time; updated each window

        if self.dim == 2 and self.u_ic_fn is None:
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

            tau = t - self.t0  # window-relative time: 0 at window start
            S = jnp.sin(jnp.pi * x) * jnp.sin(jnp.pi * y)
            N = self.network(inp)
            u_ic = self.u_ic_fn(x, y)

            return u_ic + tau * S * N

        else:
            raise ValueError(f"dim must be 1 or 2, got {self.dim}")


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
    n_windows=5,
    steps_per_window=None,
    u0_fn=None,
    v0_fn=None,
    norm="L2",
    lambda_sob=1.0,
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

    def make_ic_fn(prev, t0_val):
        """Return a spatial IC function u_ic(x,y) = prev_model(x,y,t0_val)."""
        return lambda x, y: prev(
            jnp.concatenate([x, y, jnp.full_like(x, t0_val)], axis=1)
        )

    opt = nnx.Optimizer(model, tx, wrt=nnx.Param)

    for k in range(n_windows):
        t0 = k * dt
        t1 = (k + 1) * dt

        # Update the model's window offset and IC so the hard constraint is correct
        model.t0 = t0
        model.u_ic_fn = u0_fn if prev_model is None else make_ic_fn(prev_model, t0)
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

    all_losses = jnp.array([v for h in histories for v in h["loss"]])
    loss_components = {
        "pde": [v for h in histories for v in h["pde"]],
        "ic_u": [v for h in histories for v in h["ic_u"]],
        "ic_ut": [v for h in histories for v in h["ic_ut"]],
        "histories": histories,
    }
    return model, all_losses, loss_components


if __name__ == "__main__":
    from plotting import error_report_2d_wave
    from pde import u_exact_2d

    # --- Parameters ---
    c           = 1.0
    L           = 1.0
    T_final     = 1.0
    n_windows   = 5
    steps_per_window = 2000
    N_int       = 2000
    N_ic        = 200
    lambda_ic   = 100.0
    lr          = 1e-3
    seed        = 0

    # --- Train ---
    model, losses, loss_comps = train_pinn(
        dim=2,
        layers=[3, 64, 64, 64, 1],
        activations=[jax.nn.tanh] * 3,
        n_windows=n_windows,
        steps_per_window=steps_per_window,
        N_int=N_int,
        N_ic=N_ic,
        T=T_final,
        L=L,
        c=c,
        lambda_ic=lambda_ic,
        lr=lr,
        seed=seed,
        grad_clip=1.0,
        u0_fn=u0_fn_2d,
        v0_fn=v0_fn_2d,
        norm="Sobolev",
    )

    # --- Training loss plot ---
    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.semilogy(losses)
    plt.xlabel("Step (across windows)")
    plt.ylabel("Loss")
    plt.title("Training Loss")
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

    # --- Error metrics ---
    times = jnp.linspace(0.0, T_final, n_windows + 1)
    report = error_report_2d_wave(
        model,
        u_exact_fn=u_exact_2d,
        times=times,
        c=c,
        L=L,
        Nx=61,
        Ny=61,
        logy=True,
        make_snapshot=True,
        snapshot_metric="relL2",
        title_prefix="2D PINN",
    )
