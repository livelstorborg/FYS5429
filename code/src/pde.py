import jax
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Analytical Solutions
# -----------------------------------------------------------------------------
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


def u_exact(x, t, y=None, c=1.0, dim=1):
    """
    Analytical solution for wave equation in 1D or 2D.
    """
    if dim == 1:
        T, X = jnp.meshgrid(t, x, indexing="ij")
        u = jnp.sin(jnp.pi * X) * jnp.cos(jnp.pi * c * T)

    elif dim == 2:
        if y is None:
            raise ValueError("y must be provided for 2D")
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        spatial = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        temporal = jnp.cos(jnp.sqrt(2) * jnp.pi * c * t)
        u = temporal[:, None, None] * spatial[None, :, :]

    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")

    return u


# -----------------------------------------------------------------------------
# Grid Creation
# -----------------------------------------------------------------------------
def create_grid(Nx, Ny=None, T=1.0, c=1.0, cfl=0.5, dim=1):
    if dim == 1:
        dx = 1.0 / Nx
        dt = cfl * dx / c
        Nt = int(T / dt)
        x = jnp.linspace(0, 1, Nx + 1)
        t = jnp.arange(0, Nt + 1) * dt
        return x, t, dx, dt

    elif dim == 2:
        if Ny is None:
            raise ValueError("Ny must be provided for 2D")
        dx = 1.0 / Nx
        dy = 1.0 / Ny
        dt = cfl * min(dx, dy) / (c * jnp.sqrt(2))
        Nt = int(T / dt)
        x = jnp.linspace(0, 1, Nx + 1)
        y = jnp.linspace(0, 1, Ny + 1)
        t = jnp.arange(0, Nt + 1) * dt
        return x, y, t, dx, dy, dt

    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")


# -----------------------------------------------------------------------------
# Metamaterial wave-speed helpers
# -----------------------------------------------------------------------------
def _smooth_indicator(coords, z, eps, delta):
    """
    Smooth indicator for a single particle at position z.
    coords : tuple of arrays, one per spatial dimension
    z      : tuple of scalars, particle centre coordinates
    eps    : particle radius
    delta  : interface width
    """
    dist = jnp.sqrt(sum((ci - zi) ** 2 for ci, zi in zip(coords, z)) + 1e-30)
    return 0.5 * (1.0 + jnp.tanh((eps / 2 - dist) / delta))


def make_c_fn(particles, c0, eps, delta, s_fn):
    """
    Return a wave-speed callable c(*spatial_coords, t) for a metamaterial
    with the given particle centres.  Works for any spatial dimension.

    Parameters
    ----------
    particles : list of tuples
        Particle centre coordinates, e.g. [(0.5,)] in 1D or [(0.5, 0.5)] in 2D.
    c0    : float, background wave speed
    eps   : float, particle radius
    delta : float, interface width
    s_fn  : callable t -> scalar, modulation function s(t)
    """
    def c_fn(*args):
        *spatial, t_scalar = args
        coords = tuple(spatial)
        eta = (s_fn(t_scalar) / eps**2) * sum(
            _smooth_indicator(coords, z, eps, delta) for z in particles
        )
        n2 = jnp.maximum(1.0 + eta, 0.1)
        return c0 / jnp.sqrt(n2)
    return c_fn


# -----------------------------------------------------------------------------
# Finite Difference Solver
# -----------------------------------------------------------------------------
def fd_solve(x, t, dx, dt, c=1.0, y=None, dy=None, u0=None, v0=None, f_fn=None, dim=1):
    """
    Finite Difference solver for the wave equation.

    If c is a scalar  -> constant-coefficient solver (_fd_solve_{1,2}d_const).
    If c is callable  -> variable-coefficient solver (_fd_solve_{1,2}d_var).
                         c must have signature c(x_array, t_scalar) -> array.
                         u0 must be provided; v0 and f_fn are optional.
    """
    var = callable(c)

    if dim == 1:
        if var:
            if u0 is None:
                raise ValueError(
                    "u0 must be provided for variable-coefficient fd_solve"
                )
            return _fd_solve_1d_var(x, t, dx, dt, c, u0, v0, f_fn)
        else:
            return _fd_solve_1d_const(x, t, dx, dt, c)

    elif dim == 2:
        if y is None or dy is None:
            raise ValueError("y and dy must be provided for 2D")
        if var:
            if u0 is None:
                raise ValueError(
                    "u0 must be provided for variable-coefficient fd_solve"
                )
            return _fd_solve_2d_var(x, t, dx, dt, y, dy, c, u0, v0, f_fn)
        else:
            return _fd_solve_2d_const(x, t, dx, dt, y, dy, c)

    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")


def _fd_solve_1d_const(x, t, dx, dt, c):
    Nx = len(x) - 1
    Nt = len(t) - 1
    r = (c * dt / dx) ** 2

    u0 = jnp.sin(jnp.pi * x)
    u1 = jnp.zeros(Nx + 1).at[1:Nx].set(
        u0[1:Nx] + 0.5 * r * (u0[2:Nx + 1] - 2 * u0[1:Nx] + u0[0:Nx - 1])
    )

    def step(carry, _):
        u_prev, u_curr = carry
        u_next = jnp.zeros(Nx + 1).at[1:Nx].set(
            2 * u_curr[1:Nx]
            - u_prev[1:Nx]
            + r * (u_curr[2:Nx + 1] - 2 * u_curr[1:Nx] + u_curr[0:Nx - 1])
        )
        return (u_curr, u_next), u_next

    _, u_rest = jax.lax.scan(step, (u0, u1), None, length=Nt - 1)
    return jnp.concatenate([u0[None], u1[None], u_rest], axis=0)


def _fd_solve_2d_const(x, t, dx, dt, y, dy, c):
    Nx = len(x) - 1
    Ny = len(y) - 1
    Nt = len(t) - 1
    rx = (c * dt / dx) ** 2
    ry = (c * dt / dy) ** 2

    X, Y = jnp.meshgrid(x, y, indexing="ij")
    u0 = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    u1 = jnp.zeros((Nx + 1, Ny + 1)).at[1:Nx, 1:Ny].set(
        u0[1:Nx, 1:Ny]
        + 0.5 * rx * (u0[2:Nx + 1, 1:Ny] - 2 * u0[1:Nx, 1:Ny] + u0[0:Nx - 1, 1:Ny])
        + 0.5 * ry * (u0[1:Nx, 2:Ny + 1] - 2 * u0[1:Nx, 1:Ny] + u0[1:Nx, 0:Ny - 1])
    )

    def step(carry, _):
        u_prev, u_curr = carry
        u_next = jnp.zeros((Nx + 1, Ny + 1)).at[1:Nx, 1:Ny].set(
            2 * u_curr[1:Nx, 1:Ny]
            - u_prev[1:Nx, 1:Ny]
            + rx * (u_curr[2:Nx + 1, 1:Ny] - 2 * u_curr[1:Nx, 1:Ny] + u_curr[0:Nx - 1, 1:Ny])
            + ry * (u_curr[1:Nx, 2:Ny + 1] - 2 * u_curr[1:Nx, 1:Ny] + u_curr[1:Nx, 0:Ny - 1])
        )
        return (u_curr, u_next), u_next

    _, u_rest = jax.lax.scan(step, (u0, u1), None, length=Nt - 1)
    return jnp.concatenate([u0[None], u1[None], u_rest], axis=0)


def _fd_solve_1d_var(x, t, dx, dt, c_fn, u0, v0, f_fn):
    """
    Conservative FD stencil for variable c(x,t):
        (c^2_{i+1/2} (u_{i+1}-u_i) - c^2_{i-1/2} (u_i-u_{i-1})) / dx^2
    c evaluated at half-points x_{i+1/2} = x_i + dx/2.
    """
    Nx = len(x) - 1
    Nt = len(t) - 1
    x_half = x[:-1] + 0.5 * dx  # shape (Nx,)

    def _lap(u_vec, tn):
        c_r = c_fn(x_half[1:Nx], tn)
        c_l = c_fn(x_half[0:Nx - 1], tn)
        return (
            c_r**2 * (u_vec[2:Nx + 1] - u_vec[1:Nx])
            - c_l**2 * (u_vec[1:Nx] - u_vec[0:Nx - 1])
        ) / dx**2

    lap0 = _lap(u0, t[0])
    f0 = f_fn(x[1:Nx], t[0]) if f_fn is not None else jnp.zeros(Nx - 1)

    if v0 is None:
        u1 = jnp.zeros(Nx + 1).at[1:Nx].set(u0[1:Nx] + 0.5 * dt**2 * (lap0 + f0))
    else:
        u1 = jnp.zeros(Nx + 1).at[1:Nx].set(
            u0[1:Nx] + dt * v0[1:Nx] + 0.5 * dt**2 * (lap0 + f0)
        )

    def step(carry, tn):
        u_prev, u_curr = carry
        lap = _lap(u_curr, tn)
        fn = f_fn(x[1:Nx], tn) if f_fn is not None else jnp.zeros(Nx - 1)
        u_next = jnp.zeros(Nx + 1).at[1:Nx].set(
            2 * u_curr[1:Nx] - u_prev[1:Nx] + dt**2 * (lap + fn)
        )
        return (u_curr, u_next), u_next

    _, u_rest = jax.lax.scan(step, (u0, u1), t[1:Nt])
    return jnp.concatenate([u0[None], u1[None], u_rest], axis=0)


def _fd_solve_2d_var(x, t, dx, dt, y, dy, c_fn, u0, v0, f_fn):
    """
    Conservative FD stencil for 2D variable c(x,y,t):
        c^2 evaluated at face midpoints (x_{i±1/2}, y_j) and (x_i, y_{j±1/2}).
    """
    Nx = len(x) - 1
    Ny = len(y) - 1
    Nt = len(t) - 1

    x_half = x[:-1] + 0.5 * dx  # shape (Nx,)
    y_half = y[:-1] + 0.5 * dy  # shape (Ny,)

    # Precompute face-midpoint grids for interior nodes (shape (Nx-1, Ny-1) each)
    Xr, Yr = jnp.meshgrid(x_half[1:Nx], y[1:Ny], indexing="ij")
    Xl, Yl = jnp.meshgrid(x_half[0:Nx - 1], y[1:Ny], indexing="ij")
    Xu, Yu = jnp.meshgrid(x[1:Nx], y_half[1:Ny], indexing="ij")
    Xd, Yd = jnp.meshgrid(x[1:Nx], y_half[0:Ny - 1], indexing="ij")
    if f_fn is not None:
        Xn, Yn = jnp.meshgrid(x[1:Nx], y[1:Ny], indexing="ij")

    def _lap(u_slice, tn):
        c2_r = c_fn(Xr, Yr, tn) ** 2
        c2_l = c_fn(Xl, Yl, tn) ** 2
        c2_u = c_fn(Xu, Yu, tn) ** 2
        c2_d = c_fn(Xd, Yd, tn) ** 2
        return (
            c2_r * (u_slice[2:Nx + 1, 1:Ny] - u_slice[1:Nx, 1:Ny])
            - c2_l * (u_slice[1:Nx, 1:Ny] - u_slice[0:Nx - 1, 1:Ny])
        ) / dx**2 + (
            c2_u * (u_slice[1:Nx, 2:Ny + 1] - u_slice[1:Nx, 1:Ny])
            - c2_d * (u_slice[1:Nx, 1:Ny] - u_slice[1:Nx, 0:Ny - 1])
        ) / dy**2

    lap0 = _lap(u0, t[0])
    f0 = f_fn(Xn, Yn, t[0]) if f_fn is not None else jnp.zeros((Nx - 1, Ny - 1))

    if v0 is None:
        u1 = jnp.zeros((Nx + 1, Ny + 1)).at[1:Nx, 1:Ny].set(
            u0[1:Nx, 1:Ny] + 0.5 * dt**2 * (lap0 + f0)
        )
    else:
        u1 = jnp.zeros((Nx + 1, Ny + 1)).at[1:Nx, 1:Ny].set(
            u0[1:Nx, 1:Ny] + dt * v0[1:Nx, 1:Ny] + 0.5 * dt**2 * (lap0 + f0)
        )

    def step(carry, tn):
        u_prev, u_curr = carry
        lap = _lap(u_curr, tn)
        fn = f_fn(Xn, Yn, tn) if f_fn is not None else jnp.zeros((Nx - 1, Ny - 1))
        u_next = jnp.zeros((Nx + 1, Ny + 1)).at[1:Nx, 1:Ny].set(
            2 * u_curr[1:Nx, 1:Ny] - u_prev[1:Nx, 1:Ny] + dt**2 * (lap + fn)
        )
        return (u_curr, u_next), u_next

    _, u_rest = jax.lax.scan(step, (u0, u1), t[1:Nt])
    return jnp.concatenate([u0[None], u1[None], u_rest], axis=0)


# -----------------------------------------------------------------------------
# Finite Element Solver
# -----------------------------------------------------------------------------
def fem_solve(x, t, dx, dt, c=1.0, y=None, dy=None, u0=None, v0=None, f_fn=None, dim=1):
    """
    FEM solver (linear elements, lumped mass) for the wave equation.

    If c is a scalar  -> constant-coefficient solver (_fem_solve_{1,2}d_const).
    If c is callable  -> variable-coefficient solver (_fem_solve_{1,2}d_var).
                         c must have signature c(x_array, t_scalar) -> array.
                         u0 must be provided; v0 and f_fn are optional.
    """
    var = callable(c)

    if dim == 1:
        if var:
            if u0 is None:
                raise ValueError(
                    "u0 must be provided for variable-coefficient fem_solve"
                )
            return _fem_solve_1d_var(x, t, dx, dt, c, u0, v0, f_fn)
        else:
            return _fem_solve_1d_const(x, t, dx, dt, c)

    elif dim == 2:
        if y is None or dy is None:
            raise ValueError("y and dy must be provided for 2D")
        if var:
            if u0 is None:
                raise ValueError(
                    "u0 must be provided for variable-coefficient fem_solve"
                )
            return _fem_solve_2d_var(x, t, dx, dt, y, dy, c, u0, v0, f_fn)
        else:
            return _fem_solve_2d_const(x, t, dx, dt, y, dy, c, u0)

    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")


def _fem_solve_1d_const(x, t, dx, dt, c):
    Nx = len(x) - 1
    Nt = len(t) - 1

    K_diag = jnp.ones(Nx + 1) * (2.0 / dx)
    K_off = jnp.ones(Nx) * (-1.0 / dx)
    K_diag = K_diag.at[0].set(1.0)
    K_diag = K_diag.at[Nx].set(1.0)
    K_off = K_off.at[0].set(0.0)
    K_off = K_off.at[-1].set(0.0)

    M_lumped = jnp.ones(Nx + 1) * dx
    M_lumped = M_lumped.at[0].set(1.0)
    M_lumped = M_lumped.at[Nx].set(1.0)

    coeff = (c * dt) ** 2 / M_lumped

    u0 = jnp.sin(jnp.pi * x)
    st0 = (
        K_diag[1:Nx] * u0[1:Nx]
        + K_off[0:Nx - 1] * u0[0:Nx - 1]
        + K_off[1:Nx] * u0[2:Nx + 1]
    )
    u1 = jnp.zeros(Nx + 1).at[1:Nx].set(u0[1:Nx] - 0.5 * coeff[1:Nx] * st0)

    def step(carry, _):
        u_prev, u_curr = carry
        st = (
            K_diag[1:Nx] * u_curr[1:Nx]
            + K_off[0:Nx - 1] * u_curr[0:Nx - 1]
            + K_off[1:Nx] * u_curr[2:Nx + 1]
        )
        u_next = jnp.zeros(Nx + 1).at[1:Nx].set(
            2 * u_curr[1:Nx] - u_prev[1:Nx] - coeff[1:Nx] * st
        )
        return (u_curr, u_next), u_next

    _, u_rest = jax.lax.scan(step, (u0, u1), None, length=Nt - 1)
    return jnp.concatenate([u0[None], u1[None], u_rest], axis=0)


def _fem_solve_2d_const(
    x,
    t,
    dx,
    dt,
    y,
    dy,
    c,
    u0,
    v0=None,
    f_fn=None,
):
    """
    Explicit mass-lumped FEM solver for 2D wave equation
        u_tt = c^2 Δu + f

    Parameters
    ----------
    x, y : spatial grids
    t    : time grid
    c    : constant wave speed
    u0   : initial displacement (Nx+1, Ny+1)
    v0   : initial velocity (optional)
    f_fn : forcing function f(x,y,t)
    """

    Nx = len(x) - 1
    Ny = len(y) - 1
    Nt = len(t) - 1
    if u0 is None:
        X, Y = jnp.meshgrid(x, y, indexing="ij")
        u0 = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)

    def stiffness(u_slice):
        return (
            u_slice[2:Nx + 1, 1:Ny]
            - 2 * u_slice[1:Nx, 1:Ny]
            + u_slice[0:Nx - 1, 1:Ny]
        ) / dx**2 + (
            u_slice[1:Nx, 2:Ny + 1]
            - 2 * u_slice[1:Nx, 1:Ny]
            + u_slice[1:Nx, 0:Ny - 1]
        ) / dy**2

    if f_fn is not None:
        X, Y = jnp.meshgrid(x[1:Nx], y[1:Ny], indexing="ij")

    rhs0 = c**2 * stiffness(u0)
    if f_fn is not None:
        rhs0 = rhs0 + f_fn(X, Y, t[0])

    if v0 is None:
        u1 = jnp.zeros((Nx + 1, Ny + 1)).at[1:Nx, 1:Ny].set(
            u0[1:Nx, 1:Ny] + 0.5 * dt**2 * rhs0
        )
    else:
        u1 = jnp.zeros((Nx + 1, Ny + 1)).at[1:Nx, 1:Ny].set(
            u0[1:Nx, 1:Ny] + dt * v0[1:Nx, 1:Ny] + 0.5 * dt**2 * rhs0
        )

    def step(carry, tn):
        u_prev, u_curr = carry
        rhs = c**2 * stiffness(u_curr)
        if f_fn is not None:
            rhs = rhs + f_fn(X, Y, tn)
        u_next = jnp.zeros((Nx + 1, Ny + 1)).at[1:Nx, 1:Ny].set(
            2 * u_curr[1:Nx, 1:Ny] - u_prev[1:Nx, 1:Ny] + dt**2 * rhs
        )
        return (u_curr, u_next), u_next

    _, u_rest = jax.lax.scan(step, (u0, u1), t[1:Nt])
    return jnp.concatenate([u0[None], u1[None], u_rest], axis=0)


def _fem_solve_1d_var(x, t, dx, dt, c_fn, u0, v0, f_fn):
    """
    Lumped-mass FEM with c at element midpoints.
    Note: algebraically identical to _fd_solve_1d_var for uniform grids.
    """
    Nx = len(x) - 1
    Nt = len(t) - 1
    x_mid = x[:-1] + 0.5 * dx  # shape (Nx,)

    def _stiffness_rhs(u_vec, tn):
        c2 = c_fn(x_mid, tn) ** 2
        Ku = (
            c2[1:Nx] * (u_vec[1:Nx] - u_vec[2:Nx + 1])
            + c2[0:Nx - 1] * (u_vec[1:Nx] - u_vec[0:Nx - 1])
        ) / dx
        return -Ku / dx  # divide by lumped mass M_i = dx

    rhs0 = _stiffness_rhs(u0, t[0])
    f0 = f_fn(x[1:Nx], t[0]) if f_fn is not None else jnp.zeros(Nx - 1)

    if v0 is None:
        u1 = jnp.zeros(Nx + 1).at[1:Nx].set(u0[1:Nx] + 0.5 * dt**2 * (rhs0 + f0))
    else:
        u1 = jnp.zeros(Nx + 1).at[1:Nx].set(
            u0[1:Nx] + dt * v0[1:Nx] + 0.5 * dt**2 * (rhs0 + f0)
        )

    def step(carry, tn):
        u_prev, u_curr = carry
        rhs = _stiffness_rhs(u_curr, tn)
        fn = f_fn(x[1:Nx], tn) if f_fn is not None else jnp.zeros(Nx - 1)
        u_next = jnp.zeros(Nx + 1).at[1:Nx].set(
            2 * u_curr[1:Nx] - u_prev[1:Nx] + dt**2 * (rhs + fn)
        )
        return (u_curr, u_next), u_next

    _, u_rest = jax.lax.scan(step, (u0, u1), t[1:Nt])
    return jnp.concatenate([u0[None], u1[None], u_rest], axis=0)


def _fem_solve_2d_var(x, t, dx, dt, y, dy, c_fn, u0, v0, f_fn):
    """
    Explicit mass-lumped FEM solver for

        u_tt = div(c(x,y,t)^2 grad u) + f

    using Q1 elements on a structured grid.
    c^2 evaluated at face midpoints (equivalent to conservative FD with lumped mass).
    """
    Nx = len(x) - 1
    Ny = len(y) - 1
    Nt = len(t) - 1

    x_half = x[:-1] + 0.5 * dx  # shape (Nx,)
    y_half = y[:-1] + 0.5 * dy  # shape (Ny,)

    # Precompute face-midpoint grids for interior nodes (shape (Nx-1, Ny-1) each)
    Xr, Yr = jnp.meshgrid(x_half[1:Nx], y[1:Ny], indexing="ij")
    Xl, Yl = jnp.meshgrid(x_half[0:Nx - 1], y[1:Ny], indexing="ij")
    Xu, Yu = jnp.meshgrid(x[1:Nx], y_half[1:Ny], indexing="ij")
    Xd, Yd = jnp.meshgrid(x[1:Nx], y_half[0:Ny - 1], indexing="ij")
    if f_fn is not None:
        Xn, Yn = jnp.meshgrid(x[1:Nx], y[1:Ny], indexing="ij")

    def stiffness_rhs(u_slice, tn):
        c2_r = c_fn(Xr, Yr, tn) ** 2
        c2_l = c_fn(Xl, Yl, tn) ** 2
        c2_u = c_fn(Xu, Yu, tn) ** 2
        c2_d = c_fn(Xd, Yd, tn) ** 2
        return (
            c2_r * (u_slice[2:Nx + 1, 1:Ny] - u_slice[1:Nx, 1:Ny])
            - c2_l * (u_slice[1:Nx, 1:Ny] - u_slice[0:Nx - 1, 1:Ny])
        ) / dx**2 + (
            c2_u * (u_slice[1:Nx, 2:Ny + 1] - u_slice[1:Nx, 1:Ny])
            - c2_d * (u_slice[1:Nx, 1:Ny] - u_slice[1:Nx, 0:Ny - 1])
        ) / dy**2

    rhs0 = stiffness_rhs(u0, t[0])
    f0 = f_fn(Xn, Yn, t[0]) if f_fn is not None else jnp.zeros((Nx - 1, Ny - 1))

    if v0 is None:
        u1 = jnp.zeros((Nx + 1, Ny + 1)).at[1:Nx, 1:Ny].set(
            u0[1:Nx, 1:Ny] + 0.5 * dt**2 * (rhs0 + f0)
        )
    else:
        u1 = jnp.zeros((Nx + 1, Ny + 1)).at[1:Nx, 1:Ny].set(
            u0[1:Nx, 1:Ny] + dt * v0[1:Nx, 1:Ny] + 0.5 * dt**2 * (rhs0 + f0)
        )

    def step(carry, tn):
        u_prev, u_curr = carry
        rhs = stiffness_rhs(u_curr, tn)
        fn = f_fn(Xn, Yn, tn) if f_fn is not None else jnp.zeros((Nx - 1, Ny - 1))
        u_next = jnp.zeros((Nx + 1, Ny + 1)).at[1:Nx, 1:Ny].set(
            2 * u_curr[1:Nx, 1:Ny] - u_prev[1:Nx, 1:Ny] + dt**2 * (rhs + fn)
        )
        return (u_curr, u_next), u_next

    _, u_rest = jax.lax.scan(step, (u0, u1), t[1:Nt])
    return jnp.concatenate([u0[None], u1[None], u_rest], axis=0)
