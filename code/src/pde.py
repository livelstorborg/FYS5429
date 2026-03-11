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
        x  = jnp.linspace(0, 1, Nx + 1)
        t  = jnp.arange(0, Nt + 1) * dt
        return x, t, dx, dt

    elif dim == 2:
        if Ny is None:
            raise ValueError("Ny must be provided for 2D")
        dx = 1.0 / Nx
        dy = 1.0 / Ny
        dt = cfl * min(dx, dy) / (c * jnp.sqrt(2))
        Nt = int(T / dt)
        x  = jnp.linspace(0, 1, Nx + 1)
        y  = jnp.linspace(0, 1, Ny + 1)
        t  = jnp.arange(0, Nt + 1) * dt
        return x, y, t, dx, dy, dt

    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")


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
                raise ValueError("u0 must be provided for variable-coefficient fd_solve")
            return _fd_solve_1d_var(x, t, dx, dt, c, u0, v0, f_fn)
        else:
            return _fd_solve_1d_const(x, t, dx, dt, c)

    elif dim == 2:
        if y is None or dy is None:
            raise ValueError("y and dy must be provided for 2D")
        if var:
            if u0 is None:
                raise ValueError("u0 must be provided for variable-coefficient fd_solve")
            return _fd_solve_2d_var(x, t, dx, dt, y, dy, c, u0, v0, f_fn)
        else:
            return _fd_solve_2d_const(x, t, dx, dt, y, dy, c)

    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")


def _fd_solve_1d_const(x, t, dx, dt, c):
    Nx = len(x) - 1
    Nt = len(t) - 1
    r  = (c * dt / dx) ** 2

    u = jnp.zeros((Nt + 1, Nx + 1))
    u = u.at[0, :].set(jnp.sin(jnp.pi * x))
    u = u.at[1, 1:Nx].set(
        u[0, 1:Nx] + 0.5 * r * (u[0, 2:Nx+1] - 2 * u[0, 1:Nx] + u[0, 0:Nx-1])
    )

    for n in range(1, Nt):
        u = u.at[n + 1, 1:Nx].set(
            2 * u[n, 1:Nx] - u[n-1, 1:Nx]
            + r * (u[n, 2:Nx+1] - 2 * u[n, 1:Nx] + u[n, 0:Nx-1])
        )
    return u


def _fd_solve_2d_const(x, t, dx, dt, y, dy, c):
    Nx = len(x) - 1
    Ny = len(y) - 1
    Nt = len(t) - 1
    rx = (c * dt / dx) ** 2
    ry = (c * dt / dy) ** 2

    X, Y = jnp.meshgrid(x, y, indexing="ij")
    u    = jnp.zeros((Nt + 1, Nx + 1, Ny + 1))
    u    = u.at[0, :, :].set(jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y))
    u    = u.at[1, 1:Nx, 1:Ny].set(
        u[0, 1:Nx, 1:Ny]
        + 0.5 * rx * (u[0, 2:Nx+1, 1:Ny] - 2 * u[0, 1:Nx, 1:Ny] + u[0, 0:Nx-1, 1:Ny])
        + 0.5 * ry * (u[0, 1:Nx, 2:Ny+1] - 2 * u[0, 1:Nx, 1:Ny] + u[0, 1:Nx, 0:Ny-1])
    )

    for n in range(1, Nt):
        u = u.at[n + 1, 1:Nx, 1:Ny].set(
            2 * u[n, 1:Nx, 1:Ny] - u[n-1, 1:Nx, 1:Ny]
            + rx * (u[n, 2:Nx+1, 1:Ny] - 2 * u[n, 1:Nx, 1:Ny] + u[n, 0:Nx-1, 1:Ny])
            + ry * (u[n, 1:Nx, 2:Ny+1] - 2 * u[n, 1:Nx, 1:Ny] + u[n, 1:Nx, 0:Ny-1])
        )
    return u


def _fd_solve_1d_var(x, t, dx, dt, c_fn, u0, v0, f_fn):
    """
    Conservative FD stencil for variable c(x,t):
        (c^2_{i+1/2} (u_{i+1}-u_i) - c^2_{i-1/2} (u_i-u_{i-1})) / dx^2
    c evaluated at half-points x_{i+1/2} = x_i + dx/2.
    """
    Nx     = len(x) - 1
    Nt     = len(t) - 1
    x_half = x[:-1] + 0.5 * dx  # shape (Nx,)

    u = jnp.zeros((Nt + 1, Nx + 1))
    u = u.at[0].set(u0)

    def _lap(u_vec, tn):
        c_r = c_fn(x_half[1:Nx],   tn)
        c_l = c_fn(x_half[0:Nx-1], tn)
        return (c_r**2 * (u_vec[2:Nx+1] - u_vec[1:Nx])
                - c_l**2 * (u_vec[1:Nx] - u_vec[0:Nx-1])) / dx**2

    lap0 = _lap(u[0], t[0])
    f0   = f_fn(x[1:Nx], t[0]) if f_fn is not None else jnp.zeros(Nx - 1)

    if v0 is None:
        u1 = u[0, 1:Nx] + 0.5 * dt**2 * (lap0 + f0)
    else:
        u1 = u[0, 1:Nx] + dt * v0[1:Nx] + 0.5 * dt**2 * (lap0 + f0)
    u = u.at[1, 1:Nx].set(u1)

    for n in range(1, Nt):
        lap = _lap(u[n], t[n])
        fn  = f_fn(x[1:Nx], t[n]) if f_fn is not None else jnp.zeros(Nx - 1)
        u   = u.at[n + 1, 1:Nx].set(
            2 * u[n, 1:Nx] - u[n-1, 1:Nx] + dt**2 * (lap + fn)
        )
    return u


def _fd_solve_2d_var(x, t, dx, dt, y, dy, c_fn, u0, v0, f_fn):
    raise NotImplementedError("2D variable-coefficient FD solver not yet implemented")


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
                raise ValueError("u0 must be provided for variable-coefficient fem_solve")
            return _fem_solve_1d_var(x, t, dx, dt, c, u0, v0, f_fn)
        else:
            return _fem_solve_1d_const(x, t, dx, dt, c)

    elif dim == 2:
        if y is None or dy is None:
            raise ValueError("y and dy must be provided for 2D")
        if var:
            if u0 is None:
                raise ValueError("u0 must be provided for variable-coefficient fem_solve")
            return _fem_solve_2d_var(x, t, dx, dt, y, dy, c, u0, v0, f_fn)
        else:
            return _fem_solve_2d_const(x, t, dx, dt, y, dy, c)

    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")


def _fem_solve_1d_const(x, t, dx, dt, c):
    Nx = len(x) - 1
    Nt = len(t) - 1

    K_diag = jnp.ones(Nx + 1) * (2.0 / dx)
    K_off  = jnp.ones(Nx) * (-1.0 / dx)
    K_diag = K_diag.at[0].set(1.0)
    K_diag = K_diag.at[Nx].set(1.0)
    K_off  = K_off.at[0].set(0.0)
    K_off  = K_off.at[-1].set(0.0)

    M_lumped = jnp.ones(Nx + 1) * dx
    M_lumped = M_lumped.at[0].set(1.0)
    M_lumped = M_lumped.at[Nx].set(1.0)

    coeff = (c * dt) ** 2 / M_lumped

    u   = jnp.zeros((Nt + 1, Nx + 1))
    u   = u.at[0, :].set(jnp.sin(jnp.pi * x))
    u_1 = jnp.zeros(Nx + 1)
    for i in range(1, Nx):
        st  = K_diag[i] * u[0, i] + K_off[i-1] * u[0, i-1] + K_off[i] * u[0, i+1]
        u_1 = u_1.at[i].set(u[0, i] - 0.5 * coeff[i] * st)
    u = u.at[1, :].set(u_1)
    u = u.at[1, 0].set(0.0)
    u = u.at[1, Nx].set(0.0)

    for n in range(1, Nt):
        u_new = jnp.zeros(Nx + 1)
        for i in range(1, Nx):
            st    = K_diag[i] * u[n, i] + K_off[i-1] * u[n, i-1] + K_off[i] * u[n, i+1]
            u_new = u_new.at[i].set(2 * u[n, i] - u[n-1, i] - coeff[i] * st)
        u = u.at[n + 1, :].set(u_new)
        u = u.at[n + 1, 0].set(0.0)
        u = u.at[n + 1, Nx].set(0.0)
    return u


def _fem_solve_2d_const(x, t, dx, dt, y, dy, c):
    Nx = len(x) - 1
    Ny = len(y) - 1
    Nt = len(t) - 1
    rx = (c * dt / dx) ** 2
    ry = (c * dt / dy) ** 2

    X, Y = jnp.meshgrid(x, y, indexing="ij")
    u0_  = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    u    = jnp.zeros((Nt + 1, Nx + 1, Ny + 1))
    u    = u.at[0].set(u0_)

    lap0 = (
        (u0_[2:Nx+1, 1:Ny] - 2 * u0_[1:Nx, 1:Ny] + u0_[0:Nx-1, 1:Ny]) / dx**2
      + (u0_[1:Nx, 2:Ny+1] - 2 * u0_[1:Nx, 1:Ny] + u0_[1:Nx, 0:Ny-1]) / dy**2
    )
    u1 = jnp.zeros((Nx + 1, Ny + 1))
    u1 = u1.at[1:Nx, 1:Ny].set(u0_[1:Nx, 1:Ny] + 0.5 * (c * dt)**2 * lap0)
    u  = u.at[1].set(u1)

    for n in range(1, Nt):
        lap_n = (
            (u[n, 2:Nx+1, 1:Ny] - 2 * u[n, 1:Nx, 1:Ny] + u[n, 0:Nx-1, 1:Ny]) / dx**2
          + (u[n, 1:Nx, 2:Ny+1] - 2 * u[n, 1:Nx, 1:Ny] + u[n, 1:Nx, 0:Ny-1]) / dy**2
        )
        u_new = jnp.zeros((Nx + 1, Ny + 1))
        u_new = u_new.at[1:Nx, 1:Ny].set(
            2 * u[n, 1:Nx, 1:Ny] - u[n-1, 1:Nx, 1:Ny] + (c * dt)**2 * lap_n
        )
        u = u.at[n + 1].set(u_new)
    return u


def _fem_solve_1d_var(x, t, dx, dt, c_fn, u0, v0, f_fn):
    """
    Lumped-mass FEM with c at element midpoints.
    Note: algebraically identical to _fd_solve_1d_var for uniform grids.
    """
    Nx    = len(x) - 1
    Nt    = len(t) - 1
    x_mid = x[:-1] + 0.5 * dx  # shape (Nx,)

    u = jnp.zeros((Nt + 1, Nx + 1))
    u = u.at[0].set(u0)

    def _stiffness_rhs(u_vec, tn):
        c2 = c_fn(x_mid, tn) ** 2
        Ku = (c2[1:Nx]   * (u_vec[1:Nx] - u_vec[2:Nx+1])
              + c2[0:Nx-1] * (u_vec[1:Nx] - u_vec[0:Nx-1])) / dx
        return -Ku / dx  # divide by lumped mass M_i = dx

    rhs0 = _stiffness_rhs(u[0], t[0])
    f0   = f_fn(x[1:Nx], t[0]) if f_fn is not None else jnp.zeros(Nx - 1)

    if v0 is None:
        u1 = u[0, 1:Nx] + 0.5 * dt**2 * (rhs0 + f0)
    else:
        u1 = u[0, 1:Nx] + dt * v0[1:Nx] + 0.5 * dt**2 * (rhs0 + f0)
    u = u.at[1, 1:Nx].set(u1)

    for n in range(1, Nt):
        rhs = _stiffness_rhs(u[n], t[n])
        fn  = f_fn(x[1:Nx], t[n]) if f_fn is not None else jnp.zeros(Nx - 1)
        u   = u.at[n + 1, 1:Nx].set(
            2 * u[n, 1:Nx] - u[n-1, 1:Nx] + dt**2 * (rhs + fn)
        )
    return u


def _fem_solve_2d_var(x, t, dx, dt, y, dy, c_fn, u0, v0, f_fn):
    raise NotImplementedError("2D variable-coefficient FEM solver not yet implemented")
