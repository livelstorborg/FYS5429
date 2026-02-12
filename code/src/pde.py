import jax
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Analytical Solutions
# -----------------------------------------------------------------------------
def u_exact(x, t, y=None, c=1.0, dim=1):
    """
    Analytical solution for wave equation in 1D or 2D.
    """
    if dim == 1:
        # Create meshgrid for proper broadcasting
        T, X = jnp.meshgrid(t, x, indexing='ij')  # Both shape (Nt, Nx)
        u = jnp.sin(jnp.pi * X) * jnp.cos(jnp.pi * c * T)
        
    elif dim == 2:
        if y is None:
            raise ValueError("y must be provided for 2D")
        # Spatial part
        X, Y = jnp.meshgrid(x, y, indexing='ij')  # Both (Nx, Ny)
        spatial = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
        
        # Temporal part
        temporal = jnp.cos(jnp.sqrt(2) * jnp.pi * c * t)  # (Nt,)
        
        # Combine: (Nt,) × (Nx, Ny) -> (Nt, Nx, Ny)
        u = temporal[:, None, None] * spatial[None, :, :]
        
    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")
    
    return u


# -----------------------------------------------------------------------------
# Grid Creation for FD and FE Schemes
# -----------------------------------------------------------------------------
def create_grid(Nx, Ny=None, T=1.0, c=1.0, cfl=0.5, dim=1):
    """
    Create spatial + time grid for wave equation in 1D or 2D.
    """
    if dim == 1:
        # 1D: CFL condition c*dt/dx ≤ 1
        dx = 1.0 / Nx
        dt = cfl * dx / c
        Nt = int(T / dt)
        
        x = jnp.linspace(0, 1, Nx + 1)
        t = jnp.linspace(0, T, Nt + 1)
        
        return x, t, dx, dt
        
    elif dim == 2:
        # 2D: CFL condition c*dt*sqrt(1/dx² + 1/dy²) ≤ 1
        if Ny is None:
            raise ValueError("Ny must be provided for 2D")
            
        dx = 1.0 / Nx
        dy = 1.0 / Ny
        dt = cfl * min(dx, dy) / (c * jnp.sqrt(2))
        Nt = int(T / dt)
        
        x = jnp.linspace(0, 1, Nx + 1)
        y = jnp.linspace(0, 1, Ny + 1)
        t = jnp.linspace(0, T, Nt + 1)
        
        return x, y, t, dx, dy, dt
        
    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")


# -----------------------------------------------------------------------------
# Finite Difference Solver 
# -----------------------------------------------------------------------------
def fd_solve(x, t, dx, dt, y=None, dy=None, c=1.0, dim=1):
    """
    Finite Difference solver for wave equation.
    """
    if dim == 1:
        return _fd_solve_1d(x, t, dx, dt, c)
    elif dim == 2:
        if y is None or dy is None:
            raise ValueError("y and dy must be provided for 2D")
        return _fd_solve_2d(x, t, dx, dt, y, dy, c)
    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")


def _fd_solve_1d(x, t, dx, dt, c):
    """
    1D FD solver.
    """
    Nx = len(x) - 1
    Nt = len(t) - 1
    r = (c * dt / dx) ** 2
    
    u = jnp.zeros((Nt + 1, Nx + 1))
    u = u.at[0, :].set(jnp.sin(jnp.pi * x))
    
    u = u.at[1, 1:Nx].set(
        u[0, 1:Nx] + 0.5 * r * (u[0, 2:Nx+1] - 2*u[0, 1:Nx] + u[0, 0:Nx-1])
    )
    u = u.at[1, 0].set(0.0)
    u = u.at[1, Nx].set(0.0)
    
    for n in range(1, Nt):
        u = u.at[n+1, 1:Nx].set(
            2*u[n, 1:Nx] - u[n-1, 1:Nx] + 
            r * (u[n, 2:Nx+1] - 2*u[n, 1:Nx] + u[n, 0:Nx-1])
        )
        u = u.at[n+1, 0].set(0.0)
        u = u.at[n+1, Nx].set(0.0)
    
    return u


def _fd_solve_2d(x, t, dx, dt, y, dy, c):
    """
    2D FD solver.
    """
    Nx = len(x) - 1
    Ny = len(y) - 1
    Nt = len(t) - 1
    
    rx = (c * dt / dx) ** 2
    ry = (c * dt / dy) ** 2
    
    u = jnp.zeros((Nt + 1, Nx + 1, Ny + 1))
    
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    u = u.at[0, :, :].set(jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y))
    
    u = u.at[1, 1:Nx, 1:Ny].set(
        u[0, 1:Nx, 1:Ny] + 
        0.5 * rx * (u[0, 2:Nx+1, 1:Ny] - 2*u[0, 1:Nx, 1:Ny] + u[0, 0:Nx-1, 1:Ny]) +
        0.5 * ry * (u[0, 1:Nx, 2:Ny+1] - 2*u[0, 1:Nx, 1:Ny] + u[0, 1:Nx, 0:Ny-1])
    )
    
    for n in range(1, Nt):
        u = u.at[n+1, 1:Nx, 1:Ny].set(
            2 * u[n, 1:Nx, 1:Ny] - u[n-1, 1:Nx, 1:Ny] +
            rx * (u[n, 2:Nx+1, 1:Ny] - 2*u[n, 1:Nx, 1:Ny] + u[n, 0:Nx-1, 1:Ny]) +
            ry * (u[n, 1:Nx, 2:Ny+1] - 2*u[n, 1:Nx, 1:Ny] + u[n, 1:Nx, 0:Ny-1])
        )
    
    return u





# -----------------------------------------------------------------------------
# Finite Element Solver 
# -----------------------------------------------------------------------------
def fem_solve(x, t, dx, dt, y=None, dy=None, c=1.0, dim=1):
    """
    Finite Element Method solver for wave equation using linear elements.
    """
    if dim == 1:
        return _fem_solve_1d(x, t, dx, dt, c)
    elif dim == 2:
        if y is None or dy is None:
            raise ValueError("y and dy must be provided for 2D")
        return _fem_solve_2d(x, t, dx, dt, y, dy, c)
    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")


def _fem_solve_1d(x, t, dx, dt, c):
    """
    1D FEM solver using linear elements and mass lumping.
    """
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
    
    r = (c * dt) ** 2
    coeff = r / M_lumped
    
    u = jnp.zeros((Nt + 1, Nx + 1))
    u = u.at[0, :].set(jnp.sin(jnp.pi * x))
    
    u_1 = jnp.zeros(Nx + 1)
    for i in range(1, Nx):
        stiffness_term = K_diag[i] * u[0, i] + K_off[i-1] * u[0, i-1] + K_off[i] * u[0, i+1]
        u_1 = u_1.at[i].set(u[0, i] - 0.5 * coeff[i] * stiffness_term)
    
    u = u.at[1, :].set(u_1)
    u = u.at[1, 0].set(0.0)
    u = u.at[1, Nx].set(0.0)
    
    for n in range(1, Nt):
        u_new = jnp.zeros(Nx + 1)
        
        for i in range(1, Nx):
            stiffness_term = (K_diag[i] * u[n, i] + 
                            K_off[i-1] * u[n, i-1] + 
                            K_off[i] * u[n, i+1])
            
            u_new = u_new.at[i].set(
                2 * u[n, i] - u[n-1, i] - coeff[i] * stiffness_term
            )
        
        u = u.at[n+1, :].set(u_new)
        u = u.at[n+1, 0].set(0.0)
        u = u.at[n+1, Nx].set(0.0)
    
    return u


def _fem_solve_2d(x, t, dx, dt, y, dy, c):
    """
    2D FEM solver using bilinear elements and mass lumping.
    """
    Nx = len(x) - 1
    Ny = len(y) - 1
    Nt = len(t) - 1
    
    N_dof = (Nx + 1) * (Ny + 1)
    M_lumped = jnp.ones(N_dof) * (dx * dy)
    
    for i in range(Nx + 1):
        M_lumped = M_lumped.at[i * (Ny + 1)].set(1.0)
        M_lumped = M_lumped.at[i * (Ny + 1) + Ny].set(1.0)
    for j in range(Ny + 1):
        M_lumped = M_lumped.at[j].set(1.0)
        M_lumped = M_lumped.at[Nx * (Ny + 1) + j].set(1.0)
    
    kx = 1.0 / dx
    ky = 1.0 / dy
    r = (c * dt) ** 2
    
    u = jnp.zeros((Nt + 1, Nx + 1, Ny + 1))
    
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    u = u.at[0, :, :].set(jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y))
    
    u_1 = jnp.zeros((Nx + 1, Ny + 1))
    for i in range(1, Nx):
        for j in range(1, Ny):
            laplacian = (
                kx * (u[0, i+1, j] - 2*u[0, i, j] + u[0, i-1, j]) +
                ky * (u[0, i, j+1] - 2*u[0, i, j] + u[0, i, j-1])
            )
            
            idx = i * (Ny + 1) + j
            coeff = r / M_lumped[idx]
            u_1 = u_1.at[i, j].set(u[0, i, j] - 0.5 * coeff * laplacian)
    
    u = u.at[1, :, :].set(u_1)
    
    for n in range(1, Nt):
        u_new = jnp.zeros((Nx + 1, Ny + 1))
        
        for i in range(1, Nx):
            for j in range(1, Ny):
                laplacian = (
                    kx * (u[n, i+1, j] - 2*u[n, i, j] + u[n, i-1, j]) +
                    ky * (u[n, i, j+1] - 2*u[n, i, j] + u[n, i, j-1])
                )
                
                idx = i * (Ny + 1) + j
                coeff = r / M_lumped[idx]
                
                u_new = u_new.at[i, j].set(
                    2 * u[n, i, j] - u[n-1, i, j] - coeff * laplacian
                )
        
        u = u.at[n+1, :, :].set(u_new)
    
    return u



