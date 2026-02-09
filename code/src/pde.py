import jax
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Analytical Solutions
# -----------------------------------------------------------------------------
def u_exact(x, t, y=None, c=1.0, dim=1):
    """
    Analytical solution for wave equation in 1D or 2D.
    
    Parameters:
    -----------
    x : array (Nx,)
        Spatial coordinates in x
    t : array (Nt,)
        Time coordinates
    y : array (Ny,), optional
        Spatial coordinates in y (required for dim=2)
    c : float
        Wave speed
    dim : int (1 or 2)
        Spatial dimension
    
    Returns:
    --------
    u : array
        Shape: (Nt, Nx) for 1D, (Nt, Nx, Ny) for 2D
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
# Grid Creation
# -----------------------------------------------------------------------------
def create_grid(Nx, Ny=None, T=1.0, c=1.0, cfl=0.5, dim=1):
    """
    Create spatial + time grid for wave equation in 1D or 2D.
    
    Parameters:
    -----------
    Nx : int
        Number of spatial intervals in x
    Ny : int, optional
        Number of spatial intervals in y (required for dim=2)
    T : float
        Final time
    c : float
        Wave speed
    cfl : float
        CFL number (must be ≤ 1 for stability)
    dim : int (1 or 2)
        Spatial dimension
    
    Returns:
    --------
    For dim=1: x, t, dx, dt
    For dim=2: x, y, t, dx, dy, dt
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
# Finite Difference Solver (Core)
# -----------------------------------------------------------------------------
def fd_solve(x, t, dx, dt, y=None, dy=None, c=1.0, dim=1):
    """
    Core FD solver: takes pre-computed grid as input.
    
    Wave equation: ∂²u/∂t² = c²∇²u
    
    Domain: [0,1]^dim × [0,T]
    BC: u = 0 on all boundaries
    IC: u = sin(πx)[sin(πy)], ∂u/∂t = 0
    
    Parameters:
    -----------
    x : array (Nx+1,)
        Spatial grid in x
    t : array (Nt+1,)
        Time grid
    dx : float
        Spatial step size in x
    dt : float
        Time step size
    y : array (Ny+1,), optional
        Spatial grid in y (required for dim=2)
    dy : float, optional
        Spatial step size in y (required for dim=2)
    c : float
        Wave speed
    dim : int (1 or 2)
        Spatial dimension
    
    Returns:
    --------
    u : array
        Shape: (Nt+1, Nx+1) for 1D, (Nt+1, Nx+1, Ny+1) for 2D
    """
    if dim == 1:
        # ===== 1D =====
        Nx = len(x) - 1
        Nt = len(t) - 1
        r = (c * dt / dx) ** 2
        
        u = jnp.zeros((Nt + 1, Nx + 1))
        u = u.at[0, :].set(jnp.sin(jnp.pi * x))
        
        # First time step
        u = u.at[1, 1:Nx].set(
            u[0, 1:Nx] + 0.5 * r * (u[0, 2:Nx+1] - 2*u[0, 1:Nx] + u[0, 0:Nx-1])
        )
        u = u.at[1, 0].set(0.0)
        u = u.at[1, Nx].set(0.0)
        
        # Time stepping
        for n in range(1, Nt):
            u = u.at[n+1, 1:Nx].set(
                2*u[n, 1:Nx] - u[n-1, 1:Nx] + 
                r * (u[n, 2:Nx+1] - 2*u[n, 1:Nx] + u[n, 0:Nx-1])
            )
            u = u.at[n+1, 0].set(0.0)
            u = u.at[n+1, Nx].set(0.0)
        
        return u
        
    elif dim == 2:
        # ===== 2D =====
        if y is None or dy is None:
            raise ValueError("y and dy must be provided for 2D")
            
        Nx = len(x) - 1
        Ny = len(y) - 1
        Nt = len(t) - 1
        
        rx = (c * dt / dx) ** 2
        ry = (c * dt / dy) ** 2
        
        u = jnp.zeros((Nt + 1, Nx + 1, Ny + 1))
        
        # Initial condition
        X, Y = jnp.meshgrid(x, y, indexing='ij')
        u = u.at[0, :, :].set(jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y))
        
        # First time step
        u = u.at[1, 1:Nx, 1:Ny].set(
            u[0, 1:Nx, 1:Ny] + 
            0.5 * rx * (u[0, 2:Nx+1, 1:Ny] - 2*u[0, 1:Nx, 1:Ny] + u[0, 0:Nx-1, 1:Ny]) +
            0.5 * ry * (u[0, 1:Nx, 2:Ny+1] - 2*u[0, 1:Nx, 1:Ny] + u[0, 1:Nx, 0:Ny-1])
        )
        
        # Time stepping
        for n in range(1, Nt):
            u = u.at[n+1, 1:Nx, 1:Ny].set(
                2 * u[n, 1:Nx, 1:Ny] - u[n-1, 1:Nx, 1:Ny] +
                rx * (u[n, 2:Nx+1, 1:Ny] - 2*u[n, 1:Nx, 1:Ny] + u[n, 0:Nx-1, 1:Ny]) +
                ry * (u[n, 1:Nx, 2:Ny+1] - 2*u[n, 1:Nx, 1:Ny] + u[n, 1:Nx, 0:Ny-1])
            )
        
        return u
        
    else:
        raise ValueError(f"dim must be 1 or 2, got {dim}")




