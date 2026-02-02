import jax
jax.config.update('jax_platform_name', 'cpu')
import jax.numpy as jnp


# -----------------------------------------------------------------------------
# Analytical solution
# -----------------------------------------------------------------------------
def u_exact_1d_constant_coeffs(x, t, c=1.0):

    u = jnp.sin(jnp.pi * x) * jnp.cos(jnp.pi * c * t[:, None])
    return u


# -----------------------------------------------------------------------------
# 2D Wave equation - Constant coefficients
# -----------------------------------------------------------------------------
def u_exact_2d_constant_coeffs(x, y, t, c=1.0):

    # Create meshgrid for spatial coordinates
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    
    # Spatial part: sin(πx)sin(πy)
    spatial_part = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y)
    
    # Temporal part: cos(√2·πct)
    temporal_part = jnp.cos(jnp.sqrt(2) * jnp.pi * c * t)
    
    # Combine: (Nt, Nx, Ny)
    u = temporal_part[:, None, None] * spatial_part[None, :, :]
    
    return u


# -----------------------------------------------------------------------------
# 3D Wave equation - Constant coefficients
# -----------------------------------------------------------------------------
def u_exact_3d_constant_coeffs(x, y, z, t, c=1.0):

    # Create meshgrid for spatial coordinates
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    
    # Spatial part: sin(πx)sin(πy)sin(πz)
    spatial_part = jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y) * jnp.sin(jnp.pi * Z)
    
    # Temporal part: cos(√3·πct)
    temporal_part = jnp.cos(jnp.sqrt(3) * jnp.pi * c * t)
    
    # Combine: (Nt, Nx, Ny, Nz)
    u = temporal_part[:, None, None, None] * spatial_part[None, :, :, :]
    
    return u



# -----------------------------------------------------------------------------
# Grid helper
# -----------------------------------------------------------------------------
def create_grid_wave_1d(Nx, T, c=1.0, cfl=0.5):
    """
    Create 1D spatial + time grid for wave equation.
    CFL condition: c*dt/dx ≤ 1
    """
    dx = 1.0 / Nx
    dt = cfl * dx / c
    Nt = int(T / dt)
    
    x = jnp.linspace(0, 1, Nx + 1)
    t = jnp.linspace(0, T, Nt + 1)
    
    return x, t, dx, dt


def create_grid_wave_2d(Nx, Ny, T, c=1.0, cfl=0.5):
    """
    Create 2D spatial + time grid for wave equation.
    CFL condition: c*dt*sqrt(1/dx² + 1/dy²) ≤ 1
    """
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    
    # For square grid (dx=dy): dt ≤ dx/(c*sqrt(2))
    dt = cfl * min(dx, dy) / (c * jnp.sqrt(2))
    Nt = int(T / dt)
    
    x = jnp.linspace(0, 1, Nx + 1)
    y = jnp.linspace(0, 1, Ny + 1)
    t = jnp.linspace(0, T, Nt + 1)
    
    return x, y, t, dx, dy, dt


def create_grid_wave_3d(Nx, Ny, Nz, T, c=1.0, cfl=0.5):
    """
    Create 3D spatial + time grid for wave equation.
    CFL condition: c*dt*sqrt(1/dx² + 1/dy² + 1/dz²) ≤ 1
    """
    dx = 1.0 / Nx
    dy = 1.0 / Ny
    dz = 1.0 / Nz
    
    # For cubic grid (dx=dy=dz): dt ≤ dx/(c*sqrt(3))
    dt = cfl * min(dx, dy, dz) / (c * jnp.sqrt(3))
    Nt = int(T / dt)
    
    x = jnp.linspace(0, 1, Nx + 1)
    y = jnp.linspace(0, 1, Ny + 1)
    z = jnp.linspace(0, 1, Nz + 1)
    t = jnp.linspace(0, T, Nt + 1)
    
    return x, y, z, t, dx, dy, dz, dt


# -----------------------------------------------------------------------------
# Finite difference solvers 
# -----------------------------------------------------------------------------
def fd_solve_wave_1d(Nx, T, c=1.0, cfl=0.5):
    """
    Solve 1D wave equation: ∂²u/∂t² = c²∂²u/∂x²
    
    Domain: x ∈ [0,1], t ∈ [0,T]
    BC: u(0,t) = u(1,t) = 0
    IC: u(x,0) = sin(πx), ∂u/∂t(x,0) = 0
    
    Parameters:
    -----------
    Nx : int
        Number of spatial intervals
    T : float
        Final time
    c : float
        Wave speed
    cfl : float
        CFL number (must be ≤ 1 for stability)
    
    Returns:
    --------
    u : array of shape (Nt+1, Nx+1)
        Solution at all time steps
    x : array of shape (Nx+1,)
    t : array of shape (Nt+1,)
    """
    x, t, dx, dt = create_grid_wave_1d(Nx, T, c, cfl)
    Nt = len(t) - 1
    
    # Courant number
    r = (c * dt / dx) ** 2
    
    # Initialize solution
    u = jnp.zeros((Nt + 1, Nx + 1))
    
    # Initial condition: u(x,0) = sin(πx)
    u = u.at[0, :].set(jnp.sin(jnp.pi * x))
    
    # First time step (special treatment for ∂u/∂t(x,0) = 0)
    u = u.at[1, 1:Nx].set(
        u[0, 1:Nx] + 0.5 * r * (u[0, 2:Nx+1] - 2*u[0, 1:Nx] + u[0, 0:Nx-1])
    )
    u = u.at[1, 0].set(0.0)
    u = u.at[1, Nx].set(0.0)
    
    # Time stepping: u[n+1] = 2u[n] - u[n-1] + r*(u[n,i+1] - 2u[n,i] + u[n,i-1])
    for n in range(1, Nt):
        u = u.at[n+1, 1:Nx].set(
            2*u[n, 1:Nx] - u[n-1, 1:Nx] + 
            r * (u[n, 2:Nx+1] - 2*u[n, 1:Nx] + u[n, 0:Nx-1])
        )
        u = u.at[n+1, 0].set(0.0)
        u = u.at[n+1, Nx].set(0.0)
    
    return u, x, t


def fd_solve_wave_2d(Nx, Ny, T, c=1.0, cfl=0.5):
    """
    Solve 2D wave equation: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y²)
    
    Domain: (x,y) ∈ [0,1]×[0,1], t ∈ [0,T]
    BC: u = 0 on all boundaries
    IC: u(x,y,0) = sin(πx)sin(πy), ∂u/∂t(x,y,0) = 0
    
    Parameters:
    -----------
    Nx, Ny : int
        Number of spatial intervals in x and y
    T : float
        Final time
    c : float
        Wave speed
    cfl : float
        CFL number (must be ≤ 1 for stability)
    
    Returns:
    --------
    u : array of shape (Nt+1, Nx+1, Ny+1)
        Solution at all time steps
    x : array of shape (Nx+1,)
    y : array of shape (Ny+1,)
    t : array of shape (Nt+1,)
    """
    x, y, t, dx, dy, dt = create_grid_wave_2d(Nx, Ny, T, c, cfl)
    Nt = len(t) - 1
    
    # Courant numbers
    rx = (c * dt / dx) ** 2
    ry = (c * dt / dy) ** 2
    
    # Initialize solution: u[time, x, y]
    u = jnp.zeros((Nt + 1, Nx + 1, Ny + 1))
    
    # Initial condition: u(x,y,0) = sin(πx)sin(πy)
    X, Y = jnp.meshgrid(x, y, indexing='ij')
    u = u.at[0, :, :].set(jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y))
    
    # First time step (∂u/∂t = 0 at t=0)
    u = u.at[1, 1:Nx, 1:Ny].set(
        u[0, 1:Nx, 1:Ny] + 
        0.5 * rx * (u[0, 2:Nx+1, 1:Ny] - 2*u[0, 1:Nx, 1:Ny] + u[0, 0:Nx-1, 1:Ny]) +
        0.5 * ry * (u[0, 1:Nx, 2:Ny+1] - 2*u[0, 1:Nx, 1:Ny] + u[0, 1:Nx, 0:Ny-1])
    )
    
    # Time stepping
    for n in range(1, Nt):
        # Interior points
        u = u.at[n+1, 1:Nx, 1:Ny].set(
            2 * u[n, 1:Nx, 1:Ny] - u[n-1, 1:Nx, 1:Ny] +
            rx * (u[n, 2:Nx+1, 1:Ny] - 2*u[n, 1:Nx, 1:Ny] + u[n, 0:Nx-1, 1:Ny]) +
            ry * (u[n, 1:Nx, 2:Ny+1] - 2*u[n, 1:Nx, 1:Ny] + u[n, 1:Nx, 0:Ny-1])
        )
        
        # Boundary conditions (already 0 from initialization)
    
    return u, x, y, t


def fd_solve_wave_3d(Nx, Ny, Nz, T, c=1.0, cfl=0.5):
    """
    Solve 3D wave equation: ∂²u/∂t² = c²(∂²u/∂x² + ∂²u/∂y² + ∂²u/∂z²)
    
    Domain: (x,y,z) ∈ [0,1]³, t ∈ [0,T]
    BC: u = 0 on all boundaries
    IC: u(x,y,z,0) = sin(πx)sin(πy)sin(πz), ∂u/∂t(x,y,z,0) = 0
    
    Parameters:
    -----------
    Nx, Ny, Nz : int
        Number of spatial intervals in x, y, and z
    T : float
        Final time
    c : float
        Wave speed
    cfl : float
        CFL number (must be ≤ 1 for stability)
    
    Returns:
    --------
    u : array of shape (Nt+1, Nx+1, Ny+1, Nz+1)
        Solution at all time steps
    x : array of shape (Nx+1,)
    y : array of shape (Ny+1,)
    z : array of shape (Nz+1,)
    t : array of shape (Nt+1,)
    """
    x, y, z, t, dx, dy, dz, dt = create_grid_wave_3d(Nx, Ny, Nz, T, c, cfl)
    Nt = len(t) - 1
    
    # Courant numbers
    rx = (c * dt / dx) ** 2
    ry = (c * dt / dy) ** 2
    rz = (c * dt / dz) ** 2
    
    # Initialize solution: u[time, x, y, z]
    u = jnp.zeros((Nt + 1, Nx + 1, Ny + 1, Nz + 1))
    
    # Initial condition: u(x,y,z,0) = sin(πx)sin(πy)sin(πz)
    X, Y, Z = jnp.meshgrid(x, y, z, indexing='ij')
    u = u.at[0, :, :, :].set(
        jnp.sin(jnp.pi * X) * jnp.sin(jnp.pi * Y) * jnp.sin(jnp.pi * Z)
    )
    
    # First time step (∂u/∂t = 0 at t=0)
    u = u.at[1, 1:Nx, 1:Ny, 1:Nz].set(
        u[0, 1:Nx, 1:Ny, 1:Nz] + 
        0.5 * rx * (u[0, 2:Nx+1, 1:Ny, 1:Nz] - 2*u[0, 1:Nx, 1:Ny, 1:Nz] + u[0, 0:Nx-1, 1:Ny, 1:Nz]) +
        0.5 * ry * (u[0, 1:Nx, 2:Ny+1, 1:Nz] - 2*u[0, 1:Nx, 1:Ny, 1:Nz] + u[0, 1:Nx, 0:Ny-1, 1:Nz]) +
        0.5 * rz * (u[0, 1:Nx, 1:Ny, 2:Nz+1] - 2*u[0, 1:Nx, 1:Ny, 1:Nz] + u[0, 1:Nx, 1:Ny, 0:Nz-1])
    )
    
    # Time stepping
    for n in range(1, Nt):
        # Interior points
        u = u.at[n+1, 1:Nx, 1:Ny, 1:Nz].set(
            2 * u[n, 1:Nx, 1:Ny, 1:Nz] - u[n-1, 1:Nx, 1:Ny, 1:Nz] +
            rx * (u[n, 2:Nx+1, 1:Ny, 1:Nz] - 2*u[n, 1:Nx, 1:Ny, 1:Nz] + u[n, 0:Nx-1, 1:Ny, 1:Nz]) +
            ry * (u[n, 1:Nx, 2:Ny+1, 1:Nz] - 2*u[n, 1:Nx, 1:Ny, 1:Nz] + u[n, 1:Nx, 0:Ny-1, 1:Nz]) +
            rz * (u[n, 1:Nx, 1:Ny, 2:Nz+1] - 2*u[n, 1:Nx, 1:Ny, 1:Nz] + u[n, 1:Nx, 1:Ny, 0:Nz-1])
        )
        
        # Boundary conditions (already 0 from initialization)
    
    return u, x, y, z, t


# -----------------------------------------------------------------------------
# Main - Simple comparison plots
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    # 1D
    u_fd_1d, x_1d, t_1d = fd_solve_wave_1d(Nx=100, T=2.0, c=1.0, cfl=0.5)
    u_exact_1d = u_exact_1d_constant_coeffs(x_1d, t_1d, c=1.0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_1d, u_exact_1d[-1, :], label='Analytical')
    plt.plot(x_1d, u_fd_1d[-1, :], linestyle='--', label='FD')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('1D Wave Equation at final time')
    plt.legend()
    plt.grid(True)
    plt.savefig('../figs/comparison_1d.pdf')
    
    # 2D
    u_fd_2d, x_2d, y_2d, t_2d = fd_solve_wave_2d(Nx=50, Ny=50, T=1.0, c=1.0, cfl=0.5)
    u_exact_2d = u_exact_2d_constant_coeffs(x_2d, y_2d, t_2d, c=1.0)
    
    plt.figure(figsize=(10, 6))
    plt.plot(x_2d, u_exact_2d[-1, :, len(y_2d)//2], label='Analytical')
    plt.plot(x_2d, u_fd_2d[-1, :, len(y_2d)//2], linestyle='--', label='FD')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('2D Wave Equation at final time (slice at y=0.5)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../figs/comparison_2d.pdf')

    # 3D
    u_fd_3d, x_3d, y_3d, z_3d, t_3d = fd_solve_wave_3d(
        Nx=30, Ny=30, Nz=30, T=0.5, c=1.0, cfl=0.5
    )
    u_exact_3d = u_exact_3d_constant_coeffs(x_3d, y_3d, z_3d, t_3d, c=1.0)
    
    plt.figure(figsize=(10, 6))
    y_idx = len(y_3d)//2
    z_idx = len(z_3d)//2
    plt.plot(x_3d, u_exact_3d[-1, :, y_idx, z_idx], label='Analytical')
    plt.plot(x_3d, u_fd_3d[-1, :, y_idx, z_idx], linestyle='--', label='FD')
    plt.xlabel('x')
    plt.ylabel('u')
    plt.title('3D Wave Equation at final time (slice at y=0.5, z=0.5)')
    plt.legend()
    plt.grid(True)
    plt.savefig('../figs/comparison_3d.pdf')
