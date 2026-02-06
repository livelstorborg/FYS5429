import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from IPython.display import HTML

# =============================================================================
# SETUP
# =============================================================================
Nx = 200          # Antall punkter i rom
Nt = 500          # Antall tidssteg
L = 2.0           # Lengde av domenet [0, 1]
T = 10.0          # Total tid
c_background = 1.0  # Bølgehastighet i bakgrunn

x = np.linspace(0, L, Nx)
dx = x[1] - x[0]
dt = 0.4 * dx / c_background  # CFL betingelse
t = np.arange(0, Nt) * dt

# =============================================================================
# PARTIKLER - ENDRE DISSE!
# =============================================================================
partikler = [
    {'pos': 0.6, 'radius': 0.05, 'c_squared': 2.0},  # Statisk: rask bølge
    {'pos': 1.2, 'radius': 0.08, 'c_squared': 0.25}, # Statisk: treg bølge
    # {'pos': 0.7, 'radius': 0.06, 'type': 'vibrerende', 'freq': 5.0, 'amp': 1.0},
]

# =============================================================================
# FUNKSJON: Beregn c²(x,t)
# =============================================================================
def get_c_squared(x, t, partikler):
    """Returner c²(x,t) - kan ha partikler!"""
    c2 = np.ones_like(x) * c_background**2  # Start med c² = 1
    
    for p in partikler:
        # Er vi inne i denne partikkelen?
        avstand = np.abs(x - p['pos'])
        inne = avstand < p['radius']
        
        if 'type' in p and p['type'] == 'vibrerende':
            # Tid-modulert partikkel
            c2_partikkel = 1.0 + p['amp'] * np.sin(2 * np.pi * p['freq'] * t)
        else:
            # Statisk partikkel
            c2_partikkel = p['c_squared']
        
        c2[inne] = c2_partikkel
    
    return c2

# =============================================================================
# FINITE DIFFERENCE SOLVER
# =============================================================================
def solve_wave_with_particles(x, t, partikler):
    """Løs bølgeligning med partikler: ∂²u/∂t² = c²(x,t) ∂²u/∂x²"""
    Nx = len(x)
    Nt = len(t)
    dx = x[1] - x[0]
    dt = t[1] - t[0] if len(t) > 1 else 0.01
    
    u = np.zeros((Nt, Nx))
    
    # Initial condition: Gauss-puls
    x0 = 0.2  # Start-posisjon
    sigma = 0.05
    u[0, :] = np.exp(-((x - x0) / sigma)**2)
    
    # First time step (∂u/∂t = 0 initialt)
    for n in range(0, 1):
        c2 = get_c_squared(x, t[n], partikler)
        r = c2 * (dt / dx)**2
        
        u[1, 1:-1] = u[0, 1:-1] + 0.5 * r[1:-1] * (
            u[0, 2:] - 2*u[0, 1:-1] + u[0, :-2]
        )
        u[1, 0] = 0.0
        u[1, -1] = 0.0
    
    # Time stepping
    for n in range(1, Nt-1):
        c2 = get_c_squared(x, t[n], partikler)
        r = c2 * (dt / dx)**2
        
        u[n+1, 1:-1] = (2*u[n, 1:-1] - u[n-1, 1:-1] + 
                        r[1:-1] * (u[n, 2:] - 2*u[n, 1:-1] + u[n, :-2]))
        u[n+1, 0] = 0.0
        u[n+1, -1] = 0.0
    
    return u

# =============================================================================
# LØSNING
# =============================================================================
print("Løser bølgeligning...")
u = solve_wave_with_particles(x, t, partikler)
print("Ferdig!")

# =============================================================================
# ANIMASJON
# =============================================================================
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Subplot 1: Bølgen u(x,t)
line1, = ax1.plot([], [], 'b-', linewidth=2, label='u(x,t)')
ax1.set_xlim(0, L)
ax1.set_ylim(-1.5, 1.5)
ax1.set_xlabel('x', fontsize=12)
ax1.set_ylabel('u(x,t)', fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend()
time_text = ax1.text(0.02, 0.95, '', transform=ax1.transAxes, 
                     fontsize=12, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

# Tegn partikler (grå bokser)
for p in partikler:
    ax1.axvspan(p['pos'] - p['radius'], p['pos'] + p['radius'], 
                alpha=0.2, color='gray', label='Partikkel')

# Subplot 2: c²(x,t) profil
line2, = ax2.plot([], [], 'r-', linewidth=2, label='c²(x,t)')
ax2.set_xlim(0, L)
ax2.set_ylim(0, 5)
ax2.set_xlabel('x', fontsize=12)
ax2.set_ylabel('c²(x,t)', fontsize=12)
ax2.grid(True, alpha=0.3)
ax2.legend()
ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.3, label='c²=1 (bakgrunn)')

# Animasjonsfunksjon
def animate(frame):
    # Tegn bølgen
    line1.set_data(x, u[frame, :])
    
    # Tegn c² profil
    c2_profile = get_c_squared(x, t[frame], partikler)
    line2.set_data(x, c2_profile)
    
    time_text.set_text(f't = {t[frame]:.2f} s')
    return line1, line2, time_text

# Lag animasjonen (vis hver 5. frame for hastighet)
ani = FuncAnimation(fig, animate, frames=range(0, Nt, 5), 
                   interval=50, blit=True, repeat=True)

plt.tight_layout()
plt.show()

# Hvis du vil lagre som video:
# ani.save('wave_animation.mp4', writer='ffmpeg', fps=30)