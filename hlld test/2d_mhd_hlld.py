'''
This code implements the HLLC (Harten-Lax-van Leer-Contact) approximate Riemann solver 
for the 1D Euler equations. The HLLC solver considers three waves:
1. Left-going wave (shock or rarefaction)
2. Contact discontinuity
3. Right-going wave (shock or rarefaction)

The solver computes approximate wave speeds and uses them to determine the numerical flux.
It maintains positivity of density and pressure, and accurately captures shocks, 
rarefactions, and contact discontinuities. The method is first-order accurate in space 
and time but can be extended to higher order with appropriate reconstruction methods.
'''

import numpy as np
import matplotlib.pyplot as plt
import os

# Constants
gamma = 1.4  # Ratio of specific heats
Nx = 1000     # Number of grid points in x-direction
Ny = 1000   # Number of grid points in y-direction
Lx = 1.0    # Domain length
Ly = 1.0    # Domain height
dx = Lx / (Nx - 1)  # Spatial step in x-direction
dy = Ly / (Ny - 1)  # Spatial step in y-direction
CFL = 0.5    # Courant number

# Spatial grid
x = np.linspace(0, Lx, Nx)
y = np.linspace(0, Ly, Ny)
meshX, meshY = np.meshgrid(x, y)

# Initialize all arrays

# Primitive variables
rho = np.zeros((Ny, Nx))  # Density
p = np.zeros((Ny, Nx))    # Pressure
pT = np.zeros((Nx, Ny))  # Total pressure???
u = np.zeros((Ny, Nx))    # X-Velocity
v = np.zeros((Ny, Nx))    # Y-Velocity
w = np.zeros((Ny, Nx))    # Z-Velocity???

# Conservative variables
### IN ORDER: [ρ, ρu′, ρv′, ρw, e′, B′x, B′y, Bz]
mu = np.zeros((Ny, Nx))  # X-Momentum
mv = np.zeros((Ny, Nx))  # Y-Momentum
mw = np.zeros((Ny, Nx))  # Z-Momentum???

E = np.zeros((Ny, Nx))    # Total energy

# Primitive and conservative variables
Bx = np.zeros((Ny, Nx))   # X-Magnetic field
By = np.zeros((Ny, Nx))   # Y-Magnetic field
Bz = np.zeros((Ny, Nx))   # Z-Magnetic field???

# Select initial conditions
IC_type = "sod"  # Options: "sod", "shu-osher"

# Create output directory if it doesn't exist
output_dir = "results_char"
os.makedirs(output_dir, exist_ok=True)

# Initial conditions
# if IC_type == "sod":
#     # Sod shock tube
#     rho[:] = 1.0
#     rho[x >= 0.5] = 0.125  # Right side density
    
#     p[:] = 1.0
#     p[x >= 0.5] = 0.1  # Right side pressure
    
#     u[:] = 0.0  # Velocity (zero everywhere)

#     t_final = 0.2  # Final time


# elif IC_type == "shu-osher":
#     # Shu-Osher problem
#     # Left state (post-shock)
#     rho_l = 3.857143
#     p_l = 10.33333
#     u_l = 2.629369
    
#     # Right state (pre-shock with oscillations)
#     rho_r = 1.0 + 0.2 * np.sin(5 * np.pi * x)  # Density with sine wave
#     p_r = np.ones(Nx)
#     u_r = np.zeros(Nx)
    
#     # Combine left and right states
#     shock_pos = 0.1
#     rho = np.where(x < shock_pos, rho_l, rho_r)
#     p = np.where(x < shock_pos, p_l, p_r)
#     u = np.where(x < shock_pos, u_l, u_r)
    
#     t_final = 0.18 # Final time

# Convert to conservative variables

mu = rho * u  # X-Momentum
mv = rho * v  # Y-Momentum
mw = rho * w  # Z-Momentum??

E = p / (gamma - 1) + 0.5 * rho * u**2  # Total energy


def cons_to_prim(mu, mv, mw, E):
    """
    Convert conservative variables to primitive variables
    """
    
    u = mu / rho
    v = mv / rho
    w = mw / rho
    p = (gamma - 1) * (E - 0.5 * rho * (u**2 + v**2 + w**2))
    pT = p + 0.5 * (Bx**2 + By**2 + Bz**2)  # Total pressure
    
    return u, v, w, p, pT


def prim_to_cons(rho, p, u, v, w, Bx, By, Bz):
    """
    Convert primitive variables to conservative variables
    """
    
    mu = rho * u
    mv = rho * v
    mw = rho * w
    E = p / (gamma - 1) + 0.5 * rho * (u**2 + v**2 + w**2) \
        + 0.5 * (Bx**2 + By**2 + Bz**2) # should we include the magnetic field energy?
    
    return mu, mv, mw, E


def compute_flux(rho, pT, u, v, w, Bx, By, Bz, E):
    """
    Compute fluxes for each conserved variable
    """
    
    F = np.zeros((8, Nx))
    
    F[0, :] = rho * u
    F[1, :] = rho * u**2 - pT + Bx**2
    F[2, :] = rho * u * v - Bx * By
    F[3, :] = rho * u * w - Bx * Bz
    F[4, :] = (E + pT) * u - Bx * (Bx * u + By * v + Bz * w) # v dot B where B = [Bx, By, Bz]^T
    F[5, :] = 0
    F[6, :] = By * u - Bx * v
    F[7, :] = Bz * u - Bx * w
    
    return F


def compute_wavespeeds(rho_L, rho_R, u_L, u_R, pT_R, pT_L, Bx, rho_star_L, rho_star_R):
    """
    Compute wave speeds for HLLC solver
    """
    # Compute pressure-based wave speed estimates
    ### ARE THESE CORRECT LOL????
    
    B = np.array([Bx, By, Bz])
    B_magnitude = np.linalg.norm(B)
    
    cf_L = (gamma * p_L + B_magnitude**2 + np.sqrt((gamma * p_L + B_magnitude**2)**2) - 4 * gamma * p_L * np.linalg.norm(Bx)) / (2*rho_L)
    cf_R = (gamma * p_R + B_magnitude**2 + np.sqrt((gamma * p_R + B_magnitude**2)**2) - 4 * gamma * p_R * np.linalg.norm(Bx)) / (2*rho_L)
    
    S_M = ((S_R - u_R) * rho_R * u_R - (S_L - u_L) * rho_L * u_L - pT_R + pT_L) / \
        (rho_R * (S_R - u_R) - rho_L * (S_L - u_L))
    
    S_star_L = S_M - np.linalg.norm(Bx) / np.sqrt(rho_star_L)
    S_star_R = S_M + np.linalg.norm(Bx) / np.sqrt(rho_star_R)
    
    S_L = np.minimum(u_L, u_R) - np.maximum(cf_L, cf_R)
    S_R = np.maximum(u_L, u_R) + np.maximum(cf_L, cf_R)
    
    return S_M, S_star_L, S_star_R, S_L, S_R


def HLLD_flux(rho_L, rho_R, u_L, u_R, v_L, v_R, w_L, w_R, pT_L, pT_R, pT_star, E_L, E_R, Bx_L, Bx_R, By_L, By_R, Bz_L, Bz_R):
    """
    Compute HLLD fluxes
    """
    
    # Compute wave speeds
    S_M, S_star_L, S_star_R, S_L, S_R = compute_wavespeeds(rho_L, rho_R, u_L, u_R, p_L, p_R)
    
    # Compute conserved variables
    U_L = np.array([rho_L, rho_L * u_L, E_L])
    U_R = np.array([rho_R, rho_R * u_R, E_R])
    
    # Compute physical fluxes
    F_L = np.array([
        rho_L * u_L,
        rho_L * u_L**2 - pT_L + Bx_L**2,
        rho_L * u_L * v_L - Bx_L * By_L,
        rho_L * u_L * w_L - Bx_L * Bz_L,
        (E_L + pT_L) * u_L - Bx_L * (Bx_L * u_L + By_L * v_L + Bz_L * w_L),
        0,
        By_L * u_L - Bx_L * v_L,
        Bz_L * u_L - Bx_L * w_L
    ])

    F_R = np.array([
        rho_R * u_R,
        rho_R * u_R**2 - pT_R + Bx_R**2,
        rho_R * u_R * v_R - Bx_R * By_R,
        rho_R * u_R * w_R - Bx_R * Bz_R,
        (E_R + pT_R) * u_R - Bx_R * (Bx_R * u_R + By_R * v_R + Bz_R * w_R),
        0,
        By_R * u_R - Bx_R * v_R,
        Bz_R * u_R - Bx_R * w_R
    ])
    
    # Compute intermediate states
    # Left state
    U_prim_star_L = np.array([
        rho_L * (S_L - u_L)/(S_L - S_M),
        v_L - Bx * By_L * (S_M - u_L)/(rho_L * (S_L - u_L) * (S_L - S_M) - Bx**2),
        w_L - Bx * Bz_L * (S_M - u_L)/(rho_L * (S_L - u_L) * (S_L - S_M) - Bx**2),
        By_L * (rho_L * (S_L - u_L)**2 - Bx**2)/(rho_L * (S_L - u_L) * (S_L - S_M) - Bx**2),
        Bz_L * (rho_L * (S_L - u_L)**2 - Bx**2)/(rho_L * (S_L - u_L) * (S_L - S_M) - Bx**2),
        ((S_L - u_L) * E_L - pT_L * u_L + pT_star * S_M + 
        Bx * (v_L * By_L + w_L * Bz_L - 
                    (v_L - Bx * By_L * (S_M - u_L)/(rho_L * (S_L - u_L) * (S_L - S_M) - Bx**2)) * 
                    By_L * (rho_L * (S_L - u_L)**2 - Bx**2)/(rho_L * (S_L - u_L) * (S_L - S_M) - Bx**2) -
                    (w_L - Bx * Bz_L * (S_M - u_L)/(rho_L * (S_L - u_L) * (S_L - S_M) - Bx**2)) * 
                    Bz_L * (rho_L * (S_L - u_L)**2 - Bx**2)/(rho_L * (S_L - u_L) * (S_L - S_M) - Bx**2)))
        /(S_L - S_M)
    ])

    # Right state
    U_prim_star_R = np.array([
        rho_R * (S_R - u_R)/(S_R - S_M),
        v_R - Bx * By_R * (S_M - u_R)/(rho_R * (S_R - u_R) * (S_R - S_M) - Bx**2),
        w_R - Bx * Bz_R * (S_M - u_R)/(rho_R * (S_R - u_R) * (S_R - S_M) - Bx**2),
        By_R * (rho_R * (S_R - u_R)**2 - Bx**2)/(rho_R * (S_R - u_R) * (S_R - S_M) - Bx**2),
        Bz_R * (rho_R * (S_R - u_R)**2 - Bx**2)/(rho_R * (S_R - u_R) * (S_R - S_M) - Bx**2),
        ((S_R - u_R) * E_R - pT_R * u_R + pT_star * S_M + 
        Bx * (v_R * By_R + w_R * Bz_R - 
                    (v_R - Bx * By_R * (S_M - u_R)/(rho_R * (S_R - u_R) * (S_R - S_M) - Bx**2)) * 
                    By_R * (rho_R * (S_R - u_R)**2 - Bx**2)/(rho_R * (S_R - u_R) * (S_R - S_M) - Bx**2) -
                    (w_R - Bx * Bz_R * (S_M - u_R)/(rho_R * (S_R - u_R) * (S_R - S_M) - Bx**2)) * 
                    Bz_R * (rho_R * (S_R - u_R)**2 - Bx**2)/(rho_R * (S_R - u_R) * (S_R - S_M) - Bx**2)))
        /(S_R - S_M)
    ])
    
    # Select flux based on wave speeds
    if S_L >= 0:
        return F_L
    elif S_L <= 0 and S_star >= 0:
        return F_L + S_L * (U_star_L - U_L)
    elif S_star <= 0 and S_R >= 0:
        return F_R + S_R * (U_star_R - U_R)
    else:
        return F_R

# Time integration loop
t = 0.0
step = 0
while t < t_final:
    # Convert conservative -> primitive
    rho, u, p = cons_to_prim(rho, mom, E)

    # Compute speed of sound
    c = np.sqrt(gamma * p / rho)
    
    # Compute time step based on CFL condition
    dt = CFL * dx / np.max(np.abs(u) + c)
    
    print(f"Time: {t:.6f}, Timestep Num: {step}")

    # Initialize flux arrays
    F = np.zeros((3, Nx))
    
    # Compute HLLC fluxes at each interface
    for i in range(Nx-1):
        # Left and right states
        rho_L, u_L, p_L = rho[i], u[i], p[i]
        rho_R, u_R, p_R = rho[i+1], u[i+1], p[i+1]
        E_L, E_R = E[i], E[i+1]
        
        # Compute HLLD flux
        F[:, i] = HLLD_flux(rho_L, rho_R, u_L, u_R, p_L, p_R, E_L, E_R)
    
    # Update conservative variables
    rho[1:-1] = rho[1:-1] - dt/dx * (F[0, 1:Nx-1] - F[0, :Nx-2])
    mom[1:-1] = mom[1:-1] - dt/dx * (F[1, 1:Nx-1] - F[1, :Nx-2])
    E[1:-1] = E[1:-1] - dt/dx * (F[2, 1:Nx-1] - F[2, :Nx-2])

    # Apply boundary conditions (transmissive)
    rho[0] = rho[1]
    rho[-1] = rho[-2]
    mom[0] = mom[1]
    mom[-1] = mom[-2]
    E[0] = E[1]
    E[-1] = E[-2]

    # Convert to primitive variables for plotting
    rho_plot, u_plot, p_plot = cons_to_prim(rho, mom, E)
    
    # Save plot at current timestep
    plt.figure(figsize=(15, 5))
    
    # Density plot
    plt.subplot(1, 3, 1)
    plt.plot(x, rho_plot, label=f"Density (t={t:.3f})", color='b')
    plt.xlabel("x")
    plt.ylabel("Density")
    plt.title("Density Profile")
    plt.grid()
    plt.legend()

    # Velocity plot
    plt.subplot(1, 3, 2)
    plt.plot(x, u_plot, label=f"Velocity (t={t:.3f})", color='r')
    plt.xlabel("x")
    plt.ylabel("Velocity")
    plt.title("Velocity Profile")
    plt.grid()
    plt.legend()

    # Pressure plot
    plt.subplot(1, 3, 3)
    plt.plot(x, p_plot, label=f"Pressure (t={t:.3f})", color='g')
    plt.xlabel("x")
    plt.ylabel("Pressure")
    plt.title("Pressure Profile")
    plt.grid()
    plt.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f'step_{step:04d}.png'))
    plt.close()
    
    t += dt
    step += 1

# Create video from PNG files
os.system(f'ffmpeg -y -framerate 30 -i {output_dir}/step_%04d.png -c:v libx264 -pix_fmt yuv420p {output_dir}/{IC_type}_simulation.mp4')