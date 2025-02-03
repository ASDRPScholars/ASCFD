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
Nx = 1000     # Number of grid points
L = 1.0      # Domain length
dx = L / (Nx - 1)  # Spatial step
CFL = 0.5    # Courant number

# Spatial grid
x = np.linspace(0, L, Nx)

# Initialize all arrays
rho = np.zeros(Nx)  # Density
p = np.zeros(Nx)    # Pressure
u = np.zeros(Nx)    # Velocity
E = np.zeros(Nx)    # Total energy
mom = np.zeros(Nx)  # Momentum

# Select initial conditions
IC_type = "sod"  # Options: "sod", "shu-osher"

# Create output directory if it doesn't exist
output_dir = "results_char"
os.makedirs(output_dir, exist_ok=True)

# Initial conditions
if IC_type == "sod":
    # Sod shock tube
    rho[:] = 1.0
    rho[x >= 0.5] = 0.125  # Right side density
    
    p[:] = 1.0
    p[x >= 0.5] = 0.1  # Right side pressure
    
    u[:] = 0.0  # Velocity (zero everywhere)

    t_final = 0.2  # Final time


elif IC_type == "shu-osher":
    # Shu-Osher problem
    # Left state (post-shock)
    rho_l = 3.857143
    p_l = 10.33333
    u_l = 2.629369
    
    # Right state (pre-shock with oscillations)
    rho_r = 1.0 + 0.2 * np.sin(5 * np.pi * x)  # Density with sine wave
    p_r = np.ones(Nx)
    u_r = np.zeros(Nx)
    
    # Combine left and right states
    shock_pos = 0.1
    rho = np.where(x < shock_pos, rho_l, rho_r)
    p = np.where(x < shock_pos, p_l, p_r)
    u = np.where(x < shock_pos, u_l, u_r)
    
    t_final = 0.18 # Final time

# Convert to conservative variables
E = p / (gamma - 1) + 0.5 * rho * u**2  # Total energy
mom = rho * u  # Momentum

# Function to convert conservative -> primitive variables
def cons_to_prim(rho, mom, E):
    u = mom / rho
    p = (gamma - 1) * (E - 0.5 * rho * u**2)
    return rho, u, p

# Function to convert primitive -> conservative variables
def prim_to_cons(rho, u, p):
    mom = rho * u
    E = p / (gamma - 1) + 0.5 * rho * u**2
    return rho, mom, E

# Function to compute fluxes
def compute_flux(rho, mom, E, p):
    F = np.zeros((3, Nx))
    F[0, :] = mom
    F[1, :] = mom**2 / rho + p
    F[2, :] = (E + p) * mom / rho
    return F

def compute_wavespeeds(rho_L, rho_R, u_L, u_R, p_L, p_R, c_L, c_R):
    """
    Compute wave speeds for HLLC solver
    """
    # Compute pressure-based wave speed estimates
    p_star = max(0, 0.5 * (p_L + p_R - 0.5 * (u_R - u_L) * (rho_L + rho_R) * 0.5 * (c_L + c_R)))
    q_L = 1 if p_star <= p_L else np.sqrt(1 + ((gamma + 1)/(2 * gamma)) * (p_star/p_L - 1))
    q_R = 1 if p_star <= p_R else np.sqrt(1 + ((gamma + 1)/(2 * gamma)) * (p_star/p_R - 1))
    
    # Wave speed estimates
    S_L = u_L - c_L * q_L
    S_R = u_R + c_R * q_R
    S_star = (p_R - p_L + rho_L * u_L * (S_L - u_L) - rho_R * u_R * (S_R - u_R)) / \
             (rho_L * (S_L - u_L) - rho_R * (S_R - u_R))
    
    return S_L, S_star, S_R

def HLLC_flux(rho_L, rho_R, u_L, u_R, p_L, p_R, E_L, E_R):
    """
    Compute HLLC fluxes
    """
    # Compute sound speeds
    c_L = np.sqrt(gamma * p_L / rho_L)
    c_R = np.sqrt(gamma * p_R / rho_R)
    
    # Compute wave speeds
    S_L, S_star, S_R = compute_wavespeeds(rho_L, rho_R, u_L, u_R, p_L, p_R, c_L, c_R)
    
    # Compute conserved variables
    U_L = np.array([rho_L, rho_L * u_L, E_L])
    U_R = np.array([rho_R, rho_R * u_R, E_R])
    
    # Compute physical fluxes
    F_L = np.array([rho_L * u_L, 
                    rho_L * u_L**2 + p_L, 
                    (E_L + p_L) * u_L])
    F_R = np.array([rho_R * u_R, 
                    rho_R * u_R**2 + p_R, 
                    (E_R + p_R) * u_R])
    
    # Compute intermediate states
    U_star_L = rho_L * ((S_L - u_L)/(S_L - S_star)) * \
               np.array([1, S_star, E_L/rho_L + (S_star - u_L)*(S_star + p_L/(rho_L*(S_L - u_L)))])
    U_star_R = rho_R * ((S_R - u_R)/(S_R - S_star)) * \
               np.array([1, S_star, E_R/rho_R + (S_star - u_R)*(S_star + p_R/(rho_R*(S_R - u_R)))])
    
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
        
        # Compute HLLC flux
        F[:, i] = HLLC_flux(rho_L, rho_R, u_L, u_R, p_L, p_R, E_L, E_R)
    
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
