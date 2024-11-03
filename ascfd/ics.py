from ascfd.constants import *
import numpy as np



def diagonal_advection_2d(a_x, a_y, a_var, t=0):
    """
    Diagonal advection test case for 2D Euler equations.
    """
    if a_var == 0: #RHOCOMP
        return 1.0 + 0.2 * np.sin(2 * np.pi * (a_x + a_y - t))
    elif a_var == 1: #UCOMP
        return np.ones_like(a_x)
    elif a_var == 2: #VCOMP
        return np.ones_like(a_x)
    elif a_var == 3: #PCOMP
        return np.ones_like(a_x)
    else:
        raise ValueError(f"Unexpected variable: {a_var}")

def kelvin_helmholtz_2d(a_x, a_y, a_var):
    """
    Kelvin-Helmholtz instability test case for 2D Euler equations.
    """
    if a_var == 0: #RHOCOMP
        return np.ones_like(a_x)
    elif a_var == 1: #UCOMP
        return 0.5 * (np.tanh(20 * a_y) - 1)
    elif a_var == 2: #VCOMP
        return 0.1 * np.sin(2 * np.pi * a_x)
    elif a_var == 3: #PCOMP
        return 2.5 * np.ones_like(a_x)
    else:
        raise ValueError(f"Unexpected variable: {a_var}")

def double_mach_reflection_2d(a_x, a_y, a_var):
    """
    Double Mach reflection test case for 2D Euler equations.
    """
    x0 = 1/6
    mask = a_x < x0 + a_y / np.sqrt(3)
    
    if a_var == 0: #RHOCOMP
        return np.where(mask, 8.0, 1.4)
    elif a_var == 1: #UCOMP
        return np.where(mask, 8.25 * np.cos(np.pi/6), 0.0)
    elif a_var == 2: #VCOMP
        return np.where(mask, -8.25 * np.sin(np.pi/6), 0.0)
    elif a_var == 3: #PCOMP
        return np.where(mask, 116.5, 1.0)
    else:
        raise ValueError(f"Unexpected variable: {a_var}")

def riemann_2d(a_x, a_y, a_var):
    """
    2D Riemann problem for 2D Euler equations.
    """
    # Define the four states
    rho = np.select([
        (a_x >= 0.5) & (a_y >= 0.5),
        (a_x < 0.5) & (a_y >= 0.5),
        (a_x < 0.5) & (a_y < 0.5),
        (a_x >= 0.5) & (a_y < 0.5)
    ], [1.5, 0.5323, 0.138, 0.5323])
    
    u = np.select([
        (a_x >= 0.5) & (a_y >= 0.5),
        (a_x < 0.5) & (a_y >= 0.5),
        (a_x < 0.5) & (a_y < 0.5),
        (a_x >= 0.5) & (a_y < 0.5)
    ], [0.0, 1.206, 1.206, 0.0])
    
    v = np.select([
        (a_x >= 0.5) & (a_y >= 0.5),
        (a_x < 0.5) & (a_y >= 0.5),
        (a_x < 0.5) & (a_y < 0.5),
        (a_x >= 0.5) & (a_y < 0.5)
    ], [0.0, 0.0, 1.206, 1.206])
    
    p = np.select([
        (a_x >= 0.5) & (a_y >= 0.5),
        (a_x < 0.5) & (a_y >= 0.5),
        (a_x < 0.5) & (a_y < 0.5),
        (a_x >= 0.5) & (a_y < 0.5)
    ], [1.5, 0.3, 0.029, 0.3])

    if a_var == 0: #RHOCOMP
        return rho
    elif a_var == 1: #UCOMP
        return u
    elif a_var == 2: #VCOMP
        return v
    elif a_var == 3: #PCOMP
        return p
    else:
        raise ValueError(f"Unexpected variable: {a_var}")
