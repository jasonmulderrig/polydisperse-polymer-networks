import numpy as np
from src.helpers.network_utils import a_or_v_func

def A_or_V_arg_L_func(L: np.ndarray) -> float:
    """Simulation box area or volume.

    This function calculates the simulation box area or volume given the
    simulation box side lengths.

    Args:
        L (np.ndarray): 1D np.ndarray with dim float entries of the simulation box side lengths.

    Returns:
        float: Simulation box area or volume.
    
    """
    return np.prod(L)

def A_or_V_arg_rho_func(en: float, rho: float) -> float:
    """Simulation box area or volume.

    This function calculates the simulation box area or volume given the
    number of particles and the particle number density.

    Args:
        en (float): Number of particles.
        rho (float): Particle number density.

    Returns:
        float: Simulation box area or volume.
    
    """
    return en / rho

def A_or_V_arg_eta_func(dim: int, b: float, en: float, eta: float) -> float:
    """Simulation box area or volume.
    
    This function calculates the simulation box area or volume given the
    number of particles and the particle packing density.

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Particle diameter.
        en (float): Number of particles.
        eta (float): Particle packing density.

    Returns:
        float: Simulation box area or volume.
    
    """
    return a_or_v_func(dim, b) * en / eta

def L_arg_A_or_V_func(dim: int, A_or_V: float) -> np.ndarray:
    """Simulation box side lengths.
    
    This function calculates the simulation box side lengths given the
    simulation box area or volume. This function assumes that the
    simulation box is either a square or a cube (for two-dimensional or
    three-dimensional networks, respectively).

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        A_or_V (float): Simulation box area or volume.

    Returns:
        np.ndarray: 1D np.ndarray with dim float entries of the
        simulation box side lengths.
    
    """
    return np.repeat(np.power(A_or_V, np.reciprocal(1.0*dim)), dim)

def L_arg_rho_func(dim: int, en: float, rho: float) -> np.ndarray:
    """Simulation box side lengths.
    
    This function calculates the simulation box side lengths given the
    number of particles and the particle number density. This function
    assumes that the simulation box is either a square or a cube (for
    two-dimensional or three-dimensional networks, respectively).

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        en (float): Number of particles.
        rho (float): Particle number density.

    Returns:
        np.ndarray: 1D np.ndarray with dim float entries of the
        simulation box side lengths.
    
    """
    return L_arg_A_or_V_func(dim, A_or_V_arg_rho_func(en, rho))

def L_arg_eta_func(dim: int, b: float, en: float, eta: float) -> np.ndarray:
    """Simulation box side lengths.
    
    This function calculates the simulation box side lengths given the
    number of particles and the particle packing density. This function
    assumes that the simulation box is either a square or a cube (for
    two-dimensional or three-dimensional networks, respectively).

    Args:
        dim (int): Physical dimensionality of the network; either 2 or 3 (for two-dimensional or three-dimensional networks).
        b (float): Particle diameter.
        en (float): Number of particles.
        eta (float): Particle packing density.

    Returns:
        np.ndarray: 1D np.ndarray with dim float entries of the
        simulation box side lengths.
    
    """
    return L_arg_A_or_V_func(dim, A_or_V_arg_eta_func(dim, b, en, eta))

def mic_func(coords: np.ndarray, L: np.ndarray) -> np.ndarray:
    """Minimum image criterion.
    
    This function modifies a provided coordinates np.ndarray to satisfy
    the minimum image criterion within a simulation box defined by side
    lengths L.

    Args:
        coords (np.ndarray): 2D np.ndarray with (en, dim) float entries of coordinates.
        L (np.ndarray): 1D np.ndarray with dim float entries of the simulation box side lengths.

    Returns:
        np.ndarray: Coordinates that satisfy the minimum image
        criterion.
    
    """
    coords = np.where(coords<0, coords+L, coords)
    coords = np.where(coords>=L, coords-L, coords)
    return coords

def L_min_func(L: np.ndarray) -> float:
    """Minimum simulation box length.
    
    This function calculates the minimum simulation box length.

    Args:
        L (np.ndarray): 1D np.ndarray with dim float entries of the simulation box side lengths.

    Returns:
        float: Minimum simulation box length.
    
    """
    return np.min(L)

def L_max_func(L: np.ndarray) -> float:
    """Maximum simulation box length.
    
    This function calculates the maximum simulation box length.

    Args:
        L (np.ndarray): 1D np.ndarray with dim float entries of the simulation box side lengths.

    Returns:
        float: Maximum simulation box length.
    
    """
    return np.max(L)

def L_diag_max_func(L: np.ndarray) -> float:
    """Maximum simulation box diagonal length.
    
    This function calculates the maximum simulation box diagonal length.

    Args:
        L (np.ndarray): 1D np.ndarray with dim float entries of the simulation box side lengths.

    Returns:
        float: Maximum simulation box diagonal length.
    
    """
    return np.sqrt(np.sum(L**2))