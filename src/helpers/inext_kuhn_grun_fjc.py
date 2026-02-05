import numpy as np
import numpy.typing as npt
from src.helpers.inv_langevin import (
    s_cn_inv_langevin_fjc_func,
    inv_langevin_func,
    inv_langevin_prime_func
)

def gamma_crit_func(kappa_n: float, zeta_n_char: float) -> float:
    """Critical absolute/equilibrium chain stretch.

    This function returns the critical absolute/equilibrium chain
    stretch for the inextensible Kuhn-Grun FJC model.

    Args:
        gamma_crit (float): Prescribed critical absolute/equilibrium chain stretch.
    
    Returns:
        float: Critical absolute/equilibrium chain stretch compliant
        with the restrictions of the inextensible Kuhn-Grun FJC model.
    
    """
    return 1.-1.e-5

def s_cn_func(
        gamma: npt.ArrayLike,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional chain-level entropic free energy contribution
    per segment.
    
    This function returns the nondimensional chain-level entropic
    free energy contribution per segment.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional chain-level entropic free
        energy contribution per segment.
    
    """
    gamma_isscalar = np.isscalar(gamma)
    if gamma_isscalar: gamma = np.asarray([gamma])
    s_cn = np.empty_like(gamma)

    for indx in np.ndindex(np.shape(gamma)):
        gamma_val = gamma[indx]
        if gamma_val >= gamma_crit: s_cn[indx] = np.inf
        else: s_cn[indx] = s_cn_inv_langevin_fjc_func(gamma_val)

    return s_cn.item() if gamma_isscalar else s_cn

def s_c_func(
        gamma: npt.ArrayLike,
        n: npt.ArrayLike,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional chain-level entropic free energy contribution.
    
    This function returns the nondimensional chain-level entropic
    free energy contribution.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        n (npt.ArrayLike): Number of chain segments.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional chain-level entropic free
        energy contribution.
    
    """
    return n * s_cn_func(gamma, gamma_crit)

def w_cn_func(
        gamma: npt.ArrayLike,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional chain-level Helmholtz free energy per segment.
    
    This function returns the nondimensional chain-level Helmholtz free
    energy per segment.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional chain-level Helmholtz free
        energy per segment.
    
    """
    return s_cn_func(gamma, gamma_crit)

def w_c_func(
        gamma: npt.ArrayLike,
        n: npt.ArrayLike,
        gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional chain-level Helmholtz free energy.
    
    This function returns the nondimensional chain-level Helmholtz free
    energy.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        n (npt.ArrayLike): Number of chain segments.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional chain-level Helmholtz free
        energy.
    
    """
    return s_c_func(gamma, n, gamma_crit)

def xi_c_func(gamma: npt.ArrayLike, gamma_crit: float) -> npt.ArrayLike:
    """Nondimensional chain force.
    
    This function returns the nondimensional chain force.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional chain force.
    
    """
    gamma_isscalar = np.isscalar(gamma)
    if gamma_isscalar: gamma = np.asarray([gamma])
    xi_c = np.empty_like(gamma)

    for indx in np.ndindex(np.shape(gamma)):
        gamma_val = gamma[indx]
        if gamma_val >= gamma_crit: xi_c[indx] = np.inf
        else: xi_c[indx] = inv_langevin_func(gamma_val)

    return xi_c.item() if gamma_isscalar else xi_c

def xi_c_vec_func(
        gamma_vec: npt.NDArray[np.floating],
        gamma: float,
        gamma_crit: float) -> npt.NDArray[np.floating]:
    """Nondimensional chain force vector.
    
    This function returns the nondimensional chain force vector.

    Args:
        gamma_vec (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch vector.
        gamma (float): Absolute/Equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.NDArray[np.floating]: Nondimensional chain force vector.
    
    """
    return xi_c_func(gamma, gamma_crit) * gamma_vec / gamma

def dw_c__dy_clnk_func(
        gamma_vec: npt.NDArray[np.floating],
        gamma: float,
        gamma_crit: float) -> npt.NDArray[np.floating]:
    """Nondimensional derivative of the polymer chain free energy with
    respect to the cross-link junction position for a chain in the
    cross-link structure RVE.
     
    This function returns the nondimensional derivative of the polymer
    chain free energy with respect to the cross-link junction position
    for a chain in the cross-link structure RVE.

    Args:
        gamma_vec (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch vector.
        gamma (float): Absolute/Equilibrium chain stretch.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.NDArray[np.floating]: Nondimensional derivative of the
        polymer chain free energy with respect to the cross-link
        junction position for a chain in the cross-link structure RVE.
    
    """
    return -xi_c_vec_func(gamma_vec, gamma, gamma_crit)

def d2w_c__dy_clnk_dy_clnk_func(
        gamma_vec: npt.NDArray[np.floating],
        gamma: float,
        n: float | int,
        gamma_crit: float) -> npt.NDArray[np.floating]:
    """Nondimensional second derivative of the polymer chain free energy
    with respect to the cross-link junction position for a chain in the
    cross-link structure RVE.
     
    This function returns the nondimensional second derivative of the
    polymer chain free energy with respect to the cross-link junction
    position for a chain in the cross-link structure RVE.

    Args:
        gamma_vec (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch vector.
        gamma (float): Absolute/Equilibrium chain stretch.
        n (float | int): Number of chain segments.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
    
    Returns:
        npt.NDArray[np.floating]: Nondimensional second derivative of
        the polymer chain free energy with respect to the cross-link
        junction position for a chain in the cross-link structure RVE.
    
    """
    if gamma >= gamma_crit:
        inv_langevin_prime_val = np.inf
        inv_langevin_val = np.inf
    else:
        inv_langevin_prime_val = inv_langevin_prime_func(gamma)
        inv_langevin_val = inv_langevin_func(gamma)
    unit_gamma_vec = gamma_vec / gamma
    unit_gamma_vec_outer_prod = np.outer(unit_gamma_vec, unit_gamma_vec)
    return (
        (inv_langevin_prime_val*unit_gamma_vec_outer_prod+inv_langevin_val/gamma*(np.eye(3)-unit_gamma_vec_outer_prod))
        / n
    )