import numpy as np
from src.helpers.inv_langevin_func_utils import s_cn_inv_langevin_fjc_func

def gamma_crit_inext_kuhn_grun_fjc_func(gamma_crit: float) -> float:
    if gamma_crit >= 1.: gamma_crit = 0.999
    return gamma_crit

def s_cn_inext_kuhn_grun_fjc_func(
        gamma: np.ndarray | float, gamma_crit: float) -> np.ndarray | float:
    """Nondimensional chain-level entropic free energy contribution
    per segment as calculated by the Jedynak R[9,2] inverse Langevin
    approximate.
        
    This function computes the nondimensional chain-level entropic
    free energy contribution per segment as calculated by the
    Jedynak R[9,2] inverse Langevin approximate as a function of the
    result of the equilibrium chain stretch minus the segment
    stretch plus one.
    
    """
    gamma = np.asarray([gamma])
    s_cn = np.empty_like(gamma)

    for indx in np.ndindex(np.shape(gamma)):
        gamma_val = gamma[indx]

        if gamma_val >= gamma_crit: s_cn[indx] = np.inf
        else: s_cn[indx] = s_cn_inv_langevin_fjc_func(gamma_val)
    
    if np.shape(s_cn) == (1,): return s_cn[0]
    else: return s_cn

def s_c_inext_kuhn_grun_fjc_func(
        gamma: np.ndarray | float,
        n: np.ndarray | float | int,
        gamma_crit: float) -> np.ndarray | float:
    """Gaussian end-to-end distance polymer chain conformation
    probability distribution.

    This function calculates the Gaussian end-to-end distance polymer
    chain conformation probability for a chain with a given number of
    segments and a given end-to-end distance.

    Args:
        r (np.ndarray | float | int): End-to-end chain distance.
        n (np.ndarray | float | int): Number of segments in the chain.
        b (float): Chain segment and/or cross-linker diameter.
    
    Note: If nu is an np.ndarray, then r must be a float or int.
    Likewise, if r is an np.ndarray, then nu must be a float or int.

    May need a s_cn_args: tuple[float] later
    
    Returns:
        np.ndarray | float: Gaussian end-to-end distance polymer chain
        conformation probability (distribution).
    
    """
    return n * s_cn_inext_kuhn_grun_fjc_func(gamma, gamma_crit)

def w_c_inext_kuhn_grun_fjc_func(
        gamma: np.ndarray | float,
        n: np.ndarray | float | int,
        gamma_crit: float) -> np.ndarray | float:
    return s_c_inext_kuhn_grun_fjc_func(gamma, n, gamma_crit)