import numpy as np
from src.helpers.chain_conformation_utils import (
    gamma_func,
    r_func
)
from src.helpers.chain_stretch_utils import (
    master_gamma_crit_func,
    master_gamma_rms_func
)

def master_r_crit_func(
        n: float | int,
        b: float,
        w_c_dist: str,
        w_c_args: tuple[float]) -> float:
    """Master polymer chain free energy function.

    This function returns the selected polymer chain free energy
    function.

    Args:
        w_c_dist (str): Short-hand name for the selected polymer chain free energy function.
    
    Returns:
        function: The selected polymer chain free energy function.
    
    """
    gamma_crit = master_gamma_crit_func(w_c_dist, w_c_args)
    if gamma_crit == np.inf: return np.inf
    else: return r_func(gamma_crit, n, b)

def master_r_rms_func(
        points: np.ndarray,
        weights: np.ndarray,
        r_crit: float,
        n: float | int,
        b: float,
        w_c_dist: str,
        w_c_func,
        w_c_args: tuple[float]) -> float:
    """Master polymer chain free energy function.

    This function returns the selected polymer chain free energy
    function.

    Args:
        w_c_dist (str): Short-hand name for the selected polymer chain free energy function.
    
    Returns:
        function: The selected polymer chain free energy function.
    
    """
    return (
        r_func(
            master_gamma_rms_func(
                points, weights, gamma_func(r_crit, n, b), n, w_c_dist, w_c_func, w_c_args),
            n, b)
    )