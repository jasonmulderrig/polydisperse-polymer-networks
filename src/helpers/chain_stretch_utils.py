import numpy as np
from src.helpers.inext_kuhn_grun_fjc_utils import (
    gamma_crit_inext_kuhn_grun_fjc_func
)
from src.helpers.cufjc_utils import gamma_crit_cufjc_func
from src.helpers.chain_conformation_dispersity_utils import p_gamma_func

def master_gamma_crit_func(w_c_dist: str, w_c_args: tuple[float]) -> float:
    """Master polymer chain free energy function.

    This function returns the selected polymer chain free energy
    function.

    Args:
        w_c_dist (str): Short-hand name for the selected polymer chain free energy function.
    
    Returns:
        function: The selected polymer chain free energy function.
    
    """
    if w_c_dist == "inext_gaussian_fjc": gamma_crit = np.inf
    elif w_c_dist == "inext_kuhn_grun_fjc":
        gamma_crit = gamma_crit_inext_kuhn_grun_fjc_func(w_c_args[0])
    elif w_c_dist == "cufjc": gamma_crit = gamma_crit_cufjc_func(*w_c_args)
    else:
        error_str = (
            "The called-for polymer chain free energy function is not"
            + "implemented!"
        )
        raise NotImplementedError(error_str)
    return gamma_crit

def J_func(gamma_init, gamma_crit):
    return (gamma_crit-gamma_init) / 2.

def gamma_point_func(gamma_init, gamma_crit, point):
    J = J_func(gamma_init, gamma_crit)
    return J * (1.0+point) + gamma_init

def master_gamma_rms_func(
        points: np.ndarray,
        weights: np.ndarray,
        gamma_crit: float,
        n: float | int,
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
    if w_c_dist == "inext_gaussian_fjc":
        return 1. / np.sqrt(n) # r_rms = np.sqrt(n) * b
    else:
        # Sort points in ascending order
        sort_indcs = np.argsort(points)
        points = points[sort_indcs]
        weights = weights[sort_indcs]

        # Jacobian for the master space-chain configuration space
        # transformation
        J = J_func(0.0, gamma_crit)

        # Chain stretches corresponding to the master space points for
        # the initial chain configuration
        gamma_0_points = gamma_point_func(0.0, gamma_crit, points)
        p_gamma = p_gamma_func(gamma_0_points, n, w_c_func, w_c_args)

        I_0_intgrnd = gamma_0_points**2 * p_gamma
        I_2_intgrnd = gamma_0_points**4 * p_gamma

        I_0 = np.sum(weights*I_0_intgrnd) * J
        I_2 = np.sum(weights*I_2_intgrnd) * J

        if w_c_dist == "inext_kuhn_grun_fjc": return np.sqrt(I_2/I_0)
        # The below equation for the cufjc is not correct since the full
        # implementation is not executed. Fill in the full
        # implementation later, which will involve calculating
        # epsilon_n_diss explicitly
        elif w_c_dist == "cufjc": return np.sqrt(I_2/I_0)
        else:
            error_str = (
                "The called-for polymer chain free energy function is "
                + "not implemented!"
            )
            raise NotImplementedError(error_str)