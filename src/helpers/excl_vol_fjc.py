import numpy as np
import numpy.typing as npt

def w_c_log_barrier_clnk_vol_pnlty_func(
        gamma: npt.ArrayLike,
        n: float | int) -> npt.ArrayLike:
    """Nondimensional logarithmic barrier chain-level free energy that
    penalizes chain ends from overlapping in space (and thus penalizes
    excluded volume interactions).
    
    This function returns the nondimensional logarithmic barrier
    chain-level free energy.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        n (float | int): Number of chain segments.
    
    Returns:
        npt.ArrayLike: Nondimensional logarithmic barrier chain-level
        free energy that penalizes chain ends from overlapping in space
        (and thus penalizes excluded volume interactions).
    
    """
    gamma_isscalar = np.isscalar(gamma)
    if gamma_isscalar: gamma = np.asarray([gamma])
    w_c_log_barrier = np.empty_like(gamma)

    for indx in np.ndindex(np.shape(gamma)):
        gamma_val = gamma[indx]
        if n * gamma_val > 2.: w_c_log_barrier[indx] = 0.
        elif n * gamma_val > 1.: w_c_log_barrier[indx] = -np.log(n*gamma_val-1)
        else: w_c_log_barrier[indx] = np.inf

    return w_c_log_barrier.item() if gamma_isscalar else w_c_log_barrier