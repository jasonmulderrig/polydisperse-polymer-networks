import numpy as np

def w_c_log_barrier_clnk_vol_pnlty_func(
        gamma: np.ndarray | float,
        n: float | int) -> np.ndarray | float:
    gamma = np.asarray([gamma])
    w_c_log_barrier = np.empty_like(gamma)

    for indx in np.ndindex(np.shape(gamma)):
        gamma_val = gamma[indx]

        if n * gamma_val > 2.: w_c_log_barrier[indx] = 0.
        elif n * gamma_val > 1.: w_c_log_barrier[indx] = -np.log(n*gamma_val-1)
        else: w_c_log_barrier[indx] = np.inf
    
    if np.shape(w_c_log_barrier) == (1,): return w_c_log_barrier[0]
    else: return w_c_log_barrier