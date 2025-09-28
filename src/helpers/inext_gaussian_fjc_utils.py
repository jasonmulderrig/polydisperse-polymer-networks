import numpy as np

def s_cn_inext_gaussian_fjc_func(
        gamma: np.ndarray | float) -> np.ndarray | float:
    """Gaussian end-to-end distance polymer chain conformation
    probability distribution.

    This function calculates the Gaussian end-to-end distance polymer
    chain conformation probability for a chain with a given number of
    segments and a given end-to-end distance.

    Args:
        r (np.ndarray | float | int): End-to-end chain distance.
        b (float): Chain segment and/or cross-linker diameter.
    
    Note: If nu is an np.ndarray, then r must be a float or int.
    Likewise, if r is an np.ndarray, then nu must be a float or int.
    
    Returns:
        np.ndarray | float: Gaussian end-to-end distance polymer chain
        conformation probability (distribution).
    
    """
    return 3. / 2. * gamma**2

def s_c_inext_gaussian_fjc_func(
        gamma: np.ndarray | float,
        n: np.ndarray | float | int) -> np.ndarray | float:
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
    return n * s_cn_inext_gaussian_fjc_func(gamma)

# w_c_inext_gaussian_fjc_vary_gamma_func()
# w_c_inext_gaussian_fjc_vary_n_func()

def w_c_inext_gaussian_fjc_func(
        gamma: np.ndarray | float,
        n: np.ndarray | float | int) -> np.ndarray | float:
    return s_c_inext_gaussian_fjc_func(gamma, n)