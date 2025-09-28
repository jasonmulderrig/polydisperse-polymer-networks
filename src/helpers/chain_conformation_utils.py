import numpy as np

def gamma_func(
        r: np.ndarray | float | int,
        n: np.ndarray | float | int,
        b: float) -> np.ndarray | float:
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
    
    Returns:
        np.ndarray | float: Gaussian end-to-end distance polymer chain
        conformation probability (distribution).
    
    """
    return r / (n*b)

def r_func(
        gamma: np.ndarray | float | int,
        n: np.ndarray | float | int,
        b: float) -> np.ndarray | float:
    """Gaussian end-to-end distance polymer chain conformation
    probability distribution.

    This function calculates the Gaussian end-to-end distance polymer
    chain conformation probability for a chain with a given number of
    segments and a given end-to-end distance.

    Args:
        gamma (np.ndarray | float | int): End-to-end chain distance.
        n (np.ndarray | float | int): Number of segments in the chain.
        b (float): Chain segment and/or cross-linker diameter.
    
    Note: If nu is an np.ndarray, then gamma must be a float or int.
    Likewise, if gamma is an np.ndarray, then nu must be a float or int.
    
    Returns:
        np.ndarray | float: Gaussian end-to-end distance polymer chain
        conformation probability (distribution).
    
    """
    return gamma * n * b