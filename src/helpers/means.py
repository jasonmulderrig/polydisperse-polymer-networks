import numpy as np
import numpy.typing as npt

def mean_func(arr: npt.NDArray) -> float:
    """Arithmetic mean.

    This function returns the arithmetic mean of an array.

    Args:
        arr (npt.NDArray): Array.
    
    Returns:
        float: Arithmetic mean.
    
    """
    return np.mean(arr)

def geo_mean_func(arr: npt.NDArray) -> float:
    """Geometric mean.

    This function returns the geometric mean of an array.

    Args:
        arr (npt.NDArray): Array.
    
    Returns:
        float: Geometric mean.
    
    """
    return np.power(np.prod(arr), 1./np.shape(arr)[0])