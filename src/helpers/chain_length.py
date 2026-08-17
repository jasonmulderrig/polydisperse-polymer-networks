import numpy as np
import numpy.typing as npt

def r_func(
        gamma: npt.ArrayLike,
        n: npt.ArrayLike,
        b: npt.ArrayLike) -> npt.ArrayLike:
    """End-to-end chain distance/length.

    This function calculates the end-to-end chain distance/length from
    absolute/equilibrium chain stretch.

    Args:
        gamma (npt.ArrayLike): Absolute/equilibrium chain stretch.
        n (npt.ArrayLike): Number of segments in the chain.
        b (npt.ArrayLike): Chain segment and/or cross-linker diameter.
    
    Returns:
        npt.ArrayLike: End-to-end chain distance/length.
    
    """
    return gamma * n * b