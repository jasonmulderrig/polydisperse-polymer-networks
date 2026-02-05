import numpy as np
import numpy.typing as npt

def p_boltzmann_func(e: npt.ArrayLike) -> npt.ArrayLike:
    """Boltzmann thermodynamic connection formula to yield probability
    from energy.

    This function calculates probability from energy via the Boltzmann
    thermodynamic connection formula.

    Args:
        e (npt.ArrayLike): Energy.
    
    Returns:
        npt.ArrayLike: Probability.
    
    """
    return np.exp(-e)

def e_boltzmann_func(p: npt.ArrayLike) -> npt.ArrayLike:
    """Boltzmann thermodynamic connection formula to yield energy from
    probability.

    This function calculates energy from probability via the Boltzmann
    thermodynamic connection formula.

    Args:
        p (npt.ArrayLike): Probability.
    
    Returns:
        npt.ArrayLike: Energy.
    
    """
    return -np.log(p)