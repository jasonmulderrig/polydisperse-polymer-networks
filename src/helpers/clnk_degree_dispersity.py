import numpy as np
import numpy.typing as npt
from pylimer_tools.calc.miller_macosko_theory import (
    compute_miller_macosko_probabilities,
    compute_probability_that_crosslink_is_effective
)

def k_init_func(f: int) -> npt.NDArray[np.int64]:
    """Elastically-effective cross-link degree.

    This function gathers the degree of all elastically-effective
    cross-links that exist in a polymer network.

    Args:
        f (int): Maximum cross-linker degree/functionality.
    
    Returns:
        npt.NDArray[np.int64]: Degree of all elastically-effective
        cross-links that exist in a polymer network.
    
    """
    return np.arange(3, f+1, dtype=int)

def p_finite_clnkr_out_func(chi: float, xi: float, f: int) -> float:
    """Probability that a randomly chosen cross-link is the start of a
    finite chain, i.e., that a randomly chosen cross-link is
    elastically-effective.

    This function calculates the probability that a randomly chosen
    cross-link is the start of a finite chain, i.e., that a randomly
    chosen cross-link is elastically-effective. This calculation is done
    using Miller-Macosko theory.

    Args:
        chi (float): Stoichiometric imbalance between the number of cross-linker sites and the number of chain ends.
        xi (float): Extent of the cross-linking reaction.
        f (int): Maximum cross-linker degree/functionality.
   
    Returns:
        float: Probability that a randomly chosen cross-link is the
        start of a finite chain, i.e., that a randomly chosen cross-link
        is elastically-effective.

    """
    p_finite_clnkr_out, _ = compute_miller_macosko_probabilities(chi, xi, f)
    return p_finite_clnkr_out

def p_k_clnks_init_func(
        f: int,
        k: npt.NDArray[np.int64],
        p_finite_clnkr_out: float) -> npt.NDArray[np.float64]:
    """Probability distribution that a randomly chosen cross-link with
    maximum cross-linker degree/functionality f is an
    elastically-effective cross-link with degree k.

    This function calculates the probability distribution that a
    randomly chosen cross-link with maximum cross-linker
    degree/functionality f is an elastically-effective cross-link with
    degree k. This calculation is done using Miller-Macosko theory.

    Args:
        f (int): Maximum cross-linker degree/functionality.
        k (npt.NDArray[np.int64]): Elastically-effective cross-link degree.
        p_finite_clnkr_out (float): Probability that a randomly chosen cross-link is the start of a finite chain, i.e., that a randomly chosen cross-link is elastically-effective.
   
    Returns:
        npt.NDArray[np.float64]: Probability distribution that a
        randomly chosen cross-link with maximum cross-linker
        degree/functionality f is an elastically-effective cross-link
        with degree k.

    """
    assert k[-1] == f
    k_num = np.shape(k)[0]
    p_k_clnks = np.empty(k_num)
    for k_indx in range(k_num):
        p_k_clnks[k_indx] = compute_probability_that_crosslink_is_effective(
            f, k[k_indx], p_finite_clnkr_out)
    return p_k_clnks