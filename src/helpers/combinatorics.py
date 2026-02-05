import numpy as np
import numpy.typing as npt
from scipy.special import factorial, binom

def indcs_permutations(k_num: int) -> npt.NDArray[np.integer]:
    """Permutations of indices [0, 1, ..., k_num-1].
    
    This function generates all permutations of indices
    [0, 1, ..., k_num-1].

    Args:
        k_num (int): Number of indices. Ideally, k_num <= 8.
    
    Returns:
        npt.NDArray[np.integer]: Permutations of indices
        [0, 1, ..., k_num-1].
    """
    indcs = np.arange(k_num, dtype=int)
    fctrl = factorial(k_num, exact=True)

    indcs_prmttns = np.empty((fctrl, k_num), dtype=int)
    fctrls = np.asarray(
        [factorial(k, exact=True) for k in range(k_num)], dtype=int)
    flipped_nonzero_fctrls = np.flip(fctrls[1:])

    for indx in range(fctrl):
        elmnts = indcs.copy()
        code = []
        idx = indx
        for f in flipped_nonzero_fctrls:
            q, idx = np.divmod(idx, f)
            code.append(q)
        code.append(0)
        indcs_prmttn = []
        for c in code:
            indcs_prmttn.append(elmnts[c])
            elmnts = np.delete(elmnts, c)
        indcs_prmttns[indx] = np.asarray(indcs_prmttn, dtype=int)
    
    return indcs_prmttns

def multinomial_coeff(m: npt.NDArray[np.integer]) -> int:
    """Multinomial coefficient
    
    This function calculates the multinomial coefficient for a
    collection of integers.

    Args:
        m (npt.NDArray[np.integer]): Non-negative integers
    
    Returns:
        int: Multinomial coefficient.
    
    """
    M = np.sum(m)
    return factorial(M, exact=True) // np.prod(factorial(m, exact=True))

def permutations_with_replacement(
        num_dist_objs: int,
        num_elems: int) -> npt.NDArray[np.integer]:
    """All permutations of how to arrange a number of distinguishable
    objects over a number of elements with replacement.
    
    This function deduces all permutations of how to arrange a number of
    distinguishable objects over a number of elements with replacement.

    Args:
        num_dist_objs (int): Number of distinguishable objects.
        num_elems (int): Number of elements.
    
    Returns:
        npt.NDArray[np.integer]: All permutations of how to arrange a
        number of distinguishable objects over a number of elements with
        replacement.
    
    """
    return (
        np.transpose(np.indices((num_dist_objs,)*num_elems).reshape(num_elems, -1))
    )

def num_combinations_with_replacement(num_dist_objs: int, num_elems: int) -> int:
    """Number of ways to combine a sampled number of distinguishable
    objects over a number of elements with replacement.
    
    This function calculates the Number of ways to combine a sampled
    number of distinguishable objects over a number of elements with
    replacement.

    Args:
        num_dist_objs (int): Number of distinguishable objects.
        num_elems (int): Number of elements.
    
    Returns:
        int: Number of ways to combine a sampled number of
        distinguishable objects over a number of elements with
        replacement.
    
    """
    return int(binom(num_dist_objs+num_elems-1, num_elems))

def indist_balls_in_dist_empty_bins_combinations(
        num_indist_balls: int,
        num_dist_empty_bins: int) -> npt.NDArray[np.integer]:
    """All combinations of how to arrange a number of indistinguishable
    balls in a number of distinguishable bins, allowing for empty bins.
    
    This function deduces all combinations of how to arrange a number of
    indistinguishable balls in a number of distinguishable bins,
    allowing for empty bins.

    Args:
        num_indist_balls (int): Number of indistinguishable balls.
        num_dist_empty_bins (int): Number of distinguishable bins that are each permitted to be empty.
    
    Returns:
        npt.NDArray[np.integer]: All combinations of how to arrange a
        number of indistinguishable balls in a number of distinguishable
        bins, allowing for empty bins.
    
    """
    grid = permutations_with_replacement(num_indist_balls+1, num_dist_empty_bins)
    return grid[np.sum(grid, axis=1) == num_indist_balls]