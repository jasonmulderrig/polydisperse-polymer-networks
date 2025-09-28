import numpy as np
from scipy.special import binom, factorial

def indcs_permutations(k_num: int) -> np.ndarray:
    """Permutations of indices [0, 1, ..., k_num-1].
    
    This function generates all permutations of indices
    [0, 1, ..., k_num-1].

    Args:
        k_num (int): Number of indices. Ideally, k_num <= 8.
    
    Returns:
        np.ndarray: Permutations of indices [0, 1, ..., k_num-1].
    """
    indcs = np.arange(k_num, dtype=int)
    fctrl = factorial(k_num, exact=True)

    indcs_prmttns = np.empty((fctrl, k_num), dtype=int)
    fctrls = np.asarray([factorial(k, exact=True) for k in range(k_num)], dtype=int)
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

def C_R_init_func(k: np.ndarray, N: int) -> np.ndarray:
    """Number of distinct cross-link structures, with symmetry
    equivalence, for a cross-link of degree k.

    This function calculates the number of distinct cross-link
    structures, with symmetry equivalence, for a cross-link of degree k.

    Args:
        k (np.ndarray): Cross-link degree.
        N (int): Number of salient polymer chain segment numbers.
    
    Returns:
        np.ndarray: Number of distinct cross-link structures, with
        symmetry equivalence, for a cross-link of degree k.
    
    """
    k_num = np.shape(k)[0]
    C_R = np.empty(k_num, dtype=int)
    for k_indx in range(k_num):
        C_R[k_indx] = int(binom(N+k[k_indx]-1, k[k_indx]))
    return C_R

def m_clnks_init_func(k: np.ndarray, N: int) -> list[np.ndarray]:
    """Chain segment number multiplicity for each distinct cross-link
    structure (with symmetry equivalence).

    This function computes the chain segment number multiplicity for
    each distinct cross-link structure (with symmetry equivalence).

    Args:
        k (np.ndarray): Cross-link degree.
        N (int): Number of salient polymer chain segment numbers.
    
    Returns:
        list[np.ndarray]: Chain segment number multiplicity for each
        distinct cross-link structure (with symmetry equivalence).
    
    """
    m_clnks = []
    for k_indx in range(np.shape(k)[0]):
        grid = np.transpose(np.indices((k[k_indx]+1,)*N).reshape(N, -1))
        m_clnks.append(grid[np.sum(grid, axis=1) == k[k_indx]])
    return m_clnks

def n_clnks_init_func(
        n: np.ndarray,
        m_clnks: list[np.ndarray]) -> list[np.ndarray]:
    """Chain segment number for each chain in each distinct cross-link
    structure (with symmetry equivalence).

    This function tabulates the chain segment number for each chain in
    each distinct cross-link structure (with symmetry equivalence).

    Args:
        n (np.ndarray): Salient chain segment numbers (sorted from least to greatest).
        m_clnks (list[np.ndarray]): Chain segment number multiplicity for each distinct cross-link structure (with symmetry equivalence).
    
    Returns:
        list[np.ndarray]: Chain segment number for each chain in each
        distinct cross-link structure (with symmetry equivalence)
        (sorted from least to greatest for each cross-link structure).
    
    """
    for k_indx in range(len(m_clnks)):
        if np.shape(n)[0] != np.shape(m_clnks[k_indx])[1]:
            error_str = (
                "The number of segments represented in the segment "
                + "number multiplicity array does not equal the "
                + "provided number of segments."
            )
            raise ValueError(error_str)
    
    n_clnks = []
    for k_indx in range(len(m_clnks)):
        n_clnks.append(
            np.vstack(
                [np.repeat(n, m_clnk) for m_clnk in m_clnks[k_indx]]))
    return n_clnks

def C_clnks_init_func(m_clnks: list[np.ndarray]) -> list[np.ndarray]:
    """Number of permutations that exist for each distinct cross-link
    structure due to symmetry equivalence.

    This function calculates the number of permutations that exist for
    each distinct cross-link structure due to symmetry equivalence.

    Args:
        m_clnks (list[np.ndarray]): Chain segment number multiplicity for each distinct cross-link structure (with symmetry equivalence).
    
    Returns:
        list[np.ndarray]: Number of permutations that exist for each
        distinct cross-link structure due to symmetry equivalence.
    
    """
    def multinomial_coeff(m: np.ndarray) -> int:
        """Multinomial coefficient
        
        This function calculates the multinomial coefficient for a
        collection of integers.

        Args:
            m (np.ndarray): Non-negative integers
        
        Returns:
            int: Multinomial coefficient.
        
        """
        M = np.sum(m)
        return factorial(M, exact=True) // np.prod(factorial(m, exact=True))
    
    C_clnks = []
    for k_indx in range(len(m_clnks)):
        C_clnks.append(
            np.hstack(
                [multinomial_coeff(m_clnk) for m_clnk in m_clnks[k_indx]]))
    return C_clnks

def p_n_k_clnks_init_func(
        C_clnks: list[np.ndarray],
        p_n: np.ndarray,
        m_clnks: list[np.ndarray]) -> list[np.ndarray]:
    """Probability distribution of distinct cross-link structures (with
    symmetry equivalence) with degree k.

    This function calculates the probability distribution of distinct
    cross-link structures (with symmetry equivalence) with degree k.

    Args:
        C_clnks: (list[np.ndarray]): Number of permutations that exist for each distinct cross-link structure due to symmetry equivalence.
        p_n (np.ndarray): Polymer chain segment number probability distribution.
        m_clnks: (list[np.ndarray]): Chain segment number multiplicity for each distinct cross-link structure (with symmetry equivalence).
    
    Returns:
        list[np.ndarray]: Probability distribution of distinct
        cross-link structures (with symmetry equivalence) with degree k.
    
    """
    if len(C_clnks) != len(m_clnks):
        error_str = (
            "The cross-link structures represented by C_clnks and "
            + "m_clnks are not compatible with one another."
        )
        raise ValueError(error_str)
    k_num = len(C_clnks)
    for k_indx in range(k_num):
        if np.shape(C_clnks[k_indx])[0] != np.shape(m_clnks[k_indx])[0]:
            error_str = (
                "The number of cross-link structures represented by "
                + "C_clnks and m_clnks are not compatible with one "
                + "another."
            )
            raise ValueError(error_str)
    for k_indx in range(k_num):
        if np.shape(p_n)[0] != np.shape(m_clnks[k_indx])[1]:
            error_str = (
                "The number of segments represented in the segment "
                + "number multiplicity array does not equal the "
                + "provided number of segments in the chain segment "
                + "number probability distribution."
            )
            raise ValueError(error_str)

    p_n_k_clnks = []
    for k_indx in range(k_num):
        C_R = np.shape(C_clnks[k_indx])[0]
        p_n_k_clnks_k_vals = np.empty(C_R)
        for clnk_indx in range(C_R):
            p_n_k_clnks_k_vals[clnk_indx] = (
                C_clnks[k_indx][clnk_indx]
                * np.prod(np.power(p_n, m_clnks[k_indx][clnk_indx]))
            )
        p_n_k_clnks.append(p_n_k_clnks_k_vals)
    return p_n_k_clnks

def p_clnks_init_func(
        p_k_f: np.ndarray,
        p_n_k_clnks: list[np.ndarray]) -> list[np.ndarray]:
    """Probability distribution of distinct elastically-effective
    cross-link structures (with symmetry equivalence) with degree k.

    This function calculates the probability distribution of distinct
    elastically-effective cross-link structures (with symmetry
    equivalence) with degree k.

    Args:
        p_k_f (np.ndarray): Probability distribution of elastically-effective cross-linkers with degree k.
        p_n_k_clnks (list[np.ndarray]): Probability distribution of distinct cross-link structures (with symmetry equivalence) with degree k.
    
    Returns:
        list[np.ndarray]: Probability distribution of distinct
        elastically-effective cross-link structures (with symmetry
        equivalence) with degree k.
    
    """
    if np.shape(p_k_f)[0] != len(p_n_k_clnks):
        error_str = (
            "The cross-link structures represented by C_clnks and "
            + "m_clnks are not compatible with one another."
        )
        raise ValueError(error_str)
    
    p_clnks = []
    for k_indx in range(np.shape(p_k_f)[0]):
        p_clnks.append(
            np.hstack([p_k_f[k_indx] * p_n_k_clnk for p_n_k_clnk in p_n_k_clnks[k_indx]]))
    return p_clnks