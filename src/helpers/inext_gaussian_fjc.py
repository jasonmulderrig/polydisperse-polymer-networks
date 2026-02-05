import numpy as np
import numpy.typing as npt

def gamma_crit_func(kappa_n: float, zeta_n_char: float) -> float:
    """Critical absolute/equilibrium chain stretch.

    This function returns the critical absolute/equilibrium chain
    stretch for the inextensible Gaussian FJC model.

    Args:
        gamma_crit (float): Prescribed critical absolute/equilibrium chain stretch.
    
    Returns:
        float: Critical absolute/equilibrium chain stretch compliant
        with the restrictions of the inextensible Gaussian FJC model.
    
    """
    return np.inf

def s_cn_func(gamma: npt.ArrayLike) -> npt.ArrayLike:
    """Nondimensional chain-level entropic free energy contribution
    per segment.
    
    This function returns the nondimensional chain-level entropic
    free energy contribution per segment.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional chain-level entropic free
        energy contribution per segment.
    
    """
    return 3. / 2. * gamma**2

def s_c_func(
        gamma: npt.ArrayLike,
        n: npt.ArrayLike) -> npt.ArrayLike:
    """Nondimensional chain entropic free energy contribution.
    
    This function returns the nondimensional chain entropic free energy
    contribution.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        n (npt.ArrayLike): Number of chain segments.
    
    Returns:
        npt.ArrayLike: Nondimensional chain entropic free energy
        contribution.
    
    """
    return n * s_cn_func(gamma)

def w_cn_func(gamma: npt.ArrayLike) -> npt.ArrayLike:
    """Nondimensional chain-level Helmholtz free energy per segment.
    
    This function returns the nondimensional chain-level Helmholtz free
    energy per segment.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional chain-level Helmholtz free
        energy per segment.
    
    """
    return s_cn_func(gamma)

def w_c_func(
        gamma: npt.ArrayLike,
        n: npt.ArrayLike) -> npt.ArrayLike:
    """Nondimensional chain-level Helmholtz free energy.
    
    This function returns the nondimensional chain-level Helmholtz free
    energy.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        n (npt.ArrayLike): Number of chain segments.
    
    Returns:
        npt.ArrayLike: Nondimensional chain-level Helmholtz free
        energy.
    
    """
    return s_c_func(gamma, n)

def xi_c_func(gamma: npt.ArrayLike) -> npt.ArrayLike:
    """Nondimensional chain force.
    
    This function returns the nondimensional chain force.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
    
    Returns:
        npt.ArrayLike: Nondimensional chain force.
    
    """
    return 3. * gamma

def xi_c_vec_func(
        gamma_vec: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Nondimensional chain force vector.
    
    This function returns the nondimensional chain force vector.

    Args:
        gamma_vec (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch vector.
    
    Returns:
        npt.NDArray[np.floating]: Nondimensional chain force vector.
    
    """
    return xi_c_func(gamma_vec)

def dw_c__dy_clnk_func(
        gamma_vec: npt.NDArray[np.floating],
        gamma: float) -> npt.NDArray[np.floating]:
    """Nondimensional derivative of the polymer chain free energy with
    respect to the cross-link junction position for a chain in the
    cross-link structure RVE.
     
    This function returns the nondimensional derivative of the polymer
    chain free energy with respect to the cross-link junction position
    for a chain in the cross-link structure RVE.

    Args:
        gamma_vec (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch vector.
        gamma (float): Absolute/Equilibrium chain stretch.
    
    Returns:
        npt.NDArray[np.floating]: Nondimensional derivative of the
        polymer chain free energy with respect to the cross-link
        junction position for a chain in the cross-link structure RVE.
    
    """
    return -xi_c_vec_func(gamma_vec)

def d2w_c__dy_clnk_dy_clnk_func(
        gamma_vec: npt.NDArray[np.floating],
        gamma: float,
        n: float | int) -> npt.NDArray[np.floating]:
    """Nondimensional second derivative of the polymer chain free energy
    with respect to the cross-link junction position for a chain in the
    cross-link structure RVE.
     
    This function returns the nondimensional second derivative of the
    polymer chain free energy with respect to the cross-link junction
    position for a chain in the cross-link structure RVE.

    Args:
        gamma_vec (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch vector.
        gamma (float): Absolute/Equilibrium chain stretch.
        n (float | int): Number of chain segments.
    
    Returns:
        npt.NDArray[np.floating]: Nondimensional second derivative of
        the polymer chain free energy with respect to the cross-link
        junction position for a chain in the cross-link structure RVE.
    
    """
    return 3. / n * np.eye(3)