import numpy as np
import numpy.typing as npt
from src.helpers.boltzmann import p_boltzmann_func

def gamma_func(
        r: npt.ArrayLike,
        n: npt.ArrayLike,
        b: npt.ArrayLike) -> npt.ArrayLike:
    """Absolute/Equilibrium chain stretch.

    This function calculates the absolute/equilibrium chain stretch from
    the end-to-end chain distance/length.

    Args:
        r (npt.ArrayLike): End-to-end chain distance/length.
        n (npt.ArrayLike): Number of chain segments.
        b (npt.ArrayLike): Chain segment and/or cross-linker diameter.
    
    Returns:
        npt.ArrayLike: Absolute/Equilibrium chain stretch.
    
    """
    return r / (n*b)

def master_gamma_crits_func(
        w_c_dist: str,
        kappa_n: float,
        zeta_n_char: float) -> tuple[float] | tuple[None]:
    """Master critical/fundamental absolute/equilibrium chain stretches.

    This function calculates and returns various critical/fundamental
    absolute/equilibrium chain stretches based off of the specified
    polymer chain model.

    Args:
        w_c_dist (str): Short-hand name for the selected nondimensional polymer chain free energy function, i.e., the selected polymer chain model.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        tuple[float] | tuple[None]: Critical/Fundamental
        absolute/equilibrium chain stretches.
    
    """
    if w_c_dist == "inext_gaussian_fjc": return tuple([])
    elif w_c_dist == "inext_kuhn_grun_fjc":
        from src.helpers.inext_kuhn_grun_fjc import gamma_crit_func
        return tuple([gamma_crit_func(kappa_n, zeta_n_char)])
    elif w_c_dist == "cufjc":
        from src.helpers.cufjc import (
            gamma_pade_to_bergstrom_crit_func,
            gamma_crit_func,
            gamma_n_pade_to_bergstrom_crit_func,
            gamma_n_crit_func
        )
        gamma_pade_to_bergstrom_crit = gamma_pade_to_bergstrom_crit_func(kappa_n)
        gamma_crit = gamma_crit_func(kappa_n, zeta_n_char)
        gamma_n_pade_to_bergstrom_crit = gamma_n_pade_to_bergstrom_crit_func(
            kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit, gamma_crit)
        gamma_n_crit = gamma_n_crit_func(kappa_n, zeta_n_char)
        gamma_crits = (
            [
                gamma_n_pade_to_bergstrom_crit, gamma_n_crit,
                gamma_pade_to_bergstrom_crit, gamma_crit
            ]
        )
        return tuple(gamma_crits)
    else:
        error_str = "The called-for polymer chain model is not implemented!"
        raise NotImplementedError(error_str)

def master_gamma_crit_func(
        w_c_dist: str, kappa_n: float, zeta_n_char: float) -> float:
    """Master critical absolute/equilibrium chain stretch.

    This function calculates and returns the critical
    absolute/equilibrium chain stretch based off of the specified
    polymer chain model.

    Args:
        w_c_dist (str): Short-hand name for the selected nondimensional polymer chain free energy function, i.e., the selected polymer chain model.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
    
    Returns:
        float: Critical/fundamental absolute/equilibrium chain stretch.
    
    """
    if w_c_dist == "inext_gaussian_fjc":
        from src.helpers.inext_gaussian_fjc import gamma_crit_func
    elif w_c_dist == "inext_kuhn_grun_fjc":
        from src.helpers.inext_kuhn_grun_fjc import gamma_crit_func
    elif w_c_dist == "cufjc":
        from src.helpers.cufjc import gamma_crit_func
    else:
        error_str = "The called-for polymer chain model is not implemented!"
        raise NotImplementedError(error_str)
    return gamma_crit_func(kappa_n, zeta_n_char)

def master_gamma_rms_args_func(
        w_c_dist: str,
        kappa_n: float,
        zeta_n_char: float,
        gamma_crits: tuple[float] | tuple[None]) -> tuple[float] | tuple[None]:
    """Master reference/root-mean-square absolute/equilibrium chain
    stretch function arguments.

    This function returns the arguments needed for the calculation of
    the reference/root-mean-square absolute/equilibrium chain stretch
    based off of the specified polymer chain model.

    Args:
        w_c_dist (str): Short-hand name for the selected nondimensional polymer chain free energy function, i.e., the selected polymer chain model.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_crits (tuple[float] | tuple[None]): Critical/Fundamental absolute/equilibrium chain stretches.
    
    Returns:
        tuple[float] | tuple[None]: The arguments needed for the
        reference/root-mean-square absolute/equilibrium chain stretch
        based off of the specified polymer chain model.
    
    """
    if w_c_dist == "inext_gaussian_fjc": return tuple([])
    elif w_c_dist == "inext_kuhn_grun_fjc": return tuple([])
    elif w_c_dist == "cufjc": return tuple([kappa_n, zeta_n_char, *gamma_crits])
    else:
        error_str = "The called-for polymer chain model is not implemented!"
        raise NotImplementedError(error_str)

def master_gamma_rms_func(
        points: npt.NDArray[np.float64],
        weights: npt.NDArray[np.float64],
        n: float,
        gamma_crit: float,
        gamma_n_hat_inc: float,
        w_c_dist: str,
        w_c_func,
        w_c_args: tuple[float] | tuple[None],
        gamma_rms_args: tuple[float] | tuple[None]) -> float:
    """Master reference/root-mean-square absolute/equilibrium chain
    stretch.

    This function calculates and returns the reference/root-mean-square
    absolute/equilibrium chain stretch based off of the specified
    polymer chain model.

    Args:
        points (npt.NDArray[np.float64]): Sample points for Gauss-Legendre quadrature used for numerically integrating various moments of the initial (intact) chain configuration equilibrium probability density distribution.
        weights (npt.NDArray[np.float64]): Weights for each sample point for Gauss-Legendre quadrature used for numerically integrating various moments of the initial (intact) chain configuration equilibrium probability density distribution.
        n (float): Number of chain segments.
        gamma_crit (float): Critical absolute/equilibrium chain stretch.
        gamma_n_hat_inc (float): Applied segment stretch increment (for the calculation of the nondimensional rate-independent critical dissipated segment scission energy).
        w_c_dist (str): Short-hand name for the selected nondimensional polymer chain free energy function, i.e., the selected polymer chain model.
        w_c_func (function): Nondimensional polymer chain free energy function.
        w_c_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        gamma_rms_args (tuple[float] | tuple[None]): The arguments needed for the reference/root-mean-square absolute/equilibrium chain stretch based off of the specified polymer chain model.
    
    Returns:
        float: Reference/Root-mean-square absolute/equilibrium chain
        stretch.
    
    """
    if w_c_dist == "inext_gaussian_fjc":
        return 1. / np.sqrt(n) # r_rms = np.sqrt(n) * b
    elif w_c_dist == "inext_kuhn_grun_fjc":
        def Jac_func(x_max: float, x_min: float) -> float:
            """Scalar Jacobian for a transformation between two
            one-dimensional coordinate spaces.
            
            This function computes the scalar Jacobian for a
            transformation between two one-dimensional coordinate
            spaces.

            Args:
                x_max (float): Maximum coordinate.
                x_min (float): Minimum coordinate.
            
            Returns:
                float: Scalar Jacobian.
            
            """
            return (x_max-x_min) / 2.
        
        def gamma_points_func(
                points: npt.ArrayLike,
                gamma_crit: float) -> npt.ArrayLike:
            """Absolute/Equilibrium chain stretch as a function of
            master space coordinate points.

            This function computes the absolute/equilibrium chain
            stretch as a function of master space coordinate points.

            Args:
                points (npt.ArrayLike): Master space coordinate points.
                gamma_crit (float): Critical abolute/equilibrium chain stretch.
            
            Returns:
                npt.ArrayLike: Absolute/Equilibrium chain stretch as a
                function of master space coordinate points.
            
            """
            Jac = Jac_func(gamma_crit, 0.)
            return Jac * (1.+points) + 0.
        
        # Sort points in ascending order
        sort_indcs = np.argsort(points)
        points = points[sort_indcs]
        weights = weights[sort_indcs]

        # Jacobian for the master space-to-equilibrium chain
        # configuration space transformation
        Jac = Jac_func(gamma_crit, 0.)

        # Absolute/Equilibrium chain stretches corresponding to the
        # master space points for the initial chain configuration
        gamma_0_points = gamma_points_func(points, gamma_crit)

        # Integrand involved in the equilibrium chain configuration
        # partition function integration
        Z = p_boltzmann_func(w_c_func(gamma_0_points, n, *w_c_args))
        
        # Integrands of the zeroth moment and second moment of the
        # initial chain configuration equilibrium probability density
        # distribution (without normalization)
        I_0_intrgrnd = Z * gamma_0_points**2
        I_2_intrgrnd = Z * gamma_0_points**4
        
        # Zeroth moment and second moment of the initial chain
        # configuration equilibrium probability density distribution
        # (without normalization)
        I_0 = np.sum(np.multiply(weights, I_0_intrgrnd)) * Jac
        I_2 = np.sum(np.multiply(weights, I_2_intrgrnd)) * Jac

        return np.sqrt(I_2/I_0)
    elif w_c_dist == "cufjc":
        from src.helpers.cufjc import A_n_func
        return A_n_func(points, weights, n, gamma_n_hat_inc, *gamma_rms_args)
    else:
        error_str = "The called-for polymer chain model is not implemented!"
        raise NotImplementedError(error_str)