import numpy as np
import numpy.typing as npt

def w_c_zero_func(gamma: npt.ArrayLike, n: float) -> npt.ArrayLike:
    """Always-zero nondimensional chain-level free energy.
    
    This function always returns a zero nondimensional chain-level free
    energy.

    Args:
        gamma (npt.ArrayLike): Absolute/Equilibrium chain stretch.
        n (float): Number of chain segments.
    
    Returns:
        npt.ArrayLike: Always-zero nondimensional chain-level free
        energy.
    
    """
    if np.isscalar(gamma): return 0.
    else: return np.zeros_like(gamma)

def master_w_c_func(w_c_dist: str):
    """Master nondimensional polymer chain free energy function.

    This function returns the selected nondimensional polymer chain free
    energy function.

    Args:
        w_c_dist (str): Short-hand name for the selected nondimensional polymer chain free energy function, i.e., the selected polymer chain model.
    
    Returns:
        function: The selected nondimensional polymer chain free energy
        function.
    
    """
    if w_c_dist == "inext_gaussian_fjc":
        from src.helpers.inext_gaussian_fjc import w_c_func
        return w_c_func
    elif w_c_dist == "inext_kuhn_grun_fjc":
        from src.helpers.inext_kuhn_grun_fjc import w_c_func
        return w_c_func
    elif w_c_dist == "cufjc":
        from src.helpers.cufjc import w_c_func
        return w_c_func
    else:
        error_str = (
            "The called-for polymer chain free energy function, i.e., "
            + "the polymer chain model, is not implemented!"
        )
        raise NotImplementedError(error_str)

def master_w_c_args_func(
        w_c_dist: str,
        kappa_n: float,
        zeta_n_char: float,
        gamma_crits: tuple[float] | tuple[None]) -> tuple[float] | tuple[None]:
    """Master nondimensional polymer chain free energy function
    arguments.

    This function returns the arguments needed for the selected
    nondimensional polymer chain free energy function (beyond the
    absolute/equilibrium chain stretch gamma and the number of chain
    segments n).

    Args:
        w_c_dist (str): Short-hand name for the selected nondimensional polymer chain free energy function, i.e., the selected polymer chain model.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_crits (tuple[float] | tuple[None]): Critical/Fundamental absolute/equilibrium chain stretches.
    
    Returns:
        tuple[float] | tuple[None]: The arguments needed for the
        nondimensional polymer chain free energy function (beyond the
        absolute/equilibrium chain stretch gamma and the number of chain
        segments n).
    
    """
    if w_c_dist == "inext_gaussian_fjc": return tuple([])
    elif w_c_dist == "inext_kuhn_grun_fjc": return gamma_crits # gamma_crits = gamma_crit
    elif w_c_dist == "cufjc":
        _, gamma_n_crit, gamma_pade_to_bergstrom_crit, gamma_crit = gamma_crits
        w_c_args = (
            [
                kappa_n, zeta_n_char, gamma_n_crit,
                gamma_pade_to_bergstrom_crit, gamma_crit
            ]
        )
        return tuple(w_c_args)
    else:
        error_str = (
            "The called-for polymer chain free energy function, i.e., "
            + "the polymer chain model, is not implemented!"
        )
        raise NotImplementedError(error_str)

def master_w_c_dfrmtn_func(w_c_dfrmtn_dist: str):
    """Master nondimensional polymer chain deformation free energy
    function.

    This function returns the selected nondimensional polymer chain
    deformation free energy function, which aims to introduce an
    energetic penalty for excluded volume interactions, particularly
    when chain ends overlap in space.

    Args:
        w_c_dfrmtn_dist (str): Short-hand name for the selected nondimensional polymer chain deformation free energy function.
    
    Returns:
        function: The selected nondimensional polymer chain deformation
        free energy function.
    
    """
    if w_c_dfrmtn_dist == "log_barrier_clnk_vol_pnlty":
        from src.helpers.excl_vol_fjc import w_c_log_barrier_clnk_vol_pnlty_func
        return w_c_log_barrier_clnk_vol_pnlty_func
    else: return w_c_zero_func

def master_w_c_dfrmtn_args_func(
        w_c_dfrmtn_dist: str) -> tuple[float] | tuple[None]:
    """Master nondimensional polymer chain deformation free energy
    function arguments.

    This function returns the arguments needed for the selected
    nondimensional polymer chain deformation free energy function.

    Args:
        w_c_dfrmtn_dist (str): Short-hand name for the selected nondimensional polymer chain deformation free energy function.
    
    Returns:
        tuple[float] | tuple[None]: The arguments needed for the
        selected nondimensional polymer chain deformation free energy
        function (beyond the absolute/equilibrium chain stretch gamma
        and the number of chain segments n).
    
    """
    return tuple([])

def master_dw_c__dy_clnk_func(w_c_dist: str):
    """Master nondimensional derivative of the polymer chain free energy
    with respect to the cross-link junction position function.

    This function returns the selected nondimensional derivative of the
    polymer chain free energy with respect to the cross-link junction
    position function.

    Args:
        w_c_dist (str): Short-hand name for the selected nondimensional derivative of the polymer chain free energy with respect to the cross-link junction position function, i.e., the selected polymer chain model.
    
    Returns:
        function: The selected nondimensional derivative of the polymer
        chain free energy with respect to the cross-link junction
        position function.
    
    """
    if w_c_dist == "inext_gaussian_fjc":
        from src.helpers.inext_gaussian_fjc import dw_c__dy_clnk_func
        return dw_c__dy_clnk_func
    elif w_c_dist == "inext_kuhn_grun_fjc":
        from src.helpers.inext_kuhn_grun_fjc import dw_c__dy_clnk_func
        return dw_c__dy_clnk_func
    elif w_c_dist == "cufjc":
        from src.helpers.cufjc import dw_c__dy_clnk_func
        return dw_c__dy_clnk_func
    else:
        error_str = "The called-for polymer chain model is not implemented!"
        raise NotImplementedError(error_str)

def master_dw_c__dy_clnk_args_func(
        w_c_dist: str,
        kappa_n: float,
        zeta_n_char: float,
        gamma_crits: tuple[float] | tuple[None]) -> tuple[float] | tuple[None]:
    """Master nondimensional derivative of the polymer chain free energy
    with respect to the cross-link junction position function arguments.

    This function returns the arguments needed for the selected
    nondimensional derivative of the polymer chain free energy with
    respect to the cross-link junction position function (beyond the 
    absolute/equilibrium chain stretch vector gamma_vec and the
    absolute/equilibrium chain stretch gamma).

    Args:
        w_c_dist (str): Short-hand name for the selected nondimensional derivative of the polymer chain free energy with respect to the cross-link junction position function, i.e., the selected polymer chain model.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_crits (tuple[float] | tuple[None]): Critical/Fundamental absolute/equilibrium chain stretches.
    
    Returns:
        tuple[float] | tuple[None]: The arguments needed for the
        selected nondimensional derivative of the polymer chain free
        energy with respect to the cross-link junction position function
        (beyond the absolute/equilibrium chain stretch vector gamma_vec
        and the absolute/equilibrium chain stretch gamma).
    
    """
    if w_c_dist == "inext_gaussian_fjc": return tuple([])
    elif w_c_dist == "inext_kuhn_grun_fjc": return gamma_crits # gamma_crits = gamma_crit
    elif w_c_dist == "cufjc":
        _, _, gamma_pade_to_bergstrom_crit, gamma_crit = gamma_crits
        dw_c__dy_clnk_args = (
            [kappa_n, zeta_n_char, gamma_pade_to_bergstrom_crit, gamma_crit]
        )
        return tuple(dw_c__dy_clnk_args)
    else:
        error_str = "The called-for polymer chain model is not implemented!"
        raise NotImplementedError(error_str)

def master_d2w_c__dy_clnk_dy_clnk_func(w_c_dist: str):
    """Master nondimensional second derivative of the polymer chain free
    energy with respect to the cross-link junction position function.

    This function returns the selected nondimensional second derivative
    of the polymer chain free energy with respect to the cross-link
    junction position function.

    Args:
        w_c_dist (str): Short-hand name for the selected nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function, i.e., the selected polymer chain model.
    
    Returns:
        function: The selected nondimensional second derivative of the
        polymer chain free energy with respect to the cross-link
        junction position function.
    
    """
    if w_c_dist == "inext_gaussian_fjc":
        from src.helpers.inext_gaussian_fjc import d2w_c__dy_clnk_dy_clnk_func
        return d2w_c__dy_clnk_dy_clnk_func
    elif w_c_dist == "inext_kuhn_grun_fjc":
        from src.helpers.inext_kuhn_grun_fjc import d2w_c__dy_clnk_dy_clnk_func
        return d2w_c__dy_clnk_dy_clnk_func
    elif w_c_dist == "cufjc":
        from src.helpers.cufjc import d2w_c__dy_clnk_dy_clnk_func
        return d2w_c__dy_clnk_dy_clnk_func
    else:
        error_str = "The called-for polymer chain model is not implemented!"
        raise NotImplementedError(error_str)

def master_d2w_c__dy_clnk_dy_clnk_args_func(
        w_c_dist: str,
        kappa_n: float,
        zeta_n_char: float,
        gamma_crits: tuple[float] | tuple[None]) -> tuple[float] | tuple[None]:
    """Master nondimensional second derivative of the polymer chain free
    energy with respect to the cross-link junction position function
    arguments.

    This function returns the arguments needed for the selected
    nondimensional second derivative of the polymer chain free energy
    with respect to the cross-link junction position function (beyond
    the absolute/equilibrium chain stretch vector gamma_vec, the
    absolute/equilibrium chain stretch gamma, and the number of chain
    segments n).

    Args:
        w_c_dist (str): Short-hand name for the selected nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function, i.e., the selected polymer chain model.
        kappa_n (float): Nondimensional segment stiffness.
        zeta_n_char (float): Nondimensional characteristic segment potential energy scale.
        gamma_crits (tuple[float] | tuple[None]): Critical/Fundamental absolute/equilibrium chain stretches.
    
    Returns:
        tuple[float] | tuple[None]: The arguments needed for the
        selected nondimensional second derivative of the polymer chain
        free energy with respect to the cross-link junction position
        function (beyond the absolute/equilibrium chain stretch vector
        gamma_vec, the absolute/equilibrium chain stretch gamma, and the
        number of chain segments n).
    
    """
    return (
        master_dw_c__dy_clnk_args_func(
            w_c_dist, kappa_n, zeta_n_char, gamma_crits)
    )