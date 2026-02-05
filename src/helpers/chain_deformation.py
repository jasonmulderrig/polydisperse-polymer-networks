import numpy as np
import numpy.typing as npt
from src.helpers.chain_stretch import gamma_func

def r_chn_vec_func(
        ideal_numerics_form: bool,
        F: npt.NDArray[np.floating],
        X_chn: npt.NDArray[np.floating],
        Q_clnk: npt.NDArray[np.floating],
        y_clnk: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """End-to-end chain vector for a chain in the cross-link structure
    RVE.

    This function calculates the end-to-end chain vector for a chain in
    the cross-link structure RVE.

    Args:
        ideal_numerics_form (bool): Boolean indicating if the end-to-end chain vector is calculated with respect to the ideal numerics form (if True) or the original form (if False). 
        F (npt.NDArray[np.floating]): Deformation gradient.
        X_chn (npt.NDArray[np.floating]): Initial chain end position for a chain in the cross-link structure RVE.
        Q_clnk (npt.NDArray[np.floating]): Cross-link rotation.
        y_clnk (npt.NDArray[np.floating]): Cross-link junction position.
    
    Returns:
        npt.NDArray[np.floating]: End-to-end chain vector for a chain in
        the cross-link structure RVE.
    
    """
    r_chn = np.matmul(F, np.matmul(Q_clnk, X_chn))
    if ideal_numerics_form: r_chn -= np.matmul(Q_clnk, y_clnk) 
    else: r_chn -= y_clnk
    return r_chn

def r_chn_approx_vec_func(
        F_Lmbda: npt.NDArray[np.floating],
        X_chn: npt.NDArray[np.floating],
        Q_clnk_m: npt.NDArray[np.floating],
        y_clnk_m: npt.NDArray[np.floating],
        delta_Q_clnk: npt.NDArray[np.floating],
        delta_y_clnk: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """End-to-end chain vector approximation for a chain in the
    cross-link structure RVE.

    This function calculates the end-to-end chain vector approximation
    for a chain in the cross-link structure RVE.

    Args:
        F_Lmbda (npt.NDArray[np.floating]): Deformation gradient, assuming incompressibility, of the form F_Lmbda = diag(Lmbda_1, Lmbda_2, 1/(Lmbda_1*Lmbda_2)).
        X_chn (npt.NDArray[np.floating]): Initial chain end position for a chain in the cross-link structure RVE.
        Q_clnk_m (npt.NDArray[np.floating]): Cross-link rotation for the monodisperse cross-link structure RVE.
        y_clnk_m (npt.NDArray[np.floating]): Cross-link junction position for the monodisperse cross-link structure RVE.
        delta_Q_clnk (npt.NDArray[np.floating]): Cross-link rotation perturbation.
        delta_y_clnk (npt.NDArray[np.floating]): Cross-link junction position perturbation.
    
    Returns:
        npt.NDArray[np.floating]: End-to-end chain vector approximation
        for a chain in the cross-link structure RVE.
    
    """
    return (
        np.matmul(F_Lmbda, np.matmul(delta_Q_clnk, np.matmul(Q_clnk_m, X_chn)))
        - (y_clnk_m+delta_y_clnk)
    )

def r_chn_func(
        ideal_numerics_form: bool,
        F: npt.NDArray[np.floating],
        X_chn: npt.NDArray[np.floating],
        Q_clnk: npt.NDArray[np.floating],
        y_clnk: npt.NDArray[np.floating]) -> float:
    """End-to-end chain distance/length for a chain in the cross-link
    structure RVE.

    This function calculates the end-to-end chain distance/length for a
    chain in the cross-link structure RVE.

    Args:
        ideal_numerics_form (bool): Boolean indicating if the end-to-end chain vector is calculated with respect to the ideal numerics form (if True) or the original form (if False). 
        F (npt.NDArray[np.floating]): Deformation gradient.
        X_chn (npt.NDArray[np.floating]): Initial chain end position for a chain in the cross-link structure RVE.
        Q_clnk (npt.NDArray[np.floating]): Cross-link rotation.
        y_clnk (npt.NDArray[np.floating]): Cross-link junction position.
    
    Returns:
        float: End-to-end chain distance/length for a chain in the
        cross-link structure RVE.
    
    """
    return (
        np.linalg.norm(
            r_chn_vec_func(ideal_numerics_form, F, X_chn, Q_clnk, y_clnk))
    )

def r_chn_approx_func(
        F_Lmbda: npt.NDArray[np.floating],
        X_chn: npt.NDArray[np.floating],
        Q_clnk_m: npt.NDArray[np.floating],
        y_clnk_m: npt.NDArray[np.floating],
        delta_Q_clnk: npt.NDArray[np.floating],
        delta_y_clnk: npt.NDArray[np.floating]) -> float:
    """End-to-end chain distance/length approximation for a chain in the
    cross-link structure RVE.

    This function calculates the end-to-end chain distance/length
    approximation for a chain in the cross-link structure RVE.

    Args:
        F_Lmbda (npt.NDArray[np.floating]): Deformation gradient, assuming incompressibility, of the form F_Lmbda = diag(Lmbda_1, Lmbda_2, 1/(Lmbda_1*Lmbda_2)).
        X_chn (npt.NDArray[np.floating]): Initial chain end position for a chain in the cross-link structure RVE.
        Q_clnk_m (npt.NDArray[np.floating]): Cross-link rotation for the monodisperse cross-link structure RVE.
        y_clnk_m (npt.NDArray[np.floating]): Cross-link junction position for the monodisperse cross-link structure RVE.
        delta_Q_clnk (npt.NDArray[np.floating]): Cross-link rotation perturbation.
        delta_y_clnk (npt.NDArray[np.floating]): Cross-link junction position perturbation.
    
    Returns:
        float: End-to-end chain distance/length approximation for a
        chain in the cross-link structure RVE.
    
    """
    return (
        np.linalg.norm(r_chn_approx_vec_func(
            F_Lmbda, X_chn, Q_clnk_m, y_clnk_m, delta_Q_clnk, delta_y_clnk))
    )

def gamma_chn_vec_func(
        ideal_numerics_form: bool,
        F: npt.NDArray[np.floating],
        X_chn: npt.NDArray[np.floating],
        Q_clnk: npt.NDArray[np.floating],
        y_clnk: npt.NDArray[np.floating],
        n_chn: float | int,
        b: float) -> npt.NDArray[np.floating]:
    """Absolute/Equilibrium chain stretch vector for a chain in the
    cross-link structure RVE.

    This function calculates the absolute/equilibrium chain stretch
    vector for a chain in the cross-link structure RVE.

    Args:
        ideal_numerics_form (bool): Boolean indicating if the end-to-end chain vector is calculated with respect to the ideal numerics form (if True) or the original form (if False). 
        F (npt.NDArray[np.floating]): Deformation gradient.
        X_chn (npt.NDArray[np.floating]): Initial chain end position for a chain in the cross-link structure RVE.
        Q_clnk (npt.NDArray[np.floating]): Cross-link rotation.
        y_clnk (npt.NDArray[np.floating]): Cross-link junction position.
        n_chn (float | int): Number of chain segments.
        b (float): Chain segment and/or cross-linker diameter.
    
    Returns:
        npt.NDArray[np.floating]: Absolute/Equilibrium chain stretch
        vector for a chain in the cross-link structure RVE.
    
    """
    return (
        gamma_func(
            r_chn_vec_func(ideal_numerics_form, F, X_chn, Q_clnk, y_clnk),
            n_chn, b)
    )

def gamma_chn_approx_vec_func(
        F_Lmbda: npt.NDArray[np.floating],
        X_chn: npt.NDArray[np.floating],
        Q_clnk_m: npt.NDArray[np.floating],
        y_clnk_m: npt.NDArray[np.floating],
        delta_Q_clnk: npt.NDArray[np.floating],
        delta_y_clnk: npt.NDArray[np.floating],
        n_chn: float | int,
        b: float) -> npt.NDArray[np.floating]:
    """Absolute/Equilibrium chain stretch vector approximation for a
    chain in the cross-link structure RVE.

    This function calculates the absolute/equilibrium chain stretch
    vector approximation for a chain in the cross-link structure RVE.

    Args:
        F_Lmbda (npt.NDArray[np.floating]): Deformation gradient, assuming incompressibility, of the form F_Lmbda = diag(Lmbda_1, Lmbda_2, 1/(Lmbda_1*Lmbda_2)).
        X_chn (npt.NDArray[np.floating]): Initial chain end position for a chain in the cross-link structure RVE.
        Q_clnk_m (npt.NDArray[np.floating]): Cross-link rotation for the monodisperse cross-link structure RVE.
        y_clnk_m (npt.NDArray[np.floating]): Cross-link junction position for the monodisperse cross-link structure RVE.
        delta_Q_clnk (npt.NDArray[np.floating]): Cross-link rotation perturbation.
        delta_y_clnk (npt.NDArray[np.floating]): Cross-link junction position perturbation.
        n_chn (float | int): Number of chain segments.
        b (float): Chain segment and/or cross-linker diameter.
    
    Returns:
        npt.NDArray[np.floating]: Absolute/Equilibrium chain stretch
        vector approximation for a chain in the cross-link structure
        RVE.
    
    """
    return (
        gamma_func(
            r_chn_approx_vec_func(F_Lmbda, X_chn, Q_clnk_m, y_clnk_m, delta_Q_clnk, delta_y_clnk),
            n_chn, b)
    )

def gamma_chn_func(
        ideal_numerics_form: bool,
        F: npt.NDArray[np.floating],
        X_chn: npt.NDArray[np.floating],
        Q_clnk: npt.NDArray[np.floating],
        y_clnk: npt.NDArray[np.floating],
        n_chn: float | int,
        b: float) -> float:
    """Absolute/Equilibrium chain stretch for a chain in the cross-link
    structure RVE.

    This function calculates the absolute/equilibrium chain stretch for
    a chain in the cross-link structure RVE.

    Args:
        ideal_numerics_form (bool): Boolean indicating if the end-to-end chain vector is calculated with respect to the ideal numerics form (if True) or the original form (if False). 
        F (npt.NDArray[np.floating]): Deformation gradient.
        X_chn (npt.NDArray[np.floating]): Initial chain end position for a chain in the cross-link structure RVE.
        Q_clnk (npt.NDArray[np.floating]): Cross-link rotation.
        y_clnk (npt.NDArray[np.floating]): Cross-link junction position.
        n_chn (float | int): Number of chain segments.
        b (float): Chain segment and/or cross-linker diameter.
    
    Returns:
        float: Absolute/Equilibrium chain stretch for a chain in the
        cross-link structure RVE.
    
    """
    return (
        gamma_func(
            r_chn_func(ideal_numerics_form, F, X_chn, Q_clnk, y_clnk), n_chn, b)
    )

def gamma_chn_approx_func(
        F_Lmbda: npt.NDArray[np.floating],
        X_chn: npt.NDArray[np.floating],
        Q_clnk_m: npt.NDArray[np.floating],
        y_clnk_m: npt.NDArray[np.floating],
        delta_Q_clnk: npt.NDArray[np.floating],
        delta_y_clnk: npt.NDArray[np.floating],
        n_chn: float | int,
        b: float) -> float:
    """Absolute/Equilibrium chain stretch approximation for a chain in
    the cross-link structure RVE.

    This function calculates the absolute/equilibrium chain stretch
    approximation for a chain in the cross-link structure RVE.

    Args:
        F_Lmbda (npt.NDArray[np.floating]): Deformation gradient, assuming incompressibility, of the form F_Lmbda = diag(Lmbda_1, Lmbda_2, 1/(Lmbda_1*Lmbda_2)).
        X_chn (npt.NDArray[np.floating]): Initial chain end position for a chain in the cross-link structure RVE.
        Q_clnk_m (npt.NDArray[np.floating]): Cross-link rotation for the monodisperse cross-link structure RVE.
        y_clnk_m (npt.NDArray[np.floating]): Cross-link junction position for the monodisperse cross-link structure RVE.
        delta_Q_clnk (npt.NDArray[np.floating]): Cross-link rotation perturbation.
        delta_y_clnk (npt.NDArray[np.floating]): Cross-link junction position perturbation.
        n_chn (float | int): Number of chain segments.
        b (float): Chain segment and/or cross-linker diameter.
    
    Returns:
        float: Absolute/Equilibrium chain stretch approximation for a
        chain in the cross-link structure RVE.
    
    """
    return (
        gamma_func(
            r_chn_approx_func(F_Lmbda, X_chn, Q_clnk_m, y_clnk_m, delta_Q_clnk, delta_y_clnk),
            n_chn, b)
    )

def w_chn_func(
        gamma: float,
        n: float | int,
        w_c_func,
        w_c_args: tuple[float] | tuple[None],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float] | tuple[None]) -> float:
    """Nondimensional polymer chain free energy for a chain in the
    cross-link structure RVE.

    This function calculates the nondimensional polymer chain free
    energy for a chain in the cross-link structure RVE.

    Args:
        gamma (float): Absolute/Equilibrium chain stretch for a chain in the cross-link structure RVE.
        n (float | int): Number of chain segments.
        w_c_func (function): Nondimensional polymer chain free energy function.
        w_c_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        w_c_dfrmtn_func (function): Nondimensional polymer chain deformation free energy function.
        w_c_dfrmtn_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
    
    Returns:
        float: Nondimensional polymer chain free energy for a chain in
        the cross-link structure RVE.
    
    """
    return (
        w_c_func(gamma, n, *w_c_args)
        + w_c_dfrmtn_func(gamma, n, *w_c_dfrmtn_args)
    )