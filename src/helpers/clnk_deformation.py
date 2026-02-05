import numpy as np
import numpy.typing as npt
from src.helpers.chain_deformation import (
    r_chn_vec_func,
    r_chn_approx_vec_func,
    r_chn_func,
    r_chn_approx_func,
    gamma_chn_vec_func,
    gamma_chn_approx_vec_func,
    gamma_chn_func,
    gamma_chn_approx_func,
    w_chn_func
)

def monodisperse_y_clnk() -> npt.NDArray[np.floating]:
    """Cross-link junction position for the monodisperse cross-link
    structure RVE, i.e., the origin.

    This function supplies the cross-link junction position for the
    monodisperse cross-link structure RVE, i.e., the origin.

    Returns:
        npt.ArrayLike: Cross-link junction position for the monodisperse
        cross-link structure RVE, i.e., the origin.
    
    """
    return np.zeros(3)

def r_vec_clnk_func(
        ideal_numerics_form: bool,
        F: npt.NDArray[np.floating],
        X_clnk: npt.NDArray[np.floating],
        Q_clnk: npt.NDArray[np.floating],
        y_clnk: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """End-to-end chain vector for each chain in the cross-link
    structure RVE.

    This function calculates the end-to-end chain vector for each chain
    in the cross-link structure RVE.

    Args:
        ideal_numerics_form (bool): Boolean indicating if the end-to-end chain vector is calculated with respect to the ideal numerics form (if True) or the original form (if False). 
        F (npt.NDArray[np.floating]): Deformation gradient.
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        Q_clnk (npt.NDArray[np.floating]): Cross-link rotation.
        y_clnk (npt.NDArray[np.floating]): Cross-link junction position.
    
    Returns:
        npt.NDArray[np.floating]: End-to-end chain vector for each chain
        in the cross-link structure RVE.
    
    """
    k_num = np.shape(X_clnk)[0]
    r_vec_clnk = np.empty_like(X_clnk)
    for chn_indx in range(k_num):
        r_vec_clnk[chn_indx] = r_chn_vec_func(
            ideal_numerics_form, F, X_clnk[chn_indx], Q_clnk, y_clnk)
    return r_vec_clnk

def r_approx_vec_clnk_func(
        F_Lmbda: npt.NDArray[np.floating],
        X_clnk: npt.NDArray[np.floating],
        Q_clnk_m: npt.NDArray[np.floating],
        y_clnk_m: npt.NDArray[np.floating],
        delta_Q_clnk: npt.NDArray[np.floating],
        delta_y_clnk: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """End-to-end chain vector approximation for each chain in the
    cross-link structure RVE.

    This function calculates the end-to-end chain vector approximation
    for each chain in the cross-link structure RVE.

    Args:
        F_Lmbda (npt.NDArray[np.floating]): Deformation gradient, assuming incompressibility, of the form F_Lmbda = diag(Lmbda_1, Lmbda_2, 1/(Lmbda_1*Lmbda_2)).
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        Q_clnk_m (npt.NDArray[np.floating]): Cross-link rotation for the monodisperse cross-link structure RVE.
        y_clnk_m (npt.NDArray[np.floating]): Cross-link junction position for the monodisperse cross-link structure RVE.
        delta_Q_clnk (npt.NDArray[np.floating]): Cross-link rotation perturbation.
        delta_y_clnk (npt.NDArray[np.floating]): Cross-link junction position perturbation.
    
    Returns:
        npt.NDArray[np.floating]: End-to-end chain vector approximation
        for each chain in the cross-link structure RVE.
    
    """
    k_num = np.shape(X_clnk)[0]
    r_approx_vec_clnk = np.empty_like(X_clnk)
    for chn_indx in range(k_num):
        r_approx_vec_clnk[chn_indx] = r_chn_approx_vec_func(
            F_Lmbda, X_clnk[chn_indx], Q_clnk_m, y_clnk_m,
            delta_Q_clnk, delta_y_clnk)
    return r_approx_vec_clnk

def r_clnk_func(
        ideal_numerics_form: bool,
        F: npt.NDArray[np.floating],
        X_clnk: npt.NDArray[np.floating],
        Q_clnk: npt.NDArray[np.floating],
        y_clnk: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """End-to-end chain distance/length for each chain in the cross-link
    structure RVE.

    This function calculates the end-to-end chain distance/length for
    each chain in the cross-link structure RVE.

    Args:
        ideal_numerics_form (bool): Boolean indicating if the end-to-end chain vector is calculated with respect to the ideal numerics form (if True) or the original form (if False). 
        F (npt.NDArray[np.floating]): Deformation gradient.
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        Q_clnk (npt.NDArray[np.floating]): Cross-link rotation.
        y_clnk (npt.NDArray[np.floating]): Cross-link junction position.
    
    Returns:
        npt.NDArray[np.floating]: End-to-end chain distance/length for
        each chain in the cross-link structure RVE.
    
    """
    k_num = np.shape(X_clnk)[0]
    r_clnk = np.empty(k_num)
    for chn_indx in range(k_num):
        r_clnk[chn_indx] = r_chn_func(
            ideal_numerics_form, F, X_clnk[chn_indx], Q_clnk, y_clnk)
    return r_clnk

def r_approx_clnk_func(
        F_Lmbda: npt.NDArray[np.floating],
        X_clnk: npt.NDArray[np.floating],
        Q_clnk_m: npt.NDArray[np.floating],
        y_clnk_m: npt.NDArray[np.floating],
        delta_Q_clnk: npt.NDArray[np.floating],
        delta_y_clnk: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """End-to-end chain distance/length approximation for each chain in
    the cross-link structure RVE.

    This function calculates the end-to-end chain distance/length
    approximation for each chain in the cross-link structure RVE.

    Args:
        F_Lmbda (npt.NDArray[np.floating]): Deformation gradient, assuming incompressibility, of the form F_Lmbda = diag(Lmbda_1, Lmbda_2, 1/(Lmbda_1*Lmbda_2)).
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        Q_clnk_m (npt.NDArray[np.floating]): Cross-link rotation for the monodisperse cross-link structure RVE.
        y_clnk_m (npt.NDArray[np.floating]): Cross-link junction position for the monodisperse cross-link structure RVE.
        delta_Q_clnk (npt.NDArray[np.floating]): Cross-link rotation perturbation.
        delta_y_clnk (npt.NDArray[np.floating]): Cross-link junction position perturbation.
    
    Returns:
        npt.NDArray[np.floating]: End-to-end chain distance/length
        approximation for each chain in the cross-link structure RVE.
    
    """
    k_num = np.shape(X_clnk)[0]
    r_approx_clnk = np.empty(k_num)
    for chn_indx in range(k_num):
        r_approx_clnk[chn_indx] = r_chn_approx_func(
            F_Lmbda, X_clnk[chn_indx], Q_clnk_m, y_clnk_m,
            delta_Q_clnk, delta_y_clnk)
    return r_approx_clnk

def gamma_vec_clnk_func(
        ideal_numerics_form: bool,
        F: npt.NDArray[np.floating],
        X_clnk: npt.NDArray[np.floating],
        Q_clnk: npt.NDArray[np.floating],
        y_clnk: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float) -> npt.NDArray[np.floating]:
    """Absolute/Equilibrium chain stretch vector for each chain in the
    cross-link structure RVE.

    This function calculates the absolute/equilibrium chain stretch
    vector for each chain in the cross-link structure RVE.

    Args:
        ideal_numerics_form (bool): Boolean indicating if the end-to-end chain vector is calculated with respect to the ideal numerics form (if True) or the original form (if False). 
        F (npt.NDArray[np.floating]): Deformation gradient.
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        Q_clnk (npt.NDArray[np.floating]): Cross-link rotation.
        y_clnk (npt.NDArray[np.floating]): Cross-link junction position.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
    
    Returns:
        npt.NDArray[np.floating]: Absolute/Equilibrium chain stretch
        vector for each chain in the cross-link structure RVE.
    
    """
    k_num = np.shape(X_clnk)[0]
    gamma_vec_clnk = np.empty_like(X_clnk)
    for chn_indx in range(k_num):
        gamma_vec_clnk[chn_indx] = gamma_chn_vec_func(
            ideal_numerics_form, F, X_clnk[chn_indx], Q_clnk, y_clnk,
            n_clnk[chn_indx], b)
    return gamma_vec_clnk

def gamma_approx_vec_clnk_func(
        F_Lmbda: npt.NDArray[np.floating],
        X_clnk: npt.NDArray[np.floating],
        Q_clnk_m: npt.NDArray[np.floating],
        y_clnk_m: npt.NDArray[np.floating],
        delta_Q_clnk: npt.NDArray[np.floating],
        delta_y_clnk: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float) -> npt.NDArray[np.floating]:
    """Absolute/Equilibrium chain stretch vector approximation for each
    chain in the cross-link structure RVE.

    This function calculates the absolute/equilibrium chain stretch
    vector approximation for each chain in the cross-link structure RVE.

    Args:
        F_Lmbda (npt.NDArray[np.floating]): Deformation gradient, assuming incompressibility, of the form F_Lmbda = diag(Lmbda_1, Lmbda_2, 1/(Lmbda_1*Lmbda_2)).
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        Q_clnk_m (npt.NDArray[np.floating]): Cross-link rotation for the monodisperse cross-link structure RVE.
        y_clnk_m (npt.NDArray[np.floating]): Cross-link junction position for the monodisperse cross-link structure RVE.
        delta_Q_clnk (npt.NDArray[np.floating]): Cross-link rotation perturbation.
        delta_y_clnk (npt.NDArray[np.floating]): Cross-link junction position perturbation.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
    
    Returns:
        npt.NDArray[np.floating]: Absolute/Equilibrium chain stretch
        vector approximation for each chain in the cross-link structure
        RVE.
    
    """
    k_num = np.shape(X_clnk)[0]
    gamma_approx_vec_clnk = np.empty_like(X_clnk)
    for chn_indx in range(k_num):
        gamma_approx_vec_clnk[chn_indx] = gamma_chn_approx_vec_func(
            F_Lmbda, X_clnk[chn_indx], Q_clnk_m, y_clnk_m,
            delta_Q_clnk, delta_y_clnk, n_clnk[chn_indx], b)
    return gamma_approx_vec_clnk

def gamma_clnk_func(
        ideal_numerics_form: bool,
        F: npt.NDArray[np.floating],
        X_clnk: npt.NDArray[np.floating],
        Q_clnk: npt.NDArray[np.floating],
        y_clnk: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float) -> npt.NDArray[np.floating]:
    """Absolute/Equilibrium chain stretch for each chain in the
    cross-link structure RVE.

    This function calculates the absolute/equilibrium chain stretch for
    each chain in the cross-link structure RVE.

    Args:
        ideal_numerics_form (bool): Boolean indicating if the end-to-end chain vector is calculated with respect to the ideal numerics form (if True) or the original form (if False). 
        F (npt.NDArray[np.floating]): Deformation gradient.
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        Q_clnk (npt.NDArray[np.floating]): Cross-link rotation.
        y_clnk (npt.NDArray[np.floating]): Cross-link junction position.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
    
    Returns:
        npt.NDArray[np.floating]: Absolute/Equilibrium chain stretch for
        each chain in the cross-link structure RVE.
    
    """
    k_num = np.shape(X_clnk)[0]
    gamma_clnk = np.empty(k_num)
    for chn_indx in range(k_num):
        gamma_clnk[chn_indx] = gamma_chn_func(
            ideal_numerics_form, F, X_clnk[chn_indx], Q_clnk, y_clnk,
            n_clnk[chn_indx], b)
    return gamma_clnk

def gamma_approx_clnk_func(
        F_Lmbda: npt.NDArray[np.floating],
        X_clnk: npt.NDArray[np.floating],
        Q_clnk_m: npt.NDArray[np.floating],
        y_clnk_m: npt.NDArray[np.floating],
        delta_Q_clnk: npt.NDArray[np.floating],
        delta_y_clnk: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float) -> npt.NDArray[np.floating]:
    """Absolute/Equilibrium chain stretch approximation for each chain
    in the cross-link structure RVE.

    This function calculates the absolute/equilibrium chain stretch
    approximation for each chain in the cross-link structure RVE.

    Args:
        F_Lmbda (npt.NDArray[np.floating]): Deformation gradient, assuming incompressibility, of the form F_Lmbda = diag(Lmbda_1, Lmbda_2, 1/(Lmbda_1*Lmbda_2)).
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        Q_clnk_m (npt.NDArray[np.floating]): Cross-link rotation for the monodisperse cross-link structure RVE.
        y_clnk_m (npt.NDArray[np.floating]): Cross-link junction position for the monodisperse cross-link structure RVE.
        delta_Q_clnk (npt.NDArray[np.floating]): Cross-link rotation perturbation.
        delta_y_clnk (npt.NDArray[np.floating]): Cross-link junction position perturbation.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
    
    Returns:
        float: Absolute/Equilibrium chain stretch approximation for each
        chain in the cross-link structure RVE.
    
    """
    k_num = np.shape(X_clnk)[0]
    gamma_approx_clnk = np.empty(k_num)
    for chn_indx in range(k_num):
        gamma_approx_clnk[chn_indx] = gamma_chn_approx_func(
            F_Lmbda, X_clnk[chn_indx], Q_clnk_m, y_clnk_m,
            delta_Q_clnk, delta_y_clnk, n_clnk[chn_indx], b)
    return gamma_approx_clnk

def W_clnk_chns_func(
        gamma_clnk: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        w_c_func,
        w_c_args: tuple[float] | tuple[None],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float] | tuple[None]) -> float:
    """Nondimensional cross-link polymer chain free energy.

    This function calculates the nondimensional cross-link polymer chain
    free energy.

    Args:
        gamma_clnk (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch for each chain in the cross-link structure RVE.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        w_c_func (function): Nondimensional polymer chain free energy function.
        w_c_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        w_c_dfrmtn_func (function): Nondimensional polymer chain deformation free energy function.
        w_c_dfrmtn_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
    
    Returns:
        float: Nondimensional cross-link polymer chain free energy.
    
    """
    W_clnk_chns = 0.
    for chn_indx in range(np.shape(n_clnk)[0]):
        W_clnk_chns += w_chn_func(
            gamma_clnk[chn_indx], n_clnk[chn_indx], w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    return W_clnk_chns

def dW_clnk_chns__dy_clnk_func(
        gamma_vec_clnk: npt.NDArray[np.floating],
        gamma_clnk: npt.NDArray[np.floating],
        dw_c__dy_clnk_func,
        dw_c__dy_clnk_args: tuple[float] | tuple[None]) -> npt.NDArray[np.floating]:
    """Nondimensional derivative of the cross-link polymer chain free
    energy with respect to the cross-link junction position.
    
    This function calculates the nondimensional derivative of the
    cross-link polymer chain free energy with respect to the cross-link
    junction position.

    Args:
        gamma_vec_clnk (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch vector for each chain in the cross-link structure RVE.
        gamma_clnk (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch for each chain in the cross-link structure RVE.
        dw_c__dy_clnk_func (function): Nondimensional derivative of the polymer chain free energy with respect to the cross-link junction position function.
        dw_c__dy_clnk_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec and the absolute/equilibrium chain stretch gamma).
    
    Returns:
        npt.NDArray[np.floating]: Nondimensional derivative of the
        cross-link polymer chain free energy with respect to the
        cross-link junction position.
    
    """
    dW_clnk_chns__dy_clnk = np.zeros(3)
    for chn_indx in range(np.shape(gamma_clnk)[0]):
        dW_clnk_chns__dy_clnk += (
            dw_c__dy_clnk_func(
                gamma_vec_clnk[chn_indx], gamma_clnk[chn_indx],
                *dw_c__dy_clnk_args)
        )
    return dW_clnk_chns__dy_clnk

def d2W_clnk_chns__dy_clnk_dy_clnk_func(
        gamma_vec_clnk: npt.NDArray[np.floating],
        gamma_clnk: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        d2w_c__dy_clnk_dy_clnk_func,
        d2w_c__dy_clnk_dy_clnk_args: tuple[float] | tuple[None]) -> npt.NDArray[np.floating]:
    """Nondimensional second derivative of the cross-link polymer chain
    free energy with respect to the cross-link junction position.
    
    This function calculates the nondimensional second derivative of the
    cross-link polymer chain free energy with respect to the cross-link
    junction position.

    Args:
        gamma_vec_clnk (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch vector for each chain in the cross-link structure RVE.
        gamma_clnk (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch for each chain in the cross-link structure RVE.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        d2w_c__dy_clnk_dy_clnk_func (function): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function.
        d2w_c__dy_clnk_dy_clnk_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n).
    
    Returns:
        npt.NDArray[np.floating]: Nondimensional second derivative of
        the cross-link polymer chain free energy with respect to the
        cross-link junction position.
    
    """
    d2W_clnk_chns__dy_clnk_dy_clnk = np.zeros((3, 3))
    for chn_indx in range(np.shape(gamma_clnk)[0]):
        d2W_clnk_chns__dy_clnk_dy_clnk += (
            d2w_c__dy_clnk_dy_clnk_func(
                gamma_vec_clnk[chn_indx], gamma_clnk[chn_indx],
                n_clnk[chn_indx], *d2w_c__dy_clnk_dy_clnk_args)
        )
    return d2W_clnk_chns__dy_clnk_dy_clnk

def W_clnk_y_flucts_func(
        gamma_vec_clnk: npt.NDArray[np.floating],
        gamma_clnk: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        d2w_c__dy_clnk_dy_clnk_func,
        d2w_c__dy_clnk_dy_clnk_args: tuple[float] | tuple[None]) -> float:
    """Nondimensional cross-link junction fluctuation free energy.
    
    This function calculates the nondimensional cross-link junction
    fluctuation free energy.

    Args:
        gamma_vec_clnk (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch vector for each chain in the cross-link structure RVE.
        gamma_clnk (npt.NDArray[np.floating]): Absolute/Equilibrium chain stretch for each chain in the cross-link structure RVE.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        d2w_c__dy_clnk_dy_clnk_func (function): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function.
        d2w_c__dy_clnk_dy_clnk_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n).
    
    Returns:
        float: Nondimensional cross-link junction fluctuation free
        energy.
    
    """
    return (
        0.5
        * np.log(
            np.linalg.det(
                d2W_clnk_chns__dy_clnk_dy_clnk_func(
                    gamma_vec_clnk, gamma_clnk, n_clnk,
                    d2w_c__dy_clnk_dy_clnk_func, d2w_c__dy_clnk_dy_clnk_args)))
    )