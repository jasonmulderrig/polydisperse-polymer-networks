import numpy as np
import numpy.typing as npt
from scipy.optimize import (
    NonlinearConstraint,
    Bounds,
    minimize
)
from scipy.optimize import (
    differential_evolution,
    shgo
)
from src.helpers.clnk_structure import (
    x_hat_clnk_func,
    com_x_clnk_func,
    chull_eqs_clnk_func,
    x_clnk_min_max_func,
    regular_octahedral_6_chn_clnk_X_hat_clnk_func,
    cube_8_chn_clnk_X_hat_clnk_func
)
from src.helpers.clnk_deformation import (
    monodisperse_y_clnk,
    gamma_clnk_func,
    gamma_vec_clnk_func,
    gamma_approx_clnk_func,
    gamma_approx_vec_clnk_func,
    W_clnk_chns_func,
    dW_clnk_chns__dy_clnk_func,
    d2W_clnk_chns__dy_clnk_dy_clnk_func,
    W_clnk_y_flucts_func
)
from src.helpers.rotations import Q_zyz_euler
from src.helpers.continuum_mechanics import principal_stretch_decomposition
from src.helpers.chain_free_energy import (
    master_dw_c__dy_clnk_func,
    master_dw_c__dy_clnk_args_func,
    master_d2w_c__dy_clnk_dy_clnk_func,
    master_d2w_c__dy_clnk_dy_clnk_args_func
)
from src.helpers.means import geo_mean_func

def y_clnk_frame_avrg_so3_quad_eval(
        so3_quad: npt.NDArray[np.float64],
        sph_quad_symmtry: bool,
        y_clnk_frame_avrg_so3: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """SO(3) quadrature evaluation for the optimal cross-link junction
    position.

    This function evaluates the SO(3) quadrature for the optimal
    cross-link junction position.
    
    Args:
        so3_quad (npt.NDArray[np.float64]): SO(3) quadrature scheme.
        sph_quad_symmtry (bool): Boolean indicating if the SO(3) quadrature scheme is hemispherically symmetric.
        y_clnk_frame_avrg_so3 (npt.NDArray[np.float64]): Optimal cross-link junction position for each cross-link SO(3) quadrature orientation.
    
    Returns:
        npt.NDArray[np.float64]: SO(3) quadrature optimal cross-link
        junction position.
    
    """
    # Evaluate SO(3) quadrature for y_clnk_frame_avrg_so3
    y_clnk_frame_avrg_so3_quad = np.zeros(3)
    for so3_quad_indx in range(np.shape(so3_quad)[0]):
        y_clnk_frame_avrg_so3_quad += (
            so3_quad[so3_quad_indx, -1] * y_clnk_frame_avrg_so3[so3_quad_indx]
        )
    # If necessary, account for spherical quadrature symmetry
    if sph_quad_symmtry: y_clnk_frame_avrg_so3_quad = np.zeros(3)
    
    return y_clnk_frame_avrg_so3_quad

def y_clnk_norm_frame_avrg_so3_quad_eval(
        so3_quad: npt.NDArray[np.float64],
        sph_quad_symmtry: bool,
        y_clnk_norm_frame_avrg_so3: npt.NDArray[np.float64]) -> float:
    """SO(3) quadrature evaluation for the distance between the origin
    and the optimal cross-link junction position.

    This function evaluates the SO(3) quadrature for the distance
    between the origin and the optimal cross-link junction position.
    
    Args:
        so3_quad (npt.NDArray[np.float64]): SO(3) quadrature scheme.
        sph_quad_symmtry (bool): Boolean indicating if the SO(3) quadrature scheme is hemispherically symmetric.
        y_clnk_norm_frame_avrg_so3 (npt.NDArray[np.float64]): Distance between the origin and the optimal cross-link junction position for each cross-link SO(3) quadrature orientation.
    
    Returns:
        float: SO(3) quadrature distance between the origin and the
        optimal cross-link junction position.
    
    """
    # Evaluate SO(3) quadrature for y_clnk_norm_frame_avrg_so3
    y_clnk_norm_frame_avrg_so3_quad = 0.
    for so3_quad_indx in range(np.shape(so3_quad)[0]):
        y_clnk_norm_frame_avrg_so3_quad += (
            so3_quad[so3_quad_indx, -1]
            * y_clnk_norm_frame_avrg_so3[so3_quad_indx]
        )
    # If necessary, account for spherical quadrature symmetry
    if sph_quad_symmtry: y_clnk_norm_frame_avrg_so3_quad *= 2.
    
    return y_clnk_norm_frame_avrg_so3_quad

def gamma_clnk_frame_avrg_so3_quad_eval(
        so3_quad: npt.NDArray[np.float64],
        sph_quad_symmtry: bool,
        gamma_clnk_frame_avrg_so3: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """SO(3) quadrature evaluation for the absolute/equilibrium chain
    stretch for each chain in the cross-link.

    This function evaluates the SO(3) quadrature for the
    absolute/equilibrium chain stretch for each chain in the cross-link.
    
    Args:
        so3_quad (npt.NDArray[np.float64]): SO(3) quadrature scheme.
        sph_quad_symmtry (bool): Boolean indicating if the SO(3) quadrature scheme is hemispherically symmetric.
        gamma_clnk_frame_avrg_so3 (npt.NDArray[np.float64]): Absolute/Equilibrium chain stretch for each chain in the cross-link at each cross-link SO(3) quadrature orientation.
    
    Returns:
        npt.NDArray[np.float64]: SO(3) quadrature absolute/equilibrium
        chain stretch for each chain in the cross-link.
    
    """
    # Evaluate SO(3) quadrature for gamma_clnk_frame_avrg_so3
    gamma_clnk_frame_avrg_so3_quad = np.zeros(
        np.shape(gamma_clnk_frame_avrg_so3)[1])
    for so3_quad_indx in range(np.shape(so3_quad)[0]):
        gamma_clnk_frame_avrg_so3_quad += (
            so3_quad[so3_quad_indx, -1]
            * gamma_clnk_frame_avrg_so3[so3_quad_indx]
        )
    # If necessary, account for spherical quadrature symmetry
    if sph_quad_symmtry: gamma_clnk_frame_avrg_so3_quad *= 2.
    
    return gamma_clnk_frame_avrg_so3_quad

def W_clnk_chns_frame_avrg_so3_quad_eval(
        so3_quad: npt.NDArray[np.float64],
        sph_quad_symmtry: bool,
        W_clnk_chns_frame_avrg_so3: npt.NDArray[np.float64]) -> float:
    """SO(3) quadrature evaluation for the nondimensional cross-link
    polymer chain free energy.

    This function evaluates the SO(3) quadrature for the nondimensional
    cross-link polymer chain free energy.
    
    Args:
        so3_quad (npt.NDArray[np.float64]): SO(3) quadrature scheme.
        sph_quad_symmtry (bool): Boolean indicating if the SO(3) quadrature scheme is hemispherically symmetric.
        W_clnk_chns_frame_avrg_so3 (npt.NDArray[np.float64]): Nondimensional cross-link polymer chain free energy for each cross-link SO(3) quadrature orientation.
    
    Returns:
        float: SO(3) quadrature nondimensional cross-link polymer chain
        free energy.
    
    """
    # Evaluate SO(3) quadrature for W_clnk_chns_frame_avrg_so3
    W_clnk_chns_frame_avrg_so3_quad = 0.
    for so3_quad_indx in range(np.shape(so3_quad)[0]):
        W_clnk_chns_frame_avrg_so3_quad += (
            so3_quad[so3_quad_indx, -1]
            * W_clnk_chns_frame_avrg_so3[so3_quad_indx]
        )
    # If necessary, account for spherical quadrature symmetry
    if sph_quad_symmtry: W_clnk_chns_frame_avrg_so3_quad *= 2.
    
    return W_clnk_chns_frame_avrg_so3_quad

def W_clnk_y_flucts_frame_avrg_so3_quad_eval(
        so3_quad: npt.NDArray[np.float64],
        sph_quad_symmtry: bool,
        W_clnk_y_flucts_frame_avrg_so3: npt.NDArray[np.float64]) -> float:
    """SO(3) quadrature evaluation for the nondimensional cross-link
    junction fluctuation free energy.

    This function evaluates the SO(3) quadrature for the nondimensional
    cross-link junction fluctuation free energy.
    
    Args:
        so3_quad (npt.NDArray[np.float64]): SO(3) quadrature scheme.
        sph_quad_symmtry (bool): Boolean indicating if the SO(3) quadrature scheme is hemispherically symmetric.
        W_clnk_y_flucts_frame_avrg_so3 (npt.NDArray[np.float64]): Nondimensional cross-link junction fluctuation free energy for each cross-link SO(3) quadrature orientation.
    
    Returns:
        float: SO(3) quadrature nondimensional cross-link junction
        fluctuation free energy.
    
    """
    # Evaluate SO(3) quadrature for W_clnk_y_flucts_frame_avrg_so3
    W_clnk_y_flucts_frame_avrg_so3_quad = 0.
    for so3_quad_indx in range(np.shape(so3_quad)[0]):
        W_clnk_y_flucts_frame_avrg_so3_quad += (
            so3_quad[so3_quad_indx, -1]
            * W_clnk_y_flucts_frame_avrg_so3[so3_quad_indx]
        )
    # If necessary, account for spherical quadrature symmetry
    if sph_quad_symmtry: W_clnk_y_flucts_frame_avrg_so3_quad *= 2.
    
    return W_clnk_y_flucts_frame_avrg_so3_quad

def monodisperse_clnk_frame_avrg(
        eval_W_clnk_y_flucts: bool,
        F: npt.NDArray[np.float64],
        so3_quad: npt.NDArray[np.float64],
        n_clnk: npt.NDArray[np.float64],
        b_clnk: npt.NDArray[np.float64],
        X_clnk: npt.NDArray[np.float64],
        y_clnk_init: npt.NDArray[np.float64],
        w_c_dist_clnk: npt.NDArray[np.object_],
        w_c_func_clnk: npt.NDArray[np.object_],
        w_c_args_clnk: npt.NDArray[np.object_],
        d2w_c__dy_clnk_dy_clnk_func_clnk: npt.NDArray[np.object_],
        d2w_c__dy_clnk_dy_clnk_args_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_func_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_args_clnk: npt.NDArray[np.object_]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Monodisperse cross-link structure RVE mechanical response in the
    frame averaging limit.

    This function determines the mechanical response of a monodisperse
    cross-link structure RVE in the frame averaging limit.

    Args:
        eval_W_clnk_y_flucts (bool): Boolean indicating if the nondimensional cross-link junction fluctuation free energy ought to be calculated (if True) or not (if False).
        F (npt.NDArray[np.float64]): Deformation gradient.
        so3_quad (npt.NDArray[np.float64]): SO(3) quadrature scheme.
        n_clnk (npt.NDArray[np.float64]): Number of chain segments for each chain in the cross-link structure RVE.
        b_clnk (npt.NDArray[np.float64]): Chain segment and/or cross-linker diameter for each chain in the cross-link structure RVE.
        X_clnk (npt.NDArray[np.float64]): Initial chain end position for each chain in the cross-link structure RVE.
        y_clnk_init (npt.NDArray[np.float64]): Initial cross-link junction position.
        w_c_dist_clnk (npt.NDArray[np.object_]): Nondimensional polymer chain free energy function model string for each chain in the cross-link structure RVE.
        w_c_func_clnk (npt.NDArray[np.object_]): Nondimensional polymer chain free energy function for each chain in the cross-link structure RVE.
        w_c_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE.
        d2w_c__dy_clnk_dy_clnk_func_clnk (npt.NDArray[np.object_]): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function for each chain in the cross-link structure RVE.
        d2w_c__dy_clnk_dy_clnk_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n) for each chain in the cross-link structure RVE.
        w_c_dfrmtn_func_clnk (npt.NDArray[np.object_]): Nondimensional polymer chain deformation free energy function for each chain in the cross-link structure RVE.
        w_c_dfrmtn_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE.
    
    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        Optimal monodisperse cross-link junction position for each
        cross-link SO(3) quadrature orientation, distance between the
        origin and the optimal monodisperse cross-link junction position
        for each cross-link SO(3) quadrature orientation,
        absolute/equilibrium chain stretch for each chain in the
        monodisperse cross-link at each cross-link SO(3) quadrature
        orientation, nondimensional monodisperse cross-link polymer
        chain free energy for each cross-link SO(3) quadrature
        orientation, nondimensional monodisperse cross-link junction
        fluctuation free energy for each cross-link SO(3) quadrature
        orientation.
    
    """
    # Boilerplate initialization, checks, and assertions
    so3_quad_num = np.shape(so3_quad)[0]
    k_num = np.shape(n_clnk)[0]
    X_hat_clnk = x_hat_clnk_func(X_clnk)
    com_X_hat_clnk = com_x_clnk_func(X_hat_clnk)
    full_like_n_clnk_0 = np.empty_like(n_clnk)
    full_like_b_clnk_0 = np.empty_like(b_clnk)
    full_like_w_c_dist_clnk_0 = np.empty_like(w_c_dist_clnk)
    full_like_w_c_func_clnk_0 = np.empty_like(w_c_func_clnk)
    full_like_w_c_args_clnk_0 = np.empty_like(w_c_args_clnk)
    full_like_d2w_c__dy_clnk_dy_clnk_func_clnk_0 = np.empty_like(
        d2w_c__dy_clnk_dy_clnk_func_clnk)
    full_like_d2w_c__dy_clnk_dy_clnk_args_clnk_0 = np.empty_like(
        d2w_c__dy_clnk_dy_clnk_args_clnk)
    full_like_w_c_dfrmtn_func_clnk_0 = np.empty_like(w_c_dfrmtn_func_clnk)
    full_like_w_c_dfrmtn_args_clnk_0 = np.empty_like(w_c_dfrmtn_args_clnk)
    full_like_n_clnk_0.fill(n_clnk[0])
    full_like_b_clnk_0.fill(b_clnk[0])
    full_like_w_c_dist_clnk_0.fill(w_c_dist_clnk[0])
    full_like_w_c_func_clnk_0.fill(w_c_func_clnk[0])
    full_like_w_c_args_clnk_0.fill(w_c_args_clnk[0])
    full_like_d2w_c__dy_clnk_dy_clnk_func_clnk_0.fill(
        d2w_c__dy_clnk_dy_clnk_func_clnk[0])
    full_like_d2w_c__dy_clnk_dy_clnk_args_clnk_0.fill(
        d2w_c__dy_clnk_dy_clnk_args_clnk[0])
    full_like_w_c_dfrmtn_func_clnk_0.fill(w_c_dfrmtn_func_clnk[0])
    full_like_w_c_dfrmtn_args_clnk_0.fill(w_c_dfrmtn_args_clnk[0])
    if (not np.allclose(n_clnk, full_like_n_clnk_0) or
        not np.allclose(b_clnk, full_like_b_clnk_0) or
        not np.all(np.equal(w_c_dist_clnk, full_like_w_c_dist_clnk_0)) or
        not np.all(np.equal(w_c_func_clnk, full_like_w_c_func_clnk_0)) or
        not np.all(np.equal(w_c_args_clnk, full_like_w_c_args_clnk_0)) or
        not np.all(np.equal(d2w_c__dy_clnk_dy_clnk_func_clnk, full_like_d2w_c__dy_clnk_dy_clnk_func_clnk_0)) or
        not np.all(np.equal(d2w_c__dy_clnk_dy_clnk_args_clnk, full_like_d2w_c__dy_clnk_dy_clnk_args_clnk_0)) or
        not np.all(np.equal(w_c_dfrmtn_func_clnk, full_like_w_c_dfrmtn_func_clnk_0)) or
        not np.all(np.equal(w_c_dfrmtn_args_clnk, full_like_w_c_dfrmtn_args_clnk_0)) or
        not np.allclose(com_X_hat_clnk, np.zeros(3)) or
        not np.allclose(y_clnk_init, np.zeros(3))):
        error_str = (
            "This function is only applicable for well-structured "
            + "cross-links of monodisperse chains. Make sure that each "
            + "chain attribute is of the same value/setting for every "
            + "chain, the initial position of the cross-link is at the "
            + "origin, and that the initial center-of-mass of the "
            + "cross-link is also located at the origin."
        )
        raise ValueError(error_str)
    clnk = False
    if w_c_dist_clnk[0] != "inext_gaussian_fjc":
        if k_num == 6:
            clnk = np.allclose(
                X_hat_clnk, regular_octahedral_6_chn_clnk_X_hat_clnk_func())
        elif k_num == 8:
            clnk = np.allclose(X_hat_clnk, cube_8_chn_clnk_X_hat_clnk_func())
    else: clnk = True
    if not clnk:
        error_str = (
            "This function is only applicable for well-structured "
            + "cross-links of monodisperse inextensible Gaussian "
            + "chains, for regular octahedral 6-chain cross-link "
            + "structure with monodisperse non-Gaussian chains, or for "
            + "cube 8-chain cross-link structure with monodisperse "
            + "non-Gaussian chains. Make sure that the cross-link "
            + "structure corresponds to one of the aforementioned "
            + "cross-link structures."
        )
        raise ValueError(error_str)
    
    # Additional initialization
    y_clnk_star = monodisperse_y_clnk()
    y_clnk_star_frame_avrg_so3 = np.zeros((so3_quad_num, 3))
    y_clnk_star_norm_frame_avrg_so3 = np.zeros(so3_quad_num)
    gamma_clnk_star_frame_avrg_so3 = np.zeros((so3_quad_num, k_num))
    W_clnk_chns_star_frame_avrg_so3 = np.zeros(so3_quad_num)
    W_clnk_y_flucts_star_frame_avrg_so3 = np.zeros(so3_quad_num)

    # Evaluate the monodisperse cross-link in each SO(3) quadrature
    # point orientation
    for so3_quad_indx in range(so3_quad_num):
        # Extract SO(3) quadrature point rotation matrix
        Q_0_clnk_star_so3 = Q_zyz_euler(so3_quad[so3_quad_indx, :-1])
        
        # Calculate the absolute/equilibrium chain stretch for each
        # chain
        gamma_clnk_star_so3 = gamma_clnk_func(
            False, F, X_clnk, Q_0_clnk_star_so3, y_clnk_star, n_clnk, b_clnk)
        
        # Calculate the nondimensional cross-link chain free energy
        W_clnk_chns_star_so3 = W_clnk_chns_func(
            gamma_clnk_star_so3, n_clnk, w_c_func_clnk, w_c_args_clnk,
            w_c_dfrmtn_func_clnk, w_c_dfrmtn_args_clnk)
        
        # If called for, calculate the nondimensional cross-link
        # junction fluctuation free energy
        W_clnk_y_flucts_star_so3 = 0.
        if eval_W_clnk_y_flucts:
            # Calculate the absolute/equilibrium chain stretch vector
            # for each chain
            gamma_vec_clnk_star_so3 = gamma_vec_clnk_func(
                False, F, X_clnk, Q_0_clnk_star_so3, y_clnk_star, n_clnk, b_clnk)
            W_clnk_y_flucts_star_so3 = W_clnk_y_flucts_func(
                gamma_vec_clnk_star_so3, gamma_clnk_star_so3, n_clnk, b_clnk,
                d2w_c__dy_clnk_dy_clnk_func_clnk,
                d2w_c__dy_clnk_dy_clnk_args_clnk)
        
        # Update SO(3) quadrature arrays
        gamma_clnk_star_frame_avrg_so3[so3_quad_indx] = gamma_clnk_star_so3
        W_clnk_chns_star_frame_avrg_so3[so3_quad_indx] = W_clnk_chns_star_so3
        W_clnk_y_flucts_star_frame_avrg_so3[so3_quad_indx] = (
            W_clnk_y_flucts_star_so3
        )
    
    return (
        y_clnk_star_frame_avrg_so3, y_clnk_star_norm_frame_avrg_so3,
        gamma_clnk_star_frame_avrg_so3, W_clnk_chns_star_frame_avrg_so3,
        W_clnk_y_flucts_star_frame_avrg_so3
    )

def W_clnk_chns_frame_avrg(
        y_clnk: npt.NDArray[np.float64],
        omega_clnk: npt.NDArray[np.float64],
        F: npt.NDArray[np.float64],
        n_clnk: npt.NDArray[np.float64],
        b_clnk: npt.NDArray[np.float64],
        X_clnk: npt.NDArray[np.float64],
        w_c_func_clnk: npt.NDArray[np.object_],
        w_c_args_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_func_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_args_clnk: npt.NDArray[np.object_]) -> float:
    """Nondimensional cross-link polymer chain free energy in the frame
    averaging limit.

    This function supplies the nondimensional cross-link polymer chain
    free energy in the frame averaging limit as a suitable objective
    function for constrained minimization.

    Args:
        y_clnk (npt.NDArray[np.float64]): Cross-link junction position for the cross-link structure RVE.
        omega_clnk (npt.NDArray[np.float64]): Vector of Euler angles for the cross-link structure RVE.
        F (npt.NDArray[np.float64]): Deformation gradient.
        n_clnk (npt.NDArray[np.float64]): Number of chain segments for each chain in the cross-link structure RVE.
        b_clnk (npt.NDArray[np.float64]): Chain segment and/or cross-linker diameter for each chain in the cross-link structure RVE.
        X_clnk (npt.NDArray[np.float64]): Initial chain end position for each chain in the cross-link structure RVE.
        w_c_func_clnk (npt.NDArray[np.object_]): Nondimensional polymer chain free energy function for each chain in the cross-link structure RVE.
        w_c_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE.
        w_c_dfrmtn_func_clnk (npt.NDArray[np.object_]): Nondimensional polymer chain deformation free energy function for each chain in the cross-link structure RVE.
        w_c_dfrmtn_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE.
    
    Returns:
        float: Nondimensional cross-link polymer chain free energy in
        the frame averaging limit.
    
    """
    # Calculate the absolute/equilibrium chain stretch for each chain
    # (using the ideal numerics form for the chain end-to-end vector)
    gamma_clnk = gamma_clnk_func(
        True, F, X_clnk, Q_zyz_euler(omega_clnk), y_clnk, n_clnk, b_clnk)
    
    # Calculate the nondimensional cross-link polymer chain free energy
    return (
        W_clnk_chns_func(
            gamma_clnk, n_clnk, w_c_func_clnk, w_c_args_clnk,
            w_c_dfrmtn_func_clnk, w_c_dfrmtn_args_clnk)
    )

def clnk_frame_avrg_cnstrnd_mnmztn(
        eval_W_clnk_y_flucts: bool,
        cnstrnd_mnmztn_scope: str,
        cnstrnd_mnmztn_method: str,
        rng: np.random.Generator,
        F: npt.NDArray[np.float64],
        so3_quad: npt.NDArray[np.float64],
        y_clnk_so3_quad: npt.NDArray[np.float64],
        n_clnk: npt.NDArray[np.float64],
        b_clnk: npt.NDArray[np.float64],
        X_clnk: npt.NDArray[np.float64],
        w_c_func_clnk: npt.NDArray[np.object_],
        w_c_args_clnk: npt.NDArray[np.object_],
        d2w_c__dy_clnk_dy_clnk_func_clnk: npt.NDArray[np.object_],
        d2w_c__dy_clnk_dy_clnk_args_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_func_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_args_clnk: npt.NDArray[np.object_]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cross-link structure RVE mechanical response in the frame
    averaging limit, as evaluated numerically via constrained
    minimization.

    This function determines the mechanical response of a cross-link
    structure RVE in the frame averaging limit, as evaluated numerically
    via constrained minimization.

    Args:
        eval_W_clnk_y_flucts (bool): Boolean indicating if the nondimensional cross-link junction fluctuation free energy ought to be calculated (if True) or not (if False).
        cnstrnd_mnmztn_scope (str): String indicating the use of either local ("lcl") or global ("glbl") constrained minimization.
        cnstrnd_mnmztn_method (str): String indicating the specific constrained minimization method to utilize. See the code and associated error string in the function for the different types of constrained minimization methods available for use.
        rng (np.random.Generator): Numpy random number generator object.
        F (npt.NDArray[np.float64]): Deformation gradient.
        so3_quad (npt.NDArray[np.float64]): SO(3) quadrature scheme.
        y_clnk_so3_quad (npt.NDArray[np.float64]): Prior cross-link junction position for each cross-link SO(3) quadrature orientation.
        n_clnk (npt.NDArray[np.float64]): Number of chain segments for each chain in the cross-link structure RVE.
        b_clnk (npt.NDArray[np.float64]): Chain segment and/or cross-linker diameter for each chain in the cross-link structure RVE.
        X_clnk (npt.NDArray[np.float64]): Initial chain end position for each chain in the cross-link structure RVE.
        w_c_func_clnk (npt.NDArray[np.object_]): Nondimensional polymer chain free energy function for each chain in the cross-link structure RVE.
        w_c_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE.
        d2w_c__dy_clnk_dy_clnk_func_clnk (npt.NDArray[np.object_]): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function for each chain in the cross-link structure RVE.
        d2w_c__dy_clnk_dy_clnk_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n) for each chain in the cross-link structure RVE.
        w_c_dfrmtn_func_clnk (npt.NDArray[np.object_]): Nondimensional polymer chain deformation free energy function for each chain in the cross-link structure RVE.
        w_c_dfrmtn_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE.
    
    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        Optimal cross-link junction position for each cross-link SO(3)
        quadrature orientation, distance between the origin and the
        optimal cross-link junction position for each cross-link SO(3)
        quadrature orientation, absolute/equilibrium chain stretch for
        each chain in the cross-link at each cross-link SO(3) quadrature
        orientation, nondimensional cross-link polymer chain free energy
        for each cross-link SO(3) quadrature orientation, nondimensional
        cross-link junction fluctuation free energy for each cross-link
        SO(3) quadrature orientation.
    
    """
    # Initialization
    so3_quad_num = np.shape(so3_quad)[0]
    k_num = np.shape(n_clnk)[0]
    y_clnk_frame_avrg_so3 = np.zeros((so3_quad_num, 3))
    y_clnk_norm_frame_avrg_so3 = np.zeros(so3_quad_num)
    gamma_clnk_frame_avrg_so3 = np.zeros((so3_quad_num, k_num))
    W_clnk_chns_frame_avrg_so3 = np.zeros(so3_quad_num)
    W_clnk_y_flucts_frame_avrg_so3 = np.zeros(so3_quad_num)

    # Evaluate the cross-link in each SO(3) quadrature point orientation
    for so3_quad_indx in range(so3_quad_num):
        # Extract prior cross-link junction position associated with
        # this SO(3) quadrature point orientation
        y_clnk_so3 = y_clnk_so3_quad[so3_quad_indx]
        
        # Extract SO(3) quadrature rotation matrix
        omega_clnk_so3 = so3_quad[so3_quad_indx, :-1]
        Q_0_clnk_so3 = Q_zyz_euler(omega_clnk_so3)

        # Gather the (deformed) cross-link convex hull
        x_clnk = np.empty_like(X_clnk)
        for chn_indx in range(np.shape(n_clnk)[0]):
            x_clnk[chn_indx] = np.matmul(
                F, np.matmul(Q_0_clnk_so3, X_clnk[chn_indx]))
        A_chull_eqs_clnk, b_chull_eqs_clnk = chull_eqs_clnk_func(x_clnk)

        # Gather the cross-link convex hull constraints on the
        # cross-link junction position
        def chull_clnk_cnstrnt_func(y_clnk):
            return b_chull_eqs_clnk - np.matmul(A_chull_eqs_clnk, y_clnk)
        y_clnk_cnstrnt = NonlinearConstraint(chull_clnk_cnstrnt_func, 0., np.inf)
        
        # Gather the bounds of the cross-link junction position
        y_clnk_min_bounds, y_clnk_max_bounds = x_clnk_min_max_func(x_clnk)
        y_clnk_bounds = Bounds(y_clnk_min_bounds, y_clnk_max_bounds)

        # Gather the arguments of the frame averaging limit
        # nondimensional cross-link polymer chain free energy objective
        # function W_clnk_chns_frame_avrg()
        W_clnk_frame_avrg_args = (
            omega_clnk_so3, F, n_clnk, b_clnk, X_clnk,
            w_c_func_clnk, w_c_args_clnk,
            w_c_dfrmtn_func_clnk, w_c_dfrmtn_args_clnk
        )

        # Apply the called-for method of (local or global) constrained
        # minimization over the frame averaging limit nondimensional
        # cross-link polymer chain free energy to solve for the optimal
        # cross-link junction position
        if cnstrnd_mnmztn_scope == "lcl":
            if cnstrnd_mnmztn_method in ["COBYLA", "COBYQA", "trust-constr"]:
                clnk_frame_avrg = minimize(
                    W_clnk_chns_frame_avrg, y_clnk_so3,
                    args=W_clnk_frame_avrg_args, method=cnstrnd_mnmztn_method,
                    bounds=y_clnk_bounds, constraints=(y_clnk_cnstrnt))
        elif cnstrnd_mnmztn_scope == "glbl":
            if cnstrnd_mnmztn_method == "differential-evolution":
                clnk_frame_avrg = differential_evolution(
                    W_clnk_chns_frame_avrg, y_clnk_bounds,
                    args=W_clnk_frame_avrg_args, rng=rng,
                    constraints=(y_clnk_cnstrnt), x0=y_clnk_so3)
            elif cnstrnd_mnmztn_method == "shgo":
                clnk_frame_avrg = shgo(
                    W_clnk_chns_frame_avrg, y_clnk_bounds,
                    args=W_clnk_frame_avrg_args, constraints=(y_clnk_cnstrnt),
                    minimizer_kwargs={"method": "COBYQA"})
        else:
            error_str = (
                "Several local and global constrained minimization "
                + "methods are implemented in solving for the "
                + "deformation of the cross-link structure "
                + "(cnstrnd_mnmztn_scope = ``lcl'' or ``glbl''). For "
                + "local constrained minimization, the constrained "
                + "optimization by linear approximation algorithm "
                + "(cnstrnd_mnmztn_method = ``COBYLA''), the "
                + "constrained optimization by quadratic "
                + "approximations algorithm (cnstrnd_mnmztn_method = "
                + "``COBYQA''), and the constrained trust-region "
                + "algorithm (cnstrnd_mnmztn_method = "
                + "``trust-constr'') are implemented. For global "
                + "constrained minimization, the differential "
                + "evolution method (cnstrnd_mnmztn_method = "
                + "``differential-evolution'') and the simplicial "
                + "homology global optimization method "
                + "(cnstrnd_mnmztn_method = ``shgo'') are implemented."
            )
            raise ValueError(error_str)
        
        # If the constrained minimization solved successfully, then
        # update the cross-link junction position
        if clnk_frame_avrg.success: y_clnk_so3 = clnk_frame_avrg.x
        y_clnk_norm_so3 = np.linalg.norm(y_clnk_so3)
        
        # Calculate the absolute/equilibrium chain stretch for each
        # chain
        gamma_clnk_so3 = gamma_clnk_func(
            True, F, X_clnk, Q_0_clnk_so3, y_clnk_so3, n_clnk, b_clnk)
        
        # Calculate the nondimensional cross-link chain free energy
        W_clnk_chns_so3 = W_clnk_chns_func(
            gamma_clnk_so3, n_clnk, w_c_func_clnk, w_c_args_clnk,
            w_c_dfrmtn_func_clnk, w_c_dfrmtn_args_clnk)
        
        # If called for, calculate the nondimensional cross-link
        # junction fluctuation free energy
        W_clnk_y_flucts_so3 = 0.
        if eval_W_clnk_y_flucts:
            # Calculate the absolute/equilibrium chain stretch vector
            # for each chain
            gamma_vec_clnk_so3 = gamma_vec_clnk_func(
                True, F, X_clnk, Q_0_clnk_so3, y_clnk_so3, n_clnk, b_clnk)
            W_clnk_y_flucts_so3 = W_clnk_y_flucts_func(
                gamma_vec_clnk_so3, gamma_clnk_so3, n_clnk, b_clnk,
                d2w_c__dy_clnk_dy_clnk_func_clnk,
                d2w_c__dy_clnk_dy_clnk_args_clnk)
        
        # Update SO(3) quadrature arrays
        y_clnk_frame_avrg_so3[so3_quad_indx] = y_clnk_so3
        y_clnk_norm_frame_avrg_so3[so3_quad_indx] = y_clnk_norm_so3
        gamma_clnk_frame_avrg_so3[so3_quad_indx] = gamma_clnk_so3
        W_clnk_chns_frame_avrg_so3[so3_quad_indx] = W_clnk_chns_so3
        W_clnk_y_flucts_frame_avrg_so3[so3_quad_indx] = W_clnk_y_flucts_so3
    
    return (
        y_clnk_frame_avrg_so3, y_clnk_norm_frame_avrg_so3,
        gamma_clnk_frame_avrg_so3, W_clnk_chns_frame_avrg_so3,
        W_clnk_y_flucts_frame_avrg_so3
    )

def clnk_frame_avrg_approx(
        eval_W_clnk_y_flucts: bool,
        use_inext_gaussian_fjc_delta_clnk: bool,
        F: npt.NDArray[np.float64],
        so3_quad: npt.NDArray[np.float64],
        n_clnk: npt.NDArray[np.float64],
        b_clnk: npt.NDArray[np.float64],
        X_clnk: npt.NDArray[np.float64],
        y_clnk_init: npt.NDArray[np.float64],
        w_c_func_clnk: npt.NDArray[np.object_],
        w_c_args_clnk: npt.NDArray[np.object_],
        dw_c__dy_clnk_func_clnk: npt.NDArray[np.object_],
        dw_c__dy_clnk_args_clnk: npt.NDArray[np.object_],
        d2w_c__dy_clnk_dy_clnk_func_clnk: npt.NDArray[np.object_],
        d2w_c__dy_clnk_dy_clnk_args_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_func_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_args_clnk: npt.NDArray[np.object_]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Cross-link structure RVE mechanical response in the frame
    averaging limit, as evaluated via closed-form approximation.

    This function determines the mechanical response of a cross-link
    structure RVE in the frame averaging limit, as evaluated via
    closed-form approximation.

    Args:
        eval_W_clnk_y_flucts (bool): Boolean indicating if the nondimensional cross-link junction fluctuation free energy ought to be calculated (if True) or not (if False).
        use_inext_gaussian_fjc_delta_clnk (bool): Boolean indicating if the inextensible Gaussian FJC model ought to be used to calculate the optimal cross-link junction position perturbation.
        F (npt.NDArray[np.float64]): Deformation gradient.
        so3_quad (npt.NDArray[np.float64]): SO(3) quadrature scheme.
        n_clnk (npt.NDArray[np.float64]): Number of chain segments for each chain in the cross-link structure RVE.
        b_clnk (npt.NDArray[np.float64]): Chain segment and/or cross-linker diameter for each chain in the cross-link structure RVE.
        X_clnk (npt.NDArray[np.float64]): Initial chain end position for each chain in the cross-link structure RVE.
        y_clnk_init (npt.NDArray[np.float64]): Initial cross-link junction position.
        w_c_func_clnk (npt.NDArray[np.object_]): Nondimensional polymer chain free energy function for each chain in the cross-link structure RVE.
        w_c_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE.
        dw_c__dy_clnk_func_clnk (npt.NDArray[np.object_]): Nondimensional derivative of the polymer chain free energy with respect to the cross-link junction position function for each chain in the cross-link structure RVE.
        dw_c__dy_clnk_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec and the absolute/equilibrium chain stretch gamma) for each chain in the cross-link structure RVE.
        d2w_c__dy_clnk_dy_clnk_func_clnk (npt.NDArray[np.object_]): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function for each chain in the cross-link structure RVE.
        d2w_c__dy_clnk_dy_clnk_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n) for each chain in the cross-link structure RVE.
        w_c_dfrmtn_func_clnk (npt.NDArray[np.object_]): Nondimensional polymer chain deformation free energy function for each chain in the cross-link structure RVE.
        w_c_dfrmtn_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE.
    
    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64]]:
        Optimal cross-link junction position perturbation for each
        cross-link SO(3) quadrature orientation, distance between the
        origin and the optimal cross-link junction position perturbation
        for each cross-link SO(3) quadrature orientation,
        absolute/equilibrium chain stretch for each chain in the
        cross-link at each cross-link SO(3) quadrature orientation,
        nondimensional cross-link polymer chain free energy for each
        cross-link SO(3) quadrature orientation, nondimensional
        cross-link junction fluctuation free energy for each cross-link
        SO(3) quadrature orientation.
    
    """
    # Boilerplate initialization, checks, and assertions
    k_num = np.shape(n_clnk)[0]
    X_hat_clnk = x_hat_clnk_func(X_clnk)
    com_X_hat_clnk = com_x_clnk_func(X_hat_clnk)
    if (not np.isclose(np.linalg.det(F), 1.) or
        not np.allclose(com_X_hat_clnk, np.zeros(3)) or
        not np.allclose(y_clnk_init, np.zeros(3))):
        error_str = (
            "This methodology is only applicable to well-structured "
            + "cross-link RVEs under incompressible deformation. Make "
            + "sure that the initial center-of-mass of the cross-link "
            + "composed of unit-length chains is located at the "
            + "origin, the initial position of the cross-link is at "
            + "the origin, and the deformation is incompressible."
        )
        raise ValueError(error_str)
    b_clnk_geo_mean = geo_mean_func(b_clnk)
    y_clnk_m = monodisperse_y_clnk()
    Lmbda, _ = principal_stretch_decomposition(F)
    lmbda_0, lmbda_1, _ = Lmbda
    F_Lmbda = np.diag(np.asarray([lmbda_0, lmbda_1, 1./(lmbda_0*lmbda_1)]))

    # If called for, gather the functions and function arguments needed
    # to calculate the optimal cross-link junction position perturbation
    # for the case of the inextensible Gaussian FJC model
    if use_inext_gaussian_fjc_delta_clnk:
        inext_gaussian_fjc = "inext_gaussian_fjc"
        dw_c__dy_clnk_func_clnk = np.empty_like(n_clnk, dtype=object)
        dw_c__dy_clnk_args_clnk = np.empty_like(n_clnk, dtype=object)
        d2w_c__dy_clnk_dy_clnk_func_clnk = np.empty_like(n_clnk, dtype=object)
        d2w_c__dy_clnk_dy_clnk_args_clnk = np.empty_like(n_clnk, dtype=object)
        for chn_indx in range(np.shape(n_clnk)[0]):
            dw_c__dy_clnk_func_clnk[chn_indx] = master_dw_c__dy_clnk_func(
                inext_gaussian_fjc)
            dw_c__dy_clnk_args_clnk[chn_indx] = master_dw_c__dy_clnk_args_func(
                inext_gaussian_fjc, 0., 0., tuple([]))
            d2w_c__dy_clnk_dy_clnk_func_clnk[chn_indx] = (
                master_d2w_c__dy_clnk_dy_clnk_func(inext_gaussian_fjc)
            )
            d2w_c__dy_clnk_dy_clnk_args_clnk[chn_indx] = (
                master_d2w_c__dy_clnk_dy_clnk_args_func(
                    inext_gaussian_fjc, 0., 0., tuple([]))
            )

    # Initialization
    so3_quad_num = np.shape(so3_quad)[0]
    k_num = np.shape(n_clnk)[0]
    delta_y_clnk_frame_avrg_approx_so3 = np.zeros((so3_quad_num, 3))
    delta_y_clnk_norm_frame_avrg_approx_so3 = np.zeros(so3_quad_num)
    gamma_clnk_frame_avrg_approx_so3 = np.zeros((so3_quad_num, k_num))
    W_clnk_chns_frame_avrg_approx_so3 = np.zeros(so3_quad_num)
    W_clnk_y_flucts_frame_avrg_approx_so3 = np.zeros(so3_quad_num)

    # Evaluate the cross-link in each SO(3) quadrature point orientation
    for so3_quad_indx in range(so3_quad_num):
        # Extract SO(3) quadrature point rotation matrix
        Q_0_clnk_so3 = Q_zyz_euler(so3_quad[so3_quad_indx, :-1])

        # Calculate the absolute/equilibrium chain stretch and the
        # absolute/equilibrium chain stretch vector for each chain for
        # the case where there is zero cross-link junction position
        # perturbation
        gamma_clnk_so3_0 = gamma_approx_clnk_func(
            F_Lmbda, X_clnk, np.eye(3), y_clnk_m, Q_0_clnk_so3, np.zeros(3),
            n_clnk, b_clnk)
        gamma_vec_clnk_so3_0 = gamma_approx_vec_clnk_func(
            F_Lmbda, X_clnk, np.eye(3), y_clnk_m, Q_0_clnk_so3, np.zeros(3),
            n_clnk, b_clnk)
        
        # Calculate the nondimensional first and second derivatives of
        # the cross-link polymer chain free energy with respect to the
        # cross-link junction position for the case where there is zero
        # cross-link junction position perturbation
        dW_clnk_chns__dy_clnk_0 = dW_clnk_chns__dy_clnk_func(
            gamma_vec_clnk_so3_0, gamma_clnk_so3_0, b_clnk,
            dw_c__dy_clnk_func_clnk, dw_c__dy_clnk_args_clnk)
        d2W_clnk_chns__dy_clnk_dy_clnk_0 = d2W_clnk_chns__dy_clnk_dy_clnk_func(
            gamma_vec_clnk_so3_0, gamma_clnk_so3_0, n_clnk, b_clnk,
            d2w_c__dy_clnk_dy_clnk_func_clnk, d2w_c__dy_clnk_dy_clnk_args_clnk)
        
        # Calculate the optimal cross-link junction position
        # perturbation
        delta_y_clnk_so3 = (
            -1. * b_clnk_geo_mean
            * np.matmul(
                np.linalg.inv(d2W_clnk_chns__dy_clnk_dy_clnk_0),
                dW_clnk_chns__dy_clnk_0)
        )
        delta_y_clnk_norm_so3 = np.linalg.norm(delta_y_clnk_so3)
        
        # Calculate the absolute/equilibrium chain stretch for each
        # chain
        gamma_clnk_so3 = gamma_approx_clnk_func(
            F_Lmbda, X_clnk, np.eye(3), y_clnk_m, Q_0_clnk_so3,
            delta_y_clnk_so3, n_clnk, b_clnk)
        
        # Calculate the nondimensional cross-link chain free energy
        W_clnk_chns_so3 = W_clnk_chns_func(
            gamma_clnk_so3, n_clnk, w_c_func_clnk, w_c_args_clnk,
            w_c_dfrmtn_func_clnk, w_c_dfrmtn_args_clnk)
        
        # If called for, calculate the nondimensional cross-link
        # junction fluctuation free energy
        W_clnk_y_flucts_so3 = 0.
        if eval_W_clnk_y_flucts:
            # Calculate the absolute/equilibrium chain stretch vector
            # for each chain
            gamma_vec_clnk_so3 = gamma_approx_vec_clnk_func(
                F_Lmbda, X_clnk, np.eye(3), y_clnk_m, Q_0_clnk_so3,
                delta_y_clnk_so3, n_clnk, b_clnk)
            W_clnk_y_flucts_so3 = W_clnk_y_flucts_func(
                gamma_vec_clnk_so3, gamma_clnk_so3, n_clnk, b_clnk,
                d2w_c__dy_clnk_dy_clnk_func_clnk,
                d2w_c__dy_clnk_dy_clnk_args_clnk)
        
        # Update SO(3) quadrature arrays
        delta_y_clnk_frame_avrg_approx_so3[so3_quad_indx] = delta_y_clnk_so3
        delta_y_clnk_norm_frame_avrg_approx_so3[so3_quad_indx] = (
            delta_y_clnk_norm_so3
        )
        gamma_clnk_frame_avrg_approx_so3[so3_quad_indx] = gamma_clnk_so3
        W_clnk_chns_frame_avrg_approx_so3[so3_quad_indx] = W_clnk_chns_so3
        W_clnk_y_flucts_frame_avrg_approx_so3[so3_quad_indx] = (
            W_clnk_y_flucts_so3
        )
    
    return (
        delta_y_clnk_frame_avrg_approx_so3,
        delta_y_clnk_norm_frame_avrg_approx_so3,
        gamma_clnk_frame_avrg_approx_so3, W_clnk_chns_frame_avrg_approx_so3,
        W_clnk_y_flucts_frame_avrg_approx_so3
    )