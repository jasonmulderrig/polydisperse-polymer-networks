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
    amended_3_chn_clnk_X_hat_clnk_func,
    regular_tetrahedral_4_chn_clnk_X_hat_clnk_func,
    equilateral_triangular_bipyramidal_5_chn_clnk_X_hat_clnk_func,
    regular_octahedral_6_chn_clnk_X_hat_clnk_func,
    equilateral_pentagonal_bipyramidal_7_chn_clnk_X_hat_clnk_func,
    cube_8_chn_clnk_X_hat_clnk_func,
    x_hat_clnk_func,
    com_x_clnk_func,
    chull_eqs_clnk_func,
    x_clnk_min_max_func
)
from src.helpers.continuum_mechanics import principal_stretch_decomposition
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
from src.helpers.chain_free_energy import (
    master_dw_c__dy_clnk_func,
    master_dw_c__dy_clnk_args_func,
    master_d2w_c__dy_clnk_dy_clnk_func,
    master_d2w_c__dy_clnk_dy_clnk_args_func
)

def monodisperse_clnk_frame_avrg(
        eval_W_clnk_chns: bool,
        eval_W_clnk_y_flucts: bool,
        F: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float,
        X_clnk: npt.NDArray[np.floating],
        so3_quad: npt.NDArray[np.floating],
        sph_quad_symmtry: bool,
        y_clnk_init: npt.NDArray[np.floating],
        w_c_func,
        w_c_args: tuple[float] | tuple[None],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float] | tuple[None],
        d2w_c__dy_clnk_dy_clnk_func,
        d2w_c__dy_clnk_dy_clnk_args: tuple[float] | tuple[None]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, float, float]:
    """Monodisperse cross-link structure RVE mechanical response in the
    frame averaging limit.

    This function determines the mechanical response of a monodisperse
    cross-link structure RVE in the frame averaging limit.

    Args:
        eval_W_clnk_chns (bool): Boolean indicating if the nondimensional cross-link polymer chain free energy ought to be calculated (if True) or not (if False).
        eval_W_clnk_y_flucts (bool): Boolean indicating if the nondimensional cross-link junction fluctuation free energy ought to be calculated (if True) or not (if False).
        F (npt.NDArray[np.floating]): Deformation gradient.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        so3_quad (npt.NDArray[np.floating]): SO(3) quadrature scheme.
        sph_quad_symmtry (bool): Boolean indicating if the SO(3) quadrature scheme is hemispherically symmetric.
        y_clnk_init (npt.NDArray[np.floating]): Initial cross-link junction position.
        w_c_func (function): Nondimensional polymer chain free energy function.
        w_c_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        w_c_dfrmtn_func (function): Nondimensional polymer chain deformation free energy function.
        w_c_dfrmtn_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        d2w_c__dy_clnk_dy_clnk_func (function): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function.
        d2w_c__dy_clnk_dy_clnk_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n).
    
    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, float, float]:
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
        orientation, nondimensional monodisperse cross-link free energy
        for each cross-link SO(3) quadrature orientation, SO(3)
        quadrature optimal monodisperse cross-link junction position,
        SO(3) quadrature distance between the origin and the optimal
        monodisperse cross-link junction position, SO(3) quadrature
        absolute/equilibrium chain stretch for each chain in the
        monodisperse cross-link, SO(3) quadrature nondimensional
        monodisperse cross-link polymer chain free energy, SO(3)
        quadrature nondimensional monodisperse cross-link junction
        fluctuation free energy, SO(3) quadrature nondimensional
        monodisperse cross-link free energy.
    
    """
    # Boilerplate initialization, checks, and assertions
    so3_quad_num = np.shape(so3_quad)[0]
    k_num = np.shape(n_clnk)[0]
    X_hat_clnk = x_hat_clnk_func(X_clnk)
    com_X_hat_clnk = com_x_clnk_func(X_hat_clnk)
    if (not np.allclose(n_clnk, n_clnk[0]*np.ones_like(n_clnk)) or
        not np.allclose(com_X_hat_clnk, np.zeros(3)) or
        not np.allclose(y_clnk_init, np.zeros(3))):
        error_str = (
            "This function is only applicable for well-structured "
            + "cross-links of monodisperse segment number. Make "
            + "sure that every chain has the same number of segments, "
            + "the initial position of the cross-link is at the "
            + "origin, and that the initial center-of-mass of the "
            + "cross-link is also located at the origin."
        )
        raise ValueError(error_str)
    clnk = False
    if k_num == 3:
        clnk = np.allclose(
            X_hat_clnk, amended_3_chn_clnk_X_hat_clnk_func())
    elif k_num == 4:
        clnk = np.allclose(
            X_hat_clnk, regular_tetrahedral_4_chn_clnk_X_hat_clnk_func())
    elif k_num == 5:
        clnk = np.allclose(
            X_hat_clnk,
            equilateral_triangular_bipyramidal_5_chn_clnk_X_hat_clnk_func())
    elif k_num == 6:
        clnk = np.allclose(
            X_hat_clnk, regular_octahedral_6_chn_clnk_X_hat_clnk_func())
    elif k_num == 7:
        clnk = np.allclose(
            X_hat_clnk,
            equilateral_pentagonal_bipyramidal_7_chn_clnk_X_hat_clnk_func())
    elif k_num == 8:
        clnk = np.allclose(X_hat_clnk, cube_8_chn_clnk_X_hat_clnk_func())
    if not clnk:
        error_str = (
            "This function is only applicable for the amended 3-chain "
            + "cross-link, regular tetrahedral 4-chain cross-link, "
            + "equilateral triangular bipyramidal 5-chain cross-link, "
            + "regular octahedral 6-chain cross-link, equilateral "
            + "pentagonal bipyramidal 7-chain cross-link, or cube "
            + "8-chain cross-link of monodisperse segment number. Make "
            + "sure that the cross-link structure corresponds to one "
            + "of the aforementioned cross-link structures."
        )
        raise ValueError(error_str)
    
    # Additional initialization
    y_clnk_star = monodisperse_y_clnk()
    y_clnk_star_frame_avrg_so3 = np.zeros((so3_quad_num, 3))
    y_clnk_star_norm_frame_avrg_so3 = np.zeros(so3_quad_num)
    gamma_clnk_star_frame_avrg_so3 = np.zeros((so3_quad_num, k_num))
    W_clnk_chns_star_frame_avrg_so3 = np.zeros(so3_quad_num)
    W_clnk_y_flucts_star_frame_avrg_so3 = np.zeros(so3_quad_num)
    y_clnk_star_frame_avrg_so3_quad = np.zeros(3)
    y_clnk_star_norm_frame_avrg_so3_quad = 0.
    gamma_clnk_star_frame_avrg_so3_quad = np.zeros(k_num)
    W_clnk_chns_star_frame_avrg_so3_quad = 0.
    W_clnk_y_flucts_star_frame_avrg_so3_quad = 0.

    # Evaluate the monodisperse cross-link in each SO(3) quadrature
    # point orientation
    for so3_quad_indx in range(so3_quad_num):
        # Extract SO(3) quadrature point weight and rotation matrix
        weight_so3 = so3_quad[so3_quad_indx, -1]
        omega_clnk_so3 = so3_quad[so3_quad_indx, :-1]
        Q_0_clnk_star_so3 = Q_zyz_euler(omega_clnk_so3)
        
        # Calculate the absolute/equilibrium chain stretch for each
        # chain
        gamma_clnk_star_so3 = gamma_clnk_func(
            False, F, X_clnk, Q_0_clnk_star_so3, y_clnk_star, n_clnk, b)
        
        # If called for, calculate each component of the nondimensional
        # cross-link free energy
        W_clnk_chns_star_so3 = 0.
        W_clnk_y_flucts_star_so3 = 0.
        if eval_W_clnk_chns:
            W_clnk_chns_star_so3 = W_clnk_chns_func(
                gamma_clnk_star_so3, n_clnk, w_c_func, w_c_args,
                w_c_dfrmtn_func, w_c_dfrmtn_args)
        if eval_W_clnk_y_flucts:
            # Calculate the absolute/equilibrium chain stretch vector
            # for each chain
            gamma_vec_clnk_star_so3 = gamma_vec_clnk_func(
                False, F, X_clnk, Q_0_clnk_star_so3, y_clnk_star, n_clnk, b)
            W_clnk_y_flucts_star_so3 = W_clnk_y_flucts_func(
                gamma_vec_clnk_star_so3, gamma_clnk_star_so3, n_clnk,
                d2w_c__dy_clnk_dy_clnk_func, d2w_c__dy_clnk_dy_clnk_args)
        
        # Update SO(3) quadrature arrays
        gamma_clnk_star_frame_avrg_so3[so3_quad_indx] = gamma_clnk_star_so3
        W_clnk_chns_star_frame_avrg_so3[so3_quad_indx] = W_clnk_chns_star_so3
        W_clnk_y_flucts_star_frame_avrg_so3[so3_quad_indx] = (
            W_clnk_y_flucts_star_so3
        )

        # Evaluate SO(3) quadrature
        gamma_clnk_star_frame_avrg_so3_quad += weight_so3 * gamma_clnk_star_so3
        W_clnk_chns_star_frame_avrg_so3_quad += (
            weight_so3 * W_clnk_chns_star_so3
        )
        W_clnk_y_flucts_star_frame_avrg_so3_quad += (
            weight_so3 * W_clnk_y_flucts_star_so3
        )
    
    # If necessary, account for spherical quadrature symmetry
    # considerations
    if sph_quad_symmtry:
        gamma_clnk_star_frame_avrg_so3_quad *= 2.
        W_clnk_chns_star_frame_avrg_so3_quad *= 2.
        W_clnk_y_flucts_star_frame_avrg_so3_quad *= 2.
    
    # Calculate the nondimensional cross-link free energy
    W_clnk_star_frame_avrg_so3 = (
        W_clnk_chns_star_frame_avrg_so3 + W_clnk_y_flucts_star_frame_avrg_so3
    )
    W_clnk_star_frame_avrg_so3_quad = (
        W_clnk_chns_star_frame_avrg_so3_quad
        + W_clnk_y_flucts_star_frame_avrg_so3_quad
    )
    
    return (
        y_clnk_star_frame_avrg_so3, y_clnk_star_norm_frame_avrg_so3,
        gamma_clnk_star_frame_avrg_so3, W_clnk_chns_star_frame_avrg_so3,
        W_clnk_y_flucts_star_frame_avrg_so3, W_clnk_star_frame_avrg_so3,
        y_clnk_star_frame_avrg_so3_quad, y_clnk_star_norm_frame_avrg_so3_quad,
        gamma_clnk_star_frame_avrg_so3_quad, W_clnk_chns_star_frame_avrg_so3_quad,
        W_clnk_y_flucts_star_frame_avrg_so3_quad, W_clnk_star_frame_avrg_so3_quad
    )

def W_clnk_chns_frame_avrg(
        y_clnk: npt.NDArray[np.floating],
        omega_clnk: npt.NDArray[np.floating],
        F: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float,
        X_clnk: npt.NDArray[np.floating],
        w_c_func,
        w_c_args: tuple[float] | tuple[None],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float] | tuple[None]) -> float:
    """Nondimensional cross-link polymer chain free energy in the frame
    averaging limit.

    This function supplies the nondimensional cross-link polymer chain
    free energy in the frame averaging limit as a suitable objective
    function for constrained minimization.

    Args:
        y_clnk (npt.NDArray[np.floating]): Cross-link junction position for the cross-link structure RVE.
        omega_clnk (npt.NDArray[np.floating]): Vector of Euler angles for the cross-link structure RVE.
        F (npt.NDArray[np.floating]): Deformation gradient.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        w_c_func (function): Nondimensional polymer chain free energy function.
        w_c_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        w_c_dfrmtn_func (function): Nondimensional polymer chain deformation free energy function.
        w_c_dfrmtn_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
    
    Returns:
        float: Nondimensional cross-link polymer chain free energy in
        the frame averaging limit.
    
    """
    # Calculate the absolute/equilibrium chain stretch for each chain
    # (using the ideal numerics form for the chain end-to-end vector)
    gamma_clnk = gamma_clnk_func(
        True, F, X_clnk, Q_zyz_euler(omega_clnk), y_clnk, n_clnk, b)
    # Calculate the nondimensional cross-link polymer chain free energy
    return (
        W_clnk_chns_func(
            gamma_clnk, n_clnk, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    )

def clnk_frame_avrg_cnstrnd_mnmztn(
        eval_W_clnk_chns: bool,
        eval_W_clnk_y_flucts: bool,
        cnstrnd_mnmztn_scope: str,
        cnstrnd_mnmztn_method: str,
        rng: np.random.Generator,
        F: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float,
        X_clnk: npt.NDArray[np.floating],
        so3_quad: npt.NDArray[np.floating],
        sph_quad_symmtry: bool,
        y_clnk_so3_quad: npt.NDArray[np.floating],
        w_c_func,
        w_c_args: tuple[float] | tuple[None],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float] | tuple[None],
        d2w_c__dy_clnk_dy_clnk_func,
        d2w_c__dy_clnk_dy_clnk_args: tuple[float] | tuple[None]):
    """Cross-link structure RVE mechanical response in the frame
    averaging limit, as evaluated numerically via constrained
    minimization.

    This function determines the mechanical response of a cross-link
    structure RVE in the frame averaging limit, as evaluated numerically
    via constrained minimization.

    Args:
        eval_W_clnk_chns (bool): Boolean indicating if the nondimensional cross-link polymer chain free energy ought to be calculated (if True) or not (if False).
        eval_W_clnk_y_flucts (bool): Boolean indicating if the nondimensional cross-link junction fluctuation free energy ought to be calculated (if True) or not (if False).
        cnstrnd_mnmztn_scope (str): String indicating the use of either local ("lcl") or global ("glbl") constrained minimization.
        cnstrnd_mnmztn_method (str): String indicating the specific constrained minimization method to utilize. See the code and associated error string in the function for the different types of constrained minimization methods available for use.
        rng (np.random.Generator): Numpy random number generator object.
        F (npt.NDArray[np.floating]): Deformation gradient.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        so3_quad (npt.NDArray[np.floating]): SO(3) quadrature scheme.
        sph_quad_symmtry (bool): Boolean indicating if the SO(3) quadrature scheme is hemispherically symmetric.
        y_clnk_so3_quad (npt.NDArray[np.floating]): Prior cross-link junction position for each cross-link SO(3) quadrature orientation.
        w_c_func (function): Nondimensional polymer chain free energy function.
        w_c_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        w_c_dfrmtn_func (function): Nondimensional polymer chain deformation free energy function.
        w_c_dfrmtn_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        d2w_c__dy_clnk_dy_clnk_func (function): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function.
        d2w_c__dy_clnk_dy_clnk_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n).
    
    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, float, float]:
        Optimal cross-link junction position for each cross-link SO(3)
        quadrature orientation, distance between the origin and the
        optimal cross-link junction position for each cross-link SO(3)
        quadrature orientation, absolute/equilibrium chain stretch for
        each chain in the cross-link at each cross-link SO(3) quadrature
        orientation, nondimensional cross-link polymer chain free energy
        for each cross-link SO(3) quadrature orientation, nondimensional
        cross-link junction fluctuation free energy for each cross-link
        SO(3) quadrature orientation, nondimensional cross-link free
        energy for each cross-link SO(3) quadrature orientation, SO(3)
        quadrature optimal cross-link junction position, SO(3)
        quadrature distance between the origin and the optimal
        cross-link junction position, SO(3) quadrature
        absolute/equilibrium chain stretch for each chain in the
        cross-link, SO(3) quadrature nondimensional cross-link polymer
        chain free energy, SO(3) quadrature nondimensional cross-link
        junction fluctuation free energy, SO(3) quadrature
        nondimensional cross-link free energy.
    
    """
    # Initialization
    so3_quad_num = np.shape(so3_quad)[0]
    k_num = np.shape(n_clnk)[0]
    y_clnk_frame_avrg_so3 = np.zeros((so3_quad_num, 3))
    y_clnk_norm_frame_avrg_so3 = np.zeros(so3_quad_num)
    gamma_clnk_frame_avrg_so3 = np.zeros((so3_quad_num, k_num))
    W_clnk_chns_frame_avrg_so3 = np.zeros(so3_quad_num)
    W_clnk_y_flucts_frame_avrg_so3 = np.zeros(so3_quad_num)
    y_clnk_frame_avrg_so3_quad = np.zeros(3)
    y_clnk_norm_frame_avrg_so3_quad = 0.
    gamma_clnk_frame_avrg_so3_quad = np.zeros(k_num)
    W_clnk_chns_frame_avrg_so3_quad = 0.
    W_clnk_y_flucts_frame_avrg_so3_quad = 0.

    # Evaluate the cross-link in each SO(3) quadrature point orientation
    for so3_quad_indx in range(so3_quad_num):
        # Extract prior cross-link junction position associated with
        # this SO(3) quadrature point orientation
        y_clnk_so3 = y_clnk_so3_quad[so3_quad_indx]
        
        # Extract SO(3) quadrature point weight and rotation matrix
        weight_so3 = so3_quad[so3_quad_indx, -1]
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
            omega_clnk_so3, F, n_clnk, b, X_clnk,
            w_c_func, w_c_args, w_c_dfrmtn_func, w_c_dfrmtn_args
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
            True, F, X_clnk, Q_0_clnk_so3, y_clnk_so3, n_clnk, b)
        
        # If called for, calculate each component of the nondimensional
        # cross-link free energy
        W_clnk_chns_so3 = 0.
        W_clnk_y_flucts_so3 = 0.
        if eval_W_clnk_chns:
            W_clnk_chns_so3 = W_clnk_chns_func(
                gamma_clnk_so3, n_clnk, w_c_func, w_c_args,
                w_c_dfrmtn_func, w_c_dfrmtn_args)
        if eval_W_clnk_y_flucts:
            # Calculate the absolute/equilibrium chain stretch vector
            # for each chain
            gamma_vec_clnk_so3 = gamma_vec_clnk_func(
                True, F, X_clnk, Q_0_clnk_so3, y_clnk_so3, n_clnk, b)
            W_clnk_y_flucts_so3 = W_clnk_y_flucts_func(
                gamma_vec_clnk_so3, gamma_clnk_so3, n_clnk,
                d2w_c__dy_clnk_dy_clnk_func, d2w_c__dy_clnk_dy_clnk_args)
        
        # Update SO(3) quadrature arrays
        y_clnk_frame_avrg_so3[so3_quad_indx] = y_clnk_so3
        y_clnk_norm_frame_avrg_so3[so3_quad_indx] = y_clnk_norm_so3
        gamma_clnk_frame_avrg_so3[so3_quad_indx] = gamma_clnk_so3
        W_clnk_chns_frame_avrg_so3[so3_quad_indx] = W_clnk_chns_so3
        W_clnk_y_flucts_frame_avrg_so3[so3_quad_indx] = W_clnk_y_flucts_so3

        # Evaluate SO(3) quadrature
        y_clnk_frame_avrg_so3_quad += weight_so3 * y_clnk_so3
        y_clnk_norm_frame_avrg_so3_quad += weight_so3 * y_clnk_norm_so3
        gamma_clnk_frame_avrg_so3_quad += weight_so3 * gamma_clnk_so3
        W_clnk_chns_frame_avrg_so3_quad += weight_so3 * W_clnk_chns_so3
        W_clnk_y_flucts_frame_avrg_so3_quad += weight_so3 * W_clnk_y_flucts_so3
    
    # If necessary, account for spherical quadrature symmetry
    # considerations
    if sph_quad_symmtry:
        y_clnk_frame_avrg_so3_quad = np.zeros(3)
        y_clnk_norm_frame_avrg_so3_quad *= 2.
        gamma_clnk_frame_avrg_so3_quad *= 2.
        W_clnk_chns_frame_avrg_so3_quad *= 2.
        W_clnk_y_flucts_frame_avrg_so3_quad *= 2.
    
    # Calculate the nondimensional cross-link free energy
    W_clnk_frame_avrg_so3 = (
        W_clnk_chns_frame_avrg_so3 + W_clnk_y_flucts_frame_avrg_so3
    )
    W_clnk_frame_avrg_so3_quad = (
        W_clnk_chns_frame_avrg_so3_quad + W_clnk_y_flucts_frame_avrg_so3_quad
    )
    
    return (
        y_clnk_frame_avrg_so3, y_clnk_norm_frame_avrg_so3,
        gamma_clnk_frame_avrg_so3, W_clnk_chns_frame_avrg_so3,
        W_clnk_y_flucts_frame_avrg_so3, W_clnk_frame_avrg_so3,
        y_clnk_frame_avrg_so3_quad, y_clnk_norm_frame_avrg_so3_quad,
        gamma_clnk_frame_avrg_so3_quad, W_clnk_chns_frame_avrg_so3_quad,
        W_clnk_y_flucts_frame_avrg_so3_quad, W_clnk_frame_avrg_so3_quad
    )

def clnk_frame_avrg_approx(
        eval_W_clnk_chns: bool,
        eval_W_clnk_y_flucts: bool,
        use_inext_gaussian_fjc_delta_clnk: bool,
        F: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float,
        X_clnk: npt.NDArray[np.floating],
        so3_quad: npt.NDArray[np.floating],
        sph_quad_symmtry: bool,
        w_c_func,
        w_c_args: tuple[float] | tuple[None],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float] | tuple[None],
        dw_c__dy_clnk_func,
        dw_c__dy_clnk_args: tuple[float] | tuple[None],
        d2w_c__dy_clnk_dy_clnk_func,
        d2w_c__dy_clnk_dy_clnk_args: tuple[float] | tuple[None]):
    """Cross-link structure RVE mechanical response in the frame
    averaging limit, as evaluated via closed-form approximation.

    This function determines the mechanical response of a cross-link
    structure RVE in the frame averaging limit, as evaluated via
    closed-form approximation.

    Args:
        eval_W_clnk_chns (bool): Boolean indicating if the nondimensional cross-link polymer chain free energy ought to be calculated (if True) or not (if False).
        eval_W_clnk_y_flucts (bool): Boolean indicating if the nondimensional cross-link junction fluctuation free energy ought to be calculated (if True) or not (if False).
        use_inext_gaussian_fjc_delta_clnk (bool): Boolean indicating if the inextensible Gaussian FJC model ought to be used to calculate the optimal cross-link junction position perturbation.
        F (npt.NDArray[np.floating]): Deformation gradient.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        so3_quad (npt.NDArray[np.floating]): SO(3) quadrature scheme.
        sph_quad_symmtry (bool): Boolean indicating if the SO(3) quadrature scheme is hemispherically symmetric.
        w_c_func (function): Nondimensional polymer chain free energy function.
        w_c_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        w_c_dfrmtn_func (function): Nondimensional polymer chain deformation free energy function.
        w_c_dfrmtn_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        dw_c__dy_clnk_func (function): Nondimensional derivative of the polymer chain free energy with respect to the cross-link junction position function.
        dw_c__dy_clnk_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec and the absolute/equilibrium chain stretch gamma).
        d2w_c__dy_clnk_dy_clnk_func (function): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function.
        d2w_c__dy_clnk_dy_clnk_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n).
    
    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, float, float]:
        Optimal cross-link junction position perturbation for each
        cross-link SO(3) quadrature orientation, distance between the
        origin and the optimal cross-link junction position perturbation
        for each cross-link SO(3) quadrature orientation,
        absolute/equilibrium chain stretch for each chain in the
        cross-link at each cross-link SO(3) quadrature orientation,
        nondimensional cross-link polymer chain free energy for each
        cross-link SO(3) quadrature orientation, nondimensional
        cross-link junction fluctuation free energy for each cross-link
        SO(3) quadrature orientation, nondimensional cross-link free
        energy for each cross-link SO(3) quadrature orientation, SO(3)
        quadrature optimal cross-link junction position perturbation,
        SO(3) quadrature distance between the origin and the optimal
        cross-link junction position perturbation, SO(3) quadrature
        absolute/equilibrium chain stretch for each chain in the
        cross-link, SO(3) quadrature nondimensional cross-link polymer
        chain free energy, SO(3) quadrature nondimensional cross-link
        junction fluctuation free energy, SO(3) quadrature
        nondimensional cross-link free energy.
    
    """
    # Boilerplate initialization, checks, and assertions
    k_num = np.shape(n_clnk)[0]
    X_hat_clnk = x_hat_clnk_func(X_clnk)
    com_X_hat_clnk = com_x_clnk_func(X_hat_clnk)
    if (not np.isclose(np.linalg.det(F), 1.) or
        np.allclose(n_clnk, n_clnk[0]*np.ones_like(n_clnk)) or
        not np.allclose(com_X_hat_clnk, np.zeros(3))):
        error_str = (
            "This methodology is only applicable to well-structured "
            + "polydisperse cross-link RVEs under incompressible "
            + "deformation. Make sure that the chains in the "
            + "cross-link are polydisperse in the number of segments, "
            + "the initial center-of-mass of the cross-link composed "
            + "of unit-length chains is located at the origin, and the "
            + "deformation is incompressible."
        )
        raise ValueError(error_str)
    y_clnk_m = monodisperse_y_clnk()
    Lmbda, _ = principal_stretch_decomposition(F)
    lmbda_0, lmbda_1, _ = Lmbda
    F_Lmbda = np.diag(np.asarray([lmbda_0, lmbda_1, 1./(lmbda_0*lmbda_1)]))

    # If called for, gather the functions and function arguments needed
    # to calculate the optimal cross-link junction position perturbation
    # for the case of the inextensible Gaussian FJC model
    if use_inext_gaussian_fjc_delta_clnk:
        inext_gaussian_fjc = "inext_gaussian_fjc"
        dw_c__dy_clnk_func = master_dw_c__dy_clnk_func(inext_gaussian_fjc)
        dw_c__dy_clnk_args = master_dw_c__dy_clnk_args_func(
            inext_gaussian_fjc, 0., 0., tuple([]))
        d2w_c__dy_clnk_dy_clnk_func = master_d2w_c__dy_clnk_dy_clnk_func(
            inext_gaussian_fjc)
        d2w_c__dy_clnk_dy_clnk_args = (
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
    delta_y_clnk_frame_avrg_approx_so3_quad = np.zeros(3)
    delta_y_clnk_norm_frame_avrg_approx_so3_quad = 0.
    gamma_clnk_frame_avrg_approx_so3_quad = np.zeros(k_num)
    W_clnk_chns_frame_avrg_approx_so3_quad = 0.
    W_clnk_y_flucts_frame_avrg_approx_so3_quad = 0.

    # Evaluate the cross-link in each SO(3) quadrature point orientation
    for so3_quad_indx in range(so3_quad_num):
        # Extract SO(3) quadrature point weight and rotation matrix
        weight_so3 = so3_quad[so3_quad_indx, -1]
        omega_clnk_so3 = so3_quad[so3_quad_indx, :-1]
        Q_0_clnk_so3 = Q_zyz_euler(omega_clnk_so3)

        # Calculate the absolute/equilibrium chain stretch and the
        # absolute/equilibrium chain stretch vector for each chain for
        # the case where there is zero cross-link junction position
        # perturbation
        gamma_clnk_so3_0 = gamma_approx_clnk_func(
            F_Lmbda, X_clnk, np.eye(3), y_clnk_m, Q_0_clnk_so3, np.zeros(3),
            n_clnk, b)
        gamma_vec_clnk_so3_0 = gamma_approx_vec_clnk_func(
            F_Lmbda, X_clnk, np.eye(3), y_clnk_m, Q_0_clnk_so3, np.zeros(3),
            n_clnk, b)
        
        # Calculate the nondimensional first and second derivatives of
        # the cross-link polymer chain free energy with respect to the
        # cross-link junction position for the case where there is zero
        # cross-link junction position perturbation
        dW_clnk_chns__dy_clnk_0 = dW_clnk_chns__dy_clnk_func(
            gamma_vec_clnk_so3_0, gamma_clnk_so3_0,
            dw_c__dy_clnk_func, dw_c__dy_clnk_args)
        d2W_clnk_chns__dy_clnk_dy_clnk_0 = d2W_clnk_chns__dy_clnk_dy_clnk_func(
            gamma_vec_clnk_so3_0, gamma_clnk_so3_0, n_clnk,
            d2w_c__dy_clnk_dy_clnk_func, d2w_c__dy_clnk_dy_clnk_args)
        
        # Calculate the optimal cross-link junction position
        # perturbation
        delta_y_clnk_so3 = (
            -1.
            * np.matmul(
                np.linalg.inv(d2W_clnk_chns__dy_clnk_dy_clnk_0),
                dW_clnk_chns__dy_clnk_0)
        )
        delta_y_clnk_norm_so3 = np.linalg.norm(delta_y_clnk_so3)
        
        # Calculate the absolute/equilibrium chain stretch for each
        # chain
        gamma_clnk_so3 = gamma_approx_clnk_func(
            F_Lmbda, X_clnk, np.eye(3), y_clnk_m, Q_0_clnk_so3,
            delta_y_clnk_so3, n_clnk, b)
        
        # If called for, calculate each component of the nondimensional
        # cross-link free energy
        W_clnk_chns_so3 = 0.
        W_clnk_y_flucts_so3 = 0.
        if eval_W_clnk_chns:
            W_clnk_chns_so3 = W_clnk_chns_func(
                gamma_clnk_so3, n_clnk, w_c_func, w_c_args,
                w_c_dfrmtn_func, w_c_dfrmtn_args)
        if eval_W_clnk_y_flucts:
            # Calculate the absolute/equilibrium chain stretch vector
            # for each chain
            gamma_vec_clnk_so3 = gamma_approx_vec_clnk_func(
                F_Lmbda, X_clnk, np.eye(3), y_clnk_m, Q_0_clnk_so3,
                delta_y_clnk_so3, n_clnk, b)
            W_clnk_y_flucts_so3 = W_clnk_y_flucts_func(
                gamma_vec_clnk_so3, gamma_clnk_so3, n_clnk,
                d2w_c__dy_clnk_dy_clnk_func, d2w_c__dy_clnk_dy_clnk_args)
        
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

        # Evaluate SO(3) quadrature
        delta_y_clnk_frame_avrg_approx_so3_quad += (
            weight_so3 * delta_y_clnk_so3
        )
        delta_y_clnk_norm_frame_avrg_approx_so3_quad += (
            weight_so3 * delta_y_clnk_norm_so3
        )
        gamma_clnk_frame_avrg_approx_so3_quad += weight_so3 * gamma_clnk_so3
        W_clnk_chns_frame_avrg_approx_so3_quad += weight_so3 * W_clnk_chns_so3
        W_clnk_y_flucts_frame_avrg_approx_so3_quad += (
            weight_so3 * W_clnk_y_flucts_so3
        )
    
    # If necessary, account for spherical quadrature symmetry
    # considerations
    if sph_quad_symmtry:
        delta_y_clnk_frame_avrg_approx_so3_quad = np.zeros(3)
        delta_y_clnk_norm_frame_avrg_approx_so3_quad *= 2.
        gamma_clnk_frame_avrg_approx_so3_quad *= 2.
        W_clnk_chns_frame_avrg_approx_so3_quad *= 2.
        W_clnk_y_flucts_frame_avrg_approx_so3_quad *= 2.
    
    # Calculate the nondimensional cross-link free energy
    W_clnk_frame_avrg_approx_so3 = (
        W_clnk_chns_frame_avrg_approx_so3
        + W_clnk_y_flucts_frame_avrg_approx_so3
    )
    W_clnk_frame_avrg_approx_so3_quad = (
        W_clnk_chns_frame_avrg_approx_so3_quad
        + W_clnk_y_flucts_frame_avrg_approx_so3_quad
    )

    return (
        delta_y_clnk_frame_avrg_approx_so3,
        delta_y_clnk_norm_frame_avrg_approx_so3,
        gamma_clnk_frame_avrg_approx_so3, W_clnk_chns_frame_avrg_approx_so3,
        W_clnk_y_flucts_frame_avrg_approx_so3, W_clnk_frame_avrg_approx_so3,
        delta_y_clnk_frame_avrg_approx_so3_quad,
        delta_y_clnk_norm_frame_avrg_approx_so3_quad,
        gamma_clnk_frame_avrg_approx_so3_quad,
        W_clnk_chns_frame_avrg_approx_so3_quad,
        W_clnk_y_flucts_frame_avrg_approx_so3_quad,
        W_clnk_frame_avrg_approx_so3_quad
    )