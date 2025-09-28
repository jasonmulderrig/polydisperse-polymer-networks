import numpy as np
from scipy.optimize import (
    NonlinearConstraint,
    Bounds,
    minimize
)
from scipy.optimize import (
    differential_evolution,
    shgo
)
from src.helpers.clnk_structure_utils import (
    amended_3_chn_clnk_X_hat_clnk_func,
    regular_tetrahedral_4_chn_clnk_X_hat_clnk_func,
    equilateral_triangular_bipyramidal_5_chn_clnk_X_hat_clnk_func,
    regular_octahedral_6_chn_clnk_X_hat_clnk_func,
    equilateral_pentagonal_bipyramidal_7_chn_clnk_X_hat_clnk_func,
    cube_8_chn_clnk_X_hat_clnk_func,
    x_hat_clnk_func,
    com_x_clnk_func,
    chull_eqs_clnk_func
)
from src.helpers.continuum_mechanics_utils import (
    principal_stretch_decomposition
)
from src.helpers.clnk_deformation_utils import (
    monodisperse_y_clnk,
    gamma_clnk_func,
    gamma_clnk_approx_func,
    W_clnk_func,
    W_flucts_clnk_func,
    W_flucts_clnk_approx_func
)
from src.helpers.rotations_utils import Q_zyz_euler

def monodisperse_clnk_frame_avrg(
        F: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        so3_quad: np.ndarray,
        sph_quad_symmtry: bool,
        y_clnk_init: np.ndarray,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]):
    
    so3_quad_num = np.shape(so3_quad)[0]
    k_num = np.shape(n_clnk)[0]
    X_hat_clnk = x_hat_clnk_func(X_clnk)
    com_X_clnk = com_x_clnk_func(X_clnk)
    if (not np.all(np.equal(n_clnk, n_clnk[0])) or
        not np.allclose(com_X_clnk, np.zeros(3)) or
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
    
    y_clnk_star = monodisperse_y_clnk()
    y_clnk_star_frame_avrg_so3 = np.zeros((so3_quad_num, 3))
    y_clnk_star_norm_frame_avrg_so3 = np.zeros(so3_quad_num)
    y_clnk_star_frame_avrg_so3_quad = np.zeros(3)
    y_clnk_star_norm_frame_avrg_so3_quad = 0.

    # Initialization
    gamma_clnk_star_frame_avrg_so3 = np.zeros((so3_quad_num, k_num))
    W_clnk_star_frame_avrg_so3 = np.zeros(so3_quad_num)
    gamma_clnk_star_frame_avrg_so3_quad = np.zeros(k_num)
    W_clnk_star_frame_avrg_so3_quad = 0.

    for so3_quad_indx in range(so3_quad_num):
        omega_clnk_so3 = so3_quad[so3_quad_indx, :-1]
        weight_so3 = so3_quad[so3_quad_indx, -1]

        Q_clnk_star_so3 = Q_zyz_euler(omega_clnk_so3)
        gamma_clnk_star_so3 = gamma_clnk_func(
            False, F, n_clnk, b, X_clnk, Q_clnk_star_so3, y_clnk_star)
        
        W_clnk_star_so3 = W_clnk_func(
            gamma_clnk_star_so3, n_clnk, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
        
        gamma_clnk_star_frame_avrg_so3[so3_quad_indx] = gamma_clnk_star_so3
        W_clnk_star_frame_avrg_so3[so3_quad_indx] = W_clnk_star_so3

        gamma_clnk_star_frame_avrg_so3_quad += weight_so3 * gamma_clnk_star_so3
        W_clnk_star_frame_avrg_so3_quad += weight_so3 * W_clnk_star_so3
    
    # If necessary, account for spherical quadrature symmetry
    # considerations
    if sph_quad_symmtry:
        gamma_clnk_star_frame_avrg_so3_quad *= 2.0
        W_clnk_star_frame_avrg_so3_quad *= 2.0
    
    return (
        y_clnk_star_frame_avrg_so3, y_clnk_star_norm_frame_avrg_so3,
        gamma_clnk_star_frame_avrg_so3, W_clnk_star_frame_avrg_so3,
        y_clnk_star_frame_avrg_so3_quad, y_clnk_star_norm_frame_avrg_so3_quad,
        gamma_clnk_star_frame_avrg_so3_quad, W_clnk_star_frame_avrg_so3_quad
    )

def W_flucts_clnk_star_frame_avrg(
        eval_W_flucts: bool,
        F: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        vol_quad_clnk: np.ndarray,
        so3_quad: np.ndarray,
        sph_quad_symmtry: bool,
        y_clnk_init: np.ndarray,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]):
    
    so3_quad_num = np.shape(so3_quad)[0]
    k_num = np.shape(n_clnk)[0]
    X_hat_clnk = x_hat_clnk_func(X_clnk)
    com_X_clnk = com_x_clnk_func(X_clnk)
    if (not np.all(np.equal(n_clnk, n_clnk[0])) or
        not np.allclose(com_X_clnk, np.zeros(3)) or
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
    
    # Initialization
    W_flucts_clnk_star_frame_avrg_so3 = np.zeros(so3_quad_num)
    W_flucts_clnk_star_frame_avrg_so3_quad = 0.

    for so3_quad_indx in range(so3_quad_num):
        omega_clnk_so3 = so3_quad[so3_quad_indx, :-1]
        weight_so3 = so3_quad[so3_quad_indx, -1]
        Q_clnk_star_so3 = Q_zyz_euler(omega_clnk_so3)
        
        if eval_W_flucts:
            W_flucts_clnk_star_so3 = W_flucts_clnk_func(
                False, F, Q_clnk_star_so3, n_clnk, b, X_clnk, vol_quad_clnk,
                w_c_func, w_c_args, w_c_dfrmtn_func, w_c_dfrmtn_args)
        else: W_flucts_clnk_star_so3 = 0.
        
        W_flucts_clnk_star_frame_avrg_so3[so3_quad_indx] = (
            W_flucts_clnk_star_so3
        )
        W_flucts_clnk_star_frame_avrg_so3_quad += (
            weight_so3 * W_flucts_clnk_star_so3
        )
    
    # If necessary, account for spherical quadrature symmetry
    # considerations
    if sph_quad_symmtry: W_flucts_clnk_star_frame_avrg_so3_quad *= 2.0
    
    return (
        W_flucts_clnk_star_frame_avrg_so3,
        W_flucts_clnk_star_frame_avrg_so3_quad
    )

def W_clnk_frame_avrg(
        y_clnk: np.ndarray,
        omega_clnk: np.ndarray,
        F: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]) -> float:
    gamma_clnk = gamma_clnk_func(
        True, F, n_clnk, b, X_clnk, Q_zyz_euler(omega_clnk), y_clnk)
    return (
        W_clnk_func(
            gamma_clnk, n_clnk, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    )

def clnk_frame_avrg_cnstrnd_mnmztn(
        cnstrnd_mnmztn_scope: str,
        cnstrnd_mnmztn_method: str,
        rng: np.random.Generator,
        F: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        so3_quad: np.ndarray,
        sph_quad_symmtry: bool,
        y_clnk_so3_quad: np.ndarray,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]):
    # Initialization
    so3_quad_num = np.shape(so3_quad)[0]
    k_num = np.shape(n_clnk)[0]
    y_clnk_frame_avrg_so3 = np.zeros((so3_quad_num, 3))
    y_clnk_norm_frame_avrg_so3 = np.zeros(so3_quad_num)
    gamma_clnk_frame_avrg_so3 = np.zeros((so3_quad_num, k_num))
    W_clnk_frame_avrg_so3 = np.zeros(so3_quad_num)
    y_clnk_frame_avrg_so3_quad = np.zeros(3)
    y_clnk_norm_frame_avrg_so3_quad = 0.
    gamma_clnk_frame_avrg_so3_quad = np.zeros(k_num)
    W_clnk_frame_avrg_so3_quad = 0.

    for so3_quad_indx in range(so3_quad_num):
        y_clnk_so3 = y_clnk_so3_quad[so3_quad_indx]
        omega_clnk_so3 = so3_quad[so3_quad_indx, :-1]
        weight_so3 = so3_quad[so3_quad_indx, -1]

        # Gather (deformed) cross-link convex hull
        Q_clnk_so3 = Q_zyz_euler(omega_clnk_so3)
        x_clnk = np.empty_like(X_clnk)
        for chn_indx in range(np.shape(n_clnk)[0]):
            x_clnk[chn_indx] = np.matmul(
                F, np.matmul(Q_clnk_so3, X_clnk[chn_indx]))
        A_chull_eqs_clnk, b_chull_eqs_clnk = chull_eqs_clnk_func(x_clnk)

        # Constraints on y_clnk
        def chull_clnk_cnstrnt_func(y_clnk):
            return b_chull_eqs_clnk - np.matmul(A_chull_eqs_clnk, y_clnk)
        y_clnk_cnstrnt = NonlinearConstraint(chull_clnk_cnstrnt_func, 0.0, np.inf)
        
        # Bounds of y_clnk
        y_clnk_bounds = Bounds(np.min(x_clnk, axis=0), np.max(x_clnk, axis=0))

        # Args of the objective function W_clnk_frame_avrg()
        W_clnk_frame_avrg_args = (
            omega_clnk_so3, F, n_clnk, b, X_clnk,
            w_c_func, w_c_args, w_c_dfrmtn_func, w_c_dfrmtn_args
        )

        if cnstrnd_mnmztn_scope == "lcl":
            if cnstrnd_mnmztn_method in ["COBYLA", "COBYQA", "trust-constr"]:
                clnk_frame_avrg = minimize(
                    W_clnk_frame_avrg, y_clnk_so3, args=W_clnk_frame_avrg_args,
                    method=cnstrnd_mnmztn_method, bounds=y_clnk_bounds,
                    constraints=(y_clnk_cnstrnt))
        elif cnstrnd_mnmztn_scope == "glbl":
            if cnstrnd_mnmztn_method == "differential-evolution":
                clnk_frame_avrg = differential_evolution(
                    W_clnk_frame_avrg, y_clnk_bounds,
                    args=W_clnk_frame_avrg_args, rng=rng,
                    constraints=(y_clnk_cnstrnt), x0=y_clnk_so3)
            elif cnstrnd_mnmztn_method == "shgo":
                clnk_frame_avrg = shgo(
                    W_clnk_frame_avrg, y_clnk_bounds,
                    args=W_clnk_frame_avrg_args, constraints=(y_clnk_cnstrnt))
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
        
        y_clnk_so3 = clnk_frame_avrg.x
        y_clnk_norm_so3 = np.linalg.norm(y_clnk_so3)
        gamma_clnk_so3 = gamma_clnk_func(
            True, F, n_clnk, b, X_clnk, Q_clnk_so3, y_clnk_so3)
        
        W_clnk_so3 = W_clnk_func(
            gamma_clnk_so3, n_clnk, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
        
        y_clnk_frame_avrg_so3[so3_quad_indx] = y_clnk_so3
        y_clnk_norm_frame_avrg_so3[so3_quad_indx] = y_clnk_norm_so3
        gamma_clnk_frame_avrg_so3[so3_quad_indx] = gamma_clnk_so3
        W_clnk_frame_avrg_so3[so3_quad_indx] = W_clnk_so3

        y_clnk_frame_avrg_so3_quad += weight_so3 * y_clnk_so3
        y_clnk_norm_frame_avrg_so3_quad += weight_so3 * y_clnk_norm_so3
        gamma_clnk_frame_avrg_so3_quad += weight_so3 * gamma_clnk_so3
        W_clnk_frame_avrg_so3_quad += weight_so3 * W_clnk_so3
    
    # If necessary, account for spherical quadrature symmetry
    # considerations
    if sph_quad_symmtry:
        y_clnk_frame_avrg_so3_quad = np.zeros(3)
        y_clnk_norm_frame_avrg_so3_quad *= 2.0
        gamma_clnk_frame_avrg_so3_quad *= 2.0
        W_clnk_frame_avrg_so3_quad *= 2.0
    
    return (
        y_clnk_frame_avrg_so3, y_clnk_norm_frame_avrg_so3,
        gamma_clnk_frame_avrg_so3, W_clnk_frame_avrg_so3,
        y_clnk_frame_avrg_so3_quad, y_clnk_norm_frame_avrg_so3_quad,
        gamma_clnk_frame_avrg_so3_quad, W_clnk_frame_avrg_so3_quad
    )

def W_flucts_clnk_frame_avrg(
        eval_W_flucts: bool,
        F: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        vol_quad_clnk: np.ndarray,
        so3_quad: np.ndarray,
        sph_quad_symmtry: bool,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]):
    # Initialization
    so3_quad_num = np.shape(so3_quad)[0]
    W_flucts_clnk_frame_avrg_so3 = np.zeros(so3_quad_num)
    W_flucts_clnk_frame_avrg_so3_quad = 0.

    for so3_quad_indx in range(so3_quad_num):
        omega_clnk_so3 = so3_quad[so3_quad_indx, :-1]
        weight_so3 = so3_quad[so3_quad_indx, -1]
        
        Q_clnk_so3 = Q_zyz_euler(omega_clnk_so3)
        
        if eval_W_flucts:
            W_flucts_clnk_so3 = W_flucts_clnk_func(
                True, F, Q_clnk_so3, n_clnk, b, X_clnk, vol_quad_clnk,
                w_c_func, w_c_args, w_c_dfrmtn_func, w_c_dfrmtn_args)
        else: W_flucts_clnk_so3 = 0.
        
        W_flucts_clnk_frame_avrg_so3[so3_quad_indx] = W_flucts_clnk_so3
        W_flucts_clnk_frame_avrg_so3_quad += weight_so3 * W_flucts_clnk_so3
    
    # If necessary, account for spherical quadrature symmetry
    # considerations
    if sph_quad_symmtry: W_flucts_clnk_frame_avrg_so3_quad *= 2.0
    
    return W_flucts_clnk_frame_avrg_so3, W_flucts_clnk_frame_avrg_so3_quad

def clnk_frame_avrg_approx(
        F: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        so3_quad: np.ndarray,
        sph_quad_symmtry: bool,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]):
    k_num = np.shape(n_clnk)[0]
    X_hat_clnk = x_hat_clnk_func(X_clnk)
    com_X_hat_clnk = com_x_clnk_func(X_hat_clnk)
    if (not np.isclose(np.linalg.det(F), 1.) or
        np.all(np.equal(n_clnk, n_clnk[0])) or
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
    dnmntr = np.sum(np.reciprocal(n_clnk*1.0))

    # Initialization
    so3_quad_num = np.shape(so3_quad)[0]
    k_num = np.shape(n_clnk)[0]
    delta_y_clnk_frame_avrg_approx_so3 = np.zeros((so3_quad_num, 3))
    delta_y_clnk_norm_frame_avrg_approx_so3 = np.zeros(so3_quad_num)
    gamma_clnk_frame_avrg_approx_so3 = np.zeros((so3_quad_num, k_num))
    W_clnk_frame_avrg_approx_so3 = np.zeros(so3_quad_num)
    delta_y_clnk_frame_avrg_approx_so3_quad = np.zeros(3)
    delta_y_clnk_norm_frame_avrg_approx_so3_quad = 0.
    gamma_clnk_frame_avrg_approx_so3_quad = np.zeros(k_num)
    W_clnk_frame_avrg_approx_so3_quad = 0.

    for so3_quad_indx in range(so3_quad_num):
        omega_clnk_so3 = so3_quad[so3_quad_indx, :-1]
        weight_so3 = so3_quad[so3_quad_indx, -1]

        Q_clnk_so3 = Q_zyz_euler(omega_clnk_so3)
        delta_y_clnk_so3 = np.zeros(3)
        for chn_indx in range(k_num):
            delta_y_clnk_so3 += (
                np.matmul(F_Lmbda, np.matmul(Q_clnk_so3, X_clnk[chn_indx]))
                / n_clnk[chn_indx]
            )
        delta_y_clnk_so3 /= dnmntr
        delta_y_clnk_norm_so3 = np.linalg.norm(delta_y_clnk_so3)
        gamma_clnk_so3 = gamma_clnk_approx_func(
            F_Lmbda, n_clnk, b, X_clnk, np.eye(3), y_clnk_m, Q_clnk_so3,
            delta_y_clnk_so3)
        
        W_clnk_so3 = W_clnk_func(
            gamma_clnk_so3, n_clnk, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
        
        delta_y_clnk_frame_avrg_approx_so3[so3_quad_indx] = delta_y_clnk_so3
        delta_y_clnk_norm_frame_avrg_approx_so3[so3_quad_indx] = (
            delta_y_clnk_norm_so3
        )
        gamma_clnk_frame_avrg_approx_so3[so3_quad_indx] = gamma_clnk_so3
        W_clnk_frame_avrg_approx_so3[so3_quad_indx] = W_clnk_so3

        delta_y_clnk_frame_avrg_approx_so3_quad += (
            weight_so3 * delta_y_clnk_so3
        )
        delta_y_clnk_norm_frame_avrg_approx_so3_quad += (
            weight_so3 * delta_y_clnk_norm_so3
        )
        gamma_clnk_frame_avrg_approx_so3_quad += weight_so3 * gamma_clnk_so3
        W_clnk_frame_avrg_approx_so3_quad += weight_so3 * W_clnk_so3
    
    # If necessary, account for spherical quadrature symmetry
    # considerations
    if sph_quad_symmtry:
        delta_y_clnk_frame_avrg_approx_so3_quad = np.zeros(3)
        delta_y_clnk_norm_frame_avrg_approx_so3_quad *= 2.0
        gamma_clnk_frame_avrg_approx_so3_quad *= 2.0
        W_clnk_frame_avrg_approx_so3_quad *= 2.0

    return (
        delta_y_clnk_frame_avrg_approx_so3,
        delta_y_clnk_norm_frame_avrg_approx_so3,
        gamma_clnk_frame_avrg_approx_so3, W_clnk_frame_avrg_approx_so3,
        delta_y_clnk_frame_avrg_approx_so3_quad,
        delta_y_clnk_norm_frame_avrg_approx_so3_quad,
        gamma_clnk_frame_avrg_approx_so3_quad,
        W_clnk_frame_avrg_approx_so3_quad
    )

def W_flucts_clnk_frame_avrg_approx(
        eval_W_flucts: bool,
        F: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        vol_quad_clnk: np.ndarray,
        so3_quad: np.ndarray,
        sph_quad_symmtry: bool,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]):
    com_X_clnk = com_x_clnk_func(X_clnk)
    if (not np.isclose(np.linalg.det(F), 1.) or
        np.all(np.equal(n_clnk, n_clnk[0])) or
        not np.allclose(com_X_clnk, np.zeros(3))):
        error_str = (
            "This methodology is only applicable to well-structured "
            + "polydisperse cross-link RVEs under incompressible "
            + "deformation. Make sure that the chains in the "
            + "cross-link are polydisperse in the number of segments, "
            + "the initial center-of-mass of the cross-link is located "
            + "at the origin, and the deformation is incompressible."
        )
        raise ValueError(error_str)
    
    Lmbda, _ = principal_stretch_decomposition(F)
    lmbda_0, lmbda_1, _ = Lmbda
    F_Lmbda = np.diag(np.asarray([lmbda_0, lmbda_1, 1./(lmbda_0*lmbda_1)]))

    # Initialization
    so3_quad_num = np.shape(so3_quad)[0]
    W_flucts_clnk_frame_avrg_approx_so3 = np.zeros(so3_quad_num)
    W_flucts_clnk_frame_avrg_approx_so3_quad = 0.

    for so3_quad_indx in range(so3_quad_num):
        omega_clnk_so3 = so3_quad[so3_quad_indx, :-1]
        weight_so3 = so3_quad[so3_quad_indx, -1]
        Q_clnk_so3 = Q_zyz_euler(omega_clnk_so3)
        
        if eval_W_flucts:
            W_flucts_clnk_so3 = W_flucts_clnk_approx_func(
                F_Lmbda, np.eye(3), Q_clnk_so3, n_clnk, b, X_clnk,
                vol_quad_clnk, w_c_func, w_c_args,
                w_c_dfrmtn_func, w_c_dfrmtn_args)
        else: W_flucts_clnk_so3 = 0.
        
        W_flucts_clnk_frame_avrg_approx_so3[so3_quad_indx] = (
            W_flucts_clnk_so3
        )
        W_flucts_clnk_frame_avrg_approx_so3_quad += (
            weight_so3 * W_flucts_clnk_so3
        )
    
    # If necessary, account for spherical quadrature symmetry
    # considerations
    if sph_quad_symmtry: W_flucts_clnk_frame_avrg_approx_so3_quad *= 2.0

    return (
        W_flucts_clnk_frame_avrg_approx_so3,
        W_flucts_clnk_frame_avrg_approx_so3_quad
    )
