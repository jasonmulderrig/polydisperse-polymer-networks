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
    regular_tetrahedral_4_chn_clnk_X_hat_clnk_func,
    regular_octahedral_6_chn_clnk_X_hat_clnk_func,
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
from src.helpers.rotations_utils import Q_axis_angle
from src.helpers.clnk_structure_dispersity_utils import indcs_permutations

def monodisperse_regular_tetrahedral_4_chn_clnk_free_rot(
        F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, P = principal_stretch_decomposition(F)
    Q_1 = Q_axis_angle(np.asarray([np.pi/4., 0., 0.]))
    Q_2 = Q_axis_angle(np.asarray([0., np.arccos(np.sqrt(2./3.)), 0.]))
    Q_3 = Q_axis_angle(np.asarray([0., 0., -np.pi/2.]))
    Q_clnk_star = np.matmul(Q_1, np.matmul(Q_2, np.matmul(Q_3, P)))

    return Q_clnk_star, monodisperse_y_clnk()

def monodisperse_regular_octahedral_6_chn_clnk_free_rot(
        F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    return np.full_like(F, np.nan), monodisperse_y_clnk()

def monodisperse_cube_8_chn_clnk_free_rot(
        F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    _, P = principal_stretch_decomposition(F)
    return P, monodisperse_y_clnk()

def monodisperse_clnk_free_rot_gamma(
        Lmbda: np.ndarray,
        n_clnk: np.ndarray,
        gamma_clnk_init: np.ndarray) -> np.ndarray:
    return (
        gamma_clnk_init[0] * np.sqrt(np.sum(np.power(Lmbda, 2))/3.)
        * np.ones(np.shape(n_clnk)[0])
    )

def monodisperse_clnk_free_rot(
        eval_W_flucts: bool,
        F: np.ndarray,
        Lmbda: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        vol_quad_clnk: np.ndarray,
        y_clnk_init: np.ndarray,
        gamma_clnk_init: np.ndarray,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]):
    
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
    if k_num == 4:
        clnk = np.allclose(
            X_hat_clnk, regular_tetrahedral_4_chn_clnk_X_hat_clnk_func())
    elif k_num == 6:
        clnk = np.allclose(
            X_hat_clnk, regular_octahedral_6_chn_clnk_X_hat_clnk_func())
    elif k_num == 8:
        clnk = np.allclose(X_hat_clnk, cube_8_chn_clnk_X_hat_clnk_func())
    if not clnk:
        error_str = (
            "This function is only applicable for the regular "
            + "tetrahedral 4-chain cross-link, regular octahedral "
            + "6-chain cross-link, or cube 8-chain cross-link of "
            + "monodisperse segment number. Make sure that the "
            + "cross-link structure corresponds to one of the "
            + "aforementioned cross-link structures."
        )
        raise ValueError(error_str)
    
    if k_num == 4:
        Q_clnk_star, y_clnk_star = (
            monodisperse_regular_tetrahedral_4_chn_clnk_free_rot(F)
        )
    elif k_num == 6:
        Q_clnk_star, y_clnk_star = (
            monodisperse_regular_octahedral_6_chn_clnk_free_rot(F)
        )
    elif k_num == 8:
        Q_clnk_star, y_clnk_star = monodisperse_cube_8_chn_clnk_free_rot(F)
    
    y_clnk_star_norm = np.linalg.norm(y_clnk_star)
    gamma_clnk_star = monodisperse_clnk_free_rot_gamma(
        Lmbda, n_clnk, gamma_clnk_init)
    
    W_clnk_star = W_clnk_func(
        gamma_clnk_star, n_clnk, w_c_func, w_c_args,
        w_c_dfrmtn_func, w_c_dfrmtn_args)
    if eval_W_flucts:
        W_flucts_clnk_star = W_flucts_clnk_func(
            False, F, Q_clnk_star, n_clnk, b, X_clnk, vol_quad_clnk,
            w_c_func, w_c_args, w_c_dfrmtn_func, w_c_dfrmtn_args)
    else: W_flucts_clnk_star = 0.

    return (
        Q_clnk_star, y_clnk_star, y_clnk_star_norm, gamma_clnk_star,
        W_clnk_star, W_flucts_clnk_star
    )

def W_clnk_free_rot(
        omega_clnk_y_clnk: np.ndarray,
        F: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]) -> float:
    omega_clnk, y_clnk = omega_clnk_y_clnk[:3], omega_clnk_y_clnk[3:]
    gamma_clnk = gamma_clnk_func(
        True, F, n_clnk, b, X_clnk, Q_axis_angle(omega_clnk), y_clnk)
    return (
        W_clnk_func(
            gamma_clnk, n_clnk, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    )

def clnk_free_rot_cnstrnd_mnmztn(
        eval_W_flucts: bool,
        cnstrnd_mnmztn_scope: str,
        cnstrnd_mnmztn_method: str,
        rng: np.random.Generator,
        F: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        vol_quad_clnk: np.ndarray,
        omega_clnk: np.ndarray,
        y_clnk: np.ndarray,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]):
    # Gather (deformed) cross-link convex hull
    x_clnk = np.empty_like(X_clnk)
    for chn_indx in range(np.shape(n_clnk)[0]):
        x_clnk[chn_indx] = np.matmul(F, X_clnk[chn_indx])
    A_chull_eqs_clnk, b_chull_eqs_clnk = chull_eqs_clnk_func(x_clnk)
    
    # Horizontally stack omega_clnk and y_clnk together
    omega_clnk_y_clnk = np.hstack((omega_clnk, y_clnk))

    # Constraint on omega_clnk
    def omega_clnk_cnstrnt_func(omega_clnk_y_clnk):
        return np.linalg.norm(omega_clnk_y_clnk[:3])
    omega_clnk_cnstrnt = NonlinearConstraint(
        omega_clnk_cnstrnt_func, 0.0, 2*np.pi)
    
    # Bounds of omega_clnk
    omega_clnk_min_bounds = np.zeros(3)
    omega_clnk_max_bounds = 2 * np.pi * np.ones(3)

    # Constraints on y_clnk
    def chull_clnk_cnstrnt_func(omega_clnk_y_clnk):
        return (
            b_chull_eqs_clnk
            - np.matmul(A_chull_eqs_clnk, omega_clnk_y_clnk[3:])
        )
    y_clnk_cnstrnt = NonlinearConstraint(chull_clnk_cnstrnt_func, 0.0, np.inf)
    
    # Bounds of y_clnk
    y_clnk_min_bounds = np.min(x_clnk, axis=0)
    y_clnk_max_bounds = np.max(x_clnk, axis=0)

    # Bounds of omega_clnk_y_clnk
    omega_clnk_y_clnk_bounds = Bounds(
        np.hstack((omega_clnk_min_bounds, y_clnk_min_bounds)),
        np.hstack((omega_clnk_max_bounds, y_clnk_max_bounds)))
    
    # Args of the objective function W_clnk_free_rot()
    W_clnk_free_rot_args = (
        F, n_clnk, b, X_clnk,
        w_c_func, w_c_args, w_c_dfrmtn_func, w_c_dfrmtn_args
    )

    if cnstrnd_mnmztn_scope == "lcl":
        if cnstrnd_mnmztn_method in ["COBYLA", "COBYQA", "trust-constr"]:
            clnk_free_rot = minimize(
                W_clnk_free_rot, omega_clnk_y_clnk, args=W_clnk_free_rot_args,
                method=cnstrnd_mnmztn_method, bounds=omega_clnk_y_clnk_bounds,
                constraints=(omega_clnk_cnstrnt, y_clnk_cnstrnt))
    elif cnstrnd_mnmztn_scope == "glbl":
        if cnstrnd_mnmztn_method == "differential-evolution":
            clnk_free_rot = differential_evolution(
                W_clnk_free_rot, omega_clnk_y_clnk_bounds,
                args=W_clnk_free_rot_args, rng=rng,
                constraints=(omega_clnk_cnstrnt, y_clnk_cnstrnt),
                x0=omega_clnk_y_clnk)
        elif cnstrnd_mnmztn_method == "shgo":
            clnk_free_rot = shgo(
                W_clnk_free_rot, omega_clnk_y_clnk_bounds,
                args=W_clnk_free_rot_args,
                constraints=(omega_clnk_cnstrnt, y_clnk_cnstrnt))
    else:
        error_str = (
            "Several local and global constrained minimization methods "
            + "are implemented in solving for the deformation of the "
            + "cross-link structure (cnstrnd_mnmztn_scope = ``lcl'' or "
            + " ``glbl''). For local constrained minimization, the "
            + "constrained optimization by linear approximation "
            + "algorithm (cnstrnd_mnmztn_method = ``COBYLA''), the "
            + "constrained optimization by quadratic approximations "
            + "algorithm (cnstrnd_mnmztn_method = ``COBYQA''), and the "
            + "constrained trust-region algorithm "
            + "(cnstrnd_mnmztn_method = ``trust-constr'') are "
            + "implemented. For global constrained minimization, the "
            + "differential evolution method (cnstrnd_mnmztn_method = "
            + "``differential-evolution'') and the simplicial homology "
            + "global optimization method (cnstrnd_mnmztn_method = "
            + "``shgo'') are implemented."
        )
        raise ValueError(error_str)
    
    omega_clnk_y_clnk = clnk_free_rot.x
    omega_clnk, y_clnk = omega_clnk_y_clnk[:3], omega_clnk_y_clnk[3:]
    omega_clnk_norm = np.linalg.norm(omega_clnk)
    y_clnk_norm = np.linalg.norm(y_clnk)
    Q_clnk = Q_axis_angle(omega_clnk)
    gamma_clnk = gamma_clnk_func(True, F, n_clnk, b, X_clnk, Q_clnk, y_clnk)
    
    W_clnk = W_clnk_func(
        gamma_clnk, n_clnk, w_c_func, w_c_args,
        w_c_dfrmtn_func, w_c_dfrmtn_args)
    if eval_W_flucts:
        W_flucts_clnk = W_flucts_clnk_func(
            True, F, Q_clnk, n_clnk, b, X_clnk, vol_quad_clnk,
            w_c_func, w_c_args, w_c_dfrmtn_func, w_c_dfrmtn_args)
    else: W_flucts_clnk = 0.
    
    return (
        omega_clnk, omega_clnk_norm, y_clnk, y_clnk_norm, gamma_clnk,
        W_clnk, W_flucts_clnk
    )

def inext_gaussian_fjc_regular_tetrahedral_4_chn_clnk_free_rot_general_approx_components(
        Lmbda, n_clnk, b):
    lmbda_0, lmbda_1, _ = Lmbda
    n_0, n_1, n_2, n_3 = n_clnk
    prod_n_clnk = n_0 * n_1 * n_2 * n_3
    prod_n_clnk_over_n_0 = prod_n_clnk / n_0
    prod_n_clnk_over_n_1 = prod_n_clnk / n_1
    prod_n_clnk_over_n_2 = prod_n_clnk / n_2
    prod_n_clnk_over_n_3 = prod_n_clnk / n_3
    sqrt_prod_n_clnk = np.sqrt(prod_n_clnk)
    sqrt_prod_n_clnk_over_n_0 = np.sqrt(prod_n_clnk_over_n_0)
    sqrt_prod_n_clnk_over_n_1 = np.sqrt(prod_n_clnk_over_n_1)
    sqrt_prod_n_clnk_over_n_2 = np.sqrt(prod_n_clnk_over_n_2)
    sqrt_prod_n_clnk_over_n_3 = np.sqrt(prod_n_clnk_over_n_3)

    a_0 = (
        sqrt_prod_n_clnk_over_n_0 - sqrt_prod_n_clnk_over_n_1
        - sqrt_prod_n_clnk_over_n_2 + sqrt_prod_n_clnk_over_n_3
    )
    a_1 = (
        sqrt_prod_n_clnk_over_n_0 - sqrt_prod_n_clnk_over_n_1
        + sqrt_prod_n_clnk_over_n_2 - sqrt_prod_n_clnk_over_n_3
    )
    a_2 = (
        -sqrt_prod_n_clnk_over_n_0 + sqrt_prod_n_clnk_over_n_1
        + sqrt_prod_n_clnk_over_n_2 - sqrt_prod_n_clnk_over_n_3
    )
    a_3 = (
        sqrt_prod_n_clnk_over_n_0 + sqrt_prod_n_clnk_over_n_1
        - sqrt_prod_n_clnk_over_n_2 - sqrt_prod_n_clnk_over_n_3
    )
    a_4 = (
        -sqrt_prod_n_clnk_over_n_0 + sqrt_prod_n_clnk_over_n_1
        - sqrt_prod_n_clnk_over_n_2 + sqrt_prod_n_clnk_over_n_3
    )
    a_5 = (
        -sqrt_prod_n_clnk_over_n_0 - sqrt_prod_n_clnk_over_n_1
        + sqrt_prod_n_clnk_over_n_2 + sqrt_prod_n_clnk_over_n_3
    )

    a_0 /= (np.sqrt(3.)*sqrt_prod_n_clnk)
    a_1 /= (np.sqrt(3.)*sqrt_prod_n_clnk)
    a_2 /= (np.sqrt(3.)*sqrt_prod_n_clnk)
    a_3 /= (np.sqrt(3.)*sqrt_prod_n_clnk)
    a_4 /= (np.sqrt(3.)*sqrt_prod_n_clnk)
    a_5 /= (np.sqrt(3.)*sqrt_prod_n_clnk)

    a_6 = (
        prod_n_clnk_over_n_0 + prod_n_clnk_over_n_1 + prod_n_clnk_over_n_2
        + prod_n_clnk_over_n_3
    )
    a_6 /= prod_n_clnk
    
    tilde_delta_omega_clnk_0 = (3*(-4*a_1*a_2*a_6*(3*a_1*a_4 + 4*a_6)*lmbda_0 - 4*a_1*a_2*a_6*(3*a_3*a_5 + 4*a_6 \
    + (3*a_0*a_2 + 3*a_1*a_4 - 3*a_5**2 + 4*a_6)*lmbda_0**3)*lmbda_1 - (9*(a_0*a_3*a_4 \
    + a_1*a_2*a_5)*(a_0*a_2*a_3 + a_5*(a_1**2 - a_3*a_5)) + 12*(a_0*a_1*(a_2**2 + a_1*a_4) \
    + a_0*a_3*(a_1 - a_4)*a_5 + a_1*a_2*(a_3 - a_5)*a_5)*a_6 + 16*a_1*(a_0 \
    + a_2)*a_6**2)*lmbda_0**2*lmbda_1**2 - 4*a_0*a_6*lmbda_0*(a_1*(3*a_3*a_5 + 4*a_6) \
    + (3*a_0*a_1*a_2 + 3*a_1**2*a_4 - 3*a_3*a_4*a_5 + 4*a_1*a_6)*lmbda_0**3)*lmbda_1**3 \
    - 4*a_0*a_1*a_6*(3*a_0*a_2 + 4*a_6)*lmbda_0**3*lmbda_1**4))/(4*a_6*(3*a_1*a_4 \
    + 4*a_6)*(3*a_1*a_4 + 3*a_3*a_5 + 4*a_6)*lmbda_0 + 4*a_6*((3*a_3*a_5 \
    + 4*a_6)*(3*a_1*a_4 + 3*a_3*a_5 + 4*a_6) + (3*a_1*a_4 + 4*a_6)*(3*a_0*a_2 \
    + 3*a_1*a_4 + 4*a_6)*lmbda_0**3)*lmbda_1 + (27*(a_0*a_3*a_4 + a_1*a_2*a_5)**2 \
    + 72*(a_1*a_3*a_4*a_5 + a_0*a_2*(a_1*a_4 + a_3*a_5))*a_6 + 96*(a_0*a_2 + a_1*a_4 \
    + a_3*a_5)*a_6**2 + 128*a_6**3)*lmbda_0**2*lmbda_1**2 + 4*a_6*lmbda_0*((3*a_3*a_5 \
    + 4*a_6)*(3*a_0*a_2 + 3*a_3*a_5 + 4*a_6) + (3*a_0*a_2 + 4*a_6)*(3*a_0*a_2 \
    + 3*a_1*a_4 + 4*a_6)*lmbda_0**3)*lmbda_1**3 + 4*a_6*(3*a_0*a_2 + 4*a_6)*(3*a_0*a_2 \
    + 3*a_3*a_5 + 4*a_6)*lmbda_0**3*lmbda_1**4)
    
    tilde_delta_omega_clnk_1 = (-3*(-9*a_1**3*a_2*a_4*a_5*lmbda_0**2*lmbda_1**2 \
    + 9*a_0*a_3**2*a_4**2*a_5*lmbda_0**2*lmbda_1**2 \
    + 12*a_0*a_2**2*a_6*lmbda_0*lmbda_1**2*(lmbda_0 + lmbda_1)*(a_3 \
    + a_5*lmbda_0**2*lmbda_1) + 3*a_1*a_4*lmbda_0*(4*a_2*a_3*a_6 \
    + 4*a_2*a_5*a_6*lmbda_0**2*lmbda_1 + (3*a_2*a_3*(a_0*a_2 + a_5**2) - 4*a_0*a_3*a_6 \
    + 4*a_2*(a_3 + a_5)*a_6)*lmbda_0*lmbda_1**2 - 4*a_0*a_3*a_6*lmbda_1**3) \
    + 4*a_2*a_6*(a_3 + a_5*lmbda_0**2*lmbda_1)*(1 \
    + lmbda_0*lmbda_1**2)*(3*a_3*a_5*lmbda_1 + 4*a_6*(lmbda_0 + lmbda_1)) \
    - 3*a_1**2*lmbda_0**2*lmbda_1**2*(3*a_0*a_3*a_4**2 - 3*a_2**3*a_5 + 4*a_2*a_5*(a_6 \
    + a_6*lmbda_0*lmbda_1**2))))/(4*a_6*(lmbda_0 + lmbda_1)*(4*a_6 + (3*a_0*a_2 \
    + 4*a_6)*lmbda_0**2*lmbda_1)*(4*a_6 + (3*a_0*a_2 + 4*a_6)*lmbda_0*lmbda_1**2) \
    + 9*a_1**2*lmbda_0*(3*a_2**2*a_5**2*lmbda_0*lmbda_1**2 + 4*a_4**2*(a_6 \
    + a_6*lmbda_0**2*lmbda_1)) + 9*a_3**2*lmbda_1*(3*a_0**2*a_4**2*lmbda_0**2*lmbda_1 \
    + 4*a_5**2*(a_6 + a_6*lmbda_0*lmbda_1**2)) + 12*a_3*a_5*a_6*(4*a_6*(1 \
    + lmbda_0*lmbda_1**2)*(lmbda_0 + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) \
    + 3*a_0*a_2*lmbda_0*lmbda_1**2*(lmbda_1 + lmbda_0*(2 + lmbda_0*lmbda_1**2))) \
    + 6*a_1*a_4*(3*a_3*a_5*(2*a_6*lmbda_0 + 2*a_6*lmbda_1 + (3*a_0*a_2 \
    + 4*a_6)*lmbda_0**2*lmbda_1**2) + 2*a_6*(3*a_0*a_2*lmbda_0**2*lmbda_1*(lmbda_0 \
    + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) + 4*a_6*(1 + lmbda_0**2*lmbda_1)*(lmbda_1 \
    + lmbda_0*(2 + lmbda_0*lmbda_1**2)))))
    
    tilde_delta_omega_clnk_2 = (3*(-4*a_4*a_6*(-3*a_0*a_2*a_3 + a_5*(3*a_1*a_4 + 3*a_3*a_5 + 4*a_6))*lmbda_0 \
    - 4*a_5*a_6*(a_1*(-3*a_2**2 + 3*a_1*a_4 + 3*a_3*a_5 + 4*a_6) + a_4*(3*a_1*a_4 \
    + 4*a_6)*lmbda_0**3)*lmbda_1 - (9*(a_0*a_3*a_4 + a_1*a_2*a_5)*(-(a_0*a_2**2) \
    + a_0*a_1*a_4 + a_2*a_5**2) + 12*(a_5*(-(a_1*a_2**2) + a_1**2*a_4 + a_3*a_4*a_5) \
    + a_0*a_2*(-(a_3*a_4) + (a_1 + a_4)*a_5))*a_6 + 16*(a_1 \
    + a_4)*a_5*a_6**2)*lmbda_0**2*lmbda_1**2 - 4*a_5*a_6*lmbda_0*(a_1*(3*a_3*a_5 + 4*a_6) \
    + a_4*(3*a_0*a_2 + 4*a_6)*lmbda_0**3)*lmbda_1**3 - 4*a_1*a_5*a_6*(3*a_0*a_2 \
    + 4*a_6)*lmbda_0**3*lmbda_1**4))/(4*a_6*(3*a_1*a_4 + 4*a_6)*(3*a_1*a_4 + 3*a_3*a_5 \
    + 4*a_6)*lmbda_0 + 4*a_6*((3*a_3*a_5 + 4*a_6)*(3*a_1*a_4 + 3*a_3*a_5 + 4*a_6) \
    + (3*a_1*a_4 + 4*a_6)*(3*a_0*a_2 + 3*a_1*a_4 + 4*a_6)*lmbda_0**3)*lmbda_1 \
    + (27*(a_0*a_3*a_4 + a_1*a_2*a_5)**2 + 72*(a_1*a_3*a_4*a_5 + a_0*a_2*(a_1*a_4 \
    + a_3*a_5))*a_6 + 96*(a_0*a_2 + a_1*a_4 + a_3*a_5)*a_6**2 \
    + 128*a_6**3)*lmbda_0**2*lmbda_1**2 + 4*a_6*lmbda_0*((3*a_3*a_5 + 4*a_6)*(3*a_0*a_2 \
    + 3*a_3*a_5 + 4*a_6) + (3*a_0*a_2 + 4*a_6)*(3*a_0*a_2 + 3*a_1*a_4 \
    + 4*a_6)*lmbda_0**3)*lmbda_1**3 + 4*a_6*(3*a_0*a_2 + 4*a_6)*(3*a_0*a_2 + 3*a_3*a_5 \
    + 4*a_6)*lmbda_0**3*lmbda_1**4)
    
    tilde_delta_omega_clnk = np.asarray(
        [
            tilde_delta_omega_clnk_0,
            tilde_delta_omega_clnk_1,
            tilde_delta_omega_clnk_2
        ])
    tilde_delta_omega_clnk_norm = np.linalg.norm(tilde_delta_omega_clnk)

    tilde_delta_y_clnk_0 = (-4*b*lmbda_0*(-9*a_0**2*a_3*(a_2**2 - a_1*a_4)*lmbda_0*lmbda_1**2*(lmbda_0 \
    + lmbda_1) - 3*a_0*a_2*(-4*a_5*a_6*lmbda_0*lmbda_1**2*(lmbda_0 + lmbda_1)*(1 \
    + lmbda_0**2*lmbda_1) + 3*a_3**2*a_5*lmbda_1*(1 + lmbda_0*lmbda_1**2) \
    + a_3*(lmbda_0 + lmbda_1)*(4*a_6 + (-3*a_5**2 + 4*a_6)*lmbda_0*lmbda_1**2)) \
    + a_5*(-9*a_1**3*a_4*lmbda_1*(1 + lmbda_0**2*lmbda_1) + 3*a_1*a_4*(1 \
    + lmbda_0**2*lmbda_1)*(3*a_3*a_5*lmbda_1 + 4*a_6*(lmbda_0 + lmbda_1)) + (1 \
    + lmbda_0*lmbda_1**2)*(3*a_3*a_5*lmbda_1 + 4*a_6*(lmbda_0 + lmbda_1))*(3*a_3*a_5 \
    + 4*(a_6 + a_6*lmbda_0**2*lmbda_1)) - 3*a_1**2*lmbda_1*(-3*a_2**2*(1 \
    + lmbda_0**2*lmbda_1) + (3*a_3*a_5 + 4*a_6 + 4*a_6*lmbda_0**2*lmbda_1)*(1 \
    + lmbda_0*lmbda_1**2)))))/(4*a_6*(lmbda_0 + lmbda_1)*(4*a_6 + (3*a_0*a_2 \
    + 4*a_6)*lmbda_0**2*lmbda_1)*(4*a_6 + (3*a_0*a_2 + 4*a_6)*lmbda_0*lmbda_1**2) \
    + 9*a_1**2*lmbda_0*(3*a_2**2*a_5**2*lmbda_0*lmbda_1**2 + 4*a_4**2*(a_6 \
    + a_6*lmbda_0**2*lmbda_1)) + 9*a_3**2*lmbda_1*(3*a_0**2*a_4**2*lmbda_0**2*lmbda_1 \
    + 4*a_5**2*(a_6 + a_6*lmbda_0*lmbda_1**2)) + 12*a_3*a_5*a_6*(4*a_6*(1 \
    + lmbda_0*lmbda_1**2)*(lmbda_0 + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) \
    + 3*a_0*a_2*lmbda_0*lmbda_1**2*(lmbda_1 + lmbda_0*(2 + lmbda_0*lmbda_1**2))) \
    + 6*a_1*a_4*(3*a_3*a_5*(2*a_6*lmbda_0 + 2*a_6*lmbda_1 + (3*a_0*a_2 \
    + 4*a_6)*lmbda_0**2*lmbda_1**2) + 2*a_6*(3*a_0*a_2*lmbda_0**2*lmbda_1*(lmbda_0 \
    + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) + 4*a_6*(1 + lmbda_0**2*lmbda_1)*(lmbda_1 \
    + lmbda_0*(2 + lmbda_0*lmbda_1**2)))))
    
    tilde_delta_y_clnk_1 = (4*b*lmbda_1*(-9*a_1**3*a_4**2*lmbda_0*(1 + lmbda_0**2*lmbda_1) \
    + 3*a_3*a_4*lmbda_0*(1 + lmbda_0*lmbda_1**2)*(-3*a_0*a_2*a_3 + a_5*(3*a_3*a_5 \
    + 4*(a_6 + a_6*lmbda_0**2*lmbda_1))) + a_1*(3*a_3*a_5*(3*a_4**2*(lmbda_0 \
    + lmbda_0**3*lmbda_1) - 4*a_6*(lmbda_0 + lmbda_1)*(1 + lmbda_0*lmbda_1**2)) \
    - (lmbda_0 + lmbda_1)*(-9*a_0*a_2**3*lmbda_0**2*lmbda_1 - 3*a_2**2*(4*a_6 \
    + (-3*a_5**2 + 4*a_6)*lmbda_0**2*lmbda_1) + 12*a_0*a_2*a_6*lmbda_0**2*lmbda_1*(1 \
    + lmbda_0*lmbda_1**2) + 16*a_6**2*(1 + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2))) \
    - 3*a_1**2*a_4*(3*a_0*a_2*lmbda_0**2*lmbda_1*(lmbda_0 + lmbda_1) \
    - 3*a_2**2*(lmbda_0 + lmbda_0**3*lmbda_1) + 3*a_3*a_5*lmbda_0*(1 \
    + lmbda_0*lmbda_1**2) + 4*a_6*(1 + lmbda_0**2*lmbda_1)*(lmbda_1 + lmbda_0*(2 \
    + lmbda_0*lmbda_1**2)))))/(4*a_6*(3*a_1*a_4 + 4*a_6)*(3*a_1*a_4 + 3*a_3*a_5 \
    + 4*a_6)*lmbda_0 + 4*a_6*((3*a_3*a_5 + 4*a_6)*(3*a_1*a_4 + 3*a_3*a_5 + 4*a_6) \
    + (3*a_1*a_4 + 4*a_6)*(3*a_0*a_2 + 3*a_1*a_4 + 4*a_6)*lmbda_0**3)*lmbda_1 \
    + (27*(a_0*a_3*a_4 + a_1*a_2*a_5)**2 + 72*(a_1*a_3*a_4*a_5 + a_0*a_2*(a_1*a_4 \
    + a_3*a_5))*a_6 + 96*(a_0*a_2 + a_1*a_4 + a_3*a_5)*a_6**2 \
    + 128*a_6**3)*lmbda_0**2*lmbda_1**2 + 4*a_6*lmbda_0*((3*a_3*a_5 + 4*a_6)*(3*a_0*a_2 \
    + 3*a_3*a_5 + 4*a_6) + (3*a_0*a_2 + 4*a_6)*(3*a_0*a_2 + 3*a_1*a_4 \
    + 4*a_6)*lmbda_0**3)*lmbda_1**3 + 4*a_6*(3*a_0*a_2 + 4*a_6)*(3*a_0*a_2 + 3*a_3*a_5 \
    + 4*a_6)*lmbda_0**3*lmbda_1**4)
    
    tilde_delta_y_clnk_2 = (-4*b*(9*a_0**2*a_2**3*lmbda_0**3*lmbda_1**3*(lmbda_0 + lmbda_1) \
    + 9*a_0*a_3*a_4**2*a_5*lmbda_0**2*lmbda_1**2*(1 + lmbda_0**2*lmbda_1) + a_2*(1 \
    + lmbda_0*lmbda_1**2)*(3*a_3*a_5*lmbda_1 + 4*a_6*(lmbda_0 \
    + lmbda_1))*(-3*a_5**2*lmbda_0**2*lmbda_1 + 4*a_6*(1 + lmbda_0**2*lmbda_1)) \
    - 9*a_1**2*lmbda_0**2*lmbda_1**2*(a_0*a_4**2*(1 + lmbda_0**2*lmbda_1) - a_2*a_5**2*(1 \
    + lmbda_0*lmbda_1**2)) \
    + 3*a_1*a_4*lmbda_0*(-3*a_0**2*a_2*lmbda_0**2*lmbda_1**3*(lmbda_0 + lmbda_1) \
    + 3*a_0*a_2**2*lmbda_0*lmbda_1**2*(1 + lmbda_0**2*lmbda_1) \
    - 4*a_0*a_6*lmbda_1**2*(lmbda_0 + lmbda_1)*(1 + lmbda_0**2*lmbda_1) + 4*a_2*a_6*(1 \
    + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2)) \
    + 3*a_0*a_2**2*lmbda_0*lmbda_1*(4*a_6*(lmbda_0 + lmbda_1)*(lmbda_0 + lmbda_1 \
    + 2*lmbda_0**2*lmbda_1**2) + 3*a_5*lmbda_0*lmbda_1*(a_3 + a_3*lmbda_0*lmbda_1**2 \
    - a_5*lmbda_0*lmbda_1*(lmbda_0 + lmbda_1)))))/(lmbda_0*lmbda_1*(4*a_6*(lmbda_0 \
    + lmbda_1)*(4*a_6 + (3*a_0*a_2 + 4*a_6)*lmbda_0**2*lmbda_1)*(4*a_6 + (3*a_0*a_2 \
    + 4*a_6)*lmbda_0*lmbda_1**2) + 9*a_1**2*lmbda_0*(3*a_2**2*a_5**2*lmbda_0*lmbda_1**2 \
    + 4*a_4**2*(a_6 + a_6*lmbda_0**2*lmbda_1)) \
    + 9*a_3**2*lmbda_1*(3*a_0**2*a_4**2*lmbda_0**2*lmbda_1 + 4*a_5**2*(a_6 \
    + a_6*lmbda_0*lmbda_1**2)) + 12*a_3*a_5*a_6*(4*a_6*(1 \
    + lmbda_0*lmbda_1**2)*(lmbda_0 + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) \
    + 3*a_0*a_2*lmbda_0*lmbda_1**2*(lmbda_1 + lmbda_0*(2 + lmbda_0*lmbda_1**2))) \
    + 6*a_1*a_4*(3*a_3*a_5*(2*a_6*lmbda_0 + 2*a_6*lmbda_1 + (3*a_0*a_2 \
    + 4*a_6)*lmbda_0**2*lmbda_1**2) + 2*a_6*(3*a_0*a_2*lmbda_0**2*lmbda_1*(lmbda_0 \
    + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) + 4*a_6*(1 + lmbda_0**2*lmbda_1)*(lmbda_1 \
    + lmbda_0*(2 + lmbda_0*lmbda_1**2))))))
    
    tilde_delta_y_clnk = np.asarray(
        [
            tilde_delta_y_clnk_0,
            tilde_delta_y_clnk_1,
            tilde_delta_y_clnk_2
        ])
    tilde_delta_y_clnk_norm = np.linalg.norm(tilde_delta_y_clnk)
    
    return (
        tilde_delta_omega_clnk, tilde_delta_omega_clnk_norm,
        tilde_delta_y_clnk, tilde_delta_y_clnk_norm
    )

def inext_gaussian_fjc_regular_tetrahedral_4_chn_clnk_free_rot_general_approx(
        Lmbda, n_clnk, b):
    n_clnk_indcs_permutations = indcs_permutations(np.shape(n_clnk)[0])
    n_clnk_permutations = n_clnk[n_clnk_indcs_permutations]
    num_permutations = np.shape(n_clnk_permutations)[0]
    tilde_delta_omega_clnk_permutations = np.zeros((num_permutations, 3))
    tilde_delta_omega_clnk_norm_permutations = np.zeros(num_permutations)
    tilde_delta_y_clnk_permutations = np.zeros((num_permutations, 3))
    tilde_delta_y_clnk_norm_permutations = np.zeros(num_permutations)

    for prmttn in range(num_permutations):
        (tilde_delta_omega_clnk, tilde_delta_omega_clnk_norm,
         tilde_delta_y_clnk, tilde_delta_y_clnk_norm) = (
            inext_gaussian_fjc_regular_tetrahedral_4_chn_clnk_free_rot_general_approx_components(
                Lmbda, n_clnk_permutations[prmttn], b)
        )
        tilde_delta_omega_clnk_permutations[prmttn] = tilde_delta_omega_clnk
        tilde_delta_omega_clnk_norm_permutations[prmttn] = (
            tilde_delta_omega_clnk_norm
        )
        tilde_delta_y_clnk_permutations[prmttn] = tilde_delta_y_clnk
        tilde_delta_y_clnk_norm_permutations[prmttn] = tilde_delta_y_clnk_norm
    
    return (
        n_clnk_permutations, tilde_delta_omega_clnk_permutations,
        tilde_delta_omega_clnk_norm_permutations,
        tilde_delta_y_clnk_permutations, tilde_delta_y_clnk_norm_permutations
    )

def inext_gaussian_fjc_1_3_bimodal_regular_tetrahedral_4_chn_clnk_free_rot_approx(
        Lmbda, n_clnk, b):
    unique_n_clnk, unique_n_clnk_counts = np.unique(n_clnk, return_counts=True)
    if (np.shape(n_clnk)[0] != 4 and np.shape(unique_n_clnk)[0] != 2 and
        (unique_n_clnk_counts[0] != 1 or unique_n_clnk_counts[0] != 3)):
        error_str = (
            "This function is only applicable to regular tetrahedral "
            + "4-chain cross-links with a bimodal number of chain "
            + "segment numbers where there are 3 chains with one "
            + "segment number and  chain with the other segment number."
        )
        raise ValueError(error_str)
    
    lmbda_0, lmbda_1, _ = Lmbda
    n_0, n_1 = unique_n_clnk
    n_counts_0, _ = unique_n_clnk_counts
    if n_counts_0 == 1: n_alpha, n_beta = n_0, n_1
    elif n_counts_0 == 3: n_beta, n_alpha = n_0, n_1
    a_0 = (np.sqrt(n_alpha)-np.sqrt(n_beta)) / np.sqrt(3.*n_alpha*n_beta)
    a_1 = (3.*n_alpha+n_beta) / (n_alpha*n_beta)
    
    num_permutations = 4
    tilde_delta_omega_clnk_permutations = np.zeros((num_permutations, 3))
    tilde_delta_y_clnk_permutations = np.zeros((num_permutations, 3))
    n_clnk_permutations = np.tile(n_clnk, (num_permutations, 1))
    s = np.asarray([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=int)

    tilde_delta_omega_clnk_0_term = (3*a_0**2*(-1 + lmbda_0*lmbda_1**2)*(-4*a_1*(lmbda_0 + lmbda_1)*(1 \
    + lmbda_0**2*lmbda_1) + 3*a_0**2*(lmbda_0 + lmbda_1 \
    + lmbda_0**2*lmbda_1*(3*lmbda_0 + lmbda_1))))/(2*(-18*a_0**2*a_1*(lmbda_0 \
    + lmbda_1)*(1 + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2) + 8*a_1**2*(lmbda_0 \
    + lmbda_1)*(1 + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2) + 9*a_0**4*(lmbda_0 \
    + lmbda_1 + lmbda_0**2*lmbda_1**2)*(1 + lmbda_0*lmbda_1*(lmbda_0 + lmbda_1))))
    
    tilde_delta_omega_clnk_1_term = (3*a_0**2*(-1 + lmbda_0**2*lmbda_1)*(-4*a_1*(lmbda_0 + lmbda_1)*(1 \
    + lmbda_0*lmbda_1**2) + 3*a_0**2*(lmbda_0 + lmbda_1 + lmbda_0*lmbda_1**2*(lmbda_0 \
    + 3*lmbda_1))))/(2*(-18*a_0**2*a_1*(lmbda_0 + lmbda_1)*(1 \
    + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2) + 8*a_1**2*(lmbda_0 + lmbda_1)*(1 \
    + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2) + 9*a_0**4*(lmbda_0 + lmbda_1 \
    + lmbda_0**2*lmbda_1**2)*(1 + lmbda_0*lmbda_1*(lmbda_0 + lmbda_1))))
    
    tilde_delta_omega_clnk_2_term = (3*a_0**2*(lmbda_0 - lmbda_1)*(-4*a_1*(1 + lmbda_0**2*lmbda_1)*(1 \
    + lmbda_0*lmbda_1**2) + 3*a_0**2*(3 + lmbda_0*lmbda_1*(lmbda_0 + lmbda_1 \
    + lmbda_0**2*lmbda_1**2))))/(2*(-18*a_0**2*a_1*(lmbda_0 + lmbda_1)*(1 \
    + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2) + 8*a_1**2*(lmbda_0 + lmbda_1)*(1 \
    + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2) + 9*a_0**4*(lmbda_0 + lmbda_1 \
    + lmbda_0**2*lmbda_1**2)*(1 + lmbda_0*lmbda_1*(lmbda_0 + lmbda_1))))

    tilde_delta_y_clnk_0_term = (a_0*b*lmbda_0*(8*a_1**2*(lmbda_0 + lmbda_1)*(1 + lmbda_0**2*lmbda_1)*(1 \
    + lmbda_0*lmbda_1**2) + 27*a_0**4*lmbda_1*(1 + lmbda_0*lmbda_1*(lmbda_0 + lmbda_1)) \
    - 6*a_0**2*a_1*(1 + lmbda_0*lmbda_1**2)*(5*lmbda_1 + lmbda_0*(3 \
    + lmbda_0*lmbda_1*(lmbda_0 + 3*lmbda_1)))))/(a_1*(-18*a_0**2*a_1*(lmbda_0 \
    + lmbda_1)*(1 + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2) + 8*a_1**2*(lmbda_0 \
    + lmbda_1)*(1 + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2) + 9*a_0**4*(lmbda_0 \
    + lmbda_1 + lmbda_0**2*lmbda_1**2)*(1 + lmbda_0*lmbda_1*(lmbda_0 + lmbda_1))))
    
    tilde_delta_y_clnk_1_term = (a_0*b*lmbda_1*(8*a_1**2*(lmbda_0 + lmbda_1)*(1 + lmbda_0**2*lmbda_1)*(1 \
    + lmbda_0*lmbda_1**2) + 27*a_0**4*lmbda_0*(1 + lmbda_0*lmbda_1*(lmbda_0 + lmbda_1)) \
    - 6*a_0**2*a_1*(1 + lmbda_0**2*lmbda_1)*(3*lmbda_1 + lmbda_0*(5 \
    + 3*lmbda_0*lmbda_1**2 + lmbda_1**3))))/(a_1*(-18*a_0**2*a_1*(lmbda_0 \
    + lmbda_1)*(1 + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2) + 8*a_1**2*(lmbda_0 \
    + lmbda_1)*(1 + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2) + 9*a_0**4*(lmbda_0 \
    + lmbda_1 + lmbda_0**2*lmbda_1**2)*(1 + lmbda_0*lmbda_1*(lmbda_0 + lmbda_1))))
    
    tilde_delta_y_clnk_2_term = (a_0*b*(8*a_1**2*(lmbda_0 + lmbda_1)*(1 + lmbda_0**2*lmbda_1)*(1 \
    + lmbda_0*lmbda_1**2) + 27*a_0**4*lmbda_0**2*lmbda_1**2*(1 \
    + lmbda_0*lmbda_1*(lmbda_0 + lmbda_1)) - 6*a_0**2*a_1*(lmbda_0 + lmbda_1)*(1 \
    + lmbda_0*lmbda_1*(3*lmbda_1 + lmbda_0*(3 \
    + 5*lmbda_0*lmbda_1**2)))))/(a_1*lmbda_0*lmbda_1*(-18*a_0**2*a_1*(lmbda_0 \
    + lmbda_1)*(1 + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2) + 8*a_1**2*(lmbda_0 \
    + lmbda_1)*(1 + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2) + 9*a_0**4*(lmbda_0 \
    + lmbda_1 + lmbda_0**2*lmbda_1**2)*(1 + lmbda_0*lmbda_1*(lmbda_0 + lmbda_1))))

    for prmttn in range(num_permutations):
        s_1, s_2 = s[prmttn]
        tilde_delta_omega_clnk_0 = s_1 * s_2 * tilde_delta_omega_clnk_0_term
        tilde_delta_omega_clnk_1 = s_1 * tilde_delta_omega_clnk_1_term
        tilde_delta_omega_clnk_2 = s_2 * tilde_delta_omega_clnk_2_term
        tilde_delta_y_clnk_0 = -s_1 * s_2 * tilde_delta_y_clnk_0_term
        tilde_delta_y_clnk_1 = s_1 * tilde_delta_y_clnk_1_term
        tilde_delta_y_clnk_2 = -s_2 * tilde_delta_y_clnk_2_term
        tilde_delta_omega_clnk_permutations[prmttn] = np.asarray(
            [
                tilde_delta_omega_clnk_0,
                tilde_delta_omega_clnk_1,
                tilde_delta_omega_clnk_2
            ])
        tilde_delta_y_clnk_permutations[prmttn] = np.asarray(
            [
                tilde_delta_y_clnk_0,
                tilde_delta_y_clnk_1,
                tilde_delta_y_clnk_2
            ])
    tilde_delta_omega_clnk_norm_permutations = np.linalg.norm(
        tilde_delta_omega_clnk_permutations, axis=1)
    tilde_delta_y_clnk_norm_permutations = np.linalg.norm(
        tilde_delta_y_clnk_permutations, axis=1)

    return (
        n_clnk_permutations, tilde_delta_omega_clnk_permutations,
        tilde_delta_omega_clnk_norm_permutations,
        tilde_delta_y_clnk_permutations, tilde_delta_y_clnk_norm_permutations
    )

def inext_gaussian_fjc_2_2_bimodal_regular_tetrahedral_4_chn_clnk_free_rot_approx(
        Lmbda, n_clnk, b):
    unique_n_clnk, unique_n_clnk_counts = np.unique(n_clnk, return_counts=True)
    if (np.shape(n_clnk)[0] != 4 and np.shape(unique_n_clnk)[0] != 2 and
        unique_n_clnk_counts[0] != 2):
        error_str = (
            "This function is only applicable to regular tetrahedral "
            + "4-chain cross-links with a bimodal number of chain "
            + "segment numbers where there are 2 chains with each "
            + "segment number."
        )
        raise ValueError(error_str)
    
    n_0, n_1 = unique_n_clnk
    if n_0 < n_1: n_a, n_b = n_0, n_1
    else: n_b, n_a = n_0, n_1
    n_geo_mean = np.power(np.prod(n_clnk), 1./np.shape(n_clnk)[0]) # = np.sqrt(n_a*n_b)
    n_mean = np.mean(n_clnk)
    eta = n_geo_mean / n_mean
    
    lmbda_0, lmbda_1, lmbda_2 = Lmbda
    num_permutations = 6
    tilde_delta_omega_clnk_permutations = np.zeros((num_permutations, 3))
    tilde_delta_omega_clnk_norm_permutations = np.zeros(num_permutations)
    n_clnk_permutations = np.tile(n_clnk, (num_permutations, 1))
    
    tilde_delta_y_clnk_term = (
        b * eta * (np.sqrt(n_b) - np.sqrt(n_a)) / (2.*np.sqrt(3.))
    )
    tilde_delta_y_clnk = np.asarray(
        [
            tilde_delta_y_clnk_term*lmbda_0,
            tilde_delta_y_clnk_term*lmbda_1,
            tilde_delta_y_clnk_term*lmbda_2
        ])
    tilde_delta_y_clnk_permutations = np.asarray(
        [
            [tilde_delta_y_clnk[0], 0., 0.],
            [-tilde_delta_y_clnk[0], 0., 0.],
            [0., tilde_delta_y_clnk[1], 0.],
            [0., -tilde_delta_y_clnk[1], 0.],
            [0., 0., tilde_delta_y_clnk[2]],
            [0., 0., -tilde_delta_y_clnk[2]],
        ]
    )
    tilde_delta_y_clnk_norm_permutations = np.linalg.norm(
        tilde_delta_y_clnk_permutations, axis=1)

    return (
        n_clnk_permutations, tilde_delta_omega_clnk_permutations,
        tilde_delta_omega_clnk_norm_permutations,
        tilde_delta_y_clnk_permutations, tilde_delta_y_clnk_norm_permutations
    )

def inext_gaussian_fjc_regular_tetrahedral_4_chn_clnk_free_rot_approx(
        eval_W_flucts: bool,
        F: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        vol_quad_clnk: np.ndarray,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]):
    if not np.allclose(x_hat_clnk_func(X_clnk), regular_tetrahedral_4_chn_clnk_X_hat_clnk_func()):
        error_str = (
            "This function is only applicable for the regular "
            + "tetrahedral 4-chain cross-link. Make sure that the "
            + "cross-link structure properly corresponds to the "
            + "regular tetrahedral 4-chain cross-link."
        )
        raise ValueError(error_str)
    
    Q_clnk_m, y_clnk_m = monodisperse_regular_tetrahedral_4_chn_clnk_free_rot(F)
    Lmbda, _ = principal_stretch_decomposition(F)
    lmbda_0, lmbda_1, _ = Lmbda
    F_Lmbda = np.diag(np.asarray([lmbda_0, lmbda_1, 1./(lmbda_0*lmbda_1)]))
    unique_n_clnk, unique_n_clnk_counts = np.unique(n_clnk, return_counts=True)
    if np.shape(unique_n_clnk)[0] >= 3:
        (n_clnk_permutations, tilde_delta_omega_clnk_permutations,
         tilde_delta_omega_clnk_norm_permutations,
         tilde_delta_y_clnk_permutations, tilde_delta_y_clnk_norm_permutations) = (
            inext_gaussian_fjc_regular_tetrahedral_4_chn_clnk_free_rot_general_approx(
                Lmbda, n_clnk, b)
        )
    elif np.shape(unique_n_clnk)[0] == 2:
        # bimodal 1-3 clnk RVE
        if unique_n_clnk_counts[0] == 1 or unique_n_clnk_counts[0] == 3:
            (n_clnk_permutations, tilde_delta_omega_clnk_permutations,
             tilde_delta_omega_clnk_norm_permutations,
             tilde_delta_y_clnk_permutations,
             tilde_delta_y_clnk_norm_permutations) = (
                inext_gaussian_fjc_1_3_bimodal_regular_tetrahedral_4_chn_clnk_free_rot_approx(
                    Lmbda, n_clnk, b)
            )
        # bimodal 2-2 clnk RVE
        else:
            (n_clnk_permutations, tilde_delta_omega_clnk_permutations,
             tilde_delta_omega_clnk_norm_permutations,
             tilde_delta_y_clnk_permutations,
             tilde_delta_y_clnk_norm_permutations) = (
                inext_gaussian_fjc_2_2_bimodal_regular_tetrahedral_4_chn_clnk_free_rot_approx(
                    Lmbda, n_clnk, b)
            )
    
    # Evaluate cross-link free energy
    num_permutations = np.shape(n_clnk_permutations)[0]
    W_clnk_permutations = np.zeros(num_permutations)
    
    for prmttn in range(num_permutations):
        n_clnk_prmttn = n_clnk_permutations[prmttn]
        gamma_clnk_prmttn = gamma_clnk_approx_func(
            F_Lmbda, n_clnk_prmttn, b, X_clnk, Q_clnk_m, y_clnk_m,
            Q_axis_angle(tilde_delta_omega_clnk_permutations[prmttn]),
            tilde_delta_y_clnk_permutations[prmttn])
        W_clnk_permutations[prmttn] = W_clnk_func(
            gamma_clnk_prmttn, n_clnk_prmttn, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    W_clnk_min_permutations_indx = np.argmin(W_clnk_permutations)

    n_clnk_permutation = n_clnk_permutations[W_clnk_min_permutations_indx]
    delta_omega_clnk_permutation = (
        tilde_delta_omega_clnk_permutations[W_clnk_min_permutations_indx]
    )
    delta_omega_clnk_norm_permutation = (
        tilde_delta_omega_clnk_norm_permutations[W_clnk_min_permutations_indx]
    )
    delta_y_clnk_permutation = (
        tilde_delta_y_clnk_permutations[W_clnk_min_permutations_indx]
    )
    delta_y_clnk_norm_permutation = (
        tilde_delta_y_clnk_norm_permutations[W_clnk_min_permutations_indx]
    )
    delta_Q_clnk_permutation = Q_axis_angle(delta_omega_clnk_permutation)
    gamma_clnk_permutation = gamma_clnk_approx_func(
        F_Lmbda, n_clnk_permutation, b, X_clnk, Q_clnk_m, y_clnk_m,
        delta_Q_clnk_permutation, delta_y_clnk_permutation)
    
    W_clnk_permutation = W_clnk_func(
        gamma_clnk_permutation, n_clnk_permutation, w_c_func, w_c_args,
        w_c_dfrmtn_func, w_c_dfrmtn_args)
    if eval_W_flucts:
        W_flucts_clnk_permutation = W_flucts_clnk_approx_func(
            F_Lmbda, Q_clnk_m, delta_Q_clnk_permutation, n_clnk_permutation, b,
            X_clnk, vol_quad_clnk, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    else: W_flucts_clnk_permutation = 0.

    return (
        n_clnk_permutation, delta_omega_clnk_permutation,
        delta_omega_clnk_norm_permutation, delta_y_clnk_permutation,
        delta_y_clnk_norm_permutation, gamma_clnk_permutation,
        W_clnk_permutation, W_flucts_clnk_permutation
    )

def inext_gaussian_fjc_cube_8_chn_clnk_free_rot_approx_components(
        Lmbda, n_clnk, b):
    lmbda_0, lmbda_1, _ = Lmbda
    n_0, n_1, n_2, n_3, n_4, n_5, n_6, n_7 = n_clnk
    prod_n_clnk = n_0 * n_1 * n_2 * n_3 * n_4 * n_5 * n_6 * n_7
    prod_n_clnk_over_n_0 = prod_n_clnk / n_0
    prod_n_clnk_over_n_1 = prod_n_clnk / n_1
    prod_n_clnk_over_n_2 = prod_n_clnk / n_2
    prod_n_clnk_over_n_3 = prod_n_clnk / n_3
    prod_n_clnk_over_n_4 = prod_n_clnk / n_4
    prod_n_clnk_over_n_5 = prod_n_clnk / n_5
    prod_n_clnk_over_n_6 = prod_n_clnk / n_6
    prod_n_clnk_over_n_7 = prod_n_clnk / n_7
    sqrt_prod_n_clnk = np.sqrt(prod_n_clnk)
    sqrt_prod_n_clnk_over_n_0 = np.sqrt(prod_n_clnk_over_n_0)
    sqrt_prod_n_clnk_over_n_1 = np.sqrt(prod_n_clnk_over_n_1)
    sqrt_prod_n_clnk_over_n_2 = np.sqrt(prod_n_clnk_over_n_2)
    sqrt_prod_n_clnk_over_n_3 = np.sqrt(prod_n_clnk_over_n_3)
    sqrt_prod_n_clnk_over_n_4 = np.sqrt(prod_n_clnk_over_n_4)
    sqrt_prod_n_clnk_over_n_5 = np.sqrt(prod_n_clnk_over_n_5)
    sqrt_prod_n_clnk_over_n_6 = np.sqrt(prod_n_clnk_over_n_6)
    sqrt_prod_n_clnk_over_n_7 = np.sqrt(prod_n_clnk_over_n_7)

    a_0 = (
        sqrt_prod_n_clnk_over_n_0 - sqrt_prod_n_clnk_over_n_1
        + sqrt_prod_n_clnk_over_n_2 - sqrt_prod_n_clnk_over_n_3
        + sqrt_prod_n_clnk_over_n_4 - sqrt_prod_n_clnk_over_n_5
        + sqrt_prod_n_clnk_over_n_6 - sqrt_prod_n_clnk_over_n_7
    )
    a_1 = (
        -sqrt_prod_n_clnk_over_n_0 - sqrt_prod_n_clnk_over_n_1
        + sqrt_prod_n_clnk_over_n_2 + sqrt_prod_n_clnk_over_n_3
        - sqrt_prod_n_clnk_over_n_4 - sqrt_prod_n_clnk_over_n_5
        + sqrt_prod_n_clnk_over_n_6 + sqrt_prod_n_clnk_over_n_7
    )
    a_2 = (
        -sqrt_prod_n_clnk_over_n_0 + sqrt_prod_n_clnk_over_n_1
        - sqrt_prod_n_clnk_over_n_2 + sqrt_prod_n_clnk_over_n_3
        - sqrt_prod_n_clnk_over_n_4 + sqrt_prod_n_clnk_over_n_5
        - sqrt_prod_n_clnk_over_n_6 + sqrt_prod_n_clnk_over_n_7
    )
    a_3 = (
        sqrt_prod_n_clnk_over_n_0 + sqrt_prod_n_clnk_over_n_1
        + sqrt_prod_n_clnk_over_n_2 + sqrt_prod_n_clnk_over_n_3
        - sqrt_prod_n_clnk_over_n_4 - sqrt_prod_n_clnk_over_n_5
        - sqrt_prod_n_clnk_over_n_6 - sqrt_prod_n_clnk_over_n_7
    )
    a_4 = (
        sqrt_prod_n_clnk_over_n_0 + sqrt_prod_n_clnk_over_n_1
        - sqrt_prod_n_clnk_over_n_2 - sqrt_prod_n_clnk_over_n_3
        + sqrt_prod_n_clnk_over_n_4 + sqrt_prod_n_clnk_over_n_5
        - sqrt_prod_n_clnk_over_n_6 - sqrt_prod_n_clnk_over_n_7
    )
    a_5 = (
        -sqrt_prod_n_clnk_over_n_0 - sqrt_prod_n_clnk_over_n_1
        - sqrt_prod_n_clnk_over_n_2 - sqrt_prod_n_clnk_over_n_3
        + sqrt_prod_n_clnk_over_n_4 + sqrt_prod_n_clnk_over_n_5
        + sqrt_prod_n_clnk_over_n_6 + sqrt_prod_n_clnk_over_n_7
    )

    a_0 /= (np.sqrt(3.)*sqrt_prod_n_clnk)
    a_1 /= (np.sqrt(3.)*sqrt_prod_n_clnk)
    a_2 /= (np.sqrt(3.)*sqrt_prod_n_clnk)
    a_3 /= (np.sqrt(3.)*sqrt_prod_n_clnk)
    a_4 /= (np.sqrt(3.)*sqrt_prod_n_clnk)
    a_5 /= (np.sqrt(3.)*sqrt_prod_n_clnk)

    a_6 = (
        prod_n_clnk_over_n_0 + prod_n_clnk_over_n_1 + prod_n_clnk_over_n_2
        + prod_n_clnk_over_n_3 + prod_n_clnk_over_n_4 + prod_n_clnk_over_n_5
        + prod_n_clnk_over_n_6 + prod_n_clnk_over_n_7
    )
    a_6 /= prod_n_clnk

    tilde_delta_omega_clnk_0 = (3*(-8*a_1*a_2*a_6*(3*a_1*a_4 + 8*a_6)*lmbda_0 - 8*a_1*a_2*a_6*(3*a_3*a_5 + 8*a_6 \
    + (3*a_0*a_2 + 3*a_1*a_4 - 3*a_5**2 + 8*a_6)*lmbda_0**3)*lmbda_1 - (9*(a_0*a_3*a_4 \
    + a_1*a_2*a_5)*(a_0*a_2*a_3 + a_5*(a_1**2 - a_3*a_5)) + 24*(a_0*a_1*(a_2**2 + a_1*a_4) \
    + a_0*a_3*(a_1 - a_4)*a_5 + a_1*a_2*(a_3 - a_5)*a_5)*a_6 + 64*a_1*(a_0 \
    + a_2)*a_6**2)*lmbda_0**2*lmbda_1**2 - 8*a_0*a_6*lmbda_0*(a_1*(3*a_3*a_5 + 8*a_6) \
    + (3*a_0*a_1*a_2 + 3*a_1**2*a_4 - 3*a_3*a_4*a_5 + 8*a_1*a_6)*lmbda_0**3)*lmbda_1**3 \
    - 8*a_0*a_1*a_6*(3*a_0*a_2 + 8*a_6)*lmbda_0**3*lmbda_1**4))/(8*a_6*(3*a_1*a_4 \
    + 8*a_6)*(3*a_1*a_4 + 3*a_3*a_5 + 8*a_6)*lmbda_0 + 8*a_6*((3*a_3*a_5 \
    + 8*a_6)*(3*a_1*a_4 + 3*a_3*a_5 + 8*a_6) + (3*a_1*a_4 + 8*a_6)*(3*a_0*a_2 \
    + 3*a_1*a_4 + 8*a_6)*lmbda_0**3)*lmbda_1 + (27*(a_0*a_3*a_4 + a_1*a_2*a_5)**2 \
    + 144*(a_1*a_3*a_4*a_5 + a_0*a_2*(a_1*a_4 + a_3*a_5))*a_6 + 384*(a_0*a_2 + a_1*a_4 \
    + a_3*a_5)*a_6**2 + 1024*a_6**3)*lmbda_0**2*lmbda_1**2 + 8*a_6*lmbda_0*((3*a_3*a_5 \
    + 8*a_6)*(3*a_0*a_2 + 3*a_3*a_5 + 8*a_6) + (3*a_0*a_2 + 8*a_6)*(3*a_0*a_2 \
    + 3*a_1*a_4 + 8*a_6)*lmbda_0**3)*lmbda_1**3 + 8*a_6*(3*a_0*a_2 + 8*a_6)*(3*a_0*a_2 \
    + 3*a_3*a_5 + 8*a_6)*lmbda_0**3*lmbda_1**4)

    tilde_delta_omega_clnk_1 = (-3*(-9*a_1**3*a_2*a_4*a_5*lmbda_0**2*lmbda_1**2 \
    + 9*a_0*a_3**2*a_4**2*a_5*lmbda_0**2*lmbda_1**2 \
    + 24*a_0*a_2**2*a_6*lmbda_0*lmbda_1**2*(lmbda_0 + lmbda_1)*(a_3 \
    + a_5*lmbda_0**2*lmbda_1) + 3*a_1*a_4*lmbda_0*(8*a_2*a_3*a_6 \
    + 8*a_2*a_5*a_6*lmbda_0**2*lmbda_1 + (3*a_2*a_3*(a_0*a_2 + a_5**2) - 8*a_0*a_3*a_6 \
    + 8*a_2*(a_3 + a_5)*a_6)*lmbda_0*lmbda_1**2 - 8*a_0*a_3*a_6*lmbda_1**3) \
    + 8*a_2*a_6*(a_3 + a_5*lmbda_0**2*lmbda_1)*(1 \
    + lmbda_0*lmbda_1**2)*(3*a_3*a_5*lmbda_1 + 8*a_6*(lmbda_0 + lmbda_1)) \
    - 3*a_1**2*lmbda_0**2*lmbda_1**2*(3*a_0*a_3*a_4**2 - 3*a_2**3*a_5 + 8*a_2*a_5*(a_6 \
    + a_6*lmbda_0*lmbda_1**2))))/(8*a_6*(lmbda_0 + lmbda_1)*(8*a_6 + (3*a_0*a_2 \
    + 8*a_6)*lmbda_0**2*lmbda_1)*(8*a_6 + (3*a_0*a_2 + 8*a_6)*lmbda_0*lmbda_1**2) \
    + 9*a_1**2*lmbda_0*(3*a_2**2*a_5**2*lmbda_0*lmbda_1**2 + 8*a_4**2*(a_6 \
    + a_6*lmbda_0**2*lmbda_1)) + 9*a_3**2*lmbda_1*(3*a_0**2*a_4**2*lmbda_0**2*lmbda_1 \
    + 8*a_5**2*(a_6 + a_6*lmbda_0*lmbda_1**2)) + 24*a_3*a_5*a_6*(8*a_6*(1 \
    + lmbda_0*lmbda_1**2)*(lmbda_0 + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) \
    + 3*a_0*a_2*lmbda_0*lmbda_1**2*(lmbda_1 + lmbda_0*(2 + lmbda_0*lmbda_1**2))) \
    + 6*a_1*a_4*(3*a_3*a_5*(4*a_6*lmbda_0 + 4*a_6*lmbda_1 + (3*a_0*a_2 \
    + 8*a_6)*lmbda_0**2*lmbda_1**2) + 4*a_6*(3*a_0*a_2*lmbda_0**2*lmbda_1*(lmbda_0 \
    + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) + 8*a_6*(1 + lmbda_0**2*lmbda_1)*(lmbda_1 \
    + lmbda_0*(2 + lmbda_0*lmbda_1**2)))))

    tilde_delta_omega_clnk_2 = (3*(-8*a_4*a_6*(-3*a_0*a_2*a_3 + a_5*(3*a_1*a_4 + 3*a_3*a_5 + 8*a_6))*lmbda_0 \
    - 8*a_5*a_6*(a_1*(-3*a_2**2 + 3*a_1*a_4 + 3*a_3*a_5 + 8*a_6) + a_4*(3*a_1*a_4 \
    + 8*a_6)*lmbda_0**3)*lmbda_1 - (9*(a_0*a_3*a_4 + a_1*a_2*a_5)*(-(a_0*a_2**2) \
    + a_0*a_1*a_4 + a_2*a_5**2) + 24*(a_5*(-(a_1*a_2**2) + a_1**2*a_4 + a_3*a_4*a_5) \
    + a_0*a_2*(-(a_3*a_4) + (a_1 + a_4)*a_5))*a_6 + 64*(a_1 \
    + a_4)*a_5*a_6**2)*lmbda_0**2*lmbda_1**2 - 8*a_5*a_6*lmbda_0*(a_1*(3*a_3*a_5 + 8*a_6) \
    + a_4*(3*a_0*a_2 + 8*a_6)*lmbda_0**3)*lmbda_1**3 - 8*a_1*a_5*a_6*(3*a_0*a_2 \
    + 8*a_6)*lmbda_0**3*lmbda_1**4))/(8*a_6*(3*a_1*a_4 + 8*a_6)*(3*a_1*a_4 + 3*a_3*a_5 \
    + 8*a_6)*lmbda_0 + 8*a_6*((3*a_3*a_5 + 8*a_6)*(3*a_1*a_4 + 3*a_3*a_5 + 8*a_6) \
    + (3*a_1*a_4 + 8*a_6)*(3*a_0*a_2 + 3*a_1*a_4 + 8*a_6)*lmbda_0**3)*lmbda_1 \
    + (27*(a_0*a_3*a_4 + a_1*a_2*a_5)**2 + 144*(a_1*a_3*a_4*a_5 + a_0*a_2*(a_1*a_4 \
    + a_3*a_5))*a_6 + 384*(a_0*a_2 + a_1*a_4 + a_3*a_5)*a_6**2 \
    + 1024*a_6**3)*lmbda_0**2*lmbda_1**2 + 8*a_6*lmbda_0*((3*a_3*a_5 + 8*a_6)*(3*a_0*a_2 \
    + 3*a_3*a_5 + 8*a_6) + (3*a_0*a_2 + 8*a_6)*(3*a_0*a_2 + 3*a_1*a_4 \
    + 8*a_6)*lmbda_0**3)*lmbda_1**3 + 8*a_6*(3*a_0*a_2 + 8*a_6)*(3*a_0*a_2 + 3*a_3*a_5 \
    + 8*a_6)*lmbda_0**3*lmbda_1**4)

    tilde_delta_omega_clnk = np.asarray(
        [
            tilde_delta_omega_clnk_0,
            tilde_delta_omega_clnk_1,
            tilde_delta_omega_clnk_2
        ])
    tilde_delta_omega_clnk_norm = np.linalg.norm(tilde_delta_omega_clnk)

    tilde_delta_y_clnk_0 = (-8*b*lmbda_0*(-9*a_0**2*a_3*(a_2**2 - a_1*a_4)*lmbda_0*lmbda_1**2*(lmbda_0 \
    + lmbda_1) - 3*a_0*a_2*(-8*a_5*a_6*lmbda_0*lmbda_1**2*(lmbda_0 + lmbda_1)*(1 \
    + lmbda_0**2*lmbda_1) + 3*a_3**2*a_5*lmbda_1*(1 + lmbda_0*lmbda_1**2) \
    + a_3*(lmbda_0 + lmbda_1)*(8*a_6 + (-3*a_5**2 + 8*a_6)*lmbda_0*lmbda_1**2)) \
    + a_5*(-9*a_1**3*a_4*lmbda_1*(1 + lmbda_0**2*lmbda_1) + 3*a_1*a_4*(1 \
    + lmbda_0**2*lmbda_1)*(3*a_3*a_5*lmbda_1 + 8*a_6*(lmbda_0 + lmbda_1)) + (1 \
    + lmbda_0*lmbda_1**2)*(3*a_3*a_5*lmbda_1 + 8*a_6*(lmbda_0 + lmbda_1))*(3*a_3*a_5 \
    + 8*(a_6 + a_6*lmbda_0**2*lmbda_1)) - 3*a_1**2*lmbda_1*(-3*a_2**2*(1 \
    + lmbda_0**2*lmbda_1) + (3*a_3*a_5 + 8*a_6 + 8*a_6*lmbda_0**2*lmbda_1)*(1 \
    + lmbda_0*lmbda_1**2)))))/(8*a_6*(lmbda_0 + lmbda_1)*(8*a_6 + (3*a_0*a_2 \
    + 8*a_6)*lmbda_0**2*lmbda_1)*(8*a_6 + (3*a_0*a_2 + 8*a_6)*lmbda_0*lmbda_1**2) \
    + 9*a_1**2*lmbda_0*(3*a_2**2*a_5**2*lmbda_0*lmbda_1**2 + 8*a_4**2*(a_6 \
    + a_6*lmbda_0**2*lmbda_1)) + 9*a_3**2*lmbda_1*(3*a_0**2*a_4**2*lmbda_0**2*lmbda_1 \
    + 8*a_5**2*(a_6 + a_6*lmbda_0*lmbda_1**2)) + 24*a_3*a_5*a_6*(8*a_6*(1 \
    + lmbda_0*lmbda_1**2)*(lmbda_0 + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) \
    + 3*a_0*a_2*lmbda_0*lmbda_1**2*(lmbda_1 + lmbda_0*(2 + lmbda_0*lmbda_1**2))) \
    + 6*a_1*a_4*(3*a_3*a_5*(4*a_6*lmbda_0 + 4*a_6*lmbda_1 + (3*a_0*a_2 \
    + 8*a_6)*lmbda_0**2*lmbda_1**2) + 4*a_6*(3*a_0*a_2*lmbda_0**2*lmbda_1*(lmbda_0 \
    + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) + 8*a_6*(1 + lmbda_0**2*lmbda_1)*(lmbda_1 \
    + lmbda_0*(2 + lmbda_0*lmbda_1**2)))))

    tilde_delta_y_clnk_1 = (-8*b*lmbda_1*(9*a_1**3*a_4**2*lmbda_0*(1 + lmbda_0**2*lmbda_1) \
    + 3*a_3*a_4*lmbda_0*(1 + lmbda_0*lmbda_1**2)*(3*a_0*a_2*a_3 - a_5*(3*a_3*a_5 \
    + 8*(a_6 + a_6*lmbda_0**2*lmbda_1))) + a_1*(3*a_3*a_5*(-3*a_4**2*(lmbda_0 \
    + lmbda_0**3*lmbda_1) + 8*a_6*(lmbda_0 + lmbda_1)*(1 + lmbda_0*lmbda_1**2)) \
    + (lmbda_0 + lmbda_1)*(-9*a_0*a_2**3*lmbda_0**2*lmbda_1 - 3*a_2**2*(8*a_6 \
    + (-3*a_5**2 + 8*a_6)*lmbda_0**2*lmbda_1) + 24*a_0*a_2*a_6*lmbda_0**2*lmbda_1*(1 \
    + lmbda_0*lmbda_1**2) + 64*a_6**2*(1 + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2))) \
    + 3*a_1**2*a_4*(3*a_0*a_2*lmbda_0**2*lmbda_1*(lmbda_0 + lmbda_1) \
    - 3*a_2**2*(lmbda_0 + lmbda_0**3*lmbda_1) + 3*a_3*a_5*lmbda_0*(1 \
    + lmbda_0*lmbda_1**2) + 8*a_6*(1 + lmbda_0**2*lmbda_1)*(lmbda_1 + lmbda_0*(2 \
    + lmbda_0*lmbda_1**2)))))/(8*a_6*(lmbda_0 + lmbda_1)*(8*a_6 + (3*a_0*a_2 \
    + 8*a_6)*lmbda_0**2*lmbda_1)*(8*a_6 + (3*a_0*a_2 + 8*a_6)*lmbda_0*lmbda_1**2) \
    + 9*a_1**2*lmbda_0*(3*a_2**2*a_5**2*lmbda_0*lmbda_1**2 + 8*a_4**2*(a_6 \
    + a_6*lmbda_0**2*lmbda_1)) + 9*a_3**2*lmbda_1*(3*a_0**2*a_4**2*lmbda_0**2*lmbda_1 \
    + 8*a_5**2*(a_6 + a_6*lmbda_0*lmbda_1**2)) + 24*a_3*a_5*a_6*(8*a_6*(1 \
    + lmbda_0*lmbda_1**2)*(lmbda_0 + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) \
    + 3*a_0*a_2*lmbda_0*lmbda_1**2*(lmbda_1 + lmbda_0*(2 + lmbda_0*lmbda_1**2))) \
    + 6*a_1*a_4*(3*a_3*a_5*(4*a_6*lmbda_0 + 4*a_6*lmbda_1 + (3*a_0*a_2 \
    + 8*a_6)*lmbda_0**2*lmbda_1**2) + 4*a_6*(3*a_0*a_2*lmbda_0**2*lmbda_1*(lmbda_0 \
    + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) + 8*a_6*(1 + lmbda_0**2*lmbda_1)*(lmbda_1 \
    + lmbda_0*(2 + lmbda_0*lmbda_1**2)))))

    tilde_delta_y_clnk_2 = (-8*b*(9*a_0**2*a_2**3*lmbda_0**3*lmbda_1**3*(lmbda_0 + lmbda_1) \
    + 9*a_0*a_3*a_4**2*a_5*lmbda_0**2*lmbda_1**2*(1 + lmbda_0**2*lmbda_1) + a_2*(1 \
    + lmbda_0*lmbda_1**2)*(3*a_3*a_5*lmbda_1 + 8*a_6*(lmbda_0 \
    + lmbda_1))*(-3*a_5**2*lmbda_0**2*lmbda_1 + 8*a_6*(1 + lmbda_0**2*lmbda_1)) \
    - 9*a_1**2*lmbda_0**2*lmbda_1**2*(a_0*a_4**2*(1 + lmbda_0**2*lmbda_1) - a_2*a_5**2*(1 \
    + lmbda_0*lmbda_1**2)) \
    + 3*a_1*a_4*lmbda_0*(-3*a_0**2*a_2*lmbda_0**2*lmbda_1**3*(lmbda_0 + lmbda_1) \
    + 3*a_0*a_2**2*lmbda_0*lmbda_1**2*(1 + lmbda_0**2*lmbda_1) \
    - 8*a_0*a_6*lmbda_1**2*(lmbda_0 + lmbda_1)*(1 + lmbda_0**2*lmbda_1) + 8*a_2*a_6*(1 \
    + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2)) \
    + 3*a_0*a_2**2*lmbda_0*lmbda_1*(8*a_6*(lmbda_0 + lmbda_1)*(lmbda_0 + lmbda_1 \
    + 2*lmbda_0**2*lmbda_1**2) + 3*a_5*lmbda_0*lmbda_1*(a_3 + a_3*lmbda_0*lmbda_1**2 \
    - a_5*lmbda_0*lmbda_1*(lmbda_0 + lmbda_1)))))/(lmbda_0*lmbda_1*(8*a_6*(lmbda_0 \
    + lmbda_1)*(8*a_6 + (3*a_0*a_2 + 8*a_6)*lmbda_0**2*lmbda_1)*(8*a_6 + (3*a_0*a_2 \
    + 8*a_6)*lmbda_0*lmbda_1**2) + 9*a_1**2*lmbda_0*(3*a_2**2*a_5**2*lmbda_0*lmbda_1**2 \
    + 8*a_4**2*(a_6 + a_6*lmbda_0**2*lmbda_1)) \
    + 9*a_3**2*lmbda_1*(3*a_0**2*a_4**2*lmbda_0**2*lmbda_1 + 8*a_5**2*(a_6 \
    + a_6*lmbda_0*lmbda_1**2)) + 24*a_3*a_5*a_6*(8*a_6*(1 \
    + lmbda_0*lmbda_1**2)*(lmbda_0 + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) \
    + 3*a_0*a_2*lmbda_0*lmbda_1**2*(lmbda_1 + lmbda_0*(2 + lmbda_0*lmbda_1**2))) \
    + 6*a_1*a_4*(3*a_3*a_5*(4*a_6*lmbda_0 + 4*a_6*lmbda_1 + (3*a_0*a_2 \
    + 8*a_6)*lmbda_0**2*lmbda_1**2) + 4*a_6*(3*a_0*a_2*lmbda_0**2*lmbda_1*(lmbda_0 \
    + 2*lmbda_1 + lmbda_0**2*lmbda_1**2) + 8*a_6*(1 + lmbda_0**2*lmbda_1)*(lmbda_1 \
    + lmbda_0*(2 + lmbda_0*lmbda_1**2))))))

    tilde_delta_y_clnk = np.asarray(
        [
            tilde_delta_y_clnk_0,
            tilde_delta_y_clnk_1,
            tilde_delta_y_clnk_2
        ])
    tilde_delta_y_clnk_norm = np.linalg.norm(tilde_delta_y_clnk)
    
    return (
        tilde_delta_omega_clnk, tilde_delta_omega_clnk_norm,
        tilde_delta_y_clnk, tilde_delta_y_clnk_norm
    )

def inext_gaussian_fjc_cube_8_chn_clnk_free_rot_approx(
        eval_W_flucts: bool,
        F: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        vol_quad_clnk: np.ndarray,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]):
    if not np.allclose(x_hat_clnk_func(X_clnk), cube_8_chn_clnk_X_hat_clnk_func()):
        error_str = (
            "This function is only applicable for the cube 4-chain "
            + "cross-link. Make sure that the cross-link structure "
            + "properly corresponds to the cube 8-chain cross-link."
        )
        raise ValueError(error_str)
    
    Q_clnk_m, y_clnk_m = monodisperse_cube_8_chn_clnk_free_rot(F)
    Lmbda, _ = principal_stretch_decomposition(F)
    lmbda_0, lmbda_1, _ = Lmbda
    F_Lmbda = np.diag(np.asarray([lmbda_0, lmbda_1, 1./(lmbda_0*lmbda_1)]))

    n_clnk_indcs_permutations = indcs_permutations(np.shape(n_clnk)[0])
    n_clnk_permutations = n_clnk[n_clnk_indcs_permutations]
    num_permutations = np.shape(n_clnk_permutations)[0]
    tilde_delta_omega_clnk_permutations = np.zeros((num_permutations, 3))
    tilde_delta_omega_clnk_norm_permutations = np.zeros(num_permutations)
    tilde_delta_y_clnk_permutations = np.zeros((num_permutations, 3))
    tilde_delta_y_clnk_norm_permutations = np.zeros(num_permutations)

    for prmttn in range(num_permutations):
        (tilde_delta_omega_clnk, tilde_delta_omega_clnk_norm,
         tilde_delta_y_clnk, tilde_delta_y_clnk_norm) = (
            inext_gaussian_fjc_cube_8_chn_clnk_free_rot_approx_components(
                Lmbda, n_clnk_permutations[prmttn], b)
        )
        tilde_delta_omega_clnk_permutations[prmttn] = tilde_delta_omega_clnk
        tilde_delta_omega_clnk_norm_permutations[prmttn] = (
            tilde_delta_omega_clnk_norm
        )
        tilde_delta_y_clnk_permutations[prmttn] = tilde_delta_y_clnk
        tilde_delta_y_clnk_norm_permutations[prmttn] = tilde_delta_y_clnk_norm
        
    # Evaluate cross-link free energy
    W_clnk_permutations = np.zeros(num_permutations)

    for prmttn in range(num_permutations):
        n_clnk_prmttn = n_clnk_permutations[prmttn]
        gamma_clnk_prmttn = gamma_clnk_approx_func(
            F_Lmbda, n_clnk_prmttn, b, X_clnk, Q_clnk_m, y_clnk_m,
            Q_axis_angle(tilde_delta_omega_clnk_permutations[prmttn]),
            tilde_delta_y_clnk_permutations[prmttn])
        W_clnk_permutations[prmttn] = W_clnk_func(
            gamma_clnk_prmttn, n_clnk_prmttn, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    W_clnk_min_permutations_indx = np.argmin(W_clnk_permutations)

    n_clnk_permutation = n_clnk_permutations[W_clnk_min_permutations_indx]
    delta_omega_clnk_permutation = (
        tilde_delta_omega_clnk_permutations[W_clnk_min_permutations_indx]
    )
    delta_omega_clnk_norm_permutation = (
        tilde_delta_omega_clnk_norm_permutations[W_clnk_min_permutations_indx]
    )
    delta_y_clnk_permutation = (
        tilde_delta_y_clnk_permutations[W_clnk_min_permutations_indx]
    )
    delta_y_clnk_norm_permutation = (
        tilde_delta_y_clnk_norm_permutations[W_clnk_min_permutations_indx]
    )
    delta_Q_clnk_permutation = Q_axis_angle(delta_omega_clnk_permutation)
    gamma_clnk_permutation = gamma_clnk_approx_func(
        F_Lmbda, n_clnk_permutation, b, X_clnk, Q_clnk_m, y_clnk_m,
        delta_Q_clnk_permutation, delta_y_clnk_permutation)
    
    W_clnk_permutation = W_clnk_func(
        gamma_clnk_permutation, n_clnk_permutation, w_c_func, w_c_args,
        w_c_dfrmtn_func, w_c_dfrmtn_args)
    if eval_W_flucts:
        W_flucts_clnk_permutation = W_flucts_clnk_approx_func(
            F_Lmbda, Q_clnk_m, delta_Q_clnk_permutation, n_clnk_permutation, b,
            X_clnk, vol_quad_clnk, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    else: W_flucts_clnk_permutation = 0.

    return (
        n_clnk_permutation, delta_omega_clnk_permutation,
        delta_omega_clnk_norm_permutation, delta_y_clnk_permutation,
        delta_y_clnk_norm_permutation, gamma_clnk_permutation,
        W_clnk_permutation, W_flucts_clnk_permutation
    )

def clnk_free_rot_approx(
        eval_W_flucts: bool,
        F: np.ndarray,
        n_clnk: np.ndarray,
        b: float,
        X_clnk: np.ndarray,
        vol_quad_clnk: np.ndarray,
        w_c_func,
        w_c_args: tuple[float],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float]):
    k_num = np.shape(n_clnk)[0]
    if (not np.isclose(np.linalg.det(F), 1.) or
        np.all(np.equal(n_clnk, n_clnk[0])) or
        not (k_num == 4 or k_num == 8)):
        error_str = (
            "This methodology is only applicable to polydisperse "
            + "regular tetrahedral 4-chain cross-link RVEs under "
            + "incompressible deformation or polydisperse cube 8-chain "
            + "cross-link RVEs under incompressible deformation. Make "
            + "sure that the cross-link structure corresponds to one "
            + "of the aforementioned polydisperse cross-link "
            + "structures, and the deformation is incompressible."
        )
        raise ValueError(error_str)
    if (k_num == 4 and
        not np.allclose(x_hat_clnk_func(X_clnk), regular_tetrahedral_4_chn_clnk_X_hat_clnk_func())):
        error_str = (
            "The 4-chain cross-link RVE must be a polydisperse regular "
            + "tetrahedral 4-chain cross-link RVE. Make sure that the "
            + "cross-link structure corresponds to this."
        )
        raise ValueError(error_str)
    elif (k_num == 8 and
          not np.allclose(x_hat_clnk_func(X_clnk), cube_8_chn_clnk_X_hat_clnk_func())):
        error_str = (
            "The 8-chain cross-link RVE must be a polydisperse cube "
            + "8-chain cross-link RVE. Make sure that the cross-link "
            + "structure corresponds to this."
        )
        raise ValueError(error_str)
   
    if k_num == 4:
        return inext_gaussian_fjc_regular_tetrahedral_4_chn_clnk_free_rot_approx(
            eval_W_flucts, F, n_clnk, b, X_clnk, vol_quad_clnk,
            w_c_func, w_c_args, w_c_dfrmtn_func, w_c_dfrmtn_args)
    elif k_num == 8:
        return inext_gaussian_fjc_cube_8_chn_clnk_free_rot_approx(
            eval_W_flucts, F, n_clnk, b, X_clnk, vol_quad_clnk,
            w_c_func, w_c_args, w_c_dfrmtn_func, w_c_dfrmtn_args)
