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
from src.helpers.continuum_mechanics import principal_stretch_decomposition
from src.helpers.rotations import Q_axis_angle
from src.helpers.clnk_deformation import (
    monodisperse_y_clnk,
    gamma_clnk_func,
    gamma_vec_clnk_func,
    gamma_approx_clnk_func,
    gamma_approx_vec_clnk_func,
    W_clnk_chns_func,
    W_clnk_y_flucts_func
)
from src.helpers.clnk_structure import (
    x_hat_clnk_func,
    com_x_clnk_func,
    chull_eqs_clnk_func,
    x_clnk_min_max_func,
    regular_tetrahedral_4_chn_clnk_X_hat_clnk_func,
    regular_octahedral_6_chn_clnk_X_hat_clnk_func,
    cube_8_chn_clnk_X_hat_clnk_func
)

def monodisperse_regular_tetrahedral_4_chn_clnk_free_rot(
        F: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Optimal solution to the cross-link rotation and cross-link
    junction position for the monodisperse classical regular tetrahedral
    4-chain cross-link structure RVE in the free rotation limit.
    
    This function returns the optimal solution to the cross-link
    rotation and cross-link junction position for the monodisperse
    classical regular tetrahedral 4-chain cross-link structure RVE in
    the free rotation limit.

    Args:
        F (npt.NDArray[np.float64]): Deformation gradient.

    Returns:
        npt.NDArray[np.float64]: Optimal solution to the cross-link
        rotation and cross-link junction position for the monodisperse
        classical regular tetrahedral 4-chain cross-link structure RVE
        in the free rotation limit.
    
    """
    _, P = principal_stretch_decomposition(F)
    Q_1 = Q_axis_angle(np.asarray([np.pi/4., 0., 0.]))
    Q_2 = Q_axis_angle(np.asarray([0., np.arccos(np.sqrt(2./3.)), 0.]))
    Q_3 = Q_axis_angle(np.asarray([0., 0., -np.pi/2.]))
    Q_clnk_star = np.matmul(Q_1, np.matmul(Q_2, np.matmul(Q_3, P)))

    return Q_clnk_star, monodisperse_y_clnk()

def monodisperse_regular_octahedral_6_chn_clnk_free_rot(
        F: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Optimal solution to the cross-link rotation and cross-link
    junction position for the monodisperse classical regular octahedral
    6-chain cross-link structure RVE in the free rotation limit.
    
    This function returns the optimal solution to the cross-link
    rotation and cross-link junction position for the monodisperse
    classical regular octahedral 6-chain cross-link structure RVE in
    the free rotation limit.

    Args:
        F (npt.NDArray[np.float64]): Deformation gradient.

    Returns:
        npt.NDArray[np.float64]: Optimal solution to the cross-link
        rotation and cross-link junction position for the monodisperse
        classical regular octahedral 6-chain cross-link structure RVE
        in the free rotation limit.
    
    """
    return np.full_like(F, np.nan), monodisperse_y_clnk()

def monodisperse_cube_8_chn_clnk_free_rot(
        F: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Optimal solution to the cross-link rotation and cross-link
    junction position for the monodisperse classical cube 8-chain
    cross-link structure RVE in the free rotation limit.
    
    This function returns the optimal solution to the cross-link
    rotation and cross-link junction position for the monodisperse
    classical cube 8-chain cross-link structure RVE in the free rotation
    limit.

    Args:
        F (npt.NDArray[np.float64]): Deformation gradient.

    Returns:
        npt.NDArray[np.float64]: Optimal solution to the cross-link
        rotation and cross-link junction position for the monodisperse
        classical cube 8-chain cross-link structure RVE in the free
        rotation limit.
    
    """
    _, P = principal_stretch_decomposition(F)
    return P, monodisperse_y_clnk()

def gamma_monodisperse_clnk_free_rot(
        Lmbda: npt.NDArray[np.float64],
        gamma_clnk_init: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Absolute/Equilibrium chain stretch for each chain in a
    monodisperse classical regular tetrahedral 4-chain, regular
    octahedral 6-chain, or cube 8-chain cross-link structure RVE in the
    free rotation limit.
    
    This function returns the absolute/equilibrium chain stretch for
    each chain in a monodisperse classical regular tetrahedral 4-chain,
    regular octahedral 6-chain, or cube 8-chain cross-link structure RVE
    in the free rotation limit.

    Args:
        Lmbda (npt.NDArray[np.float64]): Principal stretch matrix.
        gamma_clnk_init (npt.NDArray[np.float64]): Initial absolute/equilibrium chain stretch for each chain in the cross-link structure RVE.

    Returns:
        npt.NDArray[np.float64]: Absolute/Equilibrium chain stretch for
        each chain in a monodisperse classical regular tetrahedral
        4-chain, regular octahedral 6-chain, or cube 8-chain cross-link
        structure RVE in the free rotation limit.
    
    """
    return (
        gamma_clnk_init[0] * np.sqrt(np.sum(np.power(Lmbda, 2))/3.)
        * np.ones(np.shape(gamma_clnk_init)[0])
    )

def monodisperse_clnk_free_rot(
        eval_W_clnk_y_flucts: bool,
        F: npt.NDArray[np.float64],
        Lmbda: npt.NDArray[np.float64],
        n_clnk: npt.NDArray[np.float64],
        b_clnk: npt.NDArray[np.float64],
        X_clnk: npt.NDArray[np.float64],
        y_clnk_init: npt.NDArray[np.float64],
        gamma_clnk_init: npt.NDArray[np.float64],
        w_c_func_clnk: npt.NDArray[np.object_],
        w_c_args_clnk: npt.NDArray[np.object_],
        d2w_c__dy_clnk_dy_clnk_func_clnk: npt.NDArray[np.object_],
        d2w_c__dy_clnk_dy_clnk_args_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_func_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_args_clnk: npt.NDArray[np.object_]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, npt.NDArray[np.float64], float, float]:
    """Monodisperse cross-link structure RVE mechanical response in the
    free rotation limit.

    This function determines the mechanical response of a monodisperse
    cross-link structure RVE in the free rotation limit.

    Args:
        eval_W_clnk_y_flucts (bool): Boolean indicating if the nondimensional cross-link junction fluctuation free energy ought to be calculated (if True) or not (if False).
        F (npt.NDArray[np.float64]): Deformation gradient.
        n_clnk (npt.NDArray[np.float64]): Number of chain segments for each chain in the cross-link structure RVE.
        b_clnk (npt.NDArray[np.float64]): Chain segment and/or cross-linker diameter for each chain in the cross-link structure RVE.
        X_clnk (npt.NDArray[np.float64]): Initial chain end position for each chain in the cross-link structure RVE.
        y_clnk_init (npt.NDArray[np.float64]): Initial cross-link junction position.
        gamma_clnk_init (npt.NDArray[np.float64]): Initial absolute/equilibrium chain stretch for each chain in the cross-link structure RVE.
        w_c_func_clnk (npt.NDArray[np.object_]): Nondimensional polymer chain free energy function for each chain in the cross-link structure RVE.
        w_c_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE.
        d2w_c__dy_clnk_dy_clnk_func_clnk (npt.NDArray[np.object_]): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function for each chain in the cross-link structure RVE.
        d2w_c__dy_clnk_dy_clnk_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n) for each chain in the cross-link structure RVE.
        w_c_dfrmtn_func_clnk (npt.NDArray[np.object_]): Nondimensional polymer chain deformation free energy function for each chain in the cross-link structure RVE.
        w_c_dfrmtn_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE.
    
    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, npt.NDArray[np.float64], float, float]:
        Optimal monodisperse cross-link rotation, optimal monodisperse
        cross-link junction position, distance between the origin and
        the optimal monodisperse cross-link junction position,
        absolute/equilibrium chain stretch for each chain in the
        monodisperse cross-link, nondimensional monodisperse cross-link
        polymer chain free energy, nondimensional monodisperse
        cross-link junction fluctuation free energy.
    
    """
    # Boilerplate initialization, checks, and assertions
    k_num = np.shape(n_clnk)[0]
    X_hat_clnk = x_hat_clnk_func(X_clnk)
    com_X_hat_clnk = com_x_clnk_func(X_hat_clnk)
    full_like_n_clnk_0 = np.empty_like(n_clnk)
    full_like_b_clnk_0 = np.empty_like(b_clnk)
    full_like_gamma_clnk_init_0 = np.empty_like(gamma_clnk_init)
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
    full_like_gamma_clnk_init_0.fill(gamma_clnk_init[0])
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
        not np.allclose(gamma_clnk_init, full_like_gamma_clnk_init_0) or
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
            + "chain, the initial absolute/equilibrium chain stretch "
            + "is the same for each chain, the initial position of the "
            + "cross-link is at the origin, and that the initial "
            + "center-of-mass of the cross-link is also located at the "
            + "origin."
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
    
    # Gather the cross-link junction position and cross-link rotation
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
    
    # Gather the absolute/equilibrium chain stretch for each chain
    gamma_clnk_star = gamma_monodisperse_clnk_free_rot(Lmbda, gamma_clnk_init)
    
    # Calculate the nondimensional cross-link chain free energy
    W_clnk_chns_star = W_clnk_chns_func(
        gamma_clnk_star, n_clnk, w_c_func_clnk, w_c_args_clnk,
        w_c_dfrmtn_func_clnk, w_c_dfrmtn_args_clnk)
    
    # If called for, calculate the nondimensional cross-link junction
    # fluctuation free energy
    W_clnk_y_flucts_star = 0.
    if eval_W_clnk_y_flucts:
        # Calculate the absolute/equilibrium chain stretch vector for
        # each chain
        gamma_vec_clnk_star = gamma_vec_clnk_func(
            False, F, X_clnk, Q_clnk_star, y_clnk_star, n_clnk, b_clnk)
        W_clnk_y_flucts_star = W_clnk_y_flucts_func(
            gamma_vec_clnk_star, gamma_clnk_star, n_clnk, b_clnk,
            d2w_c__dy_clnk_dy_clnk_func_clnk, d2w_c__dy_clnk_dy_clnk_args_clnk)

    return (
        Q_clnk_star, y_clnk_star, y_clnk_star_norm, gamma_clnk_star,
        W_clnk_chns_star, W_clnk_y_flucts_star
    )

def W_clnk_chns_free_rot(
        omega_clnk_y_clnk: npt.NDArray[np.float64],
        F: npt.NDArray[np.float64],
        n_clnk: npt.NDArray[np.float64],
        b_clnk: npt.NDArray[np.float64],
        X_clnk: npt.NDArray[np.float64],
        w_c_func_clnk: npt.NDArray[np.object_],
        w_c_args_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_func_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_args_clnk: npt.NDArray[np.object_]) -> float:
    """Nondimensional cross-link polymer chain free energy in the free
    rotation limit.

    This function supplies the nondimensional cross-link polymer chain
    free energy in the free rotation limit as a suitable objective
    function for constrained minimization.

    Args:
        omega_clnk_y_clnk (npt.NDArray[np.float64]): Horizontally stacked Rodrigues vector and cross-link junction position for the cross-link structure RVE.
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
        the free rotation limit.
    
    """
    # Gather the Rodrigues vector and the cross-link junction position
    omega_clnk, y_clnk = omega_clnk_y_clnk[:3], omega_clnk_y_clnk[3:]
    
    # Calculate the absolute/equilibrium chain stretch for each chain
    # (using the ideal numerics form for the chain end-to-end vector)
    gamma_clnk = gamma_clnk_func(
        True, F, X_clnk, Q_axis_angle(omega_clnk), y_clnk, n_clnk, b_clnk)
    
    # Calculate the nondimensional cross-link polymer chain free energy
    return (
        W_clnk_chns_func(
            gamma_clnk, n_clnk, w_c_func_clnk, w_c_args_clnk,
            w_c_dfrmtn_func_clnk, w_c_dfrmtn_args_clnk)
    )

def clnk_free_rot_cnstrnd_mnmztn(
        eval_W_clnk_y_flucts: bool,
        cnstrnd_mnmztn_scope: str,
        cnstrnd_mnmztn_method: str,
        rng: np.random.Generator,
        F: npt.NDArray[np.float64],
        n_clnk: npt.NDArray[np.float64],
        b_clnk: npt.NDArray[np.float64],
        X_clnk: npt.NDArray[np.float64],
        omega_clnk: npt.NDArray[np.float64],
        y_clnk: npt.NDArray[np.float64],
        w_c_func_clnk: npt.NDArray[np.object_],
        w_c_args_clnk: npt.NDArray[np.object_],
        d2w_c__dy_clnk_dy_clnk_func_clnk: npt.NDArray[np.object_],
        d2w_c__dy_clnk_dy_clnk_args_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_func_clnk: npt.NDArray[np.object_],
        w_c_dfrmtn_args_clnk: npt.NDArray[np.object_]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], float, npt.NDArray[np.float64], float, float]:
    """Cross-link structure RVE mechanical response in the free rotation
    limit, as evaluated numerically via constrained minimization.

    This function determines the mechanical response of a cross-link
    structure RVE in the free rotation limit, as evaluated numerically
    via constrained minimization.

    Args:
        eval_W_clnk_y_flucts (bool): Boolean indicating if the nondimensional cross-link junction fluctuation free energy ought to be calculated (if True) or not (if False).
        cnstrnd_mnmztn_scope (str): String indicating the use of either local ("lcl") or global ("glbl") constrained minimization.
        cnstrnd_mnmztn_method (str): String indicating the specific constrained minimization method to utilize. See the code and associated error string in the function for the different types of constrained minimization methods available for use.
        rng (np.random.Generator): Numpy random number generator object.
        F (npt.NDArray[np.float64]): Deformation gradient.
        n_clnk (npt.NDArray[np.float64]): Number of chain segments for each chain in the cross-link structure RVE.
        b_clnk (npt.NDArray[np.float64]): Chain segment and/or cross-linker diameter for each chain in the cross-link structure RVE.
        X_clnk (npt.NDArray[np.float64]): Initial chain end position for each chain in the cross-link structure RVE.
        omega_clnk (npt.NDArray[np.float64]): Prior Rodrigues vector describing the prior cross-link structure RVE orientation.
        y_clnk (npt.NDArray[np.float64]): Prior cross-link junction position.
        w_c_func_clnk (npt.NDArray[np.object_]): Nondimensional polymer chain free energy function for each chain in the cross-link structure RVE.
        w_c_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE.
        d2w_c__dy_clnk_dy_clnk_func_clnk (npt.NDArray[np.object_]): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function for each chain in the cross-link structure RVE.
        d2w_c__dy_clnk_dy_clnk_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n) for each chain in the cross-link structure RVE.
        w_c_dfrmtn_func_clnk (npt.NDArray[np.object_]): Nondimensional polymer chain deformation free energy function for each chain in the cross-link structure RVE.
        w_c_dfrmtn_args_clnk (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE.
    
    Returns:
        tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64], float, npt.NDArray[np.float64], float, float]:
        Optimal Rodrigues vector describing the cross-link rotation,
        norm of the optimal Rodrigues vector describing the cross-link
        rotation, optimal cross-link junction position, distance between
        the origin and the optimal cross-link junction position,
        absolute/equilibrium chain stretch for each chain in the
        cross-link, nondimensional cross-link polymer chain free energy,
        nondimensional cross-link junction fluctuation free energy.
    
    """
    # Gather the (deformed) cross-link convex hull
    x_clnk = np.empty_like(X_clnk)
    for chn_indx in range(np.shape(n_clnk)[0]):
        x_clnk[chn_indx] = np.matmul(F, X_clnk[chn_indx])
    A_chull_eqs_clnk, b_chull_eqs_clnk = chull_eqs_clnk_func(x_clnk)
    
    # Horizontally stack the prior Rodrigues vector and cross-link
    # junction position together
    omega_clnk_y_clnk = np.hstack((omega_clnk, y_clnk))

    # Gather the rotational constraint on the Rodrigues vector
    def omega_clnk_cnstrnt_func(omega_clnk_y_clnk):
        return np.linalg.norm(omega_clnk_y_clnk[:3])
    omega_clnk_cnstrnt = NonlinearConstraint(
        omega_clnk_cnstrnt_func, 0., 2.*np.pi)
    
    # Gather the bounds of the Rodrigues vector
    omega_clnk_min_bounds = np.zeros(3)
    omega_clnk_max_bounds = 2. * np.pi * np.ones(3)

    # Gather the cross-link convex hull constraints on the cross-link
    # junction position
    def chull_clnk_cnstrnt_func(omega_clnk_y_clnk):
        return (
            b_chull_eqs_clnk
            - np.matmul(A_chull_eqs_clnk, omega_clnk_y_clnk[3:])
        )
    y_clnk_cnstrnt = NonlinearConstraint(chull_clnk_cnstrnt_func, 0., np.inf)
    
    # Gather the bounds of the cross-link junction position
    y_clnk_min_bounds, y_clnk_max_bounds = x_clnk_min_max_func(x_clnk)

    # Gather the bounds of the Rodrigues vector and the cross-link
    # junction position together
    omega_clnk_y_clnk_bounds = Bounds(
        np.hstack((omega_clnk_min_bounds, y_clnk_min_bounds)),
        np.hstack((omega_clnk_max_bounds, y_clnk_max_bounds)))
    
    # Gather the arguments of the free rotation limit nondimensional
    # cross-link polymer chain free energy objective function
    # W_clnk_chns_free_rot()
    W_clnk_free_rot_args = (
        F, n_clnk, b_clnk, X_clnk,
        w_c_func_clnk, w_c_args_clnk, w_c_dfrmtn_func_clnk, w_c_dfrmtn_args_clnk
    )

    # Apply the called-for method of (local or global) constrained
    # minimization over the free rotation limit nondimensional
    # cross-link polymer chain free energy to solve for the optimal
    # Rodrigues vector (i.e., cross-link rotation) and cross-link
    # junction position
    if cnstrnd_mnmztn_scope == "lcl":
        if cnstrnd_mnmztn_method in ["COBYLA", "COBYQA", "trust-constr"]:
            clnk_free_rot = minimize(
                W_clnk_chns_free_rot, omega_clnk_y_clnk,
                args=W_clnk_free_rot_args, method=cnstrnd_mnmztn_method,
                bounds=omega_clnk_y_clnk_bounds,
                constraints=(omega_clnk_cnstrnt, y_clnk_cnstrnt))
    elif cnstrnd_mnmztn_scope == "glbl":
        if cnstrnd_mnmztn_method == "differential-evolution":
            clnk_free_rot = differential_evolution(
                W_clnk_chns_free_rot, omega_clnk_y_clnk_bounds,
                args=W_clnk_free_rot_args, rng=rng,
                constraints=(omega_clnk_cnstrnt, y_clnk_cnstrnt),
                x0=omega_clnk_y_clnk)
        elif cnstrnd_mnmztn_method == "shgo":
            clnk_free_rot = shgo(
                W_clnk_chns_free_rot, omega_clnk_y_clnk_bounds,
                args=W_clnk_free_rot_args,
                constraints=(omega_clnk_cnstrnt, y_clnk_cnstrnt),
                minimizer_kwargs={"method": "COBYQA"})
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
    
    # If the constrained minimization solved successfully, then update
    # the Rodrigues vector (i.e., cross-link rotation) and cross-link
    # junction position
    if clnk_free_rot.success: omega_clnk_y_clnk = clnk_free_rot.x
    omega_clnk, y_clnk = omega_clnk_y_clnk[:3], omega_clnk_y_clnk[3:]
    omega_clnk_norm = np.linalg.norm(omega_clnk)
    y_clnk_norm = np.linalg.norm(y_clnk)
    Q_clnk = Q_axis_angle(omega_clnk)
    
    # Calculate the absolute/equilibrium chain stretch for each chain
    gamma_clnk = gamma_clnk_func(True, F, X_clnk, Q_clnk, y_clnk, n_clnk, b_clnk)

    # Calculate the nondimensional cross-link chain free energy
    W_clnk_chns = W_clnk_chns_func(
        gamma_clnk, n_clnk, w_c_func_clnk, w_c_args_clnk,
        w_c_dfrmtn_func_clnk, w_c_dfrmtn_args_clnk)
    
    # If called for, calculate the nondimensional cross-link junction
    # fluctuation free energy
    W_clnk_y_flucts = 0.
    if eval_W_clnk_y_flucts:
        # Calculate the absolute/equilibrium chain stretch vector for
        # each chain
        gamma_vec_clnk = gamma_vec_clnk_func(
            True, F, X_clnk, Q_clnk, y_clnk, n_clnk, b_clnk)
        W_clnk_y_flucts = W_clnk_y_flucts_func(
            gamma_vec_clnk, gamma_clnk, n_clnk, b_clnk,
            d2w_c__dy_clnk_dy_clnk_func_clnk, d2w_c__dy_clnk_dy_clnk_args_clnk)
    
    return (
        omega_clnk, omega_clnk_norm, y_clnk, y_clnk_norm, gamma_clnk,
        W_clnk_chns, W_clnk_y_flucts
    )

def inext_gaussian_fjc_tilde_delta_clnk_regular_tetrahedral_4_chn_clnk_free_rot_general_approx_components(
        Lmbda: npt.NDArray[np.float64],
        n_clnk: npt.NDArray[np.float64],
        b_clnk: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64], float]:
    """Rodrigues vector perturbation and the cross-link junction
    position perturbation for a polydisperse regular tetrahedral 4-chain
    inextensible Gaussian FJC cross-link structure RVE in the free
    rotation limit, as evaluated via closed-form approximation.

    This function determines the Rodrigues vector perturbation and the
    cross-link junction position perturbation for a polydisperse regular
    tetrahedral 4-chain inextensible Gaussian FJC cross-link structure
    RVE in the free rotation limit, as evaluated via closed-form
    approximation.

    Args:
        Lmbda (npt.NDArray[np.float64]): Principal stretch matrix.
        n_clnk (npt.NDArray[np.float64]): Number of chain segments for each chain in the cross-link structure RVE.
        b_clnk (npt.NDArray[np.float64]): Chain segment and/or cross-linker diameter for each chain in the cross-link structure RVE.
    
    Returns:
        tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64], float]:
        Rodrigues vector perturbation, norm of the Rodrigues vector
        perturbation, cross-link junction position perturbation,
        distance between the origin and the cross-link junction position
        perturbation.
    
    """
    # Initialization
    lmbda_0, lmbda_1, _ = Lmbda
    
    n_0, n_1, n_2, n_3 = n_clnk
    prod_n_clnk = np.prod(n_clnk)
    prod_n_clnk_over_n_0 = prod_n_clnk / n_0
    prod_n_clnk_over_n_1 = prod_n_clnk / n_1
    prod_n_clnk_over_n_2 = prod_n_clnk / n_2
    prod_n_clnk_over_n_3 = prod_n_clnk / n_3
    sqrt_prod_n_clnk = np.sqrt(prod_n_clnk)
    sqrt_prod_n_clnk_over_n_0 = np.sqrt(prod_n_clnk_over_n_0)
    sqrt_prod_n_clnk_over_n_1 = np.sqrt(prod_n_clnk_over_n_1)
    sqrt_prod_n_clnk_over_n_2 = np.sqrt(prod_n_clnk_over_n_2)
    sqrt_prod_n_clnk_over_n_3 = np.sqrt(prod_n_clnk_over_n_3)

    b_0, b_1, b_2, b_3 = b_clnk
    prod_b_clnk = np.prod(b_clnk)
    prod_b_clnk_over_b_0 = prod_b_clnk / b_0
    prod_b_clnk_over_b_1 = prod_b_clnk / b_1
    prod_b_clnk_over_b_2 = prod_b_clnk / b_2
    prod_b_clnk_over_b_3 = prod_b_clnk / b_3
    sqrd_prod_b_clnk = prod_b_clnk**2
    sqrd_prod_b_clnk_over_b_0 = prod_b_clnk_over_b_0**2
    sqrd_prod_b_clnk_over_b_1 = prod_b_clnk_over_b_1**2
    sqrd_prod_b_clnk_over_b_2 = prod_b_clnk_over_b_2**2
    sqrd_prod_b_clnk_over_b_3 = prod_b_clnk_over_b_3**2

    a_dnmntr = np.sqrt(3.) * prod_b_clnk * sqrt_prod_n_clnk
    a_0 = (
        prod_b_clnk_over_b_0 * sqrt_prod_n_clnk_over_n_0
        - prod_b_clnk_over_b_1 * sqrt_prod_n_clnk_over_n_1
        - prod_b_clnk_over_b_2 * sqrt_prod_n_clnk_over_n_2
        + prod_b_clnk_over_b_3 * sqrt_prod_n_clnk_over_n_3
    )
    a_1 = (
        prod_b_clnk_over_b_0 * sqrt_prod_n_clnk_over_n_0
        - prod_b_clnk_over_b_1 * sqrt_prod_n_clnk_over_n_1
        + prod_b_clnk_over_b_2 * sqrt_prod_n_clnk_over_n_2
        - prod_b_clnk_over_b_3 * sqrt_prod_n_clnk_over_n_3
    )
    a_2 = (
        -prod_b_clnk_over_b_0 * sqrt_prod_n_clnk_over_n_0
        + prod_b_clnk_over_b_1 * sqrt_prod_n_clnk_over_n_1
        + prod_b_clnk_over_b_2 * sqrt_prod_n_clnk_over_n_2
        - prod_b_clnk_over_b_3 * sqrt_prod_n_clnk_over_n_3
    )
    a_3 = (
        prod_b_clnk_over_b_0 * sqrt_prod_n_clnk_over_n_0
        + prod_b_clnk_over_b_1 * sqrt_prod_n_clnk_over_n_1
        - prod_b_clnk_over_b_2 * sqrt_prod_n_clnk_over_n_2
        - prod_b_clnk_over_b_3 * sqrt_prod_n_clnk_over_n_3
    )
    a_4 = (
        -prod_b_clnk_over_b_0 * sqrt_prod_n_clnk_over_n_0
        + prod_b_clnk_over_b_1 * sqrt_prod_n_clnk_over_n_1
        - prod_b_clnk_over_b_2 * sqrt_prod_n_clnk_over_n_2
        + prod_b_clnk_over_b_3 * sqrt_prod_n_clnk_over_n_3
    )
    a_5 = (
        -prod_b_clnk_over_b_0 * sqrt_prod_n_clnk_over_n_0
        - prod_b_clnk_over_b_1 * sqrt_prod_n_clnk_over_n_1
        + prod_b_clnk_over_b_2 * sqrt_prod_n_clnk_over_n_2
        + prod_b_clnk_over_b_3 * sqrt_prod_n_clnk_over_n_3
    )
    a_0 /= a_dnmntr
    a_1 /= a_dnmntr
    a_2 /= a_dnmntr
    a_3 /= a_dnmntr
    a_4 /= a_dnmntr
    a_5 /= a_dnmntr

    a_6 = (
        sqrd_prod_b_clnk_over_b_0 * prod_n_clnk_over_n_0
        + sqrd_prod_b_clnk_over_b_1 * prod_n_clnk_over_n_1
        + sqrd_prod_b_clnk_over_b_2 * prod_n_clnk_over_n_2
        + sqrd_prod_b_clnk_over_b_3 * prod_n_clnk_over_n_3
    )
    a_6 /= sqrd_prod_b_clnk * prod_n_clnk
    
    # Calculate the Rodrigues vector perturbation and the cross-link
    # junction position perturbation
    tilde_delta_omega_clnk_0 = (3*(-4*a_1*a_2*a_6*(3*a_1*a_4 + 4*a_6)*lmbda_0 \
    - 4*a_1*a_2*a_6*(3*a_3*a_5 + 4*a_6 + (3*a_0*a_2 + 3*a_1*a_4 - 3*a_5**2 \
    + 4*a_6)*lmbda_0**3)*lmbda_1 - (9*(a_0*a_3*a_4 + a_1*a_2*a_5)*(a_0*a_2*a_3 \
    + a_5*(a_1**2 - a_3*a_5)) + 12*(a_0*a_1*(a_2**2 + a_1*a_4) \
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
    
    tilde_delta_omega_clnk_2 = (3*(-4*a_4*a_6*(-3*a_0*a_2*a_3 \
    + a_5*(3*a_1*a_4 + 3*a_3*a_5 + 4*a_6))*lmbda_0 \
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
    
    tilde_delta_y_clnk_0 = (-4*lmbda_0*(-9*a_0**2*a_3*(a_2**2 - a_1*a_4)*lmbda_0*lmbda_1**2*(lmbda_0 \
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

    tilde_delta_y_clnk_1 = (4*lmbda_1*(-9*a_1**3*a_4**2*lmbda_0*(1 + lmbda_0**2*lmbda_1) \
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

    tilde_delta_y_clnk_2 = (4*(-9*a_0**2*a_2**3*lmbda_0**3*lmbda_1**3*(lmbda_0 + lmbda_1) \
    - 9*a_0*a_3*a_4**2*a_5*lmbda_0**2*lmbda_1**2*(1 + lmbda_0**2*lmbda_1) - a_2*(1 \
    + lmbda_0*lmbda_1**2)*(3*a_3*a_5*lmbda_1 + 4*a_6*(lmbda_0 \
    + lmbda_1))*(-3*a_5**2*lmbda_0**2*lmbda_1 + 4*a_6*(1 + lmbda_0**2*lmbda_1)) \
    + 9*a_1**2*lmbda_0**2*lmbda_1**2*(a_0*a_4**2*(1 + lmbda_0**2*lmbda_1) - a_2*a_5**2*(1 \
    + lmbda_0*lmbda_1**2)) \
    - 3*a_1*a_4*lmbda_0*(-3*a_0**2*a_2*lmbda_0**2*lmbda_1**3*(lmbda_0 + lmbda_1) \
    + 3*a_0*a_2**2*lmbda_0*lmbda_1**2*(1 + lmbda_0**2*lmbda_1) \
    - 4*a_0*a_6*lmbda_1**2*(lmbda_0 + lmbda_1)*(1 + lmbda_0**2*lmbda_1) + 4*a_2*a_6*(1 \
    + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2)) \
    - 3*a_0*a_2**2*lmbda_0*lmbda_1*(4*a_6*(lmbda_0 + lmbda_1)*(lmbda_0 + lmbda_1 \
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

def inext_gaussian_fjc_tilde_delta_clnk_cube_8_chn_clnk_free_rot_general_approx_components(
        Lmbda: npt.NDArray[np.float64],
        n_clnk: npt.NDArray[np.float64],
        b_clnk: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64], float]:
    """Rodrigues vector perturbation and the cross-link junction
    position perturbation for a polydisperse cube 8-chain inextensible
    Gaussian FJC cross-link structure RVE in the free rotation limit, as
    evaluated via closed-form approximation.

    This function determines the Rodrigues vector perturbation and the
    cross-link junction position perturbation for a polydisperse cube
    8-chain inextensible Gaussian FJC cross-link structure RVE in the
    free rotation limit, as evaluated via closed-form approximation.

    Args:
        Lmbda (npt.NDArray[np.float64]): Principal stretch matrix.
        n_clnk (npt.NDArray[np.float64]): Number of chain segments for each chain in the cross-link structure RVE.
        b_clnk (npt.NDArray[np.float64]): Chain segment and/or cross-linker diameter for each chain in the cross-link structure RVE.
    
    Returns:
        tuple[npt.NDArray[np.float64], float, npt.NDArray[np.float64], float]:
        Rodrigues vector perturbation, norm of the Rodrigues vector
        perturbation, cross-link junction position perturbation,
        distance between the origin and the cross-link junction position
        perturbation.
    
    """
    # Initialization
    lmbda_0, lmbda_1, _ = Lmbda

    n_0, n_1, n_2, n_3, n_4, n_5, n_6, n_7 = n_clnk
    prod_n_clnk = np.prod(n_clnk)
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

    b_0, b_1, b_2, b_3, b_4, b_5, b_6, b_7 = b_clnk
    prod_b_clnk = np.prod(b_clnk)
    prod_b_clnk_over_b_0 = prod_b_clnk / b_0
    prod_b_clnk_over_b_1 = prod_b_clnk / b_1
    prod_b_clnk_over_b_2 = prod_b_clnk / b_2
    prod_b_clnk_over_b_3 = prod_b_clnk / b_3
    prod_b_clnk_over_b_4 = prod_b_clnk / b_4
    prod_b_clnk_over_b_5 = prod_b_clnk / b_5
    prod_b_clnk_over_b_6 = prod_b_clnk / b_6
    prod_b_clnk_over_b_7 = prod_b_clnk / b_7
    sqrd_prod_b_clnk = prod_b_clnk**2
    sqrd_prod_b_clnk_over_b_0 = prod_b_clnk_over_b_0**2
    sqrd_prod_b_clnk_over_b_1 = prod_b_clnk_over_b_1**2
    sqrd_prod_b_clnk_over_b_2 = prod_b_clnk_over_b_2**2
    sqrd_prod_b_clnk_over_b_3 = prod_b_clnk_over_b_3**2
    sqrd_prod_b_clnk_over_b_4 = prod_b_clnk_over_b_4**2
    sqrd_prod_b_clnk_over_b_5 = prod_b_clnk_over_b_5**2
    sqrd_prod_b_clnk_over_b_6 = prod_b_clnk_over_b_6**2
    sqrd_prod_b_clnk_over_b_7 = prod_b_clnk_over_b_7**2

    a_dnmntr = np.sqrt(3.) * prod_b_clnk * sqrt_prod_n_clnk
    a_0 = (
        prod_b_clnk_over_b_0 * sqrt_prod_n_clnk_over_n_0
        - prod_b_clnk_over_b_1 * sqrt_prod_n_clnk_over_n_1
        + prod_b_clnk_over_b_2 * sqrt_prod_n_clnk_over_n_2
        - prod_b_clnk_over_b_3 * sqrt_prod_n_clnk_over_n_3
        + prod_b_clnk_over_b_4 * sqrt_prod_n_clnk_over_n_4
        - prod_b_clnk_over_b_5 * sqrt_prod_n_clnk_over_n_5
        + prod_b_clnk_over_b_6 * sqrt_prod_n_clnk_over_n_6
        - prod_b_clnk_over_b_7 * sqrt_prod_n_clnk_over_n_7
    )
    a_1 = (
        -prod_b_clnk_over_b_0 * sqrt_prod_n_clnk_over_n_0
        - prod_b_clnk_over_b_1 * sqrt_prod_n_clnk_over_n_1
        + prod_b_clnk_over_b_2 * sqrt_prod_n_clnk_over_n_2
        + prod_b_clnk_over_b_3 * sqrt_prod_n_clnk_over_n_3
        - prod_b_clnk_over_b_4 * sqrt_prod_n_clnk_over_n_4
        - prod_b_clnk_over_b_5 * sqrt_prod_n_clnk_over_n_5
        + prod_b_clnk_over_b_6 * sqrt_prod_n_clnk_over_n_6
        + prod_b_clnk_over_b_7 * sqrt_prod_n_clnk_over_n_7
    )
    a_2 = (
        -prod_b_clnk_over_b_0 * sqrt_prod_n_clnk_over_n_0
        + prod_b_clnk_over_b_1 * sqrt_prod_n_clnk_over_n_1
        - prod_b_clnk_over_b_2 * sqrt_prod_n_clnk_over_n_2
        + prod_b_clnk_over_b_3 * sqrt_prod_n_clnk_over_n_3
        - prod_b_clnk_over_b_4 * sqrt_prod_n_clnk_over_n_4
        + prod_b_clnk_over_b_5 * sqrt_prod_n_clnk_over_n_5
        - prod_b_clnk_over_b_6 * sqrt_prod_n_clnk_over_n_6
        + prod_b_clnk_over_b_7 * sqrt_prod_n_clnk_over_n_7
    )
    a_3 = (
        prod_b_clnk_over_b_0 * sqrt_prod_n_clnk_over_n_0
        + prod_b_clnk_over_b_1 * sqrt_prod_n_clnk_over_n_1
        + prod_b_clnk_over_b_2 * sqrt_prod_n_clnk_over_n_2
        + prod_b_clnk_over_b_3 * sqrt_prod_n_clnk_over_n_3
        - prod_b_clnk_over_b_4 * sqrt_prod_n_clnk_over_n_4
        - prod_b_clnk_over_b_5 * sqrt_prod_n_clnk_over_n_5
        - prod_b_clnk_over_b_6 * sqrt_prod_n_clnk_over_n_6
        - prod_b_clnk_over_b_7 * sqrt_prod_n_clnk_over_n_7
    )
    a_4 = (
        prod_b_clnk_over_b_0 * sqrt_prod_n_clnk_over_n_0
        + prod_b_clnk_over_b_1 * sqrt_prod_n_clnk_over_n_1
        - prod_b_clnk_over_b_2 * sqrt_prod_n_clnk_over_n_2
        - prod_b_clnk_over_b_3 * sqrt_prod_n_clnk_over_n_3
        + prod_b_clnk_over_b_4 * sqrt_prod_n_clnk_over_n_4
        + prod_b_clnk_over_b_5 * sqrt_prod_n_clnk_over_n_5
        - prod_b_clnk_over_b_6 * sqrt_prod_n_clnk_over_n_6
        - prod_b_clnk_over_b_7 * sqrt_prod_n_clnk_over_n_7
    )
    a_5 = (
        -prod_b_clnk_over_b_0 * sqrt_prod_n_clnk_over_n_0
        - prod_b_clnk_over_b_1 * sqrt_prod_n_clnk_over_n_1
        - prod_b_clnk_over_b_2 * sqrt_prod_n_clnk_over_n_2
        - prod_b_clnk_over_b_3 * sqrt_prod_n_clnk_over_n_3
        + prod_b_clnk_over_b_4 * sqrt_prod_n_clnk_over_n_4
        + prod_b_clnk_over_b_5 * sqrt_prod_n_clnk_over_n_5
        + prod_b_clnk_over_b_6 * sqrt_prod_n_clnk_over_n_6
        + prod_b_clnk_over_b_7 * sqrt_prod_n_clnk_over_n_7
    )
    a_0 /= a_dnmntr
    a_1 /= a_dnmntr
    a_2 /= a_dnmntr
    a_3 /= a_dnmntr
    a_4 /= a_dnmntr
    a_5 /= a_dnmntr

    a_6 = (
        sqrd_prod_b_clnk_over_b_0 * prod_n_clnk_over_n_0
        + sqrd_prod_b_clnk_over_b_1 * prod_n_clnk_over_n_1
        + sqrd_prod_b_clnk_over_b_2 * prod_n_clnk_over_n_2
        + sqrd_prod_b_clnk_over_b_3 * prod_n_clnk_over_n_3
        + sqrd_prod_b_clnk_over_b_4 * prod_n_clnk_over_n_4
        + sqrd_prod_b_clnk_over_b_5 * prod_n_clnk_over_n_5
        + sqrd_prod_b_clnk_over_b_6 * prod_n_clnk_over_n_6
        + sqrd_prod_b_clnk_over_b_7 * prod_n_clnk_over_n_7
    )
    a_6 /= sqrd_prod_b_clnk * prod_n_clnk
    
    # Calculate the Rodrigues vector perturbation and the cross-link
    # junction position perturbation
    tilde_delta_omega_clnk_0 = (3*(-8*a_1*a_2*a_6*(3*a_1*a_4 + 8*a_6)*lmbda_0 \
    - 8*a_1*a_2*a_6*(3*a_3*a_5 + 8*a_6 \
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
    
    tilde_delta_omega_clnk_2 = (3*(-8*a_4*a_6*(-3*a_0*a_2*a_3 \
    + a_5*(3*a_1*a_4 + 3*a_3*a_5 + 8*a_6))*lmbda_0 \
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

    
    tilde_delta_y_clnk_0 = (-8*lmbda_0*(-9*a_0**2*a_3*(a_2**2 - a_1*a_4)*lmbda_0*lmbda_1**2*(lmbda_0 \
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

    tilde_delta_y_clnk_1 = (-8*lmbda_1*(9*a_1**3*a_4**2*lmbda_0*(1 + lmbda_0**2*lmbda_1) \
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

    tilde_delta_y_clnk_2 = (8*(-9*a_0**2*a_2**3*lmbda_0**3*lmbda_1**3*(lmbda_0 + lmbda_1) \
    - 9*a_0*a_3*a_4**2*a_5*lmbda_0**2*lmbda_1**2*(1 + lmbda_0**2*lmbda_1) - a_2*(1 \
    + lmbda_0*lmbda_1**2)*(3*a_3*a_5*lmbda_1 + 8*a_6*(lmbda_0 \
    + lmbda_1))*(-3*a_5**2*lmbda_0**2*lmbda_1 + 8*a_6*(1 + lmbda_0**2*lmbda_1)) \
    + 9*a_1**2*lmbda_0**2*lmbda_1**2*(a_0*a_4**2*(1 + lmbda_0**2*lmbda_1) - a_2*a_5**2*(1 \
    + lmbda_0*lmbda_1**2)) \
    - 3*a_1*a_4*lmbda_0*(-3*a_0**2*a_2*lmbda_0**2*lmbda_1**3*(lmbda_0 + lmbda_1) \
    + 3*a_0*a_2**2*lmbda_0*lmbda_1**2*(1 + lmbda_0**2*lmbda_1) \
    - 8*a_0*a_6*lmbda_1**2*(lmbda_0 + lmbda_1)*(1 + lmbda_0**2*lmbda_1) + 8*a_2*a_6*(1 \
    + lmbda_0**2*lmbda_1)*(1 + lmbda_0*lmbda_1**2)) \
    - 3*a_0*a_2**2*lmbda_0*lmbda_1*(8*a_6*(lmbda_0 + lmbda_1)*(lmbda_0 + lmbda_1 \
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

def clnk_free_rot_approx(
        eval_W_clnk_y_flucts: bool,
        use_inext_gaussian_fjc_delta_clnk: bool,
        F: npt.NDArray[np.float64],
        n_clnks_geo_isomrphc_set: npt.NDArray[np.float64],
        b_clnks_geo_isomrphc_set: npt.NDArray[np.float64],
        X_clnks_geo_isomrphc_set: npt.NDArray[np.float64],
        y_clnks_init_geo_isomrphc_set: npt.NDArray[np.float64],
        w_c_func_clnks_geo_isomrphc_set: npt.NDArray[np.object_],
        w_c_args_clnks_geo_isomrphc_set: npt.NDArray[np.object_],
        d2w_c__dy_clnk_dy_clnk_func_clnks_geo_isomrphc_set: npt.NDArray[np.object_],
        d2w_c__dy_clnk_dy_clnk_args_clnks_geo_isomrphc_set: npt.NDArray[np.object_],
        w_c_dfrmtn_func_clnks_geo_isomrphc_set: npt.NDArray[np.object_],
        w_c_dfrmtn_args_clnks_geo_isomrphc_set: npt.NDArray[np.object_]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float, npt.NDArray[np.float64], float, npt.NDArray[np.float64], float, float]:
    """Cross-link structure RVE mechanical response in the free rotation
    limit, as evaluated via closed-form approximation.

    This function determines the mechanical response of a cross-link
    structure RVE in the free rotation limit, as evaluated via
    closed-form approximation.

    Args:
        eval_W_clnk_y_flucts (bool): Boolean indicating if the nondimensional cross-link junction fluctuation free energy ought to be calculated (if True) or not (if False).
        use_inext_gaussian_fjc_delta_clnk (bool): Boolean indicating if the inextensible Gaussian FJC model ought to be used to calculate the optimal cross-link junction position perturbation.
        F (npt.NDArray[np.float64]): Deformation gradient.
        n_clnks_geo_isomrphc_set (npt.NDArray[np.float64]): Number of chain segments for each chain in each cross-link structure RVE in the geometrically isomorphic set of cross-link structures.
        n_clnks_geo_isomrphc_set (npt.NDArray[np.float64]): Chain segment and/or cross-linker diameter for each chain in each cross-link structure RVE in the geometrically isomorphic set of cross-link structures.
        X_clnks_geo_isomrphc_set (npt.NDArray[np.float64]): Initial chain end position for each chain in the cross-link structure RVE in the geometrically isomorphic set of cross-link structures.
        y_clnks_init_geo_isomrphc_set (npt.NDArray[np.float64]): Initial cross-link junction position for each cross-link structure RVE in the geometrically isomorphic set of cross-link structures.
        w_c_func_clnks_geo_isomrphc_set (npt.NDArray[np.object_]): Nondimensional polymer chain free energy function for each chain in the cross-link structure RVE in the geometrically isomorphic set of cross-link structures.
        w_c_args_clnks_geo_isomrphc_set (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE in the geometrically isomorphic set of cross-link structures.
        d2w_c__dy_clnk_dy_clnk_func_clnks_geo_isomrphc_set (npt.NDArray[np.object_]): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function for each chain in the cross-link structure RVE in the geometrically isomorphic set of cross-link structures.
        d2w_c__dy_clnk_dy_clnk_args_clnks_geo_isomrphc_set (npt.NDArray[np.object_]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n) for each chain in the cross-link structure RVE in the geometrically isomorphic set of cross-link structures.
        w_c_dfrmtn_func_clnks_geo_isomrphc_set (npt.NDArray[np.object_]): Nondimensional polymer chain deformation free energy function for each chain in the cross-link structure RVE in the geometrically isomorphic set of cross-link structures.
        w_c_dfrmtn_args_clnks_geo_isomrphc_set (npt.NDArray[np.object_]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n) for each chain in the cross-link structure RVE in the geometrically isomorphic set of cross-link structures.
    
    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], npt.NDArray[np.float64], float, npt.NDArray[np.float64], float, npt.NDArray[np.float64], float, float]:
        Rodrigues vector perturbation describing the cross-link rotation
        for each cross-link structure RVE in the geometrically
        isomorphic set of cross-link structures, norm of the Rodrigues
        vector perturbation describing the cross-link rotation for each
        cross-link structure RVE in the geometrically isomorphic set
        of cross-link structures, cross-link junction position
        perturbation for each cross-link structure RVE in the
        geometrically isomorphic set of cross-link structures,
        distance between the origin and the optimal cross-link junction
        position perturbation for each cross-link structure RVE in the
        geometrically isomorphic set of cross-link structures,
        absolute/equilibrium chain stretch for each chain in each
        cross-link structure RVE in the geometrically isomorphic set
        of cross-link structures, nondimensional cross-link polymer
        chain free energy for each cross-link structure RVE in the
        geometrically isomorphic set of cross-link structures,
        nondimensional cross-link junction fluctuation free energy for
        each cross-link structure RVE in the geometrically isomorphic
        set of cross-link structures, minimal/optimal Rodrigues vector
        perturbation describing the cross-link rotation, minimal/optimal
        norm of the Rodrigues vector perturbation describing the
        cross-link rotation, minimal/optimal cross-link junction
        position perturbation, minimal/optimal distance between the
        origin and the optimal cross-link junction position
        perturbation, minimal/optimal absolute/equilibrium chain stretch
        for each chain in the cross-link, minimal/optimal nondimensional
        cross-link polymer chain free energy, minimal/optimal
        nondimensional cross-link junction fluctuation free energy.
    
    """
    # Boilerplate initialization, checks, and assertions
    if not np.isclose(np.linalg.det(F), 1.):
        error_str = (
            "This methodology is only applicable to cross-link RVEs "
            + "under incompressible deformation. Make sure that the "
            + "deformation is incompressible."
        )
        raise ValueError(error_str)
    clnks_num, k_num = np.shape(n_clnks_geo_isomrphc_set)
    for clnk_indx in range(clnks_num):
        X_clnk = X_clnks_geo_isomrphc_set[clnk_indx]
        X_hat_clnk = x_hat_clnk_func(X_clnk)
        com_X_hat_clnk = com_x_clnk_func(X_hat_clnk)
        y_clnk_init = y_clnks_init_geo_isomrphc_set[clnk_indx]
        if (not np.allclose(com_X_hat_clnk, np.zeros(3)) or
            not np.allclose(y_clnk_init, np.zeros(3))):
            error_str = (
                "This methodology is only applicable to "
                + "well-structured cross-link RVEs. Make sure that the "
                + "initial center-of-mass of the cross-link composed "
                + "of unit-length chains is located at the origin and "
                + "the initial position of the cross-link is at the "
                + "origin."
            )
            raise ValueError(error_str)
        clnk = False
        if k_num == 4:
            clnk = np.allclose(
                X_hat_clnk, regular_tetrahedral_4_chn_clnk_X_hat_clnk_func())
        elif k_num == 8:
            clnk = np.allclose(X_hat_clnk, cube_8_chn_clnk_X_hat_clnk_func())
        if not clnk:
            error_str = (
                "This function is only applicable for the regular "
                + "tetrahedral 4-chain cross-link or cube 8-chain "
                + "cross-link structures. Make sure that the "
                + "cross-link structure corresponds to one of the "
                + "aforementioned cross-link structures."
            )
            raise ValueError(error_str)
    if not use_inext_gaussian_fjc_delta_clnk:
        error_str = (
            "This approximated perturbed deformation response of the "
            + "polydisperse regular tetrahedral 4-chain cross-link RVE "
            + "and the polydisperse cube 8-chain cross-link RVE is "
            + "specifically formulated for the case of the "
            + "inextensible Gaussian FJC."
        )
        raise ValueError(error_str)
    if k_num == 4:
        Q_clnk_m, y_clnk_m = (
            monodisperse_regular_tetrahedral_4_chn_clnk_free_rot(F)
        )
    elif k_num == 8:
        Q_clnk_m, y_clnk_m = monodisperse_cube_8_chn_clnk_free_rot(F)
    Lmbda, _ = principal_stretch_decomposition(F)
    lmbda_0, lmbda_1, _ = Lmbda
    F_Lmbda = np.diag(np.asarray([lmbda_0, lmbda_1, 1./(lmbda_0*lmbda_1)]))
    tilde_delta_omega_clnks_geo_isomrphc_set = np.zeros((clnks_num, 3))
    tilde_delta_omega_clnk_norm_clnks_geo_isomrphc_set = np.zeros(clnks_num)
    tilde_delta_y_clnks_geo_isomrphc_set = np.zeros((clnks_num, 3))
    tilde_delta_y_clnk_norm_clnks_geo_isomrphc_set = np.zeros(clnks_num)
    gamma_clnks_geo_isomrphc_set = np.zeros((clnks_num, k_num))
    W_clnk_chns_clnks_geo_isomrphc_set = np.zeros(clnks_num)
    W_clnk_y_flucts_clnks_geo_isomrphc_set = np.zeros(clnks_num)

    # Evaluate the mechanical response of each cross-link structure RVE
    # in the geometrically isomorphic set of cross-link structures
    for clnk_indx in range(clnks_num):
        n_clnk = n_clnks_geo_isomrphc_set[clnk_indx]
        b_clnk = b_clnks_geo_isomrphc_set[clnk_indx]
        X_clnk = X_clnks_geo_isomrphc_set[clnk_indx]
        
        # For each cross-link structure in the geometrically isomorphic
        # set, calculate the Rodrigues vector perturbation and the
        # cross-link junction position perturbation
        if k_num == 4:
            (tilde_delta_omega_clnk, tilde_delta_omega_clnk_norm,
             tilde_delta_y_clnk, tilde_delta_y_clnk_norm) = (
                inext_gaussian_fjc_tilde_delta_clnk_regular_tetrahedral_4_chn_clnk_free_rot_general_approx_components(
                    Lmbda, n_clnk, b_clnk)
            )
        elif k_num == 8:
            (tilde_delta_omega_clnk, tilde_delta_omega_clnk_norm,
             tilde_delta_y_clnk, tilde_delta_y_clnk_norm) = (
                inext_gaussian_fjc_tilde_delta_clnk_cube_8_chn_clnk_free_rot_general_approx_components(
                    Lmbda, n_clnk, b_clnk)
            )
        
        # Calculate the nondimensional cross-link chain free energy
        gamma_clnk = gamma_approx_clnk_func(
            F_Lmbda, X_clnk, Q_clnk_m, y_clnk_m,
            Q_axis_angle(tilde_delta_omega_clnk), tilde_delta_y_clnk,
            n_clnk, b_clnk)
        W_clnk_chns = W_clnk_chns_func(
            gamma_clnk, n_clnk, w_c_func_clnks_geo_isomrphc_set[clnk_indx],
            w_c_args_clnks_geo_isomrphc_set[clnk_indx],
            w_c_dfrmtn_func_clnks_geo_isomrphc_set[clnk_indx],
            w_c_dfrmtn_args_clnks_geo_isomrphc_set[clnk_indx])
        
        # If called for, calculate the nondimensional cross-link
        # junction fluctuation free energy
        W_clnk_y_flucts = 0.
        if eval_W_clnk_y_flucts:
            gamma_vec_clnk = gamma_approx_vec_clnk_func(
                F_Lmbda, X_clnk, Q_clnk_m, y_clnk_m,
                Q_axis_angle(tilde_delta_omega_clnk), tilde_delta_y_clnk,
                n_clnk, b_clnk)
            W_clnk_y_flucts = W_clnk_y_flucts_func(
                gamma_vec_clnk, gamma_clnk, n_clnk,
                d2w_c__dy_clnk_dy_clnk_func_clnks_geo_isomrphc_set[clnk_indx],
                d2w_c__dy_clnk_dy_clnk_args_clnks_geo_isomrphc_set[clnk_indx])
        
        # Store the mechanical response of each cross-link structure
        tilde_delta_omega_clnks_geo_isomrphc_set[clnk_indx] = (
            tilde_delta_omega_clnk
        )
        tilde_delta_omega_clnk_norm_clnks_geo_isomrphc_set[clnk_indx] = (
            tilde_delta_omega_clnk_norm
        )
        tilde_delta_y_clnks_geo_isomrphc_set[clnk_indx] = tilde_delta_y_clnk
        tilde_delta_y_clnk_norm_clnks_geo_isomrphc_set[clnk_indx] = (
            tilde_delta_y_clnk_norm
        )
        gamma_clnks_geo_isomrphc_set[clnk_indx] = gamma_clnk
        W_clnk_chns_clnks_geo_isomrphc_set[clnk_indx] = W_clnk_chns
        W_clnk_y_flucts_clnks_geo_isomrphc_set[clnk_indx] = W_clnk_y_flucts
    
    # Extract the mechanical response associated with the
    # minimal/optimal nondimensional cross-link chain free energy from
    # the geometrically isomorphic set of cross-link structures
    W_clnk_chns_min_clnk_indx = np.argmin(W_clnk_chns_clnks_geo_isomrphc_set)
    
    return (
        tilde_delta_omega_clnks_geo_isomrphc_set,
        tilde_delta_omega_clnk_norm_clnks_geo_isomrphc_set,
        tilde_delta_y_clnks_geo_isomrphc_set,
        tilde_delta_y_clnk_norm_clnks_geo_isomrphc_set,
        gamma_clnks_geo_isomrphc_set, W_clnk_chns_clnks_geo_isomrphc_set,
        W_clnk_y_flucts_clnks_geo_isomrphc_set,
        tilde_delta_omega_clnks_geo_isomrphc_set[W_clnk_chns_min_clnk_indx],
        tilde_delta_omega_clnk_norm_clnks_geo_isomrphc_set[W_clnk_chns_min_clnk_indx],
        tilde_delta_y_clnks_geo_isomrphc_set[W_clnk_chns_min_clnk_indx],
        tilde_delta_y_clnk_norm_clnks_geo_isomrphc_set[W_clnk_chns_min_clnk_indx],
        gamma_clnks_geo_isomrphc_set[W_clnk_chns_min_clnk_indx],
        W_clnk_chns_clnks_geo_isomrphc_set[W_clnk_chns_min_clnk_indx],
        W_clnk_y_flucts_clnks_geo_isomrphc_set[W_clnk_chns_min_clnk_indx]
    )