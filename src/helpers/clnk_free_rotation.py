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
    regular_tetrahedral_4_chn_clnk_X_hat_clnk_func,
    regular_octahedral_6_chn_clnk_X_hat_clnk_func,
    cube_8_chn_clnk_X_hat_clnk_func,
    x_hat_clnk_func,
    com_x_clnk_func,
    chull_eqs_clnk_func,
    x_clnk_min_max_func,
    n_clnk_mean_func,
    n_clnk_geo_mean_func
)
from src.helpers.continuum_mechanics import principal_stretch_decomposition
from src.helpers.clnk_deformation import (
    monodisperse_y_clnk,
    gamma_clnk_func,
    gamma_vec_clnk_func,
    gamma_approx_clnk_func,
    gamma_approx_vec_clnk_func,
    W_clnk_chns_func,
    W_clnk_y_flucts_func
)
from src.helpers.rotations import Q_axis_angle
from src.helpers.combinatorics import indcs_permutations

def monodisperse_regular_tetrahedral_4_chn_clnk_free_rot(
        F: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Optimal solution to the cross-link rotation and cross-link
    junction position for the monodisperse classical regular tetrahedral
    4-chain cross-link structure RVE in the free rotation limit.
    
    This function returns the optimal solution to the cross-link
    rotation and cross-link junction position for the monodisperse
    classical regular tetrahedral 4-chain cross-link structure RVE in
    the free rotation limit.

    Args:
        F (npt.NDArray[np.floating]): Deformation gradient.

    Returns:
        npt.NDArray[np.floating]: Optimal solution to the cross-link
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
        F: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Optimal solution to the cross-link rotation and cross-link
    junction position for the monodisperse classical regular octahedral
    6-chain cross-link structure RVE in the free rotation limit.
    
    This function returns the optimal solution to the cross-link
    rotation and cross-link junction position for the monodisperse
    classical regular octahedral 6-chain cross-link structure RVE in
    the free rotation limit.

    Args:
        F (npt.NDArray[np.floating]): Deformation gradient.

    Returns:
        npt.NDArray[np.floating]: Optimal solution to the cross-link
        rotation and cross-link junction position for the monodisperse
        classical regular octahedral 6-chain cross-link structure RVE
        in the free rotation limit.
    
    """
    return np.full_like(F, np.nan), monodisperse_y_clnk()

def monodisperse_cube_8_chn_clnk_free_rot(
        F: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Optimal solution to the cross-link rotation and cross-link
    junction position for the monodisperse classical cube 8-chain
    cross-link structure RVE in the free rotation limit.
    
    This function returns the optimal solution to the cross-link
    rotation and cross-link junction position for the monodisperse
    classical cube 8-chain cross-link structure RVE in the free rotation
    limit.

    Args:
        F (npt.NDArray[np.floating]): Deformation gradient.

    Returns:
        npt.NDArray[np.floating]: Optimal solution to the cross-link
        rotation and cross-link junction position for the monodisperse
        classical cube 8-chain cross-link structure RVE in the free
        rotation limit.
    
    """
    _, P = principal_stretch_decomposition(F)
    return P, monodisperse_y_clnk()

def gamma_monodisperse_clnk_free_rot(
        Lmbda: npt.NDArray[np.floating],
        gamma_clnk_init: npt.NDArray[np.floating | np.integer]) -> npt.NDArray[np.floating]:
    """Absolute/Equilibrium chain stretch for each chain in a
    monodisperse classical regular tetrahedral 4-chain, regular
    octahedral 6-chain, or cube 8-chain cross-link structure RVE in the
    free rotation limit.
    
    This function returns the absolute/equilibrium chain stretch for
    each chain in a monodisperse classical regular tetrahedral 4-chain,
    regular octahedral 6-chain, or cube 8-chain cross-link structure RVE
    in the free rotation limit.

    Args:
        Lmbda (npt.NDArray[np.floating]): Principal stretch matrix.
        gamma_clnk_init (npt.NDArray[np.floating | np.integer]): Initial absolute/equilibrium chain stretch for each chain in the cross-link structure RVE.

    Returns:
        npt.NDArray[np.floating]: Absolute/Equilibrium chain stretch for
        each chain in a monodisperse classical regular tetrahedral
        4-chain, regular octahedral 6-chain, or cube 8-chain cross-link
        structure RVE in the free rotation limit.
    
    """
    return (
        gamma_clnk_init[0] * np.sqrt(np.sum(np.power(Lmbda, 2))/3.)
        * np.ones(np.shape(gamma_clnk_init)[0])
    )

def monodisperse_clnk_free_rot(
        eval_W_clnk_chns: bool,
        eval_W_clnk_y_flucts: bool,
        F: npt.NDArray[np.floating],
        Lmbda: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float,
        X_clnk: npt.NDArray[np.floating],
        y_clnk_init: npt.NDArray[np.floating],
        gamma_clnk_init: npt.NDArray[np.floating],
        w_c_func,
        w_c_args: tuple[float] | tuple[None],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float] | tuple[None],
        d2w_c__dy_clnk_dy_clnk_func,
        d2w_c__dy_clnk_dy_clnk_args: tuple[float] | tuple[None]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, float, float]:
    """Monodisperse cross-link structure RVE mechanical response in the
    free rotation limit.

    This function determines the mechanical response of a monodisperse
    cross-link structure RVE in the free rotation limit.

    Args:
        eval_W_clnk_chns (bool): Boolean indicating if the nondimensional cross-link polymer chain free energy ought to be calculated (if True) or not (if False).
        eval_W_clnk_y_flucts (bool): Boolean indicating if the nondimensional cross-link junction fluctuation free energy ought to be calculated (if True) or not (if False).
        F (npt.NDArray[np.floating]): Deformation gradient.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        y_clnk_init (npt.NDArray[np.floating]): Initial cross-link junction position.
        gamma_clnk_init (npt.NDArray[np.floating]): Initial absolute/equilibrium chain stretch for each chain in the cross-link structure RVE.
        w_c_func (function): Nondimensional polymer chain free energy function.
        w_c_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        w_c_dfrmtn_func (function): Nondimensional polymer chain deformation free energy function.
        w_c_dfrmtn_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        d2w_c__dy_clnk_dy_clnk_func (function): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function.
        d2w_c__dy_clnk_dy_clnk_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n).
    
    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, float, float]:
        Optimal monodisperse cross-link rotation, optimal monodisperse
        cross-link junction position, distance between the origin and
        the optimal monodisperse cross-link junction position,
        absolute/equilibrium chain stretch for each chain in the
        monodisperse cross-link, nondimensional monodisperse cross-link
        polymer chain free energy, nondimensional monodisperse
        cross-link junction fluctuation free energy, nondimensional
        monodisperse cross-link free energy.
    
    """
    # Boilerplate initialization, checks, and assertions
    k_num = np.shape(n_clnk)[0]
    X_hat_clnk = x_hat_clnk_func(X_clnk)
    com_X_hat_clnk = com_x_clnk_func(X_hat_clnk)
    if (not np.allclose(n_clnk, n_clnk[0]*np.ones_like(n_clnk)) or
        not np.allclose(gamma_clnk_init, gamma_clnk_init[0]*np.ones_like(gamma_clnk_init)) or
        not np.allclose(com_X_hat_clnk, np.zeros(3)) or
        not np.allclose(y_clnk_init, np.zeros(3))):
        error_str = (
            "This function is only applicable for well-structured "
            + "cross-links of monodisperse segment number. Make "
            + "sure that every chain has the same number of segments, "
            + "the initial absolute/equilibrium chain stretch is the "
            + "same for each chain, the initial position of the "
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
    
    # If called for, calculate each component of the nondimensional
    # cross-link free energy
    W_clnk_chns_star = 0.
    W_clnk_y_flucts_star = 0.
    if eval_W_clnk_chns:
        W_clnk_chns_star = W_clnk_chns_func(
            gamma_clnk_star, n_clnk, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    if eval_W_clnk_y_flucts:
        # Calculate the absolute/equilibrium chain stretch vector for
        # each chain
        gamma_vec_clnk_star = gamma_vec_clnk_func(
            False, F, X_clnk, Q_clnk_star, y_clnk_star, n_clnk, b)
        W_clnk_y_flucts_star = W_clnk_y_flucts_func(
            gamma_vec_clnk_star, gamma_clnk_star, n_clnk,
            d2w_c__dy_clnk_dy_clnk_func, d2w_c__dy_clnk_dy_clnk_args)

    # Calculate the nondimensional cross-link free energy
    W_clnk_star = W_clnk_chns_star + W_clnk_y_flucts_star
    
    return (
        Q_clnk_star, y_clnk_star, y_clnk_star_norm, gamma_clnk_star,
        W_clnk_chns_star, W_clnk_y_flucts_star, W_clnk_star
    )

def W_clnk_chns_free_rot(
        omega_clnk_y_clnk: npt.NDArray[np.floating],
        F: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float,
        X_clnk: npt.NDArray[np.floating],
        w_c_func,
        w_c_args: tuple[float] | tuple[None],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float] | tuple[None]) -> float:
    """Nondimensional cross-link polymer chain free energy in the free
    rotation limit.

    This function supplies the nondimensional cross-link polymer chain
    free energy in the free rotation limit as a suitable objective
    function for constrained minimization.

    Args:
        omega_clnk_y_clnk (npt.NDArray[np.floating]): Horizontally stacked Rodrigues vector and cross-link junction position for the cross-link structure RVE.
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
        the free rotation limit.
    
    """
    # Gather the Rodrigues vector and the cross-link junction position
    omega_clnk, y_clnk = omega_clnk_y_clnk[:3], omega_clnk_y_clnk[3:]
    # Calculate the absolute/equilibrium chain stretch for each chain
    # (using the ideal numerics form for the chain end-to-end vector)
    gamma_clnk = gamma_clnk_func(
        True, F, X_clnk, Q_axis_angle(omega_clnk), y_clnk, n_clnk, b)
    # Calculate the nondimensional cross-link polymer chain free energy
    return (
        W_clnk_chns_func(
            gamma_clnk, n_clnk, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    )

def clnk_free_rot_cnstrnd_mnmztn(
        eval_W_clnk_chns: bool,
        eval_W_clnk_y_flucts: bool,
        cnstrnd_mnmztn_scope: str,
        cnstrnd_mnmztn_method: str,
        rng: np.random.Generator,
        F: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float,
        X_clnk: npt.NDArray[np.floating],
        omega_clnk: npt.NDArray[np.floating],
        y_clnk: npt.NDArray[np.floating],
        w_c_func,
        w_c_args: tuple[float] | tuple[None],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float] | tuple[None],
        d2w_c__dy_clnk_dy_clnk_func,
        d2w_c__dy_clnk_dy_clnk_args: tuple[float] | tuple[None]):
    """Cross-link structure RVE mechanical response in the free rotation
    limit, as evaluated numerically via constrained minimization.

    This function determines the mechanical response of a cross-link
    structure RVE in the free rotation limit, as evaluated numerically
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
        omega_clnk (npt.NDArray[np.floating]): Prior Rodrigues vector describing the prior cross-link structure RVE orientation.
        y_clnk (npt.NDArray[np.floating]): Prior cross-link junction position.
        w_c_func (function): Nondimensional polymer chain free energy function.
        w_c_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        w_c_dfrmtn_func (function): Nondimensional polymer chain deformation free energy function.
        w_c_dfrmtn_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        d2w_c__dy_clnk_dy_clnk_func (function): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function.
        d2w_c__dy_clnk_dy_clnk_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n).
    
    Returns:
        tuple[npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, float, float]:
        Optimal Rodrigues vector describing the cross-link rotation,
        norm of the optimal Rodrigues vector describing the cross-link
        rotation, optimal cross-link junction position, distance between
        the origin and the optimal cross-link junction position,
        absolute/equilibrium chain stretch for each chain in the
        cross-link, nondimensional cross-link polymer chain free energy,
        nondimensional cross-link junction fluctuation free energy,
        nondimensional cross-link free energy.
    
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
        F, n_clnk, b, X_clnk,
        w_c_func, w_c_args, w_c_dfrmtn_func, w_c_dfrmtn_args
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
    gamma_clnk = gamma_clnk_func(True, F, X_clnk, Q_clnk, y_clnk, n_clnk, b)
    
    # If called for, calculate each component of the nondimensional
    # cross-link free energy
    W_clnk_chns = 0.
    W_clnk_y_flucts = 0.
    if eval_W_clnk_chns:
        W_clnk_chns = W_clnk_chns_func(
            gamma_clnk, n_clnk, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    if eval_W_clnk_y_flucts:
        # Calculate the absolute/equilibrium chain stretch vector for
        # each chain
        gamma_vec_clnk = gamma_vec_clnk_func(
            True, F, X_clnk, Q_clnk, y_clnk, n_clnk, b)
        W_clnk_y_flucts = W_clnk_y_flucts_func(
            gamma_vec_clnk, gamma_clnk, n_clnk,
            d2w_c__dy_clnk_dy_clnk_func, d2w_c__dy_clnk_dy_clnk_args)
    
    # Calculate the nondimensional cross-link free energy
    W_clnk = W_clnk_chns + W_clnk_y_flucts
    
    return (
        omega_clnk, omega_clnk_norm, y_clnk, y_clnk_norm, gamma_clnk,
        W_clnk_chns, W_clnk_y_flucts, W_clnk
    )

def inext_gaussian_fjc_tilde_delta_clnk_regular_tetrahedral_4_chn_clnk_free_rot_general_approx_components(
        Lmbda: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float) -> tuple[npt.NDArray[np.floating], float, npt.NDArray[np.floating], float]:
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
        Lmbda (npt.NDArray[np.floating]): Principal stretch matrix.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
    
    Returns:
        tuple[npt.NDArray[np.floating], float, npt.NDArray[np.floating], float]:
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
    sqrt_prod_n_clnk_over_n_0 = np.sqrt(prod_n_clnk_over_n_0)
    sqrt_prod_n_clnk_over_n_1 = np.sqrt(prod_n_clnk_over_n_1)
    sqrt_prod_n_clnk_over_n_2 = np.sqrt(prod_n_clnk_over_n_2)
    sqrt_prod_n_clnk_over_n_3 = np.sqrt(prod_n_clnk_over_n_3)
    n_clnk_geo_mean = n_clnk_geo_mean_func(n_clnk)

    a_0 = (
        sqrt_prod_n_clnk_over_n_3 - sqrt_prod_n_clnk_over_n_2
        - sqrt_prod_n_clnk_over_n_1 + sqrt_prod_n_clnk_over_n_0
    )
    a_1 = (
        sqrt_prod_n_clnk_over_n_3 - sqrt_prod_n_clnk_over_n_2
        + sqrt_prod_n_clnk_over_n_1 - sqrt_prod_n_clnk_over_n_0
    )
    a_2 = (
        sqrt_prod_n_clnk_over_n_3 + sqrt_prod_n_clnk_over_n_2
        - sqrt_prod_n_clnk_over_n_1 - sqrt_prod_n_clnk_over_n_0
    )
    a_3 = (
        2.
        * (-2.*n_1*n_2*np.sqrt(n_0*n_3)+prod_n_clnk_over_n_0+n_0*(n_3*(np.sqrt(n_1)-np.sqrt(n_2))**2+n_1*n_2))
    )
    a_4 = (
        -2.
        * (-2.*n_1*n_3*np.sqrt(n_0*n_2)+prod_n_clnk_over_n_0+n_0*(-2.*n_2*np.sqrt(n_1*n_3)+n_2*n_3+n_1*(n_2+n_3)))
    )
    a_5 = (
        3. * prod_n_clnk_over_n_0 - 2. * np.sqrt(n_0)
        * (n_2*n_3*np.sqrt(n_1)+n_1*(n_2*np.sqrt(n_3)+n_3*np.sqrt(n_2)))
        + n_0 * (3.*n_2*n_3+n_1*(-2.*np.sqrt(n_2*n_3)+3.*(n_2+n_3))-2.*np.sqrt(n_1)*(n_2*np.sqrt(n_3)+n_3*np.sqrt(n_2)))
    )
    a_6 = (
        prod_n_clnk_over_n_3 + prod_n_clnk_over_n_2 + prod_n_clnk_over_n_1
        + prod_n_clnk_over_n_0
    )
    
    # Calculate the Rodrigues vector perturbation and the cross-link
    # junction position perturbation
    tilde_delta_omega_clnk_dnmntr = (
        a_2 * (a_0**2*(lmbda_0*lmbda_1)**4+(a_1*lmbda_0)**2+(a_2*lmbda_1)**2)
    )
    tilde_delta_omega_clnk_0 = 0.
    tilde_delta_omega_clnk_1 = (
        a_0
        * (a_3*(lmbda_0*lmbda_1)**4-(a_1*lmbda_0)**2-(a_2*lmbda_1)**2)
        / tilde_delta_omega_clnk_dnmntr
    )
    tilde_delta_omega_clnk_2 = (
        a_1
        * (a_0**2*(lmbda_0*lmbda_1)**4+a_4*lmbda_0**2+(a_2*lmbda_1)**2)
        / tilde_delta_omega_clnk_dnmntr
    )
    
    tilde_delta_omega_clnk = np.asarray(
        [
            tilde_delta_omega_clnk_0,
            tilde_delta_omega_clnk_1,
            tilde_delta_omega_clnk_2
        ])
    tilde_delta_omega_clnk_norm = np.linalg.norm(tilde_delta_omega_clnk)

    tilde_delta_y_clnk_dnmntr = (
        np.sqrt(3.) * a_6
        * (a_0**2*(lmbda_0*lmbda_1)**4+(a_1*lmbda_0)**2+(a_2*lmbda_1)**2)
    )
    tilde_delta_y_clnk_0 = (
        -b * a_2 * a_5 * n_clnk_geo_mean**2 * lmbda_0 * lmbda_1**2
        / tilde_delta_y_clnk_dnmntr
    )
    tilde_delta_y_clnk_1 = (
        b * a_1 * a_5 * n_clnk_geo_mean**2 * lmbda_0**2 * lmbda_1
        / tilde_delta_y_clnk_dnmntr
    )
    tilde_delta_y_clnk_2 = (
        b * a_0 * a_5 * n_clnk_geo_mean**2 * (lmbda_0*lmbda_1)**3
        / tilde_delta_y_clnk_dnmntr
    )
    
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

def inext_gaussian_fjc_tilde_delta_clnk_regular_tetrahedral_4_chn_clnk_free_rot_general_approx(
        Lmbda: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Rodrigues vector perturbation and the cross-link junction
    position perturbation for all chain permutations of a polydisperse
    regular tetrahedral 4-chain inextensible Gaussian FJC cross-link
    structure RVE in the free rotation limit, as evaluated via
    closed-form approximation.

    This function determines the Rodrigues vector perturbation and the
    cross-link junction position perturbation for all chain permutations
    of a polydisperse regular tetrahedral 4-chain inextensible Gaussian
    FJC cross-link structure RVE in the free rotation limit, as
    evaluated via closed-form approximation.

    Args:
        Lmbda (npt.NDArray[np.floating]): Principal stretch matrix.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
    
    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Chain segment number permutations, Rodrigues vector perturbation
        for each cross-link permutation, norm of the Rodrigues vector
        perturbation for each cross-link permutation, cross-link
        junction position perturbation for each cross-link permutation,
        distance between the origin and the cross-link junction position
        perturbation for each cross-link permutation.
    
    """
    # Generate all cross-link structures under chain permutation
    n_clnk_indcs_permutations = indcs_permutations(np.shape(n_clnk)[0])
    n_clnk_permutations = n_clnk[n_clnk_indcs_permutations]
    num_permutations = np.shape(n_clnk_permutations)[0]
    
    # Initialization
    tilde_delta_omega_clnk_permutations = np.zeros((num_permutations, 3))
    tilde_delta_omega_clnk_norm_permutations = np.zeros(num_permutations)
    tilde_delta_y_clnk_permutations = np.zeros((num_permutations, 3))
    tilde_delta_y_clnk_norm_permutations = np.zeros(num_permutations)

    # For each cross-link structure under chain permutation, calculate
    # the Rodrigues vector perturbation and the cross-link junction
    # position perturbation
    for prmttn in range(num_permutations):
        (tilde_delta_omega_clnk, tilde_delta_omega_clnk_norm,
         tilde_delta_y_clnk, tilde_delta_y_clnk_norm) = (
            inext_gaussian_fjc_tilde_delta_clnk_regular_tetrahedral_4_chn_clnk_free_rot_general_approx_components(
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

def inext_gaussian_fjc_tilde_delta_clnk_1_3_bimodal_regular_tetrahedral_4_chn_clnk_free_rot_approx(
        Lmbda: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Rodrigues vector perturbation and the cross-link junction
    position perturbation for all chain permutations of a bimodal 1-3
    regular tetrahedral 4-chain inextensible Gaussian FJC cross-link
    structure RVE in the free rotation limit, as evaluated via
    closed-form approximation.

    This function determines the Rodrigues vector perturbation and the
    cross-link junction position perturbation for all chain permutations
    of a bimodal 1-3 regular tetrahedral 4-chain inextensible Gaussian
    FJC cross-link structure RVE in the free rotation limit, as
    evaluated via closed-form approximation.

    Args:
        Lmbda (npt.NDArray[np.floating]): Principal stretch matrix.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
    
    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Chain segment number permutations, Rodrigues vector perturbation
        for each cross-link permutation, norm of the Rodrigues vector
        perturbation for each cross-link permutation, cross-link
        junction position perturbation for each cross-link permutation,
        distance between the origin and the cross-link junction position
        perturbation for each cross-link permutation.
    
    """
    # Boilerplate initialization, checks, and assertions
    unique_n_clnk, unique_n_clnk_counts = np.unique(n_clnk, return_counts=True)
    if (np.shape(n_clnk)[0] != 4 or np.shape(unique_n_clnk)[0] != 2 or
        (unique_n_clnk_counts[0] != 1 and unique_n_clnk_counts[0] != 3)):
        error_str = (
            "This function is only applicable to regular tetrahedral "
            + "4-chain cross-links with a bimodal number of chain "
            + "segment numbers where there are 3 chains with one "
            + "segment number and 1 chain with the other segment number."
        )
        raise ValueError(error_str)
    lmbda_0, lmbda_1, _ = Lmbda
    n_0, n_1 = unique_n_clnk
    n_counts_0, _ = unique_n_clnk_counts
    if n_counts_0 == 1: n_alpha, n_beta = n_0, n_1
    elif n_counts_0 == 3: n_beta, n_alpha = n_0, n_1
    
    # Generate all cross-link structures under chain permutation
    num_permutations = 4
    n_clnk_permutations = np.tile(n_clnk, (num_permutations, 1))
    s = np.asarray([[1, 1], [1, -1], [-1, 1], [-1, -1]], dtype=int)
    tilde_delta_omega_clnk_permutations = np.zeros((num_permutations, 3))
    tilde_delta_y_clnk_permutations = np.zeros((num_permutations, 3))

    # Initialization
    tilde_delta_omega_clnk_dnmntr = (
        lmbda_0**2 + lmbda_1**2 + (lmbda_0*lmbda_1)**4
    )
    tilde_delta_omega_clnk_0_term = 0.
    tilde_delta_omega_clnk_1_term = (
        3. * (lmbda_0**2+lmbda_1**2) / tilde_delta_omega_clnk_dnmntr - 2.
    )
    tilde_delta_omega_clnk_2_term = (
        3. * lmbda_0**2 / tilde_delta_omega_clnk_dnmntr - 1.
    )

    tilde_delta_y_clnk_dnmntr = (
        2. * (3.*n_alpha + n_beta) * tilde_delta_omega_clnk_dnmntr
    )
    tilde_delta_y_clnk_0_term = (
        np.sqrt(3.*n_alpha*n_beta) * (np.sqrt(n_beta)-np.sqrt(n_alpha))
        * lmbda_0 * lmbda_1 * (lmbda_0+lmbda_1) / tilde_delta_y_clnk_dnmntr
    )
    tilde_delta_y_clnk_1_term = tilde_delta_y_clnk_0_term
    tilde_delta_y_clnk_2_term = (
        np.sqrt(3.*n_alpha*n_beta) * (np.sqrt(n_beta)-np.sqrt(n_alpha))
        * (lmbda_0*lmbda_1)**3 / tilde_delta_y_clnk_dnmntr
    )

    # For each cross-link structure under chain permutation, calculate
    # the Rodrigues vector perturbation and the cross-link junction
    # position perturbation
    for prmttn in range(num_permutations):
        s_1, s_2 = s[prmttn]
        tilde_delta_omega_clnk_0 = s_1 * s_2 * tilde_delta_omega_clnk_0_term
        tilde_delta_omega_clnk_1 = s_1 * tilde_delta_omega_clnk_1_term
        tilde_delta_omega_clnk_2 = s_2 * tilde_delta_omega_clnk_2_term
        tilde_delta_y_clnk_0 = s_1 * s_2 * tilde_delta_y_clnk_0_term
        tilde_delta_y_clnk_1 = s_1 * tilde_delta_y_clnk_1_term
        tilde_delta_y_clnk_2 = s_2 * tilde_delta_y_clnk_2_term
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

def inext_gaussian_fjc_tilde_delta_clnk_2_2_bimodal_regular_tetrahedral_4_chn_clnk_free_rot_approx(
        Lmbda: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Rodrigues vector perturbation and the cross-link junction
    position perturbation for all chain permutations of a bimodal 2-2
    regular tetrahedral 4-chain inextensible Gaussian FJC cross-link
    structure RVE in the free rotation limit, as evaluated via
    closed-form approximation.

    This function determines the Rodrigues vector perturbation and the
    cross-link junction position perturbation for all chain permutations
    of a bimodal 2-2 regular tetrahedral 4-chain inextensible Gaussian
    FJC cross-link structure RVE in the free rotation limit, as
    evaluated via closed-form approximation.

    Args:
        Lmbda (npt.NDArray[np.floating]): Principal stretch matrix.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
    
    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Chain segment number permutations, Rodrigues vector perturbation
        for each cross-link permutation, norm of the Rodrigues vector
        perturbation for each cross-link permutation, cross-link
        junction position perturbation for each cross-link permutation,
        distance between the origin and the cross-link junction position
        perturbation for each cross-link permutation.
    
    """
    # Boilerplate initialization, checks, and assertions
    unique_n_clnk, unique_n_clnk_counts = np.unique(n_clnk, return_counts=True)
    if (np.shape(n_clnk)[0] != 4 or np.shape(unique_n_clnk)[0] != 2 or
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
    n_clnk_geo_mean = n_clnk_geo_mean_func(n_clnk) # = np.sqrt(n_a*n_b)
    n_clnk_mean = n_clnk_mean_func(n_clnk)
    eta = n_clnk_geo_mean / n_clnk_mean
    lmbda_0, lmbda_1, lmbda_2 = Lmbda
    
    # Generate all cross-link structures under chain permutation
    num_permutations = 6
    n_clnk_permutations = np.tile(n_clnk, (num_permutations, 1))
    
    # For each cross-link structure under chain permutation, calculate
    # the Rodrigues vector perturbation and the cross-link junction
    # position perturbation
    tilde_delta_omega_clnk_permutations = np.zeros((num_permutations, 3))
    tilde_delta_omega_clnk_norm_permutations = np.zeros(num_permutations)
    tilde_delta_y_clnk_term = (
        b * eta * (np.sqrt(n_b)-np.sqrt(n_a)) / (2.*np.sqrt(3.))
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

def inext_gaussian_fjc_delta_clnk_regular_tetrahedral_4_chn_clnk_free_rot_approx(
        eval_W_clnk_chns: bool,
        eval_W_clnk_y_flucts: bool,
        F: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float,
        X_clnk: npt.NDArray[np.floating],
        w_c_func,
        w_c_args: tuple[float] | tuple[None],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float] | tuple[None],
        d2w_c__dy_clnk_dy_clnk_func,
        d2w_c__dy_clnk_dy_clnk_args: tuple[float] | tuple[None]):
    """Regular tetrahedral 4-chain inextensible Gaussian FJC cross-link
    structure RVE mechanical response in the free rotation limit, as
    evaluated via closed-form approximation.

    This function determines the mechanical response of a regular
    tetrahedral 4-chain inextensible Gaussian FJC cross-link structure
    RVE in the free rotation limit, as evaluated via closed-form
    approximation.

    Args:
        eval_W_clnk_chns (bool): Boolean indicating if the nondimensional cross-link polymer chain free energy ought to be calculated (if True) or not (if False).
        eval_W_clnk_y_flucts (bool): Boolean indicating if the nondimensional cross-link junction fluctuation free energy ought to be calculated (if True) or not (if False).
        F (npt.NDArray[np.floating]): Deformation gradient.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        w_c_func (function): Nondimensional polymer chain free energy function.
        w_c_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        w_c_dfrmtn_func (function): Nondimensional polymer chain deformation free energy function.
        w_c_dfrmtn_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        d2w_c__dy_clnk_dy_clnk_func (function): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function.
        d2w_c__dy_clnk_dy_clnk_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n).
    
    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, float, float]:
        Chain segment number permutation associated with the
        minimal/optimal nondimensional cross-link chain free energy,
        optimal Rodrigues vector perturbation describing the cross-link
        rotation, norm of the optimal Rodrigues vector perturbation
        describing the cross-link rotation, optimal cross-link junction
        position perturbation, distance between the origin and the
        optimal cross-link junction position perturbation,
        absolute/equilibrium chain stretch for each chain in the
        cross-link, nondimensional cross-link polymer chain free energy,
        nondimensional cross-link junction fluctuation free energy,
        nondimensional cross-link free energy.
    
    """
    # Boilerplate initialization, checks, and assertions
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
    
    # Assess the nature of the chain segment number polydispersity in
    # the regular tetrahedral 4-chain cross-link structure
    unique_n_clnk, unique_n_clnk_counts = np.unique(n_clnk, return_counts=True)
    
    # The chain segment numbers in the regular tetrahedral 4-chain
    # cross-link structure are at least trimodal
    if np.shape(unique_n_clnk)[0] >= 3:
        # Gather the cross-link perturbations for all cross-link
        # permutations
        (n_clnk_permutations, tilde_delta_omega_clnk_permutations,
         tilde_delta_omega_clnk_norm_permutations,
         tilde_delta_y_clnk_permutations, tilde_delta_y_clnk_norm_permutations) = (
            inext_gaussian_fjc_tilde_delta_clnk_regular_tetrahedral_4_chn_clnk_free_rot_general_approx(
                Lmbda, n_clnk, b)
        )
    # The chain segment numbers in the regular tetrahedral 4-chain
    # cross-link structure are bimodal
    elif np.shape(unique_n_clnk)[0] == 2:
        # Bimodal 1-3 regular tetrahedral 4-chain cross-link structure
        if unique_n_clnk_counts[0] == 1 or unique_n_clnk_counts[0] == 3:
            # Gather the cross-link perturbations for all cross-link
            # permutations
            (n_clnk_permutations, tilde_delta_omega_clnk_permutations,
             tilde_delta_omega_clnk_norm_permutations,
             tilde_delta_y_clnk_permutations,
             tilde_delta_y_clnk_norm_permutations) = (
                inext_gaussian_fjc_tilde_delta_clnk_1_3_bimodal_regular_tetrahedral_4_chn_clnk_free_rot_approx(
                    Lmbda, n_clnk, b)
            )
        # Bimodal 2-2 regular tetrahedral 4-chain cross-link structure
        else:
            # Gather the cross-link perturbations for all cross-link
            # permutations
            (n_clnk_permutations, tilde_delta_omega_clnk_permutations,
             tilde_delta_omega_clnk_norm_permutations,
             tilde_delta_y_clnk_permutations,
             tilde_delta_y_clnk_norm_permutations) = (
                inext_gaussian_fjc_tilde_delta_clnk_2_2_bimodal_regular_tetrahedral_4_chn_clnk_free_rot_approx(
                    Lmbda, n_clnk, b)
            )
    
    # Calculate nondimensional cross-link chain free energy for all
    # cross-link permutations
    num_permutations = np.shape(n_clnk_permutations)[0]
    W_clnk_chns_permutations = np.zeros(num_permutations)
    for prmttn in range(num_permutations):
        n_clnk_prmttn = n_clnk_permutations[prmttn]
        gamma_clnk_prmttn = gamma_approx_clnk_func(
            F_Lmbda, X_clnk, Q_clnk_m, y_clnk_m,
            Q_axis_angle(tilde_delta_omega_clnk_permutations[prmttn]),
            tilde_delta_y_clnk_permutations[prmttn], n_clnk_prmttn, b)
        W_clnk_chns_permutations[prmttn] = W_clnk_chns_func(
            gamma_clnk_prmttn, n_clnk_prmttn, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    
    # Extract the cross-link permutation associated with the
    # minimal/optimal nondimensional cross-link chain free energy
    W_clnk_chns_min_permutations_indx = np.argmin(W_clnk_chns_permutations)
    n_clnk_permutation = n_clnk_permutations[W_clnk_chns_min_permutations_indx]
    delta_omega_clnk_permutation = (
        tilde_delta_omega_clnk_permutations[W_clnk_chns_min_permutations_indx]
    )
    delta_omega_clnk_norm_permutation = (
        tilde_delta_omega_clnk_norm_permutations[W_clnk_chns_min_permutations_indx]
    )
    delta_y_clnk_permutation = (
        tilde_delta_y_clnk_permutations[W_clnk_chns_min_permutations_indx]
    )
    delta_y_clnk_norm_permutation = (
        tilde_delta_y_clnk_norm_permutations[W_clnk_chns_min_permutations_indx]
    )
    delta_Q_clnk_permutation = Q_axis_angle(delta_omega_clnk_permutation)
    
    # Calculate the absolute/equilibrium chain stretch for each chain
    gamma_clnk_permutation = gamma_approx_clnk_func(
        F_Lmbda, X_clnk, Q_clnk_m, y_clnk_m, delta_Q_clnk_permutation,
        delta_y_clnk_permutation, n_clnk_permutation, b)
    
    # If called for, calculate each component of the nondimensional
    # cross-link free energy
    W_clnk_chns_permutation = 0.
    W_clnk_y_flucts_permutation = 0.
    if eval_W_clnk_chns:
        W_clnk_chns_permutation = W_clnk_chns_func(
            gamma_clnk_permutation, n_clnk_permutation, w_c_func, w_c_args,
            w_c_dfrmtn_func, w_c_dfrmtn_args)
    if eval_W_clnk_y_flucts:
        # Calculate the absolute/equilibrium chain stretch vector for
        # each chain
        gamma_vec_clnk_permutation = gamma_approx_vec_clnk_func(
            F_Lmbda, X_clnk, Q_clnk_m, y_clnk_m, delta_Q_clnk_permutation,
            delta_y_clnk_permutation, n_clnk_permutation, b)
        W_clnk_y_flucts_permutation = W_clnk_y_flucts_func(
            gamma_vec_clnk_permutation, gamma_clnk_permutation, n_clnk,
            d2w_c__dy_clnk_dy_clnk_func, d2w_c__dy_clnk_dy_clnk_args)

    # Calculate the nondimensional cross-link free energy
    W_clnk_permutation = W_clnk_chns_permutation + W_clnk_y_flucts_permutation

    return (
        n_clnk_permutation, delta_omega_clnk_permutation,
        delta_omega_clnk_norm_permutation, delta_y_clnk_permutation,
        delta_y_clnk_norm_permutation, gamma_clnk_permutation,
        W_clnk_chns_permutation, W_clnk_y_flucts_permutation, W_clnk_permutation
    )

def clnk_free_rot_approx(
        eval_W_clnk_chns: bool,
        eval_W_clnk_y_flucts: bool,
        use_inext_gaussian_fjc_delta_clnk: bool,
        F: npt.NDArray[np.floating],
        n_clnk: npt.NDArray[np.floating | np.integer],
        b: float,
        X_clnk: npt.NDArray[np.floating],
        w_c_func,
        w_c_args: tuple[float] | tuple[None],
        w_c_dfrmtn_func,
        w_c_dfrmtn_args: tuple[float] | tuple[None],
        d2w_c__dy_clnk_dy_clnk_func,
        d2w_c__dy_clnk_dy_clnk_args: tuple[float] | tuple[None]):
    """Cross-link structure RVE mechanical response in the free rotation
    limit, as evaluated via closed-form approximation.

    This function determines the mechanical response of a cross-link
    structure RVE in the free rotation limit, as evaluated via
    closed-form approximation.

    Args:
        eval_W_clnk_chns (bool): Boolean indicating if the nondimensional cross-link polymer chain free energy ought to be calculated (if True) or not (if False).
        eval_W_clnk_y_flucts (bool): Boolean indicating if the nondimensional cross-link junction fluctuation free energy ought to be calculated (if True) or not (if False).
        use_inext_gaussian_fjc_delta_clnk (bool): Boolean indicating if the inextensible Gaussian FJC model ought to be used to calculate the optimal cross-link junction position perturbation.
        F (npt.NDArray[np.floating]): Deformation gradient.
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
        b (float): Chain segment and/or cross-linker diameter.
        X_clnk (npt.NDArray[np.floating]): Initial chain end position for each chain in the cross-link structure RVE.
        w_c_func (function): Nondimensional polymer chain free energy function.
        w_c_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        w_c_dfrmtn_func (function): Nondimensional polymer chain deformation free energy function.
        w_c_dfrmtn_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional polymer chain deformation free energy function (beyond the absolute/equilibrium chain stretch gamma and the number of chain segments n).
        d2w_c__dy_clnk_dy_clnk_func (function): Nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function.
        d2w_c__dy_clnk_dy_clnk_args (tuple[float] | tuple[None]): The arguments needed for the nondimensional second derivative of the polymer chain free energy with respect to the cross-link junction position function (beyond the absolute/equilibrium chain stretch vector gamma_vec, the absolute/equilibrium chain stretch gamma, and the number of chain segments n).
    
    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, npt.NDArray[np.floating], float, float, float]:
        Chain segment number permutation associated with the
        minimal/optimal nondimensional cross-link chain free energy,
        optimal Rodrigues vector perturbation describing the cross-link
        rotation, norm of the optimal Rodrigues vector perturbation
        describing the cross-link rotation, optimal cross-link junction
        position perturbation, distance between the origin and the
        optimal cross-link junction position perturbation,
        absolute/equilibrium chain stretch for each chain in the
        cross-link, nondimensional cross-link polymer chain free energy,
        nondimensional cross-link junction fluctuation free energy,
        nondimensional cross-link free energy.
    
    """
    # Boilerplate initialization, checks, and assertions
    k_num = np.shape(n_clnk)[0]
    if (not np.isclose(np.linalg.det(F), 1.) or 
        np.allclose(n_clnk, n_clnk[0]*np.ones_like(n_clnk)) or k_num != 4):
        error_str = (
            "This methodology is only applicable to polydisperse "
            + "regular tetrahedral 4-chain cross-link RVEs under "
            + "incompressible deformation. Make sure that the "
            + "cross-link structure corresponds to the aforementioned "
            + "polydisperse cross-link structure, and the deformation "
            + "is incompressible."
        )
        raise ValueError(error_str)
    if k_num == 4:
        if not np.allclose(x_hat_clnk_func(X_clnk), regular_tetrahedral_4_chn_clnk_X_hat_clnk_func()):
            error_str = (
                "The 4-chain cross-link RVE must be a polydisperse regular "
                + "tetrahedral 4-chain cross-link RVE. Make sure that the "
                + "cross-link structure corresponds to this."
            )
            raise ValueError(error_str)
    if not use_inext_gaussian_fjc_delta_clnk:
        error_str = (
            "This approximated perturbed deformation response of the "
            + "polydisperse regular tetrahedral 4-chain cross-link RVE "
            + "is specifically formulated for the case of the "
            + "inextensible Gaussian FJC."
        )
        raise ValueError(error_str)
   
    if k_num == 4:
        return (
            inext_gaussian_fjc_delta_clnk_regular_tetrahedral_4_chn_clnk_free_rot_approx(
                eval_W_clnk_chns, eval_W_clnk_y_flucts, F, n_clnk, b, X_clnk,
                w_c_func, w_c_args, w_c_dfrmtn_func, w_c_dfrmtn_args,
                d2w_c__dy_clnk_dy_clnk_func, d2w_c__dy_clnk_dy_clnk_args)
        )