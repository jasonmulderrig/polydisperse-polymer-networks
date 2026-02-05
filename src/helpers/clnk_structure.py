import numpy as np
import numpy.typing as npt
from scipy.spatial import ConvexHull

def n_clnk_mean_func(n_clnk: npt.NDArray[np.floating | np.integer]) -> float:
    """Mean cross-link chain segment number.

    This function returns the Mean cross-link chain segment number.

    Args:
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
    
    Returns:
        float: Mean cross-link chain segment number.
    
    """
    return np.mean(n_clnk)

def n_clnk_geo_mean_func(n_clnk: npt.NDArray[np.floating | np.integer]) -> float:
    """Geometric mean cross-link chain segment number.

    This function returns the geometric mean cross-link chain segment
    number.

    Args:
        n_clnk (npt.NDArray[np.floating | np.integer]): Number of chain segments for each chain in the cross-link structure RVE.
    
    Returns:
        float: Geometric mean cross-link chain segment number.
    
    """
    return np.power(np.prod(n_clnk), 1./np.shape(n_clnk)[0])

def centroid_x_clnk_func(
        x_clnk: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Centroid of the cross-link chain ends.

    This function returns the centroid of the cross-link chain ends.

    Args:
        x_clnk (npt.NDArray[np.floating]): Chain end position for each chain in the cross-link structure RVE.
    
    Returns:
        npt.NDArray[np.floating]: Centroid of the cross-link chain ends.
    
    """
    return np.mean(x_clnk, axis=0)

def com_x_clnk_func(
        x_clnk: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Center-of-mass of the cross-link chain ends.

    This function returns the center-of-mass of the cross-link chain
    ends. Here, it is assumed that the mass of each chain end is equal
    to the mass of all other chain ends. Given this assumption, the
    center-of-mass of the cross-link chain ends is equivalent to the
    centroid of the cross-link chain ends.

    Args:
        x_clnk (npt.NDArray[np.floating]): Chain end position for each chain in the cross-link structure RVE.
    
    Returns:
        npt.NDArray[np.floating]: Center-of-mass of the cross-link chain
        ends.
    
    """
    return centroid_x_clnk_func(x_clnk)

def x_clnk_jog_min_max_3_chn_clnk_func(
        x_clnk: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Jogged minimum and maximum components of the chain ends in a
    3-chain cross-link structure RVE.

    This function deduces and returns the jogged minimum and maximum
    components of the chain ends in a 3-chain cross-link structure RVE.
    The jogging of the minimum and maximum components of the chain ends
    in a 3-chain cross-link structure RVE only takes place if necessary,
    i.e., if the provided 3-chain cross-link structure RVE resides in an
    x-, y-, or z-plane.

    Args:
        x_clnk (npt.NDArray[np.floating]): Chain end position for each chain in the 3-chain cross-link structure RVE.
    
    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Jogged minimum and maximum components of the chain ends in a
        3-chain cross-link structure RVE.
    
    """
    if np.shape(x_clnk)[0] == 3:
        x_clnk_min = np.min(x_clnk, axis=0)
        x_clnk_max = np.max(x_clnk, axis=0)
        jog = 0.01 * np.max(np.abs(x_clnk)) # Inspired by Qhull
        if jog < 1.e-5: jog = 1.e-5 # Jog tolerance
        if np.isclose(x_clnk_min[0], x_clnk_max[0]):
            x_clnk_min[0] -= jog
            x_clnk_max[0] += jog
        elif np.isclose(x_clnk_min[1], x_clnk_max[1]):
            x_clnk_min[1] -= jog
            x_clnk_max[1] += jog
        elif np.isclose(x_clnk_min[2], x_clnk_max[2]):
            x_clnk_min[2] -= jog
            x_clnk_max[2] += jog
        return x_clnk_min, x_clnk_max
    else:
        error_str = (
            "This function is only applicable for 3-chain cross-link "
            + "RVEs."
        )
        raise ValueError(error_str)

def x_clnk_min_max_func(
        x_clnk: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Minimum and maximum components of the chain ends in a cross-link
    structure RVE.

    This function deduces and returns the minimum and maximum components
    of the chain ends in a cross-link structure RVE.

    Args:
        x_clnk (npt.NDArray[np.floating]): Chain end position for each chain in the cross-link structure RVE.
    
    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Minimum and maximum components of the chain ends in a cross-link
        structure RVE.
    
    """
    if np.shape(x_clnk)[0] == 3:
        return x_clnk_jog_min_max_3_chn_clnk_func(x_clnk)
    else: return np.min(x_clnk, axis=0), np.max(x_clnk, axis=0)

def x_clnk_3_chn_clnk_func(
        x_clnk: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Vertices of the rectangular prism bounding the chain ends in a
    3-chain cross-link structure RVE.

    This function deduces and returns the vertices of the rectangular
    prism bounding the chain ends in a 3-chain cross-link structure RVE.

    Args:
        x_clnk (npt.NDArray[np.floating]): Chain end position for each chain in the 3-chain cross-link structure RVE.
    
    Returns:
        npt.NDArray[np.floating]: Vertices of the rectangular prism
        bounding the chain ends in a 3-chain cross-link structure RVE.
    
    """
    if np.shape(x_clnk)[0] == 3:
        x_clnk_min, x_clnk_max = x_clnk_jog_min_max_3_chn_clnk_func(x_clnk)
        x_min, y_min, z_min = x_clnk_min
        x_max, y_max, z_max = x_clnk_max
        return (
            np.asarray(
                [
                    [x_max, y_max, z_max],
                    [x_max, y_max, z_min],
                    [x_max, y_min, z_max],
                    [x_max, y_min, z_min],
                    [x_min, y_max, z_max],
                    [x_min, y_max, z_min],
                    [x_min, y_min, z_max],
                    [x_min, y_min, z_min]
                ])
        )
    else:
        error_str = (
            "This function is only applicable for 3-chain cross-link "
            + "RVEs."
        )
        raise ValueError(error_str)

def chull_eqs_clnk_func(
        x_clnk: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Convex hull of the cross-link structure.

    This function extracts and returns the convex hull of the cross-link
    structure via using the scipy.spatial.ConvexHull() function.

    Args:
        x_clnk (npt.NDArray[np.floating]): Chain end position for each chain in the cross-link structure RVE.
    
    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating]]: ``A''
        matrix (2D array) and ``b'' vector (1D array) of coefficients
        involved in the equations that define the planes that altogether
        define the convex hull of the cross-link structure.
    
    """
    # If necessary, determine the rectangular prism that subsumes the
    # 3-chain cross-link structure
    if np.shape(x_clnk)[0] == 3: x_clnk = x_clnk_3_chn_clnk_func(x_clnk)
    
    # Extract the convex hull of the cross-link structure
    chull_eqs_clnk = np.unique(ConvexHull(x_clnk).equations, axis=0)
    return chull_eqs_clnk[:, :-1], -chull_eqs_clnk[:, -1]

def x_hat_clnk_func(
        x_clnk: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Unit chain end position for each chain in the cross-link
    structure RVE.

    This function returns the unit chain end position for each chain in
    the cross-link structure RVE.

    Args:
        x_clnk (npt.NDArray[np.floating]): Chain end position for each chain in the cross-link structure RVE.
    
    Returns:
        npt.NDArray[np.floating]: Unit chain end position for each chain
        in the cross-link structure RVE.
    
    """
    x_hat_clnk = np.empty_like(x_clnk)
    for chn_indx in range(np.shape(x_clnk)[0]):
        x_hat_clnk[chn_indx] = (
            x_clnk[chn_indx] / np.linalg.norm(x_clnk[chn_indx])
        )
    return x_hat_clnk

def classical_3_chn_clnk_X_hat_clnk_func() -> npt.NDArray[np.floating]:
    """Initial unit chain end position for each chain in the classical
    3-chain cross-link structure RVE.
    
    This function returns the initial unit chain end position for each
    chain in the classical 3-chain cross-link structure RVE.

    Returns:
        npt.NDArray[np.floating]: Initial unit chain end position for
        each chain in the classical 3-chain cross-link structure RVE.
    
    """
    return np.eye(3)

def amended_3_chn_clnk_X_hat_clnk_func() -> npt.NDArray[np.floating]:
    """Initial unit chain end position for each chain in the amended
    3-chain cross-link structure RVE.
    
    This function returns the initial unit chain end position for each
    chain in the amended 3-chain cross-link structure RVE.

    Returns:
        npt.NDArray[np.floating]: Initial unit chain end position for
        each chain in the amended 3-chain cross-link structure RVE.
    
    """
    X_hat_clnk = -np.sqrt(1./6.) * np.ones((3, 3))
    np.fill_diagonal(X_hat_clnk, np.sqrt(2./3.))
    return X_hat_clnk

def regular_tetrahedral_4_chn_clnk_X_hat_clnk_func() -> npt.NDArray[np.floating]:
    """Initial unit chain end position for each chain in the classical
    regular tetrahedral 4-chain cross-link structure RVE.
    
    This function returns the initial unit chain end position for each
    chain in the classical regular tetrahedral 4-chain cross-link
    structure RVE.

    Returns:
        npt.NDArray[np.floating]: Initial unit chain end position for
        each chain in the classical regular tetrahedral 4-chain
        cross-link structure RVE.
    
    """
    return (
        np.asarray(
            [
                [0., 0., 1.],
                [0., 2.*np.sqrt(2.)/3., -1./3.],
                [np.sqrt(2./3.), -np.sqrt(2.)/3., -1./3.],
                [-np.sqrt(2./3.), -np.sqrt(2.)/3., -1./3.]
            ])
    )

def equilateral_triangular_bipyramidal_5_chn_clnk_X_hat_clnk_func() -> npt.NDArray[np.floating]:
    """Initial unit chain end position for each chain in the equilateral
    triangular bipyramidal 5-chain cross-link structure RVE.
    
    This function returns the initial unit chain end position for each
    chain in the equilateral triangular bipyramidal 5-chain cross-link
    structure RVE.

    Returns:
        npt.NDArray[np.floating]: Initial unit chain end position for
        each chain in the equilateral triangular bipyramidal 5-chain
        cross-link structure RVE.
    
    """
    return (
        np.asarray(
            [
                [0., 0., 1.],
                [1., 0., 0.],
                [-1./2., np.sqrt(3.)/2., 0.],
                [-1./2., -np.sqrt(3.)/2., 0.],
                [0., 0., -1.]
            ])
    )

def regular_octahedral_6_chn_clnk_X_hat_clnk_func() -> npt.NDArray[np.floating]:
    """Initial unit chain end position for each chain in the classical
    regular octahedral 6-chain cross-link structure RVE.
    
    This function returns the initial unit chain end position for each
    chain in the classical regular octahedral 6-chain cross-link
    structure RVE.

    Returns:
        npt.NDArray[np.floating]: Initial unit chain end position for
        each chain in the classical regular octahedral 6-chain
        cross-link structure RVE.
    
    """
    e_hat = np.eye(3)
    X_hat_clnk = np.empty((6, 3))
    for chn_indx in range(6):
        if chn_indx < 3: X_hat_clnk[chn_indx] = e_hat[chn_indx]
        else: X_hat_clnk[chn_indx] = -e_hat[chn_indx-3]
    return X_hat_clnk

def equilateral_pentagonal_bipyramidal_7_chn_clnk_X_hat_clnk_func() -> npt.NDArray[np.floating]:
    """Initial unit chain end position for each chain in the equilateral
    pentagonal bipyramidal 7-chain cross-link structure RVE.
    
    This function returns the initial unit chain end position for each
    chain in the equilateral pentagonal bipyramidal 7-chain cross-link
    structure RVE.

    Returns:
        npt.NDArray[np.floating]: Initial unit chain end position for
        each chain in the equilateral pentagonal bipyramidal 7-chain
        cross-link structure RVE.
    
    """
    return (
        np.asarray(
            [
                [0., 0., 1.],
                [1., 0., 0.],
                [(np.sqrt(5.)-1.)/4., np.sqrt(np.sqrt(5.)/8.+5./8.), 0.],
                [(-np.sqrt(5.)-1.)/4., np.sqrt(-np.sqrt(5.)/8.+5./8.), 0.],
                [(-np.sqrt(5.)-1.)/4., -np.sqrt(-np.sqrt(5.)/8.+5./8.), 0.],
                [(np.sqrt(5.)-1.)/4., -np.sqrt(np.sqrt(5.)/8.+5./8.), 0.],
                [0., 0., -1.]
            ])
    )

def cube_8_chn_clnk_X_hat_clnk_func() -> npt.NDArray[np.floating]:
    """Initial unit chain end position for each chain in the classical
    cube 8-chain cross-link structure RVE.
    
    This function returns the initial unit chain end position for each
    chain in the classical cube 8-chain cross-link structure RVE.

    Returns:
        npt.NDArray[np.floating]: Initial unit chain end position for
        each chain in the classical cube 8-chain cross-link structure
        RVE.
    
    """
    return (
        1. / np.sqrt(3.) * (1-2*np.transpose(np.indices((2,)*3).reshape(3, -1)))
    )

def square_antiprism_8_chn_clnk_X_hat_clnk_func() -> npt.NDArray[np.floating]:
    """Initial unit chain end position for each chain in the square
    anti-prism 8-chain cross-link structure RVE.
    
    This function returns the initial unit chain end position for each
    chain in the square anti-prism 8-chain cross-link structure RVE.

    Returns:
        npt.NDArray[np.floating]: Initial unit chain end position for
        each chain in the square anti-prism 8-chain cross-link structure
        RVE.
    
    """
    X_hat_clnk_cube = cube_8_chn_clnk_X_hat_clnk_func()
    X_hat_clnk_half_cube = X_hat_clnk_cube[:4]
    X_hat_clnk_half_square_antiprism = np.asarray(
        [
            [-1., 1., 0.],
            [-1., -1., 0.],
            [-1., 0., 1.],
            [-1., 0., -1.]
        ])
    X_hat_clnk_half_square_antiprism *= 1. / np.sqrt(2.)
    X_hat_clnk = np.vstack(
        (X_hat_clnk_half_cube, X_hat_clnk_half_square_antiprism))
    return X_hat_clnk

def X_clnk_func(
        X_hat_clnk: npt.NDArray[np.floating],
        r: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Initial chain end position for each chain in the cross-link
    structure RVE.

    This function calculates and returns the initial chain end position
    for each chain in the cross-link structure RVE.

    Args:
        X_hat_clnk (npt.NDArray[np.floating]): Initial unit chain end position for each chain in the cross-link structure RVE.
        r (npt.NDArray[np.floating]): Initial end-to-end chain distance/length for each chain in the cross-link structure RVE.
    
    Returns:
        npt.NDArray[np.floating]: Initial chain end position for each
        chain in the cross-link structure RVE.
    
    """
    if np.shape(X_hat_clnk)[0] != np.shape(r)[0]:
        error_str = (
            "The number of chains in the cross-link RVE must match "
            + "between X_hat_clnk and r."
        )
        raise ValueError(error_str)
    return X_hat_clnk * r[:, np.newaxis]

def omega_clnk_init_func() -> npt.NDArray[np.floating]:
    """Initial Rodrigues vector describing the initial rotation of the
    cross-link structure RVE, i.e., the zero Rodrigues vector.

    This function supplies the initial Rodrigues vector describing the
    initial rotation of the cross-link structure RVE, i.e., the zero
    Rodrigues vector.

    Returns:
        npt.ArrayLike: Initial Rodrigues vector describing the initial
        rotation of the cross-link structure RVE, i.e., the zero
        Rodrigues vector.
    
    """
    return np.zeros(3)

def y_clnk_init_func() -> npt.NDArray[np.floating]:
    """Initial cross-link junction position for the cross-link structure
    RVE, i.e., the origin.

    This function supplies the initial cross-link junction position for
    the cross-link structure RVE, i.e., the origin.

    Returns:
        npt.ArrayLike: Initial cross-link junction position for the
        cross-link structure RVE, i.e., the origin.
    
    """
    return np.zeros(3)

def classical_3_chn_clnk_init_func(
        r: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Classical 3-chain cross-link structure RVE initialization.
    
    This function initializes the classical 3-chain cross-link structure
    RVE.

    Args:
        r (npt.NDArray[np.floating]): Initial end-to-end chain distance/length for each chain in the classical 3-chain cross-link structure RVE.

    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Initial chain end position for each chain in the classical
        3-chain cross-link structure RVE, initial Rodrigues vector
        describing the initial rotation (i.e., the zero Rodrigues
        vector), and initial cross-link junction position (i.e., the
        origin).
    
    """
    return (
        X_clnk_func(classical_3_chn_clnk_X_hat_clnk_func(), r),
        omega_clnk_init_func(), y_clnk_init_func()
    )

def amended_3_chn_clnk_init_func(
        r: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Amended 3-chain cross-link structure RVE initialization.
    
    This function initializes the amended 3-chain cross-link structure
    RVE.

    Args:
        r (npt.NDArray[np.floating]): Initial end-to-end chain distance/length for each chain in the amended 3-chain cross-link structure RVE.

    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Initial chain end position for each chain in the amended 3-chain
        cross-link structure RVE, initial Rodrigues vector describing
        the initial rotation (i.e., the zero Rodrigues vector), and
        initial cross-link junction position (i.e., the origin).
    
    """
    return (
        X_clnk_func(amended_3_chn_clnk_X_hat_clnk_func(), r),
        omega_clnk_init_func(), y_clnk_init_func()
    )

def regular_tetrahedral_4_chn_clnk_init_func(
        r: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Classical regular tetrahedral 4-chain cross-link structure RVE
    initialization.
    
    This function initializes the classical regular tetrahedral 4-chain
    cross-link structure RVE.

    Args:
        r (npt.NDArray[np.floating]): Initial end-to-end chain distance/length for each chain in the classical regular tetrahedral 4-chain cross-link structure RVE.

    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Initial chain end position for each chain in the classical
        regular tetrahedral 4-chain cross-link structure RVE, initial
        Rodrigues vector describing the initial rotation (i.e., the zero
        Rodrigues vector), and initial cross-link junction position
        (i.e., the origin).
    
    """
    return (
        X_clnk_func(regular_tetrahedral_4_chn_clnk_X_hat_clnk_func(), r),
        omega_clnk_init_func(), y_clnk_init_func()
    )

def equilateral_triangular_bipyramidal_5_chn_clnk_init_func(
        r: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Equilateral triangular bipyramidal 5-chain cross-link structure
    RVE initialization.
    
    This function initializes the equilateral triangular bipyramidal
    5-chain cross-link structure RVE.

    Args:
        r (npt.NDArray[np.floating]): Initial end-to-end chain distance/length for each chain in the equilateral triangular bipyramidal 5-chain cross-link structure RVE.

    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Initial chain end position for each chain in the equilateral
        triangular bipyramidal 5-chain cross-link structure RVE, initial
        Rodrigues vector describing the initial rotation (i.e., the zero
        Rodrigues vector), and initial cross-link junction position
        (i.e., the origin).
    
    """
    return (
        X_clnk_func(equilateral_triangular_bipyramidal_5_chn_clnk_X_hat_clnk_func(), r),
        omega_clnk_init_func(), y_clnk_init_func()
    )

def regular_octahedral_6_chn_clnk_init_func(
        r: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Classical regular octahedral 6-chain cross-link structure RVE
    initialization.
    
    This function initializes the classical regular octahedral 6-chain
    cross-link structure RVE.

    Args:
        r (npt.NDArray[np.floating]): Initial end-to-end chain distance/length for each chain in the classical regular octahedral 6-chain cross-link structure RVE.

    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Initial chain end position for each chain in the classical
        regular octahedral 6-chain cross-link structure RVE, initial
        Rodrigues vector describing the initial rotation (i.e., the zero
        Rodrigues vector), and initial cross-link junction position
        (i.e., the origin).
    
    """
    return (
        X_clnk_func(regular_octahedral_6_chn_clnk_X_hat_clnk_func(), r),
        omega_clnk_init_func(), y_clnk_init_func()
    )

def equilateral_pentagonal_bipyramidal_7_chn_clnk_init_func(
        r: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Equilateral pentagonal bipyramidal 7-chain cross-link structure
    RVE initialization.
    
    This function initializes the equilateral pentagonal bipyramidal
    7-chain cross-link structure RVE.

    Args:
        r (npt.NDArray[np.floating]): Initial end-to-end chain distance/length for each chain in the equilateral pentagonal bipyramidal 7-chain cross-link structure RVE.

    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Initial chain end position for each chain in the equilateral
        pentagonal bipyramidal 7-chain cross-link structure RVE, initial
        Rodrigues vector describing the initial rotation (i.e., the zero
        Rodrigues vector), and initial cross-link junction position
        (i.e., the origin).
    
    """
    return (
        X_clnk_func(equilateral_pentagonal_bipyramidal_7_chn_clnk_X_hat_clnk_func(), r),
        omega_clnk_init_func(), y_clnk_init_func()
    )

def cube_8_chn_clnk_init_func(
        r: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Classical cube 8-chain cross-link structure RVE initialization.
    
    This function initializes the classical cube 8-chain cross-link
    structure RVE.

    Args:
        r (npt.NDArray[np.floating]): Initial end-to-end chain distance/length for each chain in the classical cube 8-chain cross-link structure RVE.

    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Initial chain end position for each chain in the classical cube
        8-chain cross-link structure RVE, initial Rodrigues vector
        describing the initial rotation (i.e., the zero Rodrigues
        vector), and initial cross-link junction position (i.e., the
        origin).
    
    """
    return (
        X_clnk_func(cube_8_chn_clnk_X_hat_clnk_func(), r),
        omega_clnk_init_func(), y_clnk_init_func()
    )

def square_antiprism_8_chn_clnk_init_func(
        r: npt.NDArray[np.floating]) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Square anti-prism 8-chain cross-link structure RVE initialization.
    
    This function initializes the square anti-prism 8-chain cross-link
    structure RVE.

    Args:
        r (npt.NDArray[np.floating]): Initial end-to-end chain distance/length for each chain in the square anti-prism 8-chain cross-link structure RVE.

    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Initial chain end position for each chain in the square
        anti-prism 8-chain cross-link structure RVE, initial Rodrigues
        vector describing the initial rotation (i.e., the zero Rodrigues
        vector), and initial cross-link junction position (i.e., the
        origin).
    
    """
    return (
        X_clnk_func(square_antiprism_8_chn_clnk_X_hat_clnk_func(), r),
        omega_clnk_init_func(), y_clnk_init_func()
    )

def recommended_clnk_init_func(
        r: npt.NDArray[np.floating],
        type_8_chn_clnk: str) -> tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
    """Recommended cross-link structure RVE initialization.
    
    This function initializes the recommended cross-link structure RVE,
    as called for by the number of chains present in the vector of
    initial end-to-end chain distance/length.

    Args:
        r (npt.NDArray[np.floating]): Initial end-to-end chain distance/length for each chain in the cross-link structure RVE.
        type_8_chn_clnk (str): String indicating which 8-chain cross-link structure RVE to initialize; either the classical cube ("cube") or the square anti-prism ("square_antiprism") 8-chain cross-link structure RVEs are accepted.

    Returns:
        tuple[npt.NDArray[np.floating], npt.NDArray[np.floating], npt.NDArray[np.floating]]:
        Initial chain end position for each chain in the recommended
        cross-link structure RVE, initial Rodrigues vector describing
        the initial rotation (i.e., the zero Rodrigues vector), and
        initial cross-link junction position (i.e., the origin).
    
    """
    k = np.shape(r)[0]
    if k == 3: return amended_3_chn_clnk_init_func(r)
    elif k == 4: return regular_tetrahedral_4_chn_clnk_init_func(r)
    elif k == 5:
        return equilateral_triangular_bipyramidal_5_chn_clnk_init_func(r)
    elif k == 6: return regular_octahedral_6_chn_clnk_init_func(r)
    elif k == 7:
        return equilateral_pentagonal_bipyramidal_7_chn_clnk_init_func(r)
    elif k == 8:
        if type_8_chn_clnk == "cube": return cube_8_chn_clnk_init_func(r)
        elif type_8_chn_clnk == "square_antiprism":
            return square_antiprism_8_chn_clnk_init_func(r)

def recommended_X_hat_clnks_func(
        type_8_chn_clnk: str) -> list[npt.NDArray[np.floating]]:
    """Recommended unit cross-link structure RVEs initialization.
    
    This function initializes the recommended 3-chain, 4-chain, ...,
    8-chain unit cross-link structure RVEs.

    Args:
        type_8_chn_clnk (str): String indicating which 8-chain unit cross-link structure RVE to initialize; either the classical cube ("cube") or the square anti-prism ("square_antiprism") 8-chain cross-link structure RVEs are accepted.

    Returns:
        list[npt.NDArray[np.floating]]:
        Initial unit chain end position for each chain in each of the
        recommended 3-chain, 4-chain, ..., 8-chain unit cross-link
        structure RVEs.
    
    """
    recommended_X_hat_clnks = []
    for k in range(3, 9):
        if k == 3:
            recommended_X_hat_clnks.append(amended_3_chn_clnk_X_hat_clnk_func())
        elif k == 4:
            recommended_X_hat_clnks.append(
                regular_tetrahedral_4_chn_clnk_X_hat_clnk_func())
        elif k == 5:
            recommended_X_hat_clnks.append(
                equilateral_triangular_bipyramidal_5_chn_clnk_X_hat_clnk_func())
        elif k == 6:
            recommended_X_hat_clnks.append(
                regular_octahedral_6_chn_clnk_X_hat_clnk_func())
        elif k == 7:
            recommended_X_hat_clnks.append(
                equilateral_pentagonal_bipyramidal_7_chn_clnk_X_hat_clnk_func())
        elif k == 8:
            if type_8_chn_clnk == "cube":
                recommended_X_hat_clnks.append(cube_8_chn_clnk_X_hat_clnk_func())
            elif type_8_chn_clnk == "square_antiprism":
                recommended_X_hat_clnks.append(
                    square_antiprism_8_chn_clnk_X_hat_clnk_func())
    return recommended_X_hat_clnks