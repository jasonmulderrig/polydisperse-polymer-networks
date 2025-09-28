import numpy as np
from scipy.spatial import ConvexHull

def com_x_clnk_func(x_clnk: np.ndarray) -> np.ndarray:
    # The center-of-mass of the cross-link chain ends is equal to the
    # centroid of the cross-link chain ends
    return centroid_x_clnk_func(x_clnk)

def centroid_x_clnk_func(x_clnk: np.ndarray) -> np.ndarray:
    return np.mean(x_clnk, axis=0)

def x_clnk_3_chn_clnk_func(x_clnk: np.ndarray) -> np.ndarray:
    if np.shape(x_clnk)[0] == 3:
        x_min, y_min, z_min = np.min(x_clnk, axis=0)
        x_max, y_max, z_max = np.max(x_clnk, axis=0)
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

def chull_eqs_clnk_func(x_clnk: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # If necessary, determine the rectangular prism that subsumes the
    # 3-chain cross-link structure
    if np.shape(x_clnk)[0] == 3: x_clnk = x_clnk_3_chn_clnk_func(x_clnk)
    
    # Cross-link convex hull
    chull_eqs_clnk = np.unique(ConvexHull(x_clnk).equations, axis=0)
    return chull_eqs_clnk[:, :-1], -chull_eqs_clnk[:, -1]

def vol_quad_clnk_func(
        x_clnk: np.ndarray,
        vol_clnk_side_dim_points: int) -> np.ndarray:
    # Determine the rectangular prism that subsumes the 3-chain
    # cross-link structure
    if np.shape(x_clnk)[0] == 3: x_clnk = x_clnk_3_chn_clnk_func(x_clnk)

    # Determine the rectangular prism that subsumes the cross-link
    # structure
    x_min, y_min, z_min = np.min(x_clnk, axis=0)
    x_max, y_max, z_max = np.max(x_clnk, axis=0)
    
    # Initialize the quadrature points about the rectangular prism that
    # subsumes the cross-link structure
    x_vol_quad_clnk = np.linspace(x_min, x_max, vol_clnk_side_dim_points)
    y_vol_quad_clnk = np.linspace(y_min, y_max, vol_clnk_side_dim_points)
    z_vol_quad_clnk = np.linspace(z_min, z_max, vol_clnk_side_dim_points)
    x_coords_vol_quad_clnk, y_coords_vol_quad_clnk, z_coords_vol_quad_clnk = (
        np.meshgrid(
            x_vol_quad_clnk, y_vol_quad_clnk, z_vol_quad_clnk, indexing="ij")
    )
    coords_vol_quad_clnk = np.column_stack(
        (
            x_coords_vol_quad_clnk.ravel(),
            y_coords_vol_quad_clnk.ravel(),
            z_coords_vol_quad_clnk.ravel()
        ))
    # Stack in the weights of the equivolume quadrature points
    num_vol_quad_clnk = np.shape(coords_vol_quad_clnk)[0]
    return (
        np.column_stack(
            (
                coords_vol_quad_clnk,
                np.ones(num_vol_quad_clnk)/num_vol_quad_clnk
            ))
    )

def x_hat_clnk_func(x_clnk: np.ndarray) -> np.ndarray:
    x_hat_clnk = np.empty_like(x_clnk)
    k_num = np.shape(x_clnk)[0]
    for chn_indx in range(k_num):
        x_hat_clnk[chn_indx] = (
            x_clnk[chn_indx] / np.linalg.norm(x_clnk[chn_indx])
        )
    return x_hat_clnk

def classical_3_chn_clnk_X_hat_clnk_func(): return np.eye(3)

def amended_3_chn_clnk_X_hat_clnk_func():
    X_hat_clnk = -np.sqrt(1./6.) * np.ones((3, 3))
    np.fill_diagonal(X_hat_clnk, np.sqrt(2./3.))
    return X_hat_clnk

def regular_tetrahedral_4_chn_clnk_X_hat_clnk_func():
    return (
        np.asarray(
            [
                [0., 0., 1.],
                [0., 2.*np.sqrt(2.)/3., -1./3.],
                [np.sqrt(2./3.), -np.sqrt(2.)/3., -1./3.],
                [-np.sqrt(2./3.), -np.sqrt(2.)/3., -1./3.]
            ])
    )

def equilateral_triangular_bipyramidal_5_chn_clnk_X_hat_clnk_func():
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

def regular_octahedral_6_chn_clnk_X_hat_clnk_func():
    e_hat = np.eye(3)
    X_hat_clnk = np.empty((6, 3))
    for chn_indx in range(6):
        if chn_indx < 3: X_hat_clnk[chn_indx] = e_hat[chn_indx]
        else: X_hat_clnk[chn_indx] = -e_hat[chn_indx-3]
    return X_hat_clnk

def equilateral_pentagonal_bipyramidal_7_chn_clnk_X_hat_clnk_func():
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

def cube_8_chn_clnk_X_hat_clnk_func():
    return (
        1. / np.sqrt(3.) * (1-2*np.transpose(np.indices((2,)*3).reshape(3, -1)))
    )

def square_antiprism_8_chn_clnk_X_hat_clnk_func():
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

def X_clnk_func(X_hat_clnk: np.ndarray, r: np.ndarray) -> np.ndarray:
    if np.shape(X_hat_clnk)[0] != np.shape(r)[0]:
        error_str = (
            "The number of chains in the cross-link RVE must match "
            + "between X_hat_clnk and r."
        )
        raise ValueError(error_str)
    return X_hat_clnk * r[:, None]

def classical_3_chn_clnk_init_func(
        r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        X_clnk_func(classical_3_chn_clnk_X_hat_clnk_func(), r),
        np.zeros(3), np.zeros(3)
    )

def amended_3_chn_clnk_init_func(
        r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        X_clnk_func(amended_3_chn_clnk_X_hat_clnk_func(), r),
        np.zeros(3), np.zeros(3)
    )

def regular_tetrahedral_4_chn_clnk_init_func(
        r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        X_clnk_func(regular_tetrahedral_4_chn_clnk_X_hat_clnk_func(), r),
        np.zeros(3), np.zeros(3)
    )

def equilateral_triangular_bipyramidal_5_chn_clnk_init_func(
        r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        X_clnk_func(equilateral_triangular_bipyramidal_5_chn_clnk_X_hat_clnk_func(), r),
        np.zeros(3), np.zeros(3)
    )

def regular_octahedral_6_chn_clnk_init_func(
        r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        X_clnk_func(regular_octahedral_6_chn_clnk_X_hat_clnk_func(), r),
        np.zeros(3), np.zeros(3)
    )

def equilateral_pentagonal_bipyramidal_7_chn_clnk_init_func(
        r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        X_clnk_func(equilateral_pentagonal_bipyramidal_7_chn_clnk_X_hat_clnk_func(), r),
        np.zeros(3), np.zeros(3)
    )

def cube_8_chn_clnk_init_func(
        r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        X_clnk_func(cube_8_chn_clnk_X_hat_clnk_func(), r), np.zeros(3), np.zeros(3)
    )

def square_antiprism_8_chn_clnk_init_func(
        r: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        X_clnk_func(square_antiprism_8_chn_clnk_X_hat_clnk_func(), r),
        np.zeros(3), np.zeros(3)
    )

def recommended_clnk_init_func(
        r: np.ndarray,
        type_8_chn_clnk: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
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