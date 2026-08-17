import numpy as np
import numpy.typing as npt

def cartesian_to_spherical_coords(
        x_y_z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Cartesian-to-spherical coordinates transformation.

    This function transforms Cartesian coordinates to spherical
    coordinates.

    Args:
        x_y_z (npt.NDArray[np.float64]): Cartesian coordinates.
    
    Returns:
        npt.NDArray[np.float64]: Spherical coordinates.
    
    """
    r_theta_phi = np.empty_like(x_y_z)
    sum_x_sq_y_sq = x_y_z[:, 0]**2 + x_y_z[:, 1]**2
    r_theta_phi[:, 0] = np.sqrt(sum_x_sq_y_sq + x_y_z[:, 2]**2)
    r_theta_phi[:, 1] = np.arctan2(np.sqrt(sum_x_sq_y_sq), x_y_z[:, 2])
    r_theta_phi[:, 2] = np.arctan2(x_y_z[:, 1], x_y_z[:, 0])
    return r_theta_phi

def spherical_to_cartesian_coords(
        r_theta_phi: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Spherical-to-Cartesian coordinates transformation.

    This function transforms spherical coordinates to Cartesian
    coordinates.

    Args:
        r_theta_phi (npt.NDArray[np.float64]): Spherical coordinates.
    
    Returns:
        npt.NDArray[np.float64]: Cartesian coordinates.
    
    """
    x_y_z = np.empty_like(r_theta_phi)
    x_y_z[:, 0] = (
        r_theta_phi[:, 0] * np.sin(r_theta_phi[:, 1])
        * np.cos(r_theta_phi[:, 2])
    )
    x_y_z[:, 1] = (
        r_theta_phi[:, 0] * np.sin(r_theta_phi[:, 1])
        * np.sin(r_theta_phi[:, 2])
    )
    x_y_z[:, 2] = r_theta_phi[:, 0] * np.cos(r_theta_phi[:, 1])
    return x_y_z