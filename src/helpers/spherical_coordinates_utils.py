import numpy as np

def cartesian_to_spherical_coords(x_y_z: np.ndarray) -> np.ndarray:
    r_theta_phi = np.empty_like(x_y_z)
    sum_x_sq_y_sq = x_y_z[:, 0]**2 + x_y_z[:, 1]**2
    r_theta_phi[:, 0] = np.sqrt(sum_x_sq_y_sq + x_y_z[:, 2]**2)
    r_theta_phi[:, 1] = np.arctan2(np.sqrt(sum_x_sq_y_sq), x_y_z[:, 2])
    r_theta_phi[:, 2] = np.arctan2(x_y_z[:, 1], x_y_z[:, 0])

    return r_theta_phi

def spherical_to_cartesian_coords(r_theta_phi: np.ndarray) -> np.ndarray:
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