import numpy as np
import numpy.typing as npt

def A_func(u_hat: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Skew-symmetric tensor representation of the normalized Rodrigues
    vector.

    This function generates the skew-symmetric tensor representation of
    the normalized Rodrigues vector.

    Args:
        u_hat (npt.NDArray[np.floating]): Normalized Rodrigues vector.

    Returns:
        npt.NDArray[np.floating]: Skew-symmetric tensor representation
        of the normalized Rodrigues vector.
    
    """
    u_hat_0, u_hat_1, u_hat_2 = u_hat
    return (
        np.asarray(
            [
                [0., -u_hat_2, u_hat_1],
                [u_hat_2, 0., -u_hat_0],
                [-u_hat_1, u_hat_0, 0.]
            ])
    )

def Q_axis_angle(omega: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """Axis-Angle rotation matrix.

    This function generates the axis-angle rotation matrix from a
    Rodrigues vector.

    Args:
        omega (npt.NDArray[np.floating]): Rodrigues vector.

    Returns:
        npt.NDArray[np.floating]: Axis-Angle rotation matrix.
    
    """
    omega_norm = np.linalg.norm(omega)
    if omega_norm > 0:
        u_hat = omega / omega_norm
        A = A_func(u_hat)
        return (
            np.eye(3) + np.sin(omega_norm) * A
            + (1.-np.cos(omega_norm)) * np.matmul(A, A)
        )
    else: return np.eye(3)

def R_x(gamma: float) -> npt.NDArray[np.floating]:
    """Rotation matrix corresponding to a rotation in the x-plane.

    This function generates the rotation matrix corresponding to a
    rotation in the x-plane by an angle gamma.

    Args:
        gamma (float): Angle of rotation in the x-plane.

    Returns:
        npt.NDArray[np.floating]: Rotation matrix corresponding to a
        rotation in the x-plane.
    
    """
    return (
        np.asarray(
            [
                [1., 0., 0.],
                [0., np.cos(gamma), -np.sin(gamma)],
                [0., np.sin(gamma), np.cos(gamma)]
            ])
    )

def R_y(beta: float) -> npt.NDArray[np.floating]:
    """Rotation matrix corresponding to a rotation in the y-plane.

    This function generates the rotation matrix corresponding to a
    rotation in the y-plane by an angle beta.

    Args:
        beta (float): Angle of rotation in the y-plane.

    Returns:
        npt.NDArray[np.floating]: Rotation matrix corresponding to a
        rotation in the y-plane.
    
    """
    return (
        np.asarray(
            [
                [np.cos(beta), 0., np.sin(beta)],
                [0., 1., 0.],
                [-np.sin(beta), 0., np.cos(beta)]
            ])
    )

def R_z(alpha: float) -> npt.NDArray[np.floating]:
    """Rotation matrix corresponding to a rotation in the z-plane.

    This function generates the rotation matrix corresponding to a
    rotation in the z-plane by an angle alpha.

    Args:
        alpha (float): Angle of rotation in the z-plane.

    Returns:
        npt.NDArray[np.floating]: Rotation matrix corresponding to a
        rotation in the z-plane.
    
    """
    return (
        np.asarray(
            [
                [np.cos(alpha), -np.sin(alpha), 0.],
                [np.sin(alpha), np.cos(alpha), 0.],
                [0., 0., 1.]
            ])
    ) 

def Q_zyz_euler(omega: npt.NDArray[np.floating]) -> npt.NDArray[np.floating]:
    """ZYZ Euler angle rotation matrix.

    This function generates the ZYZ Euler angle rotation matrix from a
    vector of Euler angles.

    Args:
        omega (npt.NDArray[np.floating]): Vector of Euler angles.

    Returns:
        npt.NDArray[np.floating]: ZYZ Euler angle rotation matrix.
    
    """
    phi, theta, psi = omega
    return np.matmul(R_z(psi), np.matmul(R_y(theta), R_z(phi)))