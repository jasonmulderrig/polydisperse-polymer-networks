import numpy as np

def A_func(u_hat: np.ndarray) -> np.ndarray:
    u_hat_0, u_hat_1, u_hat_2 = u_hat
    
    return (
        np.asarray(
            [
                [0., -u_hat_2, u_hat_1],
                [u_hat_2, 0., -u_hat_0],
                [-u_hat_1, u_hat_0, 0.]
            ])
    )

def Q_axis_angle(omega: np.ndarray) -> np.ndarray:
    omega_norm = np.linalg.norm(omega)
    if omega_norm > 0:
        u_hat = omega / omega_norm
        A = A_func(u_hat)
        return (
            np.eye(3) + np.sin(omega_norm) * A
            + (1.-np.cos(omega_norm)) * np.matmul(A, A)
        )
    else: return np.eye(3)

def R_x(gamma: float) -> np.ndarray:
    return (
        np.asarray(
            [
                [1., 0., 0.],
                [0., np.cos(gamma), -np.sin(gamma)],
                [0., np.sin(gamma), np.cos(gamma)]
            ])
    )

def R_y(beta: float) -> np.ndarray:
    return (
        np.asarray(
            [
                [np.cos(beta), 0., np.sin(beta)],
                [0., 1., 0.],
                [-np.sin(beta), 0., np.cos(beta)]
            ])
    )

def R_z(alpha: float) -> np.ndarray:
    return (
        np.asarray(
            [
                [np.cos(alpha), -np.sin(alpha), 0.],
                [np.sin(alpha), np.cos(alpha), 0.],
                [0., 0., 1.]
            ])
    ) 

def Q_zyz_euler(omega: np.ndarray) -> np.ndarray:
    phi, theta, psi = omega
    return np.matmul(R_z(psi), np.matmul(R_y(theta), R_z(phi)))