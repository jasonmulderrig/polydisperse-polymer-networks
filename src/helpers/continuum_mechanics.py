import numpy as np
import numpy.typing as npt

def uniaxial_F_func(
        lmbda: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Deformation gradient and the associated principal stretch matrix
    in the case of uniaxial deformation.

    This function returns the deformation gradient and the associated
    principal stretch matrix in the case of uniaxial deformation.

    Args:
        lmbda (float): Uniaxial stretch state.
    
    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: 
        Deformation gradient and the associated principal stretch matrix
        in the case of uniaxial deformation.
    
    """
    Lmbda = np.asarray([lmbda, 1./np.sqrt(lmbda), 1./np.sqrt(lmbda)])
    F = np.diag(Lmbda)
    return F, Lmbda

def simple_shear_F_func(
        s: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Deformation gradient and the associated principal stretch matrix
    in the case of simple shear.

    This function returns the deformation gradient and the associated
    principal stretch matrix in the case of simple shear.

    Args:
        s (float): Simple shear state.
    
    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: 
        Deformation gradient and the associated principal stretch matrix
        in the case of simple shear.
    
    """
    e_hat = np.eye(3)
    F = np.eye(3) + s * np.outer(e_hat[0], e_hat[2])
    lmbda_0 = np.sqrt(2.+s**2+s*np.sqrt(4.+s**2)) / np.sqrt(2.)
    lmbda_1 = 1.
    lmbda_2 = np.sqrt(2.+s**2-s*np.sqrt(4.+s**2)) / np.sqrt(2.)
    Lmbda = np.asarray([lmbda_0, lmbda_1, lmbda_2])
    return F, Lmbda

def F_func(
        dfrmtn: str,
        x: float) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Deformation gradient and the associated principal stretch matrix,
    given some mode of deformation.

    This function returns the deformation gradient and the associated
    principal stretch matrix, given some mode of deformation.

    Args:
        dfrmtn (str): Called-for mode of deformation.
        x (float): Deformation state.
    
    Returns:
        tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]: 
        Deformation gradient and the associated principal stretch
        matrix, given some mode of deformation.
    
    """
    if dfrmtn == "uniaxial": return uniaxial_F_func(x)
    elif dfrmtn == "simple_shear": return simple_shear_F_func(x)
    else:
        error_str = (
            "The called-for mode of deformation is not implemented!"
        )
        raise NotImplementedError(error_str)

def C_func(F: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    """Right Cauchy-Green deformation tensor.

    This function returns the right Cauchy-Green deformation tensor.

    Args:
        F (npt.NDArray[np.float64]): Deformation gradient.
    
    Returns:
        npt.NDArray[np.float64]: Right Cauchy-Green deformation tensor.
    
    """
    return np.matmul(np.transpose(F), F)

def principal_stretch_decomposition(
        F: npt.NDArray[np.float64]) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    """Principal stretch decomposition.

    This function performs a principal stretch decomposition to a given
    deformation gradient.

    Args:
        F (npt.NDArray[np.float64]): Deformation gradient.
    
    Returns:
        npt.NDArray[np.float64]: Principal stretch matrix and
        associated rotation matrix that diagonalizes the right stretch
        tensor to the principal stretch matrix.
    
    """
    Lmdba_sqrd, P = np.linalg.eigh(C_func(F))
    Lmdba = np.sqrt(Lmdba_sqrd)
    return Lmdba, P

def deformation_protocol_init_func(
        protocol_init: str, protocol: tuple[float]) -> npt.NDArray[np.float64]:
    """Deformation protocol.

    This function initializes the deformation protocol.

    Args:
        protocol_init (str): Short-hand description for the deformation protocol initialization; either "explicit" or "linspace".
        protocol (tuple[float]): Deformation protocol, or information needed to properly initialize the deformation protocol.
    
    Returns:
        npt.NDArray[np.float64]: Deformation protocol.
    
    """
    if protocol_init == "explicit": return np.asarray(protocol)
    elif protocol_init == "linspace":
        return np.linspace(protocol[0], protocol[1], int(protocol[2]))