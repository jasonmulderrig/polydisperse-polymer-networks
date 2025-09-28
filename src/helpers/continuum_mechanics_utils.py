import numpy as np

def uniaxial_F_func(lmbda: float) -> tuple[np.ndarray, np.ndarray]:
    Lmbda = np.asarray([lmbda, 1./np.sqrt(lmbda), 1./np.sqrt(lmbda)])
    F = np.diag(Lmbda)
    return F, Lmbda

def simple_shear_F_func(s: float) -> tuple[np.ndarray, np.ndarray]:
    e_hat = np.eye(3)
    F = np.eye(3) + s * np.outer(e_hat[0], e_hat[2])
    lmbda_0 = np.sqrt(2.+s**2+s*np.sqrt(4.+s**2)) / np.sqrt(2.)
    lmbda_1 = 1.
    lmbda_2 = np.sqrt(2.+s**2-s*np.sqrt(4.+s**2)) / np.sqrt(2.)
    Lmbda = np.asarray([lmbda_0, lmbda_1, lmbda_2])
    return F, Lmbda

def F_func(dfrmtn: str, x: float) -> tuple[np.ndarray, np.ndarray]:
    if dfrmtn == "uniaxial": return uniaxial_F_func(x)
    elif dfrmtn == "simple_shear": return simple_shear_F_func(x)
    else:
        error_str = (
            "The called-for deformation gradient is not implemented!"
        )
        raise NotImplementedError(error_str)

def C_func(F: np.ndarray) -> np.ndarray:
    return np.matmul(np.transpose(F), F)

def principal_stretch_decomposition(
        F: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    Lmdba_sqrd, P = np.linalg.eigh(C_func(F))
    Lmdba = np.sqrt(Lmdba_sqrd)
    return Lmdba, P

def deformation_protocol_init_func(
        protocol_init: str, protocol: tuple[float]) -> np.ndarray:
    """Salient chain segment numbers.

    This function initializes the salient chain segment numbers.

    Args:
        n_init (str): Short-hand description for the salient chain segment number initialization protocol; either "explicit" or "linspace".
        n (tuple[int]): Salient chain segment numbers, or information needed to properly initialize the salient chain segment numbers.
    
    Returns:
        tuple[np.ndarray, int]: Salient chain segment numbers (sorted
        from least to greatest), and the number of salient chain segment
        numbers.
    
    """
    # Define the salient chain segment number array
    if protocol_init == "explicit": return np.asarray(protocol)
    elif protocol_init == "linspace":
        return np.linspace(protocol[0], protocol[1], int(protocol[2]))