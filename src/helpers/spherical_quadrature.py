import numpy as np
import numpy.typing as npt
from src.file_io.file_io import spherical_quadrature_filepath_str

def master_spherical_quadrature_func(
        sph_quad_method: str) -> tuple[npt.NDArray[np.float64], bool]:
    """Master spherical quadrature function.

    This function returns the selected spherical quadrature scheme.

    Args:
        sph_quad_method (str): Short-hand name for the selected spherical quadrature scheme.
    
    Returns:
        tuple[npt.NDArray[np.float64], bool]: Spherical quadrature
        scheme, boolean indicating if the spherical quadrature scheme is
        hemispherically symmetric.
    
    """
    # Gather spherical quadrature parameterized in unit direction
    # cosines
    sph_quad_filepath = spherical_quadrature_filepath_str()
    sph_quad = np.loadtxt(sph_quad_filepath+sph_quad_method+".dat")
    sph_quad_symmtry = bool(
        np.loadtxt(sph_quad_filepath+sph_quad_method+"_sph_quad_symmtry.dat"))
    
    # Normalize weights
    sph_quad_w = sph_quad[:, -1]
    if sph_quad_symmtry: sph_quad_w /= (2*np.sum(sph_quad_w))
    else: sph_quad_w /= np.sum(sph_quad_w)
    sph_quad[:, -1] = sph_quad_w

    return sph_quad, sph_quad_symmtry