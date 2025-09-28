import numpy as np
from src.helpers.spherical_quadrature_utils import (
    master_spherical_quadrature_func
)
from src.helpers.spherical_coordinates_utils import (
    cartesian_to_spherical_coords
)

def master_so3_quadrature_func(sph_quad_method: str, num_spin_inc: int) -> tuple[np.ndarray, bool]:
    # Gather spherical quadrature parameterized in unit direction
    # cosines
    sph_quad, sph_quad_symmtry = master_spherical_quadrature_func(
        sph_quad_method)
    
    # Convert spherical quadrature parameterization to spherical
    # coordinates
    sph_quad_x_y_z = sph_quad[:, :-1]
    sph_quad_weights = sph_quad[:, -1]
    sph_quad_r_theta_phi = cartesian_to_spherical_coords(sph_quad_x_y_z)
    # Confirm that the spherical quadrature points are unit vectors
    if not np.allclose(sph_quad_r_theta_phi[:, 0], 1.):
        error_str = (
            "At least one of the called-for spherical quadrature "
            + "points are not unit vectors!"
        )
        raise ValueError(error_str)
    
    # Order angles in Euler rotation angle order (phi before theta)
    sph_quad = np.column_stack(
        (sph_quad_r_theta_phi[:, [-1, -2]], sph_quad_weights))
    sph_quad_num = np.shape(sph_quad)[0]

    # Initialize the spin angle quadrature
    spin_quad_psi = np.linspace(0, 2*np.pi, num_spin_inc, endpoint=False)
    spin_quad_weights = np.ones(num_spin_inc) / num_spin_inc
    spin_quad = np.column_stack((spin_quad_psi, spin_quad_weights))

    # Populate the SO3 quadrature
    so3_quad_num = sph_quad_num * num_spin_inc
    so3_quad = np.empty((so3_quad_num, 4))
    for so3_quad_indx in range(so3_quad_num):
        sph_quad_indx, spin_quad_indx = np.divmod(so3_quad_indx, num_spin_inc)

        # Sequentially populate phi, theta, psi, and weight
        so3_quad[so3_quad_indx, :2] = sph_quad[sph_quad_indx, :2]
        so3_quad[so3_quad_indx, 2] = spin_quad[spin_quad_indx, 0]
        so3_quad[so3_quad_indx, -1] = (
            sph_quad[sph_quad_indx, -1] * spin_quad[spin_quad_indx, -1]
        )
    
    # Normalize weights
    so3_quad_weights = so3_quad[:, -1]
    if sph_quad_symmtry: so3_quad_weights /= (2*np.sum(so3_quad_weights))
    else: so3_quad_weights /= np.sum(so3_quad_weights)
    so3_quad[:, -1] = so3_quad_weights

    return so3_quad, sph_quad_symmtry
