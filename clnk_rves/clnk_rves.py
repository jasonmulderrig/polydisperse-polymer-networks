# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import logging
import logging
logging.disable(logging.WARNING)

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
from src.file_io.file_io import filename_str
from src.helpers.so3_quadrature_utils import master_so3_quadrature_func
from src.helpers.chain_free_energy_utils import (
    master_w_c_func,
    master_w_c_dfrmtn_func
)
from src.helpers.chain_segment_number_utils import n_init_func
from src.helpers.clnk_degree_dispersity_utils import k_init_func
from src.helpers.clnk_structure_dispersity_utils import (
    m_clnks_init_func,
    n_clnks_init_func
)
from src.helpers.chain_length_utils import (
    master_r_crit_func,
    master_r_rms_func
)
from src.helpers.clnk_structure_utils import (
    recommended_clnk_init_func,
    vol_quad_clnk_func,
    amended_3_chn_clnk_X_hat_clnk_func,
    regular_tetrahedral_4_chn_clnk_X_hat_clnk_func,
    equilateral_triangular_bipyramidal_5_chn_clnk_X_hat_clnk_func,
    regular_octahedral_6_chn_clnk_X_hat_clnk_func,
    equilateral_pentagonal_bipyramidal_7_chn_clnk_X_hat_clnk_func,
    cube_8_chn_clnk_X_hat_clnk_func,
    x_hat_clnk_func,
    com_x_clnk_func
)
from src.helpers.chain_conformation_utils import gamma_func
from src.helpers.continuum_mechanics_utils import (
    deformation_protocol_init_func,
    F_func
)
from src.helpers.clnk_free_rotation_utils import (
    monodisperse_clnk_free_rot,
    clnk_free_rot_cnstrnd_mnmztn,
    clnk_free_rot_approx
)
from src.helpers.clnk_frame_averaging_utils import (
    monodisperse_clnk_frame_avrg,
    W_flucts_clnk_star_frame_avrg,
    clnk_frame_avrg_cnstrnd_mnmztn,
    W_flucts_clnk_frame_avrg,
    clnk_frame_avrg_approx,
    W_flucts_clnk_frame_avrg_approx
)

def clnk_rves_analysis(
        label: DictConfig,
        sample: int,
        deformation: DictConfig,
        w_c_dist: str,
        w_c_args: list,
        w_c_dfrmtn_dist: str,
        w_c_dfrmtn_args: list,
        b: float,
        f: int,
        p_n_args: list,
        n: list,
        n_init: str) -> None:
    # Generate filename prefix
    filename_prefix = filename_str(
        label.workdir, label.date, label.batch, sample)
    
    # Create a seeded default random number generator for the
    # differential evolution global constrained minimization
    rng = np.random.default_rng(42)

    # Numerical quadrature scheme
    points, weights = np.polynomial.legendre.leggauss(1001)

    # SO3 quadrature scheme
    so3_quad, sph_quad_symmtry = master_so3_quadrature_func("bazant_oh_009", 5) # master_so3_quadrature_func("bazant_oh_013", 16)
    so3_quad_num = np.shape(so3_quad)[0]

    # Convex hull quadrature scheme
    vol_clnk_side_dim_points = 10 # 15

    # Evaluate W_flucts
    eval_W_flucts = True
    
    # Extract arguments for the polymer chain segment number probability
    # distribution, and convert lists to tuples
    p_n_p_args = p_n_args[0]
    p_n_n_args = p_n_args[1]
    p_n_p_args = [] if p_n_p_args == [None] else p_n_p_args
    p_n_n_args = [] if p_n_n_args == [None] else p_n_n_args
    p_n_p_args = tuple(p_n_p_args)
    p_n_n_args = tuple(p_n_n_args)
    n = tuple(n)

    # Extract arguments for the polymer chain free energy function
    w_c_args = [] if w_c_args == [None] else w_c_args
    w_c_args = tuple(w_c_args)

    # Extract polymer chain free energy function
    w_c_func = master_w_c_func(w_c_dist)

    # print(w_c_dist)
    # print(w_c_args)
    # print(w_c_func)

    # Extract arguments for the cross-link deformation polymer chain
    # free energy function
    w_c_dfrmtn_args = [] if w_c_dfrmtn_args == [None] else w_c_dfrmtn_args
    w_c_dfrmtn_args = tuple(w_c_dfrmtn_args)

    # Extract cross-link deformation polymer chain free energy function
    w_c_dfrmtn_func = master_w_c_dfrmtn_func(w_c_dfrmtn_dist)

    # print(w_c_dfrmtn_dist)
    # print(w_c_dfrmtn_args)
    # print(w_c_dfrmtn_func)

    # Initialize the salient chain segment numbers, the
    # elastically-effective cross-link degree, the chain segment number
    # multiplicity for each distinct cross-link structure, and the chain
    # segment number for each chain in each distinct cross-link
    # structure
    n, N = n_init_func(n_init, n)
    k = k_init_func(f)
    m_clnks = m_clnks_init_func(k, N)
    n_clnks = n_clnks_init_func(n, m_clnks)
    # print(n_clnks)

    # Save the chain segment number for each chain in each distinct
    # cross-link structure
    for k_indx in range(np.shape(k)[0]):
        n_clnks_k_filename = (
            filename_prefix + "-n_clnks_k_" + str(k[k_indx]) + ".dat"
        )
        np.savetxt(n_clnks_k_filename, n_clnks[k_indx], fmt="%d")

    # Calculate the critical polymer chain contour length on a
    # chain-by-chain basis
    r_crit_clnks = []
    for k_indx in range(np.shape(k)[0]):
        n_clnks_arr = n_clnks[k_indx]
        r_crit_clnks_arr = np.empty_like(n_clnks_arr, dtype=float)
        for clnk_indx in np.ndindex(np.shape(n_clnks_arr)):
            r_crit_clnks_arr[clnk_indx] = master_r_crit_func(
                n_clnks_arr[clnk_indx], b, w_c_dist, w_c_args)
        r_crit_clnks.append(r_crit_clnks_arr)
    # print(r_crit_clnks)

    # Calculate the root-mean-square polymer chain length on a
    # chain-by-chain basis
    r_rms_clnks = []
    for k_indx in range(np.shape(k)[0]):
        n_clnks_arr = n_clnks[k_indx]
        r_crit_clnks_arr = r_crit_clnks[k_indx]
        r_rms_clnks_arr = np.empty_like(n_clnks_arr, dtype=float)
        for clnk_indx in np.ndindex(np.shape(n_clnks_arr)):
            r_rms_clnks_arr[clnk_indx] = master_r_rms_func(
                points, weights, r_crit_clnks_arr[clnk_indx],
                n_clnks_arr[clnk_indx], b, w_c_dist, w_c_func, w_c_args)
        r_rms_clnks.append(r_rms_clnks_arr)
    # print(r_rms_clnks)

    # Initialize the cross-link structures
    X_clnks = []
    vol_quad_clnks = []
    omega_clnks_init = []
    y_clnks_init = []
    gamma_clnks_init = []
    delta_omega_clnks_init = []
    delta_y_clnks_init = []
    for k_indx in range(np.shape(k)[0]):
        n_clnks_arr = n_clnks[k_indx]
        r_rms_clnks_arr = r_rms_clnks[k_indx]
        C_R, k_num = np.shape(n_clnks_arr)
        X_clnks_arr = np.zeros((C_R, k_num, 3))
        X_l_clnk, _, _  = recommended_clnk_init_func(
            X_clnks_arr[0]*b, type_8_chn_clnk="cube")
        vol_quad_clnk = vol_quad_clnk_func(X_l_clnk, vol_clnk_side_dim_points)
        vol_quad_clnk_shape = np.shape(vol_quad_clnk)
        vol_quad_clnks_arr = np.zeros(
            (C_R, vol_quad_clnk_shape[0], vol_quad_clnk_shape[1]))
        omega_clnks_init_arr = np.zeros((C_R, 3))
        y_clnks_init_arr = np.zeros((C_R, 3))
        gamma_clnks_init_arr = np.zeros((C_R, k_num))
        delta_omega_clnks_init_arr = np.zeros((C_R, 3))
        delta_y_clnks_init_arr = np.zeros((C_R, 3))
        for clnk_indx in range(C_R):
            n_clnk = n_clnks_arr[clnk_indx]
            X_clnk, omega_clnk_init, y_clnk_init = recommended_clnk_init_func(
                r_rms_clnks_arr[clnk_indx], type_8_chn_clnk="cube")
            X_l_clnk, _, _ = recommended_clnk_init_func(
                n_clnk*b, type_8_chn_clnk="cube")
            vol_quad_clnk = vol_quad_clnk_func(
                X_l_clnk, vol_clnk_side_dim_points)
            X_clnks_arr[clnk_indx] = X_clnk
            vol_quad_clnks_arr[clnk_indx] = vol_quad_clnk
            omega_clnks_init_arr[clnk_indx] = omega_clnk_init
            y_clnks_init_arr[clnk_indx] = y_clnk_init
            r_clnk_init = X_clnk - y_clnk_init
            for chn_indx in range(k_num):
                gamma_clnks_init_arr[clnk_indx, chn_indx] = gamma_func(
                    np.linalg.norm(r_clnk_init[chn_indx]), n_clnk[chn_indx], b)
        X_clnks.append(X_clnks_arr)
        vol_quad_clnks.append(vol_quad_clnks_arr)
        omega_clnks_init.append(omega_clnks_init_arr)
        y_clnks_init.append(y_clnks_init_arr)
        gamma_clnks_init.append(gamma_clnks_init_arr)
        delta_omega_clnks_init.append(delta_omega_clnks_init_arr)
        delta_y_clnks_init.append(delta_y_clnks_init_arr)
    # print(X_clnks)
    # print(vol_quad_clnks)
    # print(omega_clnks_init)
    # print(y_clnks_init)
    # print(gamma_clnks_init)
    # print(delta_omega_clnks_init)
    # print(delta_y_clnks_init)
    
    # Verify that all initial cross-link positions coincide with the origin
    for k_indx in range(np.shape(k)[0]):
        y_clnks_init_arr = y_clnks_init[k_indx]
        C_R = np.shape(y_clnks_init_arr)[0]
        for clnk_indx in range(C_R):
            y_clnk_init = y_clnks_init_arr[clnk_indx]
            if not np.allclose(y_clnk_init, np.zeros(3)):
                error_str = (
                    "The initial position of a cross-link is not at "
                    + "the origin."
                )
                raise ValueError(error_str)
    
    # Gather unit cross-link structures
    amended_3_chn_clnk_X_hat_clnk = amended_3_chn_clnk_X_hat_clnk_func()
    regular_tetrahedral_4_chn_clnk_X_hat_clnk = (
        regular_tetrahedral_4_chn_clnk_X_hat_clnk_func()
    )
    equilateral_triangular_bipyramidal_5_chn_clnk_X_hat_clnk = (
        equilateral_triangular_bipyramidal_5_chn_clnk_X_hat_clnk_func()
    )
    regular_octahedral_6_chn_clnk_X_hat_clnk = (
        regular_octahedral_6_chn_clnk_X_hat_clnk_func()
    )
    equilateral_pentagonal_bipyramidal_7_chn_clnk_X_hat_clnk = (
        equilateral_pentagonal_bipyramidal_7_chn_clnk_X_hat_clnk_func()
    )
    cube_8_chn_clnk_X_hat_clnk = cube_8_chn_clnk_X_hat_clnk_func()

    # Deformation protocol initialization
    dfrmtn_protocol_class = deformation.protocol_class
    dfrmtn_protocol = []
    for protocol_indx in range(len(dfrmtn_protocol_class)):
        dfrmtn_protocol.append(
            deformation_protocol_init_func(
                deformation.protocol_init,
                deformation.protocol[protocol_indx]))
    # print(dfrmtn_protocol_class)
    # print(dfrmtn_protocol)

    # Data initialization
    omega_clnks_free_rot = []
    omega_clnks_norm_free_rot = []
    y_clnks_free_rot = []
    y_clnks_norm_free_rot = []
    gamma_clnks_free_rot = []
    W_clnks_free_rot = []
    W_flucts_clnks_free_rot = []

    delta_omega_clnks_free_rot_approx = []
    delta_omega_clnks_norm_free_rot_approx = []
    delta_y_clnks_free_rot_approx = []
    delta_y_clnks_norm_free_rot_approx = []
    gamma_clnks_free_rot_approx = []
    W_clnks_free_rot_approx = []
    W_flucts_clnks_free_rot_approx = []

    y_clnks_frame_avrg_so3 = []
    y_clnks_norm_frame_avrg_so3 = []
    gamma_clnks_frame_avrg_so3 = []
    W_clnks_frame_avrg_so3 = []
    W_flucts_clnks_frame_avrg_so3 = []
    y_clnks_frame_avrg_so3_quad = []
    y_clnks_norm_frame_avrg_so3_quad = []
    gamma_clnks_frame_avrg_so3_quad = []
    W_clnks_frame_avrg_so3_quad = []
    W_flucts_clnks_frame_avrg_so3_quad = []

    delta_y_clnks_frame_avrg_approx_so3 = []
    delta_y_clnks_norm_frame_avrg_approx_so3 = []
    gamma_clnks_frame_avrg_approx_so3 = []
    W_clnks_frame_avrg_approx_so3 = []
    W_flucts_clnks_frame_avrg_approx_so3 = []
    delta_y_clnks_frame_avrg_approx_so3_quad = []
    delta_y_clnks_norm_frame_avrg_approx_so3_quad = []
    gamma_clnks_frame_avrg_approx_so3_quad = []
    W_clnks_frame_avrg_approx_so3_quad = []
    W_flucts_clnks_frame_avrg_approx_so3_quad = []

    for protocol_indx in range(len(dfrmtn_protocol)):
        omega_clnks_free_rot_protocol = []
        omega_clnks_norm_free_rot_protocol = []
        y_clnks_free_rot_protocol = []
        y_clnks_norm_free_rot_protocol = []
        gamma_clnks_free_rot_protocol = []
        W_clnks_free_rot_protocol = []
        W_flucts_clnks_free_rot_protocol = []

        delta_omega_clnks_free_rot_approx_protocol = []
        delta_omega_clnks_norm_free_rot_approx_protocol = []
        delta_y_clnks_free_rot_approx_protocol = []
        delta_y_clnks_norm_free_rot_approx_protocol = []
        gamma_clnks_free_rot_approx_protocol = []
        W_clnks_free_rot_approx_protocol = []
        W_flucts_clnks_free_rot_approx_protocol = []

        y_clnks_frame_avrg_so3_protocol = []
        y_clnks_norm_frame_avrg_so3_protocol = []
        gamma_clnks_frame_avrg_so3_protocol = []
        W_clnks_frame_avrg_so3_protocol = []
        W_flucts_clnks_frame_avrg_so3_protocol = []
        y_clnks_frame_avrg_so3_quad_protocol = []
        y_clnks_norm_frame_avrg_so3_quad_protocol = []
        gamma_clnks_frame_avrg_so3_quad_protocol = []
        W_clnks_frame_avrg_so3_quad_protocol = []
        W_flucts_clnks_frame_avrg_so3_quad_protocol = []

        delta_y_clnks_frame_avrg_approx_so3_protocol = []
        delta_y_clnks_norm_frame_avrg_approx_so3_protocol = []
        gamma_clnks_frame_avrg_approx_so3_protocol = []
        W_clnks_frame_avrg_approx_so3_protocol = []
        W_flucts_clnks_frame_avrg_approx_so3_protocol = []
        delta_y_clnks_frame_avrg_approx_so3_quad_protocol = []
        delta_y_clnks_norm_frame_avrg_approx_so3_quad_protocol = []
        gamma_clnks_frame_avrg_approx_so3_quad_protocol = []
        W_clnks_frame_avrg_approx_so3_quad_protocol = []
        W_flucts_clnks_frame_avrg_approx_so3_quad_protocol = []
        
        dfrmtn_arr = dfrmtn_protocol[protocol_indx]
        num_dfrmtn_steps = np.shape(dfrmtn_arr)[0]

        for k_indx in range(np.shape(k)[0]):
            n_clnks_arr = n_clnks[k_indx]
            C_R, k_num = np.shape(n_clnks_arr)
            
            omega_clnks_free_rot_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, 3)))
            omega_clnks_norm_free_rot_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            y_clnks_free_rot_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, 3)))
            y_clnks_norm_free_rot_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            gamma_clnks_free_rot_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, k_num)))
            W_clnks_free_rot_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            W_flucts_clnks_free_rot_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            
            delta_omega_clnks_free_rot_approx_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, 3)))
            delta_omega_clnks_norm_free_rot_approx_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            delta_y_clnks_free_rot_approx_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, 3)))
            delta_y_clnks_norm_free_rot_approx_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            gamma_clnks_free_rot_approx_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, k_num)))
            W_clnks_free_rot_approx_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            W_flucts_clnks_free_rot_approx_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            
            y_clnks_frame_avrg_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num, 3)))
            y_clnks_norm_frame_avrg_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            gamma_clnks_frame_avrg_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num, k_num)))
            W_clnks_frame_avrg_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            W_flucts_clnks_frame_avrg_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            y_clnks_frame_avrg_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, 3)))
            y_clnks_norm_frame_avrg_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            gamma_clnks_frame_avrg_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, k_num)))
            W_clnks_frame_avrg_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            W_flucts_clnks_frame_avrg_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            
            delta_y_clnks_frame_avrg_approx_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num, 3)))
            delta_y_clnks_norm_frame_avrg_approx_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            gamma_clnks_frame_avrg_approx_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num, k_num)))
            W_clnks_frame_avrg_approx_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            W_flucts_clnks_frame_avrg_approx_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            delta_y_clnks_frame_avrg_approx_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, 3)))
            delta_y_clnks_norm_frame_avrg_approx_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            gamma_clnks_frame_avrg_approx_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, k_num)))
            W_clnks_frame_avrg_approx_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            W_flucts_clnks_frame_avrg_approx_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
        
        omega_clnks_free_rot.append(omega_clnks_free_rot_protocol)
        omega_clnks_norm_free_rot.append(omega_clnks_norm_free_rot_protocol)
        y_clnks_free_rot.append(y_clnks_free_rot_protocol)
        y_clnks_norm_free_rot.append(y_clnks_norm_free_rot_protocol)
        gamma_clnks_free_rot.append(gamma_clnks_free_rot_protocol)
        W_clnks_free_rot.append(W_clnks_free_rot_protocol)
        W_flucts_clnks_free_rot.append(W_flucts_clnks_free_rot_protocol)

        delta_omega_clnks_free_rot_approx.append(
            delta_omega_clnks_free_rot_approx_protocol)
        delta_omega_clnks_norm_free_rot_approx.append(
            delta_omega_clnks_norm_free_rot_approx_protocol)
        delta_y_clnks_free_rot_approx.append(
            delta_y_clnks_free_rot_approx_protocol)
        delta_y_clnks_norm_free_rot_approx.append(
            delta_y_clnks_norm_free_rot_approx_protocol)
        gamma_clnks_free_rot_approx.append(gamma_clnks_free_rot_approx_protocol)
        W_clnks_free_rot_approx.append(W_clnks_free_rot_approx_protocol)
        W_flucts_clnks_free_rot_approx.append(
            W_flucts_clnks_free_rot_approx_protocol)

        y_clnks_frame_avrg_so3.append(y_clnks_frame_avrg_so3_protocol)
        y_clnks_norm_frame_avrg_so3.append(y_clnks_norm_frame_avrg_so3_protocol)
        gamma_clnks_frame_avrg_so3.append(gamma_clnks_frame_avrg_so3_protocol)
        W_clnks_frame_avrg_so3.append(W_clnks_frame_avrg_so3_protocol)
        W_flucts_clnks_frame_avrg_so3.append(
            W_flucts_clnks_frame_avrg_so3_protocol)
        y_clnks_frame_avrg_so3_quad.append(y_clnks_frame_avrg_so3_quad_protocol)
        y_clnks_norm_frame_avrg_so3_quad.append(
            y_clnks_norm_frame_avrg_so3_quad_protocol)
        gamma_clnks_frame_avrg_so3_quad.append(
            gamma_clnks_frame_avrg_so3_quad_protocol)
        W_clnks_frame_avrg_so3_quad.append(W_clnks_frame_avrg_so3_quad_protocol)
        W_flucts_clnks_frame_avrg_so3_quad.append(
            W_flucts_clnks_frame_avrg_so3_quad_protocol)

        delta_y_clnks_frame_avrg_approx_so3.append(
            delta_y_clnks_frame_avrg_approx_so3_protocol)
        delta_y_clnks_norm_frame_avrg_approx_so3.append(
            delta_y_clnks_norm_frame_avrg_approx_so3_protocol)
        gamma_clnks_frame_avrg_approx_so3.append(
            gamma_clnks_frame_avrg_approx_so3_protocol)
        W_clnks_frame_avrg_approx_so3.append(
            W_clnks_frame_avrg_approx_so3_protocol)
        W_flucts_clnks_frame_avrg_approx_so3.append(
            W_flucts_clnks_frame_avrg_approx_so3_protocol)
        delta_y_clnks_frame_avrg_approx_so3_quad.append(
            delta_y_clnks_frame_avrg_approx_so3_quad_protocol)
        delta_y_clnks_norm_frame_avrg_approx_so3_quad.append(
            delta_y_clnks_norm_frame_avrg_approx_so3_quad_protocol)
        gamma_clnks_frame_avrg_approx_so3_quad.append(
            gamma_clnks_frame_avrg_approx_so3_quad_protocol)
        W_flucts_clnks_frame_avrg_approx_so3_quad.append(
            W_flucts_clnks_frame_avrg_approx_so3_quad_protocol)
        W_clnks_frame_avrg_approx_so3_quad.append(
            W_clnks_frame_avrg_approx_so3_quad_protocol)

    # Step through each deformation protocol
    for protocol_indx in range(len(dfrmtn_protocol)):
        dfrmtn_arr = dfrmtn_protocol[protocol_indx]
        num_dfrmtn_steps = np.shape(dfrmtn_arr)[0]

        # Evaluate the deformation of each cross-link structure
        for k_indx in range(np.shape(k)[0]):
            C_R, k_num = np.shape(n_clnks[k_indx])
            # Step through deformation steps
            for dfrmtn_step in range(num_dfrmtn_steps):
                # Deformation gradient
                F, Lmbda = F_func(
                    dfrmtn_protocol_class[protocol_indx],
                    dfrmtn_arr[dfrmtn_step])
                for clnk_indx in range(C_R):
                    # Gather universal cross-link structure
                    n_clnk = n_clnks[k_indx][clnk_indx]
                    X_clnk = X_clnks[k_indx][clnk_indx]
                    X_hat_clnk = x_hat_clnk_func(X_clnk)
                    com_X_clnk = com_x_clnk_func(X_clnk)
                    vol_quad_clnk = vol_quad_clnks[k_indx][clnk_indx]
                    y_clnk_init = y_clnks_init[k_indx][clnk_indx]
                    gamma_clnk_init = gamma_clnks_init[k_indx][clnk_indx]
                    
                    # Gather initial cross-link structure for the
                    # constrained minimization schemes
                    if dfrmtn_step == 0: init_dfrmtn_step = 0
                    else: init_dfrmtn_step = dfrmtn_step - 1

                    # Free rotation model
                    omega_clnk_free_rot_init = (
                        omega_clnks_free_rot[protocol_indx][k_indx][init_dfrmtn_step, clnk_indx]
                    )
                    y_clnk_free_rot_init = (
                        y_clnks_free_rot[protocol_indx][k_indx][init_dfrmtn_step, clnk_indx]
                    )

                    # Frame averaging model
                    y_clnk_frame_avrg_so3_init = (
                        y_clnks_frame_avrg_so3[protocol_indx][k_indx][init_dfrmtn_step, clnk_indx]
                    )
                    
                    if (np.all(np.equal(n_clnk, n_clnk[0])) and
                        np.allclose(com_X_clnk, np.zeros(3)) and
                        np.allclose(y_clnk_init, np.zeros(3))):
                        
                        # Evaluate the free rotation model for monodisperse
                        # cross-link structures
                        monodisperse_clnk = False
                        if k_num == 4:
                            monodisperse_clnk = np.allclose(
                                X_hat_clnk,
                                regular_tetrahedral_4_chn_clnk_X_hat_clnk)
                        elif k_num == 6:
                            monodisperse_clnk = np.allclose(
                                X_hat_clnk,
                                regular_octahedral_6_chn_clnk_X_hat_clnk)
                        elif k_num == 8:
                            monodisperse_clnk = np.allclose(
                                X_hat_clnk, cube_8_chn_clnk_X_hat_clnk)
                        if monodisperse_clnk:
                            (_, y_clnk_star, y_clnk_star_norm, gamma_clnk_star,
                             W_clnk_star, W_flucts_clnk_star) = (
                                monodisperse_clnk_free_rot(
                                    eval_W_flucts, F, Lmbda, n_clnk, b, X_clnk,
                                    vol_quad_clnk, y_clnk_init, gamma_clnk_init,
                                    w_c_func, w_c_args,
                                    w_c_dfrmtn_func, w_c_dfrmtn_args)
                            )

                            y_clnks_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star
                            )
                            y_clnks_norm_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star_norm
                            )
                            gamma_clnks_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                gamma_clnk_star
                            )
                            W_clnks_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_star
                            )
                            W_flucts_clnks_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_flucts_clnk_star
                            )

                            delta_y_clnks_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star
                            )
                            delta_y_clnks_norm_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star_norm
                            )
                            gamma_clnks_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                gamma_clnk_star
                            )
                            W_clnks_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_star
                            )
                            W_flucts_clnks_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_flucts_clnk_star
                            )
                        
                        # Evaluate the frame averaging model for
                        # monodisperse cross-link structures
                        monodisperse_clnk = False
                        if k_num == 3:
                            monodisperse_clnk = np.allclose(
                                X_hat_clnk, amended_3_chn_clnk_X_hat_clnk)
                        elif k_num == 4:
                            monodisperse_clnk = np.allclose(
                                X_hat_clnk,
                                regular_tetrahedral_4_chn_clnk_X_hat_clnk)
                        elif k_num == 5:
                            monodisperse_clnk = np.allclose(
                                X_hat_clnk,
                                equilateral_triangular_bipyramidal_5_chn_clnk_X_hat_clnk)
                        elif k_num == 6:
                            monodisperse_clnk = np.allclose(
                                X_hat_clnk,
                                regular_octahedral_6_chn_clnk_X_hat_clnk)
                        elif k_num == 7:
                            monodisperse_clnk = np.allclose(
                                X_hat_clnk,
                                equilateral_pentagonal_bipyramidal_7_chn_clnk_X_hat_clnk)
                        elif k_num == 8:
                            monodisperse_clnk = np.allclose(
                                X_hat_clnk, cube_8_chn_clnk_X_hat_clnk)
                        if monodisperse_clnk:
                            (y_clnk_star_frame_avrg_so3,
                             y_clnk_star_norm_frame_avrg_so3,
                             gamma_clnk_star_frame_avrg_so3,
                             W_clnk_star_frame_avrg_so3,
                             y_clnk_star_frame_avrg_so3_quad,
                             y_clnk_star_norm_frame_avrg_so3_quad,
                             gamma_clnk_star_frame_avrg_so3_quad,
                             W_clnk_star_frame_avrg_so3_quad) = (
                                monodisperse_clnk_frame_avrg(
                                    F, n_clnk, b, X_clnk, so3_quad,
                                    sph_quad_symmtry, y_clnk_init,
                                    w_c_func, w_c_args,
                                    w_c_dfrmtn_func, w_c_dfrmtn_args)
                            )

                            (W_flucts_clnk_star_frame_avrg_so3,
                             W_flucts_clnk_star_frame_avrg_so3_quad) = (
                                W_flucts_clnk_star_frame_avrg(
                                    eval_W_flucts, F, n_clnk, b, X_clnk,
                                    vol_quad_clnk, so3_quad,
                                    sph_quad_symmtry, y_clnk_init,
                                    w_c_func, w_c_args,
                                    w_c_dfrmtn_func, w_c_dfrmtn_args)
                            )

                            y_clnks_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star_frame_avrg_so3
                            )
                            y_clnks_norm_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star_norm_frame_avrg_so3
                            )
                            gamma_clnks_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                gamma_clnk_star_frame_avrg_so3
                            )
                            W_clnks_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_star_frame_avrg_so3
                            )
                            W_flucts_clnks_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_flucts_clnk_star_frame_avrg_so3
                            )
                            y_clnks_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star_frame_avrg_so3_quad
                            )
                            y_clnks_norm_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star_norm_frame_avrg_so3_quad
                            )
                            gamma_clnks_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                gamma_clnk_star_frame_avrg_so3_quad
                            )
                            W_clnks_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_star_frame_avrg_so3_quad
                            )
                            W_flucts_clnks_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_flucts_clnk_star_frame_avrg_so3_quad
                            )

                            delta_y_clnks_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star_frame_avrg_so3
                            )
                            delta_y_clnks_norm_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star_norm_frame_avrg_so3
                            )
                            gamma_clnks_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                gamma_clnk_star_frame_avrg_so3
                            )
                            W_clnks_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_star_frame_avrg_so3
                            )
                            W_flucts_clnks_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_flucts_clnk_star_frame_avrg_so3
                            )
                            delta_y_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star_frame_avrg_so3_quad
                            )
                            delta_y_clnks_norm_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star_norm_frame_avrg_so3_quad
                            )
                            gamma_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                gamma_clnk_star_frame_avrg_so3_quad
                            )
                            W_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_star_frame_avrg_so3_quad
                            )
                            W_flucts_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_flucts_clnk_star_frame_avrg_so3_quad
                            )
                    
                    # Evaluate the cross-link structure deformation
                    else:
                        # Set the scope of the constrained minimization
                        # solver
                        cnstrnd_mnmztn_scope = "glbl" # "lcl", "glbl"
                        cnstrnd_mnmztn_method = "shgo" # "COBYLA", "COBYQA", "trust-constr", "differential-evolution", "shgo"
                        
                        # Free rotation constrained minimization
                        (omega_clnk_free_rot, omega_clnk_norm_free_rot,
                         y_clnk_free_rot, y_clnk_norm_free_rot,
                         gamma_clnk_free_rot, W_clnk_free_rot,
                         W_flucts_clnk_free_rot) = (
                            clnk_free_rot_cnstrnd_mnmztn(
                                eval_W_flucts, cnstrnd_mnmztn_scope,
                                cnstrnd_mnmztn_method, rng, F, n_clnk, b,
                                X_clnk, vol_quad_clnk, omega_clnk_free_rot_init,
                                y_clnk_free_rot_init, w_c_func, w_c_args,
                                w_c_dfrmtn_func, w_c_dfrmtn_args)
                        )

                        omega_clnks_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            omega_clnk_free_rot
                        )
                        omega_clnks_norm_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            omega_clnk_norm_free_rot
                        )
                        y_clnks_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            y_clnk_free_rot
                        )
                        y_clnks_norm_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            y_clnk_norm_free_rot
                        )
                        gamma_clnks_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            gamma_clnk_free_rot
                        )
                        W_clnks_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_clnk_free_rot
                        )
                        W_flucts_clnks_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_flucts_clnk_free_rot
                        )

                        # Frame averaging constrained minimization
                        (y_clnk_frame_avrg_so3, y_clnk_norm_frame_avrg_so3,
                         gamma_clnk_frame_avrg_so3, W_clnk_frame_avrg_so3,
                         y_clnk_frame_avrg_so3_quad,
                         y_clnk_norm_frame_avrg_so3_quad,
                         gamma_clnk_frame_avrg_so3_quad,
                         W_clnk_frame_avrg_so3_quad) = (
                            clnk_frame_avrg_cnstrnd_mnmztn(
                                cnstrnd_mnmztn_scope, cnstrnd_mnmztn_method,
                                rng, F, n_clnk, b, X_clnk, so3_quad,
                                sph_quad_symmtry, y_clnk_frame_avrg_so3_init,
                                w_c_func, w_c_args,
                                w_c_dfrmtn_func, w_c_dfrmtn_args)
                        )

                        (W_flucts_clnk_frame_avrg_so3,
                         W_flucts_clnk_frame_avrg_so3_quad) = (
                            W_flucts_clnk_frame_avrg(
                                eval_W_flucts, F, n_clnk, b, X_clnk,
                                vol_quad_clnk, so3_quad, sph_quad_symmtry,
                                w_c_func, w_c_args,
                                w_c_dfrmtn_func, w_c_dfrmtn_args)
                        )

                        y_clnks_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            y_clnk_frame_avrg_so3
                        )
                        y_clnks_norm_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            y_clnk_norm_frame_avrg_so3
                        )
                        gamma_clnks_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            gamma_clnk_frame_avrg_so3
                        )
                        W_clnks_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_clnk_frame_avrg_so3
                        )
                        W_flucts_clnks_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_flucts_clnk_frame_avrg_so3
                        )
                        y_clnks_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            y_clnk_frame_avrg_so3_quad
                        )
                        y_clnks_norm_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            y_clnk_norm_frame_avrg_so3_quad
                        )
                        gamma_clnks_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            gamma_clnk_frame_avrg_so3_quad
                        )
                        W_clnks_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_clnk_frame_avrg_so3_quad
                        )
                        W_flucts_clnks_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_flucts_clnk_frame_avrg_so3_quad
                        )
                        
                        if not np.all(np.equal(n_clnk, n_clnk[0])):
                            # Free rotation approximation
                            clnk_approx = False
                            if k_num == 4:
                                clnk_approx = np.allclose(
                                    X_hat_clnk,
                                    regular_tetrahedral_4_chn_clnk_X_hat_clnk)
                            elif k_num == 8:
                                clnk_approx = np.allclose(
                                    X_hat_clnk, cube_8_chn_clnk_X_hat_clnk)
                            if clnk_approx:
                                (_, delta_omega_clnk_free_rot_approx,
                                 delta_omega_clnk_norm_free_rot_approx,
                                 delta_y_clnk_free_rot_approx,
                                 delta_y_clnk_norm_free_rot_approx,
                                 gamma_clnk_free_rot_approx,
                                 W_clnk_free_rot_approx,
                                 W_flucts_clnk_free_rot_approx) = (
                                    clnk_free_rot_approx(
                                        eval_W_flucts, F, n_clnk, b, X_clnk,
                                        vol_quad_clnk, w_c_func, w_c_args,
                                        w_c_dfrmtn_func, w_c_dfrmtn_args)
                                )
                                
                                delta_omega_clnks_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    delta_omega_clnk_free_rot_approx
                                )
                                delta_omega_clnks_norm_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    delta_omega_clnk_norm_free_rot_approx
                                )
                                delta_y_clnks_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    delta_y_clnk_free_rot_approx
                                )
                                delta_y_clnks_norm_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    delta_y_clnk_norm_free_rot_approx
                                )
                                gamma_clnks_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    gamma_clnk_free_rot_approx
                                )
                                W_clnks_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_clnk_free_rot_approx
                                )
                                W_flucts_clnks_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_flucts_clnk_free_rot_approx
                                )
                            
                            # Frame averaging approximation
                            if np.allclose(com_X_clnk, np.zeros(3)):
                                (delta_y_clnk_frame_avrg_approx_so3,
                                 delta_y_clnk_norm_frame_avrg_approx_so3,
                                 gamma_clnk_frame_avrg_approx_so3,
                                 W_clnk_frame_avrg_approx_so3,
                                 delta_y_clnk_frame_avrg_approx_so3_quad,
                                 delta_y_clnk_norm_frame_avrg_approx_so3_quad,
                                 gamma_clnk_frame_avrg_approx_so3_quad,
                                 W_clnk_frame_avrg_approx_so3_quad) = (
                                    clnk_frame_avrg_approx(
                                        F, n_clnk, b, X_clnk, so3_quad,
                                        sph_quad_symmtry, w_c_func, w_c_args,
                                        w_c_dfrmtn_func, w_c_dfrmtn_args)
                                )

                                (W_flucts_clnk_frame_avrg_approx_so3,
                                 W_flucts_clnk_frame_avrg_approx_so3_quad) = (
                                    W_flucts_clnk_frame_avrg_approx(
                                        eval_W_flucts, F, n_clnk, b, X_clnk,
                                        vol_quad_clnk, so3_quad,
                                        sph_quad_symmtry, w_c_func, w_c_args,
                                        w_c_dfrmtn_func, w_c_dfrmtn_args)
                                )

                                delta_y_clnks_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    delta_y_clnk_frame_avrg_approx_so3
                                )
                                delta_y_clnks_norm_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    delta_y_clnk_norm_frame_avrg_approx_so3
                                )
                                gamma_clnks_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    gamma_clnk_frame_avrg_approx_so3
                                )
                                W_clnks_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_clnk_frame_avrg_approx_so3
                                )
                                W_flucts_clnks_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_flucts_clnk_frame_avrg_approx_so3
                                )
                                delta_y_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    delta_y_clnk_frame_avrg_approx_so3_quad
                                )
                                delta_y_clnks_norm_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    delta_y_clnk_norm_frame_avrg_approx_so3_quad
                                )
                                gamma_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    gamma_clnk_frame_avrg_approx_so3_quad
                                )
                                W_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_clnk_frame_avrg_approx_so3_quad
                                )
                                W_flucts_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_flucts_clnk_frame_avrg_approx_so3_quad
                                )
    
    # Generate filenames and save data
    for protocol_indx in range(len(dfrmtn_protocol)):
        protocol_indx_str = "protocol_indx_" + str(protocol_indx)
        dfrmtn_filename = (
            filename_prefix + "-dfrmtn" + "_" + protocol_indx_str + ".npy"
        )
        np.save(dfrmtn_filename, dfrmtn_protocol[protocol_indx])
        
        for k_indx in range(np.shape(k)[0]):
            k_str = "k_" + str(k[k_indx])

            omega_clnks_free_rot_filename = (
                filename_prefix + "-omega_clnks_free_rot"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            omega_clnks_norm_free_rot_filename = (
                filename_prefix + "-omega_clnks_norm_free_rot"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            y_clnks_free_rot_filename = (
                filename_prefix + "-y_clnks_free_rot"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            y_clnks_norm_free_rot_filename = (
                filename_prefix + "-y_clnks_norm_free_rot"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            gamma_clnks_free_rot_filename = (
                filename_prefix + "-gamma_clnks_free_rot"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_free_rot_filename = (
                filename_prefix + "-W_clnks_free_rot"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_flucts_clnks_free_rot_filename = (
                filename_prefix + "-W_flucts_clnks_free_rot"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )

            delta_omega_clnks_free_rot_approx_filename = (
                filename_prefix + "-delta_omega_clnks_free_rot_approx"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            delta_omega_clnks_norm_free_rot_approx_filename = (
                filename_prefix + "-delta_omega_clnks_norm_free_rot_approx"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            delta_y_clnks_free_rot_approx_filename = (
                filename_prefix + "-delta_y_clnks_free_rot_approx"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            delta_y_clnks_norm_free_rot_approx_filename = (
                filename_prefix + "-delta_y_clnks_norm_free_rot_approx"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            gamma_clnks_free_rot_approx_filename = (
                filename_prefix + "-gamma_clnks_free_rot_approx"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_free_rot_approx_filename = (
                filename_prefix + "-W_clnks_free_rot_approx"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_flucts_clnks_free_rot_approx_filename = (
                filename_prefix + "-W_flucts_clnks_free_rot_approx"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )

            y_clnks_frame_avrg_so3_filename = (
                filename_prefix + "-y_clnks_frame_avrg_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            y_clnks_norm_frame_avrg_so3_filename = (
                filename_prefix + "-y_clnks_norm_frame_avrg_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            gamma_clnks_frame_avrg_so3_filename = (
                filename_prefix + "-gamma_clnks_frame_avrg_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_frame_avrg_so3_filename = (
                filename_prefix + "-W_clnks_frame_avrg_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_flucts_clnks_frame_avrg_so3_filename = (
                filename_prefix + "-W_flucts_clnks_frame_avrg_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            y_clnks_frame_avrg_so3_quad_filename = (
                filename_prefix + "-y_clnks_frame_avrg_so3_quad"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            y_clnks_norm_frame_avrg_so3_quad_filename = (
                filename_prefix + "-y_clnks_norm_frame_avrg_so3_quad"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            gamma_clnks_frame_avrg_so3_quad_filename = (
                filename_prefix + "-gamma_clnks_frame_avrg_so3_quad"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_frame_avrg_so3_quad_filename = (
                filename_prefix + "-W_clnks_frame_avrg_so3_quad"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_flucts_clnks_frame_avrg_so3_quad_filename = (
                filename_prefix + "-W_flucts_clnks_frame_avrg_so3_quad"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )

            delta_y_clnks_frame_avrg_approx_so3_filename = (
                filename_prefix + "-delta_y_clnks_frame_avrg_approx_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            delta_y_clnks_norm_frame_avrg_approx_so3_filename = (
                filename_prefix + "-delta_y_clnks_norm_frame_avrg_approx_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            gamma_clnks_frame_avrg_approx_so3_filename = (
                filename_prefix + "-gamma_clnks_frame_avrg_approx_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_frame_avrg_approx_so3_filename = (
                filename_prefix + "-W_clnks_frame_avrg_approx_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_flucts_clnks_frame_avrg_approx_so3_filename = (
                filename_prefix + "-W_flucts_clnks_frame_avrg_approx_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            delta_y_clnks_frame_avrg_approx_so3_quad_filename = (
                filename_prefix + "-delta_y_clnks_frame_avrg_approx_so3_quad"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            delta_y_clnks_norm_frame_avrg_approx_so3_quad_filename = (
                filename_prefix + "-delta_y_clnks_norm_frame_avrg_approx_so3_quad"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            gamma_clnks_frame_avrg_approx_so3_quad_filename = (
                filename_prefix + "-gamma_clnks_frame_avrg_approx_so3_quad"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_frame_avrg_approx_so3_quad_filename = (
                filename_prefix + "-W_clnks_frame_avrg_approx_so3_quad"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_flucts_clnks_frame_avrg_approx_so3_quad_filename = (
                filename_prefix + "-W_flucts_clnks_frame_avrg_approx_so3_quad"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )

            np.save(
                omega_clnks_free_rot_filename,
                omega_clnks_free_rot[protocol_indx][k_indx])
            np.save(
                omega_clnks_norm_free_rot_filename,
                omega_clnks_norm_free_rot[protocol_indx][k_indx])
            np.save(
                y_clnks_free_rot_filename,
                y_clnks_free_rot[protocol_indx][k_indx])
            np.save(
                y_clnks_norm_free_rot_filename,
                y_clnks_norm_free_rot[protocol_indx][k_indx])
            np.save(
                gamma_clnks_free_rot_filename,
                gamma_clnks_free_rot[protocol_indx][k_indx])
            np.save(
                W_clnks_free_rot_filename,
                W_clnks_free_rot[protocol_indx][k_indx])
            np.save(
                W_flucts_clnks_free_rot_filename,
                W_flucts_clnks_free_rot[protocol_indx][k_indx])
            
            np.save(
                delta_omega_clnks_free_rot_approx_filename,
                delta_omega_clnks_free_rot_approx[protocol_indx][k_indx])
            np.save(
                delta_omega_clnks_norm_free_rot_approx_filename,
                delta_omega_clnks_norm_free_rot_approx[protocol_indx][k_indx])
            np.save(
                delta_y_clnks_free_rot_approx_filename,
                delta_y_clnks_free_rot_approx[protocol_indx][k_indx])
            np.save(
                delta_y_clnks_norm_free_rot_approx_filename,
                delta_y_clnks_norm_free_rot_approx[protocol_indx][k_indx])
            np.save(
                gamma_clnks_free_rot_approx_filename,
                gamma_clnks_free_rot_approx[protocol_indx][k_indx])
            np.save(
                W_clnks_free_rot_approx_filename,
                W_clnks_free_rot_approx[protocol_indx][k_indx])
            np.save(
                W_flucts_clnks_free_rot_approx_filename,
                W_flucts_clnks_free_rot_approx[protocol_indx][k_indx])
            
            np.save(
                y_clnks_frame_avrg_so3_filename,
                y_clnks_frame_avrg_so3[protocol_indx][k_indx])
            np.save(
                y_clnks_norm_frame_avrg_so3_filename,
                y_clnks_norm_frame_avrg_so3[protocol_indx][k_indx])
            np.save(
                gamma_clnks_frame_avrg_so3_filename,
                gamma_clnks_frame_avrg_so3[protocol_indx][k_indx])
            np.save(
                W_clnks_frame_avrg_so3_filename,
                W_clnks_frame_avrg_so3[protocol_indx][k_indx])
            np.save(
                W_flucts_clnks_frame_avrg_so3_filename,
                W_flucts_clnks_frame_avrg_so3[protocol_indx][k_indx])
            np.save(
                y_clnks_frame_avrg_so3_quad_filename,
                y_clnks_frame_avrg_so3_quad[protocol_indx][k_indx])
            np.save(
                y_clnks_norm_frame_avrg_so3_quad_filename,
                y_clnks_norm_frame_avrg_so3_quad[protocol_indx][k_indx])
            np.save(
                gamma_clnks_frame_avrg_so3_quad_filename,
                gamma_clnks_frame_avrg_so3_quad[protocol_indx][k_indx])
            np.save(
                W_clnks_frame_avrg_so3_quad_filename,
                W_clnks_frame_avrg_so3_quad[protocol_indx][k_indx])
            np.save(
                W_flucts_clnks_frame_avrg_so3_quad_filename,
                W_flucts_clnks_frame_avrg_so3_quad[protocol_indx][k_indx])
            
            np.save(
                delta_y_clnks_frame_avrg_approx_so3_filename,
                delta_y_clnks_frame_avrg_approx_so3[protocol_indx][k_indx])
            np.save(
                delta_y_clnks_norm_frame_avrg_approx_so3_filename,
                delta_y_clnks_norm_frame_avrg_approx_so3[protocol_indx][k_indx])
            np.save(
                gamma_clnks_frame_avrg_approx_so3_filename,
                gamma_clnks_frame_avrg_approx_so3[protocol_indx][k_indx])
            np.save(
                W_clnks_frame_avrg_approx_so3_filename,
                W_clnks_frame_avrg_approx_so3[protocol_indx][k_indx])
            np.save(
                W_flucts_clnks_frame_avrg_approx_so3_filename,
                W_flucts_clnks_frame_avrg_approx_so3[protocol_indx][k_indx])
            np.save(
                delta_y_clnks_frame_avrg_approx_so3_quad_filename,
                delta_y_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx])
            np.save(
                delta_y_clnks_norm_frame_avrg_approx_so3_quad_filename,
                delta_y_clnks_norm_frame_avrg_approx_so3_quad[protocol_indx][k_indx])
            np.save(
                gamma_clnks_frame_avrg_approx_so3_quad_filename,
                gamma_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx])
            np.save(
                W_clnks_frame_avrg_approx_so3_quad_filename,
                W_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx])
            np.save(
                W_flucts_clnks_frame_avrg_approx_so3_quad_filename,
                W_flucts_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx])
    
    # print("omega_clnks_free_rot = {}".format(omega_clnks_free_rot))
    # print("y_clnks_free_rot = {}".format(y_clnks_free_rot))
    # print("gamma_clnks_free_rot = {}".format(gamma_clnks_free_rot))
    print("W_clnks_free_rot = {}".format(W_clnks_free_rot))
    print("W_flucts_clnks_free_rot = {}".format(W_flucts_clnks_free_rot))

    # print("omega_clnks_norm_free_rot = {}".format(omega_clnks_norm_free_rot))
    # print("y_clnks_norm_free_rot = {}".format(y_clnks_norm_free_rot))


    # print("delta_omega_clnks_free_rot_approx = {}".format(delta_omega_clnks_free_rot_approx))
    # print("delta_y_clnks_free_rot_approx = {}".format(delta_y_clnks_norm_free_rot_approx))
    # print("gamma_clnks_free_rot_approx = {}".format(gamma_clnks_free_rot_approx))
    print("W_clnks_free_rot_approx = {}".format(W_clnks_free_rot_approx))
    print("W_flucts_clnks_free_rot_approx = {}".format(W_flucts_clnks_free_rot_approx))

    # print("delta_omega_clnks_norm_free_rot_approx = {}".format(delta_omega_clnks_norm_free_rot_approx))
    # print("delta_y_clnks_norm_free_rot_approx = {}".format(delta_y_clnks_norm_free_rot_approx))


    # print("y_clnks_frame_avrg_so3 = {}".format(y_clnks_frame_avrg_so3))
    # print("gamma_clnks_frame_avrg_so3 = {}".format(gamma_clnks_frame_avrg_so3))
    # print("W_clnks_frame_avrg_so3 = {}".format(W_clnks_frame_avrg_so3))
    # print("W_flucts_clnks_frame_avrg_so3 = {}".format(W_flucts_clnks_frame_avrg_so3))
    # print("y_clnks_frame_avrg_so3_quad = {}".format(y_clnks_frame_avrg_so3_quad))
    # print("gamma_clnks_frame_avrg_so3_quad = {}".format(gamma_clnks_frame_avrg_so3_quad))
    print("W_clnks_frame_avrg_so3_quad = {}".format(W_clnks_frame_avrg_so3_quad))
    print("W_flucts_clnks_frame_avrg_so3_quad = {}".format(W_flucts_clnks_frame_avrg_so3_quad))

    # print("y_clnks_norm_frame_avrg_so3 = {}".format(y_clnks_norm_frame_avrg_so3))
    # print("y_clnks_norm_frame_avrg_so3_quad = {}".format(y_clnks_norm_frame_avrg_so3_quad))


    # print("delta_y_clnks_frame_avrg_approx_so3 = {}".format(delta_y_clnks_frame_avrg_approx_so3))
    # print("gamma_clnks_frame_avrg_approx_so3 = {}".format(gamma_clnks_frame_avrg_approx_so3))
    # print("W_clnks_frame_avrg_approx_so3 = {}".format(W_clnks_frame_avrg_approx_so3))
    # print("W_flucts_clnks_frame_avrg_approx_so3 = {}".format(W_flucts_clnks_frame_avrg_approx_so3))
    # print("delta_y_clnks_frame_avrg_approx_so3_quad = {}".format(delta_y_clnks_frame_avrg_approx_so3_quad))
    # print("gamma_clnks_frame_avrg_approx_so3_quad = {}".format(gamma_clnks_frame_avrg_approx_so3_quad))
    print("W_clnks_frame_avrg_approx_so3_quad = {}".format(W_clnks_frame_avrg_approx_so3_quad))
    print("W_flucts_clnks_frame_avrg_approx_so3_quad = {}".format(W_flucts_clnks_frame_avrg_approx_so3_quad))

    # print("delta_y_clnks_norm_frame_avrg_approx_so3 = {}".format(delta_y_clnks_norm_frame_avrg_approx_so3))
    # print("delta_y_clnks_norm_frame_avrg_approx_so3_quad = {}".format(delta_y_clnks_norm_frame_avrg_approx_so3_quad))

@hydra.main(
        version_base=None,
        config_path="../configs/clnk_rves",
        config_name="config")
def main(cfg: DictConfig) -> None:
    topology = cfg.topology
    
    sample = 0
    for w_c_args in topology.w_c_args:
        for w_c_dfrmtn_args in topology.w_c_dfrmtn_args:
            for b in topology.b:
                for f in topology.f:
                    for p_n_args in topology.p_n_args:
                        for n in topology.n:
                            clnk_rves_analysis(
                                cfg.label, sample, cfg.deformation,
                                topology.w_c_dist, w_c_args,
                                topology.w_c_dfrmtn_dist,
                                w_c_dfrmtn_args, b, f,
                                p_n_args, n, topology.n_init)
                            sample += 1

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"End-linked polymer network elastically-effective cross-link RVE construction took {execution_time} seconds to run")