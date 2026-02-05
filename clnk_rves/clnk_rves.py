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
from src.helpers.so3_quadrature import master_so3_quadrature_func
from src.helpers.chain_free_energy import (
    master_w_c_func,
    master_w_c_args_func,
    master_w_c_dfrmtn_func,
    master_w_c_dfrmtn_args_func,
    master_dw_c__dy_clnk_func,
    master_dw_c__dy_clnk_args_func,
    master_d2w_c__dy_clnk_dy_clnk_func,
    master_d2w_c__dy_clnk_dy_clnk_args_func
)
from src.helpers.chain_segment_number import n_init_func
from src.helpers.clnk_degree_dispersity import k_init_func
from src.helpers.clnk_structure_dispersity import (
    m_clnks_symmetric_under_chain_permutation_func,
    n_clnks_symmetric_under_chain_permutation_func
)
from src.helpers.chain_stretch import (
    master_gamma_crits_func,
    master_gamma_crit_func,
    master_gamma_rms_args_func,
    master_gamma_rms_func,
    gamma_func
)
from src.helpers.chain_length import r_func
from src.helpers.clnk_structure import (
    recommended_clnk_init_func,
    amended_3_chn_clnk_X_hat_clnk_func,
    regular_tetrahedral_4_chn_clnk_X_hat_clnk_func,
    equilateral_triangular_bipyramidal_5_chn_clnk_X_hat_clnk_func,
    regular_octahedral_6_chn_clnk_X_hat_clnk_func,
    equilateral_pentagonal_bipyramidal_7_chn_clnk_X_hat_clnk_func,
    cube_8_chn_clnk_X_hat_clnk_func,
    x_hat_clnk_func,
    com_x_clnk_func
)
from src.helpers.continuum_mechanics import (
    deformation_protocol_init_func,
    F_func
)
from src.helpers.clnk_free_rotation import (
    monodisperse_clnk_free_rot,
    clnk_free_rot_cnstrnd_mnmztn,
    clnk_free_rot_approx
)
from src.helpers.clnk_frame_averaging import (
    monodisperse_clnk_frame_avrg,
    clnk_frame_avrg_cnstrnd_mnmztn,
    clnk_frame_avrg_approx
)

@hydra.main(
        version_base=None,
        config_path="../configs/clnk_rves",
        config_name="config")
def main(cfg: DictConfig) -> None:
    # Generate filename prefix
    filename_prefix = filename_str(
        cfg.label.workdir, cfg.label.date, cfg.label.batch, cfg.label.sample)
    
    # Create a seeded default random number generator for the
    # differential evolution global constrained minimization
    rng = np.random.default_rng(cfg.deformation.seed)

    # Numerical quadrature scheme
    points, weights = np.polynomial.legendre.leggauss(
        cfg.deformation.num_quad_points)

    # SO3 quadrature scheme
    so3_quad, sph_quad_symmtry = master_so3_quadrature_func(
        cfg.deformation.sph_quad_method, cfg.deformation.num_spin_inc)
    so3_quad_num = np.shape(so3_quad)[0]

    # Gather all possible types of polymer chain stretches
    gamma_crits = master_gamma_crits_func(
        cfg.topology.w_c_dist, cfg.topology.kappa_n, cfg.topology.zeta_n_char)

    # Extract nondimensional polymer chain free energy function,
    # nondimensional polymer chain deformation free energy function,
    # nondimensional derivative of the polymer chain free energy with
    # respect to the cross-link junction position function, and
    # nondimensional second derivative of the polymer chain free energy
    # with respect to the cross-link junction position function
    w_c_func = master_w_c_func(cfg.topology.w_c_dist)
    w_c_args = master_w_c_args_func(
        cfg.topology.w_c_dist, cfg.topology.kappa_n, cfg.topology.zeta_n_char,
        gamma_crits)
    w_c_dfrmtn_func = master_w_c_dfrmtn_func(cfg.topology.w_c_dfrmtn_dist)
    w_c_dfrmtn_args = master_w_c_dfrmtn_args_func(cfg.topology.w_c_dfrmtn_dist)
    dw_c__dy_clnk_func = master_dw_c__dy_clnk_func(cfg.topology.w_c_dist)
    dw_c__dy_clnk_args = master_dw_c__dy_clnk_args_func(
        cfg.topology.w_c_dist, cfg.topology.kappa_n, cfg.topology.zeta_n_char,
        gamma_crits)
    d2w_c__dy_clnk_dy_clnk_func = master_d2w_c__dy_clnk_dy_clnk_func(
        cfg.topology.w_c_dist)
    d2w_c__dy_clnk_dy_clnk_args = master_d2w_c__dy_clnk_dy_clnk_args_func(
        cfg.topology.w_c_dist, cfg.topology.kappa_n, cfg.topology.zeta_n_char,
        gamma_crits)

    # Initialize the salient chain segment numbers, the
    # elastically-effective cross-link degree, the chain segment number
    # multiplicity for each distinct cross-link structure, and the chain
    # segment number for each chain in each distinct cross-link
    # structure
    n, N = n_init_func(cfg.topology.n_init, tuple(cfg.topology.n[0]), int)
    k = k_init_func(cfg.topology.f)
    n_clnks = []
    for k_indx in range(np.shape(k)[0]):
        n_clnks.append(
            n_clnks_symmetric_under_chain_permutation_func(
                n, m_clnks_symmetric_under_chain_permutation_func(k[k_indx], N)))
    # print(n_clnks)

    # Save the chain segment number for each chain in each distinct
    # cross-link structure
    for k_indx in range(np.shape(k)[0]):
        n_clnks_k_filename = (
            filename_prefix + f"-n_clnks_k_{k[k_indx]:d}" + ".dat"
        )
        np.savetxt(n_clnks_k_filename, n_clnks[k_indx], fmt="%d")

    # Calculate the critical absolute/equilibrium polymer chain stretch
    # on a chain-by-chain basis
    gamma_crit_clnks = []
    for k_indx in range(np.shape(k)[0]):
        n_clnks_arr = n_clnks[k_indx]
        gamma_crit_clnks_arr = np.empty_like(n_clnks_arr, dtype=float)
        for chn_indx in np.ndindex(np.shape(n_clnks_arr)):
            gamma_crit_clnks_arr[chn_indx] = master_gamma_crit_func(
                cfg.topology.w_c_dist, cfg.topology.kappa_n,
                cfg.topology.zeta_n_char)
        gamma_crit_clnks.append(gamma_crit_clnks_arr)
    
    # Calculate the critical polymer chain contour length on a
    # chain-by-chain basis
    r_crit_clnks = []
    for k_indx in range(np.shape(k)[0]):
        n_clnks_arr = n_clnks[k_indx]
        gamma_crit_clnks_arr = gamma_crit_clnks[k_indx]
        r_crit_clnks_arr = np.empty_like(n_clnks_arr, dtype=float)
        for chn_indx in np.ndindex(np.shape(n_clnks_arr)):
            r_crit_clnks_arr[chn_indx] = r_func(
                gamma_crit_clnks_arr[chn_indx], n_clnks_arr[chn_indx],
                cfg.topology.b)
        r_crit_clnks.append(r_crit_clnks_arr)
    
    # Calculate the root-mean-square absolute/equilibrium polymer chain
    # stretch on a chain-by-chain basis
    gamma_rms_args = master_gamma_rms_args_func(
        cfg.topology.w_c_dist, cfg.topology.kappa_n, cfg.topology.zeta_n_char,
        gamma_crits)
    gamma_rms_clnks = []
    for k_indx in range(np.shape(k)[0]):
        n_clnks_arr = n_clnks[k_indx]
        gamma_crit_clnks_arr = gamma_crit_clnks[k_indx]
        gamma_rms_clnks_arr = np.empty_like(n_clnks_arr, dtype=float)
        for chn_indx in np.ndindex(np.shape(n_clnks_arr)):
            gamma_rms_clnks_arr[chn_indx] = master_gamma_rms_func(
                points, weights, n_clnks_arr[chn_indx],
                gamma_crit_clnks_arr[chn_indx], cfg.deformation.gamma_n_hat_inc,
                cfg.topology.w_c_dist, w_c_func, w_c_args, gamma_rms_args)
        gamma_rms_clnks.append(gamma_rms_clnks_arr)
    
    # Calculate the root-mean-square polymer chain length on a
    # chain-by-chain basis
    r_rms_clnks = []
    for k_indx in range(np.shape(k)[0]):
        n_clnks_arr = n_clnks[k_indx]
        gamma_rms_clnks_arr = gamma_rms_clnks[k_indx]
        r_rms_clnks_arr = np.empty_like(n_clnks_arr, dtype=float)
        for chn_indx in np.ndindex(np.shape(n_clnks_arr)):
            r_rms_clnks_arr[chn_indx] = r_func(
                gamma_rms_clnks_arr[chn_indx], n_clnks_arr[chn_indx],
                cfg.topology.b)
        r_rms_clnks.append(r_rms_clnks_arr)

    # Initialize the cross-link structures
    X_clnks = []
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
        omega_clnks_init_arr = np.zeros((C_R, 3))
        y_clnks_init_arr = np.zeros((C_R, 3))
        gamma_clnks_init_arr = np.zeros((C_R, k_num))
        delta_omega_clnks_init_arr = np.zeros((C_R, 3))
        delta_y_clnks_init_arr = np.zeros((C_R, 3))
        for clnk_indx in range(C_R):
            n_clnk = n_clnks_arr[clnk_indx]
            X_clnk, omega_clnk_init, y_clnk_init = recommended_clnk_init_func(
                r_rms_clnks_arr[clnk_indx],
                type_8_chn_clnk=cfg.topology.type_8_chn_clnk)
            X_clnks_arr[clnk_indx] = X_clnk
            omega_clnks_init_arr[clnk_indx] = omega_clnk_init
            y_clnks_init_arr[clnk_indx] = y_clnk_init
            r_clnk_init = X_clnk - y_clnk_init
            for chn_indx in range(k_num):
                gamma_clnks_init_arr[clnk_indx, chn_indx] = gamma_func(
                    np.linalg.norm(r_clnk_init[chn_indx]),
                    n_clnk[chn_indx], cfg.topology.b)
        X_clnks.append(X_clnks_arr)
        omega_clnks_init.append(omega_clnks_init_arr)
        y_clnks_init.append(y_clnks_init_arr)
        gamma_clnks_init.append(gamma_clnks_init_arr)
        delta_omega_clnks_init.append(delta_omega_clnks_init_arr)
        delta_y_clnks_init.append(delta_y_clnks_init_arr)
    # print(X_clnks)
    # print(omega_clnks_init)
    # print(y_clnks_init)
    # print(gamma_clnks_init)
    # print(delta_omega_clnks_init)
    # print(delta_y_clnks_init)
    
    # Verify that all initial cross-link positions coincide with the
    # origin
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
    dfrmtn_protocol_class = cfg.deformation.protocol_class
    dfrmtn_protocol = []
    for protocol_indx in range(len(dfrmtn_protocol_class)):
        dfrmtn_protocol.append(
            deformation_protocol_init_func(
                cfg.deformation.protocol_init,
                cfg.deformation.protocol[protocol_indx]))
    # print(dfrmtn_protocol_class)
    # print(dfrmtn_protocol)

    # Data initialization
    omega_clnks_free_rot = []
    omega_clnks_norm_free_rot = []
    y_clnks_free_rot = []
    y_clnks_norm_free_rot = []
    gamma_clnks_free_rot = []
    W_clnks_chns_free_rot = []
    W_clnks_y_flucts_free_rot = []
    W_clnks_free_rot = []

    delta_omega_clnks_free_rot_approx = []
    delta_omega_clnks_norm_free_rot_approx = []
    delta_y_clnks_free_rot_approx = []
    delta_y_clnks_norm_free_rot_approx = []
    gamma_clnks_free_rot_approx = []
    W_clnks_chns_free_rot_approx = []
    W_clnks_y_flucts_free_rot_approx = []
    W_clnks_free_rot_approx = []

    y_clnks_frame_avrg_so3 = []
    y_clnks_norm_frame_avrg_so3 = []
    gamma_clnks_frame_avrg_so3 = []
    W_clnks_chns_frame_avrg_so3 = []
    W_clnks_y_flucts_frame_avrg_so3 = []
    W_clnks_frame_avrg_so3 = []
    y_clnks_frame_avrg_so3_quad = []
    y_clnks_norm_frame_avrg_so3_quad = []
    gamma_clnks_frame_avrg_so3_quad = []
    W_clnks_chns_frame_avrg_so3_quad = []
    W_clnks_y_flucts_frame_avrg_so3_quad = []
    W_clnks_frame_avrg_so3_quad = []

    delta_y_clnks_frame_avrg_approx_so3 = []
    delta_y_clnks_norm_frame_avrg_approx_so3 = []
    gamma_clnks_frame_avrg_approx_so3 = []
    W_clnks_chns_frame_avrg_approx_so3 = []
    W_clnks_y_flucts_frame_avrg_approx_so3 = []
    W_clnks_frame_avrg_approx_so3 = []
    delta_y_clnks_frame_avrg_approx_so3_quad = []
    delta_y_clnks_norm_frame_avrg_approx_so3_quad = []
    gamma_clnks_frame_avrg_approx_so3_quad = []
    W_clnks_chns_frame_avrg_approx_so3_quad = []
    W_clnks_y_flucts_frame_avrg_approx_so3_quad = []
    W_clnks_frame_avrg_approx_so3_quad = []

    for protocol_indx in range(len(dfrmtn_protocol)):
        omega_clnks_free_rot_protocol = []
        omega_clnks_norm_free_rot_protocol = []
        y_clnks_free_rot_protocol = []
        y_clnks_norm_free_rot_protocol = []
        gamma_clnks_free_rot_protocol = []
        W_clnks_chns_free_rot_protocol = []
        W_clnks_y_flucts_free_rot_protocol = []
        W_clnks_free_rot_protocol = []

        delta_omega_clnks_free_rot_approx_protocol = []
        delta_omega_clnks_norm_free_rot_approx_protocol = []
        delta_y_clnks_free_rot_approx_protocol = []
        delta_y_clnks_norm_free_rot_approx_protocol = []
        gamma_clnks_free_rot_approx_protocol = []
        W_clnks_chns_free_rot_approx_protocol = []
        W_clnks_y_flucts_free_rot_approx_protocol = []
        W_clnks_free_rot_approx_protocol = []

        y_clnks_frame_avrg_so3_protocol = []
        y_clnks_norm_frame_avrg_so3_protocol = []
        gamma_clnks_frame_avrg_so3_protocol = []
        W_clnks_chns_frame_avrg_so3_protocol = []
        W_clnks_y_flucts_frame_avrg_so3_protocol = []
        W_clnks_frame_avrg_so3_protocol = []
        y_clnks_frame_avrg_so3_quad_protocol = []
        y_clnks_norm_frame_avrg_so3_quad_protocol = []
        gamma_clnks_frame_avrg_so3_quad_protocol = []
        W_clnks_chns_frame_avrg_so3_quad_protocol = []
        W_clnks_y_flucts_frame_avrg_so3_quad_protocol = []
        W_clnks_frame_avrg_so3_quad_protocol = []

        delta_y_clnks_frame_avrg_approx_so3_protocol = []
        delta_y_clnks_norm_frame_avrg_approx_so3_protocol = []
        gamma_clnks_frame_avrg_approx_so3_protocol = []
        W_clnks_chns_frame_avrg_approx_so3_protocol = []
        W_clnks_y_flucts_frame_avrg_approx_so3_protocol = []
        W_clnks_frame_avrg_approx_so3_protocol = []
        delta_y_clnks_frame_avrg_approx_so3_quad_protocol = []
        delta_y_clnks_norm_frame_avrg_approx_so3_quad_protocol = []
        gamma_clnks_frame_avrg_approx_so3_quad_protocol = []
        W_clnks_chns_frame_avrg_approx_so3_quad_protocol = []
        W_clnks_y_flucts_frame_avrg_approx_so3_quad_protocol = []
        W_clnks_frame_avrg_approx_so3_quad_protocol = []
        
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
            W_clnks_chns_free_rot_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            W_clnks_y_flucts_free_rot_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            W_clnks_free_rot_protocol.append(
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
            W_clnks_chns_free_rot_approx_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            W_clnks_y_flucts_free_rot_approx_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            W_clnks_free_rot_approx_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            
            y_clnks_frame_avrg_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num, 3)))
            y_clnks_norm_frame_avrg_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            gamma_clnks_frame_avrg_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num, k_num)))
            W_clnks_chns_frame_avrg_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            W_clnks_y_flucts_frame_avrg_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            W_clnks_frame_avrg_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            y_clnks_frame_avrg_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, 3)))
            y_clnks_norm_frame_avrg_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            gamma_clnks_frame_avrg_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, k_num)))
            W_clnks_chns_frame_avrg_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            W_clnks_y_flucts_frame_avrg_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            W_clnks_frame_avrg_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            
            delta_y_clnks_frame_avrg_approx_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num, 3)))
            delta_y_clnks_norm_frame_avrg_approx_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            gamma_clnks_frame_avrg_approx_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num, k_num)))
            W_clnks_chns_frame_avrg_approx_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            W_clnks_y_flucts_frame_avrg_approx_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            W_clnks_frame_avrg_approx_so3_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, so3_quad_num)))
            delta_y_clnks_frame_avrg_approx_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, 3)))
            delta_y_clnks_norm_frame_avrg_approx_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            gamma_clnks_frame_avrg_approx_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R, k_num)))
            W_clnks_chns_frame_avrg_approx_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            W_clnks_y_flucts_frame_avrg_approx_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
            W_clnks_frame_avrg_approx_so3_quad_protocol.append(
                np.zeros((num_dfrmtn_steps, C_R)))
        
        omega_clnks_free_rot.append(omega_clnks_free_rot_protocol)
        omega_clnks_norm_free_rot.append(omega_clnks_norm_free_rot_protocol)
        y_clnks_free_rot.append(y_clnks_free_rot_protocol)
        y_clnks_norm_free_rot.append(y_clnks_norm_free_rot_protocol)
        gamma_clnks_free_rot.append(gamma_clnks_free_rot_protocol)
        W_clnks_chns_free_rot.append(W_clnks_chns_free_rot_protocol)
        W_clnks_y_flucts_free_rot.append(W_clnks_y_flucts_free_rot_protocol)
        W_clnks_free_rot.append(W_clnks_free_rot_protocol)

        delta_omega_clnks_free_rot_approx.append(
            delta_omega_clnks_free_rot_approx_protocol)
        delta_omega_clnks_norm_free_rot_approx.append(
            delta_omega_clnks_norm_free_rot_approx_protocol)
        delta_y_clnks_free_rot_approx.append(
            delta_y_clnks_free_rot_approx_protocol)
        delta_y_clnks_norm_free_rot_approx.append(
            delta_y_clnks_norm_free_rot_approx_protocol)
        gamma_clnks_free_rot_approx.append(gamma_clnks_free_rot_approx_protocol)
        W_clnks_chns_free_rot_approx.append(
            W_clnks_chns_free_rot_approx_protocol)
        W_clnks_y_flucts_free_rot_approx.append(
            W_clnks_y_flucts_free_rot_approx_protocol)
        W_clnks_free_rot_approx.append(W_clnks_free_rot_approx_protocol)

        y_clnks_frame_avrg_so3.append(y_clnks_frame_avrg_so3_protocol)
        y_clnks_norm_frame_avrg_so3.append(y_clnks_norm_frame_avrg_so3_protocol)
        gamma_clnks_frame_avrg_so3.append(gamma_clnks_frame_avrg_so3_protocol)
        W_clnks_chns_frame_avrg_so3.append(W_clnks_chns_frame_avrg_so3_protocol)
        W_clnks_y_flucts_frame_avrg_so3.append(
            W_clnks_y_flucts_frame_avrg_so3_protocol)
        W_clnks_frame_avrg_so3.append(W_clnks_frame_avrg_so3_protocol)
        y_clnks_frame_avrg_so3_quad.append(y_clnks_frame_avrg_so3_quad_protocol)
        y_clnks_norm_frame_avrg_so3_quad.append(
            y_clnks_norm_frame_avrg_so3_quad_protocol)
        gamma_clnks_frame_avrg_so3_quad.append(
            gamma_clnks_frame_avrg_so3_quad_protocol)
        W_clnks_chns_frame_avrg_so3_quad.append(
            W_clnks_chns_frame_avrg_so3_quad_protocol)
        W_clnks_y_flucts_frame_avrg_so3_quad.append(
            W_clnks_y_flucts_frame_avrg_so3_quad_protocol)
        W_clnks_frame_avrg_so3_quad.append(W_clnks_frame_avrg_so3_quad_protocol)

        delta_y_clnks_frame_avrg_approx_so3.append(
            delta_y_clnks_frame_avrg_approx_so3_protocol)
        delta_y_clnks_norm_frame_avrg_approx_so3.append(
            delta_y_clnks_norm_frame_avrg_approx_so3_protocol)
        gamma_clnks_frame_avrg_approx_so3.append(
            gamma_clnks_frame_avrg_approx_so3_protocol)
        W_clnks_chns_frame_avrg_approx_so3.append(
            W_clnks_chns_frame_avrg_approx_so3_protocol)
        W_clnks_y_flucts_frame_avrg_approx_so3.append(
            W_clnks_y_flucts_frame_avrg_approx_so3_protocol)
        W_clnks_frame_avrg_approx_so3.append(
            W_clnks_frame_avrg_approx_so3_protocol)
        delta_y_clnks_frame_avrg_approx_so3_quad.append(
            delta_y_clnks_frame_avrg_approx_so3_quad_protocol)
        delta_y_clnks_norm_frame_avrg_approx_so3_quad.append(
            delta_y_clnks_norm_frame_avrg_approx_so3_quad_protocol)
        gamma_clnks_frame_avrg_approx_so3_quad.append(
            gamma_clnks_frame_avrg_approx_so3_quad_protocol)
        W_clnks_chns_frame_avrg_approx_so3_quad.append(
            W_clnks_chns_frame_avrg_approx_so3_quad_protocol)
        W_clnks_y_flucts_frame_avrg_approx_so3_quad.append(
            W_clnks_y_flucts_frame_avrg_approx_so3_quad_protocol)
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
                    com_X_hat_clnk = com_x_clnk_func(X_hat_clnk)
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

                    if (np.allclose(n_clnk, n_clnk[0]*np.ones_like(n_clnk)) and
                        np.allclose(gamma_clnk_init, gamma_clnk_init[0]*np.ones_like(gamma_clnk_init)) and
                        np.allclose(com_X_hat_clnk, np.zeros(3)) and
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
                             W_clnk_chns_star, W_clnk_y_flucts_star,
                             W_clnk_star) = monodisperse_clnk_free_rot(
                                cfg.deformation.eval_W_clnk_chns,
                                cfg.deformation.eval_W_clnk_y_flucts, F, Lmbda,
                                n_clnk, cfg.topology.b, X_clnk, y_clnk_init,
                                gamma_clnk_init, w_c_func, w_c_args,
                                w_c_dfrmtn_func, w_c_dfrmtn_args,
                                d2w_c__dy_clnk_dy_clnk_func,
                                d2w_c__dy_clnk_dy_clnk_args)

                            y_clnks_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star
                            )
                            y_clnks_norm_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                y_clnk_star_norm
                            )
                            gamma_clnks_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                gamma_clnk_star
                            )
                            W_clnks_chns_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_chns_star
                            )
                            W_clnks_y_flucts_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_y_flucts_star
                            )
                            W_clnks_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_star
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
                            W_clnks_chns_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_chns_star
                            )
                            W_clnks_y_flucts_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_y_flucts_star
                            )
                            W_clnks_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_star
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
                             W_clnk_chns_star_frame_avrg_so3,
                             W_clnk_y_flucts_star_frame_avrg_so3,
                             W_clnk_star_frame_avrg_so3,
                             y_clnk_star_frame_avrg_so3_quad,
                             y_clnk_star_norm_frame_avrg_so3_quad,
                             gamma_clnk_star_frame_avrg_so3_quad,
                             W_clnk_chns_star_frame_avrg_so3_quad,
                             W_clnk_y_flucts_star_frame_avrg_so3_quad,
                             W_clnk_star_frame_avrg_so3_quad) = (
                                monodisperse_clnk_frame_avrg(
                                    cfg.deformation.eval_W_clnk_chns,
                                    cfg.deformation.eval_W_clnk_y_flucts, F,
                                    n_clnk, cfg.topology.b, X_clnk, so3_quad,
                                    sph_quad_symmtry, y_clnk_init,
                                    w_c_func, w_c_args,
                                    w_c_dfrmtn_func, w_c_dfrmtn_args,
                                    d2w_c__dy_clnk_dy_clnk_func,
                                    d2w_c__dy_clnk_dy_clnk_args)
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
                            W_clnks_chns_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_chns_star_frame_avrg_so3
                            )
                            W_clnks_y_flucts_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_y_flucts_star_frame_avrg_so3
                            )
                            W_clnks_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_star_frame_avrg_so3
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
                            W_clnks_chns_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_chns_star_frame_avrg_so3_quad
                            )
                            W_clnks_y_flucts_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_y_flucts_star_frame_avrg_so3_quad
                            )
                            W_clnks_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_star_frame_avrg_so3_quad
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
                            W_clnks_chns_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_chns_star_frame_avrg_so3
                            )
                            W_clnks_y_flucts_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_y_flucts_star_frame_avrg_so3
                            )
                            W_clnks_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_star_frame_avrg_so3
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
                            W_clnks_chns_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_chns_star_frame_avrg_so3_quad
                            )
                            W_clnks_y_flucts_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_y_flucts_star_frame_avrg_so3_quad
                            )
                            W_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                W_clnk_star_frame_avrg_so3_quad
                            )
                    
                    # Evaluate the cross-link structure deformation
                    else:
                        # Free rotation constrained minimization
                        (omega_clnk_free_rot, omega_clnk_norm_free_rot,
                         y_clnk_free_rot, y_clnk_norm_free_rot,
                         gamma_clnk_free_rot, W_clnk_chns_free_rot,
                         W_clnk_y_flucts_free_rot, W_clnk_free_rot) = (
                            clnk_free_rot_cnstrnd_mnmztn(
                                cfg.deformation.eval_W_clnk_chns,
                                cfg.deformation.eval_W_clnk_y_flucts,
                                cfg.deformation.cnstrnd_mnmztn_scope,
                                cfg.deformation.cnstrnd_mnmztn_method, rng, F,
                                n_clnk, cfg.topology.b, X_clnk,
                                omega_clnk_free_rot_init, y_clnk_free_rot_init,
                                w_c_func, w_c_args,
                                w_c_dfrmtn_func, w_c_dfrmtn_args,
                                d2w_c__dy_clnk_dy_clnk_func,
                                d2w_c__dy_clnk_dy_clnk_args)
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
                        W_clnks_chns_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_clnk_chns_free_rot
                        )
                        W_clnks_y_flucts_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_clnk_y_flucts_free_rot
                        )
                        W_clnks_free_rot[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_clnk_free_rot
                        )

                        # Frame averaging constrained minimization
                        (y_clnk_frame_avrg_so3, y_clnk_norm_frame_avrg_so3,
                         gamma_clnk_frame_avrg_so3, W_clnk_chns_frame_avrg_so3,
                         W_clnk_y_flucts_frame_avrg_so3, W_clnk_frame_avrg_so3,
                         y_clnk_frame_avrg_so3_quad,
                         y_clnk_norm_frame_avrg_so3_quad,
                         gamma_clnk_frame_avrg_so3_quad,
                         W_clnk_chns_frame_avrg_so3_quad,
                         W_clnk_y_flucts_frame_avrg_so3_quad,
                         W_clnk_frame_avrg_so3_quad) = (
                            clnk_frame_avrg_cnstrnd_mnmztn(
                                cfg.deformation.eval_W_clnk_chns,
                                cfg.deformation.eval_W_clnk_y_flucts,
                                cfg.deformation.cnstrnd_mnmztn_scope,
                                cfg.deformation.cnstrnd_mnmztn_method, rng, F,
                                n_clnk, cfg.topology.b, X_clnk, so3_quad,
                                sph_quad_symmtry, y_clnk_frame_avrg_so3_init,
                                w_c_func, w_c_args,
                                w_c_dfrmtn_func, w_c_dfrmtn_args,
                                d2w_c__dy_clnk_dy_clnk_func,
                                d2w_c__dy_clnk_dy_clnk_args)
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
                        W_clnks_chns_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_clnk_chns_frame_avrg_so3
                        )
                        W_clnks_y_flucts_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_clnk_y_flucts_frame_avrg_so3
                        )
                        W_clnks_frame_avrg_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_clnk_frame_avrg_so3
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
                        W_clnks_chns_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_clnk_chns_frame_avrg_so3_quad
                        )
                        W_clnks_y_flucts_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_clnk_y_flucts_frame_avrg_so3_quad
                        )
                        W_clnks_frame_avrg_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                            W_clnk_frame_avrg_so3_quad
                        )
                        
                        if not np.allclose(n_clnk, n_clnk[0]*np.ones_like(n_clnk)):
                            # Free rotation approximation
                            clnk_approx = False
                            if k_num == 4:
                                clnk_approx = np.allclose(
                                    X_hat_clnk,
                                    regular_tetrahedral_4_chn_clnk_X_hat_clnk)
                            if clnk_approx:
                                (_, delta_omega_clnk_free_rot_approx,
                                 delta_omega_clnk_norm_free_rot_approx,
                                 delta_y_clnk_free_rot_approx,
                                 delta_y_clnk_norm_free_rot_approx,
                                 gamma_clnk_free_rot_approx,
                                 W_clnk_chns_free_rot_approx,
                                 W_clnk_y_flucts_free_rot_approx,
                                 W_clnk_free_rot_approx) = clnk_free_rot_approx(
                                    cfg.deformation.eval_W_clnk_chns,
                                    cfg.deformation.eval_W_clnk_y_flucts,
                                    cfg.deformation.use_inext_gaussian_fjc_delta_clnk,
                                    F, n_clnk, cfg.topology.b, X_clnk,
                                    w_c_func, w_c_args,
                                    w_c_dfrmtn_func, w_c_dfrmtn_args,
                                    d2w_c__dy_clnk_dy_clnk_func,
                                    d2w_c__dy_clnk_dy_clnk_args)
                                
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
                                W_clnks_chns_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_clnk_chns_free_rot_approx
                                )
                                W_clnks_y_flucts_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_clnk_y_flucts_free_rot_approx
                                )
                                W_clnks_free_rot_approx[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_clnk_free_rot_approx
                                )
                            
                            # Frame averaging approximation
                            if np.allclose(com_X_hat_clnk, np.zeros(3)):
                                (delta_y_clnk_frame_avrg_approx_so3,
                                 delta_y_clnk_norm_frame_avrg_approx_so3,
                                 gamma_clnk_frame_avrg_approx_so3,
                                 W_clnk_chns_frame_avrg_approx_so3,
                                 W_clnk_y_flucts_frame_avrg_approx_so3,
                                 W_clnk_frame_avrg_approx_so3,
                                 delta_y_clnk_frame_avrg_approx_so3_quad,
                                 delta_y_clnk_norm_frame_avrg_approx_so3_quad,
                                 gamma_clnk_frame_avrg_approx_so3_quad,
                                 W_clnk_chns_frame_avrg_approx_so3_quad,
                                 W_clnk_y_flucts_frame_avrg_approx_so3_quad,
                                 W_clnk_frame_avrg_approx_so3_quad) = (
                                    clnk_frame_avrg_approx(
                                        cfg.deformation.eval_W_clnk_chns,
                                        cfg.deformation.eval_W_clnk_y_flucts,
                                        cfg.deformation.use_inext_gaussian_fjc_delta_clnk,
                                        F, n_clnk, cfg.topology.b, X_clnk,
                                        so3_quad, sph_quad_symmtry,
                                        w_c_func, w_c_args,
                                        w_c_dfrmtn_func, w_c_dfrmtn_args,
                                        dw_c__dy_clnk_func, dw_c__dy_clnk_args,
                                        d2w_c__dy_clnk_dy_clnk_func,
                                        d2w_c__dy_clnk_dy_clnk_args)
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
                                W_clnks_chns_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_clnk_chns_frame_avrg_approx_so3
                                )
                                W_clnks_y_flucts_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_clnk_y_flucts_frame_avrg_approx_so3
                                )
                                W_clnks_frame_avrg_approx_so3[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_clnk_frame_avrg_approx_so3
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
                                W_clnks_chns_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_clnk_chns_frame_avrg_approx_so3_quad
                                )
                                W_clnks_y_flucts_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_clnk_y_flucts_frame_avrg_approx_so3_quad
                                )
                                W_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx][dfrmtn_step, clnk_indx] = (
                                    W_clnk_frame_avrg_approx_so3_quad
                                )
    
    # Generate filenames and save data
    for protocol_indx in range(len(dfrmtn_protocol)):
        protocol_indx_str = f"protocol_indx_{protocol_indx:d}"
        dfrmtn_filename = (
            filename_prefix + "-dfrmtn" + "_" + protocol_indx_str + ".npy"
        )
        np.save(dfrmtn_filename, dfrmtn_protocol[protocol_indx])
        
        for k_indx in range(np.shape(k)[0]):
            k_str = f"k_{k[k_indx]:d}"

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
            W_clnks_chns_free_rot_filename = (
                filename_prefix + "-W_clnks_chns_free_rot"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_y_flucts_free_rot_filename = (
                filename_prefix + "-W_clnks_y_flucts_free_rot"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_free_rot_filename = (
                filename_prefix + "-W_clnks_free_rot"
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
            W_clnks_chns_free_rot_approx_filename = (
                filename_prefix + "-W_clnks_chns_free_rot_approx"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_y_flucts_free_rot_approx_filename = (
                filename_prefix + "-W_clnks_y_flucts_free_rot_approx"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_free_rot_approx_filename = (
                filename_prefix + "-W_clnks_free_rot_approx"
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
            W_clnks_chns_frame_avrg_so3_filename = (
                filename_prefix + "-W_clnks_chns_frame_avrg_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_y_flucts_frame_avrg_so3_filename = (
                filename_prefix + "-W_clnks_y_flucts_frame_avrg_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_frame_avrg_so3_filename = (
                filename_prefix + "-W_clnks_frame_avrg_so3"
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
            W_clnks_chns_frame_avrg_so3_quad_filename = (
                filename_prefix + "-W_clnks_chns_frame_avrg_so3_quad"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_y_flucts_frame_avrg_so3_quad_filename = (
                filename_prefix + "-W_clnks_y_flucts_frame_avrg_so3_quad"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_frame_avrg_so3_quad_filename = (
                filename_prefix + "-W_clnks_frame_avrg_so3_quad"
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
            W_clnks_chns_frame_avrg_approx_so3_filename = (
                filename_prefix + "-W_clnks_chns_frame_avrg_approx_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_y_flucts_frame_avrg_approx_so3_filename = (
                filename_prefix + "-W_clnks_y_flucts_frame_avrg_approx_so3"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_frame_avrg_approx_so3_filename = (
                filename_prefix + "-W_clnks_frame_avrg_approx_so3"
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
            W_clnks_y_flucts_frame_avrg_approx_so3_quad_filename = (
                filename_prefix + "-W_clnks_y_flucts_frame_avrg_approx_so3_quad"
                + "_" + protocol_indx_str + "_" + k_str + ".npy"
            )
            W_clnks_chns_frame_avrg_approx_so3_quad_filename = (
                filename_prefix + "-W_clnks_chns_frame_avrg_approx_so3_quad"
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
                W_clnks_chns_free_rot_filename,
                W_clnks_chns_free_rot[protocol_indx][k_indx])
            np.save(
                W_clnks_y_flucts_free_rot_filename,
                W_clnks_y_flucts_free_rot[protocol_indx][k_indx])
            np.save(
                W_clnks_free_rot_filename,
                W_clnks_free_rot[protocol_indx][k_indx])
            
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
                W_clnks_chns_free_rot_approx_filename,
                W_clnks_chns_free_rot_approx[protocol_indx][k_indx])
            np.save(
                W_clnks_y_flucts_free_rot_approx_filename,
                W_clnks_y_flucts_free_rot_approx[protocol_indx][k_indx])
            np.save(
                W_clnks_free_rot_approx_filename,
                W_clnks_free_rot_approx[protocol_indx][k_indx])
            
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
                W_clnks_chns_frame_avrg_so3_filename,
                W_clnks_chns_frame_avrg_so3[protocol_indx][k_indx])
            np.save(
                W_clnks_y_flucts_frame_avrg_so3_filename,
                W_clnks_y_flucts_frame_avrg_so3[protocol_indx][k_indx])
            np.save(
                W_clnks_frame_avrg_so3_filename,
                W_clnks_frame_avrg_so3[protocol_indx][k_indx])
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
                W_clnks_chns_frame_avrg_so3_quad_filename,
                W_clnks_chns_frame_avrg_so3_quad[protocol_indx][k_indx])
            np.save(
                W_clnks_y_flucts_frame_avrg_so3_quad_filename,
                W_clnks_y_flucts_frame_avrg_so3_quad[protocol_indx][k_indx])
            np.save(
                W_clnks_frame_avrg_so3_quad_filename,
                W_clnks_frame_avrg_so3_quad[protocol_indx][k_indx])
            
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
                W_clnks_chns_frame_avrg_approx_so3_filename,
                W_clnks_chns_frame_avrg_approx_so3[protocol_indx][k_indx])
            np.save(
                W_clnks_y_flucts_frame_avrg_approx_so3_filename,
                W_clnks_y_flucts_frame_avrg_approx_so3[protocol_indx][k_indx])
            np.save(
                W_clnks_frame_avrg_approx_so3_filename,
                W_clnks_frame_avrg_approx_so3[protocol_indx][k_indx])
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
                W_clnks_chns_frame_avrg_approx_so3_quad_filename,
                W_clnks_chns_frame_avrg_approx_so3_quad[protocol_indx][k_indx])
            np.save(
                W_clnks_y_flucts_frame_avrg_approx_so3_quad_filename,
                W_clnks_y_flucts_frame_avrg_approx_so3_quad[protocol_indx][k_indx])
            np.save(
                W_clnks_frame_avrg_approx_so3_quad_filename,
                W_clnks_frame_avrg_approx_so3_quad[protocol_indx][k_indx])
    
    # print("omega_clnks_free_rot = {}".format(omega_clnks_free_rot))
    # print("y_clnks_free_rot = {}".format(y_clnks_free_rot))
    # print("gamma_clnks_free_rot = {}".format(gamma_clnks_free_rot))
    print("W_clnks_chns_free_rot = {}".format(W_clnks_chns_free_rot))
    print("W_clnks_y_flucts_free_rot = {}".format(W_clnks_y_flucts_free_rot))
    print("W_clnks_free_rot = {}".format(W_clnks_free_rot))

    # print("omega_clnks_norm_free_rot = {}".format(omega_clnks_norm_free_rot))
    # print("y_clnks_norm_free_rot = {}".format(y_clnks_norm_free_rot))


    # print("delta_omega_clnks_free_rot_approx = {}".format(delta_omega_clnks_free_rot_approx))
    # print("delta_y_clnks_free_rot_approx = {}".format(delta_y_clnks_norm_free_rot_approx))
    # print("gamma_clnks_free_rot_approx = {}".format(gamma_clnks_free_rot_approx))
    print("W_clnks_chns_free_rot_approx = {}".format(W_clnks_chns_free_rot_approx))
    print("W_clnks_y_flucts_free_rot_approx = {}".format(W_clnks_y_flucts_free_rot_approx))
    print("W_clnks_free_rot_approx = {}".format(W_clnks_free_rot_approx))

    # print("delta_omega_clnks_norm_free_rot_approx = {}".format(delta_omega_clnks_norm_free_rot_approx))
    # print("delta_y_clnks_norm_free_rot_approx = {}".format(delta_y_clnks_norm_free_rot_approx))


    # print("y_clnks_frame_avrg_so3 = {}".format(y_clnks_frame_avrg_so3))
    # print("gamma_clnks_frame_avrg_so3 = {}".format(gamma_clnks_frame_avrg_so3))
    # print("W_clnks_chns_frame_avrg_so3 = {}".format(W_clnks_chns_frame_avrg_so3))
    # print("W_clnks_y_flucts_frame_avrg_so3 = {}".format(W_clnks_y_flucts_frame_avrg_so3))
    # print("W_clnks_frame_avrg_so3 = {}".format(W_clnks_frame_avrg_so3))
    # print("y_clnks_frame_avrg_so3_quad = {}".format(y_clnks_frame_avrg_so3_quad))
    # print("gamma_clnks_frame_avrg_so3_quad = {}".format(gamma_clnks_frame_avrg_so3_quad))
    print("W_clnks_chns_frame_avrg_so3_quad = {}".format(W_clnks_chns_frame_avrg_so3_quad))
    print("W_clnks_y_flucts_frame_avrg_so3_quad = {}".format(W_clnks_y_flucts_frame_avrg_so3_quad))
    print("W_clnks_frame_avrg_so3_quad = {}".format(W_clnks_frame_avrg_so3_quad))

    # print("y_clnks_norm_frame_avrg_so3 = {}".format(y_clnks_norm_frame_avrg_so3))
    # print("y_clnks_norm_frame_avrg_so3_quad = {}".format(y_clnks_norm_frame_avrg_so3_quad))


    # print("delta_y_clnks_frame_avrg_approx_so3 = {}".format(delta_y_clnks_frame_avrg_approx_so3))
    # print("gamma_clnks_frame_avrg_approx_so3 = {}".format(gamma_clnks_frame_avrg_approx_so3))
    # print("W_clnks_chns_frame_avrg_approx_so3 = {}".format(W_clnks_chns_frame_avrg_approx_so3))
    # print("W_clnks_y_flucts_frame_avrg_approx_so3 = {}".format(W_clnks_y_flucts_frame_avrg_approx_so3))
    # print("W_clnks_frame_avrg_approx_so3 = {}".format(W_clnks_frame_avrg_approx_so3))
    # print("delta_y_clnks_frame_avrg_approx_so3_quad = {}".format(delta_y_clnks_frame_avrg_approx_so3_quad))
    # print("gamma_clnks_frame_avrg_approx_so3_quad = {}".format(gamma_clnks_frame_avrg_approx_so3_quad))
    print("W_clnks_chns_frame_avrg_approx_so3_quad = {}".format(W_clnks_chns_frame_avrg_approx_so3_quad))
    print("W_clnks_y_flucts_frame_avrg_approx_so3_quad = {}".format(W_clnks_y_flucts_frame_avrg_approx_so3_quad))
    print("W_clnks_frame_avrg_approx_so3_quad = {}".format(W_clnks_frame_avrg_approx_so3_quad))

    # print("delta_y_clnks_norm_frame_avrg_approx_so3 = {}".format(delta_y_clnks_norm_frame_avrg_approx_so3))
    # print("delta_y_clnks_norm_frame_avrg_approx_so3_quad = {}".format(delta_y_clnks_norm_frame_avrg_approx_so3_quad))

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"End-linked polymer network elastically-effective cross-link RVE analysis took {execution_time} seconds to run")