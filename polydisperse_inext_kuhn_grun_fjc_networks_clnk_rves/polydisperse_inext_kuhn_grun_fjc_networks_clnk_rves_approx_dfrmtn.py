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
from src.helpers.chain_segment_number import n_init_func
from src.helpers.chain_stretch import (
    master_gamma_crits_func,
    master_gamma_crit_func,
    master_gamma_rms_args_func,
    master_gamma_rms_func,
    gamma_func
)
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
from src.helpers.chain_length import r_func
from src.helpers.clnk_structure import (
    recommended_clnk_init_func,
    x_hat_clnk_func
)
from src.helpers.clnk_structure_dispersity import (
    geometrically_isomorphic_set_clnks_from_clnk_assembly
)
from src.helpers.continuum_mechanics import (
    deformation_protocol_init_func,
    F_func
)
from src.helpers.clnk_frame_averaging import (
    clnk_frame_avrg_approx,
    W_clnk_chns_frame_avrg_so3_quad_eval,
    W_clnk_y_flucts_frame_avrg_so3_quad_eval
)
from src.helpers.clnk_free_rotation import clnk_free_rot_approx

@hydra.main(
        version_base=None,
        config_path="../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves",
        config_name="config")
def main(cfg: DictConfig) -> None:
    # Generate filename prefix
    filename_prefix = filename_str(
        cfg.label.workdir, cfg.label.date, cfg.label.batch, cfg.label.sample)

    # Numerical quadrature scheme
    points, weights = np.polynomial.legendre.leggauss(
        cfg.deformation.num_quad_points)

    # SO3 quadrature scheme
    so3_quad, sph_quad_symmtry = master_so3_quadrature_func(
        cfg.deformation.sph_quad_method, cfg.deformation.num_spin_inc)

    # Initialize and save the salient chain segment numbers in the
    # cross-link structures
    n_clnks, _ = n_init_func(cfg.topology.n_init, tuple(cfg.topology.n))
    n_clnks_filename = filename_prefix + "-n_clnks" + ".npy"
    np.save(n_clnks_filename, n_clnks)

    # Initialize the monomer length, nondimensional segment stiffness,
    # nondimensional characteristic segment potential energy scale, and
    # the polymer chain model string for each chain
    b_clnks = np.empty_like(n_clnks)
    kappa_n_clnks = np.empty_like(n_clnks)
    zeta_n_char_clnks = np.empty_like(n_clnks)
    w_c_dist_clnks = np.empty_like(n_clnks, dtype="U19")
    for chn_indx in np.ndindex(np.shape(n_clnks)):
        b_clnks[chn_indx] = cfg.topology.b
        kappa_n_clnks[chn_indx] = cfg.topology.kappa_n
        zeta_n_char_clnks[chn_indx] = cfg.topology.zeta_n_char
        w_c_dist_clnks[chn_indx] = cfg.topology.w_c_dist
    
    # Gather all possible types of critical/fundamental
    # absolute/equilibrium chain stretches for each chain
    gamma_crits_clnks = np.empty_like(n_clnks, dtype=object)
    for chn_indx in np.ndindex(np.shape(n_clnks)):
        gamma_crits_clnks[chn_indx] = master_gamma_crits_func(
            w_c_dist_clnks[chn_indx], kappa_n_clnks[chn_indx],
            zeta_n_char_clnks[chn_indx])
    
    # Extract the nondimensional polymer chain free energy function and
    # the nondimensional first and second derivatives of the polymer
    # chain free energy with respect to the cross-link junction position
    # function for each chain
    w_c_func_clnks = np.empty_like(n_clnks, dtype=object)
    w_c_args_clnks = np.empty_like(n_clnks, dtype=object)
    dw_c__dy_clnk_func_clnks = np.empty_like(n_clnks, dtype=object)
    dw_c__dy_clnk_args_clnks = np.empty_like(n_clnks, dtype=object)
    d2w_c__dy_clnk_dy_clnk_func_clnks = np.empty_like(n_clnks, dtype=object)
    d2w_c__dy_clnk_dy_clnk_args_clnks = np.empty_like(n_clnks, dtype=object)
    for chn_indx in np.ndindex(np.shape(n_clnks)):
        w_c_func_clnks[chn_indx] = master_w_c_func(w_c_dist_clnks[chn_indx])
        w_c_args_clnks[chn_indx] = master_w_c_args_func(
            w_c_dist_clnks[chn_indx], kappa_n_clnks[chn_indx],
            zeta_n_char_clnks[chn_indx], gamma_crits_clnks[chn_indx])
        dw_c__dy_clnk_func_clnks[chn_indx] = master_dw_c__dy_clnk_func(
            w_c_dist_clnks[chn_indx])
        dw_c__dy_clnk_args_clnks[chn_indx] = master_dw_c__dy_clnk_args_func(
            w_c_dist_clnks[chn_indx], kappa_n_clnks[chn_indx],
            zeta_n_char_clnks[chn_indx], gamma_crits_clnks[chn_indx])
        d2w_c__dy_clnk_dy_clnk_func_clnks[chn_indx] = (
            master_d2w_c__dy_clnk_dy_clnk_func(w_c_dist_clnks[chn_indx])
        )
        d2w_c__dy_clnk_dy_clnk_args_clnks[chn_indx] = (
            master_d2w_c__dy_clnk_dy_clnk_args_func(
                w_c_dist_clnks[chn_indx], kappa_n_clnks[chn_indx],
                zeta_n_char_clnks[chn_indx], gamma_crits_clnks[chn_indx])
        )
    
    # Initialize the polymer chain deformation free energy function
    # string for each chain
    w_c_dfrmtn_dist_clnks = np.empty_like(n_clnks, dtype="U26")
    for chn_indx in np.ndindex(np.shape(n_clnks)):
        w_c_dfrmtn_dist_clnks[chn_indx] = cfg.topology.w_c_dfrmtn_dist
    
    # Extract the nondimensional polymer chain deformation free energy
    # function for each chain
    w_c_dfrmtn_func_clnks = np.empty_like(n_clnks, dtype=object)
    w_c_dfrmtn_args_clnks = np.empty_like(n_clnks, dtype=object)
    for chn_indx in np.ndindex(np.shape(n_clnks)):
        w_c_dfrmtn_func_clnks[chn_indx] = master_w_c_dfrmtn_func(
            w_c_dfrmtn_dist_clnks[chn_indx])
        w_c_dfrmtn_args_clnks[chn_indx] = master_w_c_dfrmtn_args_func(
            w_c_dfrmtn_dist_clnks[chn_indx])
    
    # Calculate the critical absolute/equilibrium polymer chain stretch
    # for each chain
    gamma_crit_clnks = np.empty_like(n_clnks)
    for chn_indx in np.ndindex(np.shape(n_clnks)):
        gamma_crit_clnks[chn_indx] = master_gamma_crit_func(
            w_c_dist_clnks[chn_indx], kappa_n_clnks[chn_indx],
            zeta_n_char_clnks[chn_indx])
    
    # Calculate the critical polymer chain contour length for each chain
    r_crit_clnks = np.empty_like(n_clnks)
    for chn_indx in np.ndindex(np.shape(n_clnks)):
        r_crit_clnks[chn_indx] = r_func(
            gamma_crit_clnks[chn_indx], n_clnks[chn_indx], b_clnks[chn_indx])
    
    # Calculate the root-mean-square absolute/equilibrium polymer chain
    # stretch for each chain
    gamma_rms_clnks = np.empty_like(n_clnks)
    for chn_indx in np.ndindex(np.shape(n_clnks)):
        gamma_rms_args = master_gamma_rms_args_func(
            w_c_dist_clnks[chn_indx], kappa_n_clnks[chn_indx],
            zeta_n_char_clnks[chn_indx], gamma_crits_clnks[chn_indx])
        gamma_rms_clnks[chn_indx] = master_gamma_rms_func(
            points, weights, n_clnks[chn_indx], gamma_crit_clnks[chn_indx],
            cfg.deformation.gamma_n_hat_inc, w_c_dist_clnks[chn_indx],
            w_c_func_clnks[chn_indx], w_c_args_clnks[chn_indx], gamma_rms_args)
    
    # Calculate the root-mean-square polymer chain length on a
    # chain-by-chain basis
    r_rms_clnks = np.empty_like(n_clnks)
    for chn_indx in np.ndindex(np.shape(n_clnks)):
        r_rms_clnks[chn_indx] = r_func(
            gamma_rms_clnks[chn_indx], n_clnks[chn_indx], b_clnks[chn_indx])

    # Initialize the cross-link structures
    clnks_num, k_num = np.shape(n_clnks)
    X_clnks = np.zeros((clnks_num, k_num, 3))
    omega_clnks_init = np.zeros((clnks_num, 3))
    y_clnks_init = np.zeros((clnks_num, 3))
    gamma_clnks_init = np.zeros((clnks_num, k_num))
    for clnk_indx in range(clnks_num):
        X_clnk, omega_clnk_init, y_clnk_init = recommended_clnk_init_func(
            r_rms_clnks[clnk_indx], type_8_chn_clnk=cfg.topology.type_8_chn_clnk)
        X_clnks[clnk_indx] = X_clnk
        omega_clnks_init[clnk_indx] = omega_clnk_init
        y_clnks_init[clnk_indx] = y_clnk_init
        r_clnk_init = X_clnk - y_clnk_init
        for chn_indx in range(k_num):
            gamma_clnks_init[clnk_indx, chn_indx] = gamma_func(
                np.linalg.norm(r_clnk_init[chn_indx]),
                n_clnks[clnk_indx, chn_indx], b_clnks[clnk_indx, chn_indx])
    
    # Verify that all initial cross-link positions coincide with the
    # origin
    for clnk_indx in range(np.shape(y_clnks_init)[0]):
        if not np.allclose(y_clnks_init[clnk_indx], np.zeros(3)):
            error_str = (
                "The initial position of a cross-link is not at the "
                + "origin."
            )
            raise ValueError(error_str)
    
    # Initialize the geometrically isomorphic sets of cross-link
    # structures associated with each base cross-link structure
    clnks_num = geo_isomrphc_sets_num = np.shape(n_clnks)[0]
    n_clnks_geo_isomrphc_sets = []
    for clnk_indx in range(clnks_num):
        n_clnks_geo_isomrphc_set = (
            geometrically_isomorphic_set_clnks_from_clnk_assembly(
                n_clnks[clnk_indx], y_clnks_init[clnk_indx],
                x_hat_clnk_func(X_clnks[clnk_indx]))
        )
        n_clnks_geo_isomrphc_sets.append(n_clnks_geo_isomrphc_set)
        n_clnks_geo_isomrphc_set_filename = (
            filename_prefix + f"-n_clnks_geo_isomrphc_set_indx_{clnk_indx:d}"
            + ".npy"
        )
        np.save(
            n_clnks_geo_isomrphc_set_filename, n_clnks_geo_isomrphc_set)
    
    # Initialize the monomer length, nondimensional segment stiffness,
    # nondimensional characteristic segment potential energy scale, and
    # the polymer chain model string for each chain in each
    # geometrically isomorphic set of cross-link structures
    b_clnks_geo_isomrphc_sets = []
    kappa_n_clnks_geo_isomrphc_sets = []
    zeta_n_char_clnks_geo_isomrphc_sets = []
    w_c_dist_clnks_geo_isomrphc_sets = []
    for set_indx in range(geo_isomrphc_sets_num):
        n_clnks_geo_isomrphc_set = n_clnks_geo_isomrphc_sets[set_indx]
        b_clnks_geo_isomrphc_set = np.empty_like(n_clnks_geo_isomrphc_set)
        kappa_n_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set)
        zeta_n_char_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set)
        w_c_dist_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set, dtype="U19")
        for chn_indx in np.ndindex(np.shape(n_clnks_geo_isomrphc_set)):
            b_clnks_geo_isomrphc_set[chn_indx] = cfg.topology.b
            kappa_n_clnks_geo_isomrphc_set[chn_indx] = cfg.topology.kappa_n
            zeta_n_char_clnks_geo_isomrphc_set[chn_indx] = (
                cfg.topology.zeta_n_char
            )
            w_c_dist_clnks_geo_isomrphc_set[chn_indx] = cfg.topology.w_c_dist
        b_clnks_geo_isomrphc_sets.append(b_clnks_geo_isomrphc_set)
        kappa_n_clnks_geo_isomrphc_sets.append(
            kappa_n_clnks_geo_isomrphc_set)
        zeta_n_char_clnks_geo_isomrphc_sets.append(
            zeta_n_char_clnks_geo_isomrphc_set)
        w_c_dist_clnks_geo_isomrphc_sets.append(
            w_c_dist_clnks_geo_isomrphc_set)
    
    # Gather all possible types of critical/fundamental
    # absolute/equilibrium chain stretches for each chain in each
    # geometrically isomorphic set of cross-link structures
    gamma_crits_clnks_geo_isomrphc_sets = []
    for set_indx in range(geo_isomrphc_sets_num):
        n_clnks_geo_isomrphc_set = n_clnks_geo_isomrphc_sets[set_indx]
        gamma_crits_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set, dtype=object)
        for chn_indx in np.ndindex(np.shape(n_clnks_geo_isomrphc_set)):
            gamma_crits_clnks_geo_isomrphc_set[chn_indx] = (
                master_gamma_crits_func(
                    w_c_dist_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    kappa_n_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    zeta_n_char_clnks_geo_isomrphc_sets[set_indx][chn_indx])
            )
        gamma_crits_clnks_geo_isomrphc_sets.append(
            gamma_crits_clnks_geo_isomrphc_set)
    
    # Extract the nondimensional polymer chain free energy function and
    # the nondimensional first and second derivatives of the polymer
    # chain free energy with respect to the cross-link junction position
    # function for each chain in each geometrically isomorphic set of
    # cross-link structures
    w_c_func_clnks_geo_isomrphc_sets = []
    w_c_args_clnks_geo_isomrphc_sets = []
    dw_c__dy_clnk_func_clnks_geo_isomrphc_sets = []
    dw_c__dy_clnk_args_clnks_geo_isomrphc_sets = []
    d2w_c__dy_clnk_dy_clnk_func_clnks_geo_isomrphc_sets = []
    d2w_c__dy_clnk_dy_clnk_args_clnks_geo_isomrphc_sets = []
    for set_indx in range(geo_isomrphc_sets_num):
        n_clnks_geo_isomrphc_set = n_clnks_geo_isomrphc_sets[set_indx]
        w_c_func_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set, dtype=object)
        w_c_args_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set, dtype=object)
        dw_c__dy_clnk_func_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set, dtype=object)
        dw_c__dy_clnk_args_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set, dtype=object)
        d2w_c__dy_clnk_dy_clnk_func_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set, dtype=object)
        d2w_c__dy_clnk_dy_clnk_args_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set, dtype=object)
        for chn_indx in np.ndindex(np.shape(n_clnks_geo_isomrphc_set)):
            w_c_func_clnks_geo_isomrphc_set[chn_indx] = master_w_c_func(
                w_c_dist_clnks_geo_isomrphc_sets[set_indx][chn_indx])
            w_c_args_clnks_geo_isomrphc_set[chn_indx] = master_w_c_args_func(
                w_c_dist_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                kappa_n_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                zeta_n_char_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                gamma_crits_clnks_geo_isomrphc_sets[set_indx][chn_indx])
            dw_c__dy_clnk_func_clnks_geo_isomrphc_set[chn_indx] = (
                master_dw_c__dy_clnk_func(
                    w_c_dist_clnks_geo_isomrphc_sets[set_indx][chn_indx])
            )
            dw_c__dy_clnk_args_clnks_geo_isomrphc_set[chn_indx] = (
                master_dw_c__dy_clnk_args_func(
                    w_c_dist_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    kappa_n_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    zeta_n_char_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    gamma_crits_clnks_geo_isomrphc_sets[set_indx][chn_indx])
            )
            d2w_c__dy_clnk_dy_clnk_func_clnks_geo_isomrphc_set[chn_indx] = (
                master_d2w_c__dy_clnk_dy_clnk_func(
                    w_c_dist_clnks_geo_isomrphc_sets[set_indx][chn_indx])
            )
            d2w_c__dy_clnk_dy_clnk_args_clnks_geo_isomrphc_set[chn_indx] = (
                master_d2w_c__dy_clnk_dy_clnk_args_func(
                    w_c_dist_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    kappa_n_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    zeta_n_char_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    gamma_crits_clnks_geo_isomrphc_sets[set_indx][chn_indx])
            )
        w_c_func_clnks_geo_isomrphc_sets.append(
            w_c_func_clnks_geo_isomrphc_set)
        w_c_args_clnks_geo_isomrphc_sets.append(
            w_c_args_clnks_geo_isomrphc_set)
        dw_c__dy_clnk_func_clnks_geo_isomrphc_sets.append(
            dw_c__dy_clnk_func_clnks_geo_isomrphc_set)
        dw_c__dy_clnk_args_clnks_geo_isomrphc_sets.append(
            dw_c__dy_clnk_args_clnks_geo_isomrphc_set)
        d2w_c__dy_clnk_dy_clnk_func_clnks_geo_isomrphc_sets.append(
            d2w_c__dy_clnk_dy_clnk_func_clnks_geo_isomrphc_set)
        d2w_c__dy_clnk_dy_clnk_args_clnks_geo_isomrphc_sets.append(
            d2w_c__dy_clnk_dy_clnk_args_clnks_geo_isomrphc_set)
    
    # Initialize the polymer chain deformation free energy function
    # string for each chain in each geometrically isomorphic set of
    # cross-link structures
    w_c_dfrmtn_dist_clnks_geo_isomrphc_sets = []
    for set_indx in range(geo_isomrphc_sets_num):
        n_clnks_geo_isomrphc_set = n_clnks_geo_isomrphc_sets[set_indx]
        w_c_dfrmtn_dist_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set, dtype="U26")
        for chn_indx in np.ndindex(np.shape(n_clnks_geo_isomrphc_set)):
            w_c_dfrmtn_dist_clnks_geo_isomrphc_set[chn_indx] = (
                cfg.topology.w_c_dfrmtn_dist
            )
        w_c_dfrmtn_dist_clnks_geo_isomrphc_sets.append(
            w_c_dfrmtn_dist_clnks_geo_isomrphc_set)
    
    # Extract the nondimensional polymer chain deformation free energy
    # function for each chain in each geometrically isomorphic set of
    # cross-link structures
    w_c_dfrmtn_func_clnks_geo_isomrphc_sets = []
    w_c_dfrmtn_args_clnks_geo_isomrphc_sets = []
    for set_indx in range(geo_isomrphc_sets_num):
        n_clnks_geo_isomrphc_set = n_clnks_geo_isomrphc_sets[set_indx]
        w_c_dfrmtn_func_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set, dtype=object)
        w_c_dfrmtn_args_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set, dtype=object)
        for chn_indx in np.ndindex(np.shape(n_clnks_geo_isomrphc_set)):
            w_c_dfrmtn_func_clnks_geo_isomrphc_set[chn_indx] = (
                master_w_c_dfrmtn_func(
                    w_c_dfrmtn_dist_clnks_geo_isomrphc_sets[set_indx][chn_indx])
            )
            w_c_dfrmtn_args_clnks_geo_isomrphc_set[chn_indx] = (
                master_w_c_dfrmtn_args_func(
                    w_c_dfrmtn_dist_clnks_geo_isomrphc_sets[set_indx][chn_indx])
            )
        w_c_dfrmtn_func_clnks_geo_isomrphc_sets.append(
            w_c_dfrmtn_func_clnks_geo_isomrphc_set)
        w_c_dfrmtn_args_clnks_geo_isomrphc_sets.append(
            w_c_dfrmtn_args_clnks_geo_isomrphc_set)
    
    # Calculate the critical absolute/equilibrium polymer chain stretch
    # for each chain in each geometrically isomorphic set of
    # cross-link structures
    gamma_crit_clnks_geo_isomrphc_sets = []
    for set_indx in range(geo_isomrphc_sets_num):
        n_clnks_geo_isomrphc_set = n_clnks_geo_isomrphc_sets[set_indx]
        gamma_crit_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set)
        for chn_indx in np.ndindex(np.shape(n_clnks_geo_isomrphc_set)):
            gamma_crit_clnks_geo_isomrphc_set[chn_indx] = (
                master_gamma_crit_func(
                    w_c_dist_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    kappa_n_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    zeta_n_char_clnks_geo_isomrphc_sets[set_indx][chn_indx])
            )
        gamma_crit_clnks_geo_isomrphc_sets.append(
            gamma_crit_clnks_geo_isomrphc_set)
    
    # Calculate the critical polymer chain contour length for each chain
    # in each geometrically isomorphic set of cross-link structures
    r_crit_clnks_geo_isomrphc_sets = []
    for set_indx in range(geo_isomrphc_sets_num):
        n_clnks_geo_isomrphc_set = n_clnks_geo_isomrphc_sets[set_indx]
        r_crit_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set)
        for chn_indx in np.ndindex(np.shape(n_clnks_geo_isomrphc_set)):
            r_crit_clnks_geo_isomrphc_set[chn_indx] = (
                r_func(
                    gamma_crit_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    n_clnks_geo_isomrphc_set[chn_indx],
                    b_clnks_geo_isomrphc_sets[set_indx][chn_indx])
            )
        r_crit_clnks_geo_isomrphc_sets.append(r_crit_clnks_geo_isomrphc_set)
    
    # Calculate the root-mean-square absolute/equilibrium polymer chain
    # stretch for each chain in each geometrically isomorphic set of
    # cross-link structures
    gamma_rms_clnks_geo_isomrphc_sets = []
    for set_indx in range(geo_isomrphc_sets_num):
        n_clnks_geo_isomrphc_set = n_clnks_geo_isomrphc_sets[set_indx]
        gamma_rms_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set)
        for chn_indx in np.ndindex(np.shape(n_clnks_geo_isomrphc_set)):
            gamma_rms_args = master_gamma_rms_args_func(
                w_c_dist_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                kappa_n_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                zeta_n_char_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                gamma_crits_clnks_geo_isomrphc_sets[set_indx][chn_indx])
            gamma_rms_clnks_geo_isomrphc_set[chn_indx] = (
                master_gamma_rms_func(
                    points, weights, n_clnks_geo_isomrphc_set[chn_indx],
                    gamma_crit_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    cfg.deformation.gamma_n_hat_inc,
                    w_c_dist_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    w_c_func_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    w_c_args_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                    gamma_rms_args)
            )
        gamma_rms_clnks_geo_isomrphc_sets.append(
            gamma_rms_clnks_geo_isomrphc_set)
    
    # Calculate the root-mean-square polymer chain length on a
    # chain-by-chain basis in each geometrically isomorphic set of
    # cross-link structures
    r_rms_clnks_geo_isomrphc_sets = []
    for set_indx in range(geo_isomrphc_sets_num):
        n_clnks_geo_isomrphc_set = n_clnks_geo_isomrphc_sets[set_indx]
        r_rms_clnks_geo_isomrphc_set = np.empty_like(
            n_clnks_geo_isomrphc_set)
        for chn_indx in np.ndindex(np.shape(n_clnks_geo_isomrphc_set)):
            r_rms_clnks_geo_isomrphc_set[chn_indx] = r_func(
                gamma_rms_clnks_geo_isomrphc_sets[set_indx][chn_indx],
                n_clnks_geo_isomrphc_set[chn_indx],
                b_clnks_geo_isomrphc_sets[set_indx][chn_indx])
        r_rms_clnks_geo_isomrphc_sets.append(r_rms_clnks_geo_isomrphc_set)

    # Initialize the cross-link structures in each geometrically
    # isomorphic set
    X_clnks_geo_isomrphc_sets = []
    omega_clnks_init_geo_isomrphc_sets = []
    y_clnks_init_geo_isomrphc_sets = []
    gamma_clnks_init_geo_isomrphc_sets = []
    for set_indx in range(geo_isomrphc_sets_num):
        n_clnks_geo_isomrphc_set = n_clnks_geo_isomrphc_sets[set_indx]
        clnks_num, k_num = np.shape(n_clnks_geo_isomrphc_set)
        X_clnks_geo_isomrphc_set = np.zeros((clnks_num, k_num, 3))
        omega_clnks_init_geo_isomrphc_set = np.zeros((clnks_num, 3))
        y_clnks_init_geo_isomrphc_set = np.zeros((clnks_num, 3))
        gamma_clnks_init_geo_isomrphc_set = np.zeros((clnks_num, k_num))
        for clnk_indx in range(clnks_num):
            X_clnk, omega_clnk_init, y_clnk_init = recommended_clnk_init_func(
                r_rms_clnks_geo_isomrphc_sets[set_indx][clnk_indx],
                type_8_chn_clnk=cfg.topology.type_8_chn_clnk)
            X_clnks_geo_isomrphc_set[clnk_indx] = X_clnk
            omega_clnks_init_geo_isomrphc_set[clnk_indx] = omega_clnk_init
            y_clnks_init_geo_isomrphc_set[clnk_indx] = y_clnk_init
            r_clnk_init = X_clnk - y_clnk_init
            for chn_indx in range(k_num):
                gamma_clnks_init_geo_isomrphc_set[clnk_indx, chn_indx] = (
                    gamma_func(
                        np.linalg.norm(r_clnk_init[chn_indx]),
                        n_clnks_geo_isomrphc_set[clnk_indx, chn_indx],
                        b_clnks_geo_isomrphc_sets[set_indx][clnk_indx, chn_indx])
                )
        X_clnks_geo_isomrphc_sets.append(X_clnks_geo_isomrphc_set)
        omega_clnks_init_geo_isomrphc_sets.append(
            omega_clnks_init_geo_isomrphc_set)
        y_clnks_init_geo_isomrphc_sets.append(
            y_clnks_init_geo_isomrphc_set)
        gamma_clnks_init_geo_isomrphc_sets.append(
            gamma_clnks_init_geo_isomrphc_set)
    
    # Verify that all initial cross-link positions coincide with the
    # origin
    for set_indx in range(geo_isomrphc_sets_num):
        y_clnks_init_geo_isomrphc_set = (
            y_clnks_init_geo_isomrphc_sets[set_indx]
        )
        for clnk_indx in range(np.shape(y_clnks_init_geo_isomrphc_set)[0]):
            if not np.allclose(y_clnks_init_geo_isomrphc_set[clnk_indx], np.zeros(3)):
                error_str = (
                    "The initial position of a cross-link is not at the "
                    + "origin."
                )
                raise ValueError(error_str)
    
    # Deformation protocol initialization
    dfrmtn_protocol_class = cfg.deformation.protocol_class
    dfrmtn_protocol = []
    for protocol_indx in range(len(dfrmtn_protocol_class)):
        dfrmtn_protocol.append(
            deformation_protocol_init_func(
                cfg.deformation.protocol_init,
                cfg.deformation.protocol[protocol_indx]))

    # Data initialization
    dfrmtn_protocol_num = len(dfrmtn_protocol)
    clnks_num = np.shape(n_clnks)[0]
    W_clnks_frame_avrg_approx_so3_quad = []
    for protocol_indx in range(dfrmtn_protocol_num):
        dfrmtn_arr = dfrmtn_protocol[protocol_indx]
        num_dfrmtn_steps = np.shape(dfrmtn_arr)[0]
        W_clnks_frame_avrg_approx_so3_quad.append(
            np.zeros((clnks_num, num_dfrmtn_steps)))
    
    geo_isomrphc_sets_num = len(n_clnks_geo_isomrphc_sets)
    W_clnks_geo_isomrphc_sets_free_rot_approx = []
    for protocol_indx in range(dfrmtn_protocol_num):
        dfrmtn_arr = dfrmtn_protocol[protocol_indx]
        num_dfrmtn_steps = np.shape(dfrmtn_arr)[0]
        W_clnks_geo_isomrphc_sets_free_rot_approx_dfrmtn_protocol = []
        for set_indx in range(geo_isomrphc_sets_num):
            n_clnks_geo_isomrphc_set = n_clnks_geo_isomrphc_sets[set_indx]
            clnks_num = np.shape(n_clnks_geo_isomrphc_set)[0]
            W_clnks_geo_isomrphc_sets_free_rot_approx_dfrmtn_protocol.append(
                np.zeros((clnks_num, num_dfrmtn_steps)))
        W_clnks_geo_isomrphc_sets_free_rot_approx.append(
            W_clnks_geo_isomrphc_sets_free_rot_approx_dfrmtn_protocol)
    
    # Step through each deformation protocol and evaluate the
    # deformation of each cross-link structure via the frame averaging
    # approximation
    clnks_num = np.shape(n_clnks)[0]
    for protocol_indx in range(dfrmtn_protocol_num):
        dfrmtn_arr = dfrmtn_protocol[protocol_indx]
        # Step through cross-link structures
        for clnk_indx in range(clnks_num):
            # Step through deformation steps
            for dfrmtn_step in range(np.shape(dfrmtn_arr)[0]):
                # Deformation gradient
                F, _ = F_func(
                    dfrmtn_protocol_class[protocol_indx], dfrmtn_arr[dfrmtn_step])
                (_, _, _, W_clnk_chns_frame_avrg_approx_so3,
                 W_clnk_y_flucts_frame_avrg_approx_so3) = clnk_frame_avrg_approx(
                    cfg.deformation.eval_W_clnk_y_flucts,
                    cfg.deformation.use_inext_gaussian_fjc_delta_clnk, F,
                    so3_quad, n_clnks[clnk_indx], b_clnks[clnk_indx],
                    X_clnks[clnk_indx], y_clnks_init[clnk_indx],
                    w_c_func_clnks[clnk_indx], w_c_args_clnks[clnk_indx],
                    dw_c__dy_clnk_func_clnks[clnk_indx],
                    dw_c__dy_clnk_args_clnks[clnk_indx],
                    d2w_c__dy_clnk_dy_clnk_func_clnks[clnk_indx],
                    d2w_c__dy_clnk_dy_clnk_args_clnks[clnk_indx],
                    w_c_dfrmtn_func_clnks[clnk_indx],
                    w_c_dfrmtn_args_clnks[clnk_indx])
                W_clnk_chns_frame_avrg_approx_so3_quad = (
                    W_clnk_chns_frame_avrg_so3_quad_eval(
                        so3_quad, sph_quad_symmtry,
                        W_clnk_chns_frame_avrg_approx_so3)
                )
                W_clnk_y_flucts_frame_avrg_approx_so3_quad = (
                    W_clnk_y_flucts_frame_avrg_so3_quad_eval(
                        so3_quad, sph_quad_symmtry,
                        W_clnk_y_flucts_frame_avrg_approx_so3)
                )
                W_clnks_frame_avrg_approx_so3_quad[protocol_indx][clnk_indx, dfrmtn_step] = (
                    W_clnk_chns_frame_avrg_approx_so3_quad
                    + W_clnk_y_flucts_frame_avrg_approx_so3_quad
                )
            
                # Print statement tracking deformation
                dfrmtn_protocol_class_str = dfrmtn_protocol_class[protocol_indx]
                if dfrmtn_protocol_class_str == "uniaxial": dfrmtn_str = "lmbda"
                elif dfrmtn_protocol_class_str == "simple_shear": dfrmtn_str = "s"
                print("Frame averaging approximation, n_clnk = {}, {}, {}={}".format(n_clnks[clnk_indx], dfrmtn_protocol_class_str, dfrmtn_str, dfrmtn_arr[dfrmtn_step]))
    
    # Step through each deformation protocol and evaluate the
    # deformation of each geometrically isomorphic set of cross-link
    # structures via the free rotation approximation
    for protocol_indx in range(dfrmtn_protocol_num):
        dfrmtn_arr = dfrmtn_protocol[protocol_indx]
        # Step through each geometrically isomorphic set of cross-link
        # structures
        for set_indx in range(geo_isomrphc_sets_num):
            # Step through deformation steps
            for dfrmtn_step in range(np.shape(dfrmtn_arr)[0]):
                # Deformation gradient
                F, _ = F_func(
                    dfrmtn_protocol_class[protocol_indx], dfrmtn_arr[dfrmtn_step])
                (_, _, _, _, _,
                 W_clnk_chns_clnks_geo_isomrphc_set_free_rot_approx,
                 W_clnk_y_flucts_clnks_geo_isomrphc_set_free_rot_approx, _, _,
                 _, _, _, _, _) = clnk_free_rot_approx(
                    cfg.deformation.eval_W_clnk_y_flucts,
                    cfg.deformation.use_inext_gaussian_fjc_delta_clnk, F,
                    n_clnks_geo_isomrphc_sets[set_indx],
                    b_clnks_geo_isomrphc_sets[set_indx],
                    X_clnks_geo_isomrphc_sets[set_indx],
                    y_clnks_init_geo_isomrphc_sets[set_indx],
                    w_c_func_clnks_geo_isomrphc_sets[set_indx],
                    w_c_args_clnks_geo_isomrphc_sets[set_indx],
                    d2w_c__dy_clnk_dy_clnk_func_clnks_geo_isomrphc_sets[set_indx],
                    d2w_c__dy_clnk_dy_clnk_args_clnks_geo_isomrphc_sets[set_indx],
                    w_c_dfrmtn_func_clnks_geo_isomrphc_sets[set_indx],
                    w_c_dfrmtn_args_clnks_geo_isomrphc_sets[set_indx])
                W_clnks_geo_isomrphc_sets_free_rot_approx[protocol_indx][set_indx][:, dfrmtn_step] = (
                    W_clnk_chns_clnks_geo_isomrphc_set_free_rot_approx
                    + W_clnk_y_flucts_clnks_geo_isomrphc_set_free_rot_approx
                )
            
                # Print statement tracking deformation
                dfrmtn_protocol_class_str = dfrmtn_protocol_class[protocol_indx]
                if dfrmtn_protocol_class_str == "uniaxial": dfrmtn_str = "lmbda"
                elif dfrmtn_protocol_class_str == "simple_shear": dfrmtn_str = "s"
                print("Free rotation approximation, n_clnks_geo_isomrphc_set = {}, {}, {}={}".format(n_clnks_geo_isomrphc_sets[set_indx], dfrmtn_protocol_class_str, dfrmtn_str, dfrmtn_arr[dfrmtn_step]))
    
    # Generate filenames and save data
    for protocol_indx in range(dfrmtn_protocol_num):
        protocol_indx_str = f"protocol_indx_{protocol_indx:d}"
        dfrmtn_filename = (
            filename_prefix + "-dfrmtn" + "_" + protocol_indx_str + ".npy"
        )
        np.save(dfrmtn_filename, dfrmtn_protocol[protocol_indx])

        # Save frame averaging approximation data
        W_clnks_frame_avrg_approx_so3_quad_filename = (
            filename_prefix + "-W_clnks_frame_avrg_approx_so3_quad"
            + "_" + protocol_indx_str + ".npy"
        )
        np.save(
            W_clnks_frame_avrg_approx_so3_quad_filename,
            W_clnks_frame_avrg_approx_so3_quad[protocol_indx])
        
        # Save free rotation approximation data
        for set_indx in range(geo_isomrphc_sets_num):
            W_clnks_geo_isomrphc_sets_free_rot_approx_filename = (
                filename_prefix + "-W_clnks_geo_isomrphc_sets_free_rot_approx"
                + "_" + protocol_indx_str + "_" + f"set_indx_{set_indx:d}"
                + ".npy"
            )
            np.save(
                W_clnks_geo_isomrphc_sets_free_rot_approx_filename,
                W_clnks_geo_isomrphc_sets_free_rot_approx[protocol_indx][set_indx])

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse Kuhn-Grun end-linked polymer network elastically-effective cross-link RVE deformation analysis took {execution_time} seconds to run")