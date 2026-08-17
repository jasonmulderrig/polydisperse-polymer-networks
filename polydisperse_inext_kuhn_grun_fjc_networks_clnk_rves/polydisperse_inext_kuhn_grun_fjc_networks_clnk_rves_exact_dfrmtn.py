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
    master_d2w_c__dy_clnk_dy_clnk_func,
    master_d2w_c__dy_clnk_dy_clnk_args_func
)
from src.helpers.chain_length import r_func
from src.helpers.clnk_structure import recommended_clnk_init_func
from src.helpers.continuum_mechanics import (
    deformation_protocol_init_func,
    F_func
)
from src.helpers.clnk_frame_averaging import (
    clnk_frame_avrg_cnstrnd_mnmztn,
    W_clnk_chns_frame_avrg_so3_quad_eval,
    W_clnk_y_flucts_frame_avrg_so3_quad_eval
)
from src.helpers.clnk_free_rotation import clnk_free_rot_cnstrnd_mnmztn

@hydra.main(
        version_base=None,
        config_path="../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves",
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
    # the nondimensional second derivative of the polymer chain free
    # energy with respect to the cross-link junction position function
    # for each chain
    w_c_func_clnks = np.empty_like(n_clnks, dtype=object)
    w_c_args_clnks = np.empty_like(n_clnks, dtype=object)
    d2w_c__dy_clnk_dy_clnk_func_clnks = np.empty_like(n_clnks, dtype=object)
    d2w_c__dy_clnk_dy_clnk_args_clnks = np.empty_like(n_clnks, dtype=object)
    for chn_indx in np.ndindex(np.shape(n_clnks)):
        w_c_func_clnks[chn_indx] = master_w_c_func(w_c_dist_clnks[chn_indx])
        w_c_args_clnks[chn_indx] = master_w_c_args_func(
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
    y_clnks_frame_avrg_so3 = []
    W_clnks_frame_avrg_so3_quad = []
    omega_clnks_free_rot = []
    y_clnks_free_rot = []
    W_clnks_free_rot = []
    for protocol_indx in range(dfrmtn_protocol_num):
        dfrmtn_arr = dfrmtn_protocol[protocol_indx]
        num_dfrmtn_steps = np.shape(dfrmtn_arr)[0]
        y_clnks_frame_avrg_so3.append(
            np.zeros((clnks_num, so3_quad_num, 3, num_dfrmtn_steps)))
        W_clnks_frame_avrg_so3_quad.append(
            np.zeros((clnks_num, num_dfrmtn_steps)))
        omega_clnks_free_rot.append(
            np.zeros((clnks_num, 3, num_dfrmtn_steps)))
        y_clnks_free_rot.append(
            np.zeros((clnks_num, 3, num_dfrmtn_steps)))
        W_clnks_free_rot.append(
            np.zeros((clnks_num, num_dfrmtn_steps)))
    
    # Step through each deformation protocol
    for protocol_indx in range(dfrmtn_protocol_num):
        dfrmtn_arr = dfrmtn_protocol[protocol_indx]
        # Step through cross-link structures
        for clnk_indx in range(clnks_num):
            # Gather cross-link structure attributes
            n_clnk = n_clnks[clnk_indx]
            b_clnk = b_clnks[clnk_indx]
            X_clnk = X_clnks[clnk_indx]
            w_c_func_clnk = w_c_func_clnks[clnk_indx]
            w_c_args_clnk = w_c_args_clnks[clnk_indx]
            d2w_c__dy_clnk_dy_clnk_func_clnk = (
                d2w_c__dy_clnk_dy_clnk_func_clnks[clnk_indx]
            )
            d2w_c__dy_clnk_dy_clnk_args_clnk = (
                d2w_c__dy_clnk_dy_clnk_args_clnks[clnk_indx]
            )
            w_c_dfrmtn_func_clnk = w_c_dfrmtn_func_clnks[clnk_indx]
            w_c_dfrmtn_args_clnk = w_c_dfrmtn_args_clnks[clnk_indx]
        
            # Step through deformation steps
            for dfrmtn_step in range(np.shape(dfrmtn_arr)[0]):
                # Deformation gradient
                F, _ = F_func(
                    dfrmtn_protocol_class[protocol_indx], dfrmtn_arr[dfrmtn_step])
                
                # Gather initial cross-link structure for the
                # constrained minimization schemes
                if dfrmtn_step == 0: init_dfrmtn_step = 0
                else: init_dfrmtn_step = dfrmtn_step - 1

                # Frame averaging model
                y_clnk_frame_avrg_so3_init = (
                    y_clnks_frame_avrg_so3[protocol_indx][clnk_indx, :, :, init_dfrmtn_step]
                )

                # Free rotation model
                omega_clnk_free_rot_init = (
                    omega_clnks_free_rot[protocol_indx][clnk_indx, :, init_dfrmtn_step]
                )
                y_clnk_free_rot_init = (
                    y_clnks_free_rot[protocol_indx][clnk_indx, :, init_dfrmtn_step]
                )
                
                # Evaluate the cross-link structure deformation via
                # frame averaging constrained minimization
                (y_clnk_frame_avrg_so3, _, _, W_clnk_chns_frame_avrg_so3,
                 W_clnk_y_flucts_frame_avrg_so3) = (
                    clnk_frame_avrg_cnstrnd_mnmztn(
                        cfg.deformation.eval_W_clnk_y_flucts,
                        cfg.deformation.cnstrnd_mnmztn_scope,
                        cfg.deformation.cnstrnd_mnmztn_method, rng, F, so3_quad,
                        y_clnk_frame_avrg_so3_init, n_clnk, b_clnk, X_clnk,
                        w_c_func_clnk, w_c_args_clnk,
                        d2w_c__dy_clnk_dy_clnk_func_clnk,
                        d2w_c__dy_clnk_dy_clnk_args_clnk,
                        w_c_dfrmtn_func_clnk, w_c_dfrmtn_args_clnk)
                )
                W_clnk_chns_frame_avrg_so3_quad = (
                    W_clnk_chns_frame_avrg_so3_quad_eval(
                        so3_quad, sph_quad_symmtry, W_clnk_chns_frame_avrg_so3)
                )
                W_clnk_y_flucts_frame_avrg_so3_quad = (
                    W_clnk_y_flucts_frame_avrg_so3_quad_eval(
                        so3_quad, sph_quad_symmtry,
                        W_clnk_y_flucts_frame_avrg_so3)
                )

                y_clnks_frame_avrg_so3[protocol_indx][clnk_indx, :, :, dfrmtn_step] = (
                    y_clnk_frame_avrg_so3
                )
                W_clnks_frame_avrg_so3_quad[protocol_indx][clnk_indx, dfrmtn_step] = (
                    W_clnk_chns_frame_avrg_so3_quad
                    + W_clnk_y_flucts_frame_avrg_so3_quad
                )

                # Evaluate the cross-link structure deformation via free
                # rotation constrained minimization
                (omega_clnk_free_rot, _, y_clnk_free_rot, _, _,
                 W_clnk_chns_free_rot, W_clnk_y_flucts_free_rot) = (
                    clnk_free_rot_cnstrnd_mnmztn(
                        cfg.deformation.eval_W_clnk_y_flucts,
                        cfg.deformation.cnstrnd_mnmztn_scope,
                        cfg.deformation.cnstrnd_mnmztn_method, rng, F, n_clnk,
                        b_clnk, X_clnk, omega_clnk_free_rot_init,
                        y_clnk_free_rot_init, w_c_func_clnk, w_c_args_clnk,
                        d2w_c__dy_clnk_dy_clnk_func_clnk,
                        d2w_c__dy_clnk_dy_clnk_args_clnk,
                        w_c_dfrmtn_func_clnk, w_c_dfrmtn_args_clnk)
                )
                
                omega_clnks_free_rot[protocol_indx][clnk_indx, :, dfrmtn_step] = (
                    omega_clnk_free_rot
                )
                y_clnks_free_rot[protocol_indx][clnk_indx, :, dfrmtn_step] = (
                    y_clnk_free_rot
                )
                W_clnks_free_rot[protocol_indx][clnk_indx, dfrmtn_step] = (
                    W_clnk_chns_free_rot + W_clnk_y_flucts_free_rot
                )
            
                # Print statement tracking deformation
                dfrmtn_protocol_class_str = dfrmtn_protocol_class[protocol_indx]
                if dfrmtn_protocol_class_str == "uniaxial": dfrmtn_str = "lmbda"
                elif dfrmtn_protocol_class_str == "simple_shear": dfrmtn_str = "s"
                print("n_clnk = {}, {}, {}={}".format(n_clnk, dfrmtn_protocol_class_str, dfrmtn_str, dfrmtn_arr[dfrmtn_step]))
    
    # Generate filenames and save data
    for protocol_indx in range(dfrmtn_protocol_num):
        protocol_indx_str = f"protocol_indx_{protocol_indx:d}"
        dfrmtn_filename = (
            filename_prefix + "-dfrmtn" + "_" + protocol_indx_str + ".npy"
        )
        np.save(dfrmtn_filename, dfrmtn_protocol[protocol_indx])

        W_clnks_frame_avrg_so3_quad_filename = (
            filename_prefix + "-W_clnks_frame_avrg_so3_quad"
            + "_" + protocol_indx_str + ".npy"
        )
        W_clnks_free_rot_filename = (
            filename_prefix + "-W_clnks_free_rot"
            + "_" + protocol_indx_str + ".npy"
        )

        np.save(
            W_clnks_frame_avrg_so3_quad_filename,
            W_clnks_frame_avrg_so3_quad[protocol_indx])
        np.save(
            W_clnks_free_rot_filename,
            W_clnks_free_rot[protocol_indx])

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse Kuhn-Grun end-linked polymer network elastically-effective cross-link RVE deformation analysis took {execution_time} seconds to run")