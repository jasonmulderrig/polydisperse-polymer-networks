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
from src.helpers.clnk_free_rotation_utils import monodisperse_clnk_free_rot
from src.helpers.clnk_frame_averaging_utils import monodisperse_clnk_frame_avrg

def clnk_rves_deformation(
        label: DictConfig,
        sample: int,
        deformation: DictConfig,
        w_c_dist: str,
        w_c_args: list,
        w_c_dfrmtn_dist: str,
        w_c_dfrmtn_args: list,
        b: float,
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
    so3_quad, sph_quad_symmtry = master_so3_quadrature_func("bazant_oh_013", 16)
    so3_quad_num = np.shape(so3_quad)[0]

    # Convex hull quadrature scheme
    vol_clnk_side_dim_points = 10 # 15

    # Evaluate W_flucts
    eval_W_flucts = False
    
    # Extract arguments for the polymer chain segment number probability
    # distribution, and convert lists to tuples
    n = tuple(n)

    # Extract arguments for the polymer chain free energy function
    w_c_args = [] if w_c_args == [None] else w_c_args
    w_c_args = tuple(w_c_args)

    # Extract polymer chain free energy function
    w_c_func = master_w_c_func(w_c_dist)

    # Extract arguments for the cross-link deformation polymer chain
    # free energy function
    w_c_dfrmtn_args = [] if w_c_dfrmtn_args == [None] else w_c_dfrmtn_args
    w_c_dfrmtn_args = tuple(w_c_dfrmtn_args)

    # Extract cross-link deformation polymer chain free energy function
    w_c_dfrmtn_func = master_w_c_dfrmtn_func(w_c_dfrmtn_dist)

    # Initialize and save the salient chain segment numbers
    n_clnks, _ = n_init_func(n_init, n)
    n_clnks_filename = filename_prefix + "-n_clnks" + ".dat"
    np.savetxt(n_clnks_filename, n_clnks, fmt="%d")
    
    # Calculate the critical polymer chain contour length on a
    # chain-by-chain basis
    r_crit_clnks = np.empty_like(n_clnks, dtype=float)
    for clnk_indx in np.ndindex(np.shape(n_clnks)):
        r_crit_clnks[clnk_indx] = master_r_crit_func(
            n_clnks[clnk_indx], b, w_c_dist, w_c_args)
    
    # Calculate the root-mean-square polymer chain length on a
    # chain-by-chain basis
    r_rms_clnks = np.empty_like(n_clnks, dtype=float)
    for clnk_indx in np.ndindex(np.shape(n_clnks)):
        r_rms_clnks[clnk_indx] = master_r_rms_func(
            points, weights, r_crit_clnks[clnk_indx], n_clnks[clnk_indx], b,
            w_c_dist, w_c_func, w_c_args)

    # Initialize the cross-link structures
    clnks_num, k_num = np.shape(n_clnks)
    X_clnks = np.zeros((clnks_num, k_num, 3))
    X_l_clnk, _, _  = recommended_clnk_init_func(
        n_clnks[0]*b, type_8_chn_clnk="cube")
    vol_quad_clnk = vol_quad_clnk_func(X_l_clnk, vol_clnk_side_dim_points)
    vol_quad_clnk_shape = np.shape(vol_quad_clnk)
    vol_quad_clnks = np.zeros(
        (clnks_num, vol_quad_clnk_shape[0], vol_quad_clnk_shape[1]))
    omega_clnks_init = np.zeros((clnks_num, 3))
    y_clnks_init = np.zeros((clnks_num, 3))
    gamma_clnks_init = np.zeros((clnks_num, k_num))
    for clnk_indx in range(clnks_num):
        n_clnk = n_clnks[clnk_indx]
        X_clnk, omega_clnk_init, y_clnk_init = recommended_clnk_init_func(
            r_rms_clnks[clnk_indx], type_8_chn_clnk="cube")
        X_l_clnk, _, _ = recommended_clnk_init_func(
            n_clnk*b, type_8_chn_clnk="cube")
        vol_quad_clnk = vol_quad_clnk_func(
            X_l_clnk, vol_clnk_side_dim_points)
        X_clnks[clnk_indx] = X_clnk
        vol_quad_clnks[clnk_indx] = vol_quad_clnk
        omega_clnks_init[clnk_indx] = omega_clnk_init
        y_clnks_init[clnk_indx] = y_clnk_init
        r_clnk_init = X_clnk - y_clnk_init
        for chn_indx in range(k_num):
            gamma_clnks_init[clnk_indx, chn_indx] = gamma_func(
                np.linalg.norm(r_clnk_init[chn_indx]), n_clnk[chn_indx], b)
    
    # Verify that all initial cross-link positions coincide with the origin
    for clnk_indx in range(np.shape(y_clnks_init)[0]):
        if not np.allclose(y_clnks_init[clnk_indx], np.zeros(3)):
            error_str = (
                "The initial position of a cross-link is not at the "
                + "origin."
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

    # Data initialization
    W_clnks_free_rot = []
    W_clnks_frame_avrg_so3_quad = []

    for protocol_indx in range(len(dfrmtn_protocol)):
        dfrmtn_arr = dfrmtn_protocol[protocol_indx]
        num_dfrmtn_steps = np.shape(dfrmtn_arr)[0]

        clnks_num, k_num = np.shape(n_clnks)
        
        W_clnks_free_rot.append(
            np.zeros((num_dfrmtn_steps, clnks_num)))
        W_clnks_frame_avrg_so3_quad.append(
            np.zeros((num_dfrmtn_steps, clnks_num)))

    # Step through each deformation protocol
    for protocol_indx in range(len(dfrmtn_protocol)):
        dfrmtn_arr = dfrmtn_protocol[protocol_indx]
        num_dfrmtn_steps = np.shape(dfrmtn_arr)[0]

        # Evaluate the deformation of each cross-link structure
        clnks_num, k_num = np.shape(n_clnks)
        # Step through deformation steps
        for dfrmtn_step in range(num_dfrmtn_steps):
            # Deformation gradient
            F, Lmbda = F_func(
                dfrmtn_protocol_class[protocol_indx], dfrmtn_arr[dfrmtn_step])
            for clnk_indx in range(clnks_num):
                # Gather universal cross-link structure
                n_clnk = n_clnks[clnk_indx]
                X_clnk = X_clnks[clnk_indx]
                X_hat_clnk = x_hat_clnk_func(X_clnk)
                com_X_hat_clnk = com_x_clnk_func(X_hat_clnk)
                vol_quad_clnk = vol_quad_clnks[clnk_indx]
                y_clnk_init = y_clnks_init[clnk_indx]
                gamma_clnk_init = gamma_clnks_init[clnk_indx]
                
                if (np.all(np.equal(n_clnk, n_clnk[0])) and
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
                        _, _, _, _, W_clnk_star, _ = monodisperse_clnk_free_rot(
                            eval_W_flucts, F, Lmbda, n_clnk, b, X_clnk,
                            vol_quad_clnk, y_clnk_init, gamma_clnk_init,
                            w_c_func, w_c_args, w_c_dfrmtn_func, w_c_dfrmtn_args)

                        W_clnks_free_rot[protocol_indx][dfrmtn_step, clnk_indx] = (
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
                        _, _, _, _, _, _, _, W_clnk_star_frame_avrg_so3_quad = (
                            monodisperse_clnk_frame_avrg(
                                F, n_clnk, b, X_clnk, so3_quad,
                                sph_quad_symmtry, y_clnk_init,
                                w_c_func, w_c_args,
                                w_c_dfrmtn_func, w_c_dfrmtn_args)
                        )

                        W_clnks_frame_avrg_so3_quad[protocol_indx][dfrmtn_step, clnk_indx] = (
                            W_clnk_star_frame_avrg_so3_quad
                        )
                
            # Print statement tracking deformation
            dfrmtn_protocol_class_str = dfrmtn_protocol_class[protocol_indx]
            if dfrmtn_protocol_class_str == "uniaxial": dfrmtn_str = "lmbda"
            elif dfrmtn_protocol_class_str == "simple_shear": dfrmtn_str = "s"
            print("{}, {}={}".format(dfrmtn_protocol_class_str, dfrmtn_str, dfrmtn_arr[dfrmtn_step]))
    
    # Generate filenames and save data
    for protocol_indx in range(len(dfrmtn_protocol)):
        protocol_indx_str = "protocol_indx_" + str(protocol_indx)
        dfrmtn_filename = (
            filename_prefix + "-dfrmtn" + "_" + protocol_indx_str + ".npy"
        )
        np.save(dfrmtn_filename, dfrmtn_protocol[protocol_indx])

        W_clnks_free_rot_filename = (
            filename_prefix + "-W_clnks_free_rot"
            + "_" + protocol_indx_str + ".npy"
        )
        W_clnks_frame_avrg_so3_quad_filename = (
            filename_prefix + "-W_clnks_frame_avrg_so3_quad"
            + "_" + protocol_indx_str + ".npy"
        )

        np.save(
            W_clnks_free_rot_filename,
            W_clnks_free_rot[protocol_indx])
        np.save(
            W_clnks_frame_avrg_so3_quad_filename,
            W_clnks_frame_avrg_so3_quad[protocol_indx])

##### This code corresponds to the 20250724C.yaml configuration file.
##### Make sure to set the deformation and topology configuration files
##### to 20250724C.yaml in the config.yaml file before running.
@hydra.main(
        version_base=None,
        config_path="../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves",
        config_name="config")
def main(cfg: DictConfig) -> None:
    w_c_dist = cfg.topology.w_c_dist
    w_c_args = cfg.topology.w_c_args[0]
    w_c_dfrmtn_dist = cfg.topology.w_c_dfrmtn_dist
    w_c_dfrmtn_args = cfg.topology.w_c_dfrmtn_args[0]
    b = cfg.topology.b
    n = cfg.topology.n
    n_init = cfg.topology.n_init

    clnk_rves_deformation(
        cfg.label, 0, cfg.deformation, w_c_dist, w_c_args, w_c_dfrmtn_dist,
        w_c_dfrmtn_args, b, n, n_init)

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse Gaussian end-linked polymer network elastically-effective cross-link RVE deformation analysis took {execution_time} seconds to run")