# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams["mathtext.fontset"] = "cm"
from src.file_io.file_io import (
    filename_str,
    filepath_str
)

@hydra.main(
        version_base=None,
        config_path="../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves",
        config_name="config")
def main(cfg: DictConfig) -> None:
    ntwrk_color = ["tab:purple", "tab:green", "tab:blue", "tab:orange", "tab:red"]
    ntwrk_marker = ["+", "+x", "x", "s.", "s"]
    plusx_marker_list = ["+", "x"]
    markersize = 20
    dotsize = 0.5
    markerlinewidth = 0.5
    dotlinewidth = 0.125

    def bimodal_n_legend_func(bimodal_n):
        num_ntwrks, N = np.shape(bimodal_n)
        assert N == 2
        bimodal_n_legend = []
        for ntwrk_indx in range(num_ntwrks):
            ntwrk_str = "$n_a = " + f"{int(bimodal_n[ntwrk_indx, 0]):d}" + ", "
            ntwrk_str += "n_b = " + f"{int(bimodal_n[ntwrk_indx, 1]):d}" + "$"
            bimodal_n_legend.append(ntwrk_str)
        return bimodal_n_legend

    def p_legend_func(p):
        p_legend = []
        for p_indx in range(np.shape(p)[0]):
            p_legend.append("$p = " + f"{p[p_indx]:.2f}" + "$")
        return p_legend

    filepath = filepath_str("polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves")

    mono_75_75_filename_prefix = filename_str(cfg.label.workdir, "20260603", "C", 0)
    bimodal_50_100_frame_avrg_approx_filename_prefix = filename_str(
        cfg.label.workdir, "20260603", "D", 0)
    bimodal_50_100_free_rot_approx_filename_prefix = filename_str(
        cfg.label.workdir, "20260603", "E", 0)
    bimodal_25_125_frame_avrg_approx_filename_prefix = filename_str(
        cfg.label.workdir, "20260603", "F", 0)
    bimodal_25_125_free_rot_approx_filename_prefix = filename_str(
        cfg.label.workdir, "20260603", "G", 0)

    mono_75_75_n_clnks_filename = mono_75_75_filename_prefix + "-n_clnks" + ".npy"
    bimodal_50_100_frame_avrg_approx_n_clnks_filename = (
        bimodal_50_100_frame_avrg_approx_filename_prefix + "-n_clnks" + ".npy"
    )
    bimodal_50_100_free_rot_approx_n_clnks_filename = (
        bimodal_50_100_free_rot_approx_filename_prefix + "-n_clnks" + ".npy"
    )
    bimodal_25_125_frame_avrg_approx_n_clnks_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix + "-n_clnks" + ".npy"
    )
    bimodal_25_125_free_rot_approx_n_clnks_filename = (
        bimodal_25_125_free_rot_approx_filename_prefix + "-n_clnks" + ".npy"
    )
    mono_75_75_n_clnks = np.load(mono_75_75_n_clnks_filename)
    bimodal_50_100_frame_avrg_approx_n_clnks = np.load(
        bimodal_50_100_frame_avrg_approx_n_clnks_filename)
    bimodal_50_100_free_rot_approx_n_clnks = np.load(
        bimodal_50_100_free_rot_approx_n_clnks_filename)
    bimodal_25_125_frame_avrg_approx_n_clnks = np.load(
        bimodal_25_125_frame_avrg_approx_n_clnks_filename)
    bimodal_25_125_free_rot_approx_n_clnks = np.load(
        bimodal_25_125_free_rot_approx_n_clnks_filename)
    assert np.allclose(
        bimodal_50_100_frame_avrg_approx_n_clnks,
        bimodal_50_100_free_rot_approx_n_clnks)
    assert np.allclose(
        bimodal_25_125_frame_avrg_approx_n_clnks,
        bimodal_25_125_free_rot_approx_n_clnks)
    bimodal_50_100_n_clnks = bimodal_50_100_frame_avrg_approx_n_clnks
    bimodal_25_125_n_clnks = bimodal_25_125_frame_avrg_approx_n_clnks
    bimodal_50_100_clnks_num = bimodal_50_100_geo_isomrphc_sets_num = (
        np.shape(bimodal_50_100_n_clnks)[0]
    )
    bimodal_25_125_clnks_num = bimodal_25_125_geo_isomrphc_sets_num = (
        np.shape(bimodal_25_125_n_clnks)[0]
    )
    mono_n_a_75 = np.unique(mono_75_75_n_clnks)
    mono_n_a_75_n_b_75 = np.append(mono_n_a_75, mono_n_a_75)
    bimodal_n_a_50_n_b_100 = np.unique(bimodal_50_100_n_clnks)
    bimodal_n_a_25_n_b_125 = np.unique(bimodal_25_125_n_clnks)
    bimodal_n = np.vstack(
        (mono_n_a_75_n_b_75, np.vstack(
            (bimodal_n_a_50_n_b_100, bimodal_n_a_25_n_b_125))))
    bimodal_n_legend = bimodal_n_legend_func(bimodal_n)

    bimodal_50_100_frame_avrg_approx_p_filename = (
        bimodal_50_100_frame_avrg_approx_filename_prefix + "-p" + ".npy"
    )
    bimodal_50_100_free_rot_approx_p_filename = (
        bimodal_50_100_free_rot_approx_filename_prefix + "-p" + ".npy"
    )
    bimodal_25_125_frame_avrg_approx_p_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix + "-p" + ".npy"
    )
    bimodal_25_125_free_rot_approx_p_filename = (
        bimodal_25_125_free_rot_approx_filename_prefix + "-p" + ".npy"
    )
    bimodal_50_100_frame_avrg_approx_p = np.load(
        bimodal_50_100_frame_avrg_approx_p_filename)
    bimodal_50_100_free_rot_approx_p = np.load(
        bimodal_50_100_free_rot_approx_p_filename)
    bimodal_25_125_frame_avrg_approx_p = np.load(
        bimodal_25_125_frame_avrg_approx_p_filename)
    bimodal_25_125_free_rot_approx_p = np.load(
        bimodal_25_125_free_rot_approx_p_filename)
    assert np.allclose(
        bimodal_50_100_frame_avrg_approx_p, bimodal_50_100_free_rot_approx_p)
    assert np.allclose(
        bimodal_25_125_frame_avrg_approx_p, bimodal_25_125_free_rot_approx_p)
    p = bimodal_50_100_frame_avrg_approx_p
    p_half = bimodal_25_125_frame_avrg_approx_p
    p_half_indx = np.where(p==0.5)[0][0]
    assert p_half_indx == 2
    p_legend = p_legend_func(p)

    bimodal_50_100_frame_avrg_approx_p_clnks_filename = (
        bimodal_50_100_frame_avrg_approx_filename_prefix + "-p_clnks" + ".npy"
    )
    bimodal_50_100_free_rot_approx_p_clnks_filename = (
        bimodal_50_100_free_rot_approx_filename_prefix + "-p_clnks" + ".npy"
    )
    bimodal_25_125_frame_avrg_approx_p_clnks_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix + "-p_clnks" + ".npy"
    )
    bimodal_25_125_free_rot_approx_p_clnks_filename = (
        bimodal_25_125_free_rot_approx_filename_prefix + "-p_clnks" + ".npy"
    )
    bimodal_50_100_frame_avrg_approx_p_clnks = np.load(
        bimodal_50_100_frame_avrg_approx_p_clnks_filename)
    bimodal_50_100_free_rot_approx_p_clnks = np.load(
        bimodal_50_100_free_rot_approx_p_clnks_filename)
    bimodal_25_125_frame_avrg_approx_p_clnks = np.load(
        bimodal_25_125_frame_avrg_approx_p_clnks_filename)
    bimodal_25_125_free_rot_approx_p_clnks = np.load(
        bimodal_25_125_free_rot_approx_p_clnks_filename)
    assert np.allclose(
        bimodal_50_100_frame_avrg_approx_p_clnks,
        bimodal_50_100_free_rot_approx_p_clnks)
    assert np.allclose(
        bimodal_25_125_frame_avrg_approx_p_clnks,
        bimodal_25_125_free_rot_approx_p_clnks)
    bimodal_50_100_p_clnks = bimodal_50_100_frame_avrg_approx_p_clnks
    bimodal_25_125_p_clnks = bimodal_25_125_frame_avrg_approx_p_clnks

    mono_75_75_uniaxl_tens_lmbda_filename = (
        mono_75_75_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    mono_75_75_uniaxl_comp_lmbda_filename = (
        mono_75_75_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    bimodal_50_100_frame_avrg_approx_uniaxl_tens_lmbda_filename = (
        bimodal_50_100_frame_avrg_approx_filename_prefix
        + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    bimodal_50_100_frame_avrg_approx_uniaxl_comp_lmbda_filename = (
        bimodal_50_100_frame_avrg_approx_filename_prefix
        + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    bimodal_50_100_free_rot_approx_uniaxl_tens_lmbda_filename = (
        bimodal_50_100_free_rot_approx_filename_prefix
        + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    bimodal_50_100_free_rot_approx_uniaxl_comp_lmbda_filename = (
        bimodal_50_100_free_rot_approx_filename_prefix
        + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    bimodal_25_125_frame_avrg_approx_uniaxl_tens_lmbda_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix
        + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    bimodal_25_125_frame_avrg_approx_uniaxl_comp_lmbda_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix
        + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    bimodal_25_125_free_rot_approx_uniaxl_tens_lmbda_filename = (
        bimodal_25_125_free_rot_approx_filename_prefix
        + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    bimodal_25_125_free_rot_approx_uniaxl_comp_lmbda_filename = (
        bimodal_25_125_free_rot_approx_filename_prefix
        + "-dfrmtn_protocol_indx_1" + ".npy"
    )

    mono_75_75_uniaxl_tens_lmbda = np.load(mono_75_75_uniaxl_tens_lmbda_filename)
    mono_75_75_uniaxl_tens_lmbda = mono_75_75_uniaxl_tens_lmbda[1:]
    mono_75_75_uniaxl_comp_lmbda = np.flip(
        np.load(mono_75_75_uniaxl_comp_lmbda_filename))
    mono_75_75_lmbda = np.hstack(
        (mono_75_75_uniaxl_comp_lmbda, mono_75_75_uniaxl_tens_lmbda))
    bimodal_50_100_frame_avrg_approx_uniaxl_tens_lmbda = np.load(
        bimodal_50_100_frame_avrg_approx_uniaxl_tens_lmbda_filename)
    bimodal_50_100_frame_avrg_approx_uniaxl_tens_lmbda = (
        bimodal_50_100_frame_avrg_approx_uniaxl_tens_lmbda[1:]
    )
    bimodal_50_100_frame_avrg_approx_uniaxl_comp_lmbda = np.flip(
        np.load(bimodal_50_100_frame_avrg_approx_uniaxl_comp_lmbda_filename))
    bimodal_50_100_frame_avrg_approx_lmbda = np.hstack(
        (bimodal_50_100_frame_avrg_approx_uniaxl_comp_lmbda,
         bimodal_50_100_frame_avrg_approx_uniaxl_tens_lmbda))
    bimodal_50_100_free_rot_approx_uniaxl_tens_lmbda = np.load(
        bimodal_50_100_free_rot_approx_uniaxl_tens_lmbda_filename)
    bimodal_50_100_free_rot_approx_uniaxl_tens_lmbda = (
        bimodal_50_100_free_rot_approx_uniaxl_tens_lmbda[1:]
    )
    bimodal_50_100_free_rot_approx_uniaxl_comp_lmbda = np.flip(
        np.load(bimodal_50_100_free_rot_approx_uniaxl_comp_lmbda_filename))
    bimodal_50_100_free_rot_approx_lmbda = np.hstack(
        (bimodal_50_100_free_rot_approx_uniaxl_comp_lmbda,
         bimodal_50_100_free_rot_approx_uniaxl_tens_lmbda))
    bimodal_25_125_frame_avrg_approx_uniaxl_tens_lmbda = np.load(
        bimodal_25_125_frame_avrg_approx_uniaxl_tens_lmbda_filename)
    bimodal_25_125_frame_avrg_approx_uniaxl_tens_lmbda = (
        bimodal_25_125_frame_avrg_approx_uniaxl_tens_lmbda[1:]
    )
    bimodal_25_125_frame_avrg_approx_uniaxl_comp_lmbda = np.flip(
        np.load(bimodal_25_125_frame_avrg_approx_uniaxl_comp_lmbda_filename))
    bimodal_25_125_frame_avrg_approx_lmbda = np.hstack(
        (bimodal_25_125_frame_avrg_approx_uniaxl_comp_lmbda,
         bimodal_25_125_frame_avrg_approx_uniaxl_tens_lmbda))
    bimodal_25_125_free_rot_approx_uniaxl_tens_lmbda = np.load(
        bimodal_25_125_free_rot_approx_uniaxl_tens_lmbda_filename)
    bimodal_25_125_free_rot_approx_uniaxl_tens_lmbda = (
        bimodal_25_125_free_rot_approx_uniaxl_tens_lmbda[1:]
    )
    bimodal_25_125_free_rot_approx_uniaxl_comp_lmbda = np.flip(
        np.load(bimodal_25_125_free_rot_approx_uniaxl_comp_lmbda_filename))
    bimodal_25_125_free_rot_approx_lmbda = np.hstack(
        (bimodal_25_125_free_rot_approx_uniaxl_comp_lmbda,
         bimodal_25_125_free_rot_approx_uniaxl_tens_lmbda))

    bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_tens_filename = (
        bimodal_50_100_frame_avrg_approx_filename_prefix
        + "-max_gamma_clnks_frame_avrg_approx_so3_protocol_indx_0" + ".npy"
    )
    bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_comp_filename = (
        bimodal_50_100_frame_avrg_approx_filename_prefix
        + "-max_gamma_clnks_frame_avrg_approx_so3_protocol_indx_1" + ".npy"
    )

    bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_tens = np.load(
        bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_tens_filename)
    bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_tens = (
        bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_tens[:, 1:]
    )
    bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_comp = np.flip(
        np.load(bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_comp_filename),
        axis=1)
    bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl = np.hstack(
        (bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_comp,
            bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_tens))

    bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl = []
    for set_indx in range(bimodal_50_100_geo_isomrphc_sets_num):
        bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens_filename = (
            bimodal_50_100_free_rot_approx_filename_prefix
            + "-max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_protocol_indx_0"
            + "_" + f"set_indx_{set_indx:d}" + ".npy"
        )
        bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp_filename = (
            bimodal_50_100_free_rot_approx_filename_prefix
            + "-max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_protocol_indx_1"
            + "_" + f"set_indx_{set_indx:d}" + ".npy"
        )

        bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens = np.load(
            bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens_filename)
        bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens = (
            bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens[:, 1:]
        )
        bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp = np.flip(
            np.load(bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp_filename),
            axis=1)
        bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl = np.hstack(
            (bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp,
                bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens))
        bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl.append(
            bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl)

    bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_tens_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix
        + "-max_gamma_clnks_frame_avrg_approx_so3_protocol_indx_0" + ".npy"
    )
    bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_comp_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix
        + "-max_gamma_clnks_frame_avrg_approx_so3_protocol_indx_1" + ".npy"
    )

    bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_tens = np.load(
        bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_tens_filename)
    bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_tens = (
        bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_tens[:, 1:]
    )
    bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_comp = np.flip(
        np.load(bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_comp_filename),
        axis=1)
    bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl = np.hstack(
        (bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_comp,
            bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_tens))

    bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl = []
    for set_indx in range(bimodal_25_125_geo_isomrphc_sets_num):
        bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens_filename = (
            bimodal_25_125_free_rot_approx_filename_prefix
            + "-max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_protocol_indx_0"
            + "_" + f"set_indx_{set_indx:d}" + ".npy"
        )
        bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp_filename = (
            bimodal_25_125_free_rot_approx_filename_prefix
            + "-max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_protocol_indx_1"
            + "_" + f"set_indx_{set_indx:d}" + ".npy"
        )

        bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens = np.load(
            bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens_filename)
        bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens = (
            bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens[:, 1:]
        )
        bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp = np.flip(
            np.load(bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp_filename),
            axis=1)
        bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl = np.hstack(
            (bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp,
                bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens))
        bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl.append(
            bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl)

    # bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl[bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl>1.]=0.
    # bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn_indcs = (
    #     np.empty(bimodal_50_100_clnks_num, dtype=int)
    # )
    # bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn = (
    #     np.empty(bimodal_50_100_clnks_num)
    # )
    # for clnk_indx in range(bimodal_50_100_clnks_num):
    #     transtn_indx = np.argmax(
    #         bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl[clnk_indx])
    #     bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn_indcs[clnk_indx] = (
    #         transtn_indx
    #     )
    #     bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn[clnk_indx] = (
    #         bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl[clnk_indx, transtn_indx]
    #     )
    # crit_indx = np.argmin(
    #     bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn_indcs)
    # crit_transtn_indx = (
    #     bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn_indcs[crit_indx]
    # )
    # bimodal_50_100_frame_avrg_approx_lmbda_crit_transtn = (
    #     bimodal_50_100_frame_avrg_approx_lmbda[crit_transtn_indx]
    # )
    # bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_crit_transtn = (
    #     bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn[crit_indx]
    # )
    # print("bimodal_50_100_frame_avrg_approx_lmbda_crit_transtn = {}".format(bimodal_50_100_frame_avrg_approx_lmbda_crit_transtn))
    # print("bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn = {}".format(bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn))
    # print("bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_crit_transtn = {}".format(bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_crit_transtn))
    # print("\n")

    # bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn_indcs = (
    #     np.empty(bimodal_50_100_geo_isomrphc_sets_num, dtype=int)
    # )
    # bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn = (
    #     np.empty(bimodal_50_100_geo_isomrphc_sets_num)
    # )
    # for set_indx in range(bimodal_50_100_geo_isomrphc_sets_num):
    #     bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl = (
    #         bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl[set_indx]
    #     )
    #     clnks_num = (
    #         np.shape(bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl)[0]
    #     )
    #     bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl[bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl>1.]=0.
    #     bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn_indcs = (
    #         np.empty(clnks_num, dtype=int)
    #     )
    #     bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn = (
    #         np.empty(clnks_num)
    #     )
    #     for clnk_indx in range(clnks_num):
    #         transtn_indx = np.argmax(
    #             bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl[clnk_indx])
    #         bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn_indcs[clnk_indx] = (
    #             transtn_indx
    #         )
    #         bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn[clnk_indx] = (
    #             bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl[clnk_indx, transtn_indx]
    #         )
    #     crit_indx = np.argmin(
    #         bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn_indcs)
    #     crit_transtn_indx = (
    #         bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn_indcs[crit_indx]
    #     )
    #     bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn_indcs[set_indx] = (
    #         crit_transtn_indx
    #     )
    #     bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn[set_indx] = (
    #         bimodal_50_100_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn[crit_indx]
    #     )
    # crit_indx = np.argmin(
    #     bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn_indcs)
    # crit_transtn_indx = (
    #     bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn_indcs[crit_indx]
    # )
    # bimodal_50_100_free_rot_approx_lmbda_crit_transtn = (
    #     bimodal_50_100_free_rot_approx_lmbda[crit_transtn_indx]
    # )
    # bimodal_50_100_max_gamma_clnks_free_rot_uniaxl_crit_transtn = (
    #     bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn[crit_indx]
    # )
    # print("bimodal_50_100_free_rot_approx_lmbda_crit_transtn = {}".format(bimodal_50_100_free_rot_approx_lmbda_crit_transtn))
    # print("bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn_indcs = {}".format(bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn_indcs))
    # print("bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn = {}".format(bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn))
    # print("bimodal_50_100_max_gamma_clnks_free_rot_uniaxl_crit_transtn = {}".format(bimodal_50_100_max_gamma_clnks_free_rot_uniaxl_crit_transtn))
    # print("\n")

    # bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl[bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl>1.]=0.
    # bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn_indcs = (
    #     np.empty(bimodal_25_125_clnks_num, dtype=int)
    # )
    # bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn = (
    #     np.empty(bimodal_25_125_clnks_num)
    # )
    # for clnk_indx in range(bimodal_25_125_clnks_num):
    #     transtn_indx = np.argmax(
    #         bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl[clnk_indx])
    #     bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn_indcs[clnk_indx] = (
    #         transtn_indx
    #     )
    #     bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn[clnk_indx] = (
    #         bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl[clnk_indx, transtn_indx]
    #     )
    # crit_indx = np.argmin(
    #     bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn_indcs)
    # crit_transtn_indx = (
    #     bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn_indcs[crit_indx]
    # )
    # bimodal_25_125_frame_avrg_approx_lmbda_crit_transtn = (
    #     bimodal_25_125_frame_avrg_approx_lmbda[crit_transtn_indx]
    # )
    # print("bimodal_25_125_frame_avrg_approx_lmbda_crit_transtn = {}".format(bimodal_25_125_frame_avrg_approx_lmbda_crit_transtn))
    # print("bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn = {}".format(bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn))
    # print("bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_crit_transtn = {}".format(bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl_transtn[crit_indx]))
    # print("\n")

    # bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn_indcs = (
    #     np.empty(bimodal_25_125_geo_isomrphc_sets_num, dtype=int)
    # )
    # bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn = (
    #     np.empty(bimodal_25_125_geo_isomrphc_sets_num)
    # )
    # for set_indx in range(bimodal_25_125_geo_isomrphc_sets_num):
    #     bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl = (
    #         bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl[set_indx]
    #     )
    #     clnks_num = (
    #         np.shape(bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl)[0]
    #     )
    #     bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl[bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl>1.]=0.
    #     bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn_indcs = (
    #         np.empty(clnks_num, dtype=int)
    #     )
    #     bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn = (
    #         np.empty(clnks_num)
    #     )
    #     for clnk_indx in range(clnks_num):
    #         transtn_indx = np.argmax(
    #             bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl[clnk_indx])
    #         bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn_indcs[clnk_indx] = (
    #             transtn_indx
    #         )
    #         bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn[clnk_indx] = (
    #             bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl[clnk_indx, transtn_indx]
    #         )
    #     crit_indx = np.argmin(
    #         bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn_indcs)
    #     crit_transtn_indx = (
    #         bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn_indcs[crit_indx]
    #     )
    #     bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn_indcs[set_indx] = (
    #         crit_transtn_indx
    #     )
    #     bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn[set_indx] = (
    #         bimodal_25_125_max_gamma_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_transtn[crit_indx]
    #     )
    # crit_indx = np.argmin(
    #     bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn_indcs)
    # crit_transtn_indx = (
    #     bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn_indcs[crit_indx]
    # )
    # bimodal_25_125_free_rot_approx_lmbda_crit_transtn = (
    #     bimodal_25_125_free_rot_approx_lmbda[crit_transtn_indx]
    # )
    # bimodal_25_125_max_gamma_clnks_free_rot_uniaxl_crit_transtn = (
    #     bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn[crit_indx]
    # )
    # print("bimodal_25_125_free_rot_approx_lmbda_crit_transtn = {}".format(bimodal_25_125_free_rot_approx_lmbda_crit_transtn))
    # print("bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn_indcs = {}".format(bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn_indcs))
    # print("bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn = {}".format(bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_so3_uniaxl_crit_transtn))
    # print("bimodal_25_125_max_gamma_clnks_free_rot_uniaxl_crit_transtn = {}".format(bimodal_25_125_max_gamma_clnks_free_rot_uniaxl_crit_transtn))

    assert np.max(bimodal_50_100_max_gamma_clnks_frame_avrg_approx_so3_uniaxl) < 1.
    for set_indx in range(bimodal_50_100_geo_isomrphc_sets_num):
        assert np.max(bimodal_50_100_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl[set_indx]) < 1.
    assert np.max(bimodal_25_125_max_gamma_clnks_frame_avrg_approx_so3_uniaxl) < 1.
    for set_indx in range(bimodal_25_125_geo_isomrphc_sets_num):
        assert np.max(bimodal_25_125_max_gamma_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl[set_indx]) < 1.

    mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename = (
        mono_75_75_filename_prefix
        + "-W_clnks_frame_avrg_so3_quad_protocol_indx_0" + ".npy"
    )
    mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename = (
        mono_75_75_filename_prefix
        + "-W_clnks_frame_avrg_so3_quad_protocol_indx_1" + ".npy"
    )
    mono_75_75_W_clnks_free_rot_uniaxl_tens_filename = (
        mono_75_75_filename_prefix + "-W_clnks_free_rot_protocol_indx_0" + ".npy"
    )
    mono_75_75_W_clnks_free_rot_uniaxl_comp_filename = (
        mono_75_75_filename_prefix + "-W_clnks_free_rot_protocol_indx_1" + ".npy"
    )

    mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl_tens = np.load(
        mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename)
    mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl_tens = (
        mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl_tens[:, 1:]
    )
    mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl_comp = np.flip(
        np.load(mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename),
        axis=1)
    mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl = np.hstack(
        (mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl_comp,
            mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl_tens))
    mono_75_75_W_clnks_free_rot_uniaxl_tens = np.load(
        mono_75_75_W_clnks_free_rot_uniaxl_tens_filename)
    mono_75_75_W_clnks_free_rot_uniaxl_tens = (
        mono_75_75_W_clnks_free_rot_uniaxl_tens[:, 1:]
    )
    mono_75_75_W_clnks_free_rot_uniaxl_comp = np.flip(
        np.load(mono_75_75_W_clnks_free_rot_uniaxl_comp_filename), axis=1)
    mono_75_75_W_clnks_free_rot_uniaxl = np.hstack(
        (mono_75_75_W_clnks_free_rot_uniaxl_comp,
            mono_75_75_W_clnks_free_rot_uniaxl_tens))

    bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        bimodal_50_100_frame_avrg_approx_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename = (
        bimodal_50_100_frame_avrg_approx_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_1" + ".npy"
    )

    bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens = np.load(
        bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename)
    bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens = (
        bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens[:, 1:]
    )
    bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp = np.flip(
        np.load(bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename),
        axis=1)
    bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl = np.hstack(
        (bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp,
            bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens))

    bimodal_50_100_W_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl = []
    for set_indx in range(bimodal_50_100_geo_isomrphc_sets_num):
        bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens_filename = (
            bimodal_50_100_free_rot_approx_filename_prefix
            + "-W_clnks_geo_isomrphc_sets_free_rot_approx_protocol_indx_0"
            + "_" + f"set_indx_{set_indx:d}" + ".npy"
        )
        bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp_filename = (
            bimodal_50_100_free_rot_approx_filename_prefix
            + "-W_clnks_geo_isomrphc_sets_free_rot_approx_protocol_indx_1"
            + "_" + f"set_indx_{set_indx:d}" + ".npy"
        )

        bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens = np.load(
            bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens_filename)
        bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens = (
            bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens[:, 1:]
        )
        bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp = np.flip(
            np.load(bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp_filename),
            axis=1)
        
        for clnk_indx in range(np.shape(bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens)[0]):
            bimodal_50_100_W_clnk_geo_isomrphc_set_free_rot_approx_uniaxl_tens_delta_first_step = (
                bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens[clnk_indx, 0]
                - bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp[clnk_indx, -1]
            )
            bimodal_50_100_W_clnk_geo_isomrphc_set_free_rot_approx_uniaxl_comp_delta_first_step = (
                bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp[clnk_indx, -2]
                - bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp[clnk_indx, -1]
            )
            bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens[clnk_indx] += (
                bimodal_50_100_W_clnk_geo_isomrphc_set_free_rot_approx_uniaxl_comp_delta_first_step
                - bimodal_50_100_W_clnk_geo_isomrphc_set_free_rot_approx_uniaxl_tens_delta_first_step
            )
        
        bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl = np.hstack(
            (bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp,
                bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens))
        
        bimodal_50_100_W_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl.append(
            bimodal_50_100_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl)
    
    bimodal_50_100_W_clnks_free_rot_approx_uniaxl = np.empty(
        (bimodal_50_100_geo_isomrphc_sets_num,
         np.shape(bimodal_50_100_free_rot_approx_lmbda)[0]))
    for set_indx in range(bimodal_50_100_geo_isomrphc_sets_num):
        bimodal_50_100_W_clnks_free_rot_approx_uniaxl[set_indx] = np.min(
            bimodal_50_100_W_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl[set_indx],
            axis=0)

    bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_1" + ".npy"
    )

    bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens = np.load(
        bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename)
    bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens = (
        bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens[:, 1:]
    )
    bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp = np.flip(
        np.load(bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename),
        axis=1)
    bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl = np.hstack(
        (bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp,
            bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens))

    bimodal_25_125_W_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl = []
    for set_indx in range(bimodal_25_125_geo_isomrphc_sets_num):
        bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens_filename = (
            bimodal_25_125_free_rot_approx_filename_prefix
            + "-W_clnks_geo_isomrphc_sets_free_rot_approx_protocol_indx_0"
            + "_" + f"set_indx_{set_indx:d}" + ".npy"
        )
        bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp_filename = (
            bimodal_25_125_free_rot_approx_filename_prefix
            + "-W_clnks_geo_isomrphc_sets_free_rot_approx_protocol_indx_1"
            + "_" + f"set_indx_{set_indx:d}" + ".npy"
        )

        bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens = np.load(
            bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens_filename)
        bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens = (
            bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens[:, 1:]
        )
        bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp = np.flip(
            np.load(bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp_filename),
            axis=1)
        
        for clnk_indx in range(np.shape(bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens)[0]):
            bimodal_25_125_W_clnk_geo_isomrphc_set_free_rot_approx_uniaxl_tens_delta_first_step = (
                bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens[clnk_indx, 0]
                - bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp[clnk_indx, -1]
            )
            bimodal_25_125_W_clnk_geo_isomrphc_set_free_rot_approx_uniaxl_comp_delta_first_step = (
                bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp[clnk_indx, -2]
                - bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp[clnk_indx, -1]
            )
            bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens[clnk_indx] += (
                bimodal_25_125_W_clnk_geo_isomrphc_set_free_rot_approx_uniaxl_comp_delta_first_step
                - bimodal_25_125_W_clnk_geo_isomrphc_set_free_rot_approx_uniaxl_tens_delta_first_step
            )
        
        bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl = np.hstack(
            (bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp,
                bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens))
        
        bimodal_25_125_W_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl.append(
            bimodal_25_125_W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl)
    
    bimodal_25_125_W_clnks_free_rot_approx_uniaxl = np.empty(
        (bimodal_25_125_geo_isomrphc_sets_num,
         np.shape(bimodal_25_125_free_rot_approx_lmbda)[0]))
    for set_indx in range(bimodal_25_125_geo_isomrphc_sets_num):
        bimodal_25_125_W_clnks_free_rot_approx_uniaxl[set_indx] = np.min(
            bimodal_25_125_W_clnks_geo_isomrphc_sets_free_rot_approx_uniaxl[set_indx],
            axis=0)

    mono_75_75_sigma_11_clnks_frame_avrg_so3_quad_uniaxl = np.gradient(
        mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl, mono_75_75_lmbda, axis=1,
        edge_order=2)
    mono_75_75_sigma_11_clnks_free_rot_uniaxl = np.gradient(
        mono_75_75_W_clnks_free_rot_uniaxl, mono_75_75_lmbda, axis=1,
        edge_order=2)

    bimodal_50_100_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl = np.gradient(
        bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl,
        bimodal_50_100_frame_avrg_approx_lmbda, axis=1, edge_order=2)
    bimodal_50_100_sigma_11_clnks_free_rot_approx_uniaxl = np.gradient(
        bimodal_50_100_W_clnks_free_rot_approx_uniaxl,
        bimodal_50_100_free_rot_approx_lmbda, axis=1, edge_order=2)

    bimodal_25_125_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl = np.gradient(
        bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl,
        bimodal_25_125_frame_avrg_approx_lmbda, axis=1, edge_order=2)
    bimodal_25_125_sigma_11_clnks_free_rot_approx_uniaxl = np.gradient(
        bimodal_25_125_W_clnks_free_rot_approx_uniaxl,
        bimodal_25_125_free_rot_approx_lmbda, axis=1, edge_order=2)

    mono_75_75_lmbda = mono_75_75_lmbda[::50]
    mono_75_75_sigma_11_clnks_frame_avrg_so3_quad_uniaxl = (
        mono_75_75_sigma_11_clnks_frame_avrg_so3_quad_uniaxl[:, ::50]
    )
    mono_75_75_sigma_11_clnks_free_rot_uniaxl = (
        mono_75_75_sigma_11_clnks_free_rot_uniaxl[:, ::50]
    )

    bimodal_50_100_frame_avrg_approx_lmbda = (
        bimodal_50_100_frame_avrg_approx_lmbda[::50]
    )
    bimodal_50_100_free_rot_approx_lmbda = (
        bimodal_50_100_free_rot_approx_lmbda[::50]
    )
    bimodal_50_100_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl = (
        bimodal_50_100_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl[:, ::50]
    )
    bimodal_50_100_sigma_11_clnks_free_rot_approx_uniaxl = (
        bimodal_50_100_sigma_11_clnks_free_rot_approx_uniaxl[:, ::50]
    )

    bimodal_25_125_frame_avrg_approx_lmbda = (
        bimodal_25_125_frame_avrg_approx_lmbda[::50]
    )
    bimodal_25_125_free_rot_approx_lmbda = (
        bimodal_25_125_free_rot_approx_lmbda[::50]
    )
    bimodal_25_125_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl = (
        bimodal_25_125_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl[:, ::50]
    )
    bimodal_25_125_sigma_11_clnks_free_rot_approx_uniaxl = (
        bimodal_25_125_sigma_11_clnks_free_rot_approx_uniaxl[:, ::50]
    )

    mono_75_75_sigma_11_ntwrk_frame_avrg_so3_quad_uniaxl = np.squeeze(
        mono_75_75_sigma_11_clnks_frame_avrg_so3_quad_uniaxl, axis=0)
    mono_75_75_sigma_11_ntwrk_free_rot_uniaxl = np.squeeze(
        mono_75_75_sigma_11_clnks_free_rot_uniaxl, axis=0)

    p_num = np.shape(p)[0]
    bimodal_50_100_frame_avrg_approx_lmbda_num = np.shape(bimodal_50_100_frame_avrg_approx_lmbda)[0]
    bimodal_50_100_free_rot_approx_lmbda_num = np.shape(bimodal_50_100_free_rot_approx_lmbda)[0]
    bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl = np.empty(
        (p_num, bimodal_50_100_frame_avrg_approx_lmbda_num))
    bimodal_50_100_sigma_11_ntwrks_free_rot_approx_uniaxl = np.empty(
        (p_num, bimodal_50_100_free_rot_approx_lmbda_num))
    for ntwrk_indx in range(p_num):
        bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx] = np.sum(
            bimodal_50_100_p_clnks[ntwrk_indx][:, np.newaxis]*bimodal_50_100_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl,
            axis=0
        )
        bimodal_50_100_sigma_11_ntwrks_free_rot_approx_uniaxl[ntwrk_indx] = np.sum(
            bimodal_50_100_p_clnks[ntwrk_indx][:, np.newaxis]*bimodal_50_100_sigma_11_clnks_free_rot_approx_uniaxl,
            axis=0
        )

    p_half_num = np.shape(p_half)[0]
    bimodal_25_125_frame_avrg_approx_lmbda_num = np.shape(bimodal_25_125_frame_avrg_approx_lmbda)[0]
    bimodal_25_125_free_rot_approx_lmbda_num = np.shape(bimodal_25_125_free_rot_approx_lmbda)[0]
    bimodal_25_125_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl = np.empty(
        (p_half_num, bimodal_25_125_frame_avrg_approx_lmbda_num))
    bimodal_25_125_sigma_11_ntwrks_free_rot_approx_uniaxl = np.empty(
        (p_half_num, bimodal_25_125_free_rot_approx_lmbda_num))
    for ntwrk_indx in range(p_half_num):
        bimodal_25_125_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx] = np.sum(
            bimodal_25_125_p_clnks[ntwrk_indx][:, np.newaxis]*bimodal_25_125_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl,
            axis=0
        )
        bimodal_25_125_sigma_11_ntwrks_free_rot_approx_uniaxl[ntwrk_indx] = np.sum(
            bimodal_25_125_p_clnks[ntwrk_indx][:, np.newaxis]*bimodal_25_125_sigma_11_clnks_free_rot_approx_uniaxl,
            axis=0
        )
    bimodal_25_125_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl = np.squeeze(
        bimodal_25_125_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl, axis=0)
    bimodal_25_125_sigma_11_ntwrk_free_rot_approx_uniaxl = np.squeeze(
        bimodal_25_125_sigma_11_ntwrks_free_rot_approx_uniaxl, axis=0)

    sigma_11_ntwrk_p_half_bimodal_ntwrk_dfrmtn_plot_fig_filename = (
        filepath
        + "JMPS_2026_fig_13a_sigma_11_ntwrk_p_half_bimodal_ntwrk_dfrmtn_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    ntwrk_linestyle = ["-", (0, (10, 10)), (0, (24, 16))]
    ntwrks_frame_avrg_approx_lmbda = [
        mono_75_75_lmbda,
        bimodal_50_100_frame_avrg_approx_lmbda,
        bimodal_25_125_frame_avrg_approx_lmbda
    ]
    ntwrks_free_rot_approx_lmbda = [
        mono_75_75_lmbda,
        bimodal_50_100_free_rot_approx_lmbda,
        bimodal_25_125_free_rot_approx_lmbda
    ]
    sigma_11_ntwrks_frame_avrg_so3_quad_uniaxl = [
        mono_75_75_sigma_11_ntwrk_frame_avrg_so3_quad_uniaxl,
        bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[p_half_indx],
        bimodal_25_125_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl
    ]
    sigma_11_ntwrks_free_rot_uniaxl = [
        mono_75_75_sigma_11_ntwrk_free_rot_uniaxl,
        bimodal_50_100_sigma_11_ntwrks_free_rot_approx_uniaxl[p_half_indx],
        bimodal_25_125_sigma_11_ntwrk_free_rot_approx_uniaxl
    ]
    for ntwrk_indx in range(np.shape(bimodal_n)[0]):
        marker = ntwrk_marker[ntwrk_indx]
        color = ntwrk_color[ntwrk_indx]
        linestyle = ntwrk_linestyle[ntwrk_indx]
        label = bimodal_n_legend[ntwrk_indx]
        ntwrk_frame_avrg_approx_lmbda = ntwrks_frame_avrg_approx_lmbda[ntwrk_indx]
        sigma_11_ntwrk_frame_avrg_so3_quad_uniaxl = sigma_11_ntwrks_frame_avrg_so3_quad_uniaxl[ntwrk_indx]
        ntwrk_free_rot_approx_lmbda = ntwrks_free_rot_approx_lmbda[ntwrk_indx]
        sigma_11_ntwrk_free_rot_uniaxl = sigma_11_ntwrks_free_rot_uniaxl[ntwrk_indx]
        if marker == "+x":
            for plusx_marker in plusx_marker_list:
                ax.scatter(
                    ntwrk_frame_avrg_approx_lmbda,
                    sigma_11_ntwrk_frame_avrg_so3_quad_uniaxl, s=markersize,
                    marker=plusx_marker, linewidth=markerlinewidth,
                    facecolors=color, clip_on=False)
        elif marker == "s.":
            ax.scatter(
                ntwrk_frame_avrg_approx_lmbda,
                sigma_11_ntwrk_frame_avrg_so3_quad_uniaxl, s=markersize,
                marker="s", linewidth=markerlinewidth, edgecolors=color,
                facecolors="None", clip_on=False)
            ax.scatter(
                ntwrk_frame_avrg_approx_lmbda,
                sigma_11_ntwrk_frame_avrg_so3_quad_uniaxl, s=dotsize,
                marker="o", linewidth=dotlinewidth, edgecolors=color,
                facecolors=color, clip_on=False)
        else:
            ax.scatter(
                ntwrk_frame_avrg_approx_lmbda,
                sigma_11_ntwrk_frame_avrg_so3_quad_uniaxl, s=markersize,
                marker=marker, linewidth=markerlinewidth, facecolors=color,
                clip_on=False)
        ax.plot(
            ntwrk_free_rot_approx_lmbda, sigma_11_ntwrk_free_rot_uniaxl,
            linestyle=linestyle, linewidth=markerlinewidth, color=color,
            clip_on=False, label=label)
    ax.legend(
        fontsize=14, labelspacing=0, markerfirst=False, frameon=False,
        loc="lower right")
    ax.set_xlim([0.0, 8.0])
    ax.set_ylim([-20.0, 120.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ax.set_yticks([-20.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    ax.set_xticklabels(["$0$", "$1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$"])
    ax.set_yticklabels(["$-20~$", "$0~$", "$20~$", "$40~$", "$60~$", "$80~$", "$100~$", "$120~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$\\Sigma_{11}/\\leftparen Mk_BT/2\\rightparen$", fontsize=16)
    fig.tight_layout()
    fig.savefig(sigma_11_ntwrk_p_half_bimodal_ntwrk_dfrmtn_plot_fig_filename)
    plt.close()

    sigma_11_ntwrk_p_bimodal_ntwrk_dfrmtn_plot_fig_filename = (
        filepath
        + "JMPS_2026_fig_13b_sigma_11_ntwrk_p_bimodal_ntwrk_dfrmtn_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    ntwrk_linestyle = [
        "-",
        (0, (2, 2)),
        (0, (4, 4)),
        (0, (12, 4, 2, 4)),
        (0, (12, 3, 2, 3, 2, 3))
    ]
    for ntwrk_indx in range(p_num):
        marker = ntwrk_marker[ntwrk_indx]
        color = ntwrk_color[ntwrk_indx]
        linestyle = ntwrk_linestyle[ntwrk_indx]
        label = p_legend[ntwrk_indx]
        if marker == "+x":
            for plusx_marker in plusx_marker_list:
                ax.scatter(
                    bimodal_50_100_frame_avrg_approx_lmbda,
                    bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx],
                    s=markersize, marker=plusx_marker,
                    linewidth=markerlinewidth, facecolors=color, clip_on=False)
        elif marker == "s.":
            ax.scatter(
                bimodal_50_100_frame_avrg_approx_lmbda,
                bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx],
                s=markersize, marker="s", linewidth=markerlinewidth,
                edgecolors=color, facecolors="None", clip_on=False)
            ax.scatter(
                bimodal_50_100_frame_avrg_approx_lmbda,
                bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx],
                s=dotsize, marker="o", linewidth=dotlinewidth, edgecolors=color,
                facecolors=color, clip_on=False)
        else:
            ax.scatter(
                bimodal_50_100_frame_avrg_approx_lmbda,
                bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx],
                s=markersize, marker=marker, linewidth=markerlinewidth,
                facecolors=color, clip_on=False)
        ax.plot(
            bimodal_50_100_free_rot_approx_lmbda,
            bimodal_50_100_sigma_11_ntwrks_free_rot_approx_uniaxl[ntwrk_indx],
            linestyle=linestyle, linewidth=markerlinewidth, color=color,
            clip_on=False, label=label)
    ax.legend(
        fontsize=14, labelspacing=0, markerfirst=False, frameon=False,
        loc="lower right")
    ax.set_xlim([0.0, 8.0])
    ax.set_ylim([-20.0, 120.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ax.set_yticks([-20.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    ax.set_xticklabels(["$0$", "$1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$"])
    ax.set_yticklabels(["$-20~$", "$0~$", "$20~$", "$40~$", "$60~$", "$80~$", "$100~$", "$120~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    inset_ax = ax.inset_axes([0.15, 0.50, 0.50, 0.30])
    lmbda_inset_ax_min = 6.25
    lmbda_inset_ax_max = 7.25
    bimodal_50_100_frame_avrg_approx_lmbda_inset_ax_indcs = (
        np.where(np.logical_and(bimodal_50_100_frame_avrg_approx_lmbda>=lmbda_inset_ax_min, bimodal_50_100_frame_avrg_approx_lmbda<=lmbda_inset_ax_max))[0]
    )
    bimodal_50_100_free_rot_approx_lmbda_inset_ax_indcs = (
        np.where(np.logical_and(bimodal_50_100_free_rot_approx_lmbda>=lmbda_inset_ax_min, bimodal_50_100_free_rot_approx_lmbda<=lmbda_inset_ax_max))[0]
    )
    for ntwrk_indx in range(p_num):
        marker = ntwrk_marker[ntwrk_indx]
        color = ntwrk_color[ntwrk_indx]
        linestyle = ntwrk_linestyle[ntwrk_indx]
        label = p_legend[ntwrk_indx]
        if marker == "+x":
            for plusx_marker in plusx_marker_list:
                inset_ax.scatter(
                    bimodal_50_100_frame_avrg_approx_lmbda[bimodal_50_100_frame_avrg_approx_lmbda_inset_ax_indcs],
                    bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx, bimodal_50_100_frame_avrg_approx_lmbda_inset_ax_indcs],
                    s=markersize, marker=plusx_marker,
                    linewidth=markerlinewidth, facecolors=color, clip_on=False)
        elif marker == "s.":
            inset_ax.scatter(
                bimodal_50_100_frame_avrg_approx_lmbda[bimodal_50_100_frame_avrg_approx_lmbda_inset_ax_indcs],
                bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx, bimodal_50_100_frame_avrg_approx_lmbda_inset_ax_indcs],
                s=markersize, marker="s", linewidth=markerlinewidth,
                edgecolors=color, facecolors="None", clip_on=False)
            inset_ax.scatter(
                bimodal_50_100_frame_avrg_approx_lmbda[bimodal_50_100_frame_avrg_approx_lmbda_inset_ax_indcs],
                bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx, bimodal_50_100_frame_avrg_approx_lmbda_inset_ax_indcs],
                s=dotsize, marker="o", linewidth=dotlinewidth, edgecolors=color,
                facecolors=color, clip_on=False)
        else:
            inset_ax.scatter(
                bimodal_50_100_frame_avrg_approx_lmbda[bimodal_50_100_frame_avrg_approx_lmbda_inset_ax_indcs],
                bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx, bimodal_50_100_frame_avrg_approx_lmbda_inset_ax_indcs],
                s=markersize, marker=marker, linewidth=markerlinewidth,
                facecolors=color, clip_on=False)
        inset_ax.plot(
            bimodal_50_100_free_rot_approx_lmbda[bimodal_50_100_free_rot_approx_lmbda_inset_ax_indcs],
            bimodal_50_100_sigma_11_ntwrks_free_rot_approx_uniaxl[ntwrk_indx, bimodal_50_100_free_rot_approx_lmbda_inset_ax_indcs],
            linestyle=linestyle, linewidth=markerlinewidth, color=color,
            clip_on=False)
    inset_ax.set_xlim([lmbda_inset_ax_min, lmbda_inset_ax_max])
    inset_ax.set_ylim([20.0, 120.0])
    inset_ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=12)
    inset_ax.set_xticks([6.4, 6.6, 6.8, 7.0, 7.2])
    inset_ax.set_yticks([20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    inset_ax.set_xticklabels(["$6.4$", "$6.6$", "$6.8$", "$7$", "$7.2$"])
    inset_ax.set_yticklabels(["$20~$", "$40~$", "$60~$", "$80~$", "$100~$", "$120~$"])
    fig.tight_layout()
    fig.savefig(sigma_11_ntwrk_p_bimodal_ntwrk_dfrmtn_plot_fig_filename)
    plt.close()

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse Kuhn-Grun end-linked polymer network elastically-effective cross-link RVE deformation analysis plotting took {execution_time} seconds to run")