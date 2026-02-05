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
        config_path="../configs/polydisperse_cufjc_networks_clnk_rves",
        config_name="config")
def main(cfg: DictConfig) -> None:
    topology = cfg.topology
    label = cfg.label

    ntwrk_linestyle = ["-", "-.", "--", ":", "-"]
    ntwrk_color = ["tab:red", "tab:orange", "tab:blue", "tab:green", "tab:purple"]
    markersize = 50
    dotsize = 0.5
    markerlinewidth = 0.5
    dotlinewidth = 0.125
    
    filepath = filepath_str("polydisperse_cufjc_networks_clnk_rves")

    sample = 0
    mono_75_75_filename_prefix = filename_str(label.workdir, "20251204", "C", sample)
    bimodal_65_85_filename_prefix = filename_str(label.workdir, "20251204", "D", sample)
    bimodal_50_100_filename_prefix = filename_str(label.workdir, "20251204", "E", sample)
    bimodal_25_125_filename_prefix = filename_str(label.workdir, "20251204", "F", sample)

    mono_75_75_n_clnks_filename = mono_75_75_filename_prefix + "-n_clnks" + ".dat"
    bimodal_65_85_n_clnks_filename = bimodal_65_85_filename_prefix + "-n_clnks" + ".dat"
    bimodal_50_100_n_clnks_filename = bimodal_50_100_filename_prefix + "-n_clnks" + ".dat"
    bimodal_25_125_n_clnks_filename = bimodal_25_125_filename_prefix + "-n_clnks" + ".dat"
    mono_75_75_n_clnks = np.loadtxt(mono_75_75_n_clnks_filename, dtype=int)
    bimodal_65_85_n_clnks = np.loadtxt(bimodal_65_85_n_clnks_filename, dtype=int)
    bimodal_50_100_n_clnks = np.loadtxt(bimodal_50_100_n_clnks_filename, dtype=int)
    bimodal_25_125_n_clnks = np.loadtxt(bimodal_25_125_n_clnks_filename, dtype=int)
    # print(mono_75_75_n_clnks)
    # print(bimodal_65_85_n_clnks
    # print(bimodal_50_100_n_clnks))
    # print(bimodal_25_125_n_clnks)

    unique_mono_75_75_n_clnks = np.unique(mono_75_75_n_clnks)
    unique_bimodal_65_85_n_clnks = np.unique(bimodal_65_85_n_clnks)
    unique_bimodal_50_100_n_clnks = np.unique(bimodal_50_100_n_clnks)
    unique_bimodal_25_125_n_clnks = np.unique(bimodal_25_125_n_clnks)
    # print(unique_mono_75_75_n_clnks)
    # print(unique_bimodal_65_85_n_clnks)
    # print(unique_bimodal_50_100_n_clnks)
    # print(unique_bimodal_25_125_n_clnks)

    assert np.allclose(unique_mono_75_75_n_clnks, np.asarray([75], dtype=int))
    assert np.allclose(unique_bimodal_65_85_n_clnks, np.asarray([65, 85], dtype=int))
    assert np.allclose(unique_bimodal_50_100_n_clnks, np.asarray([50, 100], dtype=int))
    assert np.allclose(unique_bimodal_25_125_n_clnks, np.asarray([25, 125], dtype=int))

    n_clnks_legend = [
        "$n_a = 75, n_b = 75$", "$n_a = 65, n_b = 85$", "$n_a = 50, n_b = 100$", "$n_a = 25, n_b = 125$"
    ]

    bimodal_65_85_p_filename = bimodal_65_85_filename_prefix + "-p" + ".dat"
    bimodal_50_100_p_filename = bimodal_50_100_filename_prefix + "-p" + ".dat"
    bimodal_25_125_p_filename = bimodal_25_125_filename_prefix + "-p" + ".dat"
    bimodal_65_85_p = np.loadtxt(bimodal_65_85_p_filename)
    bimodal_50_100_p = np.loadtxt(bimodal_50_100_p_filename)
    bimodal_25_125_p = np.loadtxt(bimodal_25_125_p_filename)
    # print(bimodal_65_85_p)
    # print(bimodal_50_100_p)
    # print(bimodal_25_125_p)

    assert np.allclose(bimodal_65_85_p, bimodal_50_100_p)
    assert np.allclose(bimodal_65_85_p, bimodal_25_125_p)
    assert np.allclose(bimodal_65_85_p, np.asarray([0.1, 0.25, 0.5, 0.75, 0.9]))

    p = np.asarray([0.1, 0.25, 0.5, 0.75, 0.9])
    p_legend = ["$p = 0.10$", "$p = 0.25$", "$p = 0.50$", "$p = 0.75$", "$p = 0.90$"]

    bimodal_65_85_p_clnks_filename = bimodal_65_85_filename_prefix + "-p_clnks" + ".dat"
    bimodal_50_100_p_clnks_filename = bimodal_50_100_filename_prefix + "-p_clnks" + ".dat"
    bimodal_25_125_p_clnks_filename = bimodal_25_125_filename_prefix + "-p_clnks" + ".dat"
    bimodal_65_85_p_clnks = np.loadtxt(bimodal_65_85_p_clnks_filename)
    bimodal_50_100_p_clnks = np.loadtxt(bimodal_50_100_p_clnks_filename)
    bimodal_25_125_p_clnks = np.loadtxt(bimodal_25_125_p_clnks_filename)
    # print(bimodal_65_85_p_clnks)
    # print(bimodal_50_100_p_clnks)
    # print(bimodal_25_125_p_clnks)

    assert np.shape(bimodal_65_85_p_clnks) == (
        np.shape(bimodal_65_85_p_clnks)[0], np.shape(bimodal_65_85_n_clnks)[0]
    )
    assert np.shape(bimodal_50_100_p_clnks) == (
        np.shape(bimodal_50_100_p_clnks)[0], np.shape(bimodal_50_100_n_clnks)[0]
    )
    assert np.shape(bimodal_25_125_p_clnks) == (
        np.shape(bimodal_25_125_p_clnks)[0], np.shape(bimodal_25_125_n_clnks)[0]
    )

    mono_75_75_uniaxl_tens_dfrmtn_filename = (
        mono_75_75_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    mono_75_75_uniaxl_comp_dfrmtn_filename = (
        mono_75_75_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    bimodal_65_85_uniaxl_tens_dfrmtn_filename = (
        bimodal_65_85_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    bimodal_65_85_uniaxl_comp_dfrmtn_filename = (
        bimodal_65_85_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    bimodal_50_100_uniaxl_tens_dfrmtn_filename = (
        bimodal_50_100_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    bimodal_50_100_uniaxl_comp_dfrmtn_filename = (
        bimodal_50_100_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    bimodal_25_125_uniaxl_tens_dfrmtn_filename = (
        bimodal_25_125_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    bimodal_25_125_uniaxl_comp_dfrmtn_filename = (
        bimodal_25_125_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )

    mono_75_75_uniaxl_tens_dfrmtn = np.load(mono_75_75_uniaxl_tens_dfrmtn_filename)
    mono_75_75_uniaxl_comp_dfrmtn = np.load(mono_75_75_uniaxl_comp_dfrmtn_filename)
    bimodal_65_85_uniaxl_tens_dfrmtn = np.load(bimodal_65_85_uniaxl_tens_dfrmtn_filename)
    bimodal_65_85_uniaxl_comp_dfrmtn = np.load(bimodal_65_85_uniaxl_comp_dfrmtn_filename)
    bimodal_50_100_uniaxl_tens_dfrmtn = np.load(bimodal_50_100_uniaxl_tens_dfrmtn_filename)
    bimodal_50_100_uniaxl_comp_dfrmtn = np.load(bimodal_50_100_uniaxl_comp_dfrmtn_filename)
    bimodal_25_125_uniaxl_tens_dfrmtn = np.load(bimodal_25_125_uniaxl_tens_dfrmtn_filename)
    bimodal_25_125_uniaxl_comp_dfrmtn = np.load(bimodal_25_125_uniaxl_comp_dfrmtn_filename)

    mono_75_75_lmbda = np.hstack((np.flip(mono_75_75_uniaxl_comp_dfrmtn)[:-1], mono_75_75_uniaxl_tens_dfrmtn))
    bimodal_65_85_lmbda = np.hstack((np.flip(bimodal_65_85_uniaxl_comp_dfrmtn)[:-1], bimodal_65_85_uniaxl_tens_dfrmtn))
    bimodal_50_100_lmbda = np.hstack((np.flip(bimodal_50_100_uniaxl_comp_dfrmtn)[:-1], bimodal_50_100_uniaxl_tens_dfrmtn))
    bimodal_25_125_lmbda = np.hstack((np.flip(bimodal_25_125_uniaxl_comp_dfrmtn)[:-1], bimodal_25_125_uniaxl_tens_dfrmtn))
    # print(mono_75_75_lmbda)
    # print(bimodal_65_85_lmbda)
    # print(bimodal_50_100_lmbda)
    # print(bimodal_25_125_lmbda)
    # print(np.shape(mono_75_75_lmbda))
    # print(np.shape(bimodal_65_85_lmbda))
    # print(np.shape(bimodal_50_100_lmbda))
    # print(np.shape(bimodal_25_125_lmbda))

    mono_75_75_W_clnks_free_rot_approx_uniaxl_tens_filename = (
        mono_75_75_filename_prefix + "-W_clnks_free_rot_approx_protocol_indx_0"
        + ".npy"
    )
    mono_75_75_W_clnks_free_rot_approx_uniaxl_comp_filename = (
        mono_75_75_filename_prefix + "-W_clnks_free_rot_approx_protocol_indx_1"
        + ".npy"
    )
    bimodal_65_85_W_clnks_free_rot_approx_uniaxl_tens_filename = (
        bimodal_65_85_filename_prefix + "-W_clnks_free_rot_approx_protocol_indx_0"
        + ".npy"
    )
    bimodal_65_85_W_clnks_free_rot_approx_uniaxl_comp_filename = (
        bimodal_65_85_filename_prefix + "-W_clnks_free_rot_approx_protocol_indx_1"
        + ".npy"
    )
    bimodal_50_100_W_clnks_free_rot_approx_uniaxl_tens_filename = (
        bimodal_50_100_filename_prefix + "-W_clnks_free_rot_approx_protocol_indx_0"
        + ".npy"
    )
    bimodal_50_100_W_clnks_free_rot_approx_uniaxl_comp_filename = (
        bimodal_50_100_filename_prefix + "-W_clnks_free_rot_approx_protocol_indx_1"
        + ".npy"
    )
    bimodal_25_125_W_clnks_free_rot_approx_uniaxl_tens_filename = (
        bimodal_25_125_filename_prefix + "-W_clnks_free_rot_approx_protocol_indx_0"
        + ".npy"
    )
    bimodal_25_125_W_clnks_free_rot_approx_uniaxl_comp_filename = (
        bimodal_25_125_filename_prefix + "-W_clnks_free_rot_approx_protocol_indx_1"
        + ".npy"
    )

    mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        mono_75_75_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename = (
        mono_75_75_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_1" + ".npy"
    )
    bimodal_65_85_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        bimodal_65_85_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    bimodal_65_85_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename = (
        bimodal_65_85_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_1" + ".npy"
    )
    bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        bimodal_50_100_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename = (
        bimodal_50_100_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_1" + ".npy"
    )
    bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        bimodal_25_125_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename = (
        bimodal_25_125_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_1" + ".npy"
    )

    mono_75_75_W_clnks_free_rot_approx_uniaxl_tens = np.transpose(
        np.load(mono_75_75_W_clnks_free_rot_approx_uniaxl_tens_filename))
    mono_75_75_W_clnks_free_rot_approx_uniaxl_tens = (
        mono_75_75_W_clnks_free_rot_approx_uniaxl_tens[0]
    )
    mono_75_75_W_clnks_free_rot_approx_uniaxl_comp = np.transpose(
        np.load(mono_75_75_W_clnks_free_rot_approx_uniaxl_comp_filename))
    mono_75_75_W_clnks_free_rot_approx_uniaxl_comp = (
        mono_75_75_W_clnks_free_rot_approx_uniaxl_comp[0]
    )
    mono_75_75_W_clnks_free_rot_approx_uniaxl = np.hstack((np.flip(mono_75_75_W_clnks_free_rot_approx_uniaxl_comp)[:-1], mono_75_75_W_clnks_free_rot_approx_uniaxl_tens))
    bimodal_65_85_W_clnks_free_rot_approx_uniaxl_tens = np.transpose(
        np.load(bimodal_65_85_W_clnks_free_rot_approx_uniaxl_tens_filename))
    bimodal_65_85_W_clnks_free_rot_approx_uniaxl_comp = np.transpose(
        np.load(bimodal_65_85_W_clnks_free_rot_approx_uniaxl_comp_filename))
    bimodal_65_85_W_clnks_free_rot_approx_uniaxl = np.hstack((np.flip(bimodal_65_85_W_clnks_free_rot_approx_uniaxl_comp, axis=1)[:, :-1], bimodal_65_85_W_clnks_free_rot_approx_uniaxl_tens))
    bimodal_50_100_W_clnks_free_rot_approx_uniaxl_tens = np.transpose(
        np.load(bimodal_50_100_W_clnks_free_rot_approx_uniaxl_tens_filename))
    bimodal_50_100_W_clnks_free_rot_approx_uniaxl_comp = np.transpose(
        np.load(bimodal_50_100_W_clnks_free_rot_approx_uniaxl_comp_filename))
    bimodal_50_100_W_clnks_free_rot_approx_uniaxl = np.hstack((np.flip(bimodal_50_100_W_clnks_free_rot_approx_uniaxl_comp, axis=1)[:, :-1], bimodal_50_100_W_clnks_free_rot_approx_uniaxl_tens))
    bimodal_25_125_W_clnks_free_rot_approx_uniaxl_tens = np.transpose(
        np.load(bimodal_25_125_W_clnks_free_rot_approx_uniaxl_tens_filename))
    bimodal_25_125_W_clnks_free_rot_approx_uniaxl_comp = np.transpose(
        np.load(bimodal_25_125_W_clnks_free_rot_approx_uniaxl_comp_filename))
    bimodal_25_125_W_clnks_free_rot_approx_uniaxl = np.hstack((np.flip(bimodal_25_125_W_clnks_free_rot_approx_uniaxl_comp, axis=1)[:, :-1], bimodal_25_125_W_clnks_free_rot_approx_uniaxl_tens))

    mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens = np.transpose(
        np.load(mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens = (
        mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens[0]
    )
    mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp = np.transpose(
        np.load(mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename))
    mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp = (
        mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp[0]
    )
    mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl = np.hstack((np.flip(mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp)[:-1], mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens))
    bimodal_65_85_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens = np.transpose(
        np.load(bimodal_65_85_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    bimodal_65_85_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp = np.transpose(
        np.load(bimodal_65_85_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename))
    bimodal_65_85_W_clnks_frame_avrg_approx_so3_quad_uniaxl = np.hstack((np.flip(bimodal_65_85_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp, axis=1)[:, :-1], bimodal_65_85_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens))
    bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens = np.transpose(
        np.load(bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp = np.transpose(
        np.load(bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename))
    bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl = np.hstack((np.flip(bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp, axis=1)[:, :-1], bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens))
    bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens = np.transpose(
        np.load(bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp = np.transpose(
        np.load(bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename))
    bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl = np.hstack((np.flip(bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp, axis=1)[:, :-1], bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens))

    # print(np.shape(mono_75_75_W_clnks_free_rot_approx_uniaxl))
    # print(np.shape(bimodal_65_85_W_clnks_free_rot_approx_uniaxl))
    # print(np.shape(bimodal_50_100_W_clnks_free_rot_approx_uniaxl))
    # print(np.shape(bimodal_25_125_W_clnks_free_rot_approx_uniaxl))
    # print(np.shape(mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl))
    # print(np.shape(bimodal_65_85_W_clnks_frame_avrg_approx_so3_quad_uniaxl))
    # print(np.shape(bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl))
    # print(np.shape(bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl))

    # print(bimodal_65_85_W_clnks_free_rot_approx_uniaxl)
    # print(bimodal_50_100_W_clnks_free_rot_approx_uniaxl)
    # print(bimodal_25_125_W_clnks_free_rot_approx_uniaxl)
    # print(bimodal_65_85_W_clnks_frame_avrg_approx_so3_quad_uniaxl)
    # print(bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl)
    # print(bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl)

    mono_75_75_sigma_11_clnks_free_rot_approx_uniaxl = np.gradient(
        mono_75_75_W_clnks_free_rot_approx_uniaxl, mono_75_75_lmbda,
        edge_order=2)
    bimodal_65_85_sigma_11_clnks_free_rot_approx_uniaxl = np.gradient(
        bimodal_65_85_W_clnks_free_rot_approx_uniaxl, bimodal_65_85_lmbda,
        axis=1, edge_order=2)
    bimodal_50_100_sigma_11_clnks_free_rot_approx_uniaxl = np.gradient(
        bimodal_50_100_W_clnks_free_rot_approx_uniaxl, bimodal_50_100_lmbda,
        axis=1, edge_order=2)
    bimodal_25_125_sigma_11_clnks_free_rot_approx_uniaxl = np.gradient(
        bimodal_25_125_W_clnks_free_rot_approx_uniaxl, bimodal_25_125_lmbda,
        axis=1, edge_order=2)
    
    mono_75_75_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl = np.gradient(
        mono_75_75_W_clnks_frame_avrg_approx_so3_quad_uniaxl, mono_75_75_lmbda,
        edge_order=2)
    bimodal_65_85_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl = np.gradient(
        bimodal_65_85_W_clnks_frame_avrg_approx_so3_quad_uniaxl, bimodal_65_85_lmbda,
        axis=1, edge_order=2)
    bimodal_50_100_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl = np.gradient(
        bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl, bimodal_50_100_lmbda,
        axis=1, edge_order=2)
    bimodal_25_125_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl = np.gradient(
        bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl, bimodal_25_125_lmbda,
        axis=1, edge_order=2)
    
    bimodal_65_85_lmbda = np.append(
        bimodal_65_85_lmbda,
        bimodal_65_85_lmbda[-1]+(bimodal_65_85_lmbda[-1]-bimodal_65_85_lmbda[-2]))
    bimodal_50_100_lmbda = np.append(
        bimodal_50_100_lmbda,
        bimodal_50_100_lmbda[-1]+(bimodal_50_100_lmbda[-1]-bimodal_50_100_lmbda[-2]))
    bimodal_25_125_lmbda = np.append(
        bimodal_25_125_lmbda,
        bimodal_25_125_lmbda[-1]+(bimodal_25_125_lmbda[-1]-bimodal_25_125_lmbda[-2]))
    
    bimodal_65_85_sigma_11_clnks_free_rot_approx_uniaxl = np.column_stack(
        (bimodal_65_85_sigma_11_clnks_free_rot_approx_uniaxl, 1e9*np.ones(np.shape(bimodal_65_85_sigma_11_clnks_free_rot_approx_uniaxl)[0])))
    bimodal_50_100_sigma_11_clnks_free_rot_approx_uniaxl = np.column_stack(
        (bimodal_50_100_sigma_11_clnks_free_rot_approx_uniaxl, 1e9*np.ones(np.shape(bimodal_50_100_sigma_11_clnks_free_rot_approx_uniaxl)[0])))
    bimodal_25_125_sigma_11_clnks_free_rot_approx_uniaxl = np.column_stack(
        (bimodal_25_125_sigma_11_clnks_free_rot_approx_uniaxl, 1e9*np.ones(np.shape(bimodal_25_125_sigma_11_clnks_free_rot_approx_uniaxl)[0])))

    bimodal_65_85_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl = np.column_stack(
        (bimodal_65_85_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl, 1e9*np.ones(np.shape(bimodal_65_85_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl)[0])))
    bimodal_50_100_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl = np.column_stack(
        (bimodal_50_100_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl, 1e9*np.ones(np.shape(bimodal_50_100_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl)[0])))
    bimodal_25_125_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl = np.column_stack(
        (bimodal_25_125_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl, 1e9*np.ones(np.shape(bimodal_25_125_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl)[0])))

    p_half_indx = np.where(p == 0.5)[0][0]
    
    sigma_11_free_rot_approx_uniaxl_p_half_response_plot_fig_filename = (
        filepath + "sigma_11_free_rot_approx_uniaxl_p_half_response_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    mono_75_75_sigma_11_ntwrk_free_rot_approx_uniaxl = (
        mono_75_75_sigma_11_clnks_free_rot_approx_uniaxl
    )
    ax.plot(
        mono_75_75_lmbda, mono_75_75_sigma_11_ntwrk_free_rot_approx_uniaxl,
        linestyle=ntwrk_linestyle[1], linewidth=markerlinewidth,
        c=ntwrk_color[1], label=n_clnks_legend[0])
    p_half_bimodal_65_85_sigma_11_ntwrk_free_rot_approx_uniaxl = np.sum(
        bimodal_65_85_p_clnks[p_half_indx][:, np.newaxis]*bimodal_65_85_sigma_11_clnks_free_rot_approx_uniaxl,
        axis=0)
    ax.plot(
        bimodal_65_85_lmbda,
        p_half_bimodal_65_85_sigma_11_ntwrk_free_rot_approx_uniaxl,
        linestyle=ntwrk_linestyle[2], linewidth=markerlinewidth,
        c=ntwrk_color[2], label=n_clnks_legend[1])
    p_half_bimodal_50_100_sigma_11_ntwrk_free_rot_approx_uniaxl = np.sum(
        bimodal_50_100_p_clnks[p_half_indx][:, np.newaxis]*bimodal_50_100_sigma_11_clnks_free_rot_approx_uniaxl,
        axis=0)
    ax.plot(
        bimodal_50_100_lmbda,
        p_half_bimodal_50_100_sigma_11_ntwrk_free_rot_approx_uniaxl,
        linestyle=ntwrk_linestyle[3], linewidth=markerlinewidth,
        c=ntwrk_color[3], label=n_clnks_legend[2])
    p_half_bimodal_25_125_sigma_11_ntwrk_free_rot_approx_uniaxl = np.sum(
        bimodal_25_125_p_clnks[p_half_indx][:, np.newaxis]*bimodal_25_125_sigma_11_clnks_free_rot_approx_uniaxl,
        axis=0)
    ax.plot(
        bimodal_25_125_lmbda,
        p_half_bimodal_25_125_sigma_11_ntwrk_free_rot_approx_uniaxl,
        linestyle=ntwrk_linestyle[4], linewidth=markerlinewidth,
        c=ntwrk_color[4], label=n_clnks_legend[3])
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="best")
    ax.set_xlim([0.5, 8.0])
    ax.set_ylim([-20.0, 120.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ax.set_yticks([-20.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    ax.set_xticklabels(["$~~1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$"])
    ax.set_yticklabels(["$-20~~$", "$0$", "$20$", "$40$", "$60$", "$80$", "$100$", "$120$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$\\Sigma_{11}^{FR}/\\leftparen Mk_BT/2\\rightparen$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        sigma_11_free_rot_approx_uniaxl_p_half_response_plot_fig_filename)
    plt.close()

    sigma_11_frame_avrg_approx_so3_quad_uniaxl_p_half_response_plot_fig_filename = (
        filepath + "sigma_11_frame_avrg_approx_so3_quad_uniaxl_p_half_response_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    mono_75_75_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl = (
        mono_75_75_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl
    )
    ax.plot(
        mono_75_75_lmbda, mono_75_75_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl,
        linestyle=ntwrk_linestyle[1], linewidth=markerlinewidth,
        c=ntwrk_color[1], label=n_clnks_legend[0])
    p_half_bimodal_65_85_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl = np.sum(
        bimodal_65_85_p_clnks[p_half_indx][:, np.newaxis]*bimodal_65_85_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl,
        axis=0)
    ax.plot(
        bimodal_65_85_lmbda,
        p_half_bimodal_65_85_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl,
        linestyle=ntwrk_linestyle[2], linewidth=markerlinewidth,
        c=ntwrk_color[2], label=n_clnks_legend[1])
    p_half_bimodal_50_100_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl = np.sum(
        bimodal_50_100_p_clnks[p_half_indx][:, np.newaxis]*bimodal_50_100_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl,
        axis=0)
    ax.plot(
        bimodal_50_100_lmbda,
        p_half_bimodal_50_100_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl,
        linestyle=ntwrk_linestyle[3], linewidth=markerlinewidth,
        c=ntwrk_color[3], label=n_clnks_legend[2])
    p_half_bimodal_25_125_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl = np.sum(
        bimodal_25_125_p_clnks[p_half_indx][:, np.newaxis]*bimodal_25_125_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl,
        axis=0)
    ax.plot(
        bimodal_25_125_lmbda,
        p_half_bimodal_25_125_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl,
        linestyle=ntwrk_linestyle[4], linewidth=markerlinewidth,
        c=ntwrk_color[4], label=n_clnks_legend[3])
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="best")
    ax.set_xlim([0.5, 8.0])
    ax.set_ylim([-20.0, 120.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ax.set_yticks([-20.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    ax.set_xticklabels(["$~~1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$"])
    ax.set_yticklabels(["$-20~~$", "$0$", "$20$", "$40$", "$60$", "$80$", "$100$", "$120$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$\\Sigma_{11}^{FA}/\\leftparen Mk_BT/2\\rightparen$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        sigma_11_frame_avrg_approx_so3_quad_uniaxl_p_half_response_plot_fig_filename)
    plt.close()

    bimodal_50_100_sigma_11_free_rot_approx_response_plot_fig_filename = (
        filepath + "bimodal_50_100_sigma_11_free_rot_approx_response_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    for p_indx in range(np.shape(p)[0]):
        bimodal_50_100_sigma_11_ntwrk_free_rot_approx_uniaxl = np.sum(
            bimodal_50_100_p_clnks[p_indx][:, np.newaxis]*bimodal_50_100_sigma_11_clnks_free_rot_approx_uniaxl,
            axis=0)
        ax.plot(
            bimodal_50_100_lmbda,
            bimodal_50_100_sigma_11_ntwrk_free_rot_approx_uniaxl,
            linestyle=ntwrk_linestyle[p_indx], linewidth=markerlinewidth,
            c=ntwrk_color[p_indx], label=p_legend[p_indx])
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="best")
    ax.set_xlim([0.5, 8.0])
    ax.set_ylim([-20.0, 120.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ax.set_yticks([-20.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    ax.set_xticklabels(["$~~1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$"])
    ax.set_yticklabels(["$-20~~$", "$0$", "$20$", "$40$", "$60$", "$80$", "$100$", "$120$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$\\Sigma_{11}^{FR}/\\leftparen Mk_BT/2\\rightparen$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        bimodal_50_100_sigma_11_free_rot_approx_response_plot_fig_filename)
    plt.close()

    bimodal_50_100_sigma_11_frame_avrg_approx_so3_quad_response_plot_fig_filename = (
        filepath + "bimodal_50_100_sigma_11_frame_avrg_approx_so3_quad_response_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    for p_indx in range(np.shape(p)[0]):
        bimodal_50_100_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl = np.sum(
            bimodal_50_100_p_clnks[p_indx][:, np.newaxis]*bimodal_50_100_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl,
            axis=0)
        ax.plot(
            bimodal_50_100_lmbda,
            bimodal_50_100_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl,
            linestyle=ntwrk_linestyle[p_indx], linewidth=markerlinewidth,
            c=ntwrk_color[p_indx], label=p_legend[p_indx])
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="best")
    ax.set_xlim([0.5, 8.0])
    ax.set_ylim([-20.0, 120.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0])
    ax.set_yticks([-20.0, 0.0, 20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    ax.set_xticklabels(["$~~1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$"])
    ax.set_yticklabels(["$-20~~$", "$0$", "$20$", "$40$", "$60$", "$80$", "$100$", "$120$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$\\Sigma_{11}^{FA}/\\leftparen Mk_BT/2\\rightparen$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        bimodal_50_100_sigma_11_frame_avrg_approx_so3_quad_response_plot_fig_filename)
    plt.close()

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse cuFJC end-linked polymer network elastically-effective cross-link RVE deformation analysis plotting took {execution_time} seconds to run")