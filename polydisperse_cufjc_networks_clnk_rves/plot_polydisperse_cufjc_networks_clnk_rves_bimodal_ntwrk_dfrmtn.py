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
    ntwrk_color = ["tab:purple", "tab:green", "tab:blue", "tab:orange", "tab:red"]
    linewidth = 1

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

    filepath = filepath_str("polydisperse_cufjc_networks_clnk_rves")

    mono_75_75_filename_prefix = filename_str(cfg.label.workdir, "20260603", "C", 0)
    bimodal_50_100_frame_avrg_approx_filename_prefix = filename_str(
        cfg.label.workdir, "20260603", "D", 0)
    bimodal_25_125_frame_avrg_approx_filename_prefix = filename_str(
        cfg.label.workdir, "20260603", "E", 0)

    mono_75_75_n_clnks_filename = mono_75_75_filename_prefix + "-n_clnks" + ".npy"
    bimodal_50_100_frame_avrg_approx_n_clnks_filename = (
        bimodal_50_100_frame_avrg_approx_filename_prefix + "-n_clnks" + ".npy"
    )
    bimodal_25_125_frame_avrg_approx_n_clnks_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix + "-n_clnks" + ".npy"
    )
    mono_75_75_n_clnks = np.load(mono_75_75_n_clnks_filename)
    bimodal_50_100_frame_avrg_approx_n_clnks = np.load(
        bimodal_50_100_frame_avrg_approx_n_clnks_filename)
    bimodal_25_125_frame_avrg_approx_n_clnks = np.load(
        bimodal_25_125_frame_avrg_approx_n_clnks_filename)
    bimodal_50_100_n_clnks = bimodal_50_100_frame_avrg_approx_n_clnks
    bimodal_25_125_n_clnks = bimodal_25_125_frame_avrg_approx_n_clnks
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
    bimodal_25_125_frame_avrg_approx_p_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix + "-p" + ".npy"
    )
    bimodal_50_100_frame_avrg_approx_p = np.load(
        bimodal_50_100_frame_avrg_approx_p_filename)
    bimodal_25_125_frame_avrg_approx_p = np.load(
        bimodal_25_125_frame_avrg_approx_p_filename)
    p = bimodal_50_100_frame_avrg_approx_p
    p_half = bimodal_25_125_frame_avrg_approx_p
    p_half_indx = np.where(p==0.5)[0][0]
    assert p_half_indx == 2
    p_legend = p_legend_func(p)

    bimodal_50_100_frame_avrg_approx_p_clnks_filename = (
        bimodal_50_100_frame_avrg_approx_filename_prefix + "-p_clnks" + ".npy"
    )
    bimodal_25_125_frame_avrg_approx_p_clnks_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix + "-p_clnks" + ".npy"
    )
    bimodal_50_100_frame_avrg_approx_p_clnks = np.load(
        bimodal_50_100_frame_avrg_approx_p_clnks_filename)
    bimodal_25_125_frame_avrg_approx_p_clnks = np.load(
        bimodal_25_125_frame_avrg_approx_p_clnks_filename)
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
    bimodal_25_125_frame_avrg_approx_uniaxl_tens_lmbda_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix
        + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    bimodal_25_125_frame_avrg_approx_uniaxl_comp_lmbda_filename = (
        bimodal_25_125_frame_avrg_approx_filename_prefix
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

    mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename = (
        mono_75_75_filename_prefix
        + "-W_clnks_frame_avrg_so3_quad_protocol_indx_0" + ".npy"
    )
    mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename = (
        mono_75_75_filename_prefix
        + "-W_clnks_frame_avrg_so3_quad_protocol_indx_1" + ".npy"
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

    mono_75_75_sigma_11_clnks_frame_avrg_so3_quad_uniaxl = np.gradient(
        mono_75_75_W_clnks_frame_avrg_so3_quad_uniaxl, mono_75_75_lmbda, axis=1,
        edge_order=2)
    bimodal_50_100_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl = np.gradient(
        bimodal_50_100_W_clnks_frame_avrg_approx_so3_quad_uniaxl,
        bimodal_50_100_frame_avrg_approx_lmbda, axis=1, edge_order=2)
    bimodal_25_125_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl = np.gradient(
        bimodal_25_125_W_clnks_frame_avrg_approx_so3_quad_uniaxl,
        bimodal_25_125_frame_avrg_approx_lmbda, axis=1, edge_order=2)

    mono_75_75_sigma_11_ntwrk_frame_avrg_so3_quad_uniaxl = np.squeeze(
        mono_75_75_sigma_11_clnks_frame_avrg_so3_quad_uniaxl, axis=0)

    p_num = np.shape(p)[0]
    bimodal_50_100_frame_avrg_approx_lmbda_num = np.shape(bimodal_50_100_frame_avrg_approx_lmbda)[0]
    bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl = np.empty(
        (p_num, bimodal_50_100_frame_avrg_approx_lmbda_num))
    for ntwrk_indx in range(p_num):
        bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx] = np.sum(
            bimodal_50_100_p_clnks[ntwrk_indx][:, np.newaxis]*bimodal_50_100_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl,
            axis=0
        )

    p_half_num = np.shape(p_half)[0]
    bimodal_25_125_frame_avrg_approx_lmbda_num = np.shape(bimodal_25_125_frame_avrg_approx_lmbda)[0]
    bimodal_25_125_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl = np.empty(
        (p_half_num, bimodal_25_125_frame_avrg_approx_lmbda_num))
    for ntwrk_indx in range(p_half_num):
        bimodal_25_125_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx] = np.sum(
            bimodal_25_125_p_clnks[ntwrk_indx][:, np.newaxis]*bimodal_25_125_sigma_11_clnks_frame_avrg_approx_so3_quad_uniaxl,
            axis=0
        )
    bimodal_25_125_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl = np.squeeze(
        bimodal_25_125_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl, axis=0)

    sigma_11_ntwrk_p_half_bimodal_ntwrk_dfrmtn_plot_fig_filename = (
        filepath + "sigma_11_ntwrk_p_half_bimodal_ntwrk_dfrmtn_plot" + ".pdf"
    )
    fig, ax = plt.subplots()
    ntwrks_frame_avrg_approx_lmbda = [
        mono_75_75_lmbda,
        bimodal_50_100_frame_avrg_approx_lmbda,
        bimodal_25_125_frame_avrg_approx_lmbda
    ]
    sigma_11_ntwrks_frame_avrg_so3_quad_uniaxl = [
        mono_75_75_sigma_11_ntwrk_frame_avrg_so3_quad_uniaxl,
        bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[p_half_indx],
        bimodal_25_125_sigma_11_ntwrk_frame_avrg_approx_so3_quad_uniaxl
    ]
    for ntwrk_indx in range(np.shape(bimodal_n)[0]):
        ax.plot(
            ntwrks_frame_avrg_approx_lmbda[ntwrk_indx],
            sigma_11_ntwrks_frame_avrg_so3_quad_uniaxl[ntwrk_indx],
            linewidth=linewidth, color=ntwrk_color[ntwrk_indx],
            label=bimodal_n_legend[ntwrk_indx])
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
        filepath + "sigma_11_ntwrk_p_bimodal_ntwrk_dfrmtn_plot" + ".pdf"
    )
    fig, ax = plt.subplots()
    for ntwrk_indx in range(p_num):
        ax.plot(
            bimodal_50_100_frame_avrg_approx_lmbda,
            bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx],
            linewidth=linewidth, color=ntwrk_color[ntwrk_indx],
            label=p_legend[ntwrk_indx])
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
    lmbda_inset_ax_max = 7.55
    bimodal_50_100_frame_avrg_approx_lmbda_inset_ax_indcs = (
        np.where(np.logical_and(bimodal_50_100_frame_avrg_approx_lmbda>=lmbda_inset_ax_min, bimodal_50_100_frame_avrg_approx_lmbda<=lmbda_inset_ax_max))[0]
    )
    for ntwrk_indx in range(p_num):
        inset_ax.plot(
            bimodal_50_100_frame_avrg_approx_lmbda[bimodal_50_100_frame_avrg_approx_lmbda_inset_ax_indcs],
            bimodal_50_100_sigma_11_ntwrks_frame_avrg_approx_so3_quad_uniaxl[ntwrk_indx, bimodal_50_100_frame_avrg_approx_lmbda_inset_ax_indcs],
            linewidth=linewidth, color=ntwrk_color[ntwrk_indx],
            label=p_legend[ntwrk_indx])
    inset_ax.set_xlim([lmbda_inset_ax_min, lmbda_inset_ax_max])
    inset_ax.set_ylim([20.0, 120.0])
    inset_ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=12)
    inset_ax.set_xticks([6.4, 6.6, 6.8, 7.0, 7.2, 7.4])
    inset_ax.set_yticks([20.0, 40.0, 60.0, 80.0, 100.0, 120.0])
    inset_ax.set_xticklabels(["$6.4$", "$6.6$", "$6.8$", "$7$", "$7.2$", "$7.4$"])
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
    print(f"Polydisperse cuFJC end-linked polymer network elastically-effective cross-link RVE deformation analysis plotting took {execution_time} seconds to run")