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
    label = cfg.label

    clnk_color = ["tab:purple", "tab:green", "tab:blue", "tab:orange"]
    markerlinewidth = 0.5
    
    filepath = filepath_str("polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves")

    filename_prefix = filename_str(label.workdir, "20260203", "A", 0)

    n_clnks_filename = filename_prefix + "-n_clnks" + ".dat"
    n_clnks = np.loadtxt(n_clnks_filename, dtype=int)
    # print(n_clnks)

    clnks_num, k_num = np.shape(n_clnks)
    range_13_rves = range(0, 4)
    range_22_rves = range(-2, 0)
    n_clnks_legend = []
    for clnk_indx in range(clnks_num):
        clnk_str = "$\\leftparen"
        for chn_indx in range(k_num):
            clnk_str += f"{n_clnks[clnk_indx, chn_indx]:d}"
            if chn_indx < k_num-1: clnk_str += ","
        clnk_str += "\\rightparen$"
        n_clnks_legend.append(clnk_str)
    # print(n_clnks_legend)

    uniaxl_tens_dfrmtn_filename = (
        filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    uniaxl_comp_dfrmtn_filename = (
        filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )

    uniaxl_tens_dfrmtn = np.load(uniaxl_tens_dfrmtn_filename)
    uniaxl_comp_dfrmtn = np.load(uniaxl_comp_dfrmtn_filename)

    lmbda = np.hstack((np.flip(uniaxl_comp_dfrmtn)[:-1], uniaxl_tens_dfrmtn))
    # print(lmbda)
    # print(np.shape(lmbda))

    W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        filename_prefix
        + "-W_clnks_chns_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_comp_filename = (
        filename_prefix
        + "-W_clnks_chns_frame_avrg_approx_so3_quad_protocol_indx_1" + ".npy"
    )
    W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        filename_prefix
        + "-W_clnks_y_flucts_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_comp_filename = (
        filename_prefix
        + "-W_clnks_y_flucts_frame_avrg_approx_so3_quad_protocol_indx_1" + ".npy"
    )
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename = (
        filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_1" + ".npy"
    )

    W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_tens = np.transpose(
        np.load(W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_comp = np.transpose(
        np.load(W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_comp_filename))
    W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl = np.hstack((np.flip(W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_comp, axis=1)[:, :-1], W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_tens))
    W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_tens = np.transpose(
        np.load(W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_comp = np.transpose(
        np.load(W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_comp_filename))
    W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl = np.hstack((np.flip(W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_comp, axis=1)[:, :-1], W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_tens))
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens = np.transpose(
        np.load(W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp = np.transpose(
        np.load(W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename))
    W_clnks_frame_avrg_approx_so3_quad_uniaxl = np.hstack((np.flip(W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp, axis=1)[:, :-1], W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens))
    # print(np.shape(W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl))
    # print(np.shape(W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl))
    # print(np.shape(W_clnks_frame_avrg_approx_so3_quad_uniaxl))

    W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_13_rves_plot_fig_filename = (
        filepath + "W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_13_rves_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    for clnk_indx in range_13_rves:
        color = clnk_color[clnk_indx]
        W_clnk_chns_frame_avrg_approx_so3_quad_uniaxl = W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl[clnk_indx]
        ax.plot(
            lmbda, W_clnk_chns_frame_avrg_approx_so3_quad_uniaxl,
            linestyle="-", linewidth=markerlinewidth, c=color,
            label=n_clnks_legend[clnk_indx])
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="upper right", bbox_to_anchor=(0.55, 1.025))
    ax.set_xlim([0.5, 6.5])
    ax.set_ylim([0.0, 100.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
    ax.set_yticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    ax.set_xticklabels(["$~0.5$", "$1$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$", "$4.5$", "$5$", "$5.5$", "$6$", "$6.5$"])
    ax.set_yticklabels(["$0~$", "$20~$", "$40~$", "$60~$", "$80~$", "$100~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{ch, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_13_rves_plot_fig_filename)
    plt.close()

    W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_22_rves_plot_fig_filename = (
        filepath + "W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_22_rves_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    plt_format_indx = 0
    for clnk_indx in range_22_rves:
        color = clnk_color[plt_format_indx]
        W_clnk_chns_frame_avrg_approx_so3_quad_uniaxl = W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl[clnk_indx]
        ax.plot(
            lmbda, W_clnk_chns_frame_avrg_approx_so3_quad_uniaxl,
            linestyle="-", linewidth=markerlinewidth, c=color,
            label=n_clnks_legend[clnk_indx])
        plt_format_indx += 1
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="upper right", bbox_to_anchor=(0.525, 1.025))
    ax.set_xlim([0.5, 6.5])
    ax.set_ylim([0.0, 100.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
    ax.set_yticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    ax.set_xticklabels(["$~0.5$", "$1$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$", "$4.5$", "$5$", "$5.5$", "$6$", "$6.5$"])
    ax.set_yticklabels(["$0~$", "$20~$", "$40~$", "$60~$", "$80~$", "$100~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{ch, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_22_rves_plot_fig_filename)
    plt.close()

    W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_13_rves_plot_fig_filename = (
        filepath + "W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_13_rves_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    for clnk_indx in range_13_rves:
        color = clnk_color[clnk_indx]
        W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl = W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl[clnk_indx]
        ax.plot(
            lmbda, W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl,
            linestyle="-", linewidth=markerlinewidth, c=color,
            label=n_clnks_legend[clnk_indx])
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="upper right", bbox_to_anchor=(0.55, 1.025))
    ax.set_xlim([0.5, 6.5])
    ax.set_ylim([-4.0, -2.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
    ax.set_yticks([-4.0, -3.5, -3.0, -2.5, -2.0])
    ax.set_xticklabels(["$~0.5$", "$1$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$", "$4.5$", "$5$", "$5.5$", "$6$", "$6.5$"])
    ax.set_yticklabels(["$-4~$", "$-3.5~$", "$-3~$", "$-2.5~$", "$-2~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{flucts, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_13_rves_plot_fig_filename)
    plt.close()

    W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_22_rves_plot_fig_filename = (
        filepath + "W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_22_rves_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    plt_format_indx = 0
    for clnk_indx in range_22_rves:
        color = clnk_color[plt_format_indx]
        W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl = W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl[clnk_indx]
        ax.plot(
            lmbda, W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl,
            linestyle="-", linewidth=markerlinewidth, c=color,
            label=n_clnks_legend[clnk_indx])
        plt_format_indx += 1
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="upper right", bbox_to_anchor=(0.525, 1.025))
    ax.set_xlim([0.5, 6.5])
    ax.set_ylim([-4.0, -2.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
    ax.set_yticks([-4.0, -3.5, -3.0, -2.5, -2.0])
    ax.set_xticklabels(["$~0.5$", "$1$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$", "$4.5$", "$5$", "$5.5$", "$6$", "$6.5$"])
    ax.set_yticklabels(["$-4~$", "$-3.5~$", "$-3~$", "$-2.5~$", "$-2~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{flucts, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_22_rves_plot_fig_filename)
    plt.close()

    W_clnks_frame_avrg_approx_so3_quad_uniaxl_13_rves_plot_fig_filename = (
        filepath + "W_clnks_frame_avrg_approx_so3_quad_uniaxl_13_rves_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    for clnk_indx in range_13_rves:
        color = clnk_color[clnk_indx]
        W_clnk_frame_avrg_approx_so3_quad_uniaxl = W_clnks_frame_avrg_approx_so3_quad_uniaxl[clnk_indx]
        ax.plot(
            lmbda, W_clnk_frame_avrg_approx_so3_quad_uniaxl,
            linestyle="-", linewidth=markerlinewidth, c=color,
            label=n_clnks_legend[clnk_indx])
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="upper right", bbox_to_anchor=(0.55, 1.025))
    ax.set_xlim([0.5, 6.5])
    ax.set_ylim([0.0, 100.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
    ax.set_yticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    ax.set_xticklabels(["$~0.5$", "$1$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$", "$4.5$", "$5$", "$5.5$", "$6$", "$6.5$"])
    ax.set_yticklabels(["$0~$", "$20~$", "$40~$", "$60~$", "$80~$", "$100~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{c, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnks_frame_avrg_approx_so3_quad_uniaxl_13_rves_plot_fig_filename)
    plt.close()

    W_clnks_frame_avrg_approx_so3_quad_uniaxl_22_rves_plot_fig_filename = (
        filepath + "W_clnks_frame_avrg_approx_so3_quad_uniaxl_22_rves_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    plt_format_indx = 0
    for clnk_indx in range_22_rves:
        color = clnk_color[plt_format_indx]
        W_clnk_frame_avrg_approx_so3_quad_uniaxl = W_clnks_frame_avrg_approx_so3_quad_uniaxl[clnk_indx]
        ax.plot(
            lmbda, W_clnk_frame_avrg_approx_so3_quad_uniaxl,
            linestyle="-", linewidth=markerlinewidth, c=color,
            label=n_clnks_legend[clnk_indx])
        plt_format_indx += 1
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="upper right", bbox_to_anchor=(0.525, 1.025))
    ax.set_xlim([0.5, 6.5])
    ax.set_ylim([0.0, 100.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 6.5])
    ax.set_yticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    ax.set_xticklabels(["$~0.5$", "$1$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$", "$4.5$", "$5$", "$5.5$", "$6$", "$6.5$"])
    ax.set_yticklabels(["$0~$", "$20~$", "$40~$", "$60~$", "$80~$", "$100~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{c, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnks_frame_avrg_approx_so3_quad_uniaxl_22_rves_plot_fig_filename)
    plt.close()

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse Kuhn-Grun end-linked polymer network elastically-effective cross-link RVE deformation analysis plotting took {execution_time} seconds to run")