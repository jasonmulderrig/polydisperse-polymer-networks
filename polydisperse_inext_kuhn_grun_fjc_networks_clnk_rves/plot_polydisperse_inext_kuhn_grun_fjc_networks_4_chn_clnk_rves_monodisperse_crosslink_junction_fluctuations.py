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

    markerlinewidth = 0.5
    
    filepath = filepath_str("polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves")

    n_5_filename_prefix = filename_str(label.workdir, "20260203", "B", 0)
    n_25_filename_prefix = filename_str(label.workdir, "20260203", "C", 0)
    n_100_filename_prefix = filename_str(label.workdir, "20260203", "D", 0)

    n_5_clnks_filename = n_5_filename_prefix + "-n_clnks" + ".dat"
    n_25_clnks_filename = n_25_filename_prefix + "-n_clnks" + ".dat"
    n_100_clnks_filename = n_100_filename_prefix + "-n_clnks" + ".dat"
    n_5_clnks = np.loadtxt(n_5_clnks_filename, dtype=int)
    n_25_clnks = np.loadtxt(n_25_clnks_filename, dtype=int)
    n_100_clnks = np.loadtxt(n_100_clnks_filename, dtype=int)
    # print(n_5_clnks)
    # print(n_25_clnks)
    # print(n_100_clnks)

    k_num = np.shape(n_5_clnks)[0]
    n_5_clnk_str = "$\\leftparen"
    for chn_indx in range(k_num):
        n_5_clnk_str += f"{n_5_clnks[chn_indx]:d}"
        if chn_indx < k_num-1: n_5_clnk_str += ","
    n_5_clnk_str += "\\rightparen$"
    # print(n_5_clnk_str)

    k_num = np.shape(n_25_clnks)[0]
    n_25_clnk_str = "$\\leftparen"
    for chn_indx in range(k_num):
        n_25_clnk_str += f"{n_25_clnks[chn_indx]:d}"
        if chn_indx < k_num-1: n_25_clnk_str += ","
    n_25_clnk_str += "\\rightparen$"
    # print(n_25_clnks_label)

    k_num = np.shape(n_100_clnks)[0]
    n_100_clnk_str = "$\\leftparen"
    for chn_indx in range(k_num):
        n_100_clnk_str += f"{n_100_clnks[chn_indx]:d}"
        if chn_indx < k_num-1: n_100_clnk_str += ","
    n_100_clnk_str += "\\rightparen$"
    # print(n_100_clnks_label)

    # n=5
    uniaxl_tens_dfrmtn_filename = (
        n_5_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    lmbda = np.load(uniaxl_tens_dfrmtn_filename)
    # print(lmbda)
    # print(np.shape(lmbda))

    W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        n_5_filename_prefix
        + "-W_clnks_chns_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        n_5_filename_prefix
        + "-W_clnks_y_flucts_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        n_5_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )

    W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl = np.squeeze(
        np.load(W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl = np.squeeze(
        np.load(W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    W_clnks_frame_avrg_approx_so3_quad_uniaxl = np.squeeze(
        np.load(W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    # print(np.shape(W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl))
    # print(np.shape(W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl))
    # print(np.shape(W_clnks_frame_avrg_approx_so3_quad_uniaxl))

    W_clnk_chns_frame_avrg_approx_so3_quad_uniaxl = W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl
    W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl = W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl
    W_clnk_frame_avrg_approx_so3_quad_uniaxl = W_clnks_frame_avrg_approx_so3_quad_uniaxl

    W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop = (
        W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl
        / W_clnk_frame_avrg_approx_so3_quad_uniaxl * 100
    )

    W_clnk_frame_avrg_approx_so3_quad_uniaxl_n_5_mono_plot_fig_filename = (
        filepath + "W_clnk_frame_avrg_approx_so3_quad_uniaxl_n_5_mono_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    ax.plot(
        lmbda, W_clnk_chns_frame_avrg_approx_so3_quad_uniaxl, linestyle="-",
        linewidth=markerlinewidth,
        label="$W_{ch, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$")
    ax.plot(
        lmbda, W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl, linestyle="-",
        linewidth=markerlinewidth,
        label="$W_{flucts, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$")
    ax.plot(
        lmbda, W_clnk_frame_avrg_approx_so3_quad_uniaxl, linestyle="-",
        linewidth=markerlinewidth,
        label="$W_{c, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$")
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="upper right", bbox_to_anchor=(0.55, 1.025))
    ax.set_title(n_5_clnk_str)
    ax.set_xlim([0.5, 3.0])
    ax.set_ylim([0.0, 25.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_yticks([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    ax.set_xticklabels(["$~0.5$", "$1$", "$1.5$", "$2$", "$2.5$", "$3$"])
    ax.set_yticklabels(["$0~$", "$5~$", "$10~$", "$15~$", "$20~$", "$25~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnk_frame_avrg_approx_so3_quad_uniaxl_n_5_mono_plot_fig_filename)
    plt.close()

    W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop_n_5_mono_plot_fig_filename = (
        filepath + "W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop_n_5_mono_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    ax.plot(
        lmbda, W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop,
        linestyle="-", linewidth=markerlinewidth)
    ax.set_title(n_5_clnk_str)
    ax.set_xlim([0.5, 3.0])
    ax.set_ylim([0.0, 25.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0])
    ax.set_yticks([0.0, 5.0, 10.0, 15.0, 20.0, 25.0])
    ax.set_xticklabels(["$~0.5$", "$1$", "$1.5$", "$2$", "$2.5$", "$3$"])
    ax.set_yticklabels(["$0~$", "$5~$", "$10~$", "$15~$", "$20~$", "$25~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{flucts, \\leftparen \\cdot \\rightparen }^{FA} / W_{c, \\leftparen \\cdot \\rightparen }^{FA},~[\\%]$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop_n_5_mono_plot_fig_filename)
    plt.close()

    # n=25
    uniaxl_tens_dfrmtn_filename = (
        n_25_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    lmbda = np.load(uniaxl_tens_dfrmtn_filename)
    # print(lmbda)
    # print(np.shape(lmbda))

    W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        n_25_filename_prefix
        + "-W_clnks_chns_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        n_25_filename_prefix
        + "-W_clnks_y_flucts_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        n_25_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )

    W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl = np.squeeze(
        np.load(W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl = np.squeeze(
        np.load(W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    W_clnks_frame_avrg_approx_so3_quad_uniaxl = np.squeeze(
        np.load(W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    # print(np.shape(W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl))
    # print(np.shape(W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl))
    # print(np.shape(W_clnks_frame_avrg_approx_so3_quad_uniaxl))

    W_clnk_chns_frame_avrg_approx_so3_quad_uniaxl = W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl
    W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl = W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl
    W_clnk_frame_avrg_approx_so3_quad_uniaxl = W_clnks_frame_avrg_approx_so3_quad_uniaxl

    W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop = (
        W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl
        / W_clnk_frame_avrg_approx_so3_quad_uniaxl * 100
    )

    W_clnk_frame_avrg_approx_so3_quad_uniaxl_n_25_mono_plot_fig_filename = (
        filepath + "W_clnk_frame_avrg_approx_so3_quad_uniaxl_n_25_mono_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    ax.plot(
        lmbda, W_clnk_chns_frame_avrg_approx_so3_quad_uniaxl, linestyle="-",
        linewidth=markerlinewidth,
        label="$W_{ch, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$")
    ax.plot(
        lmbda, W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl, linestyle="-",
        linewidth=markerlinewidth,
        label="$W_{flucts, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$")
    ax.plot(
        lmbda, W_clnk_frame_avrg_approx_so3_quad_uniaxl, linestyle="-",
        linewidth=markerlinewidth,
        label="$W_{c, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$")
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="upper right", bbox_to_anchor=(0.55, 1.025))
    ax.set_title(n_25_clnk_str)
    ax.set_xlim([0.5, 5.5])
    ax.set_ylim([-5.0, 100.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
    ax.set_yticks([0.0, 20.0, 40.0, 60.0, 80.0, 100.0])
    ax.set_xticklabels(["$~0.5$", "$1$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$", "$4.5$", "$5$", "$5.5$"])
    ax.set_yticklabels(["$0~$", "$20~$", "$40~$", "$60~$", "$80~$", "$100~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnk_frame_avrg_approx_so3_quad_uniaxl_n_25_mono_plot_fig_filename)
    plt.close()

    W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop_n_25_mono_plot_fig_filename = (
        filepath + "W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop_n_25_mono_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    ax.plot(
        lmbda, W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop,
        linestyle="-", linewidth=markerlinewidth)
    ax.set_title(n_25_clnk_str)
    ax.set_xlim([0.5, 5.5])
    ax.set_ylim([-25.0, 5.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.5])
    ax.set_yticks([-25.0, -20.0, -15.0, -10.0, -5.0, 0.0, 5.0])
    ax.set_xticklabels(["$~0.5$", "$1$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$", "$4.5$", "$5$", "$5.5$"])
    ax.set_yticklabels(["$-25~$", "$-20~$", "$-15~$", "$-10~$", "$-5~$", "$0~$", "$5~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{flucts, \\leftparen \\cdot \\rightparen }^{FA} / W_{c, \\leftparen \\cdot \\rightparen }^{FA},~[\\%]$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop_n_25_mono_plot_fig_filename)
    plt.close()

    # n=100
    uniaxl_tens_dfrmtn_filename = (
        n_100_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    lmbda = np.load(uniaxl_tens_dfrmtn_filename)
    # print(lmbda)
    # print(np.shape(lmbda))

    W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        n_100_filename_prefix
        + "-W_clnks_chns_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        n_100_filename_prefix
        + "-W_clnks_y_flucts_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        n_100_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )

    W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl = np.squeeze(
        np.load(W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl = np.squeeze(
        np.load(W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    W_clnks_frame_avrg_approx_so3_quad_uniaxl = np.squeeze(
        np.load(W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    # print(np.shape(W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl))
    # print(np.shape(W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl))
    # print(np.shape(W_clnks_frame_avrg_approx_so3_quad_uniaxl))

    W_clnk_chns_frame_avrg_approx_so3_quad_uniaxl = W_clnks_chns_frame_avrg_approx_so3_quad_uniaxl
    W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl = W_clnks_y_flucts_frame_avrg_approx_so3_quad_uniaxl
    W_clnk_frame_avrg_approx_so3_quad_uniaxl = W_clnks_frame_avrg_approx_so3_quad_uniaxl

    W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop = (
        W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl
        / W_clnk_frame_avrg_approx_so3_quad_uniaxl * 100
    )

    W_clnk_frame_avrg_approx_so3_quad_uniaxl_n_100_mono_plot_fig_filename = (
        filepath + "W_clnk_frame_avrg_approx_so3_quad_uniaxl_n_100_mono_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    ax.plot(
        lmbda, W_clnk_chns_frame_avrg_approx_so3_quad_uniaxl, linestyle="-",
        linewidth=markerlinewidth,
        label="$W_{ch, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$")
    ax.plot(
        lmbda, W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl, linestyle="-",
        linewidth=markerlinewidth,
        label="$W_{flucts, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$")
    ax.plot(
        lmbda, W_clnk_frame_avrg_approx_so3_quad_uniaxl, linestyle="-",
        linewidth=markerlinewidth,
        label="$W_{c, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$")
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="upper right", bbox_to_anchor=(0.55, 1.025))
    ax.set_title(n_100_clnk_str)
    ax.set_xlim([0.5, 11.5])
    ax.set_ylim([-5.0, 350.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    ax.set_yticks([0.0, 50.0, 100.0, 150.0, 200.0, 250.0, 300.0, 350.0])
    ax.set_xticklabels(["$~1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$", "$9$", "$10$", "$11$"])
    ax.set_yticklabels(["$0~$", "$50~$", "$100~$", "$150~$", "$200~$", "$250~$", "$300~$", "$350~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnk_frame_avrg_approx_so3_quad_uniaxl_n_100_mono_plot_fig_filename)
    plt.close()

    W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop_n_100_mono_plot_fig_filename = (
        filepath + "W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop_n_100_mono_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    ax.plot(
        lmbda, W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop,
        linestyle="-", linewidth=markerlinewidth)
    ax.set_title(n_100_clnk_str)
    ax.set_xlim([0.5, 11.5])
    ax.set_ylim([-120.0, 5.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0])
    ax.set_yticks([-120.0, -100.0, -80.0, -60.0, -40.0, -20.0, 0.0])
    ax.set_xticklabels(["$~1$", "$2$", "$3$", "$4$", "$5$", "$6$", "$7$", "$8$", "$9$", "$10$", "$11$"])
    ax.set_yticklabels(["$-120~$", "$-100~$", "$-80~$", "$-60~$", "$-40~$", "$-20~$", "$0~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{flucts, \\leftparen \\cdot \\rightparen }^{FA} / W_{c, \\leftparen \\cdot \\rightparen }^{FA},~[\\%]$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnk_y_flucts_frame_avrg_approx_so3_quad_uniaxl_abs_prop_n_100_mono_plot_fig_filename)
    plt.close()

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse Kuhn-Grun end-linked polymer network elastically-effective cross-link RVE deformation analysis plotting took {execution_time} seconds to run")