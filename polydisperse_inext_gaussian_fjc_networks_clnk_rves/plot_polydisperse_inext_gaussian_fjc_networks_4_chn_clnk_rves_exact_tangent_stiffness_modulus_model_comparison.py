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
        config_path="../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves",
        config_name="config")
def main(cfg: DictConfig) -> None:
    topology = cfg.topology
    label = cfg.label

    clnk_color = ["black", "tab:purple", "tab:green", "tab:blue", "tab:orange"]
    markerlinewidth = 0.5

    filepath = filepath_str("polydisperse_inext_gaussian_fjc_networks_clnk_rves")

    sample = 0
    filename_prefix = filename_str(label.workdir, "20251204", "H", sample)
    n_clnks_filename = filename_prefix + "-n_clnks" + ".dat"
    n_clnks = np.loadtxt(n_clnks_filename, dtype=int)
    # print(n_clnks)

    clnks_num, k_num = np.shape(n_clnks)
    range_13_rves = range(0, 2)
    range_22_rves = range(2, clnks_num)
    n_clnks_13_rves_legend = []
    for clnk_indx in range_13_rves:
        clnk_str = "\\leftparen"
        for chn_indx in range(k_num):
            clnk_str += f"{n_clnks[clnk_indx, chn_indx]:d}"
            if chn_indx < k_num-1: clnk_str += ","
        clnk_str += "\\rightparen$"
        n_clnks_13_rves_legend.append(clnk_str)
    n_clnks_22_rves_legend = []
    for clnk_indx in range_22_rves:
        clnk_str = "\\leftparen"
        for chn_indx in range(k_num):
            clnk_str += f"{n_clnks[clnk_indx, chn_indx]:d}"
            if chn_indx < k_num-1: clnk_str += ","
        clnk_str += "\\rightparen$"
        n_clnks_22_rves_legend.append(clnk_str)
    # print(n_clnks_13_rves_legend)
    # print(n_clnks_22_rves_legend)

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

    W_clnks_free_rot_uniaxl_tens_filename = (
        filename_prefix + "-W_clnks_free_rot_protocol_indx_0" + ".npy"
    )
    W_clnks_free_rot_uniaxl_comp_filename = (
        filename_prefix + "-W_clnks_free_rot_protocol_indx_1" + ".npy"
    )

    W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename = (
        filename_prefix + "-W_clnks_frame_avrg_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename = (
        filename_prefix + "-W_clnks_frame_avrg_so3_quad_protocol_indx_1" + ".npy"
    )

    W_clnks_free_rot_uniaxl_tens = np.transpose(
        np.load(W_clnks_free_rot_uniaxl_tens_filename))
    W_clnks_free_rot_uniaxl_comp = np.transpose(
        np.load(W_clnks_free_rot_uniaxl_comp_filename))
    W_clnks_free_rot_uniaxl = np.hstack((np.flip(W_clnks_free_rot_uniaxl_comp, axis=1)[:, :-1], W_clnks_free_rot_uniaxl_tens))

    W_clnks_frame_avrg_so3_quad_uniaxl_tens = np.transpose(
        np.load(W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename))
    W_clnks_frame_avrg_so3_quad_uniaxl_comp = np.transpose(
        np.load(W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename))
    W_clnks_frame_avrg_so3_quad_uniaxl = np.hstack((np.flip(W_clnks_frame_avrg_so3_quad_uniaxl_comp, axis=1)[:, :-1], W_clnks_frame_avrg_so3_quad_uniaxl_tens))
    
    # print(np.shape(W_clnks_free_rot_uniaxl))
    # print(np.shape(W_clnks_frame_avrg_so3_quad_uniaxl))

    sigma_11_clnks_free_rot_uniaxl = np.gradient(
        W_clnks_free_rot_uniaxl, lmbda, axis=1, edge_order=2)
    sigma_11_clnks_frame_avrg_so3_quad_uniaxl = np.gradient(
        W_clnks_frame_avrg_so3_quad_uniaxl, lmbda, axis=1, edge_order=2)

    E_clnks_free_rot_uniaxl = np.gradient(
        sigma_11_clnks_free_rot_uniaxl, lmbda, axis=1, edge_order=2)
    E_clnks_frame_avrg_so3_quad_uniaxl = np.gradient(
        sigma_11_clnks_frame_avrg_so3_quad_uniaxl, lmbda, axis=1, edge_order=2)
    
    # print(np.shape(E_clnks_free_rot_uniaxl))
    # print(np.shape(E_clnks_frame_avrg_so3_quad_uniaxl))

    W_clnks_model_comparison_uniaxl_13_rves_plot_fig_filename = (
        filepath + "W_clnks_model_comparison_uniaxl_13_rves_plot" + ".pdf"
    )
    fig, ax = plt.subplots()
    ax.plot(
        lmbda, W_clnks_free_rot_uniaxl[0], linestyle="--",
        linewidth=markerlinewidth, c=clnk_color[0],
        label="$FR,~"+n_clnks_13_rves_legend[0])
    ax.plot(
        lmbda, W_clnks_free_rot_uniaxl[1], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[2],
        label="$FR,~"+n_clnks_13_rves_legend[1])
    ax.plot(
        lmbda, W_clnks_frame_avrg_so3_quad_uniaxl[0], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[3],
        label="$FA,~"+n_clnks_13_rves_legend[0])
    ax.plot(
        lmbda, W_clnks_frame_avrg_so3_quad_uniaxl[1], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[4],
        label="$FA,~"+n_clnks_13_rves_legend[1])
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="best")
    ax.set_xlim([0.4, 1.4])
    ax.set_ylim([5.0, 9.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_yticks([5.0, 6.0, 7.0, 8.0, 9.0])
    ax.set_xticklabels(["$~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$"])
    ax.set_yticklabels(["$5~$", "$6~$", "$7~$", "$8~$", "$9~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{c, \\leftparen \\cdot \\rightparen }/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(W_clnks_model_comparison_uniaxl_13_rves_plot_fig_filename)
    plt.close()

    W_clnks_model_comparison_uniaxl_22_rves_plot_fig_filename = (
        filepath + "W_clnks_model_comparison_uniaxl_22_rves_plot" + ".pdf"
    )
    fig, ax = plt.subplots()
    ax.plot(
        lmbda, W_clnks_free_rot_uniaxl[2], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[1],
        label="$FR,~"+n_clnks_22_rves_legend[0])
    ax.plot(
        lmbda, W_clnks_free_rot_uniaxl[3], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[2],
        label="$FR,~"+n_clnks_22_rves_legend[1])
    ax.plot(
        lmbda, W_clnks_frame_avrg_so3_quad_uniaxl[2], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[3],
        label="$FA,~"+n_clnks_22_rves_legend[0])
    ax.plot(
        lmbda, W_clnks_frame_avrg_so3_quad_uniaxl[3], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[4],
        label="$FA,~"+n_clnks_22_rves_legend[1])
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="best")
    ax.set_xlim([0.4, 1.4])
    ax.set_ylim([5.0, 9.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_yticks([5.0, 6.0, 7.0, 8.0, 9.0])
    ax.set_xticklabels(["$~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$"])
    ax.set_yticklabels(["$5~$", "$6~$", "$7~$", "$8~$", "$9~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{c, \\leftparen \\cdot \\rightparen }/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(W_clnks_model_comparison_uniaxl_22_rves_plot_fig_filename)
    plt.close()

    E_clnks_model_comparison_uniaxl_13_rves_plot_fig_filename = (
        filepath + "E_clnks_model_comparison_uniaxl_13_rves_plot" + ".pdf"
    )
    fig, ax = plt.subplots()
    ax.plot(
        lmbda, E_clnks_free_rot_uniaxl[0], linestyle="--",
        linewidth=markerlinewidth, c=clnk_color[0],
        label="$FR,~"+n_clnks_13_rves_legend[0])
    ax.plot(
        lmbda, E_clnks_free_rot_uniaxl[1], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[2],
        label="$FR,~"+n_clnks_13_rves_legend[1])
    ax.plot(
        lmbda, E_clnks_frame_avrg_so3_quad_uniaxl[0], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[3],
        label="$FA,~"+n_clnks_13_rves_legend[0])
    ax.plot(
        lmbda, E_clnks_frame_avrg_so3_quad_uniaxl[1], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[4],
        label="$FA,~"+n_clnks_13_rves_legend[1])
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="best")
    ax.set_xlim([0.4, 1.4])
    ax.set_ylim([-100.0, 75.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_yticks([-100.0, -75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0])
    ax.set_xticklabels(["$~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$"])
    ax.set_yticklabels(["$-100~$", "$-75~$", "$-50~$", "$-25~$", "$0~$", "$25~$", "$50~$", "$75~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$E_{\\leftparen \\cdot \\rightparen }/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(E_clnks_model_comparison_uniaxl_13_rves_plot_fig_filename)
    plt.close()

    E_clnks_model_comparison_uniaxl_22_rves_plot_fig_filename = (
        filepath + "E_clnks_model_comparison_uniaxl_22_rves_plot" + ".pdf"
    )
    fig, ax = plt.subplots()
    ax.plot(
        lmbda, E_clnks_free_rot_uniaxl[2], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[1],
        label="$FR,~"+n_clnks_22_rves_legend[0])
    ax.plot(
        lmbda, E_clnks_free_rot_uniaxl[3], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[2],
        label="$FR,~"+n_clnks_22_rves_legend[1])
    ax.plot(
        lmbda, E_clnks_frame_avrg_so3_quad_uniaxl[2], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[3],
        label="$FA,~"+n_clnks_22_rves_legend[0])
    ax.plot(
        lmbda, E_clnks_frame_avrg_so3_quad_uniaxl[3], linestyle="-",
        linewidth=markerlinewidth, c=clnk_color[4],
        label="$FA,~"+n_clnks_22_rves_legend[1])
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="best")
    ax.set_xlim([0.4, 1.4])
    ax.set_ylim([-100.0, 75.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4])
    ax.set_yticks([-100.0, -75.0, -50.0, -25.0, 0.0, 25.0, 50.0, 75.0])
    ax.set_xticklabels(["$~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$"])
    ax.set_yticklabels(["$-100~$", "$-75~$", "$-50~$", "$-25~$", "$0~$", "$25~$", "$50~$", "$75~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$E_{\\leftparen \\cdot \\rightparen }/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(E_clnks_model_comparison_uniaxl_22_rves_plot_fig_filename)
    plt.close()

    # # Save data in .csv files for plotting reproduction
    # lmbda_and_W_clnks_model_comparison_uniaxl_13_rves_filename = (
    #     filepath
    #     + "lmbda_and_W_clnks_model_comparison_uniaxl_13_rves" + ".csv"
    # )
    # lmbda_and_W_clnks_model_comparison_uniaxl_22_rves_filename = (
    #     filepath
    #     + "lmbda_and_W_clnks_model_comparison_uniaxl_22_rves" + ".csv"
    # )
    # lmbda_and_E_clnks_model_comparison_uniaxl_13_rves_filename = (
    #     filepath
    #     + "lmbda_and_E_clnks_model_comparison_uniaxl_13_rves" + ".csv"
    # )
    # lmbda_and_E_clnks_model_comparison_uniaxl_22_rves_filename = (
    #     filepath
    #     + "lmbda_and_E_clnks_model_comparison_uniaxl_22_rves" + ".csv"
    # )
    
    # lmbda_and_W_clnks_model_comparison_uniaxl_13_rves = lmbda.copy()
    # lmbda_and_W_clnks_model_comparison_uniaxl_22_rves = lmbda.copy()
    # lmbda_and_E_clnks_model_comparison_uniaxl_13_rves = lmbda.copy()
    # lmbda_and_E_clnks_model_comparison_uniaxl_22_rves = lmbda.copy()
    
    # lmbda_and_W_clnks_model_comparison_uniaxl_13_rves = np.column_stack(
    #     (lmbda_and_W_clnks_model_comparison_uniaxl_13_rves, W_clnks_free_rot_uniaxl[0]))
    # lmbda_and_W_clnks_model_comparison_uniaxl_13_rves = np.column_stack(
    #     (lmbda_and_W_clnks_model_comparison_uniaxl_13_rves, W_clnks_free_rot_uniaxl[1]))
    # lmbda_and_W_clnks_model_comparison_uniaxl_13_rves = np.column_stack(
    #     (lmbda_and_W_clnks_model_comparison_uniaxl_13_rves, W_clnks_frame_avrg_so3_quad_uniaxl[0]))
    # lmbda_and_W_clnks_model_comparison_uniaxl_13_rves = np.column_stack(
    #     (lmbda_and_W_clnks_model_comparison_uniaxl_13_rves, W_clnks_frame_avrg_so3_quad_uniaxl[1]))

    # lmbda_and_W_clnks_model_comparison_uniaxl_22_rves = np.column_stack(
    #     (lmbda_and_W_clnks_model_comparison_uniaxl_22_rves, W_clnks_free_rot_uniaxl[2]))
    # lmbda_and_W_clnks_model_comparison_uniaxl_22_rves = np.column_stack(
    #     (lmbda_and_W_clnks_model_comparison_uniaxl_22_rves, W_clnks_free_rot_uniaxl[3]))
    # lmbda_and_W_clnks_model_comparison_uniaxl_22_rves = np.column_stack(
    #     (lmbda_and_W_clnks_model_comparison_uniaxl_22_rves, W_clnks_frame_avrg_so3_quad_uniaxl[2]))
    # lmbda_and_W_clnks_model_comparison_uniaxl_22_rves = np.column_stack(
    #     (lmbda_and_W_clnks_model_comparison_uniaxl_22_rves, W_clnks_frame_avrg_so3_quad_uniaxl[3]))

    # lmbda_and_E_clnks_model_comparison_uniaxl_13_rves = np.column_stack(
    #     (lmbda_and_E_clnks_model_comparison_uniaxl_13_rves, E_clnks_free_rot_uniaxl[0]))
    # lmbda_and_E_clnks_model_comparison_uniaxl_13_rves = np.column_stack(
    #     (lmbda_and_E_clnks_model_comparison_uniaxl_13_rves, E_clnks_free_rot_uniaxl[1]))
    # lmbda_and_E_clnks_model_comparison_uniaxl_13_rves = np.column_stack(
    #     (lmbda_and_E_clnks_model_comparison_uniaxl_13_rves, E_clnks_frame_avrg_so3_quad_uniaxl[0]))
    # lmbda_and_E_clnks_model_comparison_uniaxl_13_rves = np.column_stack(
    #     (lmbda_and_E_clnks_model_comparison_uniaxl_13_rves, E_clnks_frame_avrg_so3_quad_uniaxl[1]))

    # lmbda_and_E_clnks_model_comparison_uniaxl_22_rves = np.column_stack(
    #     (lmbda_and_E_clnks_model_comparison_uniaxl_22_rves, E_clnks_free_rot_uniaxl[2]))
    # lmbda_and_E_clnks_model_comparison_uniaxl_22_rves = np.column_stack(
    #     (lmbda_and_E_clnks_model_comparison_uniaxl_22_rves, E_clnks_free_rot_uniaxl[3]))
    # lmbda_and_E_clnks_model_comparison_uniaxl_22_rves = np.column_stack(
    #     (lmbda_and_E_clnks_model_comparison_uniaxl_22_rves, E_clnks_frame_avrg_so3_quad_uniaxl[2]))
    # lmbda_and_E_clnks_model_comparison_uniaxl_22_rves = np.column_stack(
    #     (lmbda_and_E_clnks_model_comparison_uniaxl_22_rves, E_clnks_frame_avrg_so3_quad_uniaxl[3]))
    
    # np.savetxt(
    #     lmbda_and_W_clnks_model_comparison_uniaxl_13_rves_filename,
    #     lmbda_and_W_clnks_model_comparison_uniaxl_13_rves, delimiter=",")
    # np.savetxt(
    #     lmbda_and_W_clnks_model_comparison_uniaxl_22_rves_filename,
    #     lmbda_and_W_clnks_model_comparison_uniaxl_22_rves, delimiter=",")
    # np.savetxt(
    #     lmbda_and_E_clnks_model_comparison_uniaxl_13_rves_filename,
    #     lmbda_and_E_clnks_model_comparison_uniaxl_13_rves, delimiter=",")
    # np.savetxt(
    #     lmbda_and_E_clnks_model_comparison_uniaxl_22_rves_filename,
    #     lmbda_and_E_clnks_model_comparison_uniaxl_22_rves, delimiter=",")

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse Gaussian end-linked polymer network elastically-effective cross-link RVE deformation analysis plotting took {execution_time} seconds to run")