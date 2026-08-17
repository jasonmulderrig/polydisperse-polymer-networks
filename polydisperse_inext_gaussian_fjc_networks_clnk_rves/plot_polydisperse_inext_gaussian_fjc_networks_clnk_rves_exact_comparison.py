# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerBase
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
    clnk_color = ["tab:purple", "tab:green"]
    clnk_marker = ["^", "s"]
    markersize = 50
    dotsize = 0.5
    markerlinewidth = 0.5
    dotlinewidth = 0.125

    def n_clnks_legend_func(n_clnks):
        clnks_num, k_num = np.shape(n_clnks)
        n_clnks_legend = []
        for clnk_indx in range(clnks_num):
            clnk_str = "$\\leftparen"
            for chn_indx in range(k_num):
                clnk_str += f"{int(n_clnks[clnk_indx, chn_indx]):d}"
                if chn_indx < k_num-1: clnk_str += ","
            clnk_str += "\\rightparen$"
            n_clnks_legend.append(clnk_str)
        return n_clnks_legend

    class HandlerCompositeMarker(HandlerBase):
        def create_artists(
                self,
                legend,
                orig_handle,
                x_descent,
                y_descent,
                width,
                height,
                fontsize,
                trans):
            marker, color = orig_handle
            center_x = x_descent + width / 2
            center_y = y_descent + height / 2
            artists = []

            artists.append(
                plt.Line2D(
                    [center_x], [center_y], marker=marker,
                    markerfacecolor="None", markeredgecolor=color,
                    markersize=markersize/7, markeredgewidth=markerlinewidth,
                    linestyle="None", transform=trans))
            artists.append(
                plt.Line2D(
                    [center_x], [center_y], marker="o", color=color,
                    markersize=dotsize, markeredgewidth=dotlinewidth,
                    linestyle="None", transform=trans))
            return artists
    
    filepath = filepath_str("polydisperse_inext_gaussian_fjc_networks_clnk_rves")

    k_4_clnks_filename_prefix = filename_str(cfg.label.workdir, "20260603", "F", 0)
    k_6_clnks_filename_prefix = filename_str(cfg.label.workdir, "20260603", "G", 0)

    k_4_clnks_n_clnks_filename = k_4_clnks_filename_prefix + "-n_clnks" + ".npy"
    k_6_clnks_n_clnks_filename = k_6_clnks_filename_prefix + "-n_clnks" + ".npy"
    k_4_clnks_n_clnks = np.load(k_4_clnks_n_clnks_filename)
    k_6_clnks_n_clnks = np.load(k_6_clnks_n_clnks_filename)

    k_4_n_clnks_legend = n_clnks_legend_func(k_4_clnks_n_clnks)
    k_6_n_clnks_legend = n_clnks_legend_func(k_6_clnks_n_clnks)

    half_n_clnks_legend = []
    half_n_clnks_legend.append(k_4_n_clnks_legend[0])
    half_n_clnks_legend.append(k_6_n_clnks_legend[0])
    one_unique_n_clnks_legend = []
    one_unique_n_clnks_legend.append(k_4_n_clnks_legend[1])
    one_unique_n_clnks_legend.append(k_6_n_clnks_legend[1])

    k_4_clnks_uniaxl_tens_lmbda_filename = (
        k_4_clnks_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    k_4_clnks_uniaxl_comp_lmbda_filename = (
        k_4_clnks_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    k_6_clnks_uniaxl_tens_lmbda_filename = (
        k_6_clnks_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    k_6_clnks_uniaxl_comp_lmbda_filename = (
        k_6_clnks_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )

    k_4_clnks_uniaxl_tens_lmbda = np.load(k_4_clnks_uniaxl_tens_lmbda_filename)
    k_4_clnks_uniaxl_tens_lmbda = k_4_clnks_uniaxl_tens_lmbda[1:]
    k_4_clnks_uniaxl_comp_lmbda = np.flip(
        np.load(k_4_clnks_uniaxl_comp_lmbda_filename))
    k_4_clnks_lmbda = np.hstack(
        (k_4_clnks_uniaxl_comp_lmbda, k_4_clnks_uniaxl_tens_lmbda))
    k_6_clnks_uniaxl_tens_lmbda = np.load(k_6_clnks_uniaxl_tens_lmbda_filename)
    k_6_clnks_uniaxl_tens_lmbda = k_6_clnks_uniaxl_tens_lmbda[1:]
    k_6_clnks_uniaxl_comp_lmbda = np.flip(
        np.load(k_6_clnks_uniaxl_comp_lmbda_filename))
    k_6_clnks_lmbda = np.hstack(
        (k_6_clnks_uniaxl_comp_lmbda, k_6_clnks_uniaxl_tens_lmbda))

    k_4_clnks_W_clnks_free_rot_uniaxl_tens_filename = (
        k_4_clnks_filename_prefix + "-W_clnks_free_rot_protocol_indx_0" + ".npy"
    )
    k_4_clnks_W_clnks_free_rot_uniaxl_comp_filename = (
        k_4_clnks_filename_prefix + "-W_clnks_free_rot_protocol_indx_1" + ".npy"
    )
    k_6_clnks_W_clnks_free_rot_uniaxl_tens_filename = (
        k_6_clnks_filename_prefix + "-W_clnks_free_rot_protocol_indx_0" + ".npy"
    )
    k_6_clnks_W_clnks_free_rot_uniaxl_comp_filename = (
        k_6_clnks_filename_prefix + "-W_clnks_free_rot_protocol_indx_1" + ".npy"
    )

    k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename = (
        k_4_clnks_filename_prefix
        + "-W_clnks_frame_avrg_so3_quad_protocol_indx_0" + ".npy"
    )
    k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename = (
        k_4_clnks_filename_prefix
        + "-W_clnks_frame_avrg_so3_quad_protocol_indx_1" + ".npy"
    )
    k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename = (
        k_6_clnks_filename_prefix
        + "-W_clnks_frame_avrg_so3_quad_protocol_indx_0" + ".npy"
    )
    k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename = (
        k_6_clnks_filename_prefix
        + "-W_clnks_frame_avrg_so3_quad_protocol_indx_1" + ".npy"
    )

    k_4_clnks_W_clnks_free_rot_uniaxl_tens = np.load(
        k_4_clnks_W_clnks_free_rot_uniaxl_tens_filename)
    k_4_clnks_W_clnks_free_rot_uniaxl_tens = (
        k_4_clnks_W_clnks_free_rot_uniaxl_tens[:, 1:]
    )
    k_4_clnks_W_clnks_free_rot_uniaxl_comp = np.flip(
        np.load(k_4_clnks_W_clnks_free_rot_uniaxl_comp_filename), axis=1)
    k_4_clnks_W_clnks_free_rot_uniaxl = np.hstack(
        (k_4_clnks_W_clnks_free_rot_uniaxl_comp,
         k_4_clnks_W_clnks_free_rot_uniaxl_tens))
    k_6_clnks_W_clnks_free_rot_uniaxl_tens = np.load(
        k_6_clnks_W_clnks_free_rot_uniaxl_tens_filename)
    k_6_clnks_W_clnks_free_rot_uniaxl_tens = (
        k_6_clnks_W_clnks_free_rot_uniaxl_tens[:, 1:]
    )
    k_6_clnks_W_clnks_free_rot_uniaxl_comp = np.flip(
        np.load(k_6_clnks_W_clnks_free_rot_uniaxl_comp_filename), axis=1)
    k_6_clnks_W_clnks_free_rot_uniaxl = np.hstack(
        (k_6_clnks_W_clnks_free_rot_uniaxl_comp,
         k_6_clnks_W_clnks_free_rot_uniaxl_tens))

    k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens = np.load(
        k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename)
    k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens = (
        k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens[:, 1:]
    )
    k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp = np.flip(
        np.load(k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename),
        axis=1)
    k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl = np.hstack(
        (k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp,
         k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens))
    k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens = np.load(
        k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename)
    k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens = (
        k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens[:, 1:]
    )
    k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp = np.flip(
        np.load(k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename),
        axis=1)
    k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl = np.hstack(
        (k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp,
         k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens))

    W_clnks_free_rot_uniaxl_half_n_clnks_plot_fig_filename = (
        filepath
        + "JMPS_2026_fig_7a_W_clnks_free_rot_uniaxl_half_n_clnks_plot" + ".pdf"
    )
    fig, ax = plt.subplots()
    handles = []
    plt_format_indx = 0
    marker = clnk_marker[plt_format_indx]
    color = clnk_color[plt_format_indx]
    ax.scatter(
        k_4_clnks_lmbda, k_4_clnks_W_clnks_free_rot_uniaxl[0]/4,
        s=markersize, marker=marker, linewidth=markerlinewidth,
        edgecolors=color, facecolors="None", clip_on=False)
    ax.scatter(
        k_4_clnks_lmbda, k_4_clnks_W_clnks_free_rot_uniaxl[0]/4,
        s=dotsize, marker="o", linewidth=dotlinewidth,
        edgecolors=color, facecolors=color, clip_on=False)
    handles.append((marker, color))
    plt_format_indx += 1
    marker = clnk_marker[plt_format_indx]
    color = clnk_color[plt_format_indx]
    ax.scatter(
        k_6_clnks_lmbda, k_6_clnks_W_clnks_free_rot_uniaxl[0]/6,
        s=markersize, marker=marker, linewidth=markerlinewidth,
        edgecolors=color, facecolors="None", clip_on=False)
    ax.scatter(
        k_6_clnks_lmbda, k_6_clnks_W_clnks_free_rot_uniaxl[0]/6,
        s=dotsize, marker="o", linewidth=dotlinewidth,
        edgecolors=color, facecolors=color, clip_on=False)
    handles.append((marker, color))
    ax.legend(
        handles=handles, labels=half_n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=16,
        labelspacing=0, markerfirst=False, frameon=False, loc="upper right",
        bbox_to_anchor=(0.75, 1.025))
    ax.set_xlim([0.4, 2.0])
    ax.set_ylim([1.4, 2.6])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6])
    ax.set_xticklabels(["$~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$", "$1.6$", "$1.8$", "$2$"])
    ax.set_yticklabels(["$1.4~$", "$1.6~$", "$1.8~$", "$2~$", "$2.2~$", "$2.4~$", "$2.6~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel(
        "$\\leftparen W_{c, \\leftparen \\cdot \\rightparen }^{FR}/k_BT\\rightparen /k$",
        fontsize=16)
    fig.tight_layout()
    fig.savefig(W_clnks_free_rot_uniaxl_half_n_clnks_plot_fig_filename)
    plt.close()

    W_clnks_free_rot_uniaxl_one_unique_n_clnks_plot_fig_filename = (
        filepath
        + "JMPS_2026_fig_7b_W_clnks_free_rot_uniaxl_one_unique_n_clnks_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    handles = []
    plt_format_indx = 0
    marker = clnk_marker[plt_format_indx]
    color = clnk_color[plt_format_indx]
    ax.scatter(
        k_4_clnks_lmbda, k_4_clnks_W_clnks_free_rot_uniaxl[1]/4,
        s=markersize, marker=marker, linewidth=markerlinewidth,
        edgecolors=color, facecolors="None", clip_on=False)
    ax.scatter(
        k_4_clnks_lmbda, k_4_clnks_W_clnks_free_rot_uniaxl[1]/4,
        s=dotsize, marker="o", linewidth=dotlinewidth,
        edgecolors=color, facecolors=color, clip_on=False)
    handles.append((marker, color))
    plt_format_indx += 1
    marker = clnk_marker[plt_format_indx]
    color = clnk_color[plt_format_indx]
    ax.scatter(
        k_6_clnks_lmbda, k_6_clnks_W_clnks_free_rot_uniaxl[1]/6,
        s=markersize, marker=marker, linewidth=markerlinewidth,
        edgecolors=color, facecolors="None", clip_on=False)
    ax.scatter(
        k_6_clnks_lmbda, k_6_clnks_W_clnks_free_rot_uniaxl[1]/6,
        s=dotsize, marker="o", linewidth=dotlinewidth,
        edgecolors=color, facecolors=color, clip_on=False)
    handles.append((marker, color))
    ax.legend(
        handles=handles, labels=one_unique_n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=16,
        labelspacing=0, markerfirst=False, frameon=False, loc="upper right",
        bbox_to_anchor=(0.8, 1.025))
    ax.set_xlim([0.4, 2.0])
    ax.set_ylim([1.4, 2.6])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6])
    ax.set_xticklabels(["$~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$", "$1.6$", "$1.8$", "$2$"])
    ax.set_yticklabels(["$1.4~$", "$1.6~$", "$1.8~$", "$2~$", "$2.2~$", "$2.4~$", "$2.6~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel(
        "$\\leftparen W_{c, \\leftparen \\cdot \\rightparen }^{FR}/k_BT\\rightparen /k$",
        fontsize=16)
    fig.tight_layout()
    fig.savefig(W_clnks_free_rot_uniaxl_one_unique_n_clnks_plot_fig_filename)
    plt.close()

    W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks_plot_fig_filename = (
        filepath
        + "JMPS_2026_fig_10a_W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    handles = []
    plt_format_indx = 0
    marker = clnk_marker[plt_format_indx]
    color = clnk_color[plt_format_indx]
    ax.scatter(
        k_4_clnks_lmbda, k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl[0]/4,
        s=markersize, marker=marker, linewidth=markerlinewidth,
        edgecolors=color, facecolors="None", clip_on=False)
    ax.scatter(
        k_4_clnks_lmbda, k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl[0]/4,
        s=dotsize, marker="o", linewidth=dotlinewidth,
        edgecolors=color, facecolors=color, clip_on=False)
    handles.append((marker, color))
    plt_format_indx += 1
    marker = clnk_marker[plt_format_indx]
    color = clnk_color[plt_format_indx]
    ax.scatter(
        k_6_clnks_lmbda, k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl[0]/6,
        s=markersize, marker=marker, linewidth=markerlinewidth,
        edgecolors=color, facecolors="None", clip_on=False)
    ax.scatter(
        k_6_clnks_lmbda, k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl[0]/6,
        s=dotsize, marker="o", linewidth=dotlinewidth,
        edgecolors=color, facecolors=color, clip_on=False)
    handles.append((marker, color))
    ax.legend(
        handles=handles, labels=half_n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=16,
        labelspacing=0, markerfirst=False, frameon=False, loc="upper right",
        bbox_to_anchor=(0.75, 1.025))
    ax.set_xlim([0.4, 2.0])
    ax.set_ylim([1.4, 2.6])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6])
    ax.set_xticklabels(["$~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$", "$1.6$", "$1.8$", "$2$"])
    ax.set_yticklabels(["$1.4~$", "$1.6~$", "$1.8~$", "$2~$", "$2.2~$", "$2.4~$", "$2.6~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel(
        "$\\leftparen W_{c, \\leftparen \\cdot \\rightparen }^{FA}/k_BT\\rightparen /k$",
        fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks_plot_fig_filename)
    plt.close()

    W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks_plot_fig_filename = (
        filepath
        + "JMPS_2026_fig_10b_W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    handles = []
    plt_format_indx = 0
    marker = clnk_marker[plt_format_indx]
    color = clnk_color[plt_format_indx]
    ax.scatter(
        k_4_clnks_lmbda, k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl[1]/4,
        s=markersize, marker=marker, linewidth=markerlinewidth,
        edgecolors=color, facecolors="None", clip_on=False)
    ax.scatter(
        k_4_clnks_lmbda, k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl[1]/4,
        s=dotsize, marker="o", linewidth=dotlinewidth,
        edgecolors=color, facecolors=color, clip_on=False)
    handles.append((marker, color))
    plt_format_indx += 1
    marker = clnk_marker[plt_format_indx]
    color = clnk_color[plt_format_indx]
    ax.scatter(
        k_6_clnks_lmbda, k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl[1]/6,
        s=markersize, marker=marker, linewidth=markerlinewidth,
        edgecolors=color, facecolors="None", clip_on=False)
    ax.scatter(
        k_6_clnks_lmbda, k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl[1]/6,
        s=dotsize, marker="o", linewidth=dotlinewidth,
        edgecolors=color, facecolors=color, clip_on=False)
    handles.append((marker, color))
    ax.legend(
        handles=handles, labels=one_unique_n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=16,
        labelspacing=0, markerfirst=False, frameon=False, loc="upper right",
        bbox_to_anchor=(0.8, 1.025))
    ax.set_xlim([0.4, 2.0])
    ax.set_ylim([1.4, 2.6])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([1.4, 1.6, 1.8, 2.0, 2.2, 2.4, 2.6])
    ax.set_xticklabels(["$~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$", "$1.6$", "$1.8$", "$2$"])
    ax.set_yticklabels(["$1.4~$", "$1.6~$", "$1.8~$", "$2~$", "$2.2~$", "$2.4~$", "$2.6~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel(
        "$\\leftparen W_{c, \\leftparen \\cdot \\rightparen }^{FA}/k_BT\\rightparen /k$",
        fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks_plot_fig_filename)
    plt.close()

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse Gaussian end-linked polymer network elastically-effective cross-link RVE deformation analysis plotting took {execution_time} seconds to run")