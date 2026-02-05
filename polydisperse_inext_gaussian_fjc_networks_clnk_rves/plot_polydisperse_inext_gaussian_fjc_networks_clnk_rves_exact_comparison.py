# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import modules
import hydra
from omegaconf import DictConfig
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.markers import MarkerStyle
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
    topology = cfg.topology
    label = cfg.label

    clnk_color = ["tab:purple", "tab:green"]
    clnk_marker = ["^", "s"]
    markersize = 50
    dotsize = 0.5
    markerlinewidth = 0.5
    dotlinewidth = 0.125

    class HandlerCompositeMarker(HandlerBase):
        def create_artists(self, legend, orig_handle, x_descent, y_descent, width, height, fontsize, trans):
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

    sample = 0
    k_4_clnks_filename_prefix = filename_str(label.workdir, "20251204", "F", sample)
    k_6_clnks_filename_prefix = filename_str(label.workdir, "20251204", "G", sample)

    k_4_clnks_n_clnks_filename = k_4_clnks_filename_prefix + "-n_clnks" + ".dat"
    k_6_clnks_n_clnks_filename = k_6_clnks_filename_prefix + "-n_clnks" + ".dat"
    k_4_clnks_n_clnks = np.loadtxt(k_4_clnks_n_clnks_filename, dtype=int)
    k_6_clnks_n_clnks = np.loadtxt(k_6_clnks_n_clnks_filename, dtype=int)
    # print(k_4_clnks_n_clnks)
    # print(k_6_clnks_n_clnks)

    k_4_clnks_num, k_num = np.shape(k_4_clnks_n_clnks)
    assert k_4_clnks_num == 2
    k_4_n_clnks_legend = []
    for clnk_indx in range(k_4_clnks_num):
        clnk_str = "$\\leftparen"
        for chn_indx in range(k_num):
            clnk_str += f"{k_4_clnks_n_clnks[clnk_indx, chn_indx]:d}"
            if chn_indx < k_num-1: clnk_str += ","
        clnk_str += "\\rightparen$"
        k_4_n_clnks_legend.append(clnk_str)
    k_6_clnks_num, k_num = np.shape(k_6_clnks_n_clnks)
    assert k_6_clnks_num == 2
    k_6_n_clnks_legend = []
    for clnk_indx in range(k_6_clnks_num):
        clnk_str = "$\\leftparen"
        for chn_indx in range(k_num):
            clnk_str += f"{k_6_clnks_n_clnks[clnk_indx, chn_indx]:d}"
            if chn_indx < k_num-1: clnk_str += ","
        clnk_str += "\\rightparen$"
        k_6_n_clnks_legend.append(clnk_str)
    # print(k_4_n_clnks_legend)
    # print(k_6_n_clnks_legend)

    half_n_clnks_legend = []
    half_n_clnks_legend.append(k_4_n_clnks_legend[0])
    half_n_clnks_legend.append(k_6_n_clnks_legend[0])
    one_unique_n_clnks_legend = []
    one_unique_n_clnks_legend.append(k_4_n_clnks_legend[1])
    one_unique_n_clnks_legend.append(k_6_n_clnks_legend[1])
    # print(half_n_clnks_legend)
    # print(one_unique_n_clnks_legend)

    k_4_clnks_uniaxl_tens_dfrmtn_filename = (
        k_4_clnks_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    k_4_clnks_uniaxl_comp_dfrmtn_filename = (
        k_4_clnks_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    k_6_clnks_uniaxl_tens_dfrmtn_filename = (
        k_6_clnks_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    k_6_clnks_uniaxl_comp_dfrmtn_filename = (
        k_6_clnks_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )

    k_4_clnks_uniaxl_tens_dfrmtn = np.load(k_4_clnks_uniaxl_tens_dfrmtn_filename)
    k_4_clnks_uniaxl_comp_dfrmtn = np.load(k_4_clnks_uniaxl_comp_dfrmtn_filename)
    k_4_clnks_lmbda = np.hstack((np.flip(k_4_clnks_uniaxl_comp_dfrmtn)[:-1], k_4_clnks_uniaxl_tens_dfrmtn))

    k_6_clnks_uniaxl_tens_dfrmtn = np.load(k_6_clnks_uniaxl_tens_dfrmtn_filename)
    k_6_clnks_uniaxl_comp_dfrmtn = np.load(k_6_clnks_uniaxl_comp_dfrmtn_filename)
    k_6_clnks_lmbda = np.hstack((np.flip(k_6_clnks_uniaxl_comp_dfrmtn)[:-1], k_6_clnks_uniaxl_tens_dfrmtn))
    
    # print(k_4_clnks_lmbda)
    # print(k_6_clnks_lmbda)
    # print(np.shape(k_4_clnks_lmbda))
    # print(np.shape(k_6_clnks_lmbda))

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

    k_4_clnks_W_clnks_free_rot_uniaxl_tens = np.transpose(
        np.load(k_4_clnks_W_clnks_free_rot_uniaxl_tens_filename))
    k_4_clnks_W_clnks_free_rot_uniaxl_comp = np.transpose(
        np.load(k_4_clnks_W_clnks_free_rot_uniaxl_comp_filename))
    k_4_clnks_W_clnks_free_rot_uniaxl = np.hstack((np.flip(k_4_clnks_W_clnks_free_rot_uniaxl_comp, axis=1)[:, :-1], k_4_clnks_W_clnks_free_rot_uniaxl_tens))
    k_6_clnks_W_clnks_free_rot_uniaxl_tens = np.transpose(
        np.load(k_6_clnks_W_clnks_free_rot_uniaxl_tens_filename))
    k_6_clnks_W_clnks_free_rot_uniaxl_comp = np.transpose(
        np.load(k_6_clnks_W_clnks_free_rot_uniaxl_comp_filename))
    k_6_clnks_W_clnks_free_rot_uniaxl = np.hstack((np.flip(k_6_clnks_W_clnks_free_rot_uniaxl_comp, axis=1)[:, :-1], k_6_clnks_W_clnks_free_rot_uniaxl_tens))

    k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens = np.transpose(
        np.load(k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename))
    k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp = np.transpose(
        np.load(k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename))
    k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl = np.hstack((np.flip(k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp, axis=1)[:, :-1], k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens))
    k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens = np.transpose(
        np.load(k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename))
    k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp = np.transpose(
        np.load(k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename))
    k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl = np.hstack((np.flip(k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_comp, axis=1)[:, :-1], k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl_tens))

    # print(np.shape(k_4_clnks_W_clnks_free_rot_uniaxl))
    # print(np.shape(k_6_clnks_W_clnks_free_rot_uniaxl))
    # print(np.shape(k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl))
    # print(np.shape(k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl))

    W_clnks_free_rot_uniaxl_half_n_clnks_plot_fig_filename = (
        filepath + "W_clnks_free_rot_uniaxl_half_n_clnks_plot" + ".pdf"
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
    ax.set_ylabel("$\\leftparen W_{c, \\leftparen \\cdot \\rightparen }^{FR}/k_BT\\rightparen /k$", fontsize=16)
    fig.tight_layout()
    fig.savefig(W_clnks_free_rot_uniaxl_half_n_clnks_plot_fig_filename)
    plt.close()

    W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks_plot_fig_filename = (
        filepath + "W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks_plot"
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
    ax.set_ylabel("$\\leftparen W_{c, \\leftparen \\cdot \\rightparen }^{FA}/k_BT\\rightparen /k$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks_plot_fig_filename)
    plt.close()

    W_clnks_free_rot_uniaxl_one_unique_n_clnks_plot_fig_filename = (
        filepath + "W_clnks_free_rot_uniaxl_one_unique_n_clnks_plot" + ".pdf"
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
    ax.set_ylabel("$\\leftparen W_{c, \\leftparen \\cdot \\rightparen }^{FR}/k_BT\\rightparen /k$", fontsize=16)
    fig.tight_layout()
    fig.savefig(W_clnks_free_rot_uniaxl_one_unique_n_clnks_plot_fig_filename)
    plt.close()

    W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks_plot_fig_filename = (
        filepath + "W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks_plot"
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
    ax.set_ylabel("$\\leftparen W_{c, \\leftparen \\cdot \\rightparen }^{FA}/k_BT\\rightparen /k$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks_plot_fig_filename)
    plt.close()

    # # Save data in .csv files for plotting reproduction
    # assert np.array_equal(k_4_clnks_lmbda, k_6_clnks_lmbda)
    # lmbda = k_4_clnks_lmbda.copy()

    # lmbda_and_W_clnks_free_rot_uniaxl_half_n_clnks_filename = (
    #     filepath + "lmbda_and_W_clnks_free_rot_uniaxl_half_n_clnks"
    #     + ".csv"
    # )
    # lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks_filename = (
    #     filepath
    #     + "lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks"
    #     + ".csv"
    # )
    # lmbda_and_W_clnks_free_rot_uniaxl_one_unique_n_clnks_filename = (
    #     filepath + "lmbda_and_W_clnks_free_rot_uniaxl_one_unique_n_clnks"
    #     + ".csv"
    # )
    # lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks_filename = (
    #     filepath
    #     + "lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks"
    #     + ".csv"
    # )
    
    # lmbda_and_W_clnks_free_rot_uniaxl_half_n_clnks = lmbda.copy()
    # lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks = lmbda.copy()
    # lmbda_and_W_clnks_free_rot_uniaxl_one_unique_n_clnks = lmbda.copy()
    # lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks = lmbda.copy()
    
    # lmbda_and_W_clnks_free_rot_uniaxl_half_n_clnks = np.column_stack(
    #     (lmbda_and_W_clnks_free_rot_uniaxl_half_n_clnks, k_4_clnks_W_clnks_free_rot_uniaxl[0]/4))
    # lmbda_and_W_clnks_free_rot_uniaxl_half_n_clnks = np.column_stack(
    #     (lmbda_and_W_clnks_free_rot_uniaxl_half_n_clnks, k_6_clnks_W_clnks_free_rot_uniaxl[0]/6))
    
    # lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks = np.column_stack(
    #     (lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks, k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl[0]/4))
    # lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks = np.column_stack(
    #     (lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks, k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl[0]/6))
    
    # lmbda_and_W_clnks_free_rot_uniaxl_one_unique_n_clnks = np.column_stack(
    #     (lmbda_and_W_clnks_free_rot_uniaxl_one_unique_n_clnks, k_4_clnks_W_clnks_free_rot_uniaxl[1]/4))
    # lmbda_and_W_clnks_free_rot_uniaxl_one_unique_n_clnks = np.column_stack(
    #     (lmbda_and_W_clnks_free_rot_uniaxl_one_unique_n_clnks, k_6_clnks_W_clnks_free_rot_uniaxl[1]/6))
    
    # lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks = np.column_stack(
    #     (lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks, k_4_clnks_W_clnks_frame_avrg_so3_quad_uniaxl[1]/4))
    # lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks = np.column_stack(
    #     (lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks, k_6_clnks_W_clnks_frame_avrg_so3_quad_uniaxl[1]/6))
    
    # np.savetxt(
    #     lmbda_and_W_clnks_free_rot_uniaxl_half_n_clnks_filename,
    #     lmbda_and_W_clnks_free_rot_uniaxl_half_n_clnks, delimiter=",")
    # np.savetxt(
    #     lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks_filename,
    #     lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_half_n_clnks,
    #     delimiter=",")
    # np.savetxt(
    #     lmbda_and_W_clnks_free_rot_uniaxl_one_unique_n_clnks_filename,
    #     lmbda_and_W_clnks_free_rot_uniaxl_one_unique_n_clnks, delimiter=",")
    # np.savetxt(
    #     lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks_filename,
    #     lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_one_unique_n_clnks,
    #     delimiter=",")

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse Gaussian end-linked polymer network elastically-effective cross-link RVE deformation analysis plotting took {execution_time} seconds to run")