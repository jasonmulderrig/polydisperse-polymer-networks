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

    b = topology.b
    assert b == 1

    clnk_color = ["tab:green", "tab:purple", "tab:orange", "tab:red"]
    clnk_marker = ["s", "^", "p", "o"]
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
    mono_filename_prefix = filename_str(label.workdir, "20250724", "C", sample)
    poly_filename_prefix = filename_str(label.workdir, "20250724", "D", sample)

    mono_n_clnks_filename = mono_filename_prefix + "-n_clnks" + ".dat"
    poly_n_clnks_filename = poly_filename_prefix + "-n_clnks" + ".dat"
    mono_n_clnks = np.loadtxt(mono_n_clnks_filename, dtype=int)
    poly_n_clnks = np.loadtxt(poly_n_clnks_filename, dtype=int)
    # print(mono_n_clnks)
    # print(poly_n_clnks)

    clnks_num, k_num = np.shape(poly_n_clnks)
    poly_n_clnks_legend = []
    for clnk_indx in range(clnks_num):
        clnk_str = "$\\{"
        for chn_indx in range(k_num):
            clnk_str += str(poly_n_clnks[clnk_indx, chn_indx])
            if chn_indx < k_num-1: clnk_str += ","
        clnk_str += "\\}$"
        poly_n_clnks_legend.append(clnk_str)
    # print(poly_n_clnks_legend)

    mono_uniaxl_tens_dfrmtn_filename = (
        mono_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    mono_uniaxl_comp_dfrmtn_filename = (
        mono_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    mono_simple_shear_dfrmtn_filename = (
        mono_filename_prefix + "-dfrmtn_protocol_indx_2" + ".npy"
    )
    poly_uniaxl_tens_dfrmtn_filename = (
        poly_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    poly_uniaxl_comp_dfrmtn_filename = (
        poly_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    poly_simple_shear_dfrmtn_filename = (
        poly_filename_prefix + "-dfrmtn_protocol_indx_2" + ".npy"
    )

    mono_uniaxl_tens_dfrmtn = np.load(mono_uniaxl_tens_dfrmtn_filename)
    mono_uniaxl_comp_dfrmtn = np.load(mono_uniaxl_comp_dfrmtn_filename)
    mono_simple_shear_dfrmtn = np.load(mono_simple_shear_dfrmtn_filename)
    poly_uniaxl_tens_dfrmtn = np.load(poly_uniaxl_tens_dfrmtn_filename)
    poly_uniaxl_comp_dfrmtn = np.load(poly_uniaxl_comp_dfrmtn_filename)
    poly_simple_shear_dfrmtn = np.load(poly_simple_shear_dfrmtn_filename)

    mono_lmbda = np.hstack((np.flip(mono_uniaxl_comp_dfrmtn)[:-1], mono_uniaxl_tens_dfrmtn))
    poly_lmbda = np.hstack((np.flip(poly_uniaxl_comp_dfrmtn)[:-1], poly_uniaxl_tens_dfrmtn))
    mono_s = mono_simple_shear_dfrmtn
    poly_s = poly_simple_shear_dfrmtn
    # print(mono_lmbda)
    # print(poly_lmbda)
    # print(mono_s)
    # print(poly_s)
    # print(np.shape(mono_lmbda))
    # print(np.shape(poly_lmbda))
    # print(np.shape(mono_s))
    # print(np.shape(poly_s))

    mono_W_clnks_free_rot_uniaxl_tens_filename = (
        mono_filename_prefix + "-W_clnks_free_rot_protocol_indx_0" + ".npy"
    )
    mono_W_clnks_free_rot_uniaxl_comp_filename = (
        mono_filename_prefix + "-W_clnks_free_rot_protocol_indx_1" + ".npy"
    )
    mono_W_clnks_free_rot_simple_shear_filename = (
        mono_filename_prefix + "-W_clnks_free_rot_protocol_indx_2" + ".npy"
    )
    mono_W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename = (
        mono_filename_prefix + "-W_clnks_frame_avrg_so3_quad_protocol_indx_0"
        + ".npy"
    )
    mono_W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename = (
        mono_filename_prefix + "-W_clnks_frame_avrg_so3_quad_protocol_indx_1"
        + ".npy"
    )
    mono_W_clnks_frame_avrg_so3_quad_simple_shear_filename = (
        mono_filename_prefix + "-W_clnks_frame_avrg_so3_quad_protocol_indx_2"
        + ".npy"
    )
    
    poly_W_clnks_free_rot_uniaxl_tens_filename = (
        poly_filename_prefix + "-W_clnks_free_rot_protocol_indx_0" + ".npy"
    )
    poly_W_clnks_free_rot_uniaxl_comp_filename = (
        poly_filename_prefix + "-W_clnks_free_rot_protocol_indx_1" + ".npy"
    )
    poly_W_clnks_free_rot_simple_shear_filename = (
        poly_filename_prefix + "-W_clnks_free_rot_protocol_indx_2" + ".npy"
    )
    poly_W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename = (
        poly_filename_prefix + "-W_clnks_frame_avrg_so3_quad_protocol_indx_0"
        + ".npy"
    )
    poly_W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename = (
        poly_filename_prefix + "-W_clnks_frame_avrg_so3_quad_protocol_indx_1"
        + ".npy"
    )
    poly_W_clnks_frame_avrg_so3_quad_simple_shear_filename = (
        poly_filename_prefix + "-W_clnks_frame_avrg_so3_quad_protocol_indx_2"
        + ".npy"
    )
    poly_y_clnks_norm_free_rot_uniaxl_tens_filename = (
        poly_filename_prefix + "-y_clnks_norm_free_rot_protocol_indx_0" + ".npy"
    )
    poly_y_clnks_norm_free_rot_uniaxl_comp_filename = (
        poly_filename_prefix + "-y_clnks_norm_free_rot_protocol_indx_1" + ".npy"
    )
    poly_y_clnks_norm_free_rot_simple_shear_filename = (
        poly_filename_prefix + "-y_clnks_norm_free_rot_protocol_indx_2" + ".npy"
    )
    poly_y_clnks_norm_frame_avrg_so3_quad_uniaxl_tens_filename = (
        poly_filename_prefix
        + "-y_clnks_norm_frame_avrg_so3_quad_protocol_indx_0" + ".npy"
    )
    poly_y_clnks_norm_frame_avrg_so3_quad_uniaxl_comp_filename = (
        poly_filename_prefix
        + "-y_clnks_norm_frame_avrg_so3_quad_protocol_indx_1" + ".npy"
    )
    poly_y_clnks_norm_frame_avrg_so3_quad_simple_shear_filename = (
        poly_filename_prefix
        + "-y_clnks_norm_frame_avrg_so3_quad_protocol_indx_2" + ".npy"
    )

    mono_W_clnks_free_rot_uniaxl_tens = np.transpose(np.load(mono_W_clnks_free_rot_uniaxl_tens_filename))
    mono_W_clnks_free_rot_uniaxl_comp = np.transpose(np.load(mono_W_clnks_free_rot_uniaxl_comp_filename))
    mono_W_clnks_free_rot_uniaxl = np.hstack((np.flip(mono_W_clnks_free_rot_uniaxl_comp, axis=1)[:, :-1], mono_W_clnks_free_rot_uniaxl_tens))
    mono_W_clnks_free_rot_uniaxl = mono_W_clnks_free_rot_uniaxl[0]
    mono_W_clnks_free_rot_simple_shear = np.transpose(np.load(mono_W_clnks_free_rot_simple_shear_filename))
    mono_W_clnks_free_rot_simple_shear = mono_W_clnks_free_rot_simple_shear[0]
    mono_W_clnks_frame_avrg_so3_quad_uniaxl_tens = np.transpose(np.load(mono_W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename))
    mono_W_clnks_frame_avrg_so3_quad_uniaxl_comp = np.transpose(np.load(mono_W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename))
    mono_W_clnks_frame_avrg_so3_quad_uniaxl = np.hstack((np.flip(mono_W_clnks_frame_avrg_so3_quad_uniaxl_comp, axis=1)[:, :-1], mono_W_clnks_frame_avrg_so3_quad_uniaxl_tens))
    mono_W_clnks_frame_avrg_so3_quad_uniaxl = mono_W_clnks_frame_avrg_so3_quad_uniaxl[0]
    mono_W_clnks_frame_avrg_so3_quad_simple_shear = np.transpose(np.load(mono_W_clnks_frame_avrg_so3_quad_simple_shear_filename))
    mono_W_clnks_frame_avrg_so3_quad_simple_shear = mono_W_clnks_frame_avrg_so3_quad_simple_shear[0]

    poly_W_clnks_free_rot_uniaxl_tens = np.transpose(np.load(poly_W_clnks_free_rot_uniaxl_tens_filename))
    poly_W_clnks_free_rot_uniaxl_comp = np.transpose(np.load(poly_W_clnks_free_rot_uniaxl_comp_filename))
    poly_W_clnks_free_rot_uniaxl = np.hstack((np.flip(poly_W_clnks_free_rot_uniaxl_comp, axis=1)[:, :-1], poly_W_clnks_free_rot_uniaxl_tens))
    poly_W_clnks_free_rot_simple_shear = np.transpose(np.load(poly_W_clnks_free_rot_simple_shear_filename))
    poly_W_clnks_frame_avrg_so3_quad_uniaxl_tens = np.transpose(np.load(poly_W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename))
    poly_W_clnks_frame_avrg_so3_quad_uniaxl_comp = np.transpose(np.load(poly_W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename))
    poly_W_clnks_frame_avrg_so3_quad_uniaxl = np.hstack((np.flip(poly_W_clnks_frame_avrg_so3_quad_uniaxl_comp, axis=1)[:, :-1], poly_W_clnks_frame_avrg_so3_quad_uniaxl_tens))
    poly_W_clnks_frame_avrg_so3_quad_simple_shear = np.transpose(np.load(poly_W_clnks_frame_avrg_so3_quad_simple_shear_filename))

    poly_y_clnks_norm_free_rot_uniaxl_tens = np.transpose(np.load(poly_y_clnks_norm_free_rot_uniaxl_tens_filename))
    poly_y_clnks_norm_free_rot_uniaxl_comp = np.transpose(np.load(poly_y_clnks_norm_free_rot_uniaxl_comp_filename))
    poly_y_clnks_norm_free_rot_uniaxl = np.hstack((np.flip(poly_y_clnks_norm_free_rot_uniaxl_comp, axis=1)[:, :-1], poly_y_clnks_norm_free_rot_uniaxl_tens))
    poly_y_clnks_norm_free_rot_simple_shear = np.transpose(np.load(poly_y_clnks_norm_free_rot_simple_shear_filename))
    poly_y_clnks_norm_frame_avrg_so3_quad_uniaxl_tens = np.transpose(np.load(poly_y_clnks_norm_frame_avrg_so3_quad_uniaxl_tens_filename))
    poly_y_clnks_norm_frame_avrg_so3_quad_uniaxl_comp = np.transpose(np.load(poly_y_clnks_norm_frame_avrg_so3_quad_uniaxl_comp_filename))
    poly_y_clnks_norm_frame_avrg_so3_quad_uniaxl = np.hstack((np.flip(poly_y_clnks_norm_frame_avrg_so3_quad_uniaxl_comp, axis=1)[:, :-1], poly_y_clnks_norm_frame_avrg_so3_quad_uniaxl_tens))
    poly_y_clnks_norm_frame_avrg_so3_quad_simple_shear = np.transpose(np.load(poly_y_clnks_norm_frame_avrg_so3_quad_simple_shear_filename))
    
    # print(np.shape(mono_W_clnks_free_rot_uniaxl))
    # print(np.shape(mono_W_clnks_free_rot_simple_shear))
    # print(np.shape(mono_W_clnks_frame_avrg_so3_quad_uniaxl))
    # print(np.shape(mono_W_clnks_frame_avrg_so3_quad_simple_shear))
    # print(np.shape(poly_W_clnks_free_rot_uniaxl))
    # print(np.shape(poly_W_clnks_free_rot_simple_shear))
    # print(np.shape(poly_W_clnks_frame_avrg_so3_quad_uniaxl))
    # print(np.shape(poly_W_clnks_frame_avrg_so3_quad_simple_shear))
    # print(np.shape(poly_y_clnks_norm_free_rot_uniaxl))
    # print(np.shape(poly_y_clnks_norm_free_rot_simple_shear))
    # print(np.shape(poly_y_clnks_norm_frame_avrg_so3_quad_uniaxl))
    # print(np.shape(poly_y_clnks_norm_frame_avrg_so3_quad_simple_shear))

    W_clnks_free_rot_uniaxl_general_deformation_plot_fig_filename = (
        filepath + "W_clnks_free_rot_uniaxl_general_deformation_plot" + ".png"
    )
    fig, ax = plt.subplots()
    handles = []
    ax.plot(
        mono_lmbda, mono_W_clnks_free_rot_uniaxl, linestyle="--",
        linewidth=markerlinewidth, c="black")
    for clnk_indx in range(clnks_num):
        marker = clnk_marker[clnk_indx]
        color = clnk_color[clnk_indx]
        poly_W_clnk_free_rot_uniaxl = poly_W_clnks_free_rot_uniaxl[clnk_indx]
        ax.scatter(
            poly_lmbda, poly_W_clnk_free_rot_uniaxl, s=markersize,
            marker=marker, linewidth=markerlinewidth, edgecolors=color,
            facecolors="None", clip_on=False)
        ax.scatter(
            poly_lmbda, poly_W_clnk_free_rot_uniaxl, s=dotsize, marker="o",
            linewidth=dotlinewidth, edgecolors=color, clip_on=False)
        handles.append((marker, color))
    ax.legend(
        handles=handles, labels=poly_n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=12,
        labelspacing=0, markerfirst=False, loc="best")
    ax.set_xlim([0.4, 2.0])
    ax.set_ylim([5.0, 10.5])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5])
    ax.set_xticklabels(["$~~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$", "$1.6$", "$1.8$", "$2$"])
    ax.set_yticklabels(["$5~~$", "$5.5$", "$6$", "$6.5$", "$7$", "$7.5$", "$8$", "$8.5$", "$9$", "$9.5$", "$10$", "$10.5$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{c, \\{\\cdot\\}}^{FR}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(W_clnks_free_rot_uniaxl_general_deformation_plot_fig_filename)
    plt.close()

    W_clnks_frame_avrg_so3_quad_uniaxl_general_deformation_plot_fig_filename = (
        filepath + "W_clnks_frame_avrg_so3_quad_uniaxl_general_deformation_plot" + ".png"
    )
    fig, ax = plt.subplots()
    handles = []
    ax.plot(
        mono_lmbda, mono_W_clnks_frame_avrg_so3_quad_uniaxl, linestyle="--",
        linewidth=markerlinewidth, c="black")
    for clnk_indx in range(clnks_num):
        marker = clnk_marker[clnk_indx]
        color = clnk_color[clnk_indx]
        poly_W_clnk_frame_avrg_so3_quad_uniaxl = poly_W_clnks_frame_avrg_so3_quad_uniaxl[clnk_indx]
        ax.scatter(
            poly_lmbda, poly_W_clnk_frame_avrg_so3_quad_uniaxl, s=markersize,
            marker=marker, linewidth=markerlinewidth, edgecolors=color,
            facecolors="None", clip_on=False)
        ax.scatter(
            poly_lmbda, poly_W_clnk_frame_avrg_so3_quad_uniaxl, s=dotsize, marker="o",
            linewidth=dotlinewidth, edgecolors=color, clip_on=False)
        handles.append((marker, color))
    ax.legend(
        handles=handles, labels=poly_n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=12,
        labelspacing=0, markerfirst=False, loc="best")
    ax.set_xlim([0.4, 2.0])
    ax.set_ylim([5.0, 10.5])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5])
    ax.set_xticklabels(["$~~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$", "$1.6$", "$1.8$", "$2$"])
    ax.set_yticklabels(["$5~~$", "$5.5$", "$6$", "$6.5$", "$7$", "$7.5$", "$8$", "$8.5$", "$9$", "$9.5$", "$10$", "$10.5$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{c, \\{\\cdot\\}}^{FA}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(W_clnks_frame_avrg_so3_quad_uniaxl_general_deformation_plot_fig_filename)
    plt.close()

    W_clnks_free_rot_simple_shear_general_deformation_plot_fig_filename = (
        filepath + "W_clnks_free_rot_simple_shear_general_deformation_plot" + ".png"
    )
    fig, ax = plt.subplots()
    handles = []
    ax.plot(
        mono_s, mono_W_clnks_free_rot_simple_shear, linestyle="--",
        linewidth=markerlinewidth, c="black")
    for clnk_indx in range(clnks_num):
        marker = clnk_marker[clnk_indx]
        color = clnk_color[clnk_indx]
        poly_W_clnk_free_rot_simple_shear = poly_W_clnks_free_rot_simple_shear[clnk_indx]
        ax.scatter(
            poly_s, poly_W_clnk_free_rot_simple_shear, s=markersize,
            marker=marker, linewidth=markerlinewidth, edgecolors=color,
            facecolors="None", clip_on=False)
        ax.scatter(
            poly_s, poly_W_clnk_free_rot_simple_shear, s=dotsize, marker="o",
            linewidth=dotlinewidth, edgecolors=color, clip_on=False)
        handles.append((marker, color))
    ax.legend(
        handles=handles, labels=poly_n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=12,
        labelspacing=0, markerfirst=False, loc="best")
    ax.set_xlim([0.0, 2.0])
    ax.set_ylim([5.0, 14.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    ax.set_xticklabels(["$~~0$", "$0.5$", "$1$", "$1.5$", "$2$"])
    ax.set_yticklabels(["$5~~$", "$6$", "$7$", "$8$", "$9$", "$10$", "$11$", "$12$", "$13$", "$14$"])
    ax.set_xlabel("$s$", fontsize=16)
    ax.set_ylabel("$W_{c, \\{\\cdot\\}}^{FR}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(W_clnks_free_rot_simple_shear_general_deformation_plot_fig_filename)
    plt.close()

    W_clnks_frame_avrg_so3_quad_simple_shear_general_deformation_plot_fig_filename = (
        filepath + "W_clnks_frame_avrg_so3_quad_simple_shear_general_deformation_plot" + ".png"
    )
    fig, ax = plt.subplots()
    handles = []
    ax.plot(
        mono_s, mono_W_clnks_frame_avrg_so3_quad_simple_shear, linestyle="--",
        linewidth=markerlinewidth, c="black")
    for clnk_indx in range(clnks_num):
        marker = clnk_marker[clnk_indx]
        color = clnk_color[clnk_indx]
        poly_W_clnk_frame_avrg_so3_quad_simple_shear = poly_W_clnks_frame_avrg_so3_quad_simple_shear[clnk_indx]
        ax.scatter(
            poly_s, poly_W_clnk_frame_avrg_so3_quad_simple_shear, s=markersize,
            marker=marker, linewidth=markerlinewidth, edgecolors=color,
            facecolors="None", clip_on=False)
        ax.scatter(
            poly_s, poly_W_clnk_frame_avrg_so3_quad_simple_shear, s=dotsize, marker="o",
            linewidth=dotlinewidth, edgecolors=color, clip_on=False)
        handles.append((marker, color))
    ax.legend(
        handles=handles, labels=poly_n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=12,
        labelspacing=0, markerfirst=False, loc="best")
    ax.set_xlim([0.0, 2.0])
    ax.set_ylim([5.0, 14.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    ax.set_xticklabels(["$~~0$", "$0.5$", "$1$", "$1.5$", "$2$"])
    ax.set_yticklabels(["$5~~$", "$6$", "$7$", "$8$", "$9$", "$10$", "$11$", "$12$", "$13$", "$14$"])
    ax.set_xlabel("$s$", fontsize=16)
    ax.set_ylabel("$W_{c, \\{\\cdot\\}}^{FA}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(W_clnks_frame_avrg_so3_quad_simple_shear_general_deformation_plot_fig_filename)
    plt.close()

    y_clnks_norm_free_rot_uniaxl_general_deformation_plot_fig_filename = (
        filepath + "y_clnks_norm_free_rot_uniaxl_general_deformation_plot" + ".png"
    )
    fig, ax = plt.subplots()
    handles = []
    for clnk_indx in range(clnks_num):
        marker = clnk_marker[clnk_indx]
        color = clnk_color[clnk_indx]
        poly_y_clnk_norm_over_b_free_rot_uniaxl = (
            poly_y_clnks_norm_free_rot_uniaxl[clnk_indx] / b
        )
        ax.scatter(
            poly_lmbda, poly_y_clnk_norm_over_b_free_rot_uniaxl, s=markersize,
            marker=marker, linewidth=markerlinewidth, edgecolors=color,
            facecolors="None", clip_on=False)
        ax.scatter(
            poly_lmbda, poly_y_clnk_norm_over_b_free_rot_uniaxl, s=dotsize,
            marker="o", linewidth=dotlinewidth, edgecolors=color, clip_on=False)
        handles.append((marker, color))
    ax.legend(
        handles=handles, labels=poly_n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=12,
        labelspacing=0, markerfirst=False, loc="best")
    ax.set_xlim([0.4, 2.0])
    ax.set_ylim([1.0, 4.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ax.set_xticklabels(["$~~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$", "$1.6$", "$1.8$", "$2$"])
    ax.set_yticklabels(["$1~~$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$|\\mathbf{y}|^{FR}/b$", fontsize=16)
    fig.tight_layout()
    fig.savefig(y_clnks_norm_free_rot_uniaxl_general_deformation_plot_fig_filename)
    plt.close()

    y_clnks_norm_frame_avrg_so3_quad_uniaxl_general_deformation_plot_fig_filename = (
        filepath + "y_clnks_norm_frame_avrg_so3_quad_uniaxl_general_deformation_plot" + ".png"
    )
    fig, ax = plt.subplots()
    handles = []
    for clnk_indx in range(clnks_num):
        marker = clnk_marker[clnk_indx]
        color = clnk_color[clnk_indx]
        poly_y_clnk_norm_over_b_frame_avrg_so3_quad_uniaxl = (
            poly_y_clnks_norm_frame_avrg_so3_quad_uniaxl[clnk_indx] / b
        )
        ax.scatter(
            poly_lmbda, poly_y_clnk_norm_over_b_frame_avrg_so3_quad_uniaxl,
            s=markersize, marker=marker, linewidth=markerlinewidth,
            edgecolors=color, facecolors="None", clip_on=False)
        ax.scatter(
            poly_lmbda, poly_y_clnk_norm_over_b_frame_avrg_so3_quad_uniaxl,
            s=dotsize, marker="o", linewidth=dotlinewidth, edgecolors=color,
            clip_on=False)
        handles.append((marker, color))
    ax.legend(
        handles=handles, labels=poly_n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=12,
        labelspacing=0, markerfirst=False, loc="best")
    ax.set_xlim([0.4, 2.0])
    ax.set_ylim([1.0, 4.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ax.set_xticklabels(["$~~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$", "$1.6$", "$1.8$", "$2$"])
    ax.set_yticklabels(["$1~~$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$|\\mathbf{y}|^{FA}/b$", fontsize=16)
    fig.tight_layout()
    fig.savefig(y_clnks_norm_frame_avrg_so3_quad_uniaxl_general_deformation_plot_fig_filename)
    plt.close()

    y_clnks_norm_free_rot_simple_shear_general_deformation_plot_fig_filename = (
        filepath + "y_clnks_norm_free_rot_simple_shear_general_deformation_plot" + ".png"
    )
    fig, ax = plt.subplots()
    handles = []
    for clnk_indx in range(clnks_num):
        marker = clnk_marker[clnk_indx]
        color = clnk_color[clnk_indx]
        poly_y_clnk_norm_over_b_free_rot_simple_shear = (
            poly_y_clnks_norm_free_rot_simple_shear[clnk_indx] / b
        )
        ax.scatter(
            poly_s, poly_y_clnk_norm_over_b_free_rot_simple_shear, s=markersize,
            marker=marker, linewidth=markerlinewidth, edgecolors=color,
            facecolors="None", clip_on=False)
        ax.scatter(
            poly_s, poly_y_clnk_norm_over_b_free_rot_simple_shear, s=dotsize,
            marker="o", linewidth=dotlinewidth, edgecolors=color, clip_on=False)
        handles.append((marker, color))
    ax.legend(
        handles=handles, labels=poly_n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=12,
        labelspacing=0, markerfirst=False, loc="best")
    ax.set_xlim([0.0, 2.0])
    ax.set_ylim([1.0, 4.5])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    ax.set_xticklabels(["$~~0$", "$0.5$", "$1$", "$1.5$", "$2$"])
    ax.set_yticklabels(["$1~~$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$", "$4.5$"])
    ax.set_xlabel("$s$", fontsize=16)
    ax.set_ylabel("$|\\mathbf{y}|^{FR}/b$", fontsize=16)
    fig.tight_layout()
    fig.savefig(y_clnks_norm_free_rot_simple_shear_general_deformation_plot_fig_filename)
    plt.close()

    y_clnks_norm_frame_avrg_so3_quad_simple_shear_general_deformation_plot_fig_filename = (
        filepath + "y_clnks_norm_frame_avrg_so3_quad_simple_shear_general_deformation_plot" + ".png"
    )
    fig, ax = plt.subplots()
    handles = []
    for clnk_indx in range(clnks_num):
        marker = clnk_marker[clnk_indx]
        color = clnk_color[clnk_indx]
        poly_y_clnks_norm_over_b_frame_avrg_so3_quad_simple_shear = (
            poly_y_clnks_norm_frame_avrg_so3_quad_simple_shear[clnk_indx] / b
        )
        ax.scatter(
            poly_s, poly_y_clnks_norm_over_b_frame_avrg_so3_quad_simple_shear,
            s=markersize, marker=marker, linewidth=markerlinewidth,
            edgecolors=color, facecolors="None", clip_on=False)
        ax.scatter(
            poly_s, poly_y_clnks_norm_over_b_frame_avrg_so3_quad_simple_shear,
            s=dotsize, marker="o", linewidth=dotlinewidth, edgecolors=color,
            clip_on=False)
        handles.append((marker, color))
    ax.legend(
        handles=handles, labels=poly_n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=12,
        labelspacing=0, markerfirst=False, loc="best")
    ax.set_xlim([0.0, 2.0])
    ax.set_ylim([1.0, 4.5])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5])
    ax.set_xticklabels(["$~~0$", "$0.5$", "$1$", "$1.5$", "$2$"])
    ax.set_yticklabels(["$1~~$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$", "$4.5$"])
    ax.set_xlabel("$s$", fontsize=16)
    ax.set_ylabel("$|\\mathbf{y}|^{FA}/b$", fontsize=16)
    fig.tight_layout()
    fig.savefig(y_clnks_norm_frame_avrg_so3_quad_simple_shear_general_deformation_plot_fig_filename)
    plt.close()

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse Gaussian end-linked polymer network elastically-effective cross-link RVE deformation analysis plotting took {execution_time} seconds to run")