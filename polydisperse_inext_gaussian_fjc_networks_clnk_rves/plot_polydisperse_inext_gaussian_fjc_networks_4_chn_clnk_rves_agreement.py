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

    clnk_color = ["tab:purple", "tab:blue", "tab:green", "tab:orange", "tab:red"]
    clnk_marker = ["^", "p", "s", "+x", "o"]
    plusx_marker_list = ["+", "x"]
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

            if marker == "+x":
                for plusx_marker in plusx_marker_list:
                    artists.append(
                        plt.Line2D(
                            [center_x], [center_y], marker=plusx_marker,
                            color=color, markersize=markersize/7,
                            markeredgewidth=markerlinewidth, linestyle="None",
                            transform=trans))
            else:
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

    exact_filename_prefix = filename_str(label.workdir, "20251204", "A", 0)
    approx_filename_prefix = filename_str(label.workdir, "20251204", "B", 0)

    exact_n_clnks_filename = exact_filename_prefix + "-n_clnks" + ".dat"
    approx_n_clnks_filename = approx_filename_prefix + "-n_clnks" + ".dat"
    exact_n_clnks = np.loadtxt(exact_n_clnks_filename, dtype=int)
    approx_n_clnks = np.loadtxt(approx_n_clnks_filename, dtype=int)
    # print(exact_n_clnks)
    # print(approx_n_clnks)

    assert np.allclose(exact_n_clnks, approx_n_clnks)

    n_clnks = exact_n_clnks
    clnks_num, k_num = np.shape(n_clnks)
    n_clnks_legend = []
    for clnk_indx in range(clnks_num):
        clnk_str = "$\\leftparen"
        for chn_indx in range(k_num):
            clnk_str += f"{n_clnks[clnk_indx, chn_indx]:d}"
            if chn_indx < k_num-1: clnk_str += ","
        clnk_str += "\\rightparen$"
        n_clnks_legend.append(clnk_str)
    # print(n_clnks_legend)

    exact_uniaxl_tens_dfrmtn_filename = (
        exact_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    exact_uniaxl_comp_dfrmtn_filename = (
        exact_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    exact_simple_shear_dfrmtn_filename = (
        exact_filename_prefix + "-dfrmtn_protocol_indx_2" + ".npy"
    )
    approx_uniaxl_tens_dfrmtn_filename = (
        approx_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    approx_uniaxl_comp_dfrmtn_filename = (
        approx_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    approx_simple_shear_dfrmtn_filename = (
        approx_filename_prefix + "-dfrmtn_protocol_indx_2" + ".npy"
    )

    exact_uniaxl_tens_dfrmtn = np.load(exact_uniaxl_tens_dfrmtn_filename)
    exact_uniaxl_comp_dfrmtn = np.load(exact_uniaxl_comp_dfrmtn_filename)
    exact_simple_shear_dfrmtn = np.load(exact_simple_shear_dfrmtn_filename)
    approx_uniaxl_tens_dfrmtn = np.load(approx_uniaxl_tens_dfrmtn_filename)
    approx_uniaxl_comp_dfrmtn = np.load(approx_uniaxl_comp_dfrmtn_filename)
    approx_simple_shear_dfrmtn = np.load(approx_simple_shear_dfrmtn_filename)

    exact_lmbda = np.hstack((np.flip(exact_uniaxl_comp_dfrmtn)[:-1], exact_uniaxl_tens_dfrmtn))
    approx_lmbda = np.hstack((np.flip(approx_uniaxl_comp_dfrmtn)[:-1], approx_uniaxl_tens_dfrmtn))
    exact_s = exact_simple_shear_dfrmtn
    approx_s = approx_simple_shear_dfrmtn
    # print(exact_lmbda)
    # print(approx_lmbda)
    # print(exact_s)
    # print(approx_s)
    # print(np.shape(exact_lmbda))
    # print(np.shape(approx_lmbda))
    # print(np.shape(exact_s))
    # print(np.shape(approx_s))

    W_clnks_free_rot_uniaxl_tens_filename = (
        exact_filename_prefix + "-W_clnks_free_rot_protocol_indx_0" + ".npy"
    )
    W_clnks_free_rot_uniaxl_comp_filename = (
        exact_filename_prefix + "-W_clnks_free_rot_protocol_indx_1" + ".npy"
    )
    W_clnks_free_rot_simple_shear_filename = (
        exact_filename_prefix + "-W_clnks_free_rot_protocol_indx_2" + ".npy"
    )
    W_clnks_free_rot_approx_uniaxl_tens_filename = (
        approx_filename_prefix
        + "-W_clnks_free_rot_approx_protocol_indx_0" + ".npy"
    )
    W_clnks_free_rot_approx_uniaxl_comp_filename = (
        approx_filename_prefix
        + "-W_clnks_free_rot_approx_protocol_indx_1" + ".npy"
    )
    W_clnks_free_rot_approx_simple_shear_filename = (
        approx_filename_prefix
        + "-W_clnks_free_rot_approx_protocol_indx_2" + ".npy"
    )

    W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename = (
        exact_filename_prefix
        + "-W_clnks_frame_avrg_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename = (
        exact_filename_prefix
        + "-W_clnks_frame_avrg_so3_quad_protocol_indx_1" + ".npy"
    )
    W_clnks_frame_avrg_so3_quad_simple_shear_filename = (
        exact_filename_prefix
        + "-W_clnks_frame_avrg_so3_quad_protocol_indx_2" + ".npy"
    )
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        approx_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename = (
        approx_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_1" + ".npy"
    )
    W_clnks_frame_avrg_approx_so3_quad_simple_shear_filename = (
        approx_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_2" + ".npy"
    )

    W_clnks_free_rot_uniaxl_tens = np.transpose(
        np.load(W_clnks_free_rot_uniaxl_tens_filename))
    W_clnks_free_rot_uniaxl_comp = np.transpose(
        np.load(W_clnks_free_rot_uniaxl_comp_filename))
    W_clnks_free_rot_uniaxl = np.hstack((np.flip(W_clnks_free_rot_uniaxl_comp, axis=1)[:, :-1], W_clnks_free_rot_uniaxl_tens))
    W_clnks_free_rot_simple_shear = np.transpose(
        np.load(W_clnks_free_rot_simple_shear_filename))
    W_clnks_free_rot_approx_uniaxl_tens = np.transpose(
        np.load(W_clnks_free_rot_approx_uniaxl_tens_filename))
    W_clnks_free_rot_approx_uniaxl_comp = np.transpose(
        np.load(W_clnks_free_rot_approx_uniaxl_comp_filename))
    W_clnks_free_rot_approx_uniaxl = np.hstack((np.flip(W_clnks_free_rot_approx_uniaxl_comp, axis=1)[:, :-1], W_clnks_free_rot_approx_uniaxl_tens))
    W_clnks_free_rot_approx_simple_shear = np.transpose(
        np.load(W_clnks_free_rot_approx_simple_shear_filename))

    W_clnks_frame_avrg_so3_quad_uniaxl_tens = np.transpose(
        np.load(W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename))
    W_clnks_frame_avrg_so3_quad_uniaxl_comp = np.transpose(
        np.load(W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename))
    W_clnks_frame_avrg_so3_quad_uniaxl = np.hstack((np.flip(W_clnks_frame_avrg_so3_quad_uniaxl_comp, axis=1)[:, :-1], W_clnks_frame_avrg_so3_quad_uniaxl_tens))
    W_clnks_frame_avrg_so3_quad_simple_shear = np.transpose(
        np.load(W_clnks_frame_avrg_so3_quad_simple_shear_filename))
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens = np.transpose(
        np.load(W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename))
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp = np.transpose(
        np.load(W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename))
    W_clnks_frame_avrg_approx_so3_quad_uniaxl = np.hstack((np.flip(W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp, axis=1)[:, :-1], W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens))
    W_clnks_frame_avrg_approx_so3_quad_simple_shear = np.transpose(
        np.load(W_clnks_frame_avrg_approx_so3_quad_simple_shear_filename))
    
    # print(np.shape(W_clnks_free_rot_uniaxl))
    # print(np.shape(W_clnks_free_rot_simple_shear))
    # print(np.shape(W_clnks_free_rot_approx_uniaxl))
    # print(np.shape(W_clnks_free_rot_approx_simple_shear))
    # print(np.shape(W_clnks_frame_avrg_so3_quad_uniaxl))
    # print(np.shape(W_clnks_frame_avrg_so3_quad_simple_shear))
    # print(np.shape(W_clnks_frame_avrg_approx_so3_quad_uniaxl))
    # print(np.shape(W_clnks_frame_avrg_approx_so3_quad_simple_shear))

    W_clnks_free_rot_uniaxl_agreement_plot_fig_filename = (
        filepath + "W_clnks_free_rot_uniaxl_agreement_plot" + ".pdf"
    )
    fig, ax = plt.subplots()
    handles = []
    for clnk_indx in range(clnks_num):
        marker = clnk_marker[clnk_indx]
        color = clnk_color[clnk_indx]
        W_clnk_free_rot_uniaxl = W_clnks_free_rot_uniaxl[clnk_indx]
        W_clnk_free_rot_approx_uniaxl = W_clnks_free_rot_approx_uniaxl[clnk_indx]
        if marker == "+x":
            for plusx_marker in plusx_marker_list:
                ax.scatter(
                    exact_lmbda, W_clnk_free_rot_uniaxl, s=markersize,
                    marker=plusx_marker, linewidth=markerlinewidth,
                    facecolors=color, clip_on=False)
        else:
            ax.scatter(
                exact_lmbda, W_clnk_free_rot_uniaxl, s=markersize,
                marker=marker, linewidth=markerlinewidth, edgecolors=color,
                facecolors="None", clip_on=False)
        ax.scatter(
            exact_lmbda, W_clnk_free_rot_uniaxl, s=dotsize, marker="o",
            linewidth=dotlinewidth, edgecolors=color, facecolors=color,
            clip_on=False)
        ax.plot(
            approx_lmbda, W_clnk_free_rot_approx_uniaxl, linestyle="-",
            linewidth=markerlinewidth, c=color)
        handles.append((marker, color))
    ax.legend(
        handles=handles, labels=n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=16,
        labelspacing=0, markerfirst=False, frameon=False, loc="upper right",
        bbox_to_anchor=(0.575, 1.025))
    ax.set_xlim([0.4, 2.0])
    ax.set_ylim([5.5, 10.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
    ax.set_xticklabels(["$~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$", "$1.6$", "$1.8$", "$2$"])
    ax.set_yticklabels(["$5.5~$", "$6~$", "$6.5~$", "$7~$", "$7.5~$", "$8~$", "$8.5~$", "$9~$", "$9.5~$", "$10~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{c, \\leftparen \\cdot \\rightparen }^{FR}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(W_clnks_free_rot_uniaxl_agreement_plot_fig_filename)
    plt.close()

    W_clnks_frame_avrg_so3_quad_uniaxl_agreement_plot_fig_filename = (
        filepath + "W_clnks_frame_avrg_so3_quad_uniaxl_agreement_plot" + ".pdf"
    )
    fig, ax = plt.subplots()
    handles = []
    for clnk_indx in range(clnks_num):
        marker = clnk_marker[clnk_indx]
        color = clnk_color[clnk_indx]
        W_clnk_frame_avrg_so3_quad_uniaxl = W_clnks_frame_avrg_so3_quad_uniaxl[clnk_indx]
        W_clnk_frame_avrg_approx_so3_quad_uniaxl = W_clnks_frame_avrg_approx_so3_quad_uniaxl[clnk_indx]
        if marker == "+x":
            for plusx_marker in plusx_marker_list:
                ax.scatter(
                    exact_lmbda, W_clnk_frame_avrg_so3_quad_uniaxl,
                    s=markersize, marker=plusx_marker,
                    linewidth=markerlinewidth, facecolors=color, clip_on=False)
        else:
            ax.scatter(
                exact_lmbda, W_clnk_frame_avrg_so3_quad_uniaxl, s=markersize,
                marker=marker, linewidth=markerlinewidth, edgecolors=color,
                facecolors="None", clip_on=False)
        ax.scatter(
            exact_lmbda, W_clnk_frame_avrg_so3_quad_uniaxl, s=dotsize,
            marker="o", linewidth=dotlinewidth, edgecolors=color,
            facecolors=color, clip_on=False)
        ax.plot(
            approx_lmbda, W_clnk_frame_avrg_approx_so3_quad_uniaxl,
            linestyle="-", linewidth=markerlinewidth, c=color)
        handles.append((marker, color))
    ax.legend(
        handles=handles, labels=n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=16,
        labelspacing=0, markerfirst=False, frameon=False, loc="upper right",
        bbox_to_anchor=(0.575, 1.025))
    ax.set_xlim([0.4, 2.0])
    ax.set_ylim([5.5, 10.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0])
    ax.set_yticks([5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0])
    ax.set_xticklabels(["$~0.4$", "$0.6$", "$0.8$", "$1$", "$1.2$", "$1.4$", "$1.6$", "$1.8$", "$2$"])
    ax.set_yticklabels(["$5.5~$", "$6~$", "$6.5~$", "$7~$", "$7.5~$", "$8~$", "$8.5~$", "$9~$", "$9.5~$", "$10~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$W_{c, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(W_clnks_frame_avrg_so3_quad_uniaxl_agreement_plot_fig_filename)
    plt.close()

    W_clnks_free_rot_simple_shear_agreement_plot_fig_filename = (
        filepath + "W_clnks_free_rot_simple_shear_agreement_plot" + ".pdf"
    )
    fig, ax = plt.subplots()
    handles = []
    for clnk_indx in range(clnks_num):
        marker = clnk_marker[clnk_indx]
        color = clnk_color[clnk_indx]
        W_clnk_free_rot_simple_shear = W_clnks_free_rot_simple_shear[clnk_indx]
        W_clnk_free_rot_approx_simple_shear = W_clnks_free_rot_approx_simple_shear[clnk_indx]
        if marker == "+x":
            for plusx_marker in plusx_marker_list:
                ax.scatter(
                    exact_s, W_clnk_free_rot_simple_shear, s=markersize,
                    marker=plusx_marker, linewidth=markerlinewidth,
                    facecolors=color, clip_on=False)
        else:
            ax.scatter(
                exact_s, W_clnk_free_rot_simple_shear, s=markersize,
                marker=marker, linewidth=markerlinewidth, edgecolors=color,
                facecolors="None", clip_on=False)
        ax.scatter(
            exact_s, W_clnk_free_rot_simple_shear, s=dotsize, marker="o",
            linewidth=dotlinewidth, edgecolors=color, facecolors=color,
            clip_on=False)
        ax.plot(
            approx_s, W_clnk_free_rot_approx_simple_shear, linestyle="-",
            linewidth=markerlinewidth, c=color)
        handles.append((marker, color))
    ax.legend(
        handles=handles, labels=n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=16,
        labelspacing=0, markerfirst=False, frameon=False, loc="upper right",
        bbox_to_anchor=(0.575, 1.025))
    ax.set_xlim([0.0, 2.0])
    ax.set_ylim([5.0, 14.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    ax.set_xticklabels(["$0$", "$0.5$", "$1$", "$1.5$", "$2$"])
    ax.set_yticklabels(["$5~$", "$6~$", "$7~$", "$8~$", "$9~$", "$10~$", "$11~$", "$12~$", "$13~$", "$14~$"])
    ax.set_xlabel("$s$", fontsize=16)
    ax.set_ylabel("$W_{c, \\leftparen \\cdot \\rightparen }^{FR}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(W_clnks_free_rot_simple_shear_agreement_plot_fig_filename)
    plt.close()

    W_clnks_frame_avrg_so3_quad_simple_shear_agreement_plot_fig_filename = (
        filepath + "W_clnks_frame_avrg_so3_quad_simple_shear_agreement_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    handles = []
    for clnk_indx in range(clnks_num):
        marker = clnk_marker[clnk_indx]
        color = clnk_color[clnk_indx]
        W_clnk_frame_avrg_so3_quad_simple_shear = W_clnks_frame_avrg_so3_quad_simple_shear[clnk_indx]
        W_clnk_frame_avrg_approx_so3_quad_simple_shear = W_clnks_frame_avrg_approx_so3_quad_simple_shear[clnk_indx]
        if marker == "+x":
            for plusx_marker in plusx_marker_list:
                ax.scatter(
                    exact_s, W_clnk_frame_avrg_so3_quad_simple_shear,
                    s=markersize, marker=plusx_marker,
                    linewidth=markerlinewidth, facecolors=color, clip_on=False)
        else:
            ax.scatter(
                exact_s, W_clnk_frame_avrg_so3_quad_simple_shear, s=markersize,
                marker=marker, linewidth=markerlinewidth, edgecolors=color,
                facecolors="None", clip_on=False)
        ax.scatter(
            exact_s, W_clnk_frame_avrg_so3_quad_simple_shear, s=dotsize,
            marker="o", linewidth=dotlinewidth, edgecolors=color,
            facecolors=color, clip_on=False)
        ax.plot(
            approx_s, W_clnk_frame_avrg_approx_so3_quad_simple_shear,
            linestyle="-", linewidth=markerlinewidth, c=color)
        handles.append((marker, color))
    ax.legend(
        handles=handles, labels=n_clnks_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=16,
        labelspacing=0, markerfirst=False, frameon=False, loc="upper right",
        bbox_to_anchor=(0.575, 1.025))
    ax.set_xlim([0.0, 2.0])
    ax.set_ylim([5.0, 14.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.0, 0.5, 1.0, 1.5, 2.0])
    ax.set_yticks([5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0])
    ax.set_xticklabels(["$0$", "$0.5$", "$1$", "$1.5$", "$2$"])
    ax.set_yticklabels(["$5~$", "$6~$", "$7~$", "$8~$", "$9~$", "$10~$", "$11~$", "$12~$", "$13~$", "$14~$"])
    ax.set_xlabel("$s$", fontsize=16)
    ax.set_ylabel("$W_{c, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(
        W_clnks_frame_avrg_so3_quad_simple_shear_agreement_plot_fig_filename)
    plt.close()

    # # Save data in .csv files for plotting reproduction
    # exact_lmbda_and_W_clnks_free_rot_uniaxl_filename = (
    #     filepath + "exact_lmbda_and_W_clnks_free_rot_uniaxl" + ".csv"
    # )
    # approx_lmbda_and_W_clnks_free_rot_approx_uniaxl_filename = (
    #     filepath + "approx_lmbda_and_W_clnks_free_rot_approx_uniaxl"
    #     + ".csv"
    # )
    # exact_lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_filename = (
    #     filepath + "exact_lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl"
    #     + ".csv"
    # )
    # approx_lmbda_and_W_clnks_frame_avrg_approx_so3_quad_uniaxl_filename = (
    #     filepath
    #     + "approx_lmbda_and_W_clnks_frame_avrg_approx_so3_quad_uniaxl"
    #     + ".csv"
    # )
    # exact_s_and_W_clnks_free_rot_simple_shear_filename = (
    #     filepath + "exact_s_and_W_clnks_free_rot_simple_shear" + ".csv"
    # )
    # approx_s_and_W_clnks_free_rot_approx_simple_shear_filename = (
    #     filepath + "approx_s_and_W_clnks_free_rot_approx_simple_shear"
    #     + ".csv"
    # )
    # exact_s_and_W_clnks_frame_avrg_so3_quad_simple_shear_filename = (
    #     filepath + "exact_s_and_W_clnks_frame_avrg_so3_quad_simple_shear"
    #     + ".csv"
    # )
    # approx_s_and_W_clnks_frame_avrg_approx_so3_quad_simple_shear_filename = (
    #     filepath
    #     + "approx_s_and_W_clnks_frame_avrg_approx_so3_quad_simple_shear"
    #     + ".csv"
    # )
    
    # exact_lmbda_and_W_clnks_free_rot_uniaxl = exact_lmbda.copy()
    # approx_lmbda_and_W_clnks_free_rot_approx_uniaxl = approx_lmbda.copy()
    # exact_lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl = exact_lmbda.copy()
    # approx_lmbda_and_W_clnks_frame_avrg_approx_so3_quad_uniaxl = approx_lmbda.copy()
    # exact_s_and_W_clnks_free_rot_simple_shear = exact_s.copy()
    # approx_s_and_W_clnks_free_rot_approx_simple_shear = approx_s.copy()
    # exact_s_and_W_clnks_frame_avrg_so3_quad_simple_shear = exact_s.copy()
    # approx_s_and_W_clnks_frame_avrg_approx_so3_quad_simple_shear = approx_s.copy()
    
    # for clnk_indx in range(clnks_num):
    #     exact_lmbda_and_W_clnks_free_rot_uniaxl = np.column_stack(
    #         (exact_lmbda_and_W_clnks_free_rot_uniaxl, W_clnks_free_rot_uniaxl[clnk_indx]))
    #     approx_lmbda_and_W_clnks_free_rot_approx_uniaxl = np.column_stack(
    #         (approx_lmbda_and_W_clnks_free_rot_approx_uniaxl, W_clnks_free_rot_approx_uniaxl[clnk_indx]))
    #     exact_lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl = np.column_stack(
    #         (exact_lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl, W_clnks_frame_avrg_so3_quad_uniaxl[clnk_indx]))
    #     approx_lmbda_and_W_clnks_frame_avrg_approx_so3_quad_uniaxl = np.column_stack(
    #         (approx_lmbda_and_W_clnks_frame_avrg_approx_so3_quad_uniaxl, W_clnks_frame_avrg_approx_so3_quad_uniaxl[clnk_indx]))
    #     exact_s_and_W_clnks_free_rot_simple_shear = np.column_stack(
    #         (exact_s_and_W_clnks_free_rot_simple_shear, W_clnks_free_rot_simple_shear[clnk_indx]))
    #     approx_s_and_W_clnks_free_rot_approx_simple_shear = np.column_stack(
    #         (approx_s_and_W_clnks_free_rot_approx_simple_shear, W_clnks_free_rot_approx_simple_shear[clnk_indx]))
    #     exact_s_and_W_clnks_frame_avrg_so3_quad_simple_shear = np.column_stack(
    #         (exact_s_and_W_clnks_frame_avrg_so3_quad_simple_shear, W_clnks_frame_avrg_so3_quad_simple_shear[clnk_indx]))
    #     approx_s_and_W_clnks_frame_avrg_approx_so3_quad_simple_shear = np.column_stack(
    #         (approx_s_and_W_clnks_frame_avrg_approx_so3_quad_simple_shear, W_clnks_frame_avrg_approx_so3_quad_simple_shear[clnk_indx]))
    
    # np.savetxt(
    #     exact_lmbda_and_W_clnks_free_rot_uniaxl_filename,
    #     exact_lmbda_and_W_clnks_free_rot_uniaxl, delimiter=",")
    # np.savetxt(
    #     approx_lmbda_and_W_clnks_free_rot_approx_uniaxl_filename,
    #     approx_lmbda_and_W_clnks_free_rot_approx_uniaxl, delimiter=",")
    # np.savetxt(
    #     exact_lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl_filename,
    #     exact_lmbda_and_W_clnks_frame_avrg_so3_quad_uniaxl, delimiter=",")
    # np.savetxt(
    #     approx_lmbda_and_W_clnks_frame_avrg_approx_so3_quad_uniaxl_filename,
    #     approx_lmbda_and_W_clnks_frame_avrg_approx_so3_quad_uniaxl, delimiter=",")
    # np.savetxt(
    #     exact_s_and_W_clnks_free_rot_simple_shear_filename,
    #     exact_s_and_W_clnks_free_rot_simple_shear, delimiter=",")
    # np.savetxt(
    #     approx_s_and_W_clnks_free_rot_approx_simple_shear_filename,
    #     approx_s_and_W_clnks_free_rot_approx_simple_shear, delimiter=",")
    # np.savetxt(
    #     exact_s_and_W_clnks_frame_avrg_so3_quad_simple_shear_filename,
    #     exact_s_and_W_clnks_frame_avrg_so3_quad_simple_shear, delimiter=",")
    # np.savetxt(
    #     approx_s_and_W_clnks_frame_avrg_approx_so3_quad_simple_shear_filename,
    #     approx_s_and_W_clnks_frame_avrg_approx_so3_quad_simple_shear, delimiter=",")

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse Gaussian end-linked polymer network elastically-effective cross-link RVE deformation analysis plotting took {execution_time} seconds to run")