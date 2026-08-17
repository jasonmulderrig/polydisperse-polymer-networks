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
        config_path="../configs/polydisperse_cufjc_networks_clnk_rves",
        config_name="config")
def main(cfg: DictConfig) -> None:
    clnk_color = ["tab:purple", "tab:green", "tab:blue", "tab:orange"]
    clnk_marker = ["^", "s", "p", "+x"]
    plusx_marker_list = ["+", "x"]
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
    
    filepath = filepath_str("polydisperse_cufjc_networks_clnk_rves")

    exact_filename_prefix = filename_str(cfg.label.workdir, "20260603", "A", 0)
    approx_filename_prefix = filename_str(cfg.label.workdir, "20260603", "B", 0)

    exact_n_clnks_filename = exact_filename_prefix + "-n_clnks" + ".npy"
    approx_n_clnks_filename = approx_filename_prefix + "-n_clnks" + ".npy"
    exact_n_clnks = np.load(exact_n_clnks_filename)
    approx_n_clnks = np.load(approx_n_clnks_filename)
    assert np.allclose(exact_n_clnks, approx_n_clnks)
    n_clnks = exact_n_clnks

    clnks_num = np.shape(n_clnks)[0]
    n_clnks_13_rves_indcs = list(range(0, 4))
    n_clnks_22_rves_indcs = list(range(4, clnks_num))
    n_clnks_13_rves = n_clnks[n_clnks_13_rves_indcs]
    n_clnks_22_rves = n_clnks[n_clnks_22_rves_indcs]
    n_clnks_13_rves_legend = n_clnks_legend_func(n_clnks_13_rves)
    n_clnks_22_rves_legend = n_clnks_legend_func(n_clnks_22_rves)

    exact_uniaxl_tens_lmbda_filename = (
        exact_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    exact_uniaxl_comp_lmbda_filename = (
        exact_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )
    approx_uniaxl_tens_lmbda_filename = (
        approx_filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    approx_uniaxl_comp_lmbda_filename = (
        approx_filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )

    exact_uniaxl_tens_lmbda = np.load(exact_uniaxl_tens_lmbda_filename)
    exact_uniaxl_tens_lmbda = exact_uniaxl_tens_lmbda[1:]
    exact_uniaxl_comp_lmbda = np.flip(np.load(exact_uniaxl_comp_lmbda_filename))
    exact_lmbda = np.hstack((exact_uniaxl_comp_lmbda, exact_uniaxl_tens_lmbda))
    approx_uniaxl_tens_lmbda = np.load(approx_uniaxl_tens_lmbda_filename)
    approx_uniaxl_tens_lmbda = approx_uniaxl_tens_lmbda[1:]
    approx_uniaxl_comp_lmbda = np.flip(np.load(approx_uniaxl_comp_lmbda_filename))
    approx_lmbda = np.hstack((approx_uniaxl_comp_lmbda, approx_uniaxl_tens_lmbda))

    lmbda_inset_ax_min, lmbda_inset_ax_max = 1.0, 1.3
    exact_lmbda_inset_ax_indcs = (
        np.where(np.logical_and(exact_lmbda>=lmbda_inset_ax_min, exact_lmbda<=lmbda_inset_ax_max))[0]
    )
    approx_lmbda_inset_ax_indcs = (
        np.where(np.logical_and(approx_lmbda>=lmbda_inset_ax_min, approx_lmbda<=lmbda_inset_ax_max))[0]
    )

    W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename = (
        exact_filename_prefix
        + "-W_clnks_frame_avrg_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename = (
        exact_filename_prefix
        + "-W_clnks_frame_avrg_so3_quad_protocol_indx_1" + ".npy"
    )
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename = (
        approx_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_0" + ".npy"
    )
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename = (
        approx_filename_prefix
        + "-W_clnks_frame_avrg_approx_so3_quad_protocol_indx_1" + ".npy"
    )

    W_clnks_frame_avrg_so3_quad_uniaxl_tens = np.load(
        W_clnks_frame_avrg_so3_quad_uniaxl_tens_filename)
    W_clnks_frame_avrg_so3_quad_uniaxl_tens = (
        W_clnks_frame_avrg_so3_quad_uniaxl_tens[:, 1:]
    )
    W_clnks_frame_avrg_so3_quad_uniaxl_comp = np.flip(
        np.load(W_clnks_frame_avrg_so3_quad_uniaxl_comp_filename),
        axis=1)
    W_clnks_frame_avrg_so3_quad_uniaxl = np.hstack(
        (W_clnks_frame_avrg_so3_quad_uniaxl_comp,
            W_clnks_frame_avrg_so3_quad_uniaxl_tens))
    
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens = np.load(
        W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens_filename)
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens = (
        W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens[:, 1:]
    )
    W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp = np.flip(
        np.load(W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp_filename),
        axis=1)
    W_clnks_frame_avrg_approx_so3_quad_uniaxl = np.hstack(
        (W_clnks_frame_avrg_approx_so3_quad_uniaxl_comp,
            W_clnks_frame_avrg_approx_so3_quad_uniaxl_tens))

    W_clnks_frame_avrg_so3_quad_uniaxl_agreement_13_rves_plot_fig_filename = (
        filepath + "W_clnks_frame_avrg_so3_quad_uniaxl_agreement_13_rves_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    handles = []
    plt_format_indx = 0
    for clnk_indx in n_clnks_13_rves_indcs:
        marker = clnk_marker[plt_format_indx]
        color = clnk_color[plt_format_indx]
        W_clnk_frame_avrg_so3_quad_uniaxl = (
            W_clnks_frame_avrg_so3_quad_uniaxl[clnk_indx]
        )
        W_clnk_frame_avrg_approx_so3_quad_uniaxl = (
            W_clnks_frame_avrg_approx_so3_quad_uniaxl[clnk_indx]
        )
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
        plt_format_indx += 1
    ax.legend(
        handles=handles, labels=n_clnks_13_rves_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=16,
        labelspacing=0, markerfirst=False, frameon=False, loc="upper right",
        bbox_to_anchor=(0.55, 1.025))
    ax.set_xlim([0.5, 4.0])
    ax.set_ylim([5.0, 35.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ax.set_yticks([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    ax.set_xticklabels(["$~0.5$", "$1$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$"])
    ax.set_yticklabels(["$5~$", "$10~$", "$15~$", "$20~$", "$25~$", "$30~$", "$35~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel(
        "$W_{c, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$", fontsize=16)
    inset_ax = ax.inset_axes([0.15, 0.40, 0.40, 0.325])
    plt_format_indx = 0
    for clnk_indx in n_clnks_13_rves_indcs:
        marker = clnk_marker[plt_format_indx]
        color = clnk_color[plt_format_indx]
        W_clnk_frame_avrg_so3_quad_uniaxl = (
            W_clnks_frame_avrg_so3_quad_uniaxl[clnk_indx]
        )
        W_clnk_frame_avrg_approx_so3_quad_uniaxl = (
            W_clnks_frame_avrg_approx_so3_quad_uniaxl[clnk_indx]
        )
        if marker == "+x":
            for plusx_marker in plusx_marker_list:
                inset_ax.scatter(
                    exact_lmbda[exact_lmbda_inset_ax_indcs],
                    W_clnk_frame_avrg_so3_quad_uniaxl[exact_lmbda_inset_ax_indcs],
                    s=markersize, marker=plusx_marker, linewidth=markerlinewidth,
                    facecolors=color, clip_on=False)
        else:
            inset_ax.scatter(
                exact_lmbda[exact_lmbda_inset_ax_indcs],
                W_clnk_frame_avrg_so3_quad_uniaxl[exact_lmbda_inset_ax_indcs],
                s=markersize, marker=marker, linewidth=markerlinewidth,
                edgecolors=color, facecolors="None", clip_on=False)
        inset_ax.scatter(
            exact_lmbda[exact_lmbda_inset_ax_indcs],
            W_clnk_frame_avrg_so3_quad_uniaxl[exact_lmbda_inset_ax_indcs],
            s=dotsize, marker="o", linewidth=dotlinewidth, edgecolors=color,
            facecolors=color, clip_on=False)
        inset_ax.plot(
            approx_lmbda[approx_lmbda_inset_ax_indcs],
            W_clnk_frame_avrg_approx_so3_quad_uniaxl[approx_lmbda_inset_ax_indcs],
            linestyle="-", linewidth=markerlinewidth, c=color)
        plt_format_indx += 1
    inset_ax.set_xlim([lmbda_inset_ax_min, lmbda_inset_ax_max])
    inset_ax.set_ylim([5.8, 6.4])
    inset_ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=12)
    inset_ax.set_xticks([1.0, 1.1, 1.2, 1.3])
    inset_ax.set_yticks([5.9, 6.2])
    inset_ax.set_xticklabels(["$~1$", "$1.1$", "$1.2$", "$1.3$"])
    inset_ax.set_yticklabels(["$5.9~$", "$6.2~$"])
    fig.tight_layout()
    fig.savefig(
        W_clnks_frame_avrg_so3_quad_uniaxl_agreement_13_rves_plot_fig_filename)
    plt.close()

    W_clnks_frame_avrg_so3_quad_uniaxl_agreement_22_rves_plot_fig_filename = (
        filepath + "W_clnks_frame_avrg_so3_quad_uniaxl_agreement_22_rves_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    handles = []
    plt_format_indx = 0
    for clnk_indx in n_clnks_22_rves_indcs:
        marker = clnk_marker[plt_format_indx]
        color = clnk_color[plt_format_indx]
        W_clnk_frame_avrg_so3_quad_uniaxl = (
            W_clnks_frame_avrg_so3_quad_uniaxl[clnk_indx]
        )
        W_clnk_frame_avrg_approx_so3_quad_uniaxl = (
            W_clnks_frame_avrg_approx_so3_quad_uniaxl[clnk_indx]
        )
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
        plt_format_indx += 1
    ax.legend(
        handles=handles, labels=n_clnks_22_rves_legend,
        handler_map={tuple: HandlerCompositeMarker()}, fontsize=16,
        labelspacing=0, markerfirst=False, frameon=False, loc="upper right",
        bbox_to_anchor=(0.525, 1.025))
    ax.set_xlim([0.5, 4.0])
    ax.set_ylim([5.0, 35.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0])
    ax.set_yticks([5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0])
    ax.set_xticklabels(["$~0.5$", "$1$", "$1.5$", "$2$", "$2.5$", "$3$", "$3.5$", "$4$"])
    ax.set_yticklabels(["$5~$", "$10~$", "$15~$", "$20~$", "$25~$", "$30~$", "$35~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel(
        "$W_{c, \\leftparen \\cdot \\rightparen }^{FA}/k_BT$", fontsize=16)
    inset_ax = ax.inset_axes([0.15, 0.40, 0.40, 0.325])
    plt_format_indx = 0
    for clnk_indx in n_clnks_22_rves_indcs:
        marker = clnk_marker[plt_format_indx]
        color = clnk_color[plt_format_indx]
        W_clnk_frame_avrg_so3_quad_uniaxl = (
            W_clnks_frame_avrg_so3_quad_uniaxl[clnk_indx]
        )
        W_clnk_frame_avrg_approx_so3_quad_uniaxl = (
            W_clnks_frame_avrg_approx_so3_quad_uniaxl[clnk_indx]
        )
        if marker == "+x":
            for plusx_marker in plusx_marker_list:
                inset_ax.scatter(
                    exact_lmbda[exact_lmbda_inset_ax_indcs],
                    W_clnk_frame_avrg_so3_quad_uniaxl[exact_lmbda_inset_ax_indcs],
                    s=markersize, marker=plusx_marker, linewidth=markerlinewidth,
                    facecolors=color, clip_on=False)
        else:
            inset_ax.scatter(
                exact_lmbda[exact_lmbda_inset_ax_indcs],
                W_clnk_frame_avrg_so3_quad_uniaxl[exact_lmbda_inset_ax_indcs],
                s=markersize, marker=marker, linewidth=markerlinewidth,
                edgecolors=color, facecolors="None", clip_on=False)
        inset_ax.scatter(
            exact_lmbda[exact_lmbda_inset_ax_indcs],
            W_clnk_frame_avrg_so3_quad_uniaxl[exact_lmbda_inset_ax_indcs],
            s=dotsize, marker="o", linewidth=dotlinewidth, edgecolors=color,
            facecolors=color, clip_on=False)
        inset_ax.plot(
            approx_lmbda[approx_lmbda_inset_ax_indcs],
            W_clnk_frame_avrg_approx_so3_quad_uniaxl[approx_lmbda_inset_ax_indcs],
            linestyle="-", linewidth=markerlinewidth, c=color)
        plt_format_indx += 1
    inset_ax.set_xlim([lmbda_inset_ax_min, lmbda_inset_ax_max])
    inset_ax.set_ylim([5.8, 6.4])
    inset_ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=12)
    inset_ax.set_xticks([1.0, 1.1, 1.2, 1.3])
    inset_ax.set_yticks([5.9, 6.2])
    inset_ax.set_xticklabels(["$~1$", "$1.1$", "$1.2$", "$1.3$"])
    inset_ax.set_yticklabels(["$5.9~$", "$6.2~$"])
    fig.tight_layout()
    fig.savefig(
        W_clnks_frame_avrg_so3_quad_uniaxl_agreement_22_rves_plot_fig_filename)
    plt.close()

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse cuFJC end-linked polymer network elastically-effective cross-link RVE deformation analysis plotting took {execution_time} seconds to run")