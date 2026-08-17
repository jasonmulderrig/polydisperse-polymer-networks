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
    clnk_color = ["black", "tab:purple", "tab:blue", "tab:green", "tab:orange", "tab:red", "tab:brown"]
    markerlinewidth = 0.5

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

    filepath = filepath_str("polydisperse_inext_gaussian_fjc_networks_clnk_rves")
    filename_prefix = filename_str(cfg.label.workdir, "20260603", "E", 0)

    n_clnks_filename = filename_prefix + "-n_clnks" + ".npy"
    n_clnks = np.load(n_clnks_filename)
    clnks_num = geo_isomrphc_sets_num = np.shape(n_clnks)[0]
    n_clnks_13_rves_indcs, n_clnks_22_rves_indcs = [0], [0]
    n_clnks_13_rves_indcs.extend(list(range(1, 4)))
    n_clnks_22_rves_indcs.extend(list(range(4, clnks_num)))
    n_clnks_13_rves = n_clnks[n_clnks_13_rves_indcs]
    n_clnks_22_rves = n_clnks[n_clnks_22_rves_indcs]
    n_clnks_13_rves_legend = n_clnks_legend_func(n_clnks_13_rves)
    n_clnks_22_rves_legend = n_clnks_legend_func(n_clnks_22_rves)

    n_clnks_geo_isomrphc_sets = []
    for set_indx in range(geo_isomrphc_sets_num):
        n_clnks_geo_isomrphc_set_filename = (
            filename_prefix
            + f"-n_clnks_geo_isomrphc_set_indx_{set_indx:d}" + ".npy"
        )
        n_clnks_geo_isomrphc_sets.append(
            np.load(n_clnks_geo_isomrphc_set_filename))

    n_clnks_geo_isomrphc_sets_legends = []
    for set_indx in range(geo_isomrphc_sets_num):
        n_clnks_geo_isomrphc_sets_legends.append(
            n_clnks_legend_func(n_clnks_geo_isomrphc_sets[set_indx]))
    
    uniaxl_tens_lmbda_filename = (
        filename_prefix + "-dfrmtn_protocol_indx_0" + ".npy"
    )
    uniaxl_comp_lmbda_filename = (
        filename_prefix + "-dfrmtn_protocol_indx_1" + ".npy"
    )

    uniaxl_tens_lmbda = np.load(uniaxl_tens_lmbda_filename)
    uniaxl_tens_lmbda = uniaxl_tens_lmbda[1:]
    uniaxl_comp_lmbda = np.flip(np.load(uniaxl_comp_lmbda_filename))
    lmbda = np.hstack((uniaxl_comp_lmbda, uniaxl_tens_lmbda))
    
    E_clnks_geo_isomrphc_sets_free_rot_uniaxl = []
    for set_indx in range(geo_isomrphc_sets_num):
        W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens_filename = (
            filename_prefix
            + "-W_clnks_geo_isomrphc_sets_free_rot_approx_protocol_indx_0"
            + "_" + f"set_indx_{set_indx:d}" + ".npy"
        )
        W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp_filename = (
            filename_prefix
            + "-W_clnks_geo_isomrphc_sets_free_rot_approx_protocol_indx_1"
            + "_" + f"set_indx_{set_indx:d}" + ".npy"
        )

        W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens = np.load(
            W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens_filename)
        W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens = (
            W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens[:, 1:]
        )
        W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp = np.flip(
            np.load(W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp_filename),
            axis=1)
        
        sigma_11_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens = (
            np.gradient(
                W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens,
                uniaxl_tens_lmbda, axis=1, edge_order=2)
        )
        sigma_11_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp = (
            np.gradient(
                W_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp,
                uniaxl_comp_lmbda, axis=1, edge_order=2)
        )

        E_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens = (
            np.gradient(
                sigma_11_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens,
                uniaxl_tens_lmbda, axis=1, edge_order=2)
        )
        E_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp = (
            np.gradient(
                sigma_11_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp,
                uniaxl_comp_lmbda, axis=1, edge_order=2)
        )

        E_clnks_geo_isomrphc_sets_free_rot_uniaxl.append(
            np.hstack(
                (E_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_comp,
                 E_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_tens)))

    for set_indx in range(geo_isomrphc_sets_num):
        n_clnks_geo_isomrphc_set_legend = (
            n_clnks_geo_isomrphc_sets_legends[set_indx]
        )
        E_clnks_geo_isomrphc_set_free_rot_uniaxl = (
            E_clnks_geo_isomrphc_sets_free_rot_uniaxl[set_indx]
        )
        clnks_num = np.shape(E_clnks_geo_isomrphc_set_free_rot_uniaxl)[0]
        E_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_plot_fig_filename = (
            filepath + "E_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_plot" 
            + "_" + f"set_indx_{set_indx:d}" + ".pdf"
        )
        fig, ax = plt.subplots()
        for clnk_indx in range(clnks_num):
            ax.plot(
                lmbda, E_clnks_geo_isomrphc_set_free_rot_uniaxl[clnk_indx],
                linestyle="-", linewidth=markerlinewidth, c=clnk_color[clnk_indx],
                label=n_clnks_geo_isomrphc_set_legend[clnk_indx])
        ax.legend(
            fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
            loc="best")
        ax.set_xlim([0.5, 1.4])
        ax.set_ylim([0.0, 70.0])
        ax.tick_params(
            bottom=True, top=True, left=True, right=True, direction="in",
            labelsize=16)
        ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
        ax.set_yticks([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
        ax.set_xticklabels(["$~0.5$", "$0.6$", "$0.7$", "$0.8$", "$0.9$", "$1$", "$1.1$", "$1.2$", "$1.3$", "$1.4$"])
        ax.set_yticklabels(["$0~$", "$10~$", "$20~$", "$30~$", "$40~$", "$50~$", "$60~$", "$70~$"])
        ax.set_xlabel("$\\lambda$", fontsize=16)
        ax.set_ylabel(
            "$E_{\\leftparen \\cdot \\rightparen }/k_BT$",
            fontsize=16)
        fig.tight_layout()
        fig.savefig(E_clnks_geo_isomrphc_set_free_rot_approx_uniaxl_plot_fig_filename)
        plt.close()
    
    E_clnks_free_rot_uniaxl_13_rves_plot_fig_filename = (
        filepath + "JMPS_2026_fig_6a_E_clnks_free_rot_uniaxl_13_rves_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    plt_format_indx = 0
    for set_indx in n_clnks_13_rves_indcs:
        if plt_format_indx == 0: linestyle = "--"
        else: linestyle = "-"
        ax.plot(
            lmbda, E_clnks_geo_isomrphc_sets_free_rot_uniaxl[set_indx][-1],
            linestyle=linestyle, linewidth=markerlinewidth,
            c=clnk_color[plt_format_indx],
            label=n_clnks_13_rves_legend[plt_format_indx])
        plt_format_indx += 1
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="best")
    ax.set_xlim([0.5, 1.4])
    ax.set_ylim([0.0, 70.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    ax.set_yticks([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    ax.set_xticklabels(["$~0.5$", "$0.6$", "$0.7$", "$0.8$", "$0.9$", "$1$", "$1.1$", "$1.2$", "$1.3$", "$1.4$"])
    ax.set_yticklabels(["$0~$", "$10~$", "$20~$", "$30~$", "$40~$", "$50~$", "$60~$", "$70~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$E_{\\leftparen \\cdot \\rightparen }/k_BT$", fontsize=16)
    inset_ax = ax.inset_axes([0.4, 0.4, 0.5, 0.25])
    plt_format_indx = 0
    for set_indx in n_clnks_13_rves_indcs:
        if plt_format_indx == 0: linestyle = "--"
        else: linestyle = "-"
        inset_ax.plot(
            lmbda, E_clnks_geo_isomrphc_sets_free_rot_uniaxl[set_indx][-1],
            linestyle=linestyle, linewidth=markerlinewidth,
            c=clnk_color[plt_format_indx],
            label=n_clnks_13_rves_legend[plt_format_indx])
        plt_format_indx += 1
    inset_ax.set_xlim([0.8, 1.1])
    inset_ax.set_ylim([8.0, 20.0])
    inset_ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=12)
    inset_ax.set_xticks([0.8, 0.9, 1.0, 1.1])
    inset_ax.set_xticklabels(["$0.8$", "$0.9$", "$1$", "$1.1$"])
    inset_ax.set_yticks([])
    inset_ax.set_yticklabels([])
    fig.tight_layout()
    fig.savefig(E_clnks_free_rot_uniaxl_13_rves_plot_fig_filename)
    plt.close()

    E_clnks_free_rot_uniaxl_22_rves_plot_fig_filename = (
        filepath + "JMPS_2026_fig_6b_E_clnks_free_rot_uniaxl_22_rves_plot"
        + ".pdf"
    )
    fig, ax = plt.subplots()
    plt_format_indx = 0
    for set_indx in n_clnks_22_rves_indcs:
        if plt_format_indx == 0: linestyle = "--"
        else: linestyle = "-"
        ax.plot(
            lmbda, E_clnks_geo_isomrphc_sets_free_rot_uniaxl[set_indx][-1],
            linestyle=linestyle, linewidth=markerlinewidth,
            c=clnk_color[plt_format_indx],
            label=n_clnks_22_rves_legend[plt_format_indx])
        plt_format_indx += 1
    ax.legend(
        fontsize=16, labelspacing=0, markerfirst=False, frameon=False,
        loc="best")
    ax.set_xlim([0.5, 1.4])
    ax.set_ylim([0.0, 70.0])
    ax.tick_params(
        bottom=True, top=True, left=True, right=True, direction="in",
        labelsize=16)
    ax.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4])
    ax.set_yticks([0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0])
    ax.set_xticklabels(["$~0.5$", "$0.6$", "$0.7$", "$0.8$", "$0.9$", "$1$", "$1.1$", "$1.2$", "$1.3$", "$1.4$"])
    ax.set_yticklabels(["$0~$", "$10~$", "$20~$", "$30~$", "$40~$", "$50~$", "$60~$", "$70~$"])
    ax.set_xlabel("$\\lambda$", fontsize=16)
    ax.set_ylabel("$E_{\\leftparen \\cdot \\rightparen }/k_BT$", fontsize=16)
    fig.tight_layout()
    fig.savefig(E_clnks_free_rot_uniaxl_22_rves_plot_fig_filename)
    plt.close()

if __name__ == "__main__":
    import time
    
    start_time = time.perf_counter()
    main()
    end_time = time.perf_counter()

    execution_time = end_time - start_time
    print(f"Polydisperse Gaussian end-linked polymer network elastically-effective cross-link RVE deformation analysis plotting took {execution_time} seconds to run")