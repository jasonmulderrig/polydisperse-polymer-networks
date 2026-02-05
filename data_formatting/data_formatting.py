# Add current path to system path for direct execution
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

# Import logging
import logging
logging.disable(logging.WARNING)

# Import modules
import numpy as np
from src.file_io.file_io import data_filepath_str

if __name__ == "__main__":
    data_filepath = data_filepath_str()
    mark_filenames = [
        data_filepath+"bimodal-pdms-ntwrks-andrady-llorente-mark-end-linked-pdms-chains-ix-1980",
        data_filepath+"bimodal-pdms-ntwrks-andrady-llorente-mark-end-linked-pdms-chains-vii-1980",
        data_filepath+"bimodal-pdms-ntwrks-andrady-llorente-mark-end-linked-pdms-chains-xi-1980",
        data_filepath+"bimodal-pdms-ntwrks-mark-tang-dependence-1984",
        data_filepath+"bimodal-pdms-ntwrks-mark-tang-effect-1984",
        data_filepath+"bimodal-pdms-ntwrks-wang-mark-unimodal-1992",
        data_filepath+"bimodal-pdms-ntwrks-xu-mark-biaxial-1991"
    ]
    
    # Save network parameters as .dat files
    for filename in mark_filenames:
        ntwrk_params_csv_filename = filename + "-ntwrk-params" + ".csv"
        ntwrk_params_dat_filename = filename + "-ntwrk-params" + ".dat"
        with open(ntwrk_params_csv_filename, 'r', encoding='utf-8-sig') as ntwrk_params_csv_file: 
            ntwrk_params = np.genfromtxt(
                ntwrk_params_csv_file, dtype=float, delimiter=',')
        np.savetxt(ntwrk_params_dat_filename, ntwrk_params, fmt='%.3f')
    
    # Save proper mechanical response parameters as .dat files
    for filename in mark_filenames:
        ntwrk_params_dat_filename = filename + "-ntwrk-params" + ".dat"
        num_ntwrks = np.shape(np.loadtxt(ntwrk_params_dat_filename))[0]
        for ntwrk in range(num_ntwrks):
            ntwrk_mechanical_test_dat_filename = (
                filename + f"-ntwrk_{ntwrk:d}" + "-stress-vs-stretch" + ".dat"
            )
            if filename == data_filepath+"bimodal-pdms-ntwrks-wang-mark-unimodal-1992":
                ntwrk_mechanical_test_csv_filename = (
                    filename + f"-ntwrk_{ntwrk:d}"
                    + "-reduced-shear-stress-vs-stretch" + ".csv"
                )
            else:
                ntwrk_mechanical_test_csv_filename = (
                    filename + f"-ntwrk_{ntwrk:d}"
                    + "-reduced-stress-vs-inv-stretch" + ".csv"
                )
            with open(ntwrk_mechanical_test_csv_filename, 'r', encoding='utf-8-sig') as ntwrk_mechanical_test_csv_file: 
                ntwrk_mechanical_test = np.genfromtxt(
                    ntwrk_mechanical_test_csv_file, dtype=float, delimiter=',')
            if filename == data_filepath+"bimodal-pdms-ntwrks-wang-mark-unimodal-1992":
                stretch = ntwrk_mechanical_test[:, 0]
                reduced_shear_stress = ntwrk_mechanical_test[:, 1]
                inv_stretch = np.reciprocal(stretch)
                stress = reduced_shear_stress * (stretch-inv_stretch**3)
            else:
                inv_stretch = ntwrk_mechanical_test[:, 0]
                reduced_stress = ntwrk_mechanical_test[:, 1]
                stretch = np.reciprocal(inv_stretch)
                stress = reduced_stress * (stretch-inv_stretch**2)
            stretch_round_val = 0.005
            stretch = (
                np.round(stretch*(1./stretch_round_val)) / (1./stretch_round_val)
            )
            _, stretch_counts = np.unique(stretch, return_counts=True)
            if np.any(stretch_counts>1):
                print(filename + f"-ntwrk_{ntwrk:d}")
                print("stretch = {}".format(stretch))
                print("stress = {}".format(stress))
                print("inv_stretch = {}".format(inv_stretch))
                print("reduced_stress = {}".format(reduced_stress))
            ntwrk_mechanical_test = np.column_stack((stretch, stress))
            np.savetxt(
                ntwrk_mechanical_test_dat_filename, ntwrk_mechanical_test,
                fmt='%.3f')
