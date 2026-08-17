#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source bin/activate

# Run polydisperse_inext_gaussian_fjc_networks_clnk_rves_exact_dfrmtn.py with proper configurations
cd configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603A
  - deformation: 20260603A
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: A
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_clnk_rves_exact_dfrmtn.py
##### Time benchmark: 3412 seconds

# Run polydisperse_inext_gaussian_fjc_networks_clnk_rves_approx_dfrmtn.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603B
  - deformation: 20260603B
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: B
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_clnk_rves_approx_dfrmtn.py
##### Time benchmark: 88 seconds

# Run polydisperse_inext_gaussian_fjc_networks_clnk_rves_exact_monodisperse_dfrmtn.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603C
  - deformation: 20260603C
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: C
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_clnk_rves_exact_monodisperse_dfrmtn.py
##### Time benchmark: 5.5 seconds

# Run polydisperse_inext_gaussian_fjc_networks_clnk_rves_exact_general_dfrmtn.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603D
  - deformation: 20260603D
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: D
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_clnk_rves_exact_general_dfrmtn.py
##### Time benchmark: 6031 seconds

# Run polydisperse_inext_gaussian_fjc_networks_clnk_rves_free_rot_approx_dfrmtn.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603E
  - deformation: 20260603E
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: E
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_clnk_rves_free_rot_approx_dfrmtn.py
##### Time benchmark: 44 seconds

# Run polydisperse_inext_gaussian_fjc_networks_clnk_rves_exact_dfrmtn.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603F
  - deformation: 20260603F
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: F
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_clnk_rves_exact_dfrmtn.py
##### Time benchmark: 207 seconds

# Run polydisperse_inext_gaussian_fjc_networks_clnk_rves_exact_dfrmtn.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603G
  - deformation: 20260603G
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: G
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_clnk_rves_exact_dfrmtn.py
##### Time benchmark: 249 seconds

# Run polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_exact_dfrmtn.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603A
  - deformation: 20260603A
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: A
  sample: 0
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_exact_dfrmtn.py
##### Time benchmark: 7008 seconds

# Run polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_approx_dfrmtn.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603B
  - deformation: 20260603B
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: B
  sample: 0
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_approx_dfrmtn.py
##### Time benchmark: 111 seconds

# Run polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_exact_monodisperse_dfrmtn.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603C
  - deformation: 20260603C
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: C
  sample: 0
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_exact_monodisperse_dfrmtn.py
##### Time benchmark: 422 seconds

# Run polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_bimodal_ntwrk_frame_avrg_approx_dfrmtn.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603D
  - deformation: 20260603D
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: D
  sample: 0
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_bimodal_ntwrk_frame_avrg_approx_dfrmtn.py
##### Time benchmark: 1848 seconds

# Run polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_bimodal_ntwrk_free_rot_approx_dfrmtn.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603E
  - deformation: 20260603E
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: E
  sample: 0
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_bimodal_ntwrk_free_rot_approx_dfrmtn.py
##### Time benchmark: 21 seconds

# Run polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_bimodal_ntwrk_frame_avrg_approx_dfrmtn.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603F
  - deformation: 20260603F
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: F
  sample: 0
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_bimodal_ntwrk_frame_avrg_approx_dfrmtn.py
##### Time benchmark: 1270 seconds

# Run polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_bimodal_ntwrk_free_rot_approx_dfrmtn.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603G
  - deformation: 20260603G
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20260603
  batch: G
  sample: 0
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_bimodal_ntwrk_free_rot_approx_dfrmtn.py
##### Time benchmark: 14 seconds

# Run plotting codes
cd ../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 plot_polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_agreement.py
python3 plot_polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_general_dfrmtn.py
python3 plot_polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_free_rot_approx_tangent_stiffness_modulus.py
python3 plot_polydisperse_inext_gaussian_fjc_networks_clnk_rves_exact_comparison.py
cd ../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 plot_polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_agreement.py
python3 plot_polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves_bimodal_ntwrk_dfrmtn.py

# Deactivate virtual environment
deactivate

# Total runtime: 20736.5 seconds ~ 5.75 hours