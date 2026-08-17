#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source bin/activate

# Run polydisperse_cufjc_networks_clnk_rves_exact_dfrmtn.py with proper configurations
cd configs/polydisperse_cufjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603A
  - deformation: 20260603A
  - _self_
label:
  workdir: polydisperse_cufjc_networks_clnk_rves
  date: !!str 20260603
  batch: A
  sample: 0
EOL
cd ../../polydisperse_cufjc_networks_clnk_rves
python3 polydisperse_cufjc_networks_clnk_rves_exact_dfrmtn.py
##### Time benchmark: 11158 seconds

# Run polydisperse_cufjc_networks_clnk_rves_approx_dfrmtn.py with proper configurations
cd ../configs/polydisperse_cufjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603B
  - deformation: 20260603B
  - _self_
label:
  workdir: polydisperse_cufjc_networks_clnk_rves
  date: !!str 20260603
  batch: B
  sample: 0
EOL
cd ../../polydisperse_cufjc_networks_clnk_rves
python3 polydisperse_cufjc_networks_clnk_rves_approx_dfrmtn.py
##### Time benchmark: 409 seconds

# Run polydisperse_cufjc_networks_clnk_rves_exact_monodisperse_dfrmtn.py with proper configurations
cd ../configs/polydisperse_cufjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603C
  - deformation: 20260603C
  - _self_
label:
  workdir: polydisperse_cufjc_networks_clnk_rves
  date: !!str 20260603
  batch: C
  sample: 0
EOL
cd ../../polydisperse_cufjc_networks_clnk_rves
python3 polydisperse_cufjc_networks_clnk_rves_exact_monodisperse_dfrmtn.py
##### Time benchmark: 747 seconds

# Run polydisperse_cufjc_networks_clnk_rves_bimodal_ntwrk_approx_dfrmtn.py with proper configurations
cd ../configs/polydisperse_cufjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603D
  - deformation: 20260603D
  - _self_
label:
  workdir: polydisperse_cufjc_networks_clnk_rves
  date: !!str 20260603
  batch: D
  sample: 0
EOL
cd ../../polydisperse_cufjc_networks_clnk_rves
python3 polydisperse_cufjc_networks_clnk_rves_bimodal_ntwrk_approx_dfrmtn.py
##### Time benchmark: 3803 seconds

# Run polydisperse_cufjc_networks_clnk_rves_bimodal_ntwrk_approx_dfrmtn.py with proper configurations
cd ../configs/polydisperse_cufjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20260603E
  - deformation: 20260603E
  - _self_
label:
  workdir: polydisperse_cufjc_networks_clnk_rves
  date: !!str 20260603
  batch: E
  sample: 0
EOL
cd ../../polydisperse_cufjc_networks_clnk_rves
python3 polydisperse_cufjc_networks_clnk_rves_bimodal_ntwrk_approx_dfrmtn.py
##### Time benchmark: 2494 seconds

# Run plotting codes
python3 plot_polydisperse_cufjc_networks_4_chn_clnk_rves_agreement.py
python3 plot_polydisperse_cufjc_networks_clnk_rves_bimodal_ntwrk_dfrmtn.py

# Deactivate virtual environment
deactivate

# Total runtime: 18611 seconds ~ 5.2 hours