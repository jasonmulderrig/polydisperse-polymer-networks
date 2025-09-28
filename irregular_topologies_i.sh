#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source bin/activate

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_agreement.py with proper configurations
cd configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20250724A
  - deformation: 20250724A
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20250724
  batch: A
configs: 1
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_agreement.py

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_approx_agreement.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20250724B
  - deformation: 20250724B
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20250724
  batch: B
configs: 1
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_approx_agreement.py

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_monodisperse_deformation.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20250724C
  - deformation: 20250724C
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20250724
  batch: C
configs: 1
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_monodisperse_deformation.py

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_general_deformation.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20250724D
  - deformation: 20250724D
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20250724
  batch: D
configs: 1
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_general_deformation.py

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_tangent_stiffness_modulus.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20250724E
  - deformation: 20250724E
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20250724
  batch: E
configs: 1
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_tangent_stiffness_modulus.py

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_comparison.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20250724F
  - deformation: 20250724F
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20250724
  batch: F
configs: 1
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_comparison.py

# Run polydisperse_inext_gaussian_fjc_networks_6_chn_clnk_rves_exact_comparison.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20250724G
  - deformation: 20250724G
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20250724
  batch: G
configs: 1
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_6_chn_clnk_rves_exact_comparison.py

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_tangent_stiffness_modulus_model_comparison.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20250724H
  - deformation: 20250724H
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20250724
  batch: H
configs: 1
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_tangent_stiffness_modulus_model_comparison.py

# Run polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_exact_agreement.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20250724A
  - deformation: 20250724A
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20250724
  batch: A
configs: 1
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_exact_agreement.py

# Run polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_approx_agreement.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20250724B
  - deformation: 20250724B
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20250724
  batch: B
configs: 1
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_approx_agreement.py

# Run polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_monodisperse_network_response.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20250724C
  - deformation: 20250724C
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20250724
  batch: C
configs: 1
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_monodisperse_network_response.py

# Run polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_bimodal_network_response.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20250724D
  - deformation: 20250724D
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20250724
  batch: D
configs: 1
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_bimodal_network_response.py
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20250724E
  - deformation: 20250724E
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20250724
  batch: E
configs: 1
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_bimodal_network_response.py

# Run plotting codes
cd ../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_agreement_plotting.py
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_general_deformation_plotting.py
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_tangent_stiffness_modulus_plotting.py
python3 polydisperse_inext_gaussian_fjc_networks_clnk_rves_exact_comparison_plotting.py
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_tangent_stiffness_modulus_model_comparison_plotting.py
cd ../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_agreement_plotting.py
python3 polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_bimodal_network_response_plotting.py

# Deactivate virtual environment
deactivate