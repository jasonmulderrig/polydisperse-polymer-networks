#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source bin/activate

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_agreement.py with proper configurations
cd configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204A
  - deformation: 20251204A
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: A
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_agreement.py

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_approx_agreement.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204B
  - deformation: 20251204B
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: B
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_approx_agreement.py

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_monodisperse_deformation.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204C
  - deformation: 20251204C
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: C
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_monodisperse_deformation.py

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_general_deformation.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204D
  - deformation: 20251204D
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: D
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_general_deformation.py

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_tangent_stiffness_modulus.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204E
  - deformation: 20251204E
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: E
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_tangent_stiffness_modulus.py

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_comparison.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204F
  - deformation: 20251204F
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: F
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_comparison.py

# Run polydisperse_inext_gaussian_fjc_networks_6_chn_clnk_rves_exact_comparison.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204G
  - deformation: 20251204G
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: G
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_6_chn_clnk_rves_exact_comparison.py

# Run polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_tangent_stiffness_modulus_model_comparison.py with proper configurations
cd ../configs/polydisperse_inext_gaussian_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204H
  - deformation: 20251204H
  - _self_
label:
  workdir: polydisperse_inext_gaussian_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: H
  sample: 0
EOL
cd ../../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_tangent_stiffness_modulus_model_comparison.py

# Run polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_exact_agreement.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204A
  - deformation: 20251204A
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: A
  sample: 0
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_exact_agreement.py

# Run polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_approx_agreement.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204B
  - deformation: 20251204B
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: B
  sample: 0
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_approx_agreement.py

# Run polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_monodisperse_network_response.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204C
  - deformation: 20251204C
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: C
  sample: 0
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_monodisperse_network_response.py

# Run polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_bimodal_network_response.py with proper configurations
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204D
  - deformation: 20251204D
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: D
  sample: 0
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_bimodal_network_response.py
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204E
  - deformation: 20251204E
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: E
  sample: 0
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_bimodal_network_response.py
cd ../configs/polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
cat > config.yaml <<EOL
defaults:
  - topology: 20251204F
  - deformation: 20251204F
  - _self_
label:
  workdir: polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
  date: !!str 20251204
  batch: F
  sample: 0
EOL
cd ../../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_bimodal_network_response.py

# Run plotting codes
cd ../polydisperse_inext_gaussian_fjc_networks_clnk_rves
python3 plot_polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_agreement.py
python3 plot_polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_general_deformation.py
python3 plot_polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_tangent_stiffness_modulus.py
python3 plot_polydisperse_inext_gaussian_fjc_networks_clnk_rves_exact_comparison.py
python3 plot_polydisperse_inext_gaussian_fjc_networks_4_chn_clnk_rves_exact_tangent_stiffness_modulus_model_comparison.py
cd ../polydisperse_inext_kuhn_grun_fjc_networks_clnk_rves
python3 plot_polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_agreement.py
python3 plot_polydisperse_inext_kuhn_grun_fjc_networks_4_chn_clnk_rves_bimodal_network_response.py

# Deactivate virtual environment
deactivate