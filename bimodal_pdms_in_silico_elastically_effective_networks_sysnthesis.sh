#!/bin/bash

# Exit immediately if a command fails
set -e

# Activate virtual environment
source bin/activate

# Run bimodal_pdms_in_silico_elastically_effective_networks_synthesis.py with proper configurations
cd configs/bimodal_pdms_networks
cat > config.yaml <<EOL
defaults:
  - topology: 20260206A
  - deformation: 20260206A
  - _self_
label:
  workdir: bimodal_pdms_networks
  date: !!str 20260206
  batch: A
  sample: 0
EOL
cd ../../bimodal_pdms_networks
python3 bimodal_pdms_in_silico_elastically_effective_networks_synthesis.py

cd ../configs/bimodal_pdms_networks
cat > config.yaml <<EOL
defaults:
  - topology: 20260206B
  - deformation: 20260206B
  - _self_
label:
  workdir: bimodal_pdms_networks
  date: !!str 20260206
  batch: B
  sample: 0
EOL
cd ../../bimodal_pdms_networks
python3 bimodal_pdms_in_silico_elastically_effective_networks_synthesis.py

cd ../configs/bimodal_pdms_networks
cat > config.yaml <<EOL
defaults:
  - topology: 20260206C
  - deformation: 20260206C
  - _self_
label:
  workdir: bimodal_pdms_networks
  date: !!str 20260206
  batch: C
  sample: 0
EOL
cd ../../bimodal_pdms_networks
python3 bimodal_pdms_in_silico_elastically_effective_networks_synthesis.py

cd ../configs/bimodal_pdms_networks
cat > config.yaml <<EOL
defaults:
  - topology: 20260206D
  - deformation: 20260206D
  - _self_
label:
  workdir: bimodal_pdms_networks
  date: !!str 20260206
  batch: D
  sample: 0
EOL
cd ../../bimodal_pdms_networks
python3 bimodal_pdms_in_silico_elastically_effective_networks_synthesis.py

# Deactivate virtual environment
deactivate