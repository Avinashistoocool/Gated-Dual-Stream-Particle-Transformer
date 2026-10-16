#!/bin/bash
# One-time setup on the HPC login node:
#   bash hpc/setup_env.sh                                        # conda env "gds"
#   DATA=/scratch/$USER/JetClass bash hpc/setup_env.sh data      # + download 10M train jets
set -euo pipefail
module load anaconda3 2>/dev/null || module load python 2>/dev/null || true

if ! conda env list 2>/dev/null | grep -q "^gds "; then
  conda create -y -n gds python=3.11
fi
source activate gds 2>/dev/null || conda activate gds
pip install --upgrade pip
# pick the CUDA wheel matching the cluster driver (see `nvidia-smi`)
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install numpy uproot awkward scikit-learn matplotlib

if [ "${1:-}" = "data" ]; then
  DATA=${DATA:-/scratch/$USER/JetClass}
  # 100k jets per file: 10 files/class = 10M train jets; 1 file/class = 1M val / 1M test
  python data/fetch_jetclass_subset.py --out "$DATA" --train_files 10 --val_files 1 --test_files 1
fi
