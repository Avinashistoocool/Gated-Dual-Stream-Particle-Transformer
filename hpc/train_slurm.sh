#!/bin/bash
# SLURM job: train GDS-ParT on JetClass (SLURM counterpart of train_pbs.sh).
#   sbatch hpc/train_slurm.sh
#   sbatch --export=ALL,DATA=/scratch/$USER/JetClass,EPOCHS=20,RUN=./checkpoints/v2_10M hpc/train_slurm.sh
#   squeue -u $USER  |  scancel <jobid>  |  tail -f <RUN>/train.log
#
# Edit partition / account / modules for your cluster (`sinfo`, `module avail`).
#SBATCH -J gds_part
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 48:00:00
#SBATCH -o slurm-%j.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR:-$(pwd)}"

module load cuda 2>/dev/null || true
module load anaconda3 2>/dev/null || module load python 2>/dev/null || true
source activate gds 2>/dev/null || conda activate gds

DATA=${DATA:-/scratch/$USER/JetClass}
TRAIN=${TRAIN:-$DATA/train}
CACHE=${CACHE:-$DATA/cache}
RUN=${RUN:-./checkpoints/v2_hpc}     # keep fixed so resubmits resume
EPOCHS=${EPOCHS:-20}
BS=${BS:-512}
LR=${LR:-1e-3}
MAX_PAIRS=${MAX_PAIRS:-8000000}
EXTRA=${EXTRA:-}

nvidia-smi || true
echo "job ${SLURM_JOB_ID:-local} on $(hostname) -> $RUN"

python train_gds_part.py \
  --train "$TRAIN" --val "$DATA/val" --test "$DATA/test" \
  --cache_dir "$CACHE" --epochs "$EPOCHS" --batch_size "$BS" --lr "$LR" \
  --max_pairs_per_step "$MAX_PAIRS" --gpu_mem_fraction 1 --checkpoint_dir "$RUN" --resume $EXTRA

python make_report.py --run_dir "$RUN"
