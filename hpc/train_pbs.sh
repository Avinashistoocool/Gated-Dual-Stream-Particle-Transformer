#!/bin/bash
# PBS/Torque job: train GDS-ParT v2 on JetClass.
#   qsub hpc/train_pbs.sh
#   qsub -v DATA=/scratch/$USER/JetClass,EPOCHS=20,RUN=./checkpoints/v2_10M hpc/train_pbs.sh
#   qstat -u $USER     |  qdel <jobid>  |  tail -f <RUN>/train.log
#
# Edit the queue / resources / modules for your cluster (`qstat -Q`, `module avail`).
#PBS -N gds_part_v2
#PBS -q gpu
#PBS -l select=1:ncpus=8:ngpus=1:mem=48gb
#PBS -l walltime=48:00:00
#PBS -j oe
# Classic Torque uses this instead of select= (remove one leading # to enable):
##PBS -l nodes=1:ppn=8:gpus=1,mem=48gb

set -euo pipefail
cd "${PBS_O_WORKDIR:-$(pwd)}"

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
MAX_PAIRS=${MAX_PAIRS:-8000000}      # B*N*N per forward pass; ~8M fits 40 GB, raise on 80 GB
EXTRA=${EXTRA:-}

nvidia-smi || true
echo "job ${PBS_JOBID:-local} on $(hostname) -> $RUN"

# --resume continues from $RUN/last.pt, so if walltime kills the job just qsub again
python train_gds_part.py \
  --train "$TRAIN" --val "$DATA/val" --test "$DATA/test" \
  --cache_dir "$CACHE" --epochs "$EPOCHS" --batch_size "$BS" --lr "$LR" \
  --max_pairs_per_step "$MAX_PAIRS" --checkpoint_dir "$RUN" --resume $EXTRA

python make_report.py --run_dir "$RUN"
