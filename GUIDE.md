# GDS-ParT setup guide: lab workstation, HPC cluster, Colab

This guide tells you how to:

1. Set up the repository on a new machine.
2. Make sure that the setup is correct.
3. Start training or evaluation.

The same code operates on all machines. Only the paths and the settings for GPU memory change between machines.

---

## 0. Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA, 6 GB (laptop RTX 4050 works) | 24–80 GB (RTX 4090/A100/H100) |
| RAM | 16 GB | 32 GB+ |
| Disk | ~25 GB (10M train + 1M val + 1M test ROOT files + cache) | NVMe/local scratch for the cache |
| Python | 3.10+ | 3.11 |
| CUDA driver | any driver that supports your torch wheel (`nvidia-smi`) | |

For each split, the disk usage is about 1.5 GB of ROOT files per 1M jets, plus about 1.9 GB of cache per 1M jets.

---

## 1. Get the code

```bash
git clone https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer gds_part
cd gds_part
```

The repository does not contain caches. Usually, it also does not contain checkpoints (`*.pt`). To continue a run on a different machine, do these steps:

1. Copy the `checkpoints/<run>/last.pt` file of the run.
2. Train with `--resume` and the same `--checkpoint_dir`.

**Exception: the repository contains the 20-epoch 10M run, so that you can resume it from a clone.**
`checkpoints/v2_shared_10M_20ep/last.pt` contains the state after epoch 8 of 20 (val acc 0.8473, model + EMA + optimizer + scheduler + history). To continue the run, do these steps:

1. Set `$DATA` to your JetClass folder (`train_10M`, `val`, `test`; refer to section 3).
2. Run this command:

```bash
python train_gds_part.py --train $DATA/train_10M --val $DATA/val --test $DATA/test \
  --cache_dir $DATA/cache --epochs 20 --batch_size 512 --lr 1e-3 \
  --per_layer_gate 0 --ema_decay 0.9999 --log_every 500 \
  --checkpoint_dir ./checkpoints/v2_shared_10M_20ep --resume
```

Do not change `--epochs 20`, `--batch_size 512` or the model flags. If you change them, the LR schedule or the checkpoint will not agree. The first run on a new machine builds the cache in `--cache_dir` again. The completed 10-epoch model (test acc 0.8484, AUC 0.9853) is in `checkpoints/v2_shared_10M_10ep/best.pt`. Its report is in `results/v2_shared_10M_10ep/`.

---

## 2. Python environment

### Lab workstation (Linux or Windows)

```bash
python -m venv .venv
# Linux:   source .venv/bin/activate
# Windows: .venv\Scripts\activate
pip install --upgrade pip
# choose the wheel that matches your driver: cu121 / cu124 / cu128
pip install torch --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

### HPC cluster (conda + modules)

```bash
bash hpc/setup_env.sh          # creates conda env "gds" and installs everything
```

To change the CUDA wheel, edit the `--index-url` line in `hpc/setup_env.sh`. Run the setup on the **login node**. Compute nodes usually do not have internet access.

### Verify

```bash
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
python gds_part_colab.py       # forward/backward/step sanity check -> "SANITY CHECK PASSED"
python train_gds_part.py --synthetic --epochs 1 --synthetic_batches 4 --batch_size 8 \
  --embed_dim 32 --n_heads 4 --n_layers 2 --ffn_dim 64 --pairwise_embed_dim 16 \
  --max_particles 32 --checkpoint_dir ./checkpoints/smoke
```

---

## 3. Get JetClass data

`$DATA` is a folder that you select. It is not a fixed path. Set it before you run the commands. For example, use `export DATA=/data/$USER/JetClass` (Linux) or `$env:DATA="D:\JetClass"` (PowerShell). The trainer looks for this structure in the folder:

```
$DATA/
├── train_10M/   # 100 ROOT files, 10 per class (~14 GB)
├── val/         # 10 ROOT files (~1.4 GB)
├── test/        # 10 ROOT files (~1.4 GB)
└── cache/       # created by the trainer (~25 GB for the splits above)
```

**Option A: copy the data from a machine that has it (fastest).** Copy `train_10M`, `val` and `test` (about 17 GB). You can also copy `cache` (about 25 GB), so that you do not do the 20–40 min conversion. The names of the cache directories include the name of the split folder (for example, `cache/train_train_10M_all_128`). Thus, if you copy the cache, do not change the folder names.

```bash
rsync -avP user@laptop:/path/to/JetClass/ $DATA/          # or copy from an external drive
```

**Option B: download from Zenodo.**
Each JetClass ROOT file contains 100k jets of one class. `fetch_jetclass_subset.py` uses HTTP range reads to download from Zenodo only the files that you ask for. Thus, it is not necessary to download the full 190 GB tarballs.

```bash
# 10M train (10 files/class), 1M val, 1M test  -> $DATA/{train,val,test}
python data/fetch_jetclass_subset.py --out $DATA --train_files 10 --val_files 1 --test_files 1
mv $DATA/train $DATA/train_10M      # match the folder name used by the committed 20-epoch run
# quick experiments: --train_files 1  (1M jets)
```

On HPC, use the scratch filesystem, not `$HOME`. The quota of `$HOME` is small. Run the download on the login node or on a data-transfer node:

```bash
DATA=/scratch/$USER/JetClass bash hpc/setup_env.sh data
```

The first time that the trainer uses a split, it converts the split into a ragged cache on the disk in `--cache_dir`. For 10M jets, this conversion uses 20–40 min of CPU time. Later runs use the same cache again. The caches are plain numpy memmaps, so you can move them between machines. Thus, you can rsync `$DATA/cache` and not do the conversion step.

---

## 4. Validate the pipeline (strongly recommended on a new machine)

This step runs the **official pretrained ParT** on your cached test data. It uses the preprocessing code and the metric code of this repository. If the result agrees with the paper, your data, features and metrics are correct.

```bash
pip install --no-deps weaver-core==0.4.17   # 0.5.x is incompatible with the released weights
curl -L -o ParT_full.pt https://raw.githubusercontent.com/jet-universe/particle_transformer/main/models/ParT_full.pt
python train_gds_part.py --train $DATA/train --val $DATA/val --test $DATA/test \
  --cache_dir $DATA/cache --cache_only --checkpoint_dir ./checkpoints/cache_only
python validate_pipeline.py --ckpt ParT_full.pt --cache $DATA/cache/test_test_all_128 --max_jets 200000
```

Expected output: accuracy ≈ **0.861** and AUC ≈ **0.9877**. On 2026-10-19, the reference laptop measured 0.8612 / 0.9877 on 200k jets. On small test sets, the rejection values at high efficiency have much noise.

---

## 5. Train

### Lab workstation

```bash
python train_gds_part.py \
  --train $DATA/train --val $DATA/val --test $DATA/test --cache_dir $DATA/cache \
  --epochs 20 --batch_size 512 --lr 1e-3 \
  --max_pairs_per_step 8000000 --gpu_mem_fraction 1 \
  --checkpoint_dir ./checkpoints/v2_10M
# add --resume to continue after interruption
```

| Knob | Meaning | Guidance |
|---|---|---|
| `--max_pairs_per_step` | Upper limit of B·N² for each forward pass. The trainer divides larger batches into micro-batches with gradient accumulation (the effective batch size does not change) | 2M for 6 GB, 4–8M for 24 GB, 8–16M for 40–80 GB |
| `--gpu_mem_fraction` | Upper limit of the VRAM that PyTorch can use | Keep 0.8 on **Windows**. A higher value can let WDDM move data into shared memory, which makes training ~4× slower. Use 1 on Linux |
| `--epochs` | Number of passes over the train set | With 10M jets, the curve continues to increase at epoch 8 (refer to `results/`). Use ≥ 20 |
| `--jets_per_class` | Use a subsample of the train set | For quick experiments |

To monitor the run, use `tail -f checkpoints/v2_10M/train.log` or `python monitor_training.py --log checkpoints/v2_10M/train.log`.

### HPC: PBS/Torque

```bash
# first edit the #PBS queue/resources and the module lines in hpc/train_pbs.sh
qsub -v DATA=/scratch/$USER/JetClass,EPOCHS=20,RUN=./checkpoints/v2_10M hpc/train_pbs.sh
qstat -u $USER
```

### HPC: SLURM

```bash
# first edit #SBATCH -p / --gres / modules in hpc/train_slurm.sh
sbatch --export=ALL,DATA=/scratch/$USER/JetClass,EPOCHS=20,RUN=./checkpoints/v2_10M hpc/train_slurm.sh
squeue -u $USER
```

The two job scripts give the `--resume` flag. If a job gets to its walltime limit, submit the **same command with the same `RUN`** again. The training then continues from `last.pt`. The trainer also restores the optimizer, scheduler and EMA state. To give other trainer flags, use `EXTRA="--n_layers 8 --pair_bias 1"`.

The first job builds the cache. Your cluster can limit the CPU time on GPU nodes. In that case, build the cache first in a CPU job with `--cache_only`.

---

## 6. Ablations

```bash
python run_ablations.py --train $DATA/train --val $DATA/val --cache_dir $DATA/cache \
  --jets_per_class 100000 --epochs 3 --out ./ablations
python run_ablations.py ... --only v2_full,no_gate   # run a subset of variants
```

For short runs (fewer than ~20k steps), add `--extra "--ema_decay 0"`. An EMA with decay 0.999 is late to follow the weights, and this makes the comparison incorrect. The summaries of the 2026-10-19 runs (with EMA and without EMA) are in `results/ablation_summaries/`.

All variants use the same data and the same optimizer. The script writes `ablations/ABLATIONS.md`. When you run the script again, it does not run the completed variants again. The variants are:

| Name | What changes |
|---|---|
| `v1` | original model: shared sigmoid gate, mean pooling, 6 layers |
| `v2_full` | current default |
| `no_gate` | v2 without Q/K gates, which is attention with only a bias, as in ParT. This is the control for the question "do the gates help?" |
| `no_pair_bias` | v2 with gates only |
| `shared_gate` | v2 with one uniform gate that all layers share |
| `v1_plus_cls` | v1 + class-attention pooling |

---

## 7. Evaluate and report

When the training is complete, the trainer evaluates `best.pt` on val and test. It writes `results.json` and `{val,test}_predictions.npz`. Then run this command:

```bash
python make_report.py --run_dir ./checkpoints/v2_10M   # plots + RESULTS.md with ParT-paper table
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | Decrease `--max_pairs_per_step`. Training uses this value, and evaluation uses 2× this value |
| Training is slow on Windows, and the GPU "shared memory" usage increases | Keep `--gpu_mem_fraction 0.8` |
| `no files for <class>` when the trainer builds the cache | The directory must contain `<Prefix>_NNN.root` files (for example, `HToBB_120.root`). Examine `--train` |
| You changed `--max_particles` / `--jets_per_class` | The trainer makes a new cache directory automatically. You can delete the old caches |
| Validation gives incorrect results (~10% acc) | The weaver-core version is incorrect. Use 0.4.17 |
| Resume shows an error about missing keys | The architecture flags must be the same as in the run that made `last.pt` (refer to `args` in the checkpoint) |
