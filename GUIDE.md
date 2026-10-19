# GDS-ParT setup guide: lab workstation, HPC cluster, Colab

This guide explains how to get the repository running on a new machine, check that the setup is correct, and launch training or evaluation. The same code runs everywhere. Only the paths and the GPU-memory settings change between machines.

---

## 0. Requirements

| | Minimum | Recommended |
|---|---|---|
| GPU | NVIDIA, 6 GB (laptop RTX 4050 works) | 24–80 GB (RTX 4090/A100/H100) |
| RAM | 16 GB | 32 GB+ |
| Disk | ~25 GB (10M train + 1M val + 1M test ROOT files + cache) | NVMe/local scratch for the cache |
| Python | 3.10+ | 3.11 |
| CUDA driver | anything that supports your torch wheel (`nvidia-smi`) | |

Disk usage for each split is about 1.5 GB of ROOT files per 1M jets, plus about 1.9 GB of cache per 1M jets.

---

## 1. Get the code

```bash
git clone https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer gds_part
cd gds_part
```

The repository does not contain checkpoints (`*.pt`) or caches. To continue an existing run on another machine, copy its `checkpoints/<run>/last.pt` and train with `--resume` and the same `--checkpoint_dir`.

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

To change the CUDA wheel, edit the `--index-url` line in `hpc/setup_env.sh`. Run setup on the **login node**, because compute nodes usually have no internet access.

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

Each JetClass ROOT file contains 100k jets of one class. `fetch_jetclass_subset.py` downloads only the files you request from Zenodo, using HTTP range reads, so you do not need to download the full 190 GB tarballs.

```bash
# 10M train (10 files/class), 1M val, 1M test  -> $DATA/{train,val,test}
python data/fetch_jetclass_subset.py --out $DATA --train_files 10 --val_files 1 --test_files 1
# quick experiments: --train_files 1  (1M jets)
```

On HPC, use the scratch filesystem rather than `$HOME`, which has a small quota. Run the download on the login node or a data-transfer node:

```bash
DATA=/scratch/$USER/JetClass bash hpc/setup_env.sh data
```

When a split is first used, the trainer converts it into a ragged on-disk cache under `--cache_dir`. For 10M jets this takes 20–40 min of CPU time, and later runs reuse the cache. Caches are portable between machines because they are plain numpy memmaps, so you can rsync `$DATA/cache` to skip the conversion step.

---

## 4. Validate the pipeline (strongly recommended on a new machine)

This step runs the **official pretrained ParT** on your cached test data, using this repo's preprocessing and metric code. If the result matches the paper, your data, features, and metrics are correct.

```bash
pip install --no-deps weaver-core==0.4.17   # 0.5.x is incompatible with the released weights
curl -L -o ParT_full.pt https://raw.githubusercontent.com/jet-universe/particle_transformer/main/models/ParT_full.pt
python train_gds_part.py --train $DATA/train --val $DATA/val --test $DATA/test \
  --cache_dir $DATA/cache --cache_only --checkpoint_dir ./checkpoints/cache_only
python validate_pipeline.py --ckpt ParT_full.pt --cache $DATA/cache/test_test_all_128 --max_jets 200000
```

Expected output: accuracy ≈ **0.861** and AUC ≈ **0.9877**. On 2026-10-19 the reference laptop measured 0.8612 / 0.9877 on 200k jets. Rejection values at high efficiency are noisy on small test sets.

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
| `--max_pairs_per_step` | upper limit on B·N² for each forward pass. Larger batches are split into micro-batches with gradient accumulation (the effective batch size stays the same) | 2M for 6 GB, 4–8M for 24 GB, 8–16M for 40–80 GB |
| `--gpu_mem_fraction` | upper limit on PyTorch's share of VRAM | Keep 0.8 on **Windows** to avoid a ~4× slowdown from WDDM spilling into shared memory. Use 1 on Linux |
| `--epochs` | passes over the train set | With 10M jets, the curve is still rising at epoch 8 (see `results/`). Use ≥ 20 |
| `--jets_per_class` | subsample train | for quick experiments |

Monitor the run with `tail -f checkpoints/v2_10M/train.log`, or with `python monitor_training.py --log checkpoints/v2_10M/train.log`.

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

Both job scripts pass `--resume`. If a job reaches its walltime limit, resubmit the **same command with the same `RUN`** and training continues from `last.pt` (the optimizer, scheduler, and EMA state are restored as well). To pass any other trainer flag, use `EXTRA="--n_layers 8 --pair_bias 1"`.

The first job builds the cache. If your cluster limits CPU time on GPU nodes, build the cache beforehand in a CPU job with `--cache_only`.

---

## 6. Ablations

```bash
python run_ablations.py --train $DATA/train --val $DATA/val --cache_dir $DATA/cache \
  --jets_per_class 100000 --epochs 3 --out ./ablations
python run_ablations.py ... --only v2_full,no_gate   # run a subset of variants
```

Every variant uses the same data and optimizer, and the script writes `ablations/ABLATIONS.md`. Completed variants are skipped when you run it again. Variants:

| Name | What changes |
|---|---|
| `v1` | original model: shared sigmoid gate, mean pooling, 6 layers |
| `v2_full` | current default |
| `no_gate` | v2 without Q/K gates, which is ParT-style bias-only attention. This is the control for "do the gates help?" |
| `no_pair_bias` | v2 with gates only |
| `shared_gate` | v2 with one uniform gate shared across layers |
| `v1_plus_cls` | v1 + class-attention pooling |

---

## 7. Evaluate and report

When training finishes, the trainer evaluates `best.pt` on val and test and writes `results.json` and `{val,test}_predictions.npz`. Then run:

```bash
python make_report.py --run_dir ./checkpoints/v2_10M   # plots + RESULTS.md with ParT-paper table
```

---

## 8. Troubleshooting

| Symptom | Fix |
|---|---|
| `CUDA out of memory` | lower `--max_pairs_per_step` (training uses this value; evaluation uses 2× it) |
| Very slow on Windows, GPU "shared memory" usage growing | keep `--gpu_mem_fraction 0.8` |
| `no files for <class>` while building the cache | the directory must contain `<Prefix>_NNN.root` files (e.g. `HToBB_120.root`). Check `--train` |
| Changed `--max_particles` / `--jets_per_class` | a new cache directory is created automatically. Old caches can be deleted |
| Validation gives garbage (~10% acc) | wrong weaver-core version. Use 0.4.17 |
| Resume complains about missing keys | the architecture flags must match the run that produced `last.pt` (see `args` in the checkpoint) |
