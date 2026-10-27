# Gated-Dual-Stream-Particle-Transformer

A custom PyTorch architecture for jet classification that rethinks how pairwise particle physics enters a transformer. Instead of following ParT exactly, GDS-ParT uses a **Gated Dual-Stream** design in which pairwise interaction features modulate particle representations before attention is computed. The baseline inspiration is Particle Transformer for Jet Tagging by Qu et al. and the JetClass benchmark released alongside it.[1][2]

## Overview

ParT introduces pairwise kinematic information as an additive bias inside attention:

$$
\text{Attention} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + \psi(U)\right)
$$

where $$U_{ij}$$ denotes pairwise particle features and $$\psi(U)$$ is a learned continuous function.[1]

GDS-ParT takes a different route. It embeds pairwise features into a relational tensor, aggregates those relations into per-particle gates, and uses the resulting gates to scale query and key channels before the dot product. This moves physics fusion from additive logit biasing into multiplicative feature-space conditioning.[1][2]

## Status

| Run | Train jets | Params | Test accuracy | Test AUC | Notes |
|---|---|---|---|---|---|
| v2, 10 epochs | 10M | 2.07M | **0.8484** | **0.9853** | [full report](results/v2_shared_10M_10ep/RESULTS.md) |
| v2, 20 epochs | 10M | 2.07M | in progress | | epoch 8 of 20 saved; val acc 0.8473. Resumable from `checkpoints/v2_shared_10M_20ep/last.pt` ([GUIDE.md](GUIDE.md) §1) |

For comparison, ParticleNet trained on 10M jets reaches 0.837 accuracy and 0.9837 AUC in the ParT paper.

To set up a new machine (lab workstation, HPC, Colab) and find out where the JetClass data must go, see **[GUIDE.md](GUIDE.md)**.

**Visual explainer:** [`docs/gds_vs_part.html`](docs/gds_vs_part.html) is an interactive page that builds both architectures from tensors upward. It covers the full math of ParT's additive bias and of GDS gating, and includes a live toy model that compares the attention maps.

**Follow-up project:** [GDS-ParT-MoE](https://github.com/Avinashistoocool/GDS-parT-MoE) replaces the particle FFNs with a per-particle Mixture-of-Experts and measures what physics each expert learns.

## Why this is different

The main goal of this project is to build an original “proof of work” model rather than reproduce the official implementation. The official repository provides the reference ParT codebase and JetClass dataset tooling, but this project intentionally explores a different inductive bias for combining particle-level and pairwise information.[2]

In GDS-ParT, the model asks: **how should a particle represent itself, given its full relational environment inside the jet?** Pairwise physics therefore changes the geometry of the token embeddings used to compute attention, rather than only shifting the final attention matrix.[1]

## Physics intuition

The pairwise stream uses physically meaningful observables such as angular separation and relative kinematics, matching the general ParT setup where interaction information is provided separately from single-particle features.[1][3]

In this architecture, those pairwise observables act like a context-dependent feature selector. If a particle sits in a relational neighborhood suggestive of substructure, collimation, or strong correlated activity, the gate can amplify some query/key channels and suppress others before attention is formed. That makes the pairwise physics structurally active in the representation itself, not merely an auxiliary correction term.[1][3]

## Architecture

### Inputs

The model works with three core tensors:

- Particle features: `(B, N, 17)`
- Pairwise interaction features: `(B, N, N, 4)`
- Labels: `(B,)` for 10 jet classes

These shapes follow the project design goals for a self-contained PyTorch implementation and are compatible with synthetic testing before loading real JetClass data.[1][2]

### Modules

1. **Particle embedding**  
   Projects the 17 raw single-particle features into an `embed_dim` latent space.

2. **Pairwise embedding**  
   Projects the 4 pairwise physics features into a relational tensor of shape `(B, N, N, D)`.

3. **Relational gate**  
   Aggregates the pairwise relational tensor across neighbors and converts the summary into element-wise gates in `(0, 1)`.

4. **Gated dual-stream attention**  
   Applies separate gates to query and key channels before attention is computed.

5. **Pooling + classifier**  
   Pools over particle tokens and predicts one of 10 jet categories.

## Architecture contrast

| Aspect | ParT | GDS-ParT |
|---|---|---|
| Pairwise information enters | Additive bias in attention logits `[QK^T / sqrt(d) + ψ(U)]` [1] | Multiplicative gating on particle representations before attention |
| Fusion space | Logit space | Feature space |
| Effect of pairwise physics | Changes attention scores directly [1] | Changes how particles form queries and keys |
| Project aim | Official benchmark architecture [2] | Original proof-of-work alternative |

## Repository goals

This repository is meant to support three use cases:

- Understand the structural difference between ParT and a gated relational alternative.
- Train a clean PyTorch prototype on synthetic data first.
- Scale to the JetClass dataset once the pipeline is verified.

The official ParT repository includes JetClass data tooling and pretrained-model infrastructure, which makes it the natural reference point for dataset preparation and benchmarking.[2]

## Expected file structure

```text
.
├── README.md
├── context.md
├── gds_part_colab.py            # single-file notebook-friendly implementation
├── train_gds_part.py            # optional standalone trainer
├── checkpoints/
└── data/
    └── JetClass/
```

A notebook-friendly single-file implementation is recommended because multi-file imports often create friction in Colab and Jupyter workflows.

## Installation

Install the core dependencies first:

```bash
pip install torch numpy uproot awkward
```

These packages are sufficient for PyTorch training plus reading ROOT-based JetClass files in a lightweight custom pipeline. The official Particle Transformer repository also distributes dataset tooling through its own setup and scripts.[2]

## Getting the dataset

The official Particle Transformer repository contains the JetClass dataset tooling and is the canonical source referenced by the ParT paper.[1][2]

```bash
git clone https://github.com/jet-universe/particle_transformer
cd particle_transformer
python get_datasets.py JetClass -d ./data
```

This prepares the standard JetClass directory structure used by the reference implementation.[2]

## Important data-format note

One practical issue when adapting JetClass data is tensor axis ordering. The official ecosystem commonly uses feature-first particle tensors, while a clean PyTorch transformer implementation is often easier to write in particle-major format, so a transpose step may be required before the forward pass.[2]

Typical conversion:

```python
# official-style shape: (num_jets, 17, max_particles)
x = x.transpose(1, 2)  # -> (num_jets, max_particles, 17)
```

## Running a synthetic test

A synthetic data generator is useful before touching the real dataset. It should emit:

- particle features,
- pairwise features,
- labels,
- optional masks for padded particles.

A minimal training sanity check should include one forward pass, one cross-entropy loss computation, one backward pass, and one optimizer step. This confirms the architecture is differentiable and free of shape mismatches.

## GDS-ParT v2 (current default)

v1 averaged the pairwise embedding uniformly over all neighbours into one sigmoid gate, shared by every layer, then used mean pooling (test acc 0.827 at 10M jets). v2 keeps multiplicative Q/K gating as the core idea and fixes its bottlenecks:

| Change | Why |
|---|---|
| Gates `1 + tanh(·)`, zero-initialized | start as the identity; can amplify (≤2×) or suppress channels instead of only shrinking Q/K |
| Per-layer gates with attention-weighted neighbour aggregation `α_ij = softmax_j(rel_ij · W h_i)` | each particle chooses which relations matter, conditioned on its current state at every depth |
| Hybrid fusion: per-head pairwise logit bias, in addition to gates | restores pair-specific (i, j) information that a per-particle gate cannot represent |
| 2 class-attention blocks (CLS token) | ParT-style learned pooling instead of mean pooling |
| Label smoothing 0.1, EMA of weights (0.999), 8 layers (~2.5M params) | regularization and stable eval weights at a size comparable to ParT |

All switches are CLI flags. `--gate_mode sigmoid --per_layer_gate 0 --pair_bias 0 --n_cls_layers 0 --n_layers 6 --label_smoothing 0 --ema_decay 0` reproduces v1. Use them for ablations, for example to measure how much accuracy comes from the gates and how much from the bias.

## Running on an HPC cluster (PBS/Torque)

```bash
git clone https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer && cd Gated-Dual-Stream-Particle-Transformer
DATA=/scratch/$USER/JetClass bash hpc/setup_env.sh data   # conda env + 10M/1M/1M jets
qsub -v DATA=/scratch/$USER/JetClass,EPOCHS=20,RUN=./checkpoints/v2_10M hpc/train_pbs.sh
qstat -u $USER                                            # monitor; logs in $RUN/train.log
```

First, edit the `#PBS` queue, resource, and `module load` lines in `hpc/train_pbs.sh` to match your cluster. The job always passes `--resume`, so if the walltime limit stops it, resubmit the same command with the same `RUN` and training continues from `last.pt`. On 40–80 GB GPUs, raise `MAX_PAIRS` (for example to 8–16M) so that fewer micro-batches are needed.

## Training on JetClass (efficient pipeline)

The trainer no longer stores dense `(N, N, 4)` pairwise tensors (≈256 KB/jet). It caches only
per-particle arrays on disk, ragged, about 1.9 KB/jet, and computes the 4 pairwise features on the GPU
for each batch. Batches are bucketed by particle multiplicity, and very large-N batches are split
into micro-batches with gradient accumulation. With these changes, a 6 GB laptop GPU and 16 GB of RAM
can train on 10M+ jets.

```bash
# 1. data: either extract a full train_100M_partX.tar (10M jets) or fetch a per-class subset
python data/fetch_jetclass_subset.py --out D:/JetClass --train_files 1 --val_files 1 --test_files 1

# 2. train (the first run builds the cache under --cache_dir; later runs reuse it)
python train_gds_part.py --train D:/JetClass/train_10M --val D:/JetClass/val --test D:/JetClass/test \
  --cache_dir D:/JetClass/cache --epochs 8 --batch_size 512 --lr 1e-3 \
  --checkpoint_dir ./checkpoints/jetclass_10M          # add --resume to continue

# 3. plots + markdown tables, including a comparison with the ParT paper
python make_report.py --run_dir ./checkpoints/jetclass_10M
```

Outputs: `best.pt`, `last.pt`, `results.json`, `{val,test}_predictions.npz` and `report/`.

Metrics follow the ParT paper (Sec. 2): 10-class accuracy, macro one-vs-one AUC, and the
background rejection `Rej_X% = 1/FPR at TPR = X%` for each signal class versus QCD. The rejection
uses the two-class score `S/(S+B)` and is measured at 50% efficiency, except H→ℓνqq′ (99%) and
t→bℓν (99.5%).

## Legacy training example

Once the synthetic pipeline works, train on JetClass using train/validation/test splits. The JetClass benchmark introduced with ParT is a large-scale jet-tagging dataset with 100 million jets, designed specifically for this problem setting.[1]

Recommended training ingredients:

- `nn.CrossEntropyLoss` for multi-class classification
- `AdamW` optimizer
- learning-rate warmup plus cosine decay
- gradient clipping
- checkpointing of best and last model

Example launch pattern:

```bash
python train_gds_part.py \
  --train ./particle_transformer/data/JetClass/train_100M \
  --val   ./particle_transformer/data/JetClass/val_5M \
  --test  ./particle_transformer/data/JetClass/test_20M \
  --epochs 10 \
  --batch_size 512
```

## Accuracy and evaluation

For multi-class jet tagging, the simplest metric is top-1 accuracy:

$$
\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}
$$

In code, this is usually computed with `argmax` over the class logits and comparison against the true labels.

Useful evaluation outputs:

- training accuracy per epoch
- validation accuracy per epoch
- final test accuracy
- per-class accuracy
- confusion matrix

These metrics make it easier to compare GDS-ParT against ParT-style baselines and inspect which jet categories benefit most from relational gating.[1][2]

## Interpreting the gated mechanism

A good way to explain the model is:

> ParT injects pairwise kinematics into the attention matrix as an additive bias. GDS-ParT instead converts pairwise kinematics into per-particle gates that reshape the query and key channels before attention. So pairwise physics changes the representation geometry itself, not only the final attention scores.

That statement captures the central architectural novelty relative to the original ParT formulation.[1]

## Suggested next steps

- Finish the 20-epoch 10M run (resume from epoch 8).
- Visualize learned gates for interpretability.
- Compare gated fusion against a small ParT-like baseline.
- Test whether gate asymmetry for queries vs. keys improves performance.
- Per-particle expert interpretability is being developed in [GDS-ParT-MoE](https://github.com/Avinashistoocool/GDS-parT-MoE).

## Reference sources

- ParT paper: [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772) [1]
- Official repository: [jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer) [2]

## Disclaimer

This project is intentionally inspired by ParT while exploring a different architectural mechanism. It should be treated as an original research prototype built for understanding, experimentation, and presentation rather than as an official reproduction of the published model.[1][2]
