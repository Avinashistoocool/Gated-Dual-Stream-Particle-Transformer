# Gated-Dual-Stream-Particle-Transformer

A PyTorch model for JetClass jet classification that conditions attention on pairwise
particle kinematics. GDS-ParT uses relation-derived gates on query and key channels;
the reported v2 configuration also retains a ParT-style additive pair bias.
The [Particle Transformer (ParT)][1] and [JetClass dataset][3] provide the reference.

## Overview

ParT introduces pairwise kinematic information as an additive bias inside attention:

$$
\text{Attention} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + \psi(U)\right)
$$

where $$U_{ij}$$ denotes pairwise particle features and $$\psi(U)$$ is a learned continuous function.[1]

GDS-ParT embeds pairwise features into a relation tensor, aggregates them into
per-particle gates and scales query/key channels before the dot product.
The reported v2 model combines this conditioning with the additive pair bias.

## Status

| Run | Train jets | Params | Test accuracy | Test AUC | Notes |
|---|---|---|---|---|---|
| v2, 10 epochs | 10M | 2.07M | **0.8484** | **0.9853** | [full report](results/v2_shared_10M_10ep/RESULTS.md) |
| v2, 20 epochs | 10M | 2.07M | in progress | | epoch 8 of 20 saved; val acc 0.8473. Resumable from `checkpoints/v2_shared_10M_20ep/last.pt` ([GUIDE.md](GUIDE.md) §1) |

For context, the [ParT paper][1] reports 0.850 accuracy and 0.9860 AUC
for ParT trained on 10M jets and 0.837/0.9837 for [ParticleNet][4].
These are not controlled comparisons: the published models use a different
optimiser and training budget and the full 20M-jet test set, versus the
one-million-jet subset used here.

To set up a new machine (lab workstation, HPC, Colab) and find out where the JetClass data must go, see **[GUIDE.md](GUIDE.md)**.

**Visual explainer:** [`docs/gds_vs_part.html`](docs/gds_vs_part.html) is an interactive page that builds both architectures from tensors upward. It covers the full math of ParT's additive bias and of GDS gating, and includes a live toy model that compares the attention maps.

**Technical report:** [`docs/GDS-ParT_Technical_Report.pdf`](docs/GDS-ParT_Technical_Report.pdf)
(source: [`docs/GDS-ParT_Technical_Report.tex`](docs/GDS-ParT_Technical_Report.tex))
specifies the inputs, architecture, gate-path gradients, training recipe, evaluation,
results, ablations and limitations. It distinguishes exploratory gate diagnostics from
unproven accuracy gains. See [Documentation](#documentation).

**Follow-up project:** [GDS-ParT-MoE](https://github.com/Avinashistoocool/GDS-parT-MoE) replaces the particle FFNs with a per-particle Mixture-of-Experts and measures what physics each expert learns.

## Why this is different

This implementation investigates relation-conditioned Q/K gating, rather than
reproducing the [official ParT implementation][2].

The relation summary changes how a particle forms queries and keys for
attention; it does not change the original particle inputs or directly gate values.

## Physics intuition

The pairwise stream uses angular separation, momentum sharing, relative
transverse momentum and pair invariant mass, as specified in the technical report.

These observables condition the query/key gates. Their physical interpretation
requires controlled analyses; correlations of average gates with kinematics
alone do not establish which particles receive attention.

## Architecture

### Inputs

The model works with three core tensors:

- Particle features: `(B, N, 17)`
- Pairwise interaction features: `(B, N, N, 4)`
- Labels: `(B,)` for 10 jet classes

The relation embedding has shape `(B, N, N, 64)` in the reported model,
while the particle embedding has shape `(B, N, 128)`.

### Modules

1. **Particle embedding**  
   Projects the 17 raw single-particle features into an `embed_dim` latent space.

2. **Pairwise embedding**  
   Projects the 4 pairwise physics features into a relational tensor of shape `(B, N, N, D_p)`.

3. **Relational gate**  
   Aggregates pair relations and produces separate Q/K gates: sigmoid gates in v1,
   identity-initialised `1 + tanh` gates in v2.

4. **Gated dual-stream attention**  
   Applies separate gates to query and key channels before attention is computed.

5. **Pooling + classifier**  
   Uses masked-mean pooling in v1 or two class-attention blocks in the reported v2 model.

## Architecture contrast

| Aspect | ParT | GDS-ParT |
|---|---|---|
| Pairwise information enters | Additive bias in attention logits `[QK^T / sqrt(d) + ψ(U)]` [1] | Q/K gating and, in reported v2, additive pair bias |
| Fusion space | Logit space | Q/K feature space; also logit space in reported v2 |
| Effect of pairwise physics | Changes attention scores directly [1] | Changes how particles form queries and keys |
| Project aim | Published benchmark architecture [1] | Relation-gating research prototype |

## Installation

Install the repository dependencies:

```bash
pip install -r requirements.txt
```

## Getting the dataset

JetClass is available from the [dataset record][3]. The
[official ParT repository][2] provides a download script:

```bash
git clone https://github.com/jet-universe/particle_transformer
cd particle_transformer
python get_datasets.py JetClass -d ./data
```

The [setup guide](GUIDE.md) describes the expected paths and cache layout.
For a synthetic forward/backward smoke check, run `python gds_part_colab.py`.

## GDS-ParT v2 (current default)

v1 uses shared sigmoid gates, no pair bias and masked-mean pooling (test
accuracy 0.8272). The reported v2 run uses shared identity-initialised gates,
pair bias and two class-attention blocks; the code's v2 defaults instead use
per-layer conditioned gates:

| Change | Why |
|---|---|
| Gates `1 + tanh(·)`, zero-initialized | start as the identity; can amplify (≤2×) or suppress channels instead of only shrinking Q/K |
| Optional per-layer gates with attention-weighted neighbour aggregation `α_ij = softmax_j(rel_ij · W h_i)` | condition each layer's summary on its current particle state; not used in the reported 10M run |
| Hybrid fusion: per-head pairwise logit bias, in addition to gates | restores pair-specific (i, j) information that a per-particle gate cannot represent |
| 2 class-attention blocks (CLS token) | ParT-style learned pooling instead of mean pooling |
| Label smoothing 0.1, EMA of weights (default 0.999), 8 layers | reported shared-gate run has 2.07M parameters and used EMA decay 0.9999 |

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

## Suggested next steps

- Finish the 20-epoch 10M run (resume from epoch 8).
- Visualize learned gates for interpretability.
- Compare gated fusion against a small ParT-like baseline.
- Test whether gate asymmetry for queries vs. keys improves performance.
- Per-particle expert interpretability is being developed in [GDS-ParT-MoE](https://github.com/Avinashistoocool/GDS-parT-MoE).

## Documentation

Browse everything at **[avinashistoocool.github.io/Gated-Dual-Stream-Particle-Transformer](https://avinashistoocool.github.io/Gated-Dual-Stream-Particle-Transformer/)**.

- [**Technical report (PDF)**](docs/GDS-ParT_Technical_Report.pdf): motivation, architecture diagram, the input features, embeddings, ParT-style bias and GDS gating, gate gradients, the full forward pass, the training recipe, evaluation metrics, results, ablations, limitations and outlook. The LaTeX source is [`docs/GDS-ParT_Technical_Report.tex`](docs/GDS-ParT_Technical_Report.tex). Compile it with pdfLaTeX, running it twice.
- [**GDS vs ParT, interactive**](https://avinashistoocool.github.io/Gated-Dual-Stream-Particle-Transformer/gds_vs_part.html): the visual explainer and question list from [`docs/gds_vs_part.html`](docs/gds_vs_part.html).
- [**Setup and training guide**](GUIDE.md): environment, data, validation, training, ablations and troubleshooting.

## Citing

If GDS-ParT helps your research, cite it with GitHub's **Cite this repository** button
(from [`CITATION.cff`](CITATION.cff)). Please also cite the [ParT paper][1] and [JetClass dataset][3].

## Reference sources

- [ParT paper][1]: H. Qu, C. Li and S. Qian, *Particle Transformer for Jet Tagging*, ICML 2022.
- [Official ParT implementation][2]: jet-universe/particle_transformer.
- [JetClass dataset][3]: DOI [10.5281/zenodo.6619768][3].
- [ParticleNet paper][4]: DOI [10.1103/PhysRevD.101.056019][4].

[1]: https://arxiv.org/abs/2202.03772
[2]: https://github.com/jet-universe/particle_transformer
[3]: https://doi.org/10.5281/zenodo.6619768
[4]: https://doi.org/10.1103/PhysRevD.101.056019

## Disclaimer

This research prototype explores a different relation-conditioning mechanism
from ParT; it is not an official ParT reproduction. The single-seed
ablation does not establish a reproducible accuracy gain from gating.

## License

MIT. See [`LICENSE`](LICENSE). The JetClass dataset and the ParT reference code it is compared against keep their own licenses.
