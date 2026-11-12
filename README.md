# Gated-Dual-Stream-Particle-Transformer

GDS-ParT is a custom PyTorch architecture for jet classification. It uses a different method to put pairwise particle physics into a transformer. GDS-ParT does not follow ParT exactly. It uses a **Gated Dual-Stream** design: the pairwise interaction features change the particle representations before the model calculates attention. The baseline is Particle Transformer for Jet Tagging by Qu et al., and the JetClass benchmark that the authors released with it.[1][2]

## Overview

ParT adds pairwise kinematic information to attention as an additive bias:

$$
\text{Attention} = \text{Softmax}\left(\frac{QK^T}{\sqrt{d}} + \psi(U)\right)
$$

In this equation, $$U_{ij}$$ are the pairwise particle features, and $$\psi(U)$$ is a learned continuous function.[1]

GDS-ParT uses a different method:

1. It embeds the pairwise features into a relational tensor.
2. It aggregates these relations into gates for each particle.
3. It uses these gates to scale the query and key channels before the dot product.

Thus, the physics fusion moves from an additive bias on the logits to a multiplicative conditioning in feature space.[1][2]

## Status

| Run | Train jets | Params | Test accuracy | Test AUC | Notes |
|---|---|---|---|---|---|
| v2, 10 epochs | 10M | 2.07M | **0.8484** | **0.9853** | [full report](results/v2_shared_10M_10ep/RESULTS.md) |
| v2, 20 epochs | 10M | 2.07M | in progress | | Epoch 8 of 20 is saved; val acc 0.8473. You can resume it from `checkpoints/v2_shared_10M_20ep/last.pt` ([GUIDE.md](GUIDE.md) §1) |

For comparison, the ParT paper gives 0.837 accuracy and 0.9837 AUC for ParticleNet trained on 10M jets.

For the setup of a new machine (lab workstation, HPC or Colab) and for the location of the JetClass data, refer to **[GUIDE.md](GUIDE.md)**.

**Visual explainer:** [`docs/gds_vs_part.html`](docs/gds_vs_part.html) is an interactive page that builds the two architectures from tensors upward. It shows all the mathematics of the ParT additive bias and of the GDS gates. It also has a live toy model that compares the attention maps.

**Old slide-video prototype (replaced):** [`docs/learning_video/README.md`](docs/learning_video/README.md) keeps the earlier Hinglish production as a record of its history. The user rejected it as the final teaching course, and it is not the replacement deliverable.

**Manim mathematics and reconstruction course:** [`docs/manim_course/README.md`](docs/manim_course/README.md) describes the approved replacement. The course has these parts:

- programmatic mathematical animation
- English neural narration
- timing from sentence bookmarks
- executed code and solved exercises.

Chapters 0-1 are kept without changes. The authorized continuation covers Chapters 2-15. It has an independent companion implementation, an executed notebook, exact audits of the parameters and behavioural tests. The [quality checklist](docs/manim_course/QUALITY_CHECKLIST.md) records the status of the rendering and the upload separately. Source files alone are not proof of delivery.

**Technical report:** [`docs/GDS-ParT_Technical_Report.pdf`](docs/GDS-ParT_Technical_Report.pdf) (source: [`docs/GDS-ParT_Technical_Report.tex`](docs/GDS-ParT_Technical_Report.tex)) covers these topics:

- the architecture
- each equation, as the code calculates it
- training, metrics and results
- ablations and limitations.

The report explains the two streams and derives the gradients of the gates. It shows the difference between shared gates and per-layer gates. It also keeps the measured dependence on the gates separate from accuracy gains, which are not proved. The appendices build the mathematics from basic algebra. They also cover the MoE extension, with worked examples and exercises. Refer to [Documentation](#documentation).

**Follow-up project:** [GDS-ParT-MoE](https://github.com/Avinashistoocool/GDS-parT-MoE) replaces the particle FFNs with a Mixture-of-Experts for each particle. It measures which physics each expert learns.

## Why this is different

The main goal of this project is to build an original "proof of work" model. It is not a copy of the official implementation. The official repository supplies the reference ParT code and the tools for the JetClass dataset. This project examines a different inductive bias to combine the particle-level information and the pairwise information.[2]

GDS-ParT asks this question: **how must a particle show itself, when it knows its full relational environment in the jet?**
Thus, the pairwise physics changes the geometry of the token embeddings that the model uses to calculate attention. It does not only shift the final attention matrix.[1]

## Physics intuition

The pairwise stream uses observables with a physical meaning, for example the angular separation and the relative kinematics. This agrees with the general ParT setup, which supplies the interaction information separately from the features of single particles.[1][3]

In this architecture, these pairwise observables operate as a feature selector that changes with the context. A particle can have a relational neighbourhood that shows substructure, collimation or strong correlated activity. In that case, the gate can increase some query/key channels and decrease other channels before the model calculates attention. Thus, the pairwise physics has an active part in the representation. It is not only an auxiliary correction term.[1][3]

## Architecture

### Inputs

The model uses three core tensors:

- Particle features: `(B, N, 17)`
- Pairwise interaction features: `(B, N, N, 4)`
- Labels: `(B,)` for 10 jet classes

These shapes follow the design goals of the project for a self-contained PyTorch implementation. You can use them for synthetic tests before you load real JetClass data.[1][2]

### Modules

1. **Particle embedding**  
   Projects the 17 raw features of each particle into an `embed_dim` latent space.

2. **Pairwise embedding**  
   Projects the 4 pairwise physics features into a relational tensor of shape `(B, N, N, D)`.

3. **Relational gate**  
   Aggregates the pairwise relational tensor across the neighbours. Then it converts the summary into element-wise gates in `(0, 1)`.

4. **Gated dual-stream attention**  
   Applies separate gates to the query channels and the key channels before the model calculates attention.

5. **Pooling + classifier**  
   Pools the particle tokens and predicts one of 10 jet categories.

## Architecture contrast

| Aspect | ParT | GDS-ParT |
|---|---|---|
| Location of the pairwise information | Additive bias in attention logits `[QK^T / sqrt(d) + ψ(U)]` [1] | Multiplicative gates on the particle representations before attention |
| Fusion space | Logit space | Feature space |
| Effect of the pairwise physics | Changes the attention scores directly [1] | Changes how particles make queries and keys |
| Project aim | Official benchmark architecture [2] | Original proof-of-work alternative |

## Repository goals

This repository has three use cases:

- Understand the structural difference between ParT and a gated relational alternative.
- Train a clean PyTorch prototype on synthetic data first.
- Scale to the JetClass dataset after you verify the pipeline.

The official ParT repository has tools for the JetClass data and an infrastructure for pretrained models. Thus, it is the natural reference for dataset preparation and benchmarks.[2]

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

We recommend a single-file implementation for notebooks. Imports from many files frequently cause problems in Colab and Jupyter.

## Installation

Install the core dependencies first:

```bash
pip install torch numpy uproot awkward
```

These packages are sufficient for PyTorch training and for a small custom pipeline that reads the ROOT-based JetClass files. The official Particle Transformer repository also supplies dataset tools through its own setup and scripts.[2]

## Getting the dataset

The official Particle Transformer repository has the tools for the JetClass dataset. The ParT paper refers to it as the canonical source.[1][2]

```bash
git clone https://github.com/jet-universe/particle_transformer
cd particle_transformer
python get_datasets.py JetClass -d ./data
```

These commands prepare the standard JetClass directory structure that the reference implementation uses.[2]

## Important data-format note

A practical problem when you adapt JetClass data is the order of the tensor axes. The official tools usually put the features first in the particle tensors. A clean PyTorch transformer is frequently easier to write with the particles first. Thus, a transpose step before the forward pass can be necessary.[2]

Typical conversion:

```python
# official-style shape: (num_jets, 17, max_particles)
x = x.transpose(1, 2)  # -> (num_jets, max_particles, 17)
```

## Running a synthetic test

Use a synthetic data generator before you use the real dataset. The generator must supply these items:

- particle features,
- pairwise features,
- labels,
- optional masks for padded particles.

A minimum check of the training must include these steps:

1. One forward pass.
2. One calculation of the cross-entropy loss.
3. One backward pass.
4. One optimizer step.

This check shows that the architecture is differentiable and that the shapes agree.

## GDS-ParT v2 (current default)

v1 calculated the uniform mean of the pairwise embedding over all neighbours. It used this mean to make one sigmoid gate, which every layer shared. Then it used mean pooling (test acc 0.827 at 10M jets). v2 keeps multiplicative Q/K gates as the core idea and removes the bottlenecks of v1:

| Change | Why |
|---|---|
| Gates `1 + tanh(·)`, with zero initialization | The gates start as the identity. They can increase (≤2×) or decrease channels, and are not limited to a decrease of Q/K |
| Per-layer gates with attention-weighted aggregation of the neighbours `α_ij = softmax_j(rel_ij · W h_i)` | Each particle selects the relations that are important, from its current state at each depth |
| Hybrid fusion: a pairwise logit bias for each head, in addition to the gates | Gives back the pair-specific (i, j) information that a gate for each particle cannot show |
| 2 class-attention blocks (CLS token) | Learned pooling as in ParT, not mean pooling |
| Label smoothing 0.1, EMA of the weights (0.999), 8 layers (~2.5M params) | Regularization and stable evaluation weights, at a size similar to ParT |

All the switches are CLI flags. `--gate_mode sigmoid --per_layer_gate 0 --pair_bias 0 --n_cls_layers 0 --n_layers 6 --label_smoothing 0 --ema_decay 0` gives v1 again. Use these flags for ablations. For example, measure how much accuracy comes from the gates and how much comes from the bias.

## Running on an HPC cluster (PBS/Torque)

```bash
git clone https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer && cd Gated-Dual-Stream-Particle-Transformer
DATA=/scratch/$USER/JetClass bash hpc/setup_env.sh data   # conda env + 10M/1M/1M jets
qsub -v DATA=/scratch/$USER/JetClass,EPOCHS=20,RUN=./checkpoints/v2_10M hpc/train_pbs.sh
qstat -u $USER                                            # monitor; logs in $RUN/train.log
```

Before you start, do these steps:

1. Edit the `#PBS` queue and resource lines in `hpc/train_pbs.sh` for your cluster.
2. Edit the `module load` lines in the same file.

The job always gives the `--resume` flag. If the walltime limit stops the job, submit the same command again with the same `RUN`. The training then continues from `last.pt`. On GPUs with 40–80 GB, increase `MAX_PAIRS` (for example, to 8–16M) to use fewer micro-batches.

## Training on JetClass (efficient pipeline)

The trainer does not keep dense `(N, N, 4)` pairwise tensors (≈256 KB/jet). It keeps only ragged arrays for each particle in a cache on the disk, at about 1.9 KB/jet.
It calculates the 4 pairwise features on the GPU for each batch. The trainer puts jets with a similar number of particles into the same batch.
It divides batches with a large N into micro-batches with gradient accumulation. With these changes, a laptop GPU with 6 GB and 16 GB of RAM
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

The metrics follow the ParT paper (Sec. 2). They are the 10-class accuracy, the macro one-vs-one AUC and the background rejection.
The background rejection is `Rej_X% = 1/FPR at TPR = X%` for each signal class against QCD. The rejection
uses the two-class score `S/(S+B)`. The measurement is at 50% efficiency, but H→ℓνqq′ uses 99% and
t→bℓν uses 99.5%.

## Legacy training example

When the synthetic pipeline operates correctly, train on JetClass with the train/validation/test splits. The JetClass benchmark came with ParT. It is a large dataset for jet tagging, with 100 million jets, and its design is specially for this problem.[1]

Recommended training items:

- `nn.CrossEntropyLoss` for multi-class classification
- `AdamW` optimizer
- a warmup of the learning rate, then a cosine decay
- gradient clipping
- checkpoints of the best model and the last model

Example of a launch command:

```bash
python train_gds_part.py \
  --train ./particle_transformer/data/JetClass/train_100M \
  --val   ./particle_transformer/data/JetClass/val_5M \
  --test  ./particle_transformer/data/JetClass/test_20M \
  --epochs 10 \
  --batch_size 512
```

## Accuracy and evaluation

For multi-class jet tagging, the most simple metric is the top-1 accuracy:

$$
\text{Accuracy} = \frac{\text{correct predictions}}{\text{total predictions}}
$$

In code, you usually calculate it with `argmax` over the class logits, and then compare the result with the true labels.

Useful evaluation outputs:

- training accuracy for each epoch
- validation accuracy for each epoch
- final test accuracy
- accuracy for each class
- confusion matrix

These metrics make it easier to compare GDS-ParT with baselines similar to ParT. They also show which jet categories get the most benefit from the relational gates.[1][2]

## Interpreting the gated mechanism

A good explanation of the model is:

> ParT adds the pairwise kinematics to the attention matrix as an additive bias. GDS-ParT converts the pairwise kinematics into gates for each particle. These gates change the query and key channels before attention. Thus, the pairwise physics changes the geometry of the representation, and not only the final attention scores.

This statement gives the central new part of the architecture, when you compare it with the original ParT.[1]

## Suggested next steps

- Complete the 20-epoch 10M run (resume from epoch 8).
- Show the learned gates in plots, to help interpret them.
- Compare the gated fusion with a small baseline similar to ParT.
- Find if different gates for queries and keys increase the accuracy.
- [GDS-ParT-MoE](https://github.com/Avinashistoocool/GDS-parT-MoE) examines how to interpret the experts for each particle. This work is in progress.

## Documentation

All the documentation is at **[avinashistoocool.github.io/Gated-Dual-Stream-Particle-Transformer](https://avinashistoocool.github.io/Gated-Dual-Stream-Particle-Transformer/)**.

- [**Technical report (PDF)**](docs/GDS-ParT_Technical_Report.pdf): the motivation, an architecture diagram, the input features, the embeddings, the bias of ParT and the GDS gates. It also has the gate gradients, the full forward pass, the training recipe, the evaluation metrics, results, ablations, limitations and the outlook. The LaTeX source is [`docs/GDS-ParT_Technical_Report.tex`](docs/GDS-ParT_Technical_Report.tex). Compile it two times with pdfLaTeX.
- [**GDS vs ParT, interactive**](https://avinashistoocool.github.io/Gated-Dual-Stream-Particle-Transformer/gds_vs_part.html): the visual explainer and the list of questions from [`docs/gds_vs_part.html`](docs/gds_vs_part.html).
- [**Setup and training guide**](GUIDE.md): environment, data, validation, training, ablations and troubleshooting.

## Citing

If GDS-ParT helps your research, cite it with the GitHub **Cite this repository** button
(from [`CITATION.cff`](CITATION.cff)). Also cite the ParT paper and the JetClass dataset [1].

## Reference sources

- ParT paper: [Particle Transformer for Jet Tagging](https://arxiv.org/abs/2202.03772) [1]
- Official repository: [jet-universe/particle_transformer](https://github.com/jet-universe/particle_transformer) [2]

## Disclaimer

ParT is the inspiration for this project, but the project examines a different architectural mechanism. It is an original research prototype for understanding, experiments and presentation. It is not an official reproduction of the published model.[1][2]

## License

MIT. Refer to [`LICENSE`](LICENSE). The JetClass dataset and the ParT reference code for the comparison keep their own licenses.
