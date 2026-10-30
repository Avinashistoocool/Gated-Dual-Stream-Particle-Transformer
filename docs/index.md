# GDS-ParT: Gated Dual-Stream Particle Transformer

**GDS-ParT** is a PyTorch transformer for jet tagging on the
[JetClass](https://zenodo.org/records/6619768) benchmark. It explores another way of feeding
pairwise particle physics into attention.

[Technical report (PDF)](GDS-ParT_Technical_Report.pdf){ .md-button .md-button--primary }
[Interactive explainer](gds_vs_part.html){ .md-button }
[Source on GitHub](https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer){ .md-button }

## The idea

The [Particle Transformer (ParT)](https://arxiv.org/abs/2202.03772) adds a learned function of the
pairwise kinematics $U$ to the attention logits:

$$
A = \mathrm{softmax}\!\left(\frac{QK^\top}{\sqrt d} + \psi(U)\right)
$$

GDS-ParT keeps two streams: particle states $X \in \mathbb R^{B\times N\times 128}$ and a relation
tensor $R \in \mathbb R^{B\times N\times N\times 64}$ embedded from four pair features
($\ln\Delta$, $\ln k_T$, $\ln z$, $\ln m^2$). Each particle aggregates its row of $R$ into a summary
$s_i$. Two small MLPs map that summary to identity-initialised gates, which rescale the query and key
channels **before** the dot product:

$$
g_i^{Q,K} = 1 + \tanh\big(\mathrm{MLP}_{Q,K}(s_i)\big) \in (0,2)^{D},\qquad
q'_i = q_i \odot g_i^Q,\quad k'_j = k_j \odot g_j^K
$$

The pair physics therefore changes the space in which particles compare themselves, not only the
final attention scores. The reported v2 model is a hybrid. It keeps a ParT-style per-head pair bias
next to the gates and pools with two class-attention blocks.

## Results (10M training jets, 1M-jet balanced test subset)

| Model | Params | Test accuracy | Macro AUC |
|---|---|---|---|
| GDS-ParT v1 (sigmoid gates, mean pooling, 8 epochs) | 1.28M | 0.8272 | 0.9824 |
| **GDS-ParT v2** (shared gates + pair bias, 10 epochs) | 2.07M | **0.8484** | **0.9853** |
| ParT, 10M jets (published) | 2.14M | 0.850 | 0.9860 |
| ParticleNet, 10M jets (published) | 370k | 0.837 | 0.9837 |

The ParT comparison is useful context but not controlled. The published model saw about five times
more jets with a different optimiser and was evaluated on the full 20M test set. Small-budget
ablations show that the additive bias matters most. The difference between the gated and bias-only
models is within single-seed noise. The [technical report](GDS-ParT_Technical_Report.pdf) covers this
in detail, together with the per-class rejections and the limitations. Full tables are in
[`results/v2_shared_10M_10ep/RESULTS.md`](https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer/blob/main/results/v2_shared_10M_10ep/RESULTS.md).

## Documentation

| | |
|---|---|
| [**Technical report (PDF)**](GDS-ParT_Technical_Report.pdf) | Architecture with diagram, every equation as the code computes it, input features, gate gradients, training recipe, metrics, results, ablations, limitations and outlook. The LaTeX source is [`docs/GDS-ParT_Technical_Report.tex`](https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer/blob/main/docs/GDS-ParT_Technical_Report.tex). |
| [**GDS vs ParT, interactive**](gds_vs_part.html) | Builds both architectures from tensors upward, with a live toy model comparing attention maps and an answered list of questions. |
| [**Setup and training guide**](guide.md) | Environment, JetClass data, pipeline validation, training on a workstation or HPC cluster, ablations and reports. |

## Quick start

```bash
git clone https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer gds_part
cd gds_part
pip install -r requirements.txt
python gds_part_colab.py            # forward/backward/step sanity check on synthetic data
python train_gds_part.py --synthetic --epochs 1 --synthetic_batches 4 --batch_size 8 \
  --embed_dim 32 --n_heads 4 --n_layers 2 --ffn_dim 64 --pairwise_embed_dim 16 \
  --max_particles 32 --checkpoint_dir ./checkpoints/smoke
```

## Citing

If you use GDS-ParT, cite it with GitHub's **Cite this repository** button, which reads
[`CITATION.cff`](https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer/blob/main/CITATION.cff).
Please also cite the ParT paper and the JetClass dataset:
H. Qu, C. Li and S. Qian, *Particle Transformer for Jet Tagging*, ICML 2022,
[arXiv:2202.03772](https://arxiv.org/abs/2202.03772).
