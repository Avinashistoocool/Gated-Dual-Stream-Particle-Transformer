# GDS-ParT: Gated Dual-Stream Particle Transformer

**GDS-ParT** is a PyTorch transformer for jet tagging on the
[JetClass](https://zenodo.org/records/6619768) benchmark. It examines a different method to put
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

GDS-ParT keeps two streams. The first stream is the particle states $X \in \mathbb R^{B\times N\times 128}$. The second stream is a relation
tensor $R \in \mathbb R^{B\times N\times N\times 64}$, which the model embeds from four pair features
($\ln\Delta$, $\ln k_T$, $\ln z$, $\ln m^2$). Each particle aggregates its row of $R$ into a summary
$s_i$. Two small MLPs change that summary into gates that start as the identity. These gates scale the query and key
channels **before** the dot product:

$$
g_i^{Q,K} = 1 + \tanh\big(\mathrm{MLP}_{Q,K}(s_i)\big) \in (0,2)^{D},\qquad
q'_i = q_i \odot g_i^Q,\quad k'_j = k_j \odot g_j^K
$$

Thus, the pair physics changes the space in which the particles compare themselves. It does not only change the
final attention scores. The reported v2 model is a hybrid. It keeps a ParT-style pair bias for each head
together with the gates. It pools with two class-attention blocks.

## Results (10M training jets, 1M-jet balanced test subset)

| Model | Params | Test accuracy | Macro AUC |
|---|---|---|---|
| GDS-ParT v1 (sigmoid gates, mean pooling, 8 epochs) | 1.28M | 0.8272 | 0.9824 |
| **GDS-ParT v2** (shared gates + pair bias, 10 epochs) | 2.07M | **0.8484** | **0.9853** |
| ParT, 10M jets (published) | 2.14M | 0.850 | 0.9860 |
| ParticleNet, 10M jets (published) | 370k | 0.837 | 0.9837 |

The ParT comparison gives useful context, but it is not a controlled comparison. The published model used about five times
more jets and a different optimiser. Its evaluation used the full 20M test set. Ablations with a small budget show that the additive bias is the most important part.
The difference between the gated model and the model with only a bias is in the noise of a single seed. The [technical report](GDS-ParT_Technical_Report.pdf) gives
the details, with the rejections for each class and the limitations. The full tables are in
[`results/v2_shared_10M_10ep/RESULTS.md`](https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer/blob/main/results/v2_shared_10M_10ep/RESULTS.md).

## Documentation

| | |
|---|---|
| [**Technical report (PDF)**](GDS-ParT_Technical_Report.pdf) | The architecture with a diagram and each equation as the code calculates it. The input features, the gate gradients and the training recipe. The metrics, results, ablations, limitations and the outlook. The LaTeX source is [`docs/GDS-ParT_Technical_Report.tex`](https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer/blob/main/docs/GDS-ParT_Technical_Report.tex). |
| [**GDS vs ParT, interactive**](gds_vs_part.html) | Builds the two architectures from tensors upward. It has a live toy model that compares attention maps, and a list of questions with answers. |
| [**Setup and training guide**](guide.md) | The environment, the JetClass data, the validation of the pipeline, training on a workstation or an HPC cluster, ablations and reports. |

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

If you use GDS-ParT, cite it with the GitHub **Cite this repository** button. This button reads
[`CITATION.cff`](https://github.com/Avinashistoocool/Gated-Dual-Stream-Particle-Transformer/blob/main/CITATION.cff).
Also cite the ParT paper and the JetClass dataset:
H. Qu, C. Li and S. Qian, *Particle Transformer for Jet Tagging*, ICML 2022,
[arXiv:2202.03772](https://arxiv.org/abs/2202.03772).
