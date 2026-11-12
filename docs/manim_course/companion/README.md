# GDS-ParT executable companion — chapters 0–15

This is an **independent, explanatory implementation of the local GDS-ParT of this
repository**. It is not an implementation or a benchmark reproduction of the published ParT.
It contains actual PyTorch layers, relation summaries, attention, residual blocks
and readout. It does not import, subclass or wrap the production model.

The primary reference is `docs\GDS-ParT_Technical_Report.pdf` and its `.tex`.
The implementation details have checks against `gds_part_colab.py` and
`data\jetclass_cache.py`. The tests import those sources **only as independent
references**. The reconstructed model itself does not import from the repository.

## Run from the repository root

The Windows `py -3.14` environment has PyTorch. The Python 3.13 `.venv` of the
repository is for Manim and does not have PyTorch. Do not use that
environment for these commands. The command-line validation does not need package installation, a dataset, a GPU, network
access or the Jupyter package.

```powershell
# Complete foundation + architecture + physics + parity + overfit suite.
py -3.14 docs\manim_course\companion\tests.py

# Independent 140-step synthetic CPU training demonstration.
py -3.14 docs\manim_course\companion\train_demo.py --steps 140

# Execute every notebook code cell and persist captured outputs in the notebook.
py -3.14 docs\manim_course\companion\validate_notebook.py

# Regenerate the short progressive chapter-14 source/output artifact.
py -3.14 docs\manim_course\companion\chapter_examples.py

# Regenerate report arithmetic (LayerNorm, attention, physics, gradients, etc.).
py -3.14 docs\manim_course\companion\math_examples.py

# Print full-size parameter counts, breakdown and the CPU demo together.
py -3.14 docs\manim_course\companion\gds_part_reconstruction.py
```

An optional entry point for the architecture tests only is:

```powershell
py -3.14 docs\manim_course\companion\test_reconstruction.py
```

Open `notebook.ipynb` in a notebook environment that has PyTorch, and
**Run All**. You can start in the repository root or in this companion directory.
The notebook keeps all **67 original chapter-0/1 cells** and extends them to
all 16 chapters. At this time, it has 138 cells, with 35 executed code cells.
To update the generated sections (and not replace the kept review cells), run these commands:

```powershell
py -3.14 docs\manim_course\companion\build_notebook.py
py -3.14 docs\manim_course\companion\validate_notebook.py
```

The usual course/review exporters must not overwrite this notebook. The builder
replaces only the cells with the tag `companion_generated`.

### Standalone downloaded course bundle

The bundle needs only the `manim_course` directory (with `content.py` and
`companion`) and Python/PyTorch. It does not need the production model or the JetClass cache module.
From the delivered `manim_course` directory, run these commands:

```powershell
py -3.14 companion\tests.py
py -3.14 companion\train_demo.py --steps 140
py -3.14 companion\validate_notebook.py
```

On systems without the Windows Python launcher, use the Python executable
from a PyTorch environment in place of `py -3.14`.
The notebook also supports this delivered course directory as its working directory.
The complete independent suite continues to run. When the original
repository sources are not present, the suite explicitly reports that exactly four optional
production-reference checks are **skipped**. These checks compare the cache pair arithmetic, the full logits,
the parameter counts and the gradients. The suite does not silently download or replace modules.
The independent checks of parameter counts, pair examples, masks, symmetry and training
stay active. `output\tests.json` records the skip count and the reasons.
The delivery can include the original references with clear labels,
`manim_course\gds_part_colab.py` and `manim_course\data\jetclass_cache.py`.
In that case, the suite finds them at the course root and runs all four reference checks.
These references are comparison oracles. The independent reconstruction does not depend on them.

## Files and learning path

| File | Purpose |
|---|---|
| `gds_part_reconstruction.py` | The independent model, the physical pair calculation, the configurations, the audit and a small training function. Read its explanatory docstrings from top to bottom. |
| `tests.py` | The complete suite, with the kept review calculations. |
| `test_reconstruction.py` | Shapes, initialization, symmetry, masks, parameter counts, parity with the production code, gradients, numerical caveats and overfit tests. |
| `train_demo.py` | A reproducible CPU optimization and an explicit check that the loss decreases. |
| `chapter_examples.py` | Short sequential cells, one module at a time. `EXAMPLES` maps names to runnable source. |
| `math_examples.py` | Reproducible report examples and their numerical results. |
| `notebook.ipynb` | Explanations, code, worked exercises and captured outputs for chapters 0–15. |
| `build_notebook.py` | Adds the new chapter sections to the original review cells, and does not delete them. |
| `validate_notebook.py` | Executes each notebook cell with plain Python, checks the chapter coverage and captures the output. |
| `output\*.json` | Live artifacts for the audit, tests, mathematics, snippets, notebook validation and demo. |

The notebook has this sequence: orientation (0), algebra/products (1), tensors/masks (2),
logs/softmax/loss (3), learning (4), gradients (5), jets/features (6),
four-vectors/pair features (7), embeddings/LayerNorm (8), attention (9),
pair bias (10), gates (11), gate effects/gradients (12), complete forward (13),
progressive reconstruction (14), and experiments/limitations (15).

## Minimal forward pass

Run this code from the companion directory, or after you add the directory to `sys.path`:

```python
import torch
from gds_part_reconstruction import Config, GDSParT, pairwise_features

cfg = Config(embed_dim=16, pairwise_embed_dim=8, ffn_dim=32,
             n_heads=2, n_layers=2, n_cls_layers=1, dropout=0)
model = GDSParT(cfg).eval()
features = torch.randn(2, 4, 17)
keep = torch.tensor([[1, 1, 1, 0], [1, 1, 0, 0]], dtype=torch.bool)
vectors = torch.randn(2, 4, 4)  # construct timelike synthetic four-vectors
vectors[..., 3] = vectors[..., :3].square().sum(-1).sqrt() + 1
pairs = pairwise_features(vectors, keep)
logits, trace = model(features, pairs, keep, return_trace=True)
assert logits.shape == (2, 10)
print(trace)
```

The real 17-feature inputs come from the fixed cache preprocessing, **not** from
`pairwise_features`. The random features and the constructed vectors above are a software
example. They are not a simulation of a physical event.

### Public module interfaces

`B`=jets, `N`=particle slots, `D`=particle width, `P`=relation width, `H`=heads.
The masks are boolean `[B,N]`, and **True=real particle**.

| Module / call | Output |
|---|---|
| `ParticleEmbedding(17,D,dropout)(features)` | `[B,N,D]` |
| `PairwiseEmbedding(4,P)(pairs)` | `[B,N,N,P]` |
| `pairwise_features(vectors,keep)` with `(px,py,pz,E)` vectors | `[B,N,N,4]` |
| `RelationalGate(P,D)(relations,keep)` | query/key gates, each `[B,N,D]` |
| `RelationalGate(P,D,conditioned=True)(relations,keep,normalized_states)` | query/key gates, each `[B,N,D]` |
| `GatedAttention(D,H,dropout)(states,gates,bias,keep)` | `[B,N,D]` |
| Same attention with `return_weights=True` | output plus the weights before dropout `[B,H,N,N]` |
| `GDSParTBlock(cfg)(states,relations,shared_gates,bias,keep)` | `[B,N,D]` |
| `ClassAttentionBlock(cfg)(states,cls,keep)` | updated CLS `[B,1,D]` |
| `GDSParT(cfg)(features,pairs,keep)` | raw logits `[B,n_classes]` |

`gates` is `(gq,gk)` or `None`. `bias` is `[B,H,N,N]` or `None`.
Conditioned blocks own their gates, so give `shared_gates=None`.
The full model accepts `keep=None`, and then it treats each slot as real. The lower-level
teaching functions require explicit masks. `ParticleBlock`, `GDSParTConfig` and
`pair_features` are compatibility aliases for earlier companion names.

## Exact parameter audit

`configurations()` returns new `Config` objects:

| Key | Particle blocks | Gate / aggregation | Pair bias | CLS blocks | Parameters |
|---|---:|---|---|---:|---:|
| `v1` | 6 | sigmoid / shared masked mean | no | 0 | **1,280,684** |
| `shared` | 8 | identity / shared masked mean | yes | 2 | **2,074,420** |
| `conditioned` | 8 | identity / per-layer conditioned | yes | 2 | **2,488,116** |
| `no_gates` | 8 | absent | yes | 2 | **2,024,756** |

All configurations use `D=128`, `P=64`, `H=8`, FFN width 512, ten classes and dropout 0.1.
The default `Config()` is **conditioned**. It is not the reported shared v2.
`max_particles=128` describes the cache limit. It does not allocate a positional embedding,
and it does not cut a forward input.

Audit of the real modules of the shared v2, in the order of the report:

| Component | Parameters |
|---|---:|
| Particle embedding | 18,850 |
| Pair embedding | 4,480 |
| Shared query/key gates | 49,664 |
| Pair-bias projection | 520 |
| Eight attention modules | 528,384 |
| Sixteen particle-block LayerNorms | 4,096 |
| Eight particle-block FFNs | 1,053,696 |
| CLS token | 128 |
| Two complete CLS blocks | 396,544 |
| Final LayerNorm | 256 |
| Classifier | 17,802 |
| **Total** | **2,074,420** |

A conditioned gate adds a context projection of `128*64+64=8,256`.
This gives 57,920 for each block. Thus, the change from shared to conditioned changes the count by
`8*57,920−49,664=413,696`. No-gates removes exactly 49,664 from shared.
All totals come from real trainable tensors with `sum(p.numel())`.
There are no dummy parameters and no counting offsets.

### Source/report clarification: historical v1

The docstring of the production config says that the four switches `gate_mode="sigmoid"`,
`per_layer_gate=False`, `pair_bias=False`, `n_cls_layers=0` give v1 again.
They give its **operations** again. But they keep the production default of eight
particle blocks, if you do not also specify `n_layers=6`. Eight blocks give
**1,677,228**, not the historical **1,280,684**. The configuration table of the report
explicitly uses six. This companion follows that table and tests the two counts.
The production code did not change.

## Semantics, numerical caveats and deliberate teaching differences

- The pair features use rapidity, wrapped azimuth, natural logs and the exact
  float32 operation order/floors of the cache. At the end, the code multiplies self pairs and padded pairs
  by zero. **NaN times zero stays NaN**. Extremely large finite
  four-vectors can cause an overflow of their squared norms. The tests keep and show
  these two behaviours. They do not silently remove them.
- Raw zero self relations can embed to nonzero vectors, because affine biases
  exist. Self pairs are part of the neighbour aggregation. Padded neighbours are not.
  The code calculates the relation embedding one time for each full forward pass.
- Identity gates start at **exactly one** for finite summaries, not at 0.5. The
  range `(0,2)` is true for real arithmetic. In floating point, tanh can saturate,
  so the gate can become 0 or 2 after the code adds one. At the start, the early MLP/context gradients
  are zero, because the final gate weights are zero. But the final layer can learn from its gradients.
- The code gates only Q and K, after the projection. It does not gate V. The head scale is
  `sqrt(D/H)`. The code adds the pair bias to the logits. Padding masks the **key columns**.
  The padded query states do not have to be zero. In CLS attention, PyTorch MHA uses the opposite
  mask convention: True means "ignore".
- The full reconstruction explicitly rejects non-finite inputs, incorrect masks
  and jets without real particles. These are clearer **extra validation
  guards**. There is no claim that the production code does the same. A low-level explicit
  softmax where all items are masked gives NaNs. The production SDPA can operate differently for empty rows.
- Finite, moderate padding does not change the real outputs. This is true when the padding values
  are not zero, and when you add more padding slots. This is not a guarantee
  against an overflow in intermediate values: `0*NaN`, `0*infinity` or `+infinity−infinity`
  can make the results incorrect. The order of the additive mask in the production code does not change.
- Explicit attention shows each multiplication, but it uses more memory than the
  production SDPA. The companion does not use activation checkpointing. The
  parameterization and the evaluation semantics do not change. The tests copy identical state
  dictionaries strictly. The evaluation logits and the gradients agree within floating-point tolerances.
  They do not always agree bit for bit.
- The permutation and padding tests run with dropout disabled. The tests also move the gates
  away from the identity on purpose, so that they check the actual relational routing.
  Reductions over the batch and the heads can change the order of the floating-point sums. Thus, the tests use tolerances.
- CLS pooling updates only CLS, with attention to `[CLS;X]` that has no gates and no bias.
  Mean pooling applies the final LayerNorm before the masked mean.

## Reproducible outputs and interpretation

`output\math_examples.json` contains the actual calculated values. Some examples are:

- LayerNorm of `[2,4,6,8]`: mean 5, population variance 5, epsilon `1e-5`.
  With gamma `[1,2,.5,1]` and beta `.1`, the result is about
  `[-1.241639,-.794426,.323607,1.441639]`.
- Attention: weights `[.669762,.330238]`, mixed values `[1.339523,1.320954]`.
  The additive bias `[0,1]` gives `[.427295,.572705]`.
  A query with half the value gives `[.587479,.412521]`.
- Pair example `pT=(80,30), Δy=.15, Δφ=−.15`: the log features are about
  `[-1.5505,1.8507,-1.2993,4.6821]`.
- Ten-class smoothing with epsilon `.1`: the target probability is `.91`, and each other probability is `.01`.
- Scalar chain-rule example: prediction 7, loss 4, derivative 12, updated weight .8.

The 140-step CPU demo used seed 7, eight fixed examples, two classes, AdamW,
learning rate .015 and gradient norm clipping at 1. It gave these results:

```text
initial loss:       0.7330597639083862
final loss:         0.0000007152553962441743
training accuracy: 1.0
```

Reference validation environment: Python 3.14, PyTorch `2.11.0+cu128`, **CPU
execution**, two PyTorch threads. The complete repository suite passed **24 tests**. All
**35 notebook code cells** executed correctly across chapters 0–15.
The numerical trajectories can be slightly different
with different library or hardware versions. The demo tests only the decrease of the loss and the memorization of one batch.
It is not a dataset benchmark. It is not evidence that gates improve
jet tagging on held-out data. Refer to the controlled experiments and the limitations in the report.
