# GDS-ParT executable companion — chapters 0–15

This is an **independent, explanatory implementation of this repository's local
GDS-ParT**, not an implementation or benchmark reproduction of published ParT.
It contains actual PyTorch layers, relation summaries, attention, residual blocks
and readout; it does not import, subclass or wrap the production model.

The primary reference is `docs\GDS-ParT_Technical_Report.pdf` and its `.tex`.
Implementation details are checked against `gds_part_colab.py` and
`data\jetclass_cache.py`. Tests import those sources **only as independent
references**; the reconstructed model itself has no repository imports.

## Run from the repository root

The existing Windows `py -3.14` environment has PyTorch. The repository's
Python 3.13 `.venv` is for Manim and does not have PyTorch; do not use that
environment for these commands. No package installation, dataset, GPU, network
access or Jupyter package is required for command-line validation.

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

An optional architecture-only test entry point is:

```powershell
py -3.14 docs\manim_course\companion\test_reconstruction.py
```

Open `notebook.ipynb` in an existing PyTorch-capable notebook environment and
**Run All**, starting in either the repository root or this companion directory.
The notebook preserves all **67 original chapter-0/1 cells** and extends them to
all 16 chapters. It currently has 138 cells, including 35 executed code cells.
To refresh authored generated sections (without replacing preserved review cells):

```powershell
py -3.14 docs\manim_course\companion\build_notebook.py
py -3.14 docs\manim_course\companion\validate_notebook.py
```

The normal course/review exporters must not overwrite this notebook. The builder
replaces only cells tagged `companion_generated`.

### Standalone downloaded course bundle

The bundle needs only the `manim_course` directory (including `content.py` and
`companion`) plus Python/PyTorch. No production model or JetClass cache module
is required. From the delivered `manim_course` directory:

```powershell
py -3.14 companion\tests.py
py -3.14 companion\train_demo.py --steps 140
py -3.14 companion\validate_notebook.py
```

On systems without the Windows Python launcher, substitute the Python executable
from an existing PyTorch environment for `py -3.14`.
The notebook supports this delivered course directory as its working directory
as well. The complete independent suite still runs; exactly four optional
production-reference checks are explicitly reported **skipped** when the original
repository sources are absent. They compare cache pair arithmetic, full logits,
parameter counts, and gradients. No modules are silently downloaded or substituted.
The independent parameter-count, pair-example, masking, symmetry and training
checks remain active. `output\tests.json` records the skip count and reasons.
If the delivery includes the clearly labelled original references
`manim_course\gds_part_colab.py` and `manim_course\data\jetclass_cache.py`,
the suite discovers them at the course root and runs all four reference checks.
Those are comparison oracles, not dependencies of the independent reconstruction.

## Files and learning path

| File | Purpose |
|---|---|
| `gds_part_reconstruction.py` | Independent model, physical pair calculation, configurations, audit and tiny training function. Read its explanatory docstrings top to bottom. |
| `tests.py` | Complete suite, including preserved review calculations. |
| `test_reconstruction.py` | Shapes, initialization, symmetry, masks, parameter counts, production parity, gradients, numerical caveats and overfit tests. |
| `train_demo.py` | Reproducible CPU optimization and explicit falling-loss check. |
| `chapter_examples.py` | Short sequential module-by-module cells; `EXAMPLES` maps names to runnable source. |
| `math_examples.py` | Reproducible report examples and their numerical results. |
| `notebook.ipynb` | Explanations, code, worked exercises and captured outputs for chapters 0–15. |
| `build_notebook.py` | Non-destructive integration of original review cells with new chapter sections. |
| `validate_notebook.py` | Plain-Python execution of every notebook cell, chapter coverage check and output capture. |
| `output\*.json` | Live audit, tests, math, snippets, notebook validation and demo artifacts. |

The notebook follows: orientation (0), algebra/products (1), tensors/masks (2),
logs/softmax/loss (3), learning (4), gradients (5), jets/features (6),
four-vectors/pair features (7), embeddings/LayerNorm (8), attention (9),
pair bias (10), gates (11), gate effects/gradients (12), complete forward (13),
progressive reconstruction (14), and experiments/limitations (15).

## Minimal forward pass

From the companion directory, or after adding it to `sys.path`:

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

Real 17-feature inputs come from fixed cache preprocessing, **not** from
`pairwise_features`. Random features and constructed vectors above are a software
example, not a physical event simulation.

### Public module interfaces

`B`=jets, `N`=particle slots, `D`=particle width, `P`=relation width, `H`=heads.
Masks are boolean `[B,N]`, **True=real particle**.

| Module / call | Output |
|---|---|
| `ParticleEmbedding(17,D,dropout)(features)` | `[B,N,D]` |
| `PairwiseEmbedding(4,P)(pairs)` | `[B,N,N,P]` |
| `pairwise_features(vectors,keep)` with `(px,py,pz,E)` vectors | `[B,N,N,4]` |
| `RelationalGate(P,D)(relations,keep)` | query/key gates, each `[B,N,D]` |
| `RelationalGate(P,D,conditioned=True)(relations,keep,normalized_states)` | query/key gates, each `[B,N,D]` |
| `GatedAttention(D,H,dropout)(states,gates,bias,keep)` | `[B,N,D]` |
| Same attention with `return_weights=True` | output plus pre-dropout weights `[B,H,N,N]` |
| `GDSParTBlock(cfg)(states,relations,shared_gates,bias,keep)` | `[B,N,D]` |
| `ClassAttentionBlock(cfg)(states,cls,keep)` | updated CLS `[B,1,D]` |
| `GDSParT(cfg)(features,pairs,keep)` | raw logits `[B,n_classes]` |

`gates` is `(gq,gk)` or `None`; `bias` is `[B,H,N,N]` or `None`.
Conditioned blocks own their gates, so pass `shared_gates=None`.
The full model accepts `keep=None` to treat every slot as real; the lower-level
teaching functions require explicit masks. `ParticleBlock`, `GDSParTConfig` and
`pair_features` are compatibility aliases for earlier companion names.

## Exact parameter audit

`configurations()` returns fresh `Config` objects:

| Key | Particle blocks | Gate / aggregation | Pair bias | CLS blocks | Parameters |
|---|---:|---|---|---:|---:|
| `v1` | 6 | sigmoid / shared masked mean | no | 0 | **1,280,684** |
| `shared` | 8 | identity / shared masked mean | yes | 2 | **2,074,420** |
| `conditioned` | 8 | identity / per-layer conditioned | yes | 2 | **2,488,116** |
| `no_gates` | 8 | absent | yes | 2 | **2,024,756** |

All use `D=128`, `P=64`, `H=8`, FFN width 512, ten classes, dropout 0.1.
The default `Config()` is **conditioned**, not the reported shared v2.
`max_particles=128` describes the cache cap; it allocates no positional embedding
and does not truncate a forward input.

Shared-v2 real module audit, in report order:

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

A conditioned gate adds a context projection of `128*64+64=8,256`,
giving 57,920 per block. Thus shared→conditioned changes the count by
`8*57,920−49,664=413,696`. No-gates removes exactly 49,664 from shared.
All totals are computed from real trainable tensors with `sum(p.numel())`;
there are no dummy parameters or counting offsets.

### Source/report clarification: historical v1

The production config docstring says the four switches `gate_mode="sigmoid"`,
`per_layer_gate=False`, `pair_bias=False`, `n_cls_layers=0` recover v1.
They recover its **operations**, but retain the production default of eight
particle blocks unless `n_layers=6` is also specified. Eight blocks give
**1,677,228**, not the historical **1,280,684**. The report's configuration table
explicitly uses six. This companion follows that table and tests both counts;
production code has not been changed.

## Semantics, numerical caveats and deliberate teaching differences

- Pair features use rapidity, wrapped azimuth, natural logs and the cache's
  exact float32 operation order/floors. Self/padded pairs are multiplied by
  zero at the end. **NaN times zero remains NaN**. Extremely large finite
  four-vectors can overflow their squared norms. Tests preserve and demonstrate
  both behaviours rather than silently sanitizing them.
- Raw zero self relations can embed to nonzero vectors because affine biases
  exist. Self-pairs participate in neighbour aggregation; padded neighbours do
  not. The relation embedding is evaluated once per full forward.
- Identity gates start **exactly one** for finite summaries, not 0.5. The
  range `(0,2)` is a real-arithmetic statement; floating-point tanh may saturate
  to 0 or 2 after adding one. Their early MLP/context gradients initially vanish
  because final gate weights are zero, while final-layer gradients can learn.
- Only Q and K are gated, after projection; V is not. Head scale is
  `sqrt(D/H)`. Pair bias is added to logits; padding masks **key columns**.
  Padded query states need not be zero. In CLS attention, PyTorch MHA reverses
  the mask convention: True means ignore.
- The full reconstruction explicitly rejects non-finite inputs, malformed masks
  and jets without any real particles. These are clearer **extra validation
  guards**, not claimed production behaviours. Low-level all-masked explicit
  softmax yields NaNs; production SDPA may handle empty rows differently.
- Finite, moderate padding does not affect real outputs, even if its values
  are nonzero or extra padding slots are appended. This is not a guarantee
  against intermediate overflow: `0*NaN`, `0*infinity`, or `+infinity−infinity`
  can contaminate results. Production's additive-mask ordering is preserved.
- Explicit attention makes every multiplication visible but is less
  memory-efficient than production SDPA. Activation checkpointing is omitted;
  parameterization and evaluation semantics are unchanged. Identical state
  dictionaries are copied strictly in tests. Evaluation logits and gradients
  match within floating-point tolerances, not necessarily bit-for-bit.
- Permutation/padding tests run with dropout disabled and gates deliberately
  moved away from identity, so they check actual relational routing.
  Batch/head reductions can reorder floating-point summation, hence tolerances.
- CLS pooling updates only CLS with ungated, unbiased attention to `[CLS;X]`.
  Mean pooling applies final LayerNorm before the masked mean.

## Reproducible outputs and interpretation

`output\math_examples.json` contains actual computed values, including:

- LayerNorm of `[2,4,6,8]`: mean 5, population variance 5, epsilon `1e-5`;
  gamma `[1,2,.5,1]`, beta `.1` gives approximately
  `[-1.241639,-.794426,.323607,1.441639]`.
- Attention: weights `[.669762,.330238]`, mixed values `[1.339523,1.320954]`;
  additive bias `[0,1]` gives `[.427295,.572705]`;
  halving the query gives `[.587479,.412521]`.
- Pair example `pT=(80,30), Δy=.15, Δφ=−.15`: log features approximately
  `[-1.5505,1.8507,-1.2993,4.6821]`.
- Ten-class smoothing with epsilon `.1`: target probability `.91`, others `.01`.
- Scalar chain-rule example: prediction 7, loss 4, derivative 12, updated weight .8.

The 140-step CPU demo (seed 7, eight fixed examples, two classes, AdamW,
learning rate .015, gradient norm clipping at 1) produced:

```text
initial loss:       0.7330597639083862
final loss:         0.0000007152553962441743
training accuracy: 1.0
```

Reference validation environment: Python 3.14, PyTorch `2.11.0+cu128`, **CPU
execution**, two PyTorch threads. The complete repository suite passed **24 tests**; all
**35 notebook code cells** executed successfully across chapters 0–15.
Numerical trajectories may differ slightly
across library/hardware versions. The demo tests loss reduction/one-batch
memorization only. It is not a dataset benchmark or evidence that gates improve
held-out jet tagging. Consult the report's controlled experiments and limitations.
