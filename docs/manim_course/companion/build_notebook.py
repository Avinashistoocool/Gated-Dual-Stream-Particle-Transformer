"""Extend the preserved chapter-0/1 notebook without replacing its authored cells.

Only cells tagged companion_generated are refreshed. The original narration and
seven arithmetic code cells stay intact. Generated cells are ordinary Python:
validate_notebook.py executes them without needing a Jupyter installation.
"""
import json
from pathlib import Path

from chapter_examples import EXAMPLES


HERE = Path(__file__).resolve().parent


def cell(kind, text):
    result = {"cell_type": kind, "metadata": {"companion_generated": True},
              "source": text.strip().splitlines(keepends=True)}
    if kind == "code":
        result.update(execution_count=None, outputs=[])
    return result


def section(number, title, explanation, source, exercises):
    return [
        cell("markdown", f"# Chapter {number}: {title}\n\n{explanation}"),
        cell("code", source),
        cell("markdown", f"### Check your understanding — worked answers\n\n{exercises}"),
    ]


def build():
    path = HERE / "notebook.ipynb"
    notebook = json.loads(path.read_text(encoding="utf-8"))
    preserved = [c for c in notebook["cells"] if not c.get("metadata", {}).get("companion_generated")]
    cells = [
        cell("markdown", """# GDS-ParT: mathematics to a working classifier

This companion mirrors **chapters 0–15**. The original chapter-0/1 narration
and arithmetic cells are preserved below. Later chapters add executable
derivations, module boundaries, exercises, exact parameter counts and training.

Authority: `docs/GDS-ParT_Technical_Report.pdf` and its `.tex`; local production
code resolves implementation details. The independent model imports no production
modules. This is the repository's local GDS-ParT, **not** published ParT.
Use **Run All** with Python and PyTorch. Start in the repository root or this
companion directory (or the delivered course directory). See README for exact
PowerShell commands and caveats.
Synthetic results demonstrate software plumbing only, not physics performance."""),
        cell("code", r"""from pathlib import Path
import sys, math, json
import torch
from torch import nn
from torch.nn import functional as F
companion = Path.cwd()
candidates = [companion, companion / "companion",
              companion / "docs" / "manim_course" / "companion"]
companion = next((p for p in candidates if (p / "gds_part_reconstruction.py").exists()), None)
assert companion is not None, "Start at repository, delivered course, or companion directory"
sys.path.insert(0, str(companion))
from gds_part_reconstruction import (
    Config, configurations, ParticleEmbedding, PairwiseEmbedding,
    RelationalGate, GatedAttention, GDSParTBlock, ClassAttentionBlock,
    GDSParT, pairwise_features, parameter_audit, shared_breakdown, cpu_demo)
from math_examples import worked_examples
torch.manual_seed(17)
torch.set_num_threads(2)
torch.set_printoptions(precision=6, sci_mode=False)
print("PyTorch", torch.__version__)"""),
    ] + preserved
    cells += section(2, "Tensors, shapes and masks", r"""
An axis names a kind of address, not a physical unit. Particle input F is
[B,N,17]; pair input U is [B,N,N,4]. The two N axes select query i and neighbour j.
A permutation must act on BOTH pair axes. Padding keeps tensor sizes rectangular;
`keep=True` identifies real particles. Broadcasting [B,1,N,1] excludes neighbour
j in a relation summary. Reducing the neighbour axis removes one N, not channels.

We use two jets with four slots; only three and two slots are real. Finite padding
may contain nonzero numbers. Never infer the mask from the feature value zero.""", """
x = torch.randn(2, 4, 17)
u = torch.randn(2, 4, 4, 4)
keep = torch.tensor([[1,1,1,0], [1,1,0,0]], dtype=torch.bool)
print("F,U,keep", x.shape, u.shape, keep.shape)
weights = keep[:, None, :, None].float()
print("neighbour broadcast", weights.shape)
print("summary", ((u*weights).sum(2)/weights.sum(2)).shape)
assert keep.sum(1).tolist() == [3, 2]
""", """1. Which mask dimension addresses keys? In `[B,1,1,N]`, the final N.
2. Is an all-zero particle necessarily padding? No: only the explicit mask decides.
Summary: check axes before arithmetic, and permute both particle axes of U.""")
    cells += section(3, "Logarithms, softmax and cross-entropy", r"""
Natural logarithms convert positive multiplicative scales to additive differences.
Softmax turns logits into probabilities: p_i=exp(s_i)/sum_j exp(s_j). Subtracting
the largest logit changes neither probabilities nor odds and improves stability.
The denominator normalizes one row at a time. Cross-entropy for target t is
`-log p_t`; feed **raw logits** to PyTorch cross_entropy, not probabilities.
For `[0,ln 2,ln 3]`, probabilities are `[1/6,2/6,3/6]`.""", """
s = torch.tensor([0., math.log(2), math.log(3)], dtype=torch.float64)
p = s.softmax(0)
print("probabilities", p)
print("sum", p.sum().item())
print("CE target 2", F.cross_entropy(s[None], torch.tensor([2])).item())
torch.testing.assert_close(p, (s+100).softmax(0))
assert abs(p[2].item()-.5) < 1e-12
""", """1. Adding 100 to every logit changes what? Nothing in softmax.
2. What is loss for target index 2? `-ln(1/2)=ln 2≈0.693147`.
Summary: logits are not probabilities, and stable normalization matters.""")
    cells += section(4, "Learning from examples", r"""
A classifier learns parameters from labelled training examples, not rules for
which physical identity a hidden channel must represent. The data split prevents
training loss from being mistaken for generalization. A Linear layer computes
X Wᵀ+b in PyTorch's row storage; W has [out,in] axes. GELU adds nonlinearity.
Accuracy measures decisions, while cross-entropy also measures confidence.
Our tiny examples are synthetic and do not estimate JetClass accuracy.""", """
linear = nn.Linear(2, 3)
with torch.no_grad():
    linear.weight.copy_(torch.tensor([[1.,0.], [0.,1.], [1.,1.]]))
    linear.bias.copy_(torch.tensor([0.,1.,-1.]))
sample = torch.tensor([[2.,3.]])
print("affine output", linear(sample))
print("GELU", F.gelu(torch.tensor([-1.,0.,1.])))
print("parameters", sum(p.numel() for p in linear.parameters()))
assert sum(p.numel() for p in linear.parameters()) == 9
""", """1. Why nine parameters? Six matrix entries plus three biases.
2. Can zero training loss establish physics performance? No; use held-out,
representative data and appropriate metrics. Summary: capacity and evaluation
answer different questions.""")
    cells += section(5, "Gradients and training", r"""
For a scalar parameter w, prediction is 3w+1 and squared error is (3w+1−5)².
At w=2, prediction 7 and loss 4 give derivative 2(7−5)3=12.
Gradient descent with learning rate 0.1 changes w to 0.8. Autograd stores
the computation graph and applies the chain rule backward. Zero gradients before
each new optimizer step unless deliberately accumulating micro-batches.
Training enables dropout; evaluation disables it but does not itself disable gradients.""", """
w = torch.tensor(2., requires_grad=True)
prediction = 3*w+1
loss = (prediction-5).square()
loss.backward()
print("prediction, loss, gradient", prediction.item(), loss.item(), w.grad.item())
with torch.no_grad():
    w -= .1*w.grad
print("updated w", w.item())
assert abs(w.item()-.8) < 1e-6
""", """1. Why is the derivative 12, not 4? The chain rule multiplies
the outer derivative 2(7−5) by the inner derivative 3.
2. Why `no_grad` for a manual update? Parameter updates are not new model
operations whose graph we want to differentiate. Summary: loss → backward → step.""")
    cells += section(6, "Jets and seventeen particle features", r"""
A jet is a set of reconstructed constituents receiving one class label. The
17 features are fixed-preprocessed observations: four log scales/fractions;
distance to jet axis; Δη, Δφ; charge; five identity flags; and four tracking
features. The cache uses `(raw−shift)*scale`, clips to [-5,5], then replaces
non-finite values. Shifts/scales are not learned. Four-vectors remain separate
float32 data, while cached constituent features use float16.

Class order is QCD,Hbb,Hcc,Hgg,H4q,Hqql,Zqq,Wqq,Tbqq,Tbl.
Track displacement can be informative but does not prove a flavour label.""", """
pt, energy, jet_pt, jet_energy = 80., 82., 200., 220.
first_four = torch.tensor([
    .7*(math.log(pt)-1.7), .7*(math.log(energy)-2.),
    .7*(math.log(pt/jet_pt)+4.7), .7*(math.log(energy/jet_energy)+4.7)])
print("first four preprocessed features", first_four.clamp(-5,5))
print("feature count", 4+1+2+1+5+4)
assert 4+1+2+1+5+4 == 17
""", """1. Does a 128-dimensional embedding add 111 measurements? No, it
learns a representation from the same 17 inputs.
2. Are pair features derived from two 17-vectors? No; they use separate
four-momenta. Summary: distinguish observation, preprocessing and representation.""")
    cells += section(7, "Four-vectors and pair features", r"""
Stored order is (px,py,pz,E). Compute pT, azimuth atan2(py,px), and rapidity
y=½ln((E+pz)/(E−pz)). Wrap Δφ into [-π,π), then
Δ=sqrt(Δy²+Δφ²), kT=min(pTi,pTj)Δ, z=min(pTi,pTj)/(pTi+pTj),
and m²=(Ei+Ej)²−|pi+pj|². The four input features are their natural logs.
Floors of 1e-8 match cache operations; self/padded pairs are multiplied by zero.
This is not robust NaN removal: NaN×0 remains NaN.

For massless pT=(80,30), Δy=.15 and Δφ=−.15, m²≈108.0002.
Massless rapidity equals pseudorapidity; do not assume equality for massive particles.""", """
pt = torch.tensor([80.,30.])
y, phi = torch.tensor([.15,0.]), torch.tensor([-.15,0.])
vectors = torch.stack((pt*phi.cos(), pt*phi.sin(),
                       pt*y.sinh(), pt*y.cosh()), -1)[None]
pairs = pairwise_features(vectors, torch.ones(1,2,dtype=torch.bool))
print("pair log features", pairs[0,0,1])
torch.testing.assert_close(pairs[0,0,1],
    torch.tensor([-1.5505,1.8507,-1.2993,4.6821]), atol=1e-4, rtol=0)
torch.testing.assert_close(pairs, pairs.transpose(1,2))
""", """1. Why z≤1/2? The smaller positive pT is at most half their sum.
2. Should arbitrary constituent-pair mass peak at a parent resonance? No:
the full decay system or appropriate subjets are needed. Summary: these are
pair descriptors, not unique reconstructed physical splittings.""")
    cells += section(8, "Embeddings, LayerNorm and two streams", r"""
Particle embedding is LN17→Linear(17,D)→GELU→Linear(D,D)→Dropout.
Pair embedding is Linear(4,P)→GELU→Linear(P,P), without normalization/dropout.
LayerNorm normalizes channels independently per row with population variance
(divide by D, not D−1), epsilon 1e-5, and learned gamma/beta.
For [2,4,6,8], mean=5 and variance=5.

R is computed once and stays static. X changes through particle blocks.
Zero input pairs can become nonzero embedded relations because affine biases exist;
self pairs are not excluded from the masked neighbour mean.""", """
v = torch.tensor([2.,4.,6.,8.])
gamma, beta = torch.tensor([1.,2.,.5,1.]), torch.full((4,),.1)
print("mean, population variance", v.mean().item(), v.var(unbiased=False).item())
print("LayerNorm", F.layer_norm(v, (4,), gamma, beta))
states = ParticleEmbedding(17,16,0)(x)
relations = PairwiseEmbedding(4,8)(u)
print("X,R", states.shape, relations.shape)
""", """1. Why two stream shapes? X has one particle address, R has two.
2. Is the pair embedding recomputed each block? No; only its use changes.
Summary: learned representations and their update rules are distinct.""")
    cells += section(9, "Self-attention from arithmetic", r"""
Queries ask, keys provide comparison coordinates, and values carry the mixture.
For each head, score_ij=q_i·k_j/sqrt(d); softmax over j creates weights that sum
to one before dropout. Weighted values are then concatenated and linearly projected.
Q,K,V are separate learned affine projections, not different raw inputs.
With q=[1,0], keys the 2×2 identity and values [[2,0],[0,4]], d=2.""", """
q, keys = torch.tensor([1.,0.]), torch.eye(2)
scores = keys @ q / math.sqrt(2)
probability = scores.softmax(0)
value = torch.tensor([[2.,0.],[0.,4.]])
print("scores", scores)
print("weights", probability)
print("weighted values", probability @ value)
assert abs(probability.sum().item()-1) < 1e-6
""", """1. Why divide by sqrt(d), not D? Each head's dot product has d terms.
2. Does attention softmax normalize channels? No, neighbouring key positions j.
Summary: the comparison and the value mixture are separate operations.""")
    cells += section(10, "Pair bias and attention odds", r"""
The learned pair projection P→H supplies one scalar bias per head and pair.
Attention softmax uses score+bias. Odds obey p_j/p_k=exp(score_j−score_k),
so increasing only bias_j by one multiplies its odds by e.
Padding is a hard exclusion with −infinity, not a learned penalty.
The local model may use both additive bias and Q/K gates; these are not synonyms.""", """
biased = (scores+torch.tensor([0.,1.])).softmax(0)
print("biased weights", biased)
print("odds multiplier", (biased[1]/biased[0]/(probability[1]/probability[0])).item())
masked = scores.masked_fill(torch.tensor([False,True]), -torch.inf).softmax(0)
print("one real key", masked)
bias = nn.Linear(8,2)(relations).permute(0,3,1,2)
print("pair bias", bias.shape)
""", """1. Which learned tensor has [B,H,N,N] axes? Additive pair bias.
2. Can one valid key get attention probability below one without dropout?
No, its row normalization is one. Summary: finite preference differs from exclusion.""")
    cells += section(11, "Relational gates", r"""
Shared gates average R_ij over real neighbours j, then use independent
P→D→D MLPs for Q and K. The shared summaries/gates are reused across all blocks.
Conditioned gates instead form c_i=Linear(LN(X_i)), score each relation with
<R_ij,c_i>/sqrt(P), mask padded j and softmax before weighted summation.
Each particle block owns its conditioned gate.

Identity gates use 1+tanh with zero-initialized final affine layers, hence start
exactly at one for finite summaries. Sigmoid gates use (0,1) and are not identity.
Floating point saturation can reach range endpoints.""", """
shared_gate = RelationalGate(8,16)
gq, gk = shared_gate(relations, keep)
print("exact identity", torch.equal(gq,torch.ones_like(gq)),
      torch.equal(gk,torch.ones_like(gk)))
conditioned_gate = RelationalGate(8,16,conditioned=True)
cgq, cgk = conditioned_gate(relations, keep, nn.LayerNorm(16)(states))
print("conditioned shapes", cgq.shape, cgk.shape)
print("sigmoid zero", torch.tensor(0.).sigmoid().item())
""", """1. Are query/key gate MLP weights shared with each other? No.
2. Do all v2 configurations use conditioned gates? No, the reported v2
uses shared mean gates; production defaults are conditioned.
Summary: activation, aggregation and sharing are separate switches.""")
    cells += section(12, "Gate effects, gradients and expressivity", r"""
Gates multiply query/key channels BEFORE the dot product; values are ungated.
Halving a positive score flattens attention, but making a negative score less
negative raises its relative probability. Thus 'suppression' is not a universal
statement about final attention probability.

At identity initialization, gate last-layer weights are zero, so gradients to
earlier gate/context layers are initially zero; the last layers can still learn.
Relation summaries compress N neighbours to P channels: gates cannot encode an
arbitrary independent pair bias. Shared R is static, not a second self-attention stack.""", """
print("half-query weights", (scores*.5).softmax(0))
print("negative logits before", torch.tensor([-2.,0.]).softmax(0))
print("negative logits after", torch.tensor([-1.,0.]).softmax(0))
gate = RelationalGate(8,16,conditioned=True)
a, b = gate(relations.detach(),keep,states.detach())
(a.sum()+b.sum()).backward()
print("context gradient", gate.ctx.weight.grad.abs().sum().item())
print("last gate gradient nonzero", bool(gate.query_gate[-1].weight.grad.abs().sum()>0))
""", """1. Does smaller query norm always lower every probability? No;
normalization and score signs matter.
2. Does zero early gradient mean identity gates never learn? No; last-layer
gradients open the earlier routes after updates. Summary: reason about derivatives
and complete normalized rows rather than a single channel in isolation.""")
    cells += section(13, "The complete forward pass", r"""
Compute X,R once. For each pre-normalized block H=LN1(X):
X←X+Attention(H,Gate(R,H),bias), then X←X+FFN(LN2(X)).
FFN is D→ffn_dim→D with GELU and dropout.
Class pooling prepends a learned CLS token, normalizes [CLS;X], queries only
with CLS, and attends to CLS plus real particles. It has no relation gate or bias.
Only CLS updates; final LN and D→D→10 classifier produce logits.
Mean-pooling v1 instead applies final LN to particles BEFORE masked averaging.

Particle blocks are permutation equivariant, final logits invariant, provided
both pair axes and mask are permuted consistently and dropout is disabled.""", """
cfg = Config(embed_dim=16, pairwise_embed_dim=8, ffn_dim=32,
             n_heads=2, n_layers=2, n_cls_layers=1, dropout=0)
model = GDSParT(cfg).eval()
logits, trace = model(x,u,keep,return_trace=True)
print(json.dumps(trace,indent=2))
order = torch.tensor([2,0,3,1])
permuted = model(x[:,order],u[:,order][:,:,order],keep[:,order])
torch.testing.assert_close(logits,permuted,atol=2e-6,rtol=2e-5)
print("logits permutation invariant within tolerance")
""", """1. Does CLS attention receive the pair bias? No.
2. Is the relation stream updated after a block? No.
Summary: the forward graph specifies where information is reused, changed and pooled.""")
    cells.append(cell("markdown", """# Chapter 14: Independent PyTorch reconstruction

Read `gds_part_reconstruction.py` top-to-bottom alongside these progressively
executable cells. Every module is independently implemented using PyTorch;
no production model is imported/wrapped. The source docstrings explain tensor
axes, formulas, zero initialization, masks, residual order and pooling.

The following short cells are the same sources captured in
`output/chapter_examples.json`. Tiny widths make CPU inspection quick; the
parameter audit instantiates the full report configurations. Counts come from
real modules and `sum(p.numel())`, not offsets or spare parameters."""))
    for key, source in EXAMPLES.items():
        cells.append(cell("markdown", f"### {key.removeprefix('c14_').replace('_', ' ').title()}"))
        cells.append(cell("code", source))
    cells.append(cell("markdown", """### Check your understanding — worked answers

1. Why does v1 need six blocks? The report's historical configuration uses six;
production's four legacy operation switches alone leave eight. Six gives
1,280,684 parameters; eight would give 1,677,228.
2. Why can strict state-dictionary copying test independence? It checks every
actual parameter name/shape and then compares separate forward implementations.
It does not replace independent code with a production wrapper.

Shared audit: 18,850+4,480+49,664+520+528,384+4,096+1,053,696+
128+396,544+256+17,802 = **2,074,420**. Conditioned gates add one context
projection per block and have no shared gate module.
Summary: build, inspect, test, then train — counting alone cannot prove correctness."""))
    cells += section(15, "Experiments, workflow and limitations", r"""
This last cell overfits eight fixed synthetic examples on CPU. Loss falling and
100% training accuracy establish that gradients/updates connect, not JetClass
performance. Use held-out evaluation, fixed split/preprocessing, independent seeds,
and matched budgets to compare architectures. The report's trained shared v2
is different from the per-layer conditioned default and from published ParT.
The reported v1→v2 improvement changes multiple choices, not just gates.

Run `tests.py` for shapes, exact initialization, block equivariance, invariant
logits, padding, strict production state-dict/logit parity, physics features,
numeric caveats, all parameter counts and overfitting. Run `train_demo.py` for
the standalone loss trajectory.

Finite padding is required. Masking does not remove NaNs or arbitrary overflow.
Explicit attention stores N×N weights; production SDPA/checkpointing are more
memory efficient. Nonempty jets are enforced by this teaching implementation.
Float32 reordering means symmetry/parity tests use tolerances, not exact equality.""", """
report = cpu_demo(140)
print(json.dumps(report,indent=2))
assert report["final_loss"] < report["history"][0]["loss"]
assert report["training_accuracy"] == 1.
print("Synthetic overfit passed; no physics-performance claim.")
""", """1. Does matching parameter count prove the architecture matches?
No; also check state dictionaries, forward values, masking, symmetries and gradients.
2. Is a 100% toy training accuracy a benchmark? No; the examples are memorized
and have no independent evaluation set. Summary: transparent assumptions and
reproducible tests are necessary before drawing experimental conclusions.""")
    notebook["cells"] = cells
    notebook["metadata"].update({
        "kernelspec": {"display_name": "Python 3 (PyTorch)", "language": "python", "name": "python3"},
        "language_info": {"name": "python", "version": "3.14"},
        "companion_chapters": list(range(16)),
        "companion_preserved_review_cells": len(preserved),
    })
    for i, current in enumerate(cells):
        current.setdefault("id", f"companion-{i:04d}")
    notebook["nbformat"], notebook["nbformat_minor"] = 4, 5
    path.write_text(json.dumps(notebook, ensure_ascii=False, indent=1)+"\n", encoding="utf-8")
    print(f"Preserved {len(preserved)} review cells; wrote {len(cells)} cells across 16 chapters.")


if __name__ == "__main__":
    build()
