"""Short sequential chapter-14 cells with captured, never transcribed, stdout.

Run cells in insertion order. Each cell builds on named tensors from the previous
cells. The model's implementation remains in gds_part_reconstruction.py; these
cells expose every module boundary rather than hiding construction in one call.
"""
import contextlib
import io
import json
from pathlib import Path


EXAMPLES = {
    "c14_imports": """import torch
from gds_part_reconstruction import (
    Config, ParticleEmbedding,
    PairwiseEmbedding, RelationalGate,
    GatedAttention, GDSParTBlock,
    ClassAttentionBlock, GDSParT,
    parameter_audit, shared_breakdown)
torch.manual_seed(17)
torch.set_num_threads(2)""",
    "c14_inputs": """x = torch.randn(2, 4, 17)
u = torch.randn(2, 4, 4, 4)
keep = torch.tensor([[1, 1, 1, 0],
                     [1, 1, 0, 0]]).bool()
print(tuple(x.shape))
print(tuple(u.shape))
print(keep.sum(1).tolist())""",
    "c14_particle": """embed = ParticleEmbedding(17, 16, 0)
states = embed(x)
print(tuple(states.shape))""",
    "c14_pair": """pair_embed = PairwiseEmbedding(4, 8)
relations = pair_embed(u)
print(tuple(relations.shape))""",
    "c14_shared_gate": """gate = RelationalGate(8, 16)
gq, gk = gate(relations, keep)
print(tuple(gq.shape), tuple(gk.shape))
print(torch.equal(gq, torch.ones_like(gq)))
print(torch.equal(gk, torch.ones_like(gk)))""",
    "c14_conditioned": """gate = RelationalGate(8, 16,
                      conditioned=True)
norm = torch.nn.LayerNorm(16)
gq, gk = gate(relations, keep, norm(states))
print(tuple(gq.shape))
print(tuple(gate.ctx.weight.shape))""",
    "c14_bias": """bias_layer = torch.nn.Linear(8, 2)
bias = bias_layer(relations)
bias = bias.permute(0, 3, 1, 2)
print(tuple(bias.shape))""",
    "c14_attention": """attn = GatedAttention(16, 2, 0)
mixed, weights = attn(
    states, (gq, gk), bias, keep, True)
print(tuple(mixed.shape))
print(tuple(weights.shape))
print(weights[0, 0, 0, 3].item())""",
    "c14_block": """cfg = Config(embed_dim=16, n_heads=2,
    pairwise_embed_dim=8, ffn_dim=32,
    n_layers=2, n_cls_layers=1, dropout=0)
block = GDSParTBlock(cfg)
updated = block(
    states, relations, None, bias, keep)
print(tuple(updated.shape))""",
    "c14_cls": """cls = torch.zeros(2, 1, 16)
pool = ClassAttentionBlock(cfg)
cls = pool(updated, cls, keep)
print(tuple(cls.shape))""",
    "c14_logits": """model = GDSParT(cfg).eval()
logits, trace = model(
    x, u, keep, return_trace=True)
print(tuple(logits.shape))
print(trace["relation_evaluations"])
print(trace["blocks"])""",
    "c14_backward": """labels = torch.tensor([0, 1])
loss = torch.nn.functional.cross_entropy(
    logits, labels)
loss.backward()
grad = model.classifier[-1].weight.grad
print(bool(torch.isfinite(grad).all()))
print(bool(grad.abs().sum() > 0))""",
    "c14_counts": """for name, count in parameter_audit().items():
    print(name, count)""",
    "c14_breakdown": """for name, count in shared_breakdown().items():
    print(name, count)
print("total", sum(shared_breakdown().values()))""",
}


def execute_examples():
    namespace, records = {}, {}
    for key, source in EXAMPLES.items():
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(source, f"<{key}>", "exec"), namespace)
        records[key] = {"source": source, "stdout": buffer.getvalue()}
    return records


def main():
    records = execute_examples()
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    (output / "chapter_examples.json").write_text(
        json.dumps(records, indent=2), encoding="utf-8")
    for key, record in records.items():
        print(f"[{key}]\n{record['stdout']}", end="")


if __name__ == "__main__":
    main()
