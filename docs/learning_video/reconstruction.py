"""Educational assembly of GDS-ParT, with executable mathematical checks.

Run from any directory with the repository's PyTorch environment:
    python docs\\learning_video\\reconstruction.py
    python docs\\learning_video\\reconstruction.py --train-demo

The component classes are deliberately imported from the production module.
The top-level assembly is explicit; it is not an independent ParT reimplementation.
"""
from __future__ import annotations

import argparse
import math
import sys
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
from data.jetclass_cache import pairwise_features
from gds_part_colab import (
    ClassAttentionBlock,
    GatedDualStreamAttention,
    GDSParT,
    GDSParTBlock,
    GDSParTConfig,
    PairwiseEmbedding,
    ParticleEmbedding,
    RelationalGate,
)


class ReconstructedGDS(nn.Module):
    """Same module names and equations as GDSParT, without activation checkpointing."""

    def __init__(self, cfg: GDSParTConfig):
        super().__init__()
        self.cfg = cfg
        self.particle_embed = ParticleEmbedding(
            cfg.n_particle_features, cfg.embed_dim, cfg.dropout)
        self.pairwise_embed = PairwiseEmbedding(
            cfg.n_pairwise_features, cfg.pairwise_embed_dim)
        self.rel_gate = (
            RelationalGate(cfg.pairwise_embed_dim, cfg.embed_dim, cfg.gate_mode)
            if not cfg.per_layer_gate and cfg.gate_mode != "none" else None
        )
        self.pair_bias = (
            nn.Linear(cfg.pairwise_embed_dim, cfg.n_heads) if cfg.pair_bias else None
        )
        self.blocks = nn.ModuleList(GDSParTBlock(cfg) for _ in range(cfg.n_layers))
        if cfg.n_cls_layers > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.cls_blocks = nn.ModuleList(
                ClassAttentionBlock(cfg) for _ in range(cfg.n_cls_layers))
        self.final_norm = nn.LayerNorm(cfg.embed_dim)
        self.classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, cfg.n_classes),
        )

    def forward(self, features, pairs, mask):
        if features.ndim != 3 or features.shape[-1] != self.cfg.n_particle_features:
            raise ValueError("features must have shape [batch, particles, configured features]")
        batch, particles, _ = features.shape
        if batch == 0 or particles == 0:
            raise ValueError("A batch must contain at least one nonempty jet")
        if pairs.shape != (batch, particles, particles, self.cfg.n_pairwise_features):
            raise ValueError("pairs must have shape [batch, particles, particles, pair features]")
        if mask.dtype != torch.bool or mask.shape != (batch, particles):
            raise ValueError("mask must be boolean [batch, particles], with True meaning real")
        if not bool(mask.any(dim=1).all()):
            raise ValueError("Empty jets are not supported: every jet needs at least one real particle")
        if features.device != pairs.device or features.device != mask.device:
            raise ValueError("features, pairs and mask must be on the same device")

        x = self.particle_embed(features)
        rel = self.pairwise_embed(pairs)
        if self.pair_bias is not None:
            attention_mask = self.pair_bias(rel).permute(0, 3, 1, 2)
            attention_mask = attention_mask.masked_fill(
                ~mask[:, None, None, :], float("-inf"))
        else:
            attention_mask = mask[:, None, None, :]
        gates = self.rel_gate(rel, mask) if self.rel_gate is not None else None
        for block in self.blocks:
            x = block(x, rel, gates, attention_mask, mask)

        if self.cfg.n_cls_layers > 0:
            cls = self.cls_token.expand(batch, -1, -1)
            for block in self.cls_blocks:
                cls = block(x, cls, mask)
            pooled = self.final_norm(cls.squeeze(1))
        else:
            x = self.final_norm(x)
            valid = mask.unsqueeze(-1).to(x.dtype)
            pooled = (x * valid).sum(dim=1) / valid.sum(dim=1)
        return self.classifier(pooled)


def build_model(tiny=False, **options):
    settings = dict(embed_dim=16, pairwise_embed_dim=8, ffn_dim=32,
                    n_heads=2, n_layers=2, dropout=0.0) if tiny else {}
    settings.update(options)
    return ReconstructedGDS(GDSParTConfig(**settings))


def tiny_batch():
    features = torch.randn(2, 5, 17)
    mask = torch.tensor([[True, True, True, True, False],
                         [True, True, False, False, False]])
    momentum = torch.randn(2, 5, 3)
    energy = torch.sqrt(momentum.square().sum(-1, keepdim=True) + .25)
    vectors = torch.cat((momentum, energy), dim=-1) * mask.unsqueeze(-1)
    features = features * mask.unsqueeze(-1)
    pairs = pairwise_features(vectors, mask)
    return features, pairs, mask, torch.tensor([2, 7])


def nonneutral_gates(model):
    for module in model.modules():
        if isinstance(module, RelationalGate):
            for mlp in (module.query_gate, module.key_gate):
                nn.init.uniform_(mlp[-1].weight, -.12, .12)
                nn.init.uniform_(mlp[-1].bias, -.08, .08)


class ReconstructionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(2026)
        self.batch = tiny_batch()

    def assertClose(self, actual, expected, **kwargs):
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-6, **kwargs)

    def variants(self):
        return [
            dict(per_layer_gate=False, gate_mode="identity", pair_bias=True, n_cls_layers=2),
            dict(per_layer_gate=True, gate_mode="identity", pair_bias=True, n_cls_layers=2),
            dict(per_layer_gate=False, gate_mode="none", pair_bias=True, n_cls_layers=0),
            dict(per_layer_gate=False, gate_mode="sigmoid", pair_bias=False, n_cls_layers=0),
            dict(per_layer_gate=True, gate_mode="identity", pair_bias=False, n_cls_layers=1),
        ]

    def test_reference_outputs_and_gradients(self):
        features, pairs, mask, labels = self.batch
        for options in self.variants():
            with self.subTest(**options):
                study = build_model(tiny=True, **options)
                nonneutral_gates(study)
                reference = GDSParT(study.cfg)
                reference.load_state_dict(study.state_dict(), strict=True)
                study.eval()
                reference.eval()
                self.assertClose(study(features, pairs, mask), reference(features, pairs, mask))
                study.train()
                reference.train()
                F.cross_entropy(study(features, pairs, mask), labels).backward()
                F.cross_entropy(reference(features, pairs, mask), labels).backward()
                original = dict(reference.named_parameters())
                for name, parameter in study.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertIsNotNone(original[name].grad, name)
                    self.assertClose(parameter.grad, original[name].grad)

    def test_padding_and_permutation(self):
        features, pairs, mask, _ = self.batch
        permutation = torch.tensor([4, 2, 0, 3, 1])
        for options in self.variants():
            with self.subTest(**options), torch.no_grad():
                model = build_model(tiny=True, **options).eval()
                nonneutral_gates(model)
                expected = model(features, pairs, mask)
                changed_f = features.clone()
                changed_f[~mask] = 100 * torch.randn_like(changed_f[~mask])
                valid_pair = mask[:, :, None] & mask[:, None, :]
                changed_u = pairs.clone()
                changed_u[~valid_pair] = 100 * torch.randn_like(changed_u[~valid_pair])
                self.assertClose(model(changed_f, changed_u, mask), expected)
                shuffled = model(features[:, permutation],
                                 pairs[:, permutation][:, :, permutation],
                                 mask[:, permutation])
                self.assertClose(shuffled, expected)
                self.assertEqual(model(features[:1], pairs[:1], mask[:1]).shape, (1, 10))

    def test_identity_gates_and_manual_summaries(self):
        _, _, mask, _ = self.batch
        rel = torch.randn(2, 5, 5, 8)
        x = torch.randn(2, 5, 16)
        for conditioned in (False, True):
            gate = RelationalGate(8, 16, mode="identity", conditioned=conditioned)
            gq, gk = gate(rel, mask, x=x)
            self.assertTrue(torch.equal(gq, torch.ones_like(gq)))
            self.assertTrue(torch.equal(gk, torch.ones_like(gk)))
            nonneutral_gates(gate)
            if conditioned:
                context = gate.ctx(x)
                scores = (rel * context[:, :, None, :]).sum(-1) / math.sqrt(8)
                scores = scores.masked_fill(~mask[:, None, :], float("-inf"))
                summary = (scores.softmax(-1)[..., None] * rel).sum(2)
            else:
                valid = mask[:, None, :, None]
                summary = (rel * valid).sum(2) / valid.sum(2)
            actual_q, actual_k = gate(rel, mask, x=x)
            self.assertClose(actual_q, 1 + gate.query_gate(summary).tanh())
            self.assertClose(actual_k, 1 + gate.key_gate(summary).tanh())

    def test_manual_gated_attention(self):
        layer = GatedDualStreamAttention(16, 2, dropout=0).eval()
        x = torch.randn(2, 5, 16)
        gq, gk = torch.rand_like(x) * 2, torch.rand_like(x) * 2
        bias = torch.randn(2, 2, 5, 5)
        bias[..., -1] = float("-inf")
        split = lambda t: t.reshape(2, 5, 2, 8).transpose(1, 2)
        q = split(layer.q_proj(x) * gq)
        k = split(layer.k_proj(x) * gk)
        v = split(layer.v_proj(x))
        weights = (q @ k.transpose(-2, -1) / math.sqrt(8) + bias).softmax(-1)
        self.assertClose(weights.sum(-1), torch.ones(2, 2, 5))
        mixed = (weights @ v).transpose(1, 2).contiguous().view(2, 5, 16)
        self.assertClose(layer(x, gq, gk, bias), layer.out_proj(mixed))

    def test_layernorm_and_gate_counterexample(self):
        x = torch.tensor([[1., 3.]])
        gamma, beta = torch.tensor([2., 3.]), torch.tensor([.5, -.5])
        mean = x.mean(-1, keepdim=True)
        variance = (x - mean).square().mean(-1, keepdim=True)
        manual = gamma * (x - mean) / torch.sqrt(variance + 1e-5) + beta
        self.assertClose(manual, F.layer_norm(x, (2,), gamma, beta, 1e-5))
        first = torch.tensor([-1., 0.]).softmax(0)[0].item()
        amplified = torch.tensor([-2., 0.]).softmax(0)[0].item()
        self.assertAlmostEqual(first, .2689414, places=6)
        self.assertAlmostEqual(amplified, .1192029, places=6)
        self.assertLess(amplified, first)

    def test_physical_pair_features(self):
        vectors = torch.tensor([[[20., 0., 0., 20.],
                                 [5*math.cos(.2), 5*math.sin(.2), 0., 5.],
                                 [0., 0., 0., 0.]]])
        mask = torch.tensor([[True, True, False]])
        pairs = pairwise_features(vectors, mask)
        expected = torch.tensor([math.log(.2), 0., math.log(.2),
                                 math.log(200 * (1-math.cos(.2)))])
        self.assertClose(pairs[0, 0, 1], expected)
        self.assertClose(pairs, pairs.transpose(1, 2))
        self.assertEqual(torch.count_nonzero(pairs[:, 2]).item(), 0)
        self.assertEqual(torch.count_nonzero(pairs.diagonal(dim1=1, dim2=2)).item(), 0)

    def test_parameter_counts(self):
        shared = build_model(per_layer_gate=False)
        conditioned = build_model()
        self.assertEqual(sum(p.numel() for p in shared.parameters()), 2074420)
        self.assertEqual(sum(p.numel() for p in conditioned.parameters()), 2488116)

    def test_invalid_inputs(self):
        features, pairs, mask, _ = self.batch
        model = build_model(tiny=True)
        with self.assertRaisesRegex(ValueError, "Empty jets"):
            model(features, pairs, torch.zeros_like(mask))
        with self.assertRaisesRegex(ValueError, "boolean"):
            model(features, pairs, mask.float())
        with self.assertRaisesRegex(ValueError, "pairs"):
            model(features, pairs[:, :-1], mask)
        with self.assertRaisesRegex(ValueError, "features"):
            model(features[..., :-1], pairs, mask)


def train_demo():
    torch.manual_seed(42)
    model = build_model(tiny=True, per_layer_gate=False).train()
    features, pairs, mask, labels = tiny_batch()
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-3)
    losses = []
    initial = next(model.parameters()).detach().clone()
    for step in range(40):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(features, pairs, mask), labels)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Nonfinite loss at step {step}")
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
    if torch.equal(initial, next(model.parameters()).detach()):
        raise RuntimeError("The optimizer did not change parameters")
    if losses[-1] >= losses[0]:
        raise RuntimeError(f"Tiny fixed-batch loss did not decrease: {losses[0]} -> {losses[-1]}")
    print(f"Synthetic plumbing demo only: loss {losses[0]:.4f} -> {losses[-1]:.4f}")
    print("This is not a real-data performance result.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-demo", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(2)
    if args.train_demo:
        train_demo()
    else:
        suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReconstructionChecks)
        result = unittest.TextTestRunner(verbosity=2).run(suite)
        raise SystemExit(0 if result.wasSuccessful() else 1)
