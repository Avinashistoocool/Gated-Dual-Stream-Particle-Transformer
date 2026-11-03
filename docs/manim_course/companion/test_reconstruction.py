"""Numerical, structural and behavioural checks; no physics performance claim."""
import json
import math
import sys
import unittest
from dataclasses import replace
from pathlib import Path
import torch
from torch.nn import functional as F
from gds_part_reconstruction import (
    Config, GDSParT, RelationalGate, configurations, pair_features, GDSParTBlock,
    ParticleEmbedding, PairwiseEmbedding, GatedAttention, ClassAttentionBlock,
    parameter_audit, shared_breakdown, cpu_demo,
)

_parents = Path(__file__).resolve().parents
_reference_candidates = [_parents[1]] + ([_parents[3]] if len(_parents) > 3 else [])
ROOT = next((p for p in _reference_candidates if (p / "gds_part_colab.py").is_file()
             and (p / "data" / "jetclass_cache.py").is_file()), None)
HAS_REFERENCE = ROOT is not None
REFERENCE_REASON = "Optional production references are not included in the standalone course bundle."
OUT = Path(__file__).resolve().parent / "output"
torch.set_num_threads(2)


class ReconstructionChecks(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(17)
        self.cfg = Config(embed_dim=16, pairwise_embed_dim=8, ffn_dim=32, n_heads=2,
                          n_layers=2, dropout=0, n_cls_layers=1)
        self.x, self.u = torch.randn(2, 4, 17), torch.randn(2, 4, 4, 4)
        self.keep = torch.tensor([[True, True, True, False], [True, True, False, False]])

    def test_exact_counts_and_breakdown(self):
        counts = parameter_audit()
        self.assertEqual(counts, {"v1": 1280684, "shared": 2074420,
                                 "conditioned": 2488116, "no_gates": 2024756})
        parts = shared_breakdown()
        self.assertEqual(list(parts.values()), [18850, 4480, 49664, 520, 528384, 4096,
                                                1053696, 128, 396544, 256, 17802])
        eight = replace(configurations()["v1"], n_layers=8)
        self.assertEqual(sum(p.numel() for p in GDSParT(eight).parameters()), 1677228)
        OUT.mkdir(exist_ok=True)
        (OUT / "parameter_audit.json").write_text(json.dumps(
            {"counts": counts, "shared_breakdown": parts}, indent=2), encoding="utf-8")

    def test_all_module_shapes_and_backward(self):
        for cfg in (self.cfg, replace(self.cfg, per_layer_gate=False),
                    replace(self.cfg, n_cls_layers=0), replace(self.cfg, gate_mode="none")):
            model = GDSParT(cfg)
            logits, trace = model(self.x, self.u, self.keep, return_trace=True)
            self.assertEqual(tuple(logits.shape), (2, 10))
            self.assertEqual(trace["relations"], [2, 4, 4, 8])
            self.assertEqual(trace["blocks"], [[2, 4, 16], [2, 4, 16]])
            F.cross_entropy(logits, torch.tensor([0, 1])).backward()
            gradients = [p.grad for p in model.parameters() if p.grad is not None]
            self.assertTrue(all(torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(float(g.abs().sum()) for g in gradients), 0)

    def test_individual_module_shapes(self):
        x = ParticleEmbedding(17, 16, 0)(self.x)
        r = PairwiseEmbedding(4, 8)(self.u)
        self.assertEqual(x.shape, (2, 4, 16))
        self.assertEqual(r.shape, (2, 4, 4, 8))
        for conditioned in (False, True):
            gate = RelationalGate(8, 16, conditioned=conditioned)
            summary = gate.aggregate(r, self.keep, x)
            self.assertEqual(summary.shape, (2, 4, 8))
            gates = gate(r, self.keep, x)
            self.assertEqual(gates[0].shape, (2, 4, 16))
            self.assertEqual(gates[1].shape, (2, 4, 16))
            bias = torch.nn.Linear(8, 2)(r).permute(0, 3, 1, 2)
            self.assertEqual(bias.shape, (2, 2, 4, 4))
            y, weights = GatedAttention(16, 2, 0)(x, gates, bias, self.keep, True)
            self.assertEqual(y.shape, x.shape)
            self.assertEqual(weights.shape, (2, 2, 4, 4))
        block = GDSParTBlock(self.cfg)
        y = block(x, r, None, bias, self.keep)
        self.assertEqual(y.shape, x.shape)
        cls = ClassAttentionBlock(self.cfg)(y, torch.zeros(2, 1, 16), self.keep)
        self.assertEqual(cls.shape, (2, 1, 16))

    def test_identity_gates_and_attention(self):
        model = GDSParT(self.cfg).eval()
        rel = model.pairwise_embed(self.u)
        normalized = model.blocks[0].norm1(model.particle_embed(self.x))
        gate = model.blocks[0].gate
        gq, gk = gate(rel, self.keep, normalized)
        torch.testing.assert_close(gq, torch.ones_like(gq), rtol=0, atol=0)
        torch.testing.assert_close(gk, torch.ones_like(gk), rtol=0, atol=0)
        attn = model.blocks[0].attn
        torch.testing.assert_close(attn(normalized, (gq, gk), None, self.keep),
                                   attn(normalized, None, None, self.keep))
        _, weights = attn(normalized, None, None, self.keep, True)
        torch.testing.assert_close(weights.sum(-1), torch.ones_like(weights.sum(-1)))
        self.assertTrue((weights.masked_select(~self.keep[:, None, None, :]) == 0).all())
        for conditioned in (False, True):
            for mode in ("identity", "sigmoid"):
                module = RelationalGate(8, 16, mode=mode, conditioned=conditioned)
                gq, gk = module(rel, self.keep, normalized)
                if mode == "identity":
                    self.assertTrue(torch.equal(gq, torch.ones_like(gq)))
                    self.assertTrue(torch.equal(gk, torch.ones_like(gk)))
                else:
                    self.assertTrue(((gq > 0) & (gq < 1)).all())
                    self.assertTrue(((gk > 0) & (gk < 1)).all())

    def test_masked_mean_and_conditioning(self):
        relations = torch.tensor([[[[2., 4.], [6., 8.], [900., 900.]]]])
        keep = torch.tensor([[True, True, False]])
        gate = RelationalGate(2, 4)
        torch.testing.assert_close(gate.aggregate(relations, keep), torch.tensor([[[4., 6.]]]))
        gate = RelationalGate(2, 4, conditioned=True)
        torch.nn.init.zeros_(gate.ctx.weight)
        torch.nn.init.zeros_(gate.ctx.bias)
        torch.testing.assert_close(gate.aggregate(relations, keep, torch.zeros(1, 1, 4)),
                                   torch.tensor([[[4., 6.]]]))

    def test_permutation_and_padding_invariance(self):
        variants = [
            replace(self.cfg, n_cls_layers=cls_layers, **switches)
            for cls_layers in (0, 1)
            for switches in ({}, {"per_layer_gate": False}, {"gate_mode": "none"},
                             {"gate_mode": "sigmoid", "per_layer_gate": False,
                              "pair_bias": False})
        ]
        for cfg in variants:
            model = GDSParT(cfg).eval()
            # Move identity gates away from 1 so this tests actual relation routing.
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, RelationalGate):
                        torch.nn.init.normal_(module.query_gate[-1].weight, std=.1)
                        torch.nn.init.normal_(module.key_gate[-1].weight, std=.1)
            original = model(self.x, self.u, self.keep)
            order = torch.tensor([2, 0, 3, 1])
            reordered = model(self.x[:, order], self.u[:, order][:, :, order], self.keep[:, order])
            torch.testing.assert_close(original, reordered, atol=2e-6, rtol=2e-5)
            x, u = self.x.clone(), self.u.clone()
            x[~self.keep] = 100
            valid_pairs = self.keep[:, :, None] & self.keep[:, None, :]
            u[~valid_pairs] = -75
            torch.testing.assert_close(original, model(x, u, self.keep), atol=2e-6, rtol=2e-5)
            padded_x = F.pad(self.x, (0, 0, 0, 2), value=30)
            padded_u = F.pad(self.u, (0, 0, 0, 2, 0, 2), value=20)
            padded_mask = F.pad(self.keep, (0, 2), value=False)
            torch.testing.assert_close(original, model(padded_x, padded_u, padded_mask),
                                       atol=2e-6, rtol=2e-5)
            for b in range(2):
                real = self.keep[b]
                cropped = model(self.x[b:b+1, real], self.u[b:b+1, real][:, :, real])
                torch.testing.assert_close(original[b:b+1], cropped, atol=2e-6, rtol=2e-5)

    def test_block_permutation_equivariance(self):
        order = torch.tensor([2, 0, 3, 1])
        for switches in ({}, {"per_layer_gate": False}, {"gate_mode": "none"}):
            model = GDSParT(replace(self.cfg, **switches)).eval()
            with torch.no_grad():
                for module in model.modules():
                    if isinstance(module, RelationalGate):
                        torch.nn.init.normal_(module.query_gate[-1].weight, std=.1)
                        torch.nn.init.normal_(module.key_gate[-1].weight, std=.1)
            x, r = model.particle_embed(self.x), model.pairwise_embed(self.u)
            gates = model.rel_gate(r, self.keep) if model.rel_gate else None
            bias = model.pair_bias(r).permute(0, 3, 1, 2)
            y = model.blocks[0](x, r, gates, bias, self.keep)
            permuted_gates = tuple(g[:, order] for g in gates) if gates else None
            yp = model.blocks[0](x[:, order], r[:, order][:, :, order],
                                 permuted_gates, bias[:, :, order][:, :, :, order],
                                 self.keep[:, order])
            torch.testing.assert_close(yp, y[:, order], atol=2e-6, rtol=2e-5)

    def test_identity_initialization_gradient_routes(self):
        gate = RelationalGate(8, 16, conditioned=True)
        r = torch.randn(2, 4, 4, 8)
        x = torch.randn(2, 4, 16)
        gq, gk = gate(r, self.keep, x)
        (gq.sum() + gk.sum()).backward()
        self.assertEqual(gate.ctx.weight.grad.abs().sum(), 0)
        self.assertEqual(gate.query_gate[0].weight.grad.abs().sum(), 0)
        self.assertGreater(gate.query_gate[-1].weight.grad.abs().sum(), 0)
        self.assertGreater(gate.key_gate[-1].weight.grad.abs().sum(), 0)

    def test_input_guards_and_static_relation(self):
        model = GDSParT(self.cfg).eval()
        with self.assertRaises(ValueError):
            model(self.x, self.u, torch.zeros_like(self.keep))
        calls = []
        handle = model.pairwise_embed.register_forward_hook(lambda *args: calls.append(1))
        model(self.x, self.u, self.keep)
        handle.remove()
        self.assertEqual(len(calls), 1)

    def test_layernorm_softmax_and_smoothing(self):
        x = torch.tensor([2., 4., 6., 8.])
        self.assertEqual(x.mean(), 5)
        self.assertEqual(x.var(unbiased=False), 5)
        normalized = (x - 5) / math.sqrt(5 + 1e-5)
        gamma = torch.tensor([1., 2., .5, 1.])
        expected = normalized * gamma + .1
        torch.testing.assert_close(F.layer_norm(x, (4,), gamma, torch.full((4,), .1)), expected)
        torch.testing.assert_close(expected, torch.tensor([-1.242, -.794, .324, 1.442]),
                                   atol=.0005, rtol=0)
        logits = torch.tensor([0., math.log(2), math.log(3)], dtype=torch.float64)
        torch.testing.assert_close(logits.softmax(0), torch.tensor([1/6, 2/6, 3/6], dtype=torch.float64))
        torch.testing.assert_close(logits.softmax(0), (logits + 100).softmax(0))
        target = torch.full((10,), .01)
        target[3] += .9
        self.assertAlmostEqual(float(target[3]), .91, places=6)
        z = torch.randn(1, 10)
        torch.testing.assert_close(F.cross_entropy(z, torch.tensor([3]), label_smoothing=.1),
                                   -(target * z.log_softmax(-1)).sum())

    def test_attention_bias_gate_numbers(self):
        q, k = torch.tensor([1., 0.]), torch.eye(2)
        scores = k @ q / math.sqrt(2)
        weights = scores.softmax(0)
        torch.testing.assert_close(weights, torch.tensor([.6698, .3302]), atol=.0001, rtol=0)
        torch.testing.assert_close(weights @ torch.tensor([[2., 0.], [0., 4.]]),
                                   torch.tensor([1.3395, 1.3210]), atol=.0001, rtol=0)
        torch.testing.assert_close((scores + torch.tensor([0., 1.])).softmax(0),
                                   torch.tensor([.4273, .5727]), atol=.0001, rtol=0)
        torch.testing.assert_close((scores * .5).softmax(0), torch.tensor([.5875, .4125]),
                                   atol=.0001, rtol=0)
        self.assertLess(float(torch.tensor([-2., 0.]).softmax(0)[0]),
                        float(torch.tensor([-1., 0.]).softmax(0)[0]))

    def test_pair_features_report_example(self):
        pt = torch.tensor([80., 30.])
        y, phi = torch.tensor([.15, 0.]), torch.tensor([-.15, 0.])
        v = torch.stack((pt * phi.cos(), pt * phi.sin(), pt * y.sinh(), pt * y.cosh()), -1)[None]
        keep = torch.ones(1, 2, dtype=torch.bool)
        result = pair_features(v, keep)
        torch.testing.assert_close(result[0, 0, 1], torch.tensor([-1.5505, 1.8507, -1.2993, 4.6821]),
                                   atol=.0001, rtol=0)
        self.assertTrue((result[0, torch.arange(2), torch.arange(2)] == 0).all())
        torch.testing.assert_close(result, result.transpose(1, 2))

    @unittest.skipUnless(HAS_REFERENCE, REFERENCE_REASON)
    def test_pair_features_against_cache(self):
        sys.path.insert(0, str(ROOT))
        from data.jetclass_cache import pairwise_features
        vectors = torch.randn(2, 4, 4)
        vectors[..., 3] = vectors[..., :3].square().sum(-1).sqrt() + 1
        torch.testing.assert_close(pair_features(vectors, self.keep),
                                   pairwise_features(vectors, self.keep), rtol=0, atol=0)
        vectors = torch.tensor([[[1., 0., 0., 2.], [float("nan"), 0., 0., 0.]]])
        keep = torch.tensor([[True, False]])
        torch.testing.assert_close(pair_features(vectors, keep), pairwise_features(vectors, keep),
                                   rtol=0, atol=0, equal_nan=True)

    def test_numerical_masking_caveats_are_not_hidden(self):
        vectors = torch.tensor([[[1., 0., 0., 2.], [float("nan"), 0., 0., 0.]]])
        keep = torch.tensor([[True, False]])
        ours = pair_features(vectors, keep)
        self.assertTrue(torch.isnan(ours).any())
        vectors = torch.tensor([[[1e30, 0., 0., 1e30], [1., 0., 0., 2.]]])
        self.assertFalse(torch.isfinite(pair_features(vectors, torch.ones_like(keep))).all())
        gate = RelationalGate(2, 4)
        relations = torch.zeros(1, 2, 2, 2)
        relations[:, :, 1] = float("nan")
        self.assertTrue(torch.isnan(gate.aggregate(relations, keep)).any())
        model = GDSParT(self.cfg).eval()
        x = self.x.clone()
        x[~self.keep] = float("nan")
        with self.assertRaisesRegex(ValueError, "finite"):
            model(x, self.u, self.keep)
        # A low-level all-masked explicit softmax is undefined. The full model
        # rejects empty jets rather than depending on backend-specific SDPA output.
        attn = GatedAttention(4, 2, 0)
        out = attn(torch.zeros(1, 2, 4), None, None, torch.zeros_like(keep))
        self.assertTrue(torch.isnan(out).all())

    @unittest.skipUnless(HAS_REFERENCE, REFERENCE_REASON)
    def test_matches_production_on_identical_weights(self):
        sys.path.insert(0, str(ROOT))
        from gds_part_colab import GDSParT as Production, GDSParTConfig
        for overrides in ({}, {"per_layer_gate": False}, {"gate_mode": "none"},
                          {"gate_mode": "sigmoid", "per_layer_gate": False,
                           "pair_bias": False, "n_cls_layers": 0}):
            cfg = replace(self.cfg, **overrides)
            rebuilt = GDSParT(cfg).eval()
            production = Production(GDSParTConfig(**vars(cfg))).eval()
            production.load_state_dict(rebuilt.state_dict(), strict=True)
            torch.testing.assert_close(rebuilt(self.x, self.u, self.keep),
                                       production(self.x, self.u, self.keep), atol=2e-6, rtol=2e-5)
            with torch.no_grad():
                for module in rebuilt.modules():
                    if isinstance(module, RelationalGate):
                        torch.nn.init.normal_(module.query_gate[-1].weight, std=.1)
                        torch.nn.init.normal_(module.key_gate[-1].weight, std=.1)
            production.load_state_dict(rebuilt.state_dict(), strict=True)
            torch.testing.assert_close(rebuilt(self.x, self.u, self.keep),
                                       production(self.x, self.u, self.keep), atol=2e-6, rtol=2e-5)

    @unittest.skipUnless(HAS_REFERENCE, REFERENCE_REASON)
    def test_production_parameter_counts_and_legacy_depth(self):
        sys.path.insert(0, str(ROOT))
        from gds_part_colab import GDSParT as Production, GDSParTConfig
        expected = {"v1": 1280684, "shared": 2074420,
                    "conditioned": 2488116, "no_gates": 2024756}
        for name, cfg in configurations().items():
            source = Production(GDSParTConfig(**vars(cfg)))
            self.assertEqual(sum(p.numel() for p in source.parameters()), expected[name])

    @unittest.skipUnless(HAS_REFERENCE, REFERENCE_REASON)
    def test_input_gradients_match_production(self):
        sys.path.insert(0, str(ROOT))
        from gds_part_colab import GDSParT as Production, GDSParTConfig
        for cfg in (self.cfg, replace(self.cfg, per_layer_gate=False)):
            rebuilt, source = GDSParT(cfg).eval(), Production(GDSParTConfig(**vars(cfg))).eval()
            with torch.no_grad():
                for module in rebuilt.modules():
                    if isinstance(module, RelationalGate):
                        torch.nn.init.normal_(module.query_gate[-1].weight, std=.1)
                        torch.nn.init.normal_(module.key_gate[-1].weight, std=.1)
            source.load_state_dict(rebuilt.state_dict())
            xa, ua = self.x.clone().requires_grad_(), self.u.clone().requires_grad_()
            xb, ub = self.x.clone().requires_grad_(), self.u.clone().requires_grad_()
            rebuilt(xa, ua, self.keep).square().sum().backward()
            source(xb, ub, self.keep).square().sum().backward()
            torch.testing.assert_close(xa.grad, xb.grad, atol=2e-6, rtol=2e-4)
            torch.testing.assert_close(ua.grad, ub.grad, atol=2e-6, rtol=2e-4)
            for name, parameter in rebuilt.named_parameters():
                reference = dict(source.named_parameters())[name]
                torch.testing.assert_close(parameter.grad, reference.grad, atol=2e-6, rtol=2e-4)

    def test_cpu_overfit(self):
        report = cpu_demo()
        self.assertEqual(report["training_accuracy"], 1.)
        self.assertLess(report["final_loss"], .05)
        OUT.mkdir(exist_ok=True)
        (OUT / "cpu_demo.json").write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(ReconstructionChecks)
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    OUT.mkdir(exist_ok=True)
    (OUT / "reconstruction_tests.json").write_text(json.dumps(
        {"tests": result.testsRun, "failures": len(result.failures), "errors": len(result.errors),
         "skipped": len(result.skipped), "skip_reasons": [reason for _, reason in result.skipped],
         "success": result.wasSuccessful()}, indent=2), encoding="utf-8")
    sys.exit(not result.wasSuccessful())
