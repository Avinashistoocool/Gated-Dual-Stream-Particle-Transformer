"""Reproduce the report's worked numbers using PyTorch and explicit formulas."""
import json
import math
from pathlib import Path

import torch
from torch.nn import functional as F

from gds_part_reconstruction import pairwise_features


def worked_examples():
    a, b = torch.tensor([1., 2.]), torch.tensor([3., 4.])
    v = torch.tensor([2., 4., 6., 8.], dtype=torch.float64)
    normalized = (v - v.mean()) / torch.sqrt(v.var(unbiased=False) + 1e-5)
    gamma = torch.tensor([1., 2., .5, 1.])
    ln = normalized * gamma + .1
    logits = torch.tensor([0., math.log(2), math.log(3)], dtype=torch.float64)
    scores = torch.tensor([1., 0.], dtype=torch.float64) / math.sqrt(2)
    values = torch.tensor([[2., 0.], [0., 4.]], dtype=torch.float64)
    weights = scores.softmax(0)
    biased = (scores + torch.tensor([0., 1.])).softmax(0)
    pt, y, phi = torch.tensor([80., 30.]), torch.tensor([.15, 0.]), torch.tensor([-.15, 0.])
    vectors = torch.stack((pt * phi.cos(), pt * phi.sin(),
                           pt * y.sinh(), pt * y.cosh()), -1)[None]
    pair = pairwise_features(vectors, torch.ones(1, 2, dtype=torch.bool))[0, 0, 1]
    target = torch.full((10,), .01, dtype=torch.float64)
    target[3] += .9
    w = torch.tensor(2., dtype=torch.float64, requires_grad=True)
    prediction = w * 3 + 1
    loss = (prediction - 5).square()
    loss.backward()
    return {
        "products": {"hadamard": (a*b).tolist(), "dot": (a@b).item(),
                     "outer": torch.outer(a, b).tolist()},
        "composition_x4": {"g_of_f": (2*4+1)**2, "f_of_g": 2*4**2+1},
        "layernorm": {"mean": v.mean().item(), "population_variance": v.var(unbiased=False).item(),
                      "epsilon": 1e-5, "normalized": normalized.tolist(),
                      "gamma": gamma.tolist(), "beta": .1, "output": ln.tolist()},
        "softmax": {"logits": logits.tolist(), "probabilities": logits.softmax(0).tolist(),
                    "shifted_by_100": (logits+100).softmax(0).tolist(),
                    "cross_entropy_class2": -logits.log_softmax(0)[2].item()},
        "attention": {"scaled_scores": scores.tolist(), "weights": weights.tolist(),
                      "weighted_values": (weights @ values).tolist(),
                      "bias_0_1_weights": biased.tolist(),
                      "half_query_gate_weights": (scores*.5).softmax(0).tolist(),
                      "bias_odds_multiplier": math.exp(1),
                      "negative_logit_suppression": {
                          "before": torch.tensor([-2., 0.]).softmax(0).tolist(),
                          "after": torch.tensor([-1., 0.]).softmax(0).tolist()}},
        "pair": {"input_order": ["px", "py", "pz", "E"], "vectors": vectors[0].tolist(),
                 "delta": math.sqrt(.15**2+.15**2), "kt": 30*math.sqrt(.15**2+.15**2),
                 "z": 30/110, "m2_analytic": 4800*(math.cosh(.15)-math.cos(.15)),
                 "log_features_float32": pair.tolist()},
        "gates": {"identity_at_zero": (1+torch.zeros(3).tanh()).tolist(),
                  "sigmoid_at_zero": torch.zeros(3).sigmoid().tolist(),
                  "masked_mean_of_2_4_and_6_8": [4., 6.]},
        "label_smoothing": {"epsilon": .1, "classes": 10, "target_class": 3,
                            "distribution": target.tolist(), "sum": target.sum().item()},
        "gradient": {"prediction": prediction.item(), "target": 5, "squared_error": loss.item(),
                     "d_loss_d_w": w.grad.item(), "learning_rate": .1,
                     "updated_w": (w-.1*w.grad).item()},
        "gelu": {"x": [-1., 0., 1.], "output": F.gelu(torch.tensor([-1., 0., 1.])).tolist()},
    }


def main():
    examples = worked_examples()
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    text = json.dumps(examples, indent=2)
    (output / "math_examples.json").write_text(text, encoding="utf-8")
    print(text)


if __name__ == "__main__":
    main()
