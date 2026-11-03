"""Numerical checks for the exact Chapters 2-5 calculation and animation data."""
import json
import math
import contextlib
import io
from pathlib import Path
import torch
from course_content import CHAPTERS, CODE


def main():
    checks = []
    def close(name, actual, expected, tolerance=1e-6):
        torch.testing.assert_close(torch.as_tensor(actual, dtype=torch.float64),
                                   torch.as_tensor(expected, dtype=torch.float64),
                                   atol=tolerance, rtol=tolerance)
        checks.append(name)

    x = torch.arange(2*3*128).reshape(2, 3, 128)
    split = x.reshape(2, 3, 8, 16).transpose(1, 2)
    close("split-head reshape preserves entries", split.transpose(1, 2).reshape_as(x), x)
    close("masked mean counts real particles", (2+6)/(1+1), 4.)
    logits = torch.tensor([0., math.log(2), math.log(3)], dtype=torch.float64)
    close("exact three-way softmax", logits.softmax(0), [1/6, 2/6, 3/6])
    close("constant-shift invariance", (logits+100).softmax(0), logits.softmax(0))
    for temperature, bars in zip((1., .5, 2.), CHAPTERS[3]["temperature"].values):
        scores = torch.tensor([.2, .3, .5], dtype=torch.float64).log()/temperature
        close(f"temperature animation T={temperature}", bars, scores.softmax(0), tolerance=2e-6)
    close("correct prediction still incurs loss", -math.log(.51), .6733445532637656)
    matrix = torch.tensor([[1., 3.], [-2., 4.]])
    close("affine expanded dot products", matrix @ torch.tensor([2., -1.]) + torch.tensor([1., -1.]),
          [0., -9.])
    close("GELU illustrative values", torch.nn.functional.gelu(torch.tensor([.5, -.5])),
          [.3457312306, -.1542687694])
    close("affine parameter count", 17*128+128, 2304)
    weight, bias = torch.tensor(1., requires_grad=True), torch.tensor(0., requires_grad=True)
    loss = (weight*2+bias-5).square()/2
    loss.backward()
    close("scalar weight gradient", weight.grad, -6.)
    close("scalar bias gradient", bias.grad, -3.)
    with torch.no_grad():
        weight -= .1*weight.grad
        bias -= .1*bias.grad
    close("updated prediction", weight*2+bias, 3.5)
    close("updated squared-error loss", (weight*2+bias-5).square()/2, 1.125)
    namespace = {}
    with contextlib.redirect_stdout(io.StringIO()):
        exec(CODE["gradient"], namespace)
    weight, bias = namespace["w"], namespace["b"]
    assert weight.requires_grad and bias.requires_grad and weight.is_leaf and bias.is_leaf
    checks.append("displayed update preserves trainable leaf tensors")
    weight.grad, bias.grad = None, None
    ((2*weight+bias-5).square()/2).backward()
    close("second weight gradient after displayed update", weight.grad, -3.)
    close("second bias gradient after displayed update", bias.grad, -1.5)
    output = Path(__file__).resolve().parent / "output" / "foundations_validation.json"
    output.write_text(json.dumps({"success": True, "checks": checks, "count": len(checks)}, indent=2),
                      encoding="utf-8")
    print(f"PASS: {len(checks)} foundations calculation and animation checks.")


if __name__ == "__main__":
    main()
