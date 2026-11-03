"""One entry point for foundation, reconstruction, physics and learning tests."""
import contextlib
import io
import json
import sys
import unittest
from pathlib import Path

import torch
from test_reconstruction import ReconstructionChecks
from chapter_examples import execute_examples as execute_reconstruction_examples
from math_examples import worked_examples

COURSE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(COURSE))
from content import CODE


def execute_examples():
    outputs = {}
    for key, source in CODE.items():
        buffer = io.StringIO()
        with contextlib.redirect_stdout(buffer):
            exec(compile(source, f"<chapter-01-{key}>", "exec"), {})
        outputs[key] = buffer.getvalue()
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    (output / "executed_examples.json").write_text(
        json.dumps(outputs, indent=2), encoding="utf-8")
    return outputs


class FoundationChecks(unittest.TestCase):
    def test_report_example(self):
        a, b = torch.tensor([1.,2.]), torch.tensor([3.,4.])
        torch.testing.assert_close(a*b, torch.tensor([3.,8.]))
        self.assertEqual(torch.dot(a,b).item(), 11)
        torch.testing.assert_close(torch.outer(a,b), torch.tensor([[3.,4.],[6.,8.]]))
        self.assertEqual((a*b).shape, (2,))
        self.assertEqual(torch.dot(a,b).shape, ())
        self.assertEqual(torch.outer(a,b).shape, (2,2))

    def test_appendix_i(self):
        a, b = torch.tensor([2.,-1.]), torch.tensor([3.,4.])
        torch.testing.assert_close(a*b, torch.tensor([6.,-4.]))
        self.assertEqual(torch.dot(a,b).item(), 2)
        torch.testing.assert_close(torch.outer(a,b), torch.tensor([[6.,8.],[-3.,-4.]]))

    def test_functions_and_reductions(self):
        x = torch.tensor(4.)
        self.assertEqual((2*x+3).item(), 11)
        f, g = lambda x: 2*x+1, lambda x: x*x
        self.assertEqual(g(f(x)).item(), 81)
        self.assertEqual(f(g(x)).item(), 33)
        v = torch.tensor([2.,4.,6.,8.])
        self.assertEqual(v.sum().item(), 20)
        self.assertEqual(v.mean().item(), 5)

    def test_every_displayed_code_block_executes(self):
        outputs = execute_examples()
        self.assertEqual(len(outputs), 7)
        self.assertIn("11.", outputs["scalar"])
        self.assertIn("torch.Size([])", outputs["dot"])

    def test_progressive_reconstruction_cells(self):
        examples = execute_reconstruction_examples()
        self.assertEqual(len(examples), 14)
        self.assertIn("True\nTrue", examples["c14_shared_gate"]["stdout"])
        self.assertIn("(2, 10)", examples["c14_logits"]["stdout"])
        self.assertIn("total 2074420", examples["c14_breakdown"]["stdout"])
        output = Path(__file__).resolve().parent / "output"
        output.mkdir(exist_ok=True)
        (output / "chapter_examples.json").write_text(json.dumps(examples, indent=2), encoding="utf-8")

    def test_reproducible_report_math_artifact(self):
        values = worked_examples()
        self.assertEqual(values["products"]["dot"], 11)
        self.assertEqual(values["layernorm"]["population_variance"], 5)
        self.assertAlmostEqual(values["attention"]["weights"][0], .6697615493, places=9)
        self.assertAlmostEqual(values["pair"]["m2_analytic"], 108.000151875, places=8)
        self.assertEqual(values["gradient"]["d_loss_d_w"], 12)
        self.assertAlmostEqual(values["label_smoothing"]["distribution"][3], .91, places=12)
        output = Path(__file__).resolve().parent / "output"
        output.mkdir(exist_ok=True)
        (output / "math_examples.json").write_text(json.dumps(values, indent=2), encoding="utf-8")


if __name__ == "__main__":
    suite = unittest.defaultTestLoader.loadTestsFromModule(sys.modules[__name__])
    result = unittest.TextTestRunner(verbosity=2).run(suite)
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    (output / "tests.json").write_text(json.dumps({
        "tests": result.testsRun, "failures": len(result.failures),
        "errors": len(result.errors), "success": result.wasSuccessful(),
        "skipped": len(result.skipped), "skip_reasons": [reason for _, reason in result.skipped],
        "python": sys.version, "torch": torch.__version__,
    }, indent=2), encoding="utf-8")
    sys.exit(not result.wasSuccessful())
