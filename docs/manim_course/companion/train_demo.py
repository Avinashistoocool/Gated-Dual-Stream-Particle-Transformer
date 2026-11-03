"""Reproducible CPU plumbing check, not a JetClass training or validation result."""
import argparse
import json
from pathlib import Path

from gds_part_reconstruction import cpu_demo


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=140)
    args = parser.parse_args()
    report = cpu_demo(args.steps)
    report["initial_loss"] = report["history"][0]["loss"]
    report["loss_fell"] = report["final_loss"] < report["initial_loss"]
    output = Path(__file__).resolve().parent / "output"
    output.mkdir(exist_ok=True)
    (output / "cpu_demo.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not report["loss_fell"]:
        raise SystemExit("Synthetic training loss did not fall.")


if __name__ == "__main__":
    main()
