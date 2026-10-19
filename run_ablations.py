"""
run_ablations.py — architecture ablations for GDS-ParT on a fixed small budget.

Each variant trains with identical data/optimizer settings and is evaluated on
the same validation subset; a summary table is written to <out>/ABLATIONS.md.
Runs are resumable (skipped if <out>/<name>/results.json already exists).

    python run_ablations.py --train D:/JetClass/train_10M --val D:/JetClass/val \
        --cache_dir D:/JetClass/cache --jets_per_class 100000 --val_jets_per_class 50000 \
        --epochs 2 --out ./ablations
    python run_ablations.py ... --only v2_full,no_gate      # subset of variants
"""
import argparse
import json
import os
import subprocess
import sys

V1 = "--gate_mode sigmoid --per_layer_gate 0 --pair_bias 0 --n_cls_layers 0 --n_layers 6 " \
     "--label_smoothing 0 --ema_decay 0"
VARIANTS = {
    # name: (extra train_gds_part.py args, description)
    "v1":            (V1, "v1: shared sigmoid gate, mean pooling, 6 layers"),
    "v2_full":       ("", "v2: per-layer identity gates + pair bias + 2 CLS blocks, 8 layers"),
    "no_gate":       ("--gate_mode none", "v2 without Q/K gates (= ParT-style bias-only attention)"),
    "no_pair_bias":  ("--pair_bias 0", "v2 without additive pair bias (gates only)"),
    "shared_gate":   ("--per_layer_gate 0", "v2 with one uniform-mean gate shared by all layers"),
    "v1_plus_cls":   (V1.replace("--n_cls_layers 0", "--n_cls_layers 2"), "v1 + CLS pooling"),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--cache_dir", required=True)
    ap.add_argument("--jets_per_class", type=int, default=100_000)
    ap.add_argument("--val_jets_per_class", type=int, default=50_000)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", default="./ablations")
    ap.add_argument("--only", default=None, help="comma-separated variant names")
    ap.add_argument("--extra", default="", help="extra args passed to every run")
    args = ap.parse_args()

    names = args.only.split(",") if args.only else list(VARIANTS)
    here = os.path.dirname(os.path.abspath(__file__))
    for name in names:
        run_dir = os.path.join(args.out, name)
        if os.path.exists(os.path.join(run_dir, "results.json")):
            print(f"[skip] {name} (done)", flush=True)
            continue
        cmd = [sys.executable, os.path.join(here, "train_gds_part.py"),
               "--train", args.train, "--val", args.val, "--cache_dir", args.cache_dir,
               "--jets_per_class", str(args.jets_per_class),
               "--val_jets_per_class", str(args.val_jets_per_class),
               "--epochs", str(args.epochs), "--lr", str(args.lr),
               "--checkpoint_dir", run_dir, "--log_every", "200", "--resume"]
        cmd += VARIANTS[name][0].split() + args.extra.split()
        print(f"[run] {name}: {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, check=True)
    summarize(args.out)


def summarize(out):
    rows = []
    for name, (_, desc) in VARIANTS.items():
        p = os.path.join(out, name, "results.json")
        if not os.path.exists(p):
            continue
        r = json.load(open(p))
        v = r.get("val", {})
        rows.append((name, desc, r["n_params"], v.get("accuracy", float("nan")),
                     v.get("auc_macro_ovo", float("nan")),
                     " / ".join(f"{h['val_acc']:.4f}" for h in r["history"] if "val_acc" in h)))
    lines = ["# GDS-ParT ablations", "",
             "| Variant | Description | Params | Val acc | Val AUC | Val acc per epoch |",
             "|---|---|---|---|---|---|"]
    for n, d, p, a, auc, hist in sorted(rows, key=lambda r: -r[3]):
        lines.append(f"| {n} | {d} | {p:,} | {a:.4f} | {auc:.4f} | {hist} |")
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, "ABLATIONS.md"), "w") as f:
        f.write("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
