"""
make_report.py — turn a finished training run into presentable artifacts.

Reads <run_dir>/results.json and <run_dir>/test_predictions.npz written by
train_gds_part.py and produces, in <run_dir>/report/:
    training_curves.png    loss / accuracy / AUC per epoch
    confusion_matrix.png   row-normalized test confusion matrix
    roc_curves.png         signal-vs-QCD ROC curves (rejection vs efficiency)
    comparison.png         GDS-ParT vs published JetClass results
    RESULTS.md             tables (overall, per-class rejection, vs ParT paper)

Usage:
    python make_report.py --run_dir ./checkpoints/jetclass_10M
"""
import argparse
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Published JetClass test-set results, Qu, Li & Qian, "Particle Transformer
# for Jet Tagging", ICML 2022 (arXiv:2202.03772), Tables 1 and 3.
# Columns: accuracy, AUC, then Rej at the per-class signal efficiency.
REJ_COLS = ["Hbb_Rej50", "Hcc_Rej50", "Hgg_Rej50", "H4q_Rej50", "Hqql_Rej99",
            "Tbqq_Rej50", "Tbl_Rej99.5", "Wqq_Rej50", "Zqq_Rej50"]
PAPER = {
    # name: (train jets, params, acc, auc, rejections...)
    "PFN (100M)":         ("100M", "86.1k", 0.772, 0.9714, [2924, 841, 75, 198, 265, 797, 721, 189, 159]),
    "P-CNN (100M)":       ("100M", "354k", 0.809, 0.9789, [4890, 1276, 88, 474, 947, 2907, 2304, 241, 204]),
    "ParticleNet (2M)":   ("2M", "370k", 0.828, 0.9820, [5540, 1681, 90, 662, 1654, 4049, 4673, 260, 215]),
    "ParticleNet (10M)":  ("10M", "370k", 0.837, 0.9837, [5848, 2070, 96, 770, 2350, 5495, 6803, 307, 253]),
    "ParticleNet (100M)": ("100M", "370k", 0.844, 0.9849, [7634, 2475, 104, 954, 3339, 10526, 11173, 347, 283]),
    "ParT (2M)":          ("2M", "2.14M", 0.836, 0.9834, [5587, 1982, 93, 761, 1609, 6061, 4474, 307, 236]),
    "ParT (10M)":         ("10M", "2.14M", 0.850, 0.9860, [8734, 3040, 110, 1274, 3257, 12579, 8969, 431, 324]),
    "ParT (100M)":        ("100M", "2.14M", 0.861, 0.9877, [10638, 4149, 123, 1864, 5479, 32787, 15873, 543, 402]),
    "ParT plain (100M)":  ("100M", "2.13M", 0.849, 0.9859, [9569, 2911, 112, 1185, 3868, 17699, 12987, 384, 311]),
}


def _fmt_rej(v):
    return "inf" if not np.isfinite(v) else f"{v:,.0f}"


def plot_curves(hist, out):
    ep = [h["epoch"] for h in hist]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    ax[0].plot(ep, [h["train_loss"] for h in hist], "o-", label="train")
    if "val_loss" in hist[0]:
        ax[0].plot(ep, [h["val_loss"] for h in hist], "s-", label="val")
    ax[0].set(title="Cross-entropy loss", xlabel="epoch")
    ax[1].plot(ep, [h["train_acc"] for h in hist], "o-", label="train")
    if "val_acc" in hist[0]:
        ax[1].plot(ep, [h["val_acc"] for h in hist], "s-", label="val")
    for name in ("ParT (10M)", "ParticleNet (10M)"):
        ax[1].axhline(PAPER[name][2], ls="--", lw=1, color="gray" if "Net" in name else "red",
                      label=f"{name} paper (test)")
    ax[1].set(title="Accuracy (10-class)", xlabel="epoch")
    if "val_auc" in hist[0]:
        ax[2].plot(ep, [h["val_auc"] for h in hist], "s-", color="C1", label="val")
        ax[2].axhline(PAPER["ParT (10M)"][3], ls="--", lw=1, color="red", label="ParT (10M) paper")
    ax[2].set(title="AUC (macro, one-vs-one)", xlabel="epoch")
    for a in ax:
        a.grid(alpha=0.3)
        a.legend()
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_confusion(conf, names, out):
    cm = conf / np.maximum(conf.sum(1, keepdims=True), 1)
    fig, ax = plt.subplots(figsize=(8, 7))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(len(names)), names, rotation=45)
    ax.set_yticks(range(len(names)), names)
    for i in range(len(names)):
        for j in range(len(names)):
            ax.text(j, i, f"{cm[i, j]:.2f}", ha="center", va="center", fontsize=7,
                    color="white" if cm[i, j] > 0.5 else "black")
    ax.set(xlabel="predicted", ylabel="true", title="GDS-ParT test confusion matrix (row-normalized)")
    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_roc(scores, labels, names, out):
    from sklearn.metrics import roc_curve
    if "QCD" not in names:
        return
    b = names.index("QCD")
    fig, ax = plt.subplots(figsize=(8, 6))
    for s_idx, name in enumerate(names):
        if s_idx == b:
            continue
        sel = (labels == s_idx) | (labels == b)
        y = labels[sel] == s_idx
        sc = scores[sel, s_idx] / (scores[sel, s_idx] + scores[sel, b] + 1e-12)
        fpr, tpr, _ = roc_curve(y, sc)
        ok = fpr > 0
        ax.plot(tpr[ok], 1 / fpr[ok], label=f"{name} vs QCD")
    ax.set_yscale("log")
    ax.set(xlabel="signal efficiency (TPR)", ylabel="background rejection (1/FPR)",
           title="GDS-ParT: signal vs QCD", xlim=(0, 1))
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def plot_comparison(ours_name, ours, out):
    names = ["PFN (100M)", "P-CNN (100M)", "ParticleNet (10M)", "ParT (2M)", "ParT (10M)", "ParT (100M)"]
    accs = [PAPER[n][2] for n in names] + [ours["accuracy"]]
    labels = names + [ours_name]
    colors = ["#bbb"] * len(names) + ["C3"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    bars = ax.barh(labels, accs, color=colors)
    for bar, a in zip(bars, accs):
        ax.text(a + 0.001, bar.get_y() + bar.get_height() / 2, f"{a:.3f}", va="center")
    ax.set(xlim=(0.75, 0.875), xlabel="test accuracy (10-class)",
           title="JetClass accuracy: GDS-ParT vs published results (arXiv:2202.03772)")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    args = ap.parse_args()

    res = json.load(open(os.path.join(args.run_dir, "results.json")))
    pred = np.load(os.path.join(args.run_dir, "test_predictions.npz"))
    scores, labels = pred["scores"], pred["labels"]
    names = [str(n) for n in pred["class_names"]]
    test, hist, run_args = res["test"], res["history"], res["args"]
    out_dir = os.path.join(args.run_dir, "report")
    os.makedirs(out_dir, exist_ok=True)

    plot_curves(hist, os.path.join(out_dir, "training_curves.png"))
    plot_confusion(np.array(test["confusion"]), names, os.path.join(out_dir, "confusion_matrix.png"))
    plot_roc(scores, labels, names, os.path.join(out_dir, "roc_curves.png"))

    train_tag = os.path.basename(os.path.normpath(run_args.get("train") or "synthetic"))
    ours_name = f"GDS-ParT ({train_tag.replace('train_', '')})"
    plot_comparison(ours_name, test, os.path.join(out_dir, "comparison.png"))

    rej = test.get("rejection", {})
    best_ep = max(hist, key=lambda h: h.get("val_acc", h["train_acc"]))
    L = []
    L.append(f"# GDS-ParT results — {train_tag}\n")
    L.append("## Setup\n")
    L.append(f"- Training set: `{run_args.get('train')}` ({train_tag})")
    L.append(f"- Validation set: `{run_args.get('val')}` — {res['val']['n_jets']:,} jets" if "val" in res else "")
    L.append(f"- Test set: `{run_args.get('test')}` — **{test['n_jets']:,} jets** (balanced, 10 classes)")
    L.append(f"- Model: {res['n_params']:,} parameters; embed_dim={run_args['embed_dim']}, "
             f"layers={run_args['n_layers']}, heads={run_args['n_heads']}, "
             f"pairwise_embed_dim={run_args['pairwise_embed_dim']}, max_particles={run_args['max_particles']}")
    L.append(f"- Optimizer: AdamW lr={run_args['lr']}, wd={run_args['weight_decay']}, "
             f"batch={run_args['batch_size']}, {run_args['epochs']} epochs, warmup {run_args['warmup_frac']:.0%} + cosine, AMP")
    L.append(f"- Checkpoint used for test: best validation accuracy (epoch {best_ep['epoch']})\n")

    L.append("## Headline test metrics\n")
    L.append("| Metric | Value |\n|---|---|")
    L.append(f"| Accuracy (10-class) | **{test['accuracy']:.4f}** |")
    L.append(f"| AUC (macro, one-vs-one) | **{test['auc_macro_ovo']:.4f}** |")
    L.append(f"| Cross-entropy loss | {test['loss']:.4f} |\n")

    L.append("## Comparison with the ParT paper (JetClass test set)\n")
    L.append("Published numbers from Qu et al., *Particle Transformer for Jet Tagging*, ICML 2022 "
             "(arXiv:2202.03772), Tables 1 and 3. Higher is better for every column.\n")
    hdr = "| Model | Train jets | Params | Accuracy | AUC | " + " | ".join(REJ_COLS) + " |"
    L.append(hdr)
    L.append("|" + "---|" * (5 + len(REJ_COLS)))
    for n, (tj, pp, acc, auc, r) in PAPER.items():
        L.append(f"| {n} | {tj} | {pp} | {acc:.3f} | {auc:.4f} | " + " | ".join(f"{x:,}" for x in r) + " |")
    ours_rej = " | ".join(_fmt_rej(rej.get(c, float("nan"))) for c in REJ_COLS)
    L.append(f"| **{ours_name}** | {train_tag.replace('train_', '')} | {res['n_params'] / 1e6:.2f}M | "
             f"**{test['accuracy']:.3f}** | **{test['auc_macro_ovo']:.4f}** | {ours_rej} |\n")

    ref = PAPER["ParT (10M)"]
    L.append(f"Closest like-for-like reference is **ParT (10M)**: accuracy {ref[2]:.3f}, AUC {ref[3]:.4f}. "
             f"GDS-ParT difference: accuracy {test['accuracy'] - ref[2]:+.3f}, "
             f"AUC {test['auc_macro_ovo'] - ref[3]:+.4f}.\n")

    L.append("### Caveats for a fair reading\n")
    L.append(f"- **Test set size**: ours is a {test['n_jets']:,}-jet balanced subset of the official 20M test split "
             "(1 file per class). Accuracy/AUC are stable at this size (stat. error ~0.0004 on accuracy), "
             "but high rejection values (e.g. Tbl Rej99.5, Hbb Rej50) rely on very few surviving QCD jets "
             "and carry large statistical uncertainty (~10–30%).")
    L.append(f"- **Training budget**: the paper trains for 1M iterations x batch 512 (~512M jets seen). "
             f"This run saw {run_args['epochs']} x {train_tag.replace('train_', '')} jets on a single 6 GB laptop GPU.")
    L.append(f"- **Model size**: GDS-ParT here has {res['n_params'] / 1e6:.2f}M params vs 2.14M for ParT; "
             "no class-attention blocks; pairwise information used via Q/K gates instead of attention bias.")
    L.append("- **Optimizer**: AdamW + cosine (ours) vs Lookahead(RAdam) + step decay (paper).")
    L.append("- Rejection uses the paper's definition: Rej_X% = 1/FPR at TPR = X%, with score S/(S+B) "
             "and QCD as background.\n")

    L.append("## Per-class test accuracy (argmax)\n")
    L.append("| Class | Accuracy |\n|---|---|")
    for n, a in zip(names, test["per_class_accuracy"]):
        L.append(f"| {n} | {a:.4f} |")
    L.append("")

    L.append("## Training history\n")
    L.append("| Epoch | Train loss | Train acc | Val loss | Val acc | Val AUC |\n|---|---|---|---|---|---|")
    for h in hist:
        L.append(f"| {h['epoch']} | {h['train_loss']:.4f} | {h['train_acc']:.4f} | "
                 f"{h.get('val_loss', float('nan')):.4f} | {h.get('val_acc', float('nan')):.4f} | "
                 f"{h.get('val_auc', float('nan')):.4f} |")
    L.append("")

    L.append("## Figures\n")
    for f, t in (("training_curves.png", "Training curves"), ("comparison.png", "Accuracy vs published"),
                 ("roc_curves.png", "Signal vs QCD ROC"), ("confusion_matrix.png", "Confusion matrix")):
        L.append(f"### {t}\n![{t}]({f})\n")

    with open(os.path.join(out_dir, "RESULTS.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(l for l in L if l is not None))
    print(f"report written to {out_dir}")


if __name__ == "__main__":
    main()
