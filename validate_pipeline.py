"""
validate_pipeline.py — independent check of our data pipeline + metric code.

Runs the OFFICIAL pretrained ParT checkpoint (ParT_full.pt from
jet-universe/particle_transformer) on our cached JetClass test set, using our
preprocessing (data/jetclass_cache.py) and our metric implementation
(train_gds_part.compute_jet_tagging_metrics). If the numbers match the
published ParT results (acc 0.861, AUC 0.9877, Table 1 of arXiv:2202.03772),
then the features, labels and metrics used for GDS-ParT are correct.

    pip install --no-deps weaver-core==0.4.17   # 0.5.x changed ParT internals; old weights give garbage
    python validate_pipeline.py --ckpt path/to/ParT_full.pt --cache D:/JetClass/cache/test_test_all_128
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from data.jetclass_cache import CLASS_LIST, CachedJetLoader
from run_logger import RunLogger
from train_gds_part import compute_jet_tagging_metrics, REJ_TARGETS

# our feature order -> official JetClass_full.yaml pf_features order
# ours:     0-4 kinematics, 5 deta, 6 dphi, 7 charge, 8-12 pid, 13 d0, 14 d0err, 15 dz, 16 dzerr
# official: 0-4 kinematics, charge, pid x5, d0, d0err, dz, dzerr, deta, dphi
TO_OFFICIAL = [0, 1, 2, 3, 4, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 5, 6]

PAPER_PART = {"accuracy": 0.861, "auc_macro_ovo": 0.9877,
              "rejection": dict(zip([f"{n}_Rej{t * 100:g}" for n, t in REJ_TARGETS],
                                    [10638, 4149, 123, 1864, 5479, 32787, 15873, 543, 402]))}


def build_official_part():
    from weaver.nn.model.ParticleTransformer import ParticleTransformer
    return ParticleTransformer(
        input_dim=17, num_classes=10, pair_input_dim=4, use_pre_activation_pair=False,
        embed_dims=[128, 512, 128], pair_embed_dims=[64, 64, 64], num_heads=8, num_layers=8,
        num_cls_layers=2, block_params=None,
        cls_block_params={"dropout": 0, "attn_dropout": 0, "activation_dropout": 0},
        fc_params=[], activation="gelu", trim=True, for_inference=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--max_jets", type=int, default=None)
    ap.add_argument("--out_dir", default="./validation")
    args = ap.parse_args()

    log = RunLogger(args.out_dir, name="validate_pipeline")
    log.env(args)
    import weaver
    log.info(f"weaver-core {getattr(weaver, '__version__', '?')} from {weaver.__file__}")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = build_official_part()
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = {k[len("mod."):] if k.startswith("mod.") else k: v for k, v in sd.items()}
    model.load_state_dict(sd, strict=True)
    model.to(device).eval()
    log.info(f"loaded official ParT weights from {args.ckpt} "
             f"({sum(p.numel() for p in model.parameters()):,} params)")

    loader = CachedJetLoader(args.cache, args.batch_size, shuffle=False, max_jets=args.max_jets)
    scores, labels = [], []
    t0 = time.time()
    with torch.no_grad():
        for i, (f, v, m, y) in enumerate(loader):
            f = f[..., TO_OFFICIAL].to(device).transpose(1, 2)        # (B, 17, N)
            v = v.to(device).transpose(1, 2)                          # (B, 4, N)
            mask = m.to(device).unsqueeze(1).float()                  # (B, 1, N)
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                p = model(f, v=v, mask=mask)
            scores.append(p.float().cpu().numpy())
            labels.append(y.numpy())
            if i % 200 == 0:
                log.info(f"batch {i}/{len(loader)}  {sum(len(l) for l in labels) / (time.time() - t0):.0f} jets/s")

    scores, labels = np.concatenate(scores), np.concatenate(labels)
    m = compute_jet_tagging_metrics(scores, labels, CLASS_LIST)
    m["n_jets"] = int(len(labels))
    m["class_counts"] = np.bincount(labels, minlength=10).tolist()

    log.info(f"official ParT on our pipeline: n_jets={m['n_jets']:,}  "
             f"accuracy={m['accuracy']:.4f} (paper {PAPER_PART['accuracy']})  "
             f"AUC={m['auc_macro_ovo']:.4f} (paper {PAPER_PART['auc_macro_ovo']})")
    rows = []
    for k, ours in m["rejection"].items():
        paper = PAPER_PART["rejection"][k]
        rows.append({"metric": k, "ours": ours, "paper": paper, "ratio": ours / paper})
        log.info(f"  {k:14s} ours={ours:10.1f}  paper={paper:8d}  ratio={ours / paper:.2f}")
    result = {"official_part_on_our_pipeline": m, "paper": PAPER_PART, "rejection_comparison": rows}
    with open(os.path.join(args.out_dir, "validation_results.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    np.savez_compressed(os.path.join(args.out_dir, "part_official_test_scores.npz"),
                        scores=scores, labels=labels)
    log.metrics("validation", m)
    log.mark_done({"accuracy": m["accuracy"], "auc": m["auc_macro_ovo"]})


if __name__ == "__main__":
    main()
