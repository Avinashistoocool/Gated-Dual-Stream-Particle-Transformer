"""
train_gds_part.py — trainer for GDS-ParT on JetClass.

Example:
    python train_gds_part.py \
        --train ./particle_transformer/data/JetClass/train_100M \
        --val   ./particle_transformer/data/JetClass/val_5M \
        --test  ./particle_transformer/data/JetClass/test_20M \
        --epochs 10 \
        --batch_size 512

Synthetic dry run (no data needed, just checks the training loop wiring):
    python train_gds_part.py --synthetic --epochs 1
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gds_part_colab import GDSParT, GDSParTConfig, make_synthetic_batch
from run_logger import RunLogger

log = None  # RunLogger, created in train()


def _say(msg=""):
    if log is not None:
        for line in str(msg).strip("\n").splitlines() or [""]:
            log.info(line)
    else:
        print(msg, flush=True)


# ----------------------------------------------------------------------
# LR schedule: linear warmup then cosine decay
# ----------------------------------------------------------------------
def make_lr_lambda(warmup_steps, total_steps):
    def lr_lambda(step):
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        progress = min(1.0, progress)
        return 0.5 * (1 + math.cos(math.pi * progress))
    return lr_lambda


# ----------------------------------------------------------------------
# Metrics (same definitions as the ParT paper, arXiv:2202.03772 Sec. 2)
# ----------------------------------------------------------------------
# signal classes vs. QCD background, and the signal efficiency (TPR) used
# for the background-rejection metric RejX% = 1 / FPR at TPR = X%
REJ_TARGETS = [("Hbb", 0.5), ("Hcc", 0.5), ("Hgg", 0.5), ("H4q", 0.5), ("Hqql", 0.99),
               ("Tbqq", 0.5), ("Tbl", 0.995), ("Wqq", 0.5), ("Zqq", 0.5)]


def compute_jet_tagging_metrics(scores, labels, class_names):
    """scores: (n, C) softmax outputs; labels: (n,) ints."""
    from sklearn.metrics import roc_auc_score, roc_curve

    out = {}
    preds = scores.argmax(1)
    out["accuracy"] = float((preds == labels).mean())
    try:
        out["auc_macro_ovo"] = float(roc_auc_score(labels, scores, average="macro",
                                                   multi_class="ovo"))
    except ValueError:
        out["auc_macro_ovo"] = float("nan")

    if "QCD" in class_names:
        b = class_names.index("QCD")
        rej = {}
        for name, tpr_target in REJ_TARGETS:
            if name not in class_names:
                continue
            s_idx = class_names.index(name)
            sel = (labels == s_idx) | (labels == b)
            y = (labels[sel] == s_idx).astype(int)
            if y.sum() == 0 or (1 - y).sum() == 0:
                continue
            sc = scores[sel, s_idx] / (scores[sel, s_idx] + scores[sel, b] + 1e-12)
            fpr, tpr, _ = roc_curve(y, sc)
            fpr_at = float(np.interp(tpr_target, tpr, fpr))
            rej[f"{name}_Rej{tpr_target * 100:g}"] = (1.0 / fpr_at) if fpr_at > 0 else float("inf")
        out["rejection"] = rej
    return out


@torch.no_grad()
def evaluate(model, loader, device, n_classes, max_batches=None, class_names=None, use_amp=False,
             max_pairs=None):
    model.eval()
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    total_loss, total = 0.0, 0
    all_scores, all_labels = [], []

    for i, batch in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        feats, pairs, mask, labels = prepare_batch(batch, device)
        N = feats.shape[1]
        micro = max(1, max_pairs // (N * N)) if max_pairs else feats.shape[0]
        chunks = []
        for s0 in range(0, feats.shape[0], micro):
            sl = slice(s0, s0 + micro)
            with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                chunks.append(model(feats[sl], pairs[sl], mask=mask[sl]).float())
        logits = torch.cat(chunks)
        total_loss += F.cross_entropy(logits, labels, reduction="sum").item()
        total += labels.size(0)
        all_scores.append(torch.softmax(logits, -1).cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    scores = np.concatenate(all_scores)
    labels = np.concatenate(all_labels)
    preds = scores.argmax(1)
    np.add.at(confusion, (labels, preds), 1)
    per_class_acc = np.divide(
        np.diag(confusion), confusion.sum(axis=1),
        out=np.zeros(n_classes), where=confusion.sum(axis=1) > 0
    )
    class_names = class_names or [f"class_{i}" for i in range(n_classes)]
    metrics = compute_jet_tagging_metrics(scores, labels, class_names)
    metrics.update({"loss": total_loss / max(total, 1), "confusion": confusion,
                    "per_class_accuracy": per_class_acc, "n_jets": int(total),
                    "scores": scores, "labels": labels})
    model.train()
    return metrics


def prepare_batch(batch, device):
    """Real-data batches carry 4-vectors; pairwise features are built on the
    GPU here. Synthetic batches already carry pairwise features."""
    feats, second, mask, labels = batch
    feats, second = feats.to(device, non_blocking=True), second.to(device, non_blocking=True)
    mask, labels = mask.to(device, non_blocking=True), labels.to(device, non_blocking=True)
    if second.dim() == 3:  # (B, N, 4) four-vectors -> (B, N, N, 4) pairwise
        from data.jetclass_cache import pairwise_features
        second = pairwise_features(second, mask)
    return feats, second, mask, labels


# ----------------------------------------------------------------------
# Synthetic loader (dry-run / CI use)
# ----------------------------------------------------------------------
class SyntheticLoader:
    """Mimics a DataLoader, yielding synthetic batches. Lets you exercise
    the entire training loop (optimizer, scheduler, checkpointing, eval)
    without any real data."""

    def __init__(self, cfg, n_batches, batch_size, device):
        self.cfg = cfg
        self.n_batches = n_batches
        self.batch_size = batch_size
        self.device = device

    def __iter__(self):
        for _ in range(self.n_batches):
            feats, pairs, mask, labels = make_synthetic_batch(
                batch_size=self.batch_size, max_particles=self.cfg.max_particles,
                cfg=self.cfg, device="cpu"
            )
            yield feats, pairs, mask, labels

    def __len__(self):
        return self.n_batches


# ----------------------------------------------------------------------
# Training loop
# ----------------------------------------------------------------------
def _print_metrics(tag, m, class_names):
    _say(f"[{tag}] n_jets={m['n_jets']:,}  accuracy={m['accuracy']:.4f}  "
          f"AUC(macro,ovo)={m['auc_macro_ovo']:.4f}  loss={m['loss']:.4f}")
    if m.get("rejection"):
        _say("  background rejection vs QCD (1/FPR at fixed signal efficiency):")
        for k, v in m["rejection"].items():
            _say(f"    {k:14s} {v:10.1f}")
    _say("  per-class accuracy:")
    for name, acc in zip(class_names, m["per_class_accuracy"]):
        _say(f"    {name:6s} {acc:.4f}")


def _jsonable(m):
    return {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in m.items()}


def _make_real_loaders(args, cfg):
    from data.jetclass_cache import build_cache, CachedJetLoader
    cache_root = args.cache_dir
    loaders = {}
    for split, path, jets, shuffle in (
        ("train", args.train, args.jets_per_class, True),
        ("val", args.val, args.val_jets_per_class, False),
        ("test", args.test, args.test_jets_per_class, False),
    ):
        if not path:
            continue
        tag = os.path.basename(os.path.normpath(path))
        cdir = os.path.join(cache_root, f"{split}_{tag}_{jets or 'all'}_{cfg.max_particles}")
        build_cache(path, cdir, max_particles=cfg.max_particles, jets_per_class=jets)
        bs = args.batch_size if split == "train" else args.eval_batch_size
        loaders[split] = CachedJetLoader(cdir, bs, shuffle=shuffle, seed=args.seed)
    return loaders


def train(args):
    global log
    log = RunLogger(args.checkpoint_dir, name="train")
    log.env(args)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    use_amp = args.amp and device == "cuda"
    _say(f"device: {device}  amp: {use_amp}")

    cfg = GDSParTConfig(
        embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        ffn_dim=args.ffn_dim, pairwise_embed_dim=args.pairwise_embed_dim,
        dropout=args.dropout, max_particles=args.max_particles,
        gate_mode=args.gate_mode, per_layer_gate=bool(args.per_layer_gate),
        pair_bias=bool(args.pair_bias), n_cls_layers=args.n_cls_layers,
    )
    model = GDSParT(cfg).to(device)
    ema = None
    if args.ema_decay > 0:
        from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
        ema = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay),
                            use_buffers=True)
    eval_model = lambda: ema.module if ema is not None else model
    n_params = sum(p.numel() for p in model.parameters())
    _say(f"model parameters: {n_params:,}")
    log.event("model", n_params=n_params, config=vars(cfg))

    if args.synthetic:
        train_loader = SyntheticLoader(cfg, n_batches=args.synthetic_batches,
                                        batch_size=args.batch_size, device=device)
        val_loader = SyntheticLoader(cfg, n_batches=max(2, args.synthetic_batches // 4),
                                      batch_size=args.batch_size, device=device)
        test_loader = SyntheticLoader(cfg, n_batches=max(2, args.synthetic_batches // 4),
                                       batch_size=args.batch_size, device=device)
        class_names = [f"class_{i}" for i in range(cfg.n_classes)]
    else:
        from data.jetclass_cache import CLASS_LIST
        loaders = _make_real_loaders(args, cfg)
        train_loader, val_loader = loaders["train"], loaders.get("val")
        test_loader = loaders.get("test")
        class_names = CLASS_LIST
    n_classes = len(class_names)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup_steps, total_steps))
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc, global_step, start_epoch = -1.0, 0, 0
    history = []
    last_path = os.path.join(args.checkpoint_dir, "last.pt")
    if args.resume and os.path.exists(last_path):
        ck = torch.load(last_path, map_location=device, weights_only=False)
        model.load_state_dict(ck.get("raw_model_state", ck["model_state"]))
        optimizer.load_state_dict(ck["optimizer_state"])
        if "scheduler_state" in ck:
            scheduler.load_state_dict(ck["scheduler_state"])
        if ema is not None and "ema_state" in ck:
            ema.load_state_dict(ck["ema_state"])
        start_epoch = ck["epoch"] + 1
        global_step = ck.get("global_step", start_epoch * steps_per_epoch)
        best_val_acc = ck.get("best_val_acc", ck.get("val_acc", -1.0))
        history = ck.get("history", [])
        _say(f"resumed from {last_path} at epoch {start_epoch}")

    def save(name, epoch, val_metrics):
        # "model_state" holds the weights used for evaluation (EMA if enabled);
        # "raw_model_state" keeps the raw weights so training can resume exactly
        torch.save({
            "epoch": epoch, "global_step": global_step,
            "model_state": eval_model().state_dict(),
            "raw_model_state": model.state_dict(),
            "ema_state": ema.state_dict() if ema is not None else None,
            "optimizer_state": optimizer.state_dict(),
            "scheduler_state": scheduler.state_dict(), "config": vars(cfg),
            "val_acc": val_metrics["accuracy"] if val_metrics else None,
            "best_val_acc": best_val_acc, "history": history, "args": vars(args),
        }, os.path.join(args.checkpoint_dir, name))

    _say(f"steps/epoch: {steps_per_epoch:,}  total steps: {total_steps:,}")
    for epoch in range(start_epoch, args.epochs):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
        t0 = tlog = time.time()

        for batch in train_loader:
            feats, pairs, mask, labels = prepare_batch(batch, device)
            optimizer.zero_grad(set_to_none=True)
            # split batches with many particles into micro-batches (gradient
            # accumulation) so the O(B*N^2) pairwise tensor fits in GPU memory
            B, N = feats.shape[:2]
            micro = max(1, min(B, args.max_pairs_per_step // (N * N)))
            loss_sum, logits_all = 0.0, []
            for s0 in range(0, B, micro):
                sl = slice(s0, s0 + micro)
                with torch.autocast("cuda", dtype=torch.float16, enabled=use_amp):
                    lg = model(feats[sl], pairs[sl], mask=mask[sl])
                    l = F.cross_entropy(lg.float(), labels[sl], reduction="sum",
                                        label_smoothing=args.label_smoothing) / B
                scaler.scale(l).backward()
                loss_sum += l.item()
                logits_all.append(lg.detach())
            logits = torch.cat(logits_all)
            loss = torch.tensor(loss_sum)
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            if ema is not None:
                ema.update_parameters(model)
            global_step += 1

            epoch_loss += loss.item() * labels.size(0)
            epoch_correct += (logits.argmax(-1) == labels).sum().item()
            epoch_total += labels.size(0)

            if global_step % args.log_every == 0:
                log.event("step", epoch=epoch + 1, step=global_step, loss=loss.item(),
                          run_acc=epoch_correct / epoch_total, lr=scheduler.get_last_lr()[0],
                          gpu_mem_gb=torch.cuda.max_memory_allocated() / 2**30 if device == "cuda" else 0)
                lr_now = scheduler.get_last_lr()[0]
                rate = args.log_every * args.batch_size / (time.time() - tlog)
                tlog = time.time()
                _say(f"  ep {epoch+1} step {global_step}/{total_steps}  loss={loss.item():.4f}  "
                      f"run_acc={epoch_correct / epoch_total:.4f}  lr={lr_now:.2e}  "
                      f"{rate:.0f} jets/s")

        train_acc = epoch_correct / max(epoch_total, 1)
        train_loss = epoch_loss / max(epoch_total, 1)
        val_metrics = None
        if val_loader is not None:
            val_metrics = evaluate(eval_model(), val_loader, device, n_classes,
                                   class_names=class_names, use_amp=use_amp,
                                   max_pairs=2 * args.max_pairs_per_step)
        dt = time.time() - t0
        msg = (f"[epoch {epoch+1}/{args.epochs}] train_loss={train_loss:.4f} "
               f"train_acc={train_acc:.4f}")
        if val_metrics:
            msg += (f"  val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f} "
                    f"val_auc={val_metrics['auc_macro_ovo']:.4f}")
        _say(msg + f"  ({dt / 60:.1f} min)")
        if val_metrics:
            log.metrics("val_epoch", {"epoch": epoch + 1, **val_metrics})
        history.append({"epoch": epoch + 1, "train_loss": train_loss, "train_acc": train_acc,
                        **({"val_loss": val_metrics["loss"], "val_acc": val_metrics["accuracy"],
                            "val_auc": val_metrics["auc_macro_ovo"]} if val_metrics else {})})

        cur = val_metrics["accuracy"] if val_metrics else train_acc
        if cur > best_val_acc:
            best_val_acc = cur
            save("best.pt", epoch, val_metrics)
            log.event("checkpoint", file="best.pt", epoch=epoch + 1, val_acc=best_val_acc)
            _say(f"  -> new best model saved (val_acc={best_val_acc:.4f})")
        save("last.pt", epoch, val_metrics)
        log.event("epoch", **history[-1], minutes=dt / 60)

    _say(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")

    results = {"n_params": n_params, "history": history, "args": vars(args)}
    best_path = os.path.join(args.checkpoint_dir, "best.pt")
    final_model = eval_model()
    if os.path.exists(best_path):
        final_model.load_state_dict(torch.load(best_path, map_location=device,
                                               weights_only=False)["model_state"])
        _say("loaded best.pt for final evaluation")
    for tag, loader in (("val (best ckpt)", val_loader), ("TEST (best ckpt)", test_loader)):
        if loader is None:
            continue
        m = evaluate(final_model, loader, device, n_classes, class_names=class_names, use_amp=use_amp,
                                   max_pairs=2 * args.max_pairs_per_step)
        _print_metrics(tag, m, class_names)
        if tag.startswith("TEST"):
            _say("\nConfusion matrix (rows=true, cols=pred):")
            _say(np.array2string(np.asarray(m["confusion"]), max_line_width=200))
        split = tag.split()[0].lower()
        np.savez_compressed(os.path.join(args.checkpoint_dir, f"{split}_predictions.npz"),
                            scores=m.pop("scores"), labels=m.pop("labels"),
                            class_names=np.array(class_names))
        results[split] = _jsonable(m)
        log.metrics(f"final_{split}", m)

    with open(os.path.join(args.checkpoint_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2, default=float)
    _say(f"metrics written to {os.path.join(args.checkpoint_dir, 'results.json')}")
    log.mark_done({"best_val_acc": best_val_acc,
                   "test_accuracy": results.get("test", {}).get("accuracy"),
                   "test_auc": results.get("test", {}).get("auc_macro_ovo")})


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default=None, help="Path to train split dir")
    ap.add_argument("--val", type=str, default=None, help="Path to val split dir")
    ap.add_argument("--test", type=str, default=None, help="Path to test split dir")
    ap.add_argument("--synthetic", action="store_true",
                     help="Run a synthetic-data dry run instead of loading JetClass")
    ap.add_argument("--synthetic_batches", type=int, default=20)
    ap.add_argument("--jets_per_class", type=int, default=None,
                     help="train jets per class (default: all jets in --train)")
    ap.add_argument("--val_jets_per_class", type=int, default=None)
    ap.add_argument("--test_jets_per_class", type=int, default=None)
    ap.add_argument("--cache_dir", type=str, default="./cache",
                     help="where preprocessed per-particle memmaps are stored")

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--eval_batch_size", type=int, default=512)
    ap.add_argument("--max_pairs_per_step", type=int, default=2_000_000,
                     help="max B*N*N per forward pass; larger batches are split into "
                          "micro-batches with gradient accumulation (tune to GPU memory)")
    ap.add_argument("--amp", type=int, default=1, help="mixed precision on CUDA (1/0)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="resume from checkpoint_dir/last.pt")
    ap.add_argument("--log_every", type=int, default=50)

    ap.add_argument("--label_smoothing", type=float, default=0.1)
    ap.add_argument("--ema_decay", type=float, default=0.999,
                     help="EMA of weights used for eval/checkpoints (0 = off)")

    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=8)
    ap.add_argument("--ffn_dim", type=int, default=512)
    ap.add_argument("--pairwise_embed_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_particles", type=int, default=128)
    # v2 architecture switches (v1 = --gate_mode sigmoid --per_layer_gate 0
    # --pair_bias 0 --n_cls_layers 0 --n_layers 6)
    ap.add_argument("--gate_mode", choices=["identity", "sigmoid"], default="identity")
    ap.add_argument("--per_layer_gate", type=int, default=1)
    ap.add_argument("--pair_bias", type=int, default=1)
    ap.add_argument("--n_cls_layers", type=int, default=2)

    ap.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not args.synthetic and not args.train:
        raise SystemExit("Provide --train/--val/--test paths, or pass --synthetic for a dry run.")
    train(args)
