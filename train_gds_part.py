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
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from gds_part_colab import GDSParT, GDSParTConfig, make_synthetic_batch


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
# Metrics
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, n_classes, max_batches=None):
    model.eval()
    total, correct = 0, 0
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    total_loss = 0.0

    for i, (feats, pairs, mask, labels) in enumerate(loader):
        if max_batches and i >= max_batches:
            break
        feats, pairs, mask, labels = (
            feats.to(device), pairs.to(device), mask.to(device), labels.to(device)
        )
        logits = model(feats, pairs, mask=mask)
        loss = F.cross_entropy(logits, labels)
        total_loss += loss.item() * labels.size(0)

        preds = logits.argmax(dim=-1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        for t, p in zip(labels.cpu().numpy(), preds.cpu().numpy()):
            confusion[t, p] += 1

    acc = correct / max(total, 1)
    avg_loss = total_loss / max(total, 1)
    per_class_acc = np.divide(
        np.diag(confusion), confusion.sum(axis=1),
        out=np.zeros(n_classes), where=confusion.sum(axis=1) > 0
    )
    model.train()
    return {"accuracy": acc, "loss": avg_loss,
            "confusion": confusion, "per_class_accuracy": per_class_acc}


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
def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    cfg = GDSParTConfig(
        embed_dim=args.embed_dim, n_heads=args.n_heads, n_layers=args.n_layers,
        ffn_dim=args.ffn_dim, pairwise_embed_dim=args.pairwise_embed_dim,
        dropout=args.dropout, max_particles=args.max_particles,
    )
    model = GDSParT(cfg).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model parameters: {n_params:,}")

    if args.synthetic:
        train_loader = SyntheticLoader(cfg, n_batches=args.synthetic_batches,
                                        batch_size=args.batch_size, device=device)
        val_loader = SyntheticLoader(cfg, n_batches=max(2, args.synthetic_batches // 4),
                                      batch_size=args.batch_size, device=device)
        n_classes = cfg.n_classes
    else:
        from data.jetclass_dataset import JetClassDataset, collate_fn, CLASS_LIST
        train_ds = JetClassDataset(args.train, split_jets_per_class=args.jets_per_class,
                                    max_particles=cfg.max_particles)
        val_ds = JetClassDataset(args.val, split_jets_per_class=max(500, args.jets_per_class // 10),
                                  max_particles=cfg.max_particles)
        train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                                   collate_fn=collate_fn, num_workers=args.num_workers)
        val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                                 collate_fn=collate_fn, num_workers=args.num_workers)
        n_classes = len(CLASS_LIST)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    steps_per_epoch = len(train_loader)
    total_steps = steps_per_epoch * args.epochs
    warmup_steps = int(args.warmup_frac * total_steps)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, make_lr_lambda(warmup_steps, total_steps))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_val_acc = -1.0
    global_step = 0

    for epoch in range(args.epochs):
        model.train()
        epoch_loss, epoch_correct, epoch_total = 0.0, 0, 0
        t0 = time.time()

        for feats, pairs, mask, labels in train_loader:
            feats, pairs, mask, labels = (
                feats.to(device), pairs.to(device), mask.to(device), labels.to(device)
            )
            logits = model(feats, pairs, mask=mask)
            loss = F.cross_entropy(logits, labels)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
            optimizer.step()
            scheduler.step()
            global_step += 1

            epoch_loss += loss.item() * labels.size(0)
            epoch_correct += (logits.argmax(-1) == labels).sum().item()
            epoch_total += labels.size(0)

            if global_step % args.log_every == 0:
                lr_now = scheduler.get_last_lr()[0]
                print(f"  step {global_step}/{total_steps}  "
                      f"loss={loss.item():.4f}  lr={lr_now:.2e}")

        train_acc = epoch_correct / max(epoch_total, 1)
        train_loss = epoch_loss / max(epoch_total, 1)

        val_metrics = evaluate(model, val_loader, device, n_classes)
        dt = time.time() - t0
        print(f"[epoch {epoch+1}/{args.epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.4f}  "
              f"val_loss={val_metrics['loss']:.4f} val_acc={val_metrics['accuracy']:.4f}  "
              f"({dt:.1f}s)")

        # checkpointing
        ckpt = {
            "epoch": epoch, "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(), "config": vars(cfg)
            if hasattr(cfg, "__dict__") else cfg.__dict__,
            "val_acc": val_metrics["accuracy"],
        }
        torch.save(ckpt, os.path.join(args.checkpoint_dir, "last.pt"))
        if val_metrics["accuracy"] > best_val_acc:
            best_val_acc = val_metrics["accuracy"]
            torch.save(ckpt, os.path.join(args.checkpoint_dir, "best.pt"))
            print(f"  -> new best model saved (val_acc={best_val_acc:.4f})")

    print(f"\nTraining complete. Best val_acc={best_val_acc:.4f}")

    # final test evaluation
    if args.synthetic:
        test_loader = SyntheticLoader(cfg, n_batches=max(2, args.synthetic_batches // 4),
                                       batch_size=args.batch_size, device=device)
        class_names = [f"class_{i}" for i in range(n_classes)]
    elif args.test:
        from data.jetclass_dataset import JetClassDataset, collate_fn, CLASS_LIST
        test_ds = JetClassDataset(args.test, split_jets_per_class=max(500, args.jets_per_class // 10),
                                   max_particles=cfg.max_particles)
        test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                                  collate_fn=collate_fn, num_workers=args.num_workers)
        class_names = CLASS_LIST
    else:
        return

    test_metrics = evaluate(model, test_loader, device, n_classes)
    print(f"\nFinal test accuracy: {test_metrics['accuracy']:.4f}")
    print("Per-class accuracy:")
    for name, acc in zip(class_names, test_metrics["per_class_accuracy"]):
        print(f"  {name:8s}: {acc:.4f}")
    print("\nConfusion matrix (rows=true, cols=pred):")
    print(test_metrics["confusion"])


def build_argparser():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", type=str, default=None, help="Path to train split dir")
    ap.add_argument("--val", type=str, default=None, help="Path to val split dir")
    ap.add_argument("--test", type=str, default=None, help="Path to test split dir")
    ap.add_argument("--synthetic", action="store_true",
                     help="Run a synthetic-data dry run instead of loading JetClass")
    ap.add_argument("--synthetic_batches", type=int, default=20)
    ap.add_argument("--jets_per_class", type=int, default=5000)

    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch_size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-2)
    ap.add_argument("--warmup_frac", type=float, default=0.05)
    ap.add_argument("--grad_clip", type=float, default=1.0)
    ap.add_argument("--num_workers", type=int, default=4)
    ap.add_argument("--log_every", type=int, default=50)

    ap.add_argument("--embed_dim", type=int, default=128)
    ap.add_argument("--n_heads", type=int, default=8)
    ap.add_argument("--n_layers", type=int, default=6)
    ap.add_argument("--ffn_dim", type=int, default=512)
    ap.add_argument("--pairwise_embed_dim", type=int, default=64)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--max_particles", type=int, default=128)

    ap.add_argument("--checkpoint_dir", type=str, default="./checkpoints")
    return ap


if __name__ == "__main__":
    args = build_argparser().parse_args()
    if not args.synthetic and not args.train:
        raise SystemExit("Provide --train/--val/--test paths, or pass --synthetic for a dry run.")
    train(args)
