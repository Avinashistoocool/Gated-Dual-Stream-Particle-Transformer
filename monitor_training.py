"""
monitor_training.py — live terminal dashboard for a train_gds_part.py run.

Parses the training log and shows a progress bar, ETA, throughput, loss,
accuracy, per-epoch validation results, and GPU stats (via nvidia-smi).

    python monitor_training.py --log ./checkpoints/train_jetclass_10M.log
"""
import argparse
import os
import re
import subprocess
import time
from collections import deque

STEP_RE = re.compile(r"ep (\d+) step (\d+)/(\d+)\s+loss=([\d.]+)\s+run_acc=([\d.]+)\s+lr=([\d.e+-]+)\s+(\d+) jets/s")
EPOCH_RE = re.compile(r"\[epoch (\d+)/(\d+)\] train_loss=([\d.]+) train_acc=([\d.]+)"
                      r"(?:\s+val_loss=([\d.]+) val_acc=([\d.]+) val_auc=([\d.na]+))?\s+\(([\d.]+) min\)")
PARAMS_RE = re.compile(r"model parameters: ([\d,]+)")
SPE_RE = re.compile(r"steps/epoch: ([\d,]+)")

PART_10M = {"acc": 0.850, "auc": 0.9860}


def bar(frac, width=50):
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    return "█" * full + "░" * (width - full)


def gpu_stats():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        name, util, used, total, temp, power = [x.strip() for x in out.split(",")]
        return f"{name} | util {util}% | mem {used}/{total} MiB | {temp} C | {power} W"
    except Exception:
        return "nvidia-smi unavailable"


def fmt_dur(sec):
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}h {m:02d}m {s:02d}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", default="./checkpoints/train_jetclass_10M.log")
    ap.add_argument("--refresh", type=float, default=5.0)
    args = ap.parse_args()

    recent_loss = deque(maxlen=20)
    while True:
        try:
            with open(args.log, "rb") as f:
                raw = f.read()
            # Tee-Object on Windows PowerShell writes UTF-16 LE
            enc = "utf-16" if raw[:2] in (b"\xff\xfe", b"\xfe\xff") else "utf-8"
            text = raw.decode(enc, errors="replace")
        except FileNotFoundError:
            print(f"waiting for {args.log} ...")
            time.sleep(args.refresh)
            continue

        steps = STEP_RE.findall(text)
        epochs = EPOCH_RE.findall(text)
        params = PARAMS_RE.search(text)
        spe = SPE_RE.search(text)
        done = "metrics written to" in text
        crashed = "Traceback" in text

        recent_loss.clear()
        for s in steps[-20:]:
            recent_loss.append(float(s[3]))

        lines = []
        lines.append("=" * 78)
        lines.append("  GDS-ParT training monitor".ljust(50) + time.strftime("%Y-%m-%d %H:%M:%S"))
        lines.append("=" * 78)
        lines.append(f"  log: {os.path.abspath(args.log)}")
        if params:
            lines.append(f"  model parameters: {params.group(1)}   steps/epoch: {spe.group(1) if spe else '?'}")
        lines.append("")

        if steps:
            ep, step, total, loss, racc, lr, rate = steps[-1]
            step, total, rate = int(step), int(total), int(float(rate))
            n_ep = int(epochs[0][1]) if epochs else None
            spe_n = int(spe.group(1).replace(",", "")) if spe else total
            frac = step / total
            ep_frac = (step - (int(ep) - 1) * spe_n) / spe_n
            rates = [int(float(s[6])) for s in steps[-10:]]
            avg_rate = sum(rates) / len(rates)
            eta_train = (total - step) * 512 / max(avg_rate, 1)
            eval_overhead = (n_ep or 8) - len(epochs)
            eta = eta_train + eval_overhead * 60

            lines.append(f"  OVERALL  [{bar(frac)}] {frac * 100:5.1f}%")
            lines.append(f"           step {step:,}/{total:,}   ETA ~{fmt_dur(eta)}")
            lines.append(f"  EPOCH {ep}{'/' + str(n_ep) if n_ep else ''} [{bar(ep_frac)}] {ep_frac * 100:5.1f}%")
            lines.append("")
            lines.append(f"  loss (last)     {float(loss):.4f}     loss (avg last 20 logs) {sum(recent_loss) / len(recent_loss):.4f}")
            lines.append(f"  running acc     {float(racc):.4f}     (cumulative over current epoch)")
            lines.append(f"  learning rate   {float(lr):.2e}")
            lines.append(f"  throughput      {rate:,} jets/s   (avg {avg_rate:,.0f})")
        else:
            lines.append("  (building cache / waiting for first log line...)")

        lines.append("")
        lines.append(f"  GPU: {gpu_stats()}")
        lines.append("")
        lines.append("  Per-epoch validation (500k jets)        ParT paper (10M train): "
                     f"acc {PART_10M['acc']:.3f}  AUC {PART_10M['auc']:.4f}")
        lines.append("  " + "-" * 74)
        lines.append("  epoch | train_loss | train_acc | val_loss | val_acc | val_auc  | time")
        if epochs:
            best = max(float(e[5] or 0) for e in epochs)
            for e in epochs:
                star = " *best" if e[5] and float(e[5]) == best else ""
                lines.append(f"  {e[0]:>5} | {float(e[2]):10.4f} | {float(e[3]):9.4f} | "
                             f"{float(e[4] or 'nan'):8.4f} | {float(e[5] or 'nan'):7.4f} | "
                             f"{float(e[6]) if e[6] not in ('', 'nan') else float('nan'):8.4f} | {e[7]} min{star}")
        else:
            lines.append("  (first epoch not finished yet)")

        if done:
            lines.append("")
            lines.append("  TRAINING COMPLETE. Final test metrics:")
            tail = text[text.rfind("[TEST"):]
            lines += ["  " + l for l in tail.splitlines()[:25]]
        elif crashed:
            lines.append("")
            lines.append("  !! Traceback found in log — training may have crashed:")
            lines += ["  " + l for l in text[text.rfind("Traceback"):].splitlines()[-8:]]

        os.system("cls" if os.name == "nt" else "clear")
        print("\n".join(lines), flush=True)
        if done:
            break
        time.sleep(args.refresh)


if __name__ == "__main__":
    main()
