"""
run_logger.py — persistent, append-only logging for every run.

Each run writes to <out_dir>/:
    <name>.log        human-readable UTF-8 log (everything printed + timestamps)
    events.jsonl      one JSON object per event (env, step, epoch, eval, checkpoint, error)

and appends a one-line summary to ./runs_index.jsonl at the repo root, so
there is a single place listing every run ever started and how it ended.
"""
import atexit
import datetime as dt
import json
import logging
import os
import platform
import socket
import subprocess
import sys
import traceback

import numpy as np

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))


def _git(*a):
    try:
        return subprocess.run(["git", *a], cwd=REPO_ROOT, capture_output=True, text=True,
                              timeout=10).stdout.strip()
    except Exception:
        return ""


def _jsonable(o):
    if isinstance(o, dict):
        return {str(k): _jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    return o


class RunLogger:
    def __init__(self, out_dir, name="train"):
        os.makedirs(out_dir, exist_ok=True)
        self.out_dir = out_dir
        self.name = name
        self.run_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        self.started = dt.datetime.now()
        self.status = "running"

        self.logger = logging.getLogger(f"{name}-{self.run_id}")
        self.logger.setLevel(logging.INFO)
        self.logger.propagate = False
        fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s", "%Y-%m-%d %H:%M:%S")
        fh = logging.FileHandler(os.path.join(out_dir, f"{name}.log"), encoding="utf-8")
        fh.setFormatter(fmt)
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(logging.Formatter("%(message)s"))
        self.logger.addHandler(fh)
        self.logger.addHandler(sh)
        self._events = open(os.path.join(out_dir, "events.jsonl"), "a", encoding="utf-8")

        self._prev_excepthook = sys.excepthook
        sys.excepthook = self._excepthook
        atexit.register(self._finish)
        self.info(f"===== run {self.run_id} ({name}) started =====")

    # ------------------------------------------------------------------
    def event(self, kind, **data):
        rec = {"time": dt.datetime.now().isoformat(timespec="seconds"), "run_id": self.run_id,
               "kind": kind, **_jsonable(data)}
        self._events.write(json.dumps(rec, default=str) + "\n")
        self._events.flush()

    def info(self, msg, **data):
        self.logger.info(msg)
        for h in self.logger.handlers:
            h.flush()
        if data:
            self.event("info", message=msg, **data)

    def warning(self, msg):
        self.logger.warning(msg)
        self.event("warning", message=msg)

    def metrics(self, kind, m):
        self.event(kind, **{k: v for k, v in m.items() if k not in ("scores", "labels")})

    def env(self, args):
        import torch
        gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
        info = {
            "argv": sys.argv, "args": vars(args), "cwd": os.getcwd(),
            "host": socket.gethostname(), "platform": platform.platform(),
            "python": sys.version.split()[0], "torch": torch.__version__,
            "cuda": torch.version.cuda, "gpu": gpu,
            "git_commit": _git("rev-parse", "HEAD"), "git_dirty": bool(_git("status", "--porcelain")),
        }
        self.info(f"command: {' '.join(sys.argv)}")
        self.info(f"git {info['git_commit'][:10]}{' (dirty)' if info['git_dirty'] else ''} | "
                  f"torch {info['torch']} cuda {info['cuda']} | gpu {gpu} | python {info['python']}")
        self.info("args: " + json.dumps(vars(args), default=str))
        self.event("env", **info)
        self._index({"status": "started", **{k: info[k] for k in ("argv", "git_commit", "gpu")}})

    # ------------------------------------------------------------------
    def _index(self, rec):
        with open(os.path.join(REPO_ROOT, "runs_index.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"time": dt.datetime.now().isoformat(timespec="seconds"),
                                "run_id": self.run_id, "name": self.name,
                                "out_dir": os.path.abspath(self.out_dir), **rec}, default=str) + "\n")

    def _excepthook(self, et, ev, tb):
        self.status = "crashed"
        self.logger.error("UNCAUGHT EXCEPTION:\n" + "".join(traceback.format_exception(et, ev, tb)))
        self.event("error", error=repr(ev), traceback="".join(traceback.format_exception(et, ev, tb)))
        self._prev_excepthook(et, ev, tb)

    def mark_done(self, summary=None):
        self.status = "completed"
        self.event("completed", **(summary or {}))

    def _finish(self):
        if self.status == "running":
            self.status = "interrupted"
        mins = (dt.datetime.now() - self.started).total_seconds() / 60
        self.logger.info(f"===== run {self.run_id} {self.status} after {mins:.1f} min =====")
        self.event("end", status=self.status, minutes=mins)
        self._index({"status": self.status, "minutes": round(mins, 1)})
        self._events.close()
