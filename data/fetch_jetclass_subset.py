"""
fetch_jetclass_subset.py — download a subset of JetClass ROOT files without
downloading the full multi-GB tar archives.

Zenodo serves the JetClass tars with HTTP range support, so we read the tar
headers remotely to find each member's byte offset, then fetch only the
.root files we want (each file holds 100k jets of a single class).

Example (1 train file/class = 1M jets, 1 val file/class, 1 test file/class):
    python data/fetch_jetclass_subset.py --out D:/JetClass --train_files 1 --val_files 1 --test_files 1
"""
import argparse
import os
import re
import sys
import urllib.request
from concurrent.futures import ThreadPoolExecutor

BASE = "https://zenodo.org/records/6619768/files/"
TARS = {
    "train": ["JetClass_Pythia_train_100M_part%d.tar" % i for i in range(10)],
    "val": ["JetClass_Pythia_val_5M.tar"],
    "test": ["JetClass_Pythia_test_20M.tar"],
}
CLASS_PREFIXES = ["ZJetsToNuNu", "HToBB", "HToCC", "HToGG", "HToWW4Q",
                  "HToWW2Q1L", "ZToQQ", "WToQQ", "TTBar", "TTBarLep"]
CHUNK = 8 * 1024 * 1024


def _range(url, start, end, timeout=120):
    req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end}"})
    return urllib.request.urlopen(req, timeout=timeout)


def index_tar(url, cache_path):
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return [(n, int(o), int(s)) for n, o, s in (l.rstrip("\n").split("\t") for l in f)]
    members, off = [], 0
    while True:
        h = _range(url, off, off + 511).read()
        if len(h) < 512 or h[:1] == b"\0":
            break
        name = h[:100].rstrip(b"\0").decode()
        prefix = h[345:500].rstrip(b"\0").decode()
        size = int(h[124:136].rstrip(b"\0 ").decode() or "0", 8)
        members.append((f"{prefix}/{name}" if prefix else name, off + 512, size))
        off += 512 + ((size + 511) // 512) * 512
    with open(cache_path, "w") as f:
        for n, o, s in members:
            f.write(f"{n}\t{o}\t{s}\n")
    return members


def download_member(url, offset, size, dest):
    tmp = dest + ".part"
    have = os.path.getsize(tmp) if os.path.exists(tmp) else 0
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        return dest
    while have < size:
        try:
            with _range(url, offset + have, offset + size - 1) as r, open(tmp, "ab") as f:
                while True:
                    buf = r.read(CHUNK)
                    if not buf:
                        break
                    f.write(buf)
                    have += len(buf)
        except Exception as e:  # resume on transient network errors
            print(f"[retry] {os.path.basename(dest)} at {have}/{size}: {e}", flush=True)
        have = os.path.getsize(tmp)
    os.replace(tmp, dest)
    print(f"[done] {dest} ({size / 1e6:.0f} MB)", flush=True)
    return dest


def select(members, prefix, n):
    pat = re.compile(rf"(^|/){prefix}_\d+\.root$")
    return sorted([m for m in members if pat.search(m[0])])[:n]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--train_files", type=int, default=1, help="files per class (100k jets each)")
    ap.add_argument("--val_files", type=int, default=1)
    ap.add_argument("--test_files", type=int, default=1)
    ap.add_argument("--workers", type=int, default=10)
    args = ap.parse_args()

    jobs = []
    for split, n in (("train", args.train_files), ("val", args.val_files), ("test", args.test_files)):
        if n <= 0:
            continue
        split_dir = os.path.join(args.out, split)
        os.makedirs(split_dir, exist_ok=True)
        needed = {p: n for p in CLASS_PREFIXES}
        for tar in TARS[split]:
            if not any(needed.values()):
                break
            url = BASE + tar
            members = index_tar(url, os.path.join(args.out, tar + ".index"))
            for p in CLASS_PREFIXES:
                for name, off, size in select(members, p, needed[p]):
                    jobs.append((url, off, size, os.path.join(split_dir, os.path.basename(name))))
                    needed[p] -= 1

    print(f"{len(jobs)} files, {sum(j[2] for j in jobs) / 1e9:.1f} GB total", flush=True)
    with ThreadPoolExecutor(args.workers) as ex:
        list(ex.map(lambda j: download_member(*j), jobs))
    print("ALL DOWNLOADS COMPLETE", flush=True)


if __name__ == "__main__":
    sys.exit(main())
