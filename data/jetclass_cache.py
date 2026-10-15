"""
jetclass_cache.py — memory-efficient JetClass pipeline for GDS-ParT.

Why: storing the dense (N, N, 4) pairwise tensor per jet costs ~256 KB/jet
at N=128 (~256 GB for 1M jets). Instead we cache only per-particle arrays
on disk (numpy memmaps) and compute the 4 pairwise features on the GPU per
batch from the particle 4-vectors, exactly like the official ParT does.

Cache layout (one directory per split, ragged = no padding stored):
    feats.bin    float16 (P, 17)  normalized particle features, all jets concatenated
    vectors.bin  float32 (P, 4)   (px, py, pz, E)
    offsets.npy  int64   (J+1,)   jet j owns particles offsets[j]:offsets[j+1]
    labels.npy   int64   (J,)
~1.8 KB/jet on disk (vs ~6.4 KB padded to 128).

Particle features (17), normalized as in the official JetClass_full.yaml
(x - shift) * scale, clipped to [-5, 5]:
    0 part_pt_log  1 part_e_log  2 part_logptrel  3 part_logerel  4 part_deltaR
    5 part_deta    6 part_dphi   7 part_charge
    8-12 isChargedHadron, isNeutralHadron, isPhoton, isElectron, isMuon
    13 tanh(d0val) 14 d0err (clip 0..1) 15 tanh(dzval) 16 dzerr (clip 0..1)
"""
import glob
import json
import os
import threading
import queue

import numpy as np

CLASS_PREFIXES = {
    "QCD":  "ZJetsToNuNu",
    "Hbb":  "HToBB",
    "Hcc":  "HToCC",
    "Hgg":  "HToGG",
    "H4q":  "HToWW4Q",
    "Hqql": "HToWW2Q1L",
    "Zqq":  "ZToQQ",
    "Wqq":  "WToQQ",
    "Tbqq": "TTBar",
    "Tbl":  "TTBarLep",
}
CLASS_LIST = list(CLASS_PREFIXES.keys())

BRANCHES = ["part_px", "part_py", "part_pz", "part_energy", "part_deta", "part_dphi",
            "part_charge", "part_isChargedHadron", "part_isNeutralHadron",
            "part_isPhoton", "part_isElectron", "part_isMuon",
            "part_d0val", "part_d0err", "part_dzval", "part_dzerr",
            "jet_pt", "jet_energy"]


def _files_for_class(data_dir, prefix):
    # "TTBar_*" would also match "TTBarLep_*", so require digits after the prefix
    files = glob.glob(os.path.join(data_dir, "**", f"{prefix}_*.root"), recursive=True)
    return sorted(f for f in files if os.path.basename(f)[len(prefix) + 1:-5].isdigit())


def _process_chunk(a, max_particles):
    """a: awkward record array of BRANCHES for a chunk of jets.
    Returns flat per-particle arrays (first `max_particles` per jet, which
    are the highest-pt ones) plus per-jet particle counts."""
    import awkward as ak
    counts = ak.to_numpy(ak.num(a["part_energy"][:, :max_particles])).astype(np.int64)
    F = lambda b: ak.to_numpy(ak.flatten(a[b][:, :max_particles])).astype(np.float32)
    px, py, pz, e = F("part_px"), F("part_py"), F("part_pz"), F("part_energy")
    pt = np.hypot(px, py)
    jet_pt = np.repeat(np.asarray(a["jet_pt"], dtype=np.float32), counts)
    jet_e = np.repeat(np.asarray(a["jet_energy"], dtype=np.float32), counts)
    deta, dphi = F("part_deta"), F("part_dphi")

    with np.errstate(divide="ignore", invalid="ignore"):
        cols = [
            (np.log(pt) - 1.7) * 0.7,
            (np.log(e) - 2.0) * 0.7,
            (np.log(pt / jet_pt) + 4.7) * 0.7,
            (np.log(e / jet_e) + 4.7) * 0.7,
            (np.hypot(deta, dphi) - 0.2) * 4.0,
            deta, dphi, F("part_charge"),
            F("part_isChargedHadron"), F("part_isNeutralHadron"), F("part_isPhoton"),
            F("part_isElectron"), F("part_isMuon"),
            np.tanh(F("part_d0val")), np.clip(F("part_d0err"), 0, 1),
            np.tanh(F("part_dzval")), np.clip(F("part_dzerr"), 0, 1),
        ]
    feats = np.stack(cols, axis=-1)
    feats = np.nan_to_num(np.clip(feats, -5, 5), nan=0.0, posinf=0.0, neginf=0.0)
    vectors = np.stack([px, py, pz, e], axis=-1)
    return feats.astype(np.float16), vectors, counts


def build_cache(data_dir, cache_dir, max_particles=128, jets_per_class=None, chunk=100_000):
    """Convert JetClass ROOT files in `data_dir` into ragged memmaps in
    `cache_dir`. Particles beyond `max_particles` (pt-sorted) are dropped.
    Skips work if a complete cache already exists."""
    import uproot

    done_flag = os.path.join(cache_dir, "DONE")
    if os.path.exists(done_flag):
        return cache_dir
    os.makedirs(cache_dir, exist_ok=True)

    plan = []  # (label, file, n_entries)
    for label, prefix in enumerate(CLASS_PREFIXES.values()):
        remaining = jets_per_class if jets_per_class else float("inf")
        files = _files_for_class(data_dir, prefix)
        if not files:
            print(f"[warn] no files for {CLASS_LIST[label]} ({prefix}) in {data_dir}")
        for f in files:
            if remaining <= 0:
                break
            with uproot.open(f) as rf:
                n = min(rf["tree"].num_entries, remaining)
            plan.append((label, f, int(n)))
            remaining -= n
    n_jets = sum(p[2] for p in plan)
    print(f"[cache] {data_dir} -> {cache_dir}: {n_jets:,} jets from {len(plan)} files", flush=True)

    feats_path = os.path.join(cache_dir, "feats.bin")
    vec_path = os.path.join(cache_dir, "vectors.bin")
    counts = np.empty(n_jets, dtype=np.int64)
    labels = np.empty(n_jets, dtype=np.int64)
    jpos, ppos = 0, 0
    with open(feats_path, "wb") as ff, open(vec_path, "wb") as fv:
        for label, f, n in plan:
            with uproot.open(f) as rf:
                for a in rf["tree"].iterate(BRANCHES, entry_stop=n, step_size=chunk, library="ak"):
                    feats, vectors, c = _process_chunk(a, max_particles)
                    ff.write(feats.tobytes())
                    fv.write(vectors.tobytes())
                    k = len(c)
                    counts[jpos:jpos + k] = c
                    labels[jpos:jpos + k] = label
                    jpos += k
                    ppos += len(feats)
            print(f"  [{CLASS_LIST[label]}] {os.path.basename(f)}: {n:,} jets", flush=True)
    offsets = np.zeros(n_jets + 1, dtype=np.int64)
    np.cumsum(counts, out=offsets[1:])
    np.save(os.path.join(cache_dir, "offsets.npy"), offsets)
    np.save(os.path.join(cache_dir, "labels.npy"), labels)
    with open(os.path.join(cache_dir, "meta.json"), "w") as f:
        json.dump({"n_jets": n_jets, "n_particles": int(ppos), "max_particles": max_particles,
                   "classes": CLASS_LIST}, f)
    open(done_flag, "w").close()
    return cache_dir


class CachedJetLoader:
    """Batch iterator over a ragged memmapped cache. Each batch is padded
    only to the longest jet in it, which cuts the O(N^2) pairwise cost
    substantially. A background thread prefetches batches."""

    def __init__(self, cache_dir, batch_size, shuffle, max_jets=None, seed=0, prefetch=6):
        self.offsets = np.load(os.path.join(cache_dir, "offsets.npy"))
        self.labels = np.load(os.path.join(cache_dir, "labels.npy"))
        n_part = int(self.offsets[-1])
        self.feats = np.memmap(os.path.join(cache_dir, "feats.bin"), dtype=np.float16,
                               mode="r", shape=(n_part, 17))
        self.vectors = np.memmap(os.path.join(cache_dir, "vectors.bin"), dtype=np.float32,
                                 mode="r", shape=(n_part, 4))
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self.prefetch = prefetch
        n = len(self.labels)
        if max_jets and max_jets < n:
            self.index = np.sort(np.random.default_rng(1234).choice(n, max_jets, replace=False))
        else:
            self.index = np.arange(n)

    def __len__(self):
        return (len(self.index) + self.batch_size - 1) // self.batch_size

    @staticmethod
    def _pad(feats, vec, off, labels, sel):
        """Build a padded batch from in-memory ragged arrays. `off` holds the
        local particle offsets of the jets in the pool; `sel` picks jets."""
        start = off[sel]
        cnt = off[sel + 1] - start
        n = max(int(cnt.max()), 2)
        ar = np.arange(n)
        mask = ar[None, :] < cnt[:, None]
        rows = (start[:, None] + ar[None, :])[mask]
        f = np.zeros((len(sel), n, 17), dtype=np.float32)
        v = np.zeros((len(sel), n, 4), dtype=np.float32)
        f[mask] = feats[rows]
        v[mask] = vec[rows]
        return f, v, mask, labels[sel]

    def _read_block(self, j0, j1):
        """Sequential read of jets [j0, j1) -> (feats, vec, local offsets, labels)."""
        p0, p1 = self.offsets[j0], self.offsets[j1]
        return (np.asarray(self.feats[p0:p1]), np.asarray(self.vectors[p0:p1]),
                self.offsets[j0:j1 + 1] - p0, self.labels[j0:j1])

    def _batches(self):
        bs = self.batch_size
        if not self.shuffle:
            # contiguous index ranges are read sequentially
            idx = self.index
            for s in range(0, len(idx), bs):
                sel = idx[s:s + bs]
                if sel[-1] - sel[0] == len(sel) - 1:
                    f, v, off, y = self._read_block(sel[0], sel[-1] + 1)
                    yield self._pad(f, v, off, y, np.arange(len(sel)))
                else:
                    yield self._pad(self.feats, self.vectors, self.offsets, self.labels, sel)
            return

        # Disk-friendly shuffling for caches larger than RAM: the dataset is
        # split into contiguous blocks of `block` jets; a pool is built from
        # `blocks_per_pool` randomly chosen blocks (each one a sequential
        # read), then jets inside the pool are shuffled. Within a pool, jets
        # are bucketed by multiplicity so each batch is padded only to a
        # similar length, which cuts the O(N^2) pairwise cost.
        block, blocks_per_pool = 256, 256
        n = len(self.labels)
        if len(self.index) != n:
            raise NotImplementedError("max_jets subsampling is only supported for eval loaders")
        starts = self.rng.permutation(np.arange(0, n, block))
        for g in range(0, len(starts), blocks_per_pool):
            parts = [self._read_block(j0, min(j0 + block, n)) for j0 in starts[g:g + blocks_per_pool]]
            feats = np.concatenate([p[0] for p in parts])
            vec = np.concatenate([p[1] for p in parts])
            counts = np.concatenate([np.diff(p[2]) for p in parts])
            labels = np.concatenate([p[3] for p in parts])
            off = np.zeros(len(counts) + 1, dtype=np.int64)
            np.cumsum(counts, out=off[1:])
            perm = self.rng.permutation(len(counts))
            perm = perm[np.argsort(counts[perm], kind="stable")]
            batches = [perm[k:k + bs] for k in range(0, len(perm), bs)]
            for i in self.rng.permutation(len(batches)):
                yield self._pad(feats, vec, off, labels, batches[i])

    def __iter__(self):
        import torch
        q = queue.Queue(maxsize=self.prefetch)
        stop = object()

        def worker():
            for item in self._batches():
                q.put(item)
            q.put(stop)

        threading.Thread(target=worker, daemon=True).start()
        while True:
            item = q.get()
            if item is stop:
                return
            yield tuple(torch.from_numpy(x) for x in item)


def pairwise_features(vectors, mask, eps=1e-8):
    """GPU computation of ParT's 4 interaction features for every pair.
    vectors: (B, N, 4) float tensor of (px, py, pz, E); mask: (B, N) bool.
    Returns (B, N, N, 4): ln(delta), ln(kt), ln(z), ln(m2). Self-pairs and
    pairs involving padding are set to 0."""
    import torch
    v = vectors.float()
    px, py, pz, e = v.unbind(-1)
    pt = torch.sqrt(px ** 2 + py ** 2).clamp(min=eps)
    rap = 0.5 * torch.log(((e + pz).clamp(min=eps)) / ((e - pz).clamp(min=eps)))
    phi = torch.atan2(py, px)

    drap = rap[:, :, None] - rap[:, None, :]
    dphi = torch.remainder(phi[:, :, None] - phi[:, None, :] + torch.pi, 2 * torch.pi) - torch.pi
    delta = torch.sqrt(drap ** 2 + dphi ** 2).clamp(min=eps)
    pt_min = torch.minimum(pt[:, :, None], pt[:, None, :])
    kt = (pt_min * delta).clamp(min=eps)
    z = (pt_min / (pt[:, :, None] + pt[:, None, :])).clamp(min=eps)
    s = v[:, :, None, :] + v[:, None, :, :]
    m2 = (s[..., 3] ** 2 - s[..., 0] ** 2 - s[..., 1] ** 2 - s[..., 2] ** 2).clamp(min=eps)

    out = torch.stack([torch.log(delta), torch.log(kt), torch.log(z), torch.log(m2)], dim=-1)
    n = mask.shape[1]
    valid = mask[:, :, None] & mask[:, None, :] & ~torch.eye(n, dtype=torch.bool, device=mask.device)
    return out * valid[..., None]
