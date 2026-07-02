"""
jetclass_dataset.py — loads real JetClass ROOT files and computes the
17 particle-level features and 4 pairwise interaction features used by
GDS-ParT, following the standard ParT/ParticleNet feature conventions.

Particle features (17), all jet-axis relative:
    0  part_pt_log        = log(pt)
    1  part_e_log         = log(energy)
    2  part_logptrel      = log(pt / jet_pt)
    3  part_logerel       = log(energy / jet_energy)
    4  part_deltaR        = sqrt(deta^2 + dphi^2)
    5  part_deta
    6  part_dphi
    7  part_charge
    8  part_isChargedHadron
    9  part_isNeutralHadron
    10 part_isPhoton
    11 part_isElectron
    12 part_isMuon
    13 part_d0val
    14 part_d0err
    15 part_dzval
    16 part_dzerr

Pairwise features (4), computed per particle pair (i, j) within a jet:
    0  ln(delta)   = ln(deltaR_ij)
    1  ln(kt)      = ln(min(pti, ptj) * deltaR_ij)
    2  ln(z)       = ln(min(pti, ptj) / (pti + ptj))
    3  ln(m2)      = ln(invariant mass^2 of the pair), clipped to be finite

These 4 features match the standard "interaction" input block used in the
official ParT / weaver feature config.
"""
import glob
import os

import numpy as np

PARTICLE_ID_TO_ONEHOT = {
    # (isChargedHadron, isNeutralHadron, isPhoton, isElectron, isMuon)
    211: (1, 0, 0, 0, 0), -211: (1, 0, 0, 0, 0),
    321: (1, 0, 0, 0, 0), -321: (1, 0, 0, 0, 0),
    2212: (1, 0, 0, 0, 0), -2212: (1, 0, 0, 0, 0),
    130: (0, 1, 0, 0, 0), 2112: (0, 1, 0, 0, 0), -2112: (0, 1, 0, 0, 0),
    22: (0, 0, 1, 0, 0),
    11: (0, 0, 0, 1, 0), -11: (0, 0, 0, 1, 0),
    13: (0, 0, 0, 0, 1), -13: (0, 0, 0, 0, 1),
}

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
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_LIST)}


def _onehot_pid(pid_array):
    out = np.zeros((len(pid_array), 5), dtype=np.float32)
    for i, pid in enumerate(pid_array):
        out[i] = PARTICLE_ID_TO_ONEHOT.get(int(pid), (0, 0, 0, 0, 0))
    return out


def build_particle_features(part, jet_pt, jet_energy, max_particles):
    """part: dict of numpy arrays (raw per-particle branches) for ONE jet.
    Returns (features (max_particles, 17), mask (max_particles,))."""
    n = len(part["part_energy"])
    n_keep = min(n, max_particles)

    px, py = part["part_px"][:n_keep], part["part_py"][:n_keep]
    pz, e = part["part_pz"][:n_keep], part["part_energy"][:n_keep]
    deta, dphi = part["part_deta"][:n_keep], part["part_dphi"][:n_keep]
    charge = part["part_charge"][:n_keep]
    pid = part["part_pid"][:n_keep]
    d0 = part.get("part_d0val", np.zeros(n_keep))[:n_keep]
    d0err = part.get("part_d0err", np.zeros(n_keep))[:n_keep]
    dz = part.get("part_dzval", np.zeros(n_keep))[:n_keep]
    dzerr = part.get("part_dzerr", np.zeros(n_keep))[:n_keep]

    pt = np.sqrt(px**2 + py**2)
    eps = 1e-6

    pt_log = np.log(pt + eps)
    e_log = np.log(e + eps)
    logptrel = np.log(pt / max(jet_pt, eps) + eps)
    logerel = np.log(e / max(jet_energy, eps) + eps)
    deltaR = np.sqrt(deta**2 + dphi**2)

    onehot = _onehot_pid(pid)  # (n_keep, 5)

    feats = np.zeros((max_particles, 17), dtype=np.float32)
    feats[:n_keep, 0] = pt_log
    feats[:n_keep, 1] = e_log
    feats[:n_keep, 2] = logptrel
    feats[:n_keep, 3] = logerel
    feats[:n_keep, 4] = deltaR
    feats[:n_keep, 5] = deta
    feats[:n_keep, 6] = dphi
    feats[:n_keep, 7] = charge
    feats[:n_keep, 8:13] = onehot
    feats[:n_keep, 13] = d0
    feats[:n_keep, 14] = d0err
    feats[:n_keep, 15] = dz
    feats[:n_keep, 16] = dzerr

    mask = np.zeros(max_particles, dtype=bool)
    mask[:n_keep] = True
    return feats, mask, (px, py, pz, e, n_keep)


def build_pairwise_features(px, py, pz, e, n_keep, max_particles):
    """Vectorized ln(delta), ln(kt), ln(z), ln(m2) for all pairs within a jet."""
    out = np.zeros((max_particles, max_particles, 4), dtype=np.float32)
    if n_keep < 2:
        return out

    px, py, pz, e = px[:n_keep], py[:n_keep], pz[:n_keep], e[:n_keep]
    pt = np.sqrt(px**2 + py**2) + 1e-8
    eta = np.arcsinh(pz / (pt + 1e-8))
    phi = np.arctan2(py, px)

    deta = eta[:, None] - eta[None, :]
    dphi = phi[:, None] - phi[None, :]
    dphi = (dphi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
    delta = np.sqrt(deta**2 + dphi**2) + 1e-8

    pt_i = pt[:, None] * np.ones((1, n_keep))
    pt_j = pt[None, :] * np.ones((n_keep, 1))
    pt_min = np.minimum(pt_i, pt_j)

    kt = pt_min * delta + 1e-8
    z = pt_min / (pt_i + pt_j + 1e-8) + 1e-8

    E_sum = e[:, None] + e[None, :]
    px_sum = px[:, None] + px[None, :]
    py_sum = py[:, None] + py[None, :]
    pz_sum = pz[:, None] + pz[None, :]
    m2 = np.clip(E_sum**2 - (px_sum**2 + py_sum**2 + pz_sum**2), 1e-8, None)

    ln_delta = np.log(delta)
    ln_kt = np.log(kt)
    ln_z = np.log(z)
    ln_m2 = np.log(m2)

    out[:n_keep, :n_keep, 0] = ln_delta
    out[:n_keep, :n_keep, 1] = ln_kt
    out[:n_keep, :n_keep, 2] = ln_z
    out[:n_keep, :n_keep, 3] = ln_m2
    return out


class JetClassDataset:
    """Minimal in-memory JetClass dataset. For 100M-jet-scale training,
    replace with an IterableDataset that streams via uproot.iterate() —
    this version is intended for validation-scale (5M) or subsampled use.
    """

    def __init__(self, data_dir: str, split_jets_per_class: int = 5000,
                 max_particles: int = 128, treename: str = "tree"):
        import uproot

        self.max_particles = max_particles
        self.samples = []  # list of (particle_feats, pairwise_feats, mask, label)

        for label, prefix in CLASS_PREFIXES.items():
            matches = sorted(glob.glob(
                os.path.join(data_dir, "**", f"{prefix}_*.root"), recursive=True))
            if not matches:
                print(f"[warn] no files for class {label} (prefix {prefix})")
                continue

            branches = ["part_px", "part_py", "part_pz", "part_energy",
                        "part_deta", "part_dphi", "part_charge", "part_pid",
                        "part_d0val", "part_d0err", "part_dzval", "part_dzerr"]

            n_loaded = 0
            for fpath in matches:
                if n_loaded >= split_jets_per_class:
                    break
                with uproot.open(fpath) as f:
                    tree = f[treename]
                    avail = [b for b in branches if b in tree.keys()]
                    arrays = tree.arrays(avail, entry_stop=split_jets_per_class - n_loaded,
                                          library="ak")

                n = len(arrays["part_energy"])
                for i in range(n):
                    part = {b: np.asarray(arrays[b][i]) for b in avail}
                    px, py = part["part_px"], part["part_py"]
                    pz, e = part["part_pz"], part["part_energy"]
                    jet_pt = float(np.sqrt(px.sum()**2 + py.sum()**2))
                    jet_energy = float(e.sum())

                    feats, mask, (px_k, py_k, pz_k, e_k, n_keep) = build_particle_features(
                        part, jet_pt, jet_energy, max_particles)
                    pair_feats = build_pairwise_features(
                        px_k, py_k, pz_k, e_k, n_keep, max_particles)

                    self.samples.append((feats, pair_feats, mask, CLASS_TO_IDX[label]))
                n_loaded += n
            print(f"[loaded] {label}: {n_loaded} jets")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]


def collate_fn(batch):
    import torch
    feats = torch.stack([torch.from_numpy(b[0]) for b in batch])
    pairs = torch.stack([torch.from_numpy(b[1]) for b in batch])
    masks = torch.stack([torch.from_numpy(b[2]) for b in batch])
    labels = torch.tensor([b[3] for b in batch], dtype=torch.long)
    return feats.float(), pairs.float(), masks.bool(), labels
