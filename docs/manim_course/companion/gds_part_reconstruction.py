"""Build the report's local GDS-ParT from elementary PyTorch modules.

This is an independent implementation, not an import of ``gds_part_colab``.
Read in order: configuration -> embeddings -> relation summaries -> gates ->
attention -> residual blocks -> pooling -> classifier -> physical pair features.

Axes throughout are B=batch, N=particles, D=particle channels, P=pair channels,
H=heads and d=D/H. ``keep[b, i]`` means particle i is real. Inputs must be
finite and each jet nonempty; masking is NOT a general NaN sanitizer.

The mathematics and trainable tensors follow GDS-ParT_Technical_Report.tex
and the local production model. Explicit attention replaces production SDPA
so its scores and probabilities can be inspected. Activation checkpointing is
omitted: it saves activation memory, not parameters. Identical state dictionaries
give numerically close (not promised bitwise-identical) CPU evaluation logits.
"""
from dataclasses import dataclass
import math
import torch
from torch import nn
from torch.nn import functional as F


@dataclass
class Config:
    n_particle_features: int = 17
    n_pairwise_features: int = 4
    n_classes: int = 10
    embed_dim: int = 128
    n_heads: int = 8
    n_layers: int = 8
    ffn_dim: int = 512
    pairwise_embed_dim: int = 64
    dropout: float = .1
    max_particles: int = 128
    gate_mode: str = "identity"
    per_layer_gate: bool = True
    pair_bias: bool = True
    n_cls_layers: int = 2

    def __post_init__(self):
        sizes = (self.n_particle_features, self.n_pairwise_features, self.n_classes,
                 self.embed_dim, self.n_heads, self.ffn_dim, self.pairwise_embed_dim,
                 self.max_particles)
        if any(size <= 0 for size in sizes) or min(self.n_layers, self.n_cls_layers) < 0:
            raise ValueError("Widths must be positive and layer counts nonnegative.")
        if self.embed_dim % self.n_heads:
            raise ValueError("The particle width must divide into whole heads.")
        if self.gate_mode not in ("identity", "sigmoid", "none"):
            raise ValueError("Unknown gate activation.")
        if not 0 <= self.dropout < 1:
            raise ValueError("Dropout must be in [0, 1).")


def configurations():
    """Return fresh configurations, including the report's six-block v1.

    Production's four legacy switches recover v1 *operations*, but still leave
    n_layers=8 unless explicitly changed. Six blocks give the historical count.
    max_particles documents the cache cap; it is not a positional embedding or
    a forward-time truncation. Widths, counts and ordering are not inferred.
    """
    return {
        "v1": Config(n_layers=6, gate_mode="sigmoid",
                     per_layer_gate=False, pair_bias=False, n_cls_layers=0),
        "shared": Config(per_layer_gate=False),
        "conditioned": Config(),
        "no_gates": Config(gate_mode="none", per_layer_gate=False),
    }


class ParticleEmbedding(nn.Module):
    """LN(F) -> Linear(F,D) -> GELU -> Linear(D,D) -> Dropout.

    LayerNorm acts on the last (feature) axis independently for each particle,
    using population variance and epsilon=1e-5. No particle positions are mixed.
    The 17 raw/preprocessed channels are not the separate four-vector input.
    """
    def __init__(self, features, width, dropout=.1):
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(features), nn.Linear(features, width),
                                 nn.GELU(), nn.Linear(width, width), nn.Dropout(dropout))

    def forward(self, particles):
        # Linear transforms only the last axis: every particle shares the weights.
        return self.net(particles)


class PairwiseEmbedding(nn.Module):
    """Map each four-feature pair independently to P channels.

    A zero raw self/padded pair need not embed to zero because both affine
    layers have biases. Self relations remain in the neighbour summary; only
    padded neighbours are excluded. There is no pair LayerNorm or dropout.
    """
    def __init__(self, features, width):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(features, width), nn.GELU(),
                                 nn.Linear(width, width))

    def forward(self, pairs):
        return self.net(pairs)


class RelationalGate(nn.Module):
    """Convert R[B,N,N,P] into two channel gates of shape [B,N,D].

    Shared mode computes s_i=sum_j keep_j R_ij / sum_j keep_j once for all
    blocks. Conditioned mode computes c_i=Linear(LN(X_i)) and
    alpha_ij=softmax_j(<R_ij,c_i>/sqrt(P)), then s_i=sum_j alpha_ij R_ij.
    Query/key MLPs have separate parameters: P -> D -> D with one GELU.

    Identity activation 1+tanh starts at exactly one because the *last* affine
    layers are zero-initialized. It is not a sigmoid centered on 1/2.
    With finite real arithmetic the range is (0,2); floating point may saturate
    to an endpoint. Zero last-layer weights initially block gradients to the
    earlier gate/context layers, but the last layers themselves can learn.
    """
    def __init__(self, relation_width, width, mode="identity", conditioned=False):
        super().__init__()
        if mode not in ("identity", "sigmoid"):
            raise ValueError("A gate module supports identity or sigmoid; none omits it.")
        self.mode = mode
        self.scale = relation_width ** -.5
        self.ctx = nn.Linear(width, relation_width) if conditioned else None
        self.query_gate = self.make_mlp(relation_width, width)
        self.key_gate = self.make_mlp(relation_width, width)

    def make_mlp(self, incoming, outgoing):
        network = nn.Sequential(nn.Linear(incoming, outgoing), nn.GELU(),
                                nn.Linear(outgoing, outgoing))
        if self.mode == "identity":
            # tanh(0)=0, so both gates begin exactly at one, not at one half.
            nn.init.zeros_(network[-1].weight)
            nn.init.zeros_(network[-1].bias)
        return network

    def aggregate(self, relations, keep, states=None):
        if self.ctx is None:
            weights = keep[:, None, :, None].to(relations.dtype)
            return (relations * weights).sum(2) / weights.sum(2).clamp_min(1)
        if states is None:
            raise ValueError("Conditioned gates need the current normalized states.")
        context = self.ctx(states).to(relations.dtype)
        scores = (relations * context[:, :, None, :]).sum(-1).float() * self.scale
        scores = scores.masked_fill(~keep[:, None, :], -torch.inf)
        weights = scores.softmax(-1).to(relations.dtype)
        return (weights[..., None] * relations).sum(2)

    def forward(self, relations, keep, states=None):
        summary = self.aggregate(relations, keep, states)
        activation = (lambda z: 1 + z.tanh()) if self.mode == "identity" else torch.sigmoid
        return activation(self.query_gate(summary)), activation(self.key_gate(summary))


class GatedAttention(nn.Module):
    """Q'=Linear(X)*gq, K'=Linear(X)*gk, V=Linear(X) (ungated).

    Split D into H heads of d channels, then compute
    softmax(Q'K'^T/sqrt(d) + bias + padding) V and concatenate heads.
    Pair bias has [B,H,N,N] axes. Padding excludes key columns, not query
    rows. Returned weights are pre-dropout row probabilities. Dropout affects
    weights in training only; evaluation is used for symmetry tests.
    """
    def __init__(self, width, heads, dropout=.1):
        super().__init__()
        if heads <= 0 or width % heads:
            raise ValueError("Attention width must be divisible by positive heads.")
        self.heads, self.head_width = heads, width // heads
        self.q_proj = nn.Linear(width, width)
        self.k_proj = nn.Linear(width, width)
        self.v_proj = nn.Linear(width, width)
        self.out_proj = nn.Linear(width, width)
        self.dropout = nn.Dropout(dropout)

    def split(self, tensor):
        batch, particles, width = tensor.shape
        return tensor.reshape(batch, particles, self.heads, self.head_width).transpose(1, 2)

    def forward(self, states, gates, bias, keep, return_weights=False):
        query, key = self.q_proj(states), self.k_proj(states)
        if gates is not None:
            query, key = query * gates[0], key * gates[1]
        query, key, value = self.split(query), self.split(key), self.split(self.v_proj(states))
        scores = query @ key.transpose(-2, -1) / math.sqrt(self.head_width)
        if bias is not None:
            # Match production's additive mask: +inf + -inf can still be NaN.
            bias = bias.to(query.dtype).masked_fill(~keep[:, None, None, :], -torch.inf)
            scores = scores + bias
        else:
            scores = scores.masked_fill(~keep[:, None, None, :], -torch.inf)
        # Padding masks keys. Padded query rows may exist, but cannot influence real keys.
        weights = scores.softmax(-1)
        mixed = self.dropout(weights) @ value
        mixed = mixed.transpose(1, 2).contiguous().reshape_as(states)
        output = self.out_proj(mixed)
        return (output, weights) if return_weights else output


def feed_forward(cfg):
    return nn.Sequential(nn.Linear(cfg.embed_dim, cfg.ffn_dim), nn.GELU(),
                         nn.Dropout(cfg.dropout), nn.Linear(cfg.ffn_dim, cfg.embed_dim),
                         nn.Dropout(cfg.dropout))


class GDSParTBlock(nn.Module):
    """Pre-LayerNorm residual recurrence, without modifying relation states.

    H=LN1(X); X <- X+Attention(H, Gate(R,H), bias);
    X <- X+FFN(LN2(X)). FFN is D->ffn_dim->D with GELU and two
    dropout operations. Shared gates are passed in, conditioned gates belong
    to this block, and no-gate blocks have no gate parameters.
    """
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(cfg.embed_dim), nn.LayerNorm(cfg.embed_dim)
        self.attn = GatedAttention(cfg.embed_dim, cfg.n_heads, cfg.dropout)
        self.ffn = feed_forward(cfg)
        self.gate = (RelationalGate(cfg.pairwise_embed_dim, cfg.embed_dim, cfg.gate_mode, True)
                     if cfg.per_layer_gate and cfg.gate_mode != "none" else None)

    def forward(self, states, relations, shared_gates, bias, keep):
        normalized = self.norm1(states)
        gates = self.gate(relations, keep, normalized) if self.gate is not None else shared_gates
        states = states + self.attn(normalized, gates, bias, keep)
        return states + self.ffn(self.norm2(states))


class ClassAttentionBlock(nn.Module):
    """Update only a [B,1,D] CLS token, leaving [B,N,D] particles unchanged.

    Normalize [CLS; X], query with its first row and use every normalized row
    as keys/values. Thus CLS can attend to itself. No relation gates or pair
    bias enter class attention. PyTorch MHA uses True=IGNORE, opposite to keep.
    """
    def __init__(self, cfg):
        super().__init__()
        self.norm1, self.norm2 = nn.LayerNorm(cfg.embed_dim), nn.LayerNorm(cfg.embed_dim)
        self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.n_heads,
                                          dropout=cfg.dropout, batch_first=True)
        self.ffn = feed_forward(cfg)

    def forward(self, states, cls, keep):
        normalized = self.norm1(torch.cat((cls, states), 1))
        valid = torch.cat((torch.ones_like(keep[:, :1]), keep), 1)
        # MultiheadAttention reverses the boolean convention: True means ignore.
        update, _ = self.attn(normalized[:, :1], normalized, normalized,
                             key_padding_mask=~valid, need_weights=False)
        cls = cls + update
        return cls + self.ffn(self.norm2(cls))


class GDSParT(nn.Module):
    """Two static input embeddings, L particle blocks and invariant readout.

    R and pair bias are computed once. Shared gates are also computed once;
    conditioned gates read the same R but the changing normalized X.
    With CLS pooling, only the final CLS vector is normalized/classified.
    With mean pooling, final normalization occurs BEFORE the masked mean.
    Classifier output is raw logits; cross_entropy performs log-softmax.
    """
    def __init__(self, cfg=None):
        super().__init__()
        self.cfg = cfg or Config()
        c = self.cfg
        self.particle_embed = ParticleEmbedding(c.n_particle_features, c.embed_dim, c.dropout)
        self.pairwise_embed = PairwiseEmbedding(c.n_pairwise_features, c.pairwise_embed_dim)
        self.rel_gate = (RelationalGate(c.pairwise_embed_dim, c.embed_dim, c.gate_mode)
                         if not c.per_layer_gate and c.gate_mode != "none" else None)
        self.pair_bias = nn.Linear(c.pairwise_embed_dim, c.n_heads) if c.pair_bias else None
        self.blocks = nn.ModuleList(GDSParTBlock(c) for _ in range(c.n_layers))
        if c.n_cls_layers:
            self.cls_token = nn.Parameter(torch.empty(1, 1, c.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=.02)
            self.cls_blocks = nn.ModuleList(ClassAttentionBlock(c) for _ in range(c.n_cls_layers))
        self.final_norm = nn.LayerNorm(c.embed_dim)
        self.classifier = nn.Sequential(nn.Linear(c.embed_dim, c.embed_dim), nn.GELU(),
                                        nn.Dropout(c.dropout), nn.Linear(c.embed_dim, c.n_classes))

    def forward(self, particles, pairs, keep=None, return_trace=False):
        batch, count, features = particles.shape
        if features != self.cfg.n_particle_features:
            raise ValueError("Unexpected particle feature count.")
        if pairs.shape != (batch, count, count, self.cfg.n_pairwise_features):
            raise ValueError("Pairs need both particle axes in the same order.")
        if keep is None:
            keep = torch.ones((batch, count), dtype=torch.bool, device=particles.device)
        if keep.dtype != torch.bool or keep.shape != (batch, count) or not keep.any(1).all():
            raise ValueError("Each jet needs at least one real particle and a boolean keep mask.")
        if not torch.isfinite(particles).all() or not torch.isfinite(pairs).all():
            raise ValueError("Supply finite inputs, including finite padding.")
        states = self.particle_embed(particles)
        relations = self.pairwise_embed(pairs)
        bias = self.pair_bias(relations).permute(0, 3, 1, 2) if self.pair_bias is not None else None
        gates = self.rel_gate(relations, keep) if self.rel_gate is not None else None
        trace = {"particles": list(states.shape), "relations": list(relations.shape),
                 "blocks": [], "relation_evaluations": 1}
        for block in self.blocks:
            states = block(states, relations, gates, bias, keep)
            trace["blocks"].append(list(states.shape))
        if self.cfg.n_cls_layers:
            cls = self.cls_token.expand(batch, -1, -1)
            for block in self.cls_blocks:
                cls = block(states, cls, keep)
            pooled = self.final_norm(cls[:, 0])
        else:
            normalized = self.final_norm(states)
            pooled = (normalized * keep[..., None]).sum(1) / keep.sum(1, keepdim=True)
        logits = self.classifier(pooled)
        trace["logits"] = list(logits.shape)
        return (logits, trace) if return_trace else logits


def pairwise_features(vectors, keep, eps=1e-8):
    """Reproduce cache arithmetic, including its finite-precision caveats.

    Four-vectors are (px,py,pz,E), not the 17 classifier input features.
    Use rapidity, not pseudorapidity; wrap delta phi to [-pi,pi).
    Return ln(delta), ln(min(pt_i,pt_j)*delta), ln(z), ln(mass_squared).
    Floors occur at the same intermediate operations as jetclass_cache.
    Self/padded pairs are multiplied by zero at the end, not sanitized with
    nan_to_num. Consequently NaN*0 remains NaN, and squared norms can overflow
    for very large finite float32 values. These are source semantics, not an
    assertion of robustness to arbitrary invalid physical inputs.
    """
    v = vectors.float()
    px, py, pz, energy = v.unbind(-1)
    pt = torch.sqrt(px ** 2 + py ** 2).clamp_min(eps)
    rapidity = .5 * ((energy + pz).clamp_min(eps) / (energy - pz).clamp_min(eps)).log()
    phi = torch.atan2(py, px)
    dy = rapidity[:, :, None] - rapidity[:, None, :]
    dphi = (phi[:, :, None] - phi[:, None, :] + math.pi).remainder(2 * math.pi) - math.pi
    distance = torch.sqrt(dy ** 2 + dphi ** 2).clamp_min(eps)
    lesser = torch.minimum(pt[:, :, None], pt[:, None, :])
    kt = (lesser * distance).clamp_min(eps)
    z = (lesser / (pt[:, :, None] + pt[:, None, :])).clamp_min(eps)
    total = v[:, :, None] + v[:, None, :]
    mass2 = (total[..., 3] ** 2 - total[..., 0] ** 2 -
             total[..., 1] ** 2 - total[..., 2] ** 2).clamp_min(eps)
    result = torch.stack((distance.log(), kt.log(), z.log(), mass2.log()), -1)
    diagonal = torch.eye(keep.shape[1], dtype=torch.bool, device=keep.device)
    valid = keep[:, :, None] & keep[:, None, :] & ~diagonal
    return result * valid[..., None]


# Keep the earlier review notebook imports working; neither alias wraps production.
ParticleBlock = GDSParTBlock
GDSParTConfig = Config
pair_features = pairwise_features


def parameter_audit():
    return {name: sum(p.numel() for p in GDSParT(cfg).parameters())
            for name, cfg in configurations().items()}


def shared_breakdown():
    model = GDSParT(configurations()["shared"])
    count = lambda module: sum(p.numel() for p in module.parameters())
    return {
        "particle_embedding": count(model.particle_embed),
        "pair_embedding": count(model.pairwise_embed), "gates": count(model.rel_gate),
        "bias": count(model.pair_bias),
        "attention": sum(count(b.attn) for b in model.blocks),
        "block_norms": sum(count(b.norm1) + count(b.norm2) for b in model.blocks),
        "feed_forward": sum(count(b.ffn) for b in model.blocks),
        "cls_token": model.cls_token.numel(), "cls_blocks": count(model.cls_blocks),
        "final_norm": count(model.final_norm), "classifier": count(model.classifier),
    }


def cpu_demo(steps=140):
    """Overfit eight synthetic examples: a plumbing test, never a physics result."""
    if steps < 1:
        raise ValueError("At least one optimizer step is required.")
    torch.manual_seed(7)
    torch.set_num_threads(2)
    cfg = Config(embed_dim=16, pairwise_embed_dim=8, ffn_dim=32, n_heads=2,
                 n_layers=1, n_cls_layers=1, n_classes=2, dropout=0)
    model = GDSParT(cfg)
    particles = torch.randn(8, 4, 17)
    pairs = torch.randn(8, 4, 4, 4)
    keep = torch.ones(8, 4, dtype=torch.bool)
    labels = torch.arange(8) % 2
    optimizer = torch.optim.AdamW(model.parameters(), lr=.015, weight_decay=.01)
    history = []
    model.train()
    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        loss = F.cross_entropy(model(particles, pairs, keep), labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1)
        optimizer.step()
        if step % 20 == 0:
            history.append({"step": step, "loss": float(loss.detach())})
    model.eval()
    with torch.no_grad():
        logits = model(particles, pairs, keep)
        final = float(F.cross_entropy(logits, labels))
        accuracy = float((logits.argmax(-1) == labels).float().mean())
    return {"kind": "synthetic memorisation only", "steps": steps, "history": history,
            "final_loss": final, "training_accuracy": accuracy}


if __name__ == "__main__":
    import json
    print(json.dumps({"parameters": parameter_audit(), "shared_breakdown": shared_breakdown(),
                      "cpu_demo": cpu_demo()}, indent=2))
