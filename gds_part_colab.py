"""
gds_part_colab.py — Gated Dual-Stream Particle Transformer (GDS-ParT)

A single-file, Colab/Jupyter-friendly PyTorch implementation.

Core idea (contrast with ParT):
    ParT:     Attention = Softmax( QK^T/sqrt(d) + psi(U) )        [additive logit bias]
    GDS-ParT: Q' = Q * gate_Q(U),  K' = K * gate_K(U)
              Attention = Softmax( Q'K'^T / sqrt(d) )              [multiplicative feature gating]

where U is the pairwise interaction tensor (B, N, N, 4) and gate_Q/gate_K are
learned, relation-aggregated, per-particle gates.

v2 (default config) extends this with:
    - identity-initialized gates 1 + tanh(.) in (0, 2) instead of sigmoid (0, 1)
    - per-layer gates whose neighbour aggregation is attention-weighted by the
      block's current particle state (instead of one uniform mean for all layers)
    - hybrid fusion: a ParT-style per-head pairwise logit bias alongside the gates
    - class-attention (CLS token) pooling instead of masked mean pooling
See GDSParTConfig for the switches that recover the v1 model.

Shapes:
    particle features:  (B, N, 17)
    pairwise features:  (B, N, N, 4)
    padding mask:        (B, N)      True = real particle, False = padded
    labels:              (B,)        10 jet classes

Run this file directly to execute a synthetic forward/backward sanity check:
    python gds_part_colab.py
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint


# ----------------------------------------------------------------------
# Config
# ----------------------------------------------------------------------
class GDSParTConfig:
    """v2 options (defaults = full v2 model):
        gate_mode       "identity": gate = 1 + tanh(.), zero-initialized so every gate
                        starts at exactly 1 and can amplify (up to 2x) or suppress channels.
                        "sigmoid": v1 behaviour, gate in (0, 1).
                        "none": no Q/K gating (ablation; with pair_bias this is plain ParT-style attention).
        per_layer_gate  each block owns a RelationalGate whose neighbour aggregation is
                        an attention-weighted sum conditioned on the block's current
                        particle state (v1: one uniform-mean gate shared by all blocks).
        pair_bias       also add a ParT-style per-head pairwise bias to the attention logits
                        (hybrid: multiplicative Q/K gating + additive logit bias).
        n_cls_layers    class-attention blocks with a learned CLS token for pooling
                        (0 = masked mean pooling, as in v1).
    Setting gate_mode="sigmoid", per_layer_gate=False, pair_bias=False, n_cls_layers=0
    gives the v1 architecture.
    """

    def __init__(
        self,
        n_particle_features: int = 17,
        n_pairwise_features: int = 4,
        n_classes: int = 10,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 8,
        ffn_dim: int = 512,
        pairwise_embed_dim: int = 64,
        dropout: float = 0.1,
        max_particles: int = 128,
        gate_mode: str = "identity",
        per_layer_gate: bool = True,
        pair_bias: bool = True,
        n_cls_layers: int = 2,
    ):
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
        assert gate_mode in ("identity", "sigmoid", "none")
        self.n_particle_features = n_particle_features
        self.n_pairwise_features = n_pairwise_features
        self.n_classes = n_classes
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.n_layers = n_layers
        self.ffn_dim = ffn_dim
        self.pairwise_embed_dim = pairwise_embed_dim
        self.dropout = dropout
        self.max_particles = max_particles
        self.gate_mode = gate_mode
        self.per_layer_gate = bool(per_layer_gate)
        self.pair_bias = bool(pair_bias)
        self.n_cls_layers = n_cls_layers


# ----------------------------------------------------------------------
# 1. Particle embedding
# ----------------------------------------------------------------------
class ParticleEmbedding(nn.Module):
    """Projects (B, N, 17) raw single-particle features into (B, N, embed_dim)."""

    def __init__(self, in_features: int, embed_dim: int, dropout: float = 0.1):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(in_features),
            nn.Linear(in_features, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


# ----------------------------------------------------------------------
# 2. Pairwise embedding
# ----------------------------------------------------------------------
class PairwiseEmbedding(nn.Module):
    """Projects (B, N, N, 4) pairwise physics features into a relational
    tensor of shape (B, N, N, pairwise_embed_dim)."""

    def __init__(self, in_features: int, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_features, out_dim),
            nn.GELU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, u):
        return self.net(u)


# ----------------------------------------------------------------------
# 3. Relational gate
# ----------------------------------------------------------------------
class RelationalGate(nn.Module):
    """Aggregates the (B, N, N, D) relational tensor over neighbours j into a
    per-particle summary and maps it to two gates (query and key).

    Aggregation:
        x is None -> masked uniform mean over neighbours (v1)
        x given   -> attention-weighted: alpha_ij = softmax_j(<rel_ij, W x_i> / sqrt(D)),
                     so each particle decides which relations matter given its
                     current representation.
    Gates:
        "identity" -> 1 + tanh(MLP(summary)), last layer zero-init (gate == 1 at init)
        "sigmoid"  -> sigmoid(MLP(summary))
    """

    def __init__(self, pairwise_dim: int, embed_dim: int, mode: str = "identity",
                 conditioned: bool = False):
        super().__init__()
        self.mode = mode
        self.scale = pairwise_dim ** -0.5
        self.ctx = nn.Linear(embed_dim, pairwise_dim) if conditioned else None
        self.query_gate = self._mlp(pairwise_dim, embed_dim)
        self.key_gate = self._mlp(pairwise_dim, embed_dim)

    def _mlp(self, d_in, d_out):
        last = nn.Linear(d_out, d_out)
        if self.mode == "identity":
            nn.init.zeros_(last.weight)
            nn.init.zeros_(last.bias)
        return nn.Sequential(nn.Linear(d_in, d_out), nn.GELU(), last)

    def _act(self, z):
        return 1.0 + torch.tanh(z) if self.mode == "identity" else torch.sigmoid(z)

    def forward(self, rel, key_padding_mask=None, x=None):
        """
        rel: (B, N, N, D), rel[b, i, j] = relation of particle i to neighbour j
        key_padding_mask: (B, N) bool, True = real particle
        x: (B, N, embed_dim) current particle state (only if conditioned)
        Returns gate_q, gate_k: (B, N, embed_dim)
        """
        if self.ctx is not None and x is not None:
            c = self.ctx(x).to(rel.dtype)                                  # (B, N, D)
            score = torch.einsum("bijd,bid->bij", rel, c).float() * self.scale
            if key_padding_mask is not None:
                score = score.masked_fill(~key_padding_mask[:, None, :], float("-inf"))
            alpha = torch.softmax(score, dim=-1).to(rel.dtype)             # (B, N, N)
            summary = torch.einsum("bij,bijd->bid", alpha, rel)            # (B, N, D)
        elif key_padding_mask is not None:
            m = key_padding_mask[:, None, :, None].to(rel.dtype)           # (B, 1, N, 1)
            denom = m.sum(dim=2).clamp(min=1.0)
            summary = (rel * m).sum(dim=2) / denom
        else:
            summary = rel.mean(dim=2)

        return self._act(self.query_gate(summary)), self._act(self.key_gate(summary))


# ----------------------------------------------------------------------
# 4. Gated dual-stream attention
# ----------------------------------------------------------------------
class GatedDualStreamAttention(nn.Module):
    """Multi-head self-attention where Q and K are element-wise gated by
    relation-derived gates *before* the dot product. V is left ungated.
    An optional additive per-head bias (B, H, N, N) can be added to the logits."""

    def __init__(self, embed_dim: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def _split_heads(self, x):
        B, N, D = x.shape
        return x.view(B, N, self.n_heads, self.head_dim).transpose(1, 2)  # (B,H,N,hd)

    def forward(self, x, gate_q, gate_k, attn_mask=None):
        """
        x: (B, N, embed_dim); gate_q, gate_k: (B, N, embed_dim)
        attn_mask: bool (B,1,1,N) True=keep, or float additive bias (B,H,N,N)
        """
        B, N, D = x.shape
        q, k = self.q_proj(x), self.k_proj(x)
        if gate_q is not None:
            q, k = q * gate_q, k * gate_k
        q, k = self._split_heads(q), self._split_heads(k)
        v = self._split_heads(self.v_proj(x))
        if attn_mask is not None and attn_mask.is_floating_point():
            attn_mask = attn_mask.to(q.dtype)
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


def _ffn(cfg):
    return nn.Sequential(
        nn.Linear(cfg.embed_dim, cfg.ffn_dim),
        nn.GELU(),
        nn.Dropout(cfg.dropout),
        nn.Linear(cfg.ffn_dim, cfg.embed_dim),
        nn.Dropout(cfg.dropout),
    )


# ----------------------------------------------------------------------
# Transformer block: (optionally per-layer) gate + gated attention + FFN, pre-LN
# ----------------------------------------------------------------------
class GDSParTBlock(nn.Module):
    def __init__(self, cfg: GDSParTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = GatedDualStreamAttention(cfg.embed_dim, cfg.n_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.ffn = _ffn(cfg)
        self.gate = (RelationalGate(cfg.pairwise_embed_dim, cfg.embed_dim, cfg.gate_mode,
                                    conditioned=True)
                     if cfg.per_layer_gate and cfg.gate_mode != "none" else None)

    def forward(self, x, rel, gates, attn_mask, key_padding_mask):
        h = self.norm1(x)
        if self.gate is not None:
            gate_q, gate_k = self.gate(rel, key_padding_mask, x=h)
        else:
            gate_q, gate_k = gates if gates is not None else (None, None)
        x = x + self.attn(h, gate_q, gate_k, attn_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ----------------------------------------------------------------------
# Class-attention block (CaiT / ParT style): only the CLS token is updated
# ----------------------------------------------------------------------
class ClassAttentionBlock(nn.Module):
    def __init__(self, cfg: GDSParTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = nn.MultiheadAttention(cfg.embed_dim, cfg.n_heads, dropout=cfg.dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.ffn = _ffn(cfg)

    def forward(self, x, cls, key_padding_mask=None):
        """x: (B, N, D) particle tokens, cls: (B, 1, D). Returns updated cls."""
        kv = self.norm1(torch.cat([cls, x], dim=1))
        pad = None
        if key_padding_mask is not None:
            keep = torch.cat([key_padding_mask.new_ones(x.shape[0], 1), key_padding_mask], dim=1)
            pad = ~keep  # nn.MultiheadAttention: True = ignore
        out, _ = self.attn(kv[:, :1], kv, kv, key_padding_mask=pad, need_weights=False)
        cls = cls + out
        cls = cls + self.ffn(self.norm2(cls))
        return cls


# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class GDSParT(nn.Module):
    def __init__(self, cfg: GDSParTConfig):
        super().__init__()
        self.cfg = cfg
        self.particle_embed = ParticleEmbedding(cfg.n_particle_features, cfg.embed_dim,
                                                cfg.dropout)
        self.pairwise_embed = PairwiseEmbedding(cfg.n_pairwise_features,
                                                cfg.pairwise_embed_dim)
        # shared (non-conditioned) gate, used only when per_layer_gate=False
        self.rel_gate = (None if cfg.per_layer_gate or cfg.gate_mode == "none" else
                         RelationalGate(cfg.pairwise_embed_dim, cfg.embed_dim, cfg.gate_mode))
        # per-head additive pairwise bias, shared by all blocks (as in ParT)
        self.pair_bias = (nn.Linear(cfg.pairwise_embed_dim, cfg.n_heads)
                          if cfg.pair_bias else None)

        self.blocks = nn.ModuleList([GDSParTBlock(cfg) for _ in range(cfg.n_layers)])

        if cfg.n_cls_layers > 0:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, cfg.embed_dim))
            nn.init.trunc_normal_(self.cls_token, std=0.02)
            self.cls_blocks = nn.ModuleList(
                [ClassAttentionBlock(cfg) for _ in range(cfg.n_cls_layers)])
        self.final_norm = nn.LayerNorm(cfg.embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, cfg.n_classes),
        )

    def _pair_inputs(self, pairwise_features, mask):
        """Relational tensor + attention mask (bool, or additive float bias)."""
        rel = self.pairwise_embed(pairwise_features)                  # (B, N, N, Dp)
        attn_mask = None
        if self.pair_bias is not None:
            attn_mask = self.pair_bias(rel).permute(0, 3, 1, 2)       # (B, H, N, N)
            if mask is not None:
                attn_mask = attn_mask.masked_fill(~mask[:, None, None, :], float("-inf"))
        elif mask is not None:
            attn_mask = mask[:, None, None, :]                        # (B, 1, 1, N)
        gates = None if self.rel_gate is None else self.rel_gate(rel, key_padding_mask=mask)
        return rel, attn_mask, gates

    def forward(self, particle_features, pairwise_features, mask=None):
        """
        particle_features: (B, N, 17)
        pairwise_features: (B, N, N, 4)
        mask: (B, N) bool, True = real particle (optional but recommended)
        Returns: logits (B, n_classes)
        """
        x = self.particle_embed(particle_features)
        rel, attn_mask, gates = self._pair_inputs(pairwise_features, mask)

        # the (B, N, N, D) tensors dominate activation memory: during training
        # each block is recomputed in backward instead of storing its internals
        ckpt = self.training and torch.is_grad_enabled()
        for block in self.blocks:
            if ckpt:
                x = torch.utils.checkpoint.checkpoint(
                    block, x, rel, gates, attn_mask, mask, use_reentrant=False)
            else:
                x = block(x, rel, gates, attn_mask, mask)

        if self.cfg.n_cls_layers > 0:
            cls = self.cls_token.expand(x.shape[0], -1, -1)
            for blk in self.cls_blocks:
                cls = blk(x, cls, key_padding_mask=mask)
            pooled = self.final_norm(cls.squeeze(1))
        else:
            x = self.final_norm(x)
            if mask is not None:
                m = mask.unsqueeze(-1).to(x.dtype)
                pooled = (x * m).sum(dim=1) / m.sum(dim=1).clamp(min=1.0)
            else:
                pooled = x.mean(dim=1)

        return self.classifier(pooled)


# ----------------------------------------------------------------------
# Synthetic data for sanity-checking the architecture
# ----------------------------------------------------------------------
def make_synthetic_batch(batch_size=16, max_particles=64, cfg: GDSParTConfig = None,
                          device="cpu"):
    cfg = cfg or GDSParTConfig()
    # random, variable particle multiplicity per jet (padded to max_particles)
    n_real = torch.randint(low=8, high=max_particles + 1, size=(batch_size,))
    mask = torch.zeros(batch_size, max_particles, dtype=torch.bool)
    for i, n in enumerate(n_real):
        mask[i, :n] = True

    particle_features = torch.randn(batch_size, max_particles, cfg.n_particle_features)
    particle_features = particle_features * mask.unsqueeze(-1)

    pairwise_features = torch.randn(
        batch_size, max_particles, max_particles, cfg.n_pairwise_features
    )
    pair_mask = mask.unsqueeze(1) & mask.unsqueeze(2)
    pairwise_features = pairwise_features * pair_mask.unsqueeze(-1)

    labels = torch.randint(low=0, high=cfg.n_classes, size=(batch_size,))

    return (
        particle_features.to(device),
        pairwise_features.to(device),
        mask.to(device),
        labels.to(device),
    )


def sanity_check(device="cpu"):
    """One forward pass, one loss computation, one backward pass, one
    optimizer step — confirms the architecture is differentiable and free
    of shape mismatches."""
    torch.manual_seed(0)
    cfg = GDSParTConfig(embed_dim=64, n_heads=4, n_layers=3, ffn_dim=128,
                         pairwise_embed_dim=32, max_particles=64)
    model = GDSParT(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    particle_features, pairwise_features, mask, labels = make_synthetic_batch(
        batch_size=8, max_particles=cfg.max_particles, cfg=cfg, device=device
    )

    print(f"particle_features: {tuple(particle_features.shape)}")
    print(f"pairwise_features: {tuple(pairwise_features.shape)}")
    print(f"mask:               {tuple(mask.shape)}")
    print(f"labels:             {tuple(labels.shape)}")

    logits = model(particle_features, pairwise_features, mask=mask)
    print(f"logits:             {tuple(logits.shape)}  (expect (8, {cfg.n_classes}))")

    loss = F.cross_entropy(logits, labels)
    print(f"initial loss:       {loss.item():.4f}")

    optimizer.zero_grad()
    loss.backward()

    total_grad_norm = 0.0
    n_params_with_grad = 0
    for p in model.parameters():
        if p.grad is not None:
            total_grad_norm += p.grad.norm().item() ** 2
            n_params_with_grad += 1
    total_grad_norm = total_grad_norm ** 0.5
    print(f"grad norm:          {total_grad_norm:.4f} over {n_params_with_grad} tensors")

    optimizer.step()
    print("optimizer step:     OK")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"total parameters:   {n_params:,}")

    print("\nSANITY CHECK PASSED: forward -> loss -> backward -> step, no shape errors.")
    return model


if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}\n")
    sanity_check(device=device)
