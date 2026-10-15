"""
gds_part_colab.py — Gated Dual-Stream Particle Transformer (GDS-ParT)

A single-file, Colab/Jupyter-friendly PyTorch implementation.

Core idea (contrast with ParT):
    ParT:     Attention = Softmax( QK^T/sqrt(d) + psi(U) )        [additive logit bias]
    GDS-ParT: Q' = Q * gate_Q(U),  K' = K * gate_K(U)
              Attention = Softmax( Q'K'^T / sqrt(d) )              [multiplicative feature gating]

where U is the pairwise interaction tensor (B, N, N, 4) and gate_Q/gate_K are
learned, relation-aggregated, per-particle gates in (0, 1).

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
    def __init__(
        self,
        n_particle_features: int = 17,
        n_pairwise_features: int = 4,
        n_classes: int = 10,
        embed_dim: int = 128,
        n_heads: int = 8,
        n_layers: int = 6,
        ffn_dim: int = 512,
        pairwise_embed_dim: int = 64,
        dropout: float = 0.1,
        max_particles: int = 128,
    ):
        assert embed_dim % n_heads == 0, "embed_dim must be divisible by n_heads"
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
        # x: (B, N, in_features)
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
        # u: (B, N, N, in_features)
        return self.net(u)


# ----------------------------------------------------------------------
# 3. Relational gate
# ----------------------------------------------------------------------
class RelationalGate(nn.Module):
    """Aggregates the (B, N, N, D) relational tensor across the neighbor
    axis (dim=2) into a per-particle summary, then maps that summary to
    two independent element-wise gates in (0, 1): one for queries, one
    for keys. Padded neighbors are excluded from the aggregation via mask.
    """

    def __init__(self, pairwise_dim: int, embed_dim: int):
        super().__init__()
        self.query_gate = nn.Sequential(
            nn.Linear(pairwise_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )
        self.key_gate = nn.Sequential(
            nn.Linear(pairwise_dim, embed_dim),
            nn.GELU(),
            nn.Linear(embed_dim, embed_dim),
            nn.Sigmoid(),
        )

    def forward(self, rel, key_padding_mask=None):
        """
        rel: (B, N, N, D) relational tensor, rel[b, i, j] describes the
             relation between particle i and particle j.
        key_padding_mask: (B, N) bool, True = real particle.
        Returns: gate_q, gate_k each of shape (B, N, embed_dim)
        """
        if key_padding_mask is not None:
            # zero out contributions from padded neighbors (dim=2 is neighbor j)
            m = key_padding_mask[:, None, :, None].to(rel.dtype)  # (B,1,N,1)
            rel = rel * m
            denom = m.sum(dim=2).clamp(min=1.0)  # (B,1,1), same count for every i
            summary = rel.sum(dim=2) / denom  # (B, N, D)
        else:
            summary = rel.mean(dim=2)  # (B, N, D)

        gate_q = self.query_gate(summary)  # (B, N, embed_dim)
        gate_k = self.key_gate(summary)    # (B, N, embed_dim)
        return gate_q, gate_k


# ----------------------------------------------------------------------
# 4. Gated dual-stream attention
# ----------------------------------------------------------------------
class GatedDualStreamAttention(nn.Module):
    """Multi-head self-attention where Q and K are element-wise gated by
    relation-derived gates *before* the dot product is computed. V is left
    ungated so the value stream still carries the raw learned content.
    """

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

    def forward(self, x, gate_q, gate_k, key_padding_mask=None):
        """
        x: (B, N, embed_dim)
        gate_q, gate_k: (B, N, embed_dim) in (0, 1)
        key_padding_mask: (B, N) bool, True = real particle
        """
        B, N, D = x.shape

        q = self.q_proj(x) * gate_q     # multiplicative gating pre-attention
        k = self.k_proj(x) * gate_k
        v = self.v_proj(x)

        q = self._split_heads(q)  # (B,H,N,hd)
        k = self._split_heads(k)
        v = self._split_heads(v)

        attn_mask = None
        if key_padding_mask is not None:
            attn_mask = key_padding_mask[:, None, None, :]  # (B,1,1,N) True=keep

        # fused memory-efficient kernel: never materializes the (B,H,N,N) scores
        out = F.scaled_dot_product_attention(
            q, k, v, attn_mask=attn_mask,
            dropout_p=self.dropout.p if self.training else 0.0)  # (B,H,N,hd)
        out = out.transpose(1, 2).contiguous().view(B, N, D)
        return self.out_proj(out)


# ----------------------------------------------------------------------
# Transformer block: gated attention + FFN, pre-LN residual
# ----------------------------------------------------------------------
class GDSParTBlock(nn.Module):
    def __init__(self, cfg: GDSParTConfig):
        super().__init__()
        self.norm1 = nn.LayerNorm(cfg.embed_dim)
        self.attn = GatedDualStreamAttention(cfg.embed_dim, cfg.n_heads, cfg.dropout)
        self.norm2 = nn.LayerNorm(cfg.embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.ffn_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.ffn_dim, cfg.embed_dim),
            nn.Dropout(cfg.dropout),
        )

    def forward(self, x, gate_q, gate_k, key_padding_mask=None):
        x = x + self.attn(self.norm1(x), gate_q, gate_k, key_padding_mask)
        x = x + self.ffn(self.norm2(x))
        return x


# ----------------------------------------------------------------------
# Full model
# ----------------------------------------------------------------------
class GDSParT(nn.Module):
    def __init__(self, cfg: GDSParTConfig):
        super().__init__()
        self.cfg = cfg
        self.particle_embed = ParticleEmbedding(
            cfg.n_particle_features, cfg.embed_dim, cfg.dropout
        )
        self.pairwise_embed = PairwiseEmbedding(
            cfg.n_pairwise_features, cfg.pairwise_embed_dim
        )
        self.rel_gate = RelationalGate(cfg.pairwise_embed_dim, cfg.embed_dim)

        self.blocks = nn.ModuleList(
            [GDSParTBlock(cfg) for _ in range(cfg.n_layers)]
        )
        self.final_norm = nn.LayerNorm(cfg.embed_dim)

        self.classifier = nn.Sequential(
            nn.Linear(cfg.embed_dim, cfg.embed_dim),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.embed_dim, cfg.n_classes),
        )

        # gates are recomputed once per forward pass from the pairwise
        # stream and reused across all layers (cheap, and keeps the
        # relational signal consistent throughout depth). Set
        # `regate_every_layer=True` below to recompute per-block instead.
        self.regate_every_layer = False
        if self.regate_every_layer:
            self.rel_gates = nn.ModuleList(
                [RelationalGate(cfg.pairwise_embed_dim, cfg.embed_dim)
                 for _ in range(cfg.n_layers)]
            )

    def _compute_gates(self, pairwise_features, mask):
        rel = self.pairwise_embed(pairwise_features)        # (B, N, N, pairwise_embed_dim)
        return self.rel_gate(rel, key_padding_mask=mask)

    def forward(self, particle_features, pairwise_features, mask=None):
        """
        particle_features: (B, N, 17)
        pairwise_features: (B, N, N, 4)
        mask: (B, N) bool, True = real particle (optional but recommended)
        Returns: logits (B, n_classes)
        """
        x = self.particle_embed(particle_features)         # (B, N, embed_dim)
        rel = None
        if self.regate_every_layer:
            rel = self.pairwise_embed(pairwise_features)    # (B, N, N, pairwise_embed_dim)
            gate_q, gate_k = self.rel_gate(rel, key_padding_mask=mask)
        elif self.training and torch.is_grad_enabled():
            # the (B, N, N, D) relational tensor dominates activation memory;
            # recompute it in backward instead of storing it
            gate_q, gate_k = torch.utils.checkpoint.checkpoint(
                self._compute_gates, pairwise_features, mask, use_reentrant=False)
        else:
            gate_q, gate_k = self._compute_gates(pairwise_features, mask)

        for i, block in enumerate(self.blocks):
            if self.regate_every_layer:
                gate_q, gate_k = self.rel_gates[i](rel, key_padding_mask=mask)
            x = block(x, gate_q, gate_k, key_padding_mask=mask)

        x = self.final_norm(x)

        # masked mean pooling over particle tokens
        if mask is not None:
            m = mask.unsqueeze(-1).to(x.dtype)              # (B, N, 1)
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
