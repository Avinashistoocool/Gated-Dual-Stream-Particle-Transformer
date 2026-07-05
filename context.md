project context for GDS-ParT

## What this is
A from-scratch PyTorch implementation of GDS-ParT (Gated Dual-Stream
Particle Transformer), an original alternative to ParT's additive
attention-bias fusion of pairwise physics. Here pairwise features become
multiplicative gates applied to Q/K *before* the dot product, rather than
an additive term added to attention logits.

## Files
- `gds_part_colab.py` — single-file model definition (ParticleEmbedding,
  PairwiseEmbedding, RelationalGate, GatedDualStreamAttention, GDSParTBlock,
  GDSParT). Also contains `make_synthetic_batch()` and `sanity_check()`.
  Running this file directly (`python gds_part_colab.py`) does one
  forward/backward/optimizer-step sanity check — this has been verified to
  pass (124,460 params in the default sanity-check config, loss ~2.09 on
  a random 10-class synthetic batch).
- `data/jetclass_dataset.py` — real JetClass ROOT loader. Computes the
  standard 17 ParT particle features and 4 pairwise interaction features
  (ln delta, ln kt, ln z, ln m2) from raw `part_px/py/pz/energy/deta/dphi/
  charge/pid/d0val/d0err/dzval/dzerr` branches. In-memory dataset, fine for
  val/test scale (5M/20M jets subsampled); for full 100M-jet training,
  swap in an IterableDataset using `uproot.iterate()`.
- `train_gds_part.py` — trainer: AdamW, linear warmup + cosine decay,
  grad clipping, best/last checkpointing, per-class accuracy + confusion
  matrix at the end. Supports `--synthetic` for a no-data dry run (verified
  working end to end: 2 epochs, checkpoints saved and reloadable).
- `checkpoints/` — `best.pt` / `last.pt`, each a dict with
  `epoch, model_state, optimizer_state, config, val_acc`.

## Known limitations / TODO (see README "Suggested next steps")
- No query/key gate asymmetry study yet (both gates exist, untested for
  which helps more).
- No gate visualization/interpretability tooling yet.
- No baseline ParT comparison harness yet.
- `JetClassDataset` loads everything into memory — fine for a few thousand
  jets per class, will not scale to 100M jets without rewriting as a
  streaming/iterable dataset.
- Pairwise feature computation in `jetclass_dataset.py` loops jet-by-jet in
  Python; fine for building a val-scale dataset once, too slow to call
  per-batch during training — precompute and cache, don't call live.

## Verified test commands
```bash
# model-only sanity check (forward/backward/step)
python gds_part_colab.py

# full training loop wiring, no data required
python train_gds_part.py --synthetic --epochs 2 --synthetic_batches 6 \
  --batch_size 8 --embed_dim 32 --n_heads 4 --n_layers 2 --ffn_dim 64 \
  --pairwise_embed_dim 16 --max_particles 32 --log_every 2
```

## Real-data run (requires JetClass download, see main README)
```bash
git clone https://github.com/jet-universe/particle_transformer
cd particle_transformer && python get_datasets.py JetClass -d ./data
cd ..
python train_gds_part.py \
  --train ./particle_transformer/data/JetClass/train_100M \
  --val   ./particle_transformer/data/JetClass/val_5M \
  --test  ./particle_transformer/data/JetClass/test_20M \
  --epochs 10 --batch_size 512
```
