# GDS-ParT ablations

Copied from the git-ignored `ablations_noema/ABLATIONS.md`. Settings: 30,000 training jets per class (300k total), 2 epochs, 50,000 validation jets per class (500k), lr 1e-3, batch 512, seed 0, **EMA off** (`--ema_decay 0`).

| Variant | Description | Params | Val acc | Val AUC | Val acc per epoch |
|---|---|---|---|---|---|
| shared_gate | v2 with one uniform-mean gate shared by all layers | 2,074,420 | 0.6966 | 0.9509 | 0.6238 / 0.6966 |
| no_gate | v2 without Q/K gates (= ParT-style bias-only attention) | 2,024,756 | 0.6942 | 0.9508 | 0.6176 / 0.6942 |
| v2_full | v2: per-layer identity gates + pair bias + 2 CLS blocks, 8 layers | 2,488,116 | 0.6933 | 0.9507 | 0.6176 / 0.6933 |
| no_pair_bias | v2 without additive pair bias (gates only) | 2,487,596 | 0.6223 | 0.9315 | 0.5324 / 0.6223 |
