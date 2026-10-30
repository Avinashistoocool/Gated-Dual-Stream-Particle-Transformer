# GDS-ParT ablations

Copied from the git-ignored `ablations/ABLATIONS.md`. Same budget as `ABLATIONS_noema.md` (300k training jets, 2 epochs, 500k validation jets, seed 0), but v2 variants use the default **EMA 0.999**, which lags badly over only 1,172 steps; v1 variants use no EMA and no label smoothing.

| Variant | Description | Params | Val acc | Val AUC | Val acc per epoch |
|---|---|---|---|---|---|
| v1 | v1: shared sigmoid gate, mean pooling, 6 layers | 1,280,684 | 0.6557 | 0.9412 | 0.5927 / 0.6557 |
| v1_plus_cls | v1 + CLS pooling | 1,677,356 | 0.6114 | 0.9292 | 0.5274 / 0.6114 |
| v2_full | v2: per-layer identity gates + pair bias + 2 CLS blocks, 8 layers | 2,488,116 | 0.5407 | 0.9113 | 0.3103 / 0.5407 |
| no_gate | v2 without Q/K gates (= ParT-style bias-only attention) | 2,024,756 | 0.5085 | 0.9028 | 0.2853 / 0.5085 |
| shared_gate | v2 with one uniform-mean gate shared by all layers | 2,074,420 | 0.4979 | 0.8962 | 0.2857 / 0.4979 |
| no_pair_bias | v2 without additive pair bias (gates only) | 2,487,596 | 0.3741 | 0.8430 | 0.1827 / 0.3741 |
