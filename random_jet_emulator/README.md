# Random Jet Emulator

This is a toy event display for a collider. It is a copy of the hero canvas of https://avinashistoocool.github.io (`eventDisplay()` in `app.js`).

1. Open `index.html` in a browser.
2. To show a new event, click the detector or click **New collision**.

## How the simulation works
- **Detector (transverse view):** the beam pipe, 5 tracker layers, ECAL, HCAL (72 φ-towers) and a muon system with 12 segments. Each part has a radius that is a fraction of `R`.
- **Event generation (`newEvent`):** selects one of 10 classes (H→bb̄, H→cc̄, H→gg, H→4q, H→ℓνqq', t→bqq', t→bℓν, W→qq', Z→qq̄, QCD). Each event has 2–3 jets. The lead jet gets the prong count of the class. A Gaussian with a width of `0.08 + 0.05·prongs` moves the sub-axes by random amounts.
- **Constituents:**
  - 62% charged particles (helical tracks, radius ∝ pT, random charge)
  - 23% photons (ECAL)
  - 15% neutral hadrons (HCAL).

  The pT of each constituent is ∝ `ptJet·exp(-U(0,2.6))`. The event also has 18 soft pile-up tracks.
- **Muons:** the emulator adds muons for leptonic classes, or with a probability of 35%.
- **Calorimeter:** the emulator puts energy in the φ-towers and shares some energy with the adjacent towers (`deposit`).
- **Tagger read-out:** a softmax over fake logits, where the true class gets U(2.4, 3.6). The HUD shows the top 3 classes.
- **Animation:**
  1. The vertex flashes (0.35 s).
  2. The tracks grow (1.8 s).
  3. The calorimeter towers and the labels show.
  4. The display holds until 9 s.
  5. The display fades until 10 s.
  6. A new event starts.
