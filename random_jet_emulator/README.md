# Random Jet Emulator

Toy collider event display copied from the hero canvas of https://avinashistoocool.github.io (`eventDisplay()` in `app.js`). Open `index.html` in a browser; click the detector or **New collision** for a new event.

## How the simulation works
- **Detector (transverse view):** beam pipe, 5 tracker layers, ECAL, HCAL (72 φ-towers), and a 12-segment muon system, as fractions of radius `R`.
- **Event generation (`newEvent`):** picks one of 10 classes (H→bb̄, H→cc̄, H→gg, H→4q, H→ℓνqq', t→bqq', t→bℓν, W→qq', Z→qq̄, QCD). 2–3 jets; the lead jet gets the class's prong count, sub-axes smeared by a Gaussian width of `0.08 + 0.05·prongs`.
- **Constituents:** 62% charged (helical tracks, radius ∝ pT, random charge), 23% photons (ECAL), 15% neutral hadrons (HCAL); pT ∝ `ptJet·exp(-U(0,2.6))`. Plus 18 soft pile-up tracks.
- **Muons:** added for leptonic classes or with 35% probability.
- **Calorimeter:** energy deposited in φ-towers with neighbour sharing (`deposit`).
- **Tagger read-out:** softmax over fake logits where the true class gets U(2.4, 3.6); top-3 shown in the HUD.
- **Animation:** vertex flash (0.35 s) → tracks grow (1.8 s) → calo towers + labels → hold until 9 s → fade by 10 s → new event.
