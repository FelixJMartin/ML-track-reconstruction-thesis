# Spiking Neural Networks as Track Filters in High-Energy Physics

Bachelor thesis project exploring the use of **Spiking Neural Networks (SNNs)** for charged particle track filtering in high-energy physics detectors, using the [TrackML dataset](https://www.kaggle.com/c/trackml-particle-identification).

![Focus](https://img.shields.io/badge/Focus-Neuromorphic%20Computing%20%26%20Particle%20Physics-4A90D9?style=flat-square)
![Tools](https://img.shields.io/badge/Tools-Python%20|%20PyTorch%20|%20SPArch-555555?style=flat-square)
![Dataset](https://img.shields.io/badge/Dataset-TrackML%20(CERN)-6f42c1?style=flat-square)

---

## Overview

Particle detectors at CERN produce on the order of 10⁵ hits per collision event. Identifying which hits belong to the same charged particle track — *track reconstruction* — is one of the most computationally demanding steps in the analysis pipeline. At the upcoming High-Luminosity LHC (HL-LHC), the number of simultaneous proton–proton collisions per event will rise to up to 200, placing combinatorial demands that are expected to exceed the budget available with conventional CPUs.

This project implements a two-stage SNN pipeline that reduces the number of candidate hits before full reconstruction begins — filtering in both geometric space (cone-level) and angular space (φ-bin level). The networks are built on **Recurrent Adaptive LIF (RadLIF)** neurons, a biologically-inspired architecture that communicates via discrete binary spikes rather than continuous signals, offering potential energy efficiency advantages on neuromorphic hardware.

<p align="center">
  <img src="img/MLtrack.png" width="700">
  <br>
  <img src="img/Filtering.png" width="700">
  <br>
  <em>Particle track reconstruction in LHC detector data — raw hits (left), after Stage 1 cone filter (centre), after Stage 2 φ-bin localiser (right)</em>
</p>

---

## Pipeline

### Stage 1 — Cone Filter

The detector volume is split into 8 octant cones. For each cone, hits are projected onto a **100×100 r-z grid** and fed into a RadLIF SNN that classifies whether the cone contains a signal track. Cones classified as background are discarded entirely.

### Stage 2 — φ-Bin Localiser

Cones that pass Stage 1 are projected onto a **100×100 r-φ grid**, augmented with a column occupancy feature that encodes the coherent columnar structure of genuine tracks. A second RadLIF SNN identifies which φ-bins contain signal hits, further narrowing the candidate set for downstream reconstruction.

---

## Model Architecture

Both stages use a fully connected two-layer **RadLIF** network from the [SPArch framework](https://github.com/idiap/sparch), trained with backpropagation through time (BPTT) and a boxcar surrogate gradient. The RadLIF neuron extends the standard leaky integrate-and-fire model with recurrent connections and spike-frequency adaptation.

```
Input grid (100×100)  →  RadLIF layer 1 (512)  →  RadLIF layer 2 (256)  →  Linear  →  Output
```

Six parameter groups are learned jointly: feedforward weights **W**, recurrent weights **V**, membrane decay rate α, adaptation decay rate β, subthreshold coupling *a*, and spike-triggered adaptation *b*.

---

## Dataset

The [TrackML Particle Tracking Challenge](https://www.kaggle.com/c/trackml-particle-identification) dataset simulates HL-LHC detector conditions, including high-pileup environments with up to 200 simultaneous proton–proton collisions per event.

- ~10⁵ hits per event from ~10⁴ charged particles
- Signal selection: pT ≥ 10 GeV/c and nhits > 10
- Background subsampled to 1% per event during training
- Training scales evaluated: 100, 1000, and 1800 events
- Strict event-level train/test split (no shared particles between sets)

---

## Results

Evaluated on 20 held-out test events with classification thresholds p_S1 = 0.5, p_S2 = 0.6.

| Metric | 100 events | 1000 events |
|---|---|---|
| S1 signal efficiency | 80.8% | 80.8% |
| S1 noise throwrate | 46.0% | 55.6% |
| S2 detection rate | 92.9% | 97.6% |
| S2 precision | 71.4% | 85.3% |
| Track survival | 80.8% | 84.5% |

**Optimised configuration** (512×256, lr=1e-3, dropout=0.1) via grid search over 12 Stage 1 configurations:
- Signal efficiency: **92.8%**
- Noise throwrate: **96.9%**
- Stage 2 IoU: **81.7%** (up from 70.0% baseline)

---

## Repository Structure

```
build_datasets/    # Scripts that build training matrices from raw TrackML events
training/          # Stage 1 / Stage 2 model training scripts
grid_search/       # Hyperparameter grid search + optimisation summaries
evaluation/        # ROC curves, threshold sweeps, full-pipeline tests
plotting/          # Visualisation scripts
cluster_scripts/   # SLURM/UPPMAX submission scripts
exploratory/       # Early exploratory scripts (pre-pipeline EDA, baseline, param inspection)
Functions.py       # Shared preprocessing utilities
SNN/               # RadLIFLayer model definition (snns.py)
figures/           # Output plots and PDFs
img/               # README figures
```

Large files (raw TrackML events, `.npy` matrices, `.pt` model weights) are excluded via `.gitignore`.

---

## Requirements

```
torch
numpy
pandas
scikit-learn
matplotlib
trackml          # pip install git+https://github.com/LAL/trackml-library
```

The SPArch RadLIF implementation is included directly in `SNN/snns.py`, adapted from [Bittar & Garner (2022)](https://doi.org/10.3389/fnins.2022.865897).

---

## Usage

**1. Build training matrices**
```bash
python build_datasets/build_data100.py
```

**2. Train Stage 1 (cone filter)**
```bash
python training/train_s1_100.py
```

**3. Train Stage 2 (φ-bin localiser)**
```bash
python training/train_s2_100.py
```

For cluster (UPPMAX/SLURM) runs, see the submission scripts in `cluster_scripts/`.

---


## Author

Felix Martin · Engineering Physics, Uppsala University · [GitHub](https://github.com/FelixJMartin)
