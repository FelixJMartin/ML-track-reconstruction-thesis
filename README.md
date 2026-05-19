# Spiking Neural Networks for Particle Track Reconstruction

Bachelor thesis project exploring the use of **Spiking Neural Networks (SNNs)** for charged particle track reconstruction in high-energy physics detectors, using the [TrackML dataset](https://www.kaggle.com/c/trackml-particle-identification).

---

## Overview

Particle detectors at experiments like CERN produce millions of hits per collision event. Identifying which hits belong to the same charged particle track — *track reconstruction* — is a core challenge in high-energy physics, and one that becomes increasingly difficult as luminosity grows.

This project implements a two-stage SNN pipeline that narrows down candidate tracks in both geometric space and angular space, using biologically-inspired leaky integrate-and-fire (LIF) neurons.

---

## Pipeline

### Stage 1 — Cone Filter
The detector volume is split into 8 octant cones. For each cone, hits are projected onto a 2D **r-z grid** and fed into an SNN that classifies whether the cone contains a signal track.

### Stage 2 — Phi Localiser
Cones that pass Stage 1 are projected onto an **r-φ grid**. A second SNN identifies which φ bins contain track hits, narrowing the search window for downstream reconstruction.

<!-- pipeline figure here -->

---

## Model Architecture

Both stages use a two-layer radial LIF network (`RadLIFLayer`) with batch normalisation and dropout, implemented in PyTorch via [snntorch](https://github.com/jeshraghian/snntorch).

```
Input grid  →  LIF layer 1  →  LIF layer 2  →  Linear  →  Output
```

---

## Repository Structure

```
Felix/
├── Code/          # Training scripts, data builders, evaluation
├── SNN/           # RadLIFLayer model definition (snns.py)
└── OlderPlots/    # Early exploratory scripts

figures/           # Output plots and PDFs
```

Large files (raw TrackML events, `.npy` matrices, `.pt` model weights) are excluded via `.gitignore`.

---

## Requirements

```
torch
snntorch
numpy
pandas
scikit-learn
trackml          # pip install git+https://github.com/LAL/trackml-library
matplotlib
```

---

## Usage

**1. Build training matrices**
```bash
python Felix/Code/build_data100.py
```

**2. Train Stage 1 (cone filter)**
```bash
python Felix/Code/Stage1_ConeFilter100.py
```

**3. Train Stage 2 (phi localiser)**
```bash
python Felix/Code/Stage2_PhiFilter100.py
```

For cluster (UPPMAX) runs, see the SLURM submission scripts in `Felix/Code/submit_*.sh`.

---

## Results

<!-- Add figures and result tables here -->

---

## Author

Felix Martin — Bachelor thesis, 2026
