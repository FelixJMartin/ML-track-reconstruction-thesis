# threshold_sweep.py
# Sweeps S1 and S2 thresholds over 500 held-out test events.
# Plots precision-recall curves and finds the F1-optimal threshold for each stage.

import sys, os
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\SNN")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\build_datasets")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from trackml.dataset import load_event
from snns import RadLIFLayer
from Functions import preprocess_particles
from build_data8500 import (
    make_rz_grid, make_rphi_grid,
    SNN_R_BINS, SNN_PHI_BINS, CONE_PHI_RANGES, SAMPLE_FRACTION,
)

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"text.usetex": False, "figure.dpi": 150,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_5"
S1_PATH  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\s1_8500_ep115.pt"
S2_PATH  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\s2_h64_32_ep78.pt"
TEST_IDS = [f"event{i:09d}" for i in range(9500, 10000)]

SIGNS = [
    ( 1,  1,  1), ( 1,  1, -1), ( 1, -1,  1), ( 1, -1, -1),
    (-1,  1,  1), (-1,  1, -1), (-1, -1,  1), (-1, -1, -1),
]
SNN_INPUT_SIZE = SNN_R_BINS + 1

# ── MODELS ────────────────────────────────────────────────────────────────────
class ConeFilterSNN(nn.Module):
    def __init__(self, input_size=100, h1=64, h2=32, bs=1):
        super().__init__()
        self.lif1 = RadLIFLayer(input_size, h1, bs, normalization="batchnorm", dropout=0.2)
        self.lif2 = RadLIFLayer(h1, h2,     bs, normalization="batchnorm", dropout=0.2)
        self.fc   = nn.Linear(h2, 1)
    def forward(self, x):
        return self.fc(self.lif2(self.lif1(x)).mean(dim=1)).squeeze(-1)

class PhiFinderSNN(nn.Module):
    def __init__(self, h1=64, h2=32, bs=1):
        super().__init__()
        self.lif1 = RadLIFLayer(SNN_INPUT_SIZE, h1, bs, normalization="batchnorm", dropout=0.2)
        self.lif2 = RadLIFLayer(h1, h2,         bs, normalization="batchnorm", dropout=0.2)
        self.fc   = nn.Linear(h2, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.fc(self.lif2(self.lif1(x))).squeeze(-1)

def load_model(cls, path, **kwargs):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = cls(**kwargs)
    m.load_state_dict(ckpt["model_state"])
    return m.eval()

model_s1 = load_model(ConeFilterSNN, S1_PATH)
model_s2 = load_model(PhiFinderSNN,  S2_PATH)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def octant_mask(df, sx, sy, sz):
    return (
        ((df["x"] > 0) if sx > 0 else (df["x"] < 0)) &
        ((df["y"] > 0) if sy > 0 else (df["y"] < 0)) &
        ((df["z"] > 0) if sz > 0 else (df["z"] < 0))
    )

def pr_sweep(scores, labels, n_thr=400):
    thresholds = np.linspace(0, 1, n_thr)
    precs, recs, f1s = [], [], []
    for thr in thresholds:
        pred = scores > thr
        tp = int((pred &  labels).sum())
        fp = int((pred & ~labels).sum())
        fn = int((~pred & labels).sum())
        p  = tp / max(tp + fp, 1)
        r  = tp / max(tp + fn, 1)
        f1 = 2 * p * r / max(p + r, 1e-9)
        precs.append(p); recs.append(r); f1s.append(f1)
    return np.array(thresholds), np.array(precs), np.array(recs), np.array(f1s)

# ── COLLECT SCORES OVER ALL TEST EVENTS ──────────────────────────────────────
s1_scores, s1_labels = [], []   # one entry per octant per event
s2_scores, s2_labels = [], []   # one entry per phi bin (signal octants only)

print(f"Sweeping {len(TEST_IDS)} test events...\n")

for k, event_id in enumerate(TEST_IDS):
    try:
        h, _, p, t = load_event(os.path.join(DATA_DIR, event_id))
        h["r"]   = np.hypot(h["x"], h["y"])
        h["phi"] = np.arctan2(h["y"], h["x"])
        p_cond, _, _ = preprocess_particles(10, 10, p, h)
        true_sig_ids  = set(t[t["particle_id"].isin(p_cond["particle_id"])]["hit_id"])
        noise_sample  = h[np.random.rand(len(h)) < SAMPLE_FRACTION].copy()

        for sx, sy, sz in SIGNS:
            cone     = h[octant_mask(h, sx, sy, sz)]
            noise    = noise_sample[octant_mask(noise_sample, sx, sy, sz)]
            sig      = cone[cone["hit_id"].isin(true_sig_ids)]
            has_sig  = len(sig) > 0

            # model input: 1% noise + all signal hits
            if has_sig:
                inp = pd.concat([noise[~noise["hit_id"].isin(true_sig_ids)], sig])
            else:
                inp = noise

            if len(inp) == 0:
                continue

            # ── Stage 1 ──────────────────────────────────────────────────────
            grid = torch.tensor(
                make_rz_grid(inp["r"].values, inp["z"].values),
                dtype=torch.float32,
            ).unsqueeze(0)
            with torch.no_grad():
                s1_conf = torch.sigmoid(model_s1(grid)).item()
            s1_scores.append(s1_conf)
            s1_labels.append(has_sig)

            # ── Stage 2 (signal octants only — independent of S1) ─────────
            if not has_sig:
                continue

            phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
            rphi_grid = torch.tensor(
                make_rphi_grid(inp["r"].values, inp["phi"].values, sx, sy),
                dtype=torch.float32,
            ).unsqueeze(0)
            with torch.no_grad():
                phi_probs = torch.sigmoid(model_s2(rphi_grid)).squeeze(0).numpy()

            # true label per phi bin: does any signal hit fall there?
            sig_bins = np.clip(
                ((sig["phi"].values - phi_min) / (phi_max - phi_min) * SNN_PHI_BINS
                 ).astype(int),
                0, SNN_PHI_BINS - 1,
            )
            true_bins = np.zeros(SNN_PHI_BINS, dtype=bool)
            true_bins[sig_bins] = True

            s2_scores.extend(phi_probs.tolist())
            s2_labels.extend(true_bins.tolist())

        if (k + 1) % 50 == 0 or k == 0:
            print(f"  {k+1:4d}/{len(TEST_IDS)}  "
                  f"s1={len(s1_scores)} octants ({sum(s1_labels)} sig)  "
                  f"s2={len(s2_scores)} phi bins ({sum(s2_labels)} sig)")

    except Exception as e:
        print(f"  {event_id}: skipped — {e}")

s1_scores = np.array(s1_scores, dtype=np.float32)
s1_labels = np.array(s1_labels, dtype=bool)
s2_scores = np.array(s2_scores, dtype=np.float32)
s2_labels = np.array(s2_labels, dtype=bool)

print(f"\nFinal totals:")
print(f"  S1: {len(s1_scores)} octants  —  {s1_labels.sum()} signal  {(~s1_labels).sum()} noise")
print(f"  S2: {len(s2_scores)} phi bins —  {s2_labels.sum()} signal  {(~s2_labels).sum()} noise")

# ── SWEEP ─────────────────────────────────────────────────────────────────────
thr_s1, prec_s1, rec_s1, f1_s1 = pr_sweep(s1_scores, s1_labels)
thr_s2, prec_s2, rec_s2, f1_s2 = pr_sweep(s2_scores, s2_labels)

best1 = np.argmax(f1_s1)
best2 = np.argmax(f1_s2)

print(f"\nOptimal thresholds (max F1):")
print(f"  S1  thr={thr_s1[best1]:.3f}  prec={prec_s1[best1]*100:.1f}%  "
      f"recall={rec_s1[best1]*100:.1f}%  F1={f1_s1[best1]:.3f}")
print(f"  S2  thr={thr_s2[best2]:.3f}  prec={prec_s2[best2]*100:.1f}%  "
      f"recall={rec_s2[best2]*100:.1f}%  F1={f1_s2[best2]:.3f}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(13, 5))

for ax, (precs, recs, f1s, thrs, bi, title) in zip(axes, [
    (prec_s1, rec_s1, f1_s1, thr_s1, best1, "Stage 1 — octant classifier"),
    (prec_s2, rec_s2, f1_s2, thr_s2, best2, "Stage 2 — phi bin classifier"),
]):
    # PR curve
    ax.plot(recs, precs, color="#0077BB", lw=1.5)
    ax.scatter([recs[bi]], [precs[bi]], color="tomato", s=60, zorder=5,
               label=f"thr={thrs[bi]:.2f}  F1={f1s[bi]:.3f}")
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title(title, fontsize=10)
    ax.grid(True, color="#cccccc", linewidth=0.6, linestyle="--")
    ax.legend(fontsize=8, frameon=False)

    # inset: F1 vs threshold
    ax2 = ax.inset_axes([0.55, 0.55, 0.42, 0.38])
    ax2.plot(thrs, f1s, color="#009988", lw=1.2)
    ax2.axvline(thrs[bi], color="tomato", lw=0.9, linestyle="--")
    ax2.set_xlabel("threshold", fontsize=6)
    ax2.set_ylabel("F1", fontsize=6)
    ax2.tick_params(labelsize=5)
    ax2.grid(True, color="#cccccc", linewidth=0.4, linestyle="--")

plt.tight_layout()
plt.show()
