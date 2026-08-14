# ================================================================================
# ROC_stage2.py
# ================================================================================

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\SNN")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\training")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\build_datasets")

import torch
import os
import numpy as np
import matplotlib.pyplot as plt

from trackml.dataset import load_event
from Functions import preprocess_particles
from build_data import (make_rz_grid, make_rphi_grid, get_cone, CONE_SIGNS,
                        SNN_R_BINS, SNN_PHI_BINS, SNN_R_MAX,
                        SAMPLE_FRACTION, CONE_PHI_RANGES)
from train_s1_1000 import ConeFilterSNN
from train_s2_8500 import PhiFinderSNN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.rcParams.update({
    "text.usetex":       False,
    "figure.dpi":        150,
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

C = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#33BBEE", "#EE3377"]

# ── CONFIG ────────────────────────────────────────────────────────────────────
THRESH_S1 = 0.5
data_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_1"
Save_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices100"
test_ids  = [f"event{i:09d}" for i in range(2800, 2820)]   # 20 unseen events

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
print("Loading models...")
model_s1 = ConeFilterSNN(SNN_R_BINS, bs=1)
model_s1.load_state_dict(
    torch.load(os.path.join(Save_dir, "model_stage1.pt"))["model_state"]
)
model_s1.eval()
model_s1.set_bs(1)

model_s2 = PhiFinderSNN(bs=1)   # PhiFinderSNN uses SNN_INPUT_SIZE internally
model_s2.load_state_dict(
    torch.load(os.path.join(Save_dir, "model_stage2.pt"))["model_state"]
)
model_s2.eval()
model_s2.set_bs(1)

# ── EVALUATE ──────────────────────────────────────────────────────────────────
all_phi_probs   = []
all_true_active = []

print(f"\nEvaluating {len(test_ids)} test events...")

for event_id in test_ids:
    try:
        h, _, p, t = load_event(os.path.join(data_dir, event_id))
        h["r"]   = np.hypot(h["x"], h["y"])
        h["phi"] = np.arctan2(h["y"], h["x"])
        p_cond, _, _ = preprocess_particles(10, 10, p, h)
        backr = h[np.random.rand(len(h)) < SAMPLE_FRACTION].copy()

        for sx, sy, sz in CONE_SIGNS:
            sig, combined = get_cone(h, t, p_cond, sx, sy, sz, backr)
            noi = combined[~combined['hit_id'].isin(sig['hit_id'])]

            all_r = np.concatenate([sig['r'].values, noi['r'].values])
            all_z = np.concatenate([sig['z'].values, noi['z'].values])

            with torch.no_grad():
                s1_prob = torch.sigmoid(
                    model_s1(torch.tensor(make_rz_grid(all_r, all_z).T).unsqueeze(0))
                ).item()

            if s1_prob > THRESH_S1 and len(sig) > 0:
                phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]

                rphi_r   = np.concatenate([sig['r'].values,   noi['r'].values])
                rphi_phi = np.concatenate([sig['phi'].values, noi['phi'].values])
                rphi_grid = torch.tensor(
                    make_rphi_grid(rphi_r, rphi_phi, sx, sy)
                ).unsqueeze(0)                              # (1, 101, 100)

                with torch.no_grad():
                    phi_probs = torch.sigmoid(
                        model_s2(rphi_grid)
                    ).squeeze(0).numpy()                   # (100,)

                phi_idx = np.clip(
                    ((sig['phi'].values - phi_min) / (phi_max - phi_min)
                     * SNN_PHI_BINS).astype(int),
                    0, SNN_PHI_BINS - 1
                )
                true_active = np.zeros(SNN_PHI_BINS, dtype=bool)
                true_active[phi_idx] = True

                all_phi_probs.append(phi_probs)
                all_true_active.append(true_active)

        print(f"  {event_id}: done  ({len(all_phi_probs)} signal cones so far)")

    except Exception as e:
        print(f"  {event_id}: skipped — {e}")

all_phi_probs   = np.array(all_phi_probs,   dtype=np.float32)  # (N, 100)
all_true_active = np.array(all_true_active, dtype=bool)         # (N, 100)

n_cones = len(all_phi_probs)
print(f"\nTotal signal cones evaluated : {n_cones}")
print(f"Mean true active bins        : {all_true_active.sum(axis=1).mean():.1f}")

# ── THRESHOLD SWEEP ───────────────────────────────────────────────────────────
thresholds      = np.linspace(0, 1, 200)
detection_rates = []
precisions      = []
mean_pred_bins  = []

for thr in thresholds:
    pred_active = all_phi_probs > thr

    overlap   = pred_active & all_true_active
    detected  = overlap.any(axis=1).sum()
    detection_rates.append(detected / n_cones)

    n_pred  = pred_active.sum()
    n_corr  = overlap.sum()
    if n_pred == 0:
        precisions.append(np.nan)  # don't plot this point
    else:
        precisions.append(n_corr / n_pred)

    mean_pred_bins.append(pred_active.sum(axis=1).mean())

detection_rates = np.array(detection_rates)
precisions      = np.array(precisions)
mean_pred_bins  = np.array(mean_pred_bins)

mean_true = all_true_active.sum(axis=1).mean()

# ── PLOT 1 — Detection vs Precision ──────────────────────────────────────────
fig, ax0 = plt.subplots(figsize=(6, 5))
ax0.plot(precisions, detection_rates, color=C[0], linewidth=1.8)
ax0.plot([0, 1], [0, 1], '--', color='grey', linewidth=0.8, alpha=0.5)
for thr, offset in zip([0.3, 0.5, 0.6, 0.8],
                        [(5, -12), (5, 5), (5, 5), (5, -12)]):
    idx  = np.argmin(np.abs(thresholds - thr))
    det  = detection_rates[idx]
    prec = precisions[idx]
    ax0.plot(prec, det, 'o', color=C[1], markersize=6, zorder=5)
    ax0.annotate(f'p={thr}', (prec, det),
                 textcoords="offset points", xytext=offset, fontsize=8)
ax0.set_xlabel('Precision  (correct bins / predicted bins)')
ax0.set_ylabel('Detection rate  (tracks with any overlap)')
ax0.set_title(f'Stage 2 — Detection vs Precision\n'
              f'{len(test_ids)} events · {n_cones} signal cones')
ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
plt.tight_layout()
plt.show()

# ── PLOT 2 — Detection rate & Precision vs threshold ─────────────────────────
fig, ax2 = plt.subplots(figsize=(6, 5))
ax2.plot(thresholds, detection_rates * 100, color=C[1],
         linewidth=1.8, label='Detection rate %')
ax2.plot(thresholds, precisions * 100,      color=C[0],
         linewidth=1.8, label='Precision %')
ax2.set_xlabel('Threshold')
ax2.set_ylabel('%')
ax2.set_title('Detection rate & Precision vs threshold')
ax2.legend()
plt.tight_layout()
plt.show()

# ── SUMMARY ───────────────────────────────────────────────────────────────────
print(f"\n{'threshold':>10}  {'detection':>12}  {'precision':>10}  {'mean bins':>10}")
print("─" * 50)
for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    idx = np.argmin(np.abs(thresholds - thr))
    print(f"{thr:>10.2f}  {detection_rates[idx]*100:>11.1f}%  "
          f"{precisions[idx]*100:>9.1f}%  {mean_pred_bins[idx]:>10.1f}")