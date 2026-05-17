# ================================================================================
# ROC_stage2.py
# Evaluates stage 2 phi localiser on held-out test events.
# Plots detection rate vs precision curve (stage 2 equivalent of ROC).
# ================================================================================

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Felix\SNN")

import torch
import torch.nn as nn
import os
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

from trackml.dataset import load_event
from Functions import preprocess_particles
from build_data100 import (make_rz_grid, make_rphi_grid, get_cone, CONE_SIGNS,
                        SNN_R_BINS, SNN_PHI_BINS, SNN_R_MAX, SNN_Z_MAX,
                        SAMPLE_FRACTION, CONE_PHI_RANGES)
from Stage1_ConeFilter100 import ConeFilterSNN
from Stage2_PhiFilter100   import PhiFinderSNN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.style.use(["science", "no-latex"])
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
THRESH_S1  = 0.5
data_dir   = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_1"
Save_dir   = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices1000"
test_ids   = [f"event{i:09d}" for i in range(2800, 2820)]  # 20 unseen events

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
print("Loading models...")
model_s1 = ConeFilterSNN(SNN_R_BINS, bs=1)
model_s1.load_state_dict(
    torch.load(os.path.join(Save_dir, "model_stage1.pt"))["model_state"]
)
model_s1.eval(); model_s1.set_bs(1)

model_s2 = PhiFinderSNN(bs=1)
model_s2.load_state_dict(
    torch.load(os.path.join(Save_dir, "model_stage2.pt"))["model_state"]
)
model_s2.eval(); model_s2.set_bs(1)

# ── EVALUATE ON TEST EVENTS ───────────────────────────────────────────────────
# For each signal cone that passes stage 1, collect:
#   phi_probs  — model output (100,) probability per phi bin
#   true_bins  — ground truth active phi bins

all_phi_probs  = []   # list of (100,) arrays
all_true_active = []  # list of (100,) bool arrays

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

            # stage 1 filter
            all_r = np.concatenate([sig['r'].values, noi['r'].values])
            all_z = np.concatenate([sig['z'].values, noi['z'].values])
            with torch.no_grad():
                s1_prob = torch.sigmoid(
                    model_s1(torch.tensor(make_rz_grid(all_r, all_z).T).unsqueeze(0))
                ).item()

            # only evaluate stage 2 on cones that pass stage 1 AND have signal
            if s1_prob > THRESH_S1 and len(sig) > 0:
                phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]

                rphi_r   = np.concatenate([sig['r'].values,   noi['r'].values])
                rphi_phi = np.concatenate([sig['phi'].values, noi['phi'].values])
                rphi_grid = torch.tensor(
                    make_rphi_grid(rphi_r, rphi_phi, sx, sy)
                ).unsqueeze(0)

                with torch.no_grad():
                    phi_probs = torch.sigmoid(
                        model_s2(rphi_grid)
                    ).squeeze(0).numpy()

                # true active bins
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
print(f"\nTotal signal cones evaluated: {n_cones}")
print(f"  Mean true active bins: {all_true_active.sum(axis=1).mean():.1f}")


# ── THRESHOLD SWEEP ───────────────────────────────────────────────────────────
thresholds      = np.linspace(0, 1, 200)
detection_rates = []   # fraction of cones where at least one bin found correctly
precisions      = []   # fraction of predicted bins that are correct
mean_pred_bins  = []   # mean number of predicted bins per cone

for thr in thresholds:
    pred_active = all_phi_probs > thr   # (N, 100) bool

    # detection: at least one predicted bin overlaps true bins
    overlap     = pred_active & all_true_active          # (N, 100)
    detected    = overlap.any(axis=1).sum()              # scalar
    detection_rates.append(detected / n_cones)

    # precision: of all predicted bins, how many are correct
    n_pred      = pred_active.sum()
    n_correct   = overlap.sum()
    if n_pred == 0:
        precisions.append(np.nan)  # don't plot this point
    else:
        precisions.append(n_correct / n_pred)

    mean_pred_bins.append(pred_active.sum(axis=1).mean())

detection_rates = np.array(detection_rates)
precisions      = np.array(precisions)
mean_pred_bins  = np.array(mean_pred_bins)


# ── PLOT 1 — Detection vs Precision ──────────────────────────────────────────
fig, ax0 = plt.subplots(figsize=(6, 5))
ax0.plot(precisions, detection_rates, color=C[0], linewidth=1.8)
ax0.plot([0, 1], [0, 1], '--', color='grey', linewidth=0.8, alpha=0.5)
ax0.set_xlabel('Precision  (correct bins / predicted bins)')
ax0.set_ylabel(f'Detection rate  (tracks with any overlap)')
ax0.set_title(
    f'Stage 2 — Detection vs Precision\n'
    f'{len(test_ids)} events · {n_cones} signal cones'
)
ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
for thr, offset in zip([0.3, 0.5, 0.6, 0.8], [(5,-12),(5,5),(5,5),(5,-12)]):
    idx  = np.argmin(np.abs(thresholds - thr))
    det  = detection_rates[idx]
    prec = precisions[idx]
    ax0.plot(prec, det, 'o', color=C[1], markersize=6, zorder=5)
    ax0.annotate(f'p={thr}', (prec, det),
                 textcoords="offset points", xytext=offset, fontsize=8)
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

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print(f"\n{'threshold':>10}  {'detection':>12}  {'precision':>10}  {'mean bins':>10}")
for thr in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]:
    idx  = np.argmin(np.abs(thresholds - thr))
    print(f"{thr:>10.2f}  {detection_rates[idx]*100:>11.1f}%  "
          f"{precisions[idx]*100:>9.1f}%  {mean_pred_bins[idx]:>10.1f}")
    



import numpy as np

paths = {
    "100 events":  r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices100\y_s2_filt.npy",
    "1000 events": r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices1000\y_s2_filt.npy",
}

print(f"\n{'='*55}")
print(f"  pos_weight analysis across dataset sizes")
print(f"{'='*55}\n")

for name, path in paths.items():
    y = np.load(path)
    n_pos = y.sum()
    n_neg = y.size - n_pos
    ratio = n_neg / n_pos
    capped = min(ratio, 10.0)
    suppression = ratio / 10.0

    print(f"  [{name}]")
    print(f"    shape          : {y.shape}")
    print(f"    n_pos          : {int(n_pos)}")
    print(f"    n_neg          : {int(n_neg)}")
    print(f"    true ratio     : {ratio:.1f}")
    print(f"    capped at 10   : {capped:.1f}")
    if ratio > 10:
        print(f"    *** suppressed by factor {suppression:.1f}x — cap is active! ***")
    else:
        print(f"    cap not active (ratio below 10)")
    print()