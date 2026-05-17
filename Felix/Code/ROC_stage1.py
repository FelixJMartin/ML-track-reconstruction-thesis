# ================================================================================
# ROC.py
# Evaluates stage 1 cone filter on held-out test events and plots ROC curve.
# ================================================================================

from trackml.dataset import load_event

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Felix\SNN")
import torch
import torch.nn as nn

# plotting
import matplotlib.pyplot as plt

# other
import os
import numpy as np

# local
from Functions import preprocess_particles
from build_data import (make_rz_grid, get_cone, CONE_SIGNS,
                        SNN_R_BINS, SNN_Z_BINS, SNN_R_MAX, SNN_Z_MAX,
                        SAMPLE_FRACTION)

from Stage1_ConeFilter import ConeFilterSNN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.rcParams.update({
    "text.usetex":        False,
    "figure.dpi":         150,
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

C = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#33BBEE", "#EE3377"]

# ── CONFIG ────────────────────────────────────────────────────────────────────
THRESHOLD = 0.5
data_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_1"
Save_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices1000"
test_ids  = [f"event{i:09d}" for i in range(2800, 2820)]  # 20 unseen events



# ── LOAD MODEL ────────────────────────────────────────────────────────────────
model_path = os.path.join(Save_dir, "model_stage1.pt")
model = ConeFilterSNN(SNN_R_BINS, bs=1)
checkpoint = torch.load(model_path)
model.load_state_dict(checkpoint['model_state'])
model.eval()
model.set_bs(1)
print(f"Model loaded from {model_path}")

# ── EVALUATE ALL TEST CONES ───────────────────────────────────────────────────
all_probs      = []
all_has_signal = []

print(f"\nEvaluating {len(test_ids)} test events ({len(test_ids)*8} cones total)...")

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
                prob = torch.sigmoid(
                    model(torch.tensor(make_rz_grid(all_r, all_z).T).unsqueeze(0))
                ).item()

            has_signal = len(sig) > 0
            all_probs.append(prob)
            all_has_signal.append(has_signal)

        print(f"  {event_id}: done")

    except Exception as e:
        print(f"  {event_id}: skipped — {e}")

# ── dtype fix ─────────────────────────────────────────────────────────────────
all_probs      = np.array(all_probs,      dtype=np.float32)
all_has_signal = np.array(all_has_signal, dtype=bool)

n_signal = all_has_signal.sum()
n_noise  = (~all_has_signal).sum()
n_total  = len(all_probs)
print(f"\nTotal cones evaluated : {n_total}")
print(f"  Signal cones        : {n_signal}")
print(f"  Noise-only cones    : {n_noise}")


# ── THRESHOLD SWEEP → ROC ─────────────────────────────────────────────────────
thresholds   = np.linspace(0, 1, 200)
efficiencies = []
throwrates   = []

for thr in thresholds:
    pred_track = all_probs > thr
    TP = ( pred_track &  all_has_signal).sum()
    FN = (~pred_track &  all_has_signal).sum()
    TN = (~pred_track & ~all_has_signal).sum()
    FP = ( pred_track & ~all_has_signal).sum()
    efficiencies.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
    throwrates.append(  TN / (TN + FP) if (TN + FP) > 0 else 0)

efficiencies = np.array(efficiencies)
throwrates   = np.array(throwrates)


# ── CONFUSION MATRIX at chosen threshold ──────────────────────────────────────
pred = all_probs > THRESHOLD
TP = ( pred &  all_has_signal).sum()
FN = (~pred &  all_has_signal).sum()
TN = (~pred & ~all_has_signal).sum()
FP = ( pred & ~all_has_signal).sum()

cm  = np.array([[TP, FN],
                [FP, TN]], dtype=int)
acc   = (TP + TN) / n_total
eff   = TP / (TP + FN) if (TP + FN) > 0 else 0
thr_r = TN / (TN + FP) if (TN + FP) > 0 else 0

print(f"\nConfusion matrix at threshold={THRESHOLD}")
print(f"  TP={TP}  FN={FN}  TN={TN}  FP={FP}")
print(f"  Accuracy={acc*100:.1f}%  Efficiency={eff*100:.1f}%  Throwrate={thr_r*100:.1f}%")


# ── PLOT 1: ROC curve ─────────────────────────────────────────────────────────
fig, ax0 = plt.subplots(figsize=(6, 5))

ax0.plot(throwrates, efficiencies, color=C[0], linewidth=1.8, label='SNN cone filter')

marker_styles = {
    0.5:  ('o', 'black', '$p=0.50$'),
    0.8:  ('s', 'black', '$p=0.80$'),
    0.9:  ('^', 'black', '$p=0.90$'),
    0.95: ('D', 'black', '$p=0.95$'),
}

for thr, (marker, color, label) in marker_styles.items():
    p_ = all_probs > thr
    tp = ( p_ &  all_has_signal).sum()
    fn = (~p_ &  all_has_signal).sum()
    tn = (~p_ & ~all_has_signal).sum()
    fp = ( p_ & ~all_has_signal).sum()
    e  = tp / (tp + fn) if (tp + fn) > 0 else 0
    tr = tn / (tn + fp) if (tn + fp) > 0 else 0
    ax0.plot(tr, e, marker=marker, color=color,
             markersize=7, zorder=5, label=label)

ax0.set_xlabel(f'Noise throwrate  ({n_noise} noise cones)')
ax0.set_ylabel(f'Signal efficiency  ({n_signal} signal cones)')
ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
ax0.legend(loc='lower left', framealpha=0.9, fontsize=9)
plt.tight_layout()
ax0.grid(True, alpha=0.3, linestyle='--')
ax0.spines['top'].set_visible(True)
ax0.spines['right'].set_visible(True)
plt.show()

# ── PLOT 2: Confidence histogram ──────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(6, 5))
ax1.hist(all_probs[ all_has_signal], bins=40, alpha=0.65,
         color=C[1], label=f'Signal  (n={n_signal})',  density=True)
ax1.hist(all_probs[~all_has_signal], bins=40, alpha=0.65,
         color=C[0], label=f'Noise  (n={n_noise})', density=True)
ax1.axvline(THRESHOLD, color='black', lw=1.0, ls='--', label=f'threshold={THRESHOLD}')
ax1.set_xlabel('Model confidence  p')
ax1.set_ylabel('Density')
ax1.legend()
plt.tight_layout()
plt.show()

# ── PLOT 3: Confusion matrix ───────────────────────────────────────────────────
fig, ax2 = plt.subplots(figsize=(5, 5))
ax2.set_aspect('equal')
ax2.imshow(cm, cmap='Blues', vmin=0)

labels = [['TP', 'FN'], ['FP', 'TN']]
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f'{labels[i][j]}\n{cm[i,j]}',
                 ha='center', va='center', fontsize=12,
                 color='white' if cm[i, j] > cm.max() * 0.5 else 'black')

ax2.set_xticks([0, 1])
ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Pred: Signal', 'Pred: Noise'])
ax2.set_yticklabels(['True: Signal', 'True: Noise'])
plt.tight_layout()
plt.show()


# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print(f"\n{'threshold':>10}  {'efficiency':>12}  {'throwrate':>10}  {'cones kept':>12}")
for thr in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
    p_  = all_probs > thr
    tp  = ( p_ &  all_has_signal).sum()
    fn  = (~p_ &  all_has_signal).sum()
    tn  = (~p_ & ~all_has_signal).sum()
    fp  = ( p_ & ~all_has_signal).sum()
    e   = tp / (tp + fn) if (tp + fn) > 0 else 0
    tr  = tn / (tn + fp) if (tn + fp) > 0 else 0
    print(f"{thr:>10.2f}  {e*100:>11.1f}%  {tr*100:>9.1f}%  {p_.sum():>12}")