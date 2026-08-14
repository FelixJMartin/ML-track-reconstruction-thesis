# ================================================================================
# roc_s1_150.py
# Evaluates the 150-bin stage 1 cone filter on held-out test events (2800-2820).
# ================================================================================

from trackml.dataset import load_event

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\SNN")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\training")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\build_datasets")
import torch
import torch.nn as nn

import scienceplots
import matplotlib.pyplot as plt

import os
import numpy as np

from Functions import preprocess_particles
from build_data150_multinoise import (make_rz_grid, get_cone, CONE_SIGNS, CONE_PHI_RANGES)
from train_s1_1000 import ConeFilterSNN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

SNN_R_BINS = 150
SNN_Z_BINS = 150

plt.rcParams.update({
    "text.usetex":        False,
    "figure.dpi":         150,
    "font.size":          10,
    "axes.titlesize":     11,
    "axes.labelsize":     10,
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    8,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

C = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#33BBEE", "#EE3377"]

# ── SELECT MODEL ─────────────────────────────────────────────────────────────
NOISE = 1   # 1% noise
# NOISE = 2   # 2% noise
# NOISE = 5   # 5% noise
# NOISE = 10  # 10% noise

# ── CONFIG ────────────────────────────────────────────────────────────────────
THRESHOLD       = 0.5
SAMPLE_FRACTION = NOISE / 100
data_dir        = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_1"
_BASE           = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\models_150"
model_path      = os.path.join(_BASE, f"s1_h512_256_lr0.001_dr0.1_noise{NOISE}pct.pt")
test_ids        = [f"event{i:09d}" for i in range(2800, 2820)]  # 20 unseen events

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
model = ConeFilterSNN(SNN_R_BINS, h1=512, h2=256, bs=1)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state'])
model.eval()
model.set_bs(1)
print(f"Model loaded from {model_path}")

# ── PLOT TRAINING CURVES ──────────────────────────────────────────────────────
if all(k in checkpoint for k in ('train_losses', 'val_losses', 'val_accs', 'val_effs', 'val_throws')):
    import scienceplots
    plt.style.use(["science", "no-latex"])
    train_losses = checkpoint['train_losses']
    val_losses   = checkpoint['val_losses']
    val_accs     = checkpoint['val_accs']
    val_effs     = checkpoint['val_effs']
    val_throws   = checkpoint['val_throws']
    epochs       = range(1, len(train_losses) + 1)

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))

    ax_l.plot(epochs, train_losses, color='#0077BB', label='Train loss')
    ax_l.plot(epochs, val_losses,   color='#EE7733', label='Val loss')
    ax_l.set_xlabel('Epoch')
    ax_l.set_ylabel('BCE Loss')
    ax_l.legend()
    ax_l.grid(True, alpha=0.3, linestyle='--')

    ax_a.plot(epochs, [v * 100 for v in val_effs],    color='#0077BB', label='Efficiency %')
    ax_a.plot(epochs, [v * 100 for v in val_throws],  color='#009988', label='Throwrate %')
    ax_a.plot(epochs, [v * 100 for v in val_accs],    color='#EE7733', label='Accuracy %', linestyle='--')
    ax_a.set_xlabel('Epoch')
    ax_a.set_ylabel('%')
    ax_a.set_ylim(0, 105)
    ax_a.legend()
    ax_a.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()
else:
    print("No training curves found in checkpoint.")

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
                    model(torch.tensor(
                        make_rz_grid(all_r, all_z, SNN_R_BINS, SNN_Z_BINS).T
                    ).unsqueeze(0))
                ).item()

            all_probs.append(prob)
            all_has_signal.append(len(sig) > 0)

        print(f"  {event_id}: done")

    except Exception as e:
        print(f"  {event_id}: skipped — {e}")

all_probs      = np.array(all_probs,      dtype=np.float32)
all_has_signal = np.array(all_has_signal, dtype=bool)

n_signal = all_has_signal.sum()
n_noise  = (~all_has_signal).sum()
n_total  = len(all_probs)
print(f"\nTotal cones : {n_total}  signal={n_signal}  noise={n_noise}")

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
pred  = all_probs > THRESHOLD
TP    = ( pred &  all_has_signal).sum()
FN    = (~pred &  all_has_signal).sum()
TN    = (~pred & ~all_has_signal).sum()
FP    = ( pred & ~all_has_signal).sum()
cm    = np.array([[TP, FN], [FP, TN]], dtype=int)
acc   = (TP + TN) / n_total
eff   = TP / (TP + FN) if (TP + FN) > 0 else 0
thr_r = TN / (TN + FP) if (TN + FP) > 0 else 0

print(f"\nAt threshold={THRESHOLD}:  TP={TP}  FN={FN}  TN={TN}  FP={FP}")
print(f"  Accuracy={acc*100:.1f}%  Efficiency={eff*100:.1f}%  Throwrate={thr_r*100:.1f}%")

# ── PLOT 1: ROC curve ─────────────────────────────────────────────────────────
fig, ax0 = plt.subplots(figsize=(6, 5))

ax0.plot(throwrates, efficiencies, color=C[0], linewidth=1.8, label='SNN cone filter (150-bin)')

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
    ax0.plot(tr, e, marker=marker, color=color, markersize=7, zorder=5, label=label)

ax0.set_xlabel(f'Noise throwrate  ({n_noise} noise cones)')
ax0.set_ylabel(f'Signal efficiency  ({n_signal} signal cones)')
ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
ax0.legend(loc='lower left', frameon=True, framealpha=0.7,
           facecolor='white', edgecolor='#888888')
ax0.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# ── PLOT 2: Confidence histogram ──────────────────────────────────────────────
fig, ax1 = plt.subplots(figsize=(6, 5))
ax1.hist(all_probs[ all_has_signal], bins=40, alpha=0.65,
         color=C[1], label=f'Signal  (n={n_signal})', density=True)
ax1.hist(all_probs[~all_has_signal], bins=40, alpha=0.65,
         color=C[0], label=f'Noise  (n={n_noise})',  density=True)
ax1.axvline(THRESHOLD, color='black', lw=1.0, ls='--', label=f'threshold={THRESHOLD}')
ax1.set_xlabel('Model confidence  p')
ax1.set_ylabel('Density')
ax1.legend(frameon=True, framealpha=0.7, facecolor='white', edgecolor='#888888')
ax1.grid(True, alpha=0.3, linestyle='--')
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
                 color='white' if cm[i,j] > cm.max()*0.5 else 'black')
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
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
