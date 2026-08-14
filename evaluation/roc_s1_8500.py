# ================================================================================
# ROC_8500.py
# Evaluates the 8500-event stage 1 model on held-out test events (9500-9999).
# ================================================================================

from trackml.dataset import load_event

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\SNN")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\build_datasets")
import torch
import torch.nn as nn

import matplotlib.pyplot as plt
import os
import numpy as np

from Functions import preprocess_particles
from build_data8500 import (make_rz_grid, get_cone, CONE_SIGNS,
                             SNN_R_BINS, SNN_Z_BINS, SNN_R_MAX, SNN_Z_MAX,
                             SAMPLE_FRACTION)

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
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

C = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#33BBEE", "#EE3377"]

# ── CONFIG ────────────────────────────────────────────────────────────────────
THRESHOLD  = 0.5
SNN_R_BINS = 100
data_dir   = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_5"
model_path = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\s1_8500_h64_32_lr0.001_dr0.2.pt"
log_path   = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\stage1_8500_5714155.log"
test_ids   = [f"event{i:09d}" for i in range(9500, 10000)]

# ── MODEL ─────────────────────────────────────────────────────────────────────
class ConeFilterSNN(nn.Module):
    def __init__(self, input_size, h1, h2, bs):
        super().__init__()
        from snns import RadLIFLayer
        self.lif1 = RadLIFLayer(input_size, h1, bs,
                                normalization='batchnorm', dropout=0.2)
        self.lif2 = RadLIFLayer(h1, h2, bs,
                                normalization='batchnorm', dropout=0.2)
        self.fc   = nn.Linear(h2, 1)

    def forward(self, x):
        return self.fc(self.lif2(self.lif1(x)).mean(dim=1)).squeeze(-1)

    def set_bs(self, bs):
        self.lif1.batch_size = bs
        self.lif2.batch_size = bs

# ── LOAD MODEL ────────────────────────────────────────────────────────────────
model = ConeFilterSNN(SNN_R_BINS, h1=64, h2=32, bs=1)
checkpoint = torch.load(model_path, map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['model_state'])
model.eval()
model.set_bs(1)
print(f"Model loaded from {model_path}")
print(f"  Best epoch: {checkpoint['best_epoch']}  Best val: {min(checkpoint['val_losses']):.4f}")

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

            all_probs.append(prob)
            all_has_signal.append(len(sig) > 0)

        print(f"  {event_id}: done", flush=True)

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

cm    = np.array([[TP, FN], [FP, TN]], dtype=int)
acc   = (TP + TN) / n_total
eff   = TP / (TP + FN) if (TP + FN) > 0 else 0
thr_r = TN / (TN + FP) if (TN + FP) > 0 else 0

print(f"\nConfusion matrix at threshold={THRESHOLD}")
print(f"  TP={TP}  FN={FN}  TN={TN}  FP={FP}")
print(f"  Accuracy={acc*100:.1f}%  Efficiency={eff*100:.1f}%  Throwrate={thr_r*100:.1f}%")

# ── PLOT 1: ROC curve ─────────────────────────────────────────────────────────
fig, ax0 = plt.subplots(figsize=(6, 5))
ax0.plot(throwrates, efficiencies, color=C[0], linewidth=1.8,
         label='SNN cone filter (8500 events, h=64/32)')

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
ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
ax0.legend(loc='lower left', framealpha=0.9, fontsize=9)
ax0.grid(True, alpha=0.3, linestyle='--')
ax0.spines['top'].set_visible(True)
ax0.spines['right'].set_visible(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(model_path), "ROC_8500.png"), dpi=150)
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
ax1.legend()
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(model_path), "confidence_hist_8500.png"), dpi=150)
plt.show()

# ── PLOT 3: Confusion matrix ──────────────────────────────────────────────────
fig, ax2 = plt.subplots(figsize=(5, 5))
ax2.set_aspect('equal')
ax2.imshow(cm, cmap='Blues', vmin=0)
labels = [['TP', 'FN'], ['FP', 'TN']]
for i in range(2):
    for j in range(2):
        ax2.text(j, i, f'{labels[i][j]}\n{cm[i,j]}',
                 ha='center', va='center', fontsize=12,
                 color='white' if cm[i, j] > cm.max() * 0.5 else 'black')
ax2.set_xticks([0, 1]); ax2.set_yticks([0, 1])
ax2.set_xticklabels(['Pred: Signal', 'Pred: Noise'])
ax2.set_yticklabels(['True: Signal', 'True: Noise'])
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(model_path), "confusion_8500.png"), dpi=150)
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



# ── PLOT 4: Training and validation loss from log ─────────────────────────────
import re

train_losses, val_losses = [], []

with open(log_path) as f:
    for line in f:
        m = re.search(r"loss=([\d.]+)\s+val=([\d.]+)", line)
        if m:
            train_losses.append(float(m.group(1)))
            val_losses.append(float(m.group(2)))

epochs = range(1, len(train_losses) + 1)

fig, ax_l = plt.subplots(figsize=(6, 4))
ax_l.plot(epochs, train_losses, color='#0077BB', label='Train loss')
ax_l.plot(epochs, val_losses,   color='#EE7733', label='Val loss')
ax_l.set_xlabel('Epoch')
ax_l.set_ylabel('BCE Loss')
ax_l.legend()
ax_l.grid(True, alpha=0.3, linestyle='--')
ax_l.spines['top'].set_visible(True)
ax_l.spines['right'].set_visible(True)
plt.tight_layout()
plt.savefig(os.path.join(os.path.dirname(model_path), "loss_8500_final.pdf"), dpi=150)
plt.show()