# ================================================================================
# ROC_stage2.py
# Evaluates the 8500-event stage 2 model on held-out test events (9500-9999).
# ================================================================================

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\SNN")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\build_datasets")
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os
import re
from trackml.dataset import load_event
from snns import RadLIFLayer
from Functions import preprocess_particles
from build_data8500 import (make_rz_grid, make_rphi_grid, get_cone, CONE_SIGNS,
                             SNN_R_BINS, SNN_PHI_BINS, SNN_R_MAX, SNN_Z_MAX,
                             SAMPLE_FRACTION, CONE_PHI_RANGES)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.rcParams.update({
    "text.usetex": False, "figure.dpi": 150,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    "axes.spines.top": True, "axes.spines.right": True,
})

C = ["#0077BB", "#EE7733", "#009988", "#CC3311"]

# ── CONFIG ────────────────────────────────────────────────────────────────────
THRESH_S1  = 0.5
THRESH_S2  = 0.5
SNN_INPUT_SIZE = SNN_R_BINS + 1

S1_PATH    = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\s1_8500_ep115.pt"
S2_PATH    = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\s2_h64_32_final.pt"
LOG_PATH   = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\stage2_8500_5714591_final.log"
OUT_DIR    = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500"
data_dir   = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_5"
test_ids   = [f"event{i:09d}" for i in range(9500, 10000)]

# ── MODELS ────────────────────────────────────────────────────────────────────
class ConeFilterSNN(nn.Module):
    def __init__(self, input_size=100, h1=64, h2=32, bs=1):
        super().__init__()
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


class PhiFinderSNN(nn.Module):
    def __init__(self, h1=64, h2=32, bs=1):
        super().__init__()
        self.lif1 = RadLIFLayer(SNN_INPUT_SIZE, h1, bs,
                                normalization='batchnorm', dropout=0.2)
        self.lif2 = RadLIFLayer(h1, h2, bs,
                                normalization='batchnorm', dropout=0.2)
        self.fc   = nn.Linear(h2, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.fc(self.lif2(self.lif1(x))).squeeze(-1)

    def set_bs(self, bs):
        self.lif1.batch_size = bs
        self.lif2.batch_size = bs


# ── LOAD MODELS ───────────────────────────────────────────────────────────────
s1 = ConeFilterSNN(input_size=SNN_R_BINS, h1=64, h2=32, bs=1)
s1.load_state_dict(torch.load(S1_PATH, map_location='cpu')['model_state'])
s1.eval()
s1.set_bs(1)
print(f"Stage 1 loaded from {S1_PATH}")

s2_ckpt = torch.load(S2_PATH, map_location='cpu')
cfg     = s2_ckpt.get('config', {'H1': 64, 'H2': 32})
s2 = PhiFinderSNN(h1=cfg['H1'], h2=cfg['H2'], bs=1)
s2.load_state_dict(s2_ckpt['model_state'])
s2.eval()
s2.set_bs(1)
print(f"Stage 2 loaded — H1={cfg['H1']} H2={cfg['H2']}")

# ── PARSE LOG ─────────────────────────────────────────────────────────────────
train_losses, val_losses, val_ious, val_overlaps = [], [], [], []
with open(LOG_PATH) as f:
    for line in f:
        m = re.search(r"loss=([\d.]+)\s+val=([\d.]+)\s+iou=([\d.]+)%\s+overlap=([\d.]+)%", line)
        if m:
            train_losses.append(float(m.group(1)))
            val_losses.append(float(m.group(2)))
            val_ious.append(float(m.group(3)))
            val_overlaps.append(float(m.group(4)))

print(f"Log parsed: {len(train_losses)} epochs")

# ── PLOT 1: loss and IoU curves from log ──────────────────────────────────────
epochs = range(1, len(train_losses) + 1)
fig, (ax_l, ax_m) = plt.subplots(1, 2, figsize=(11, 4))

ax_l.plot(epochs, train_losses, color=C[0], label='Train loss')
ax_l.plot(epochs, val_losses,   color=C[1], label='Val loss')
ax_l.set_xlabel('Epoch')
ax_l.set_ylabel('BCE Loss')
ax_l.legend()
ax_l.grid(True, alpha=0.3, linestyle='--')

ax_m.plot(epochs, val_ious,     color=C[0], label='IoU %')
ax_m.plot(epochs, val_overlaps, color=C[1], label='Overlap %')
ax_m.set_xlabel('Epoch')
ax_m.set_ylabel('%')
ax_m.set_ylim(0, 105)
ax_m.legend()
ax_m.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

best_epoch = int(np.argmin(val_losses)) + 1
print(f"\nStage 2 training summary ({len(train_losses)} epochs)")
print(f"  Best val loss : {min(val_losses):.4f}  (epoch {best_epoch})")
print(f"  Final val loss: {val_losses[-1]:.4f}")
print(f"  Best IoU      : {max(val_ious):.1f}%  (epoch {int(np.argmax(val_ious))+1})")
print(f"  Final IoU     : {val_ious[-1]:.1f}%")
print(f"  Best overlap  : {max(val_overlaps):.1f}%")
print(f"  Final overlap : {val_overlaps[-1]:.1f}%")




# ── EVALUATE ON 500 HELD-OUT EVENTS ───────────────────────────────────────────
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
            if len(sig) == 0:
                continue
            noi = combined[~combined['hit_id'].isin(sig['hit_id'])]

            all_r = np.concatenate([sig['r'].values, noi['r'].values])
            all_z = np.concatenate([sig['z'].values, noi['z'].values])

            with torch.no_grad():
                s1_prob = torch.sigmoid(
                    s1(torch.tensor(make_rz_grid(all_r, all_z).T).unsqueeze(0))
                ).item()

            if s1_prob > THRESH_S1:
                phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
                rphi_r   = np.concatenate([sig['r'].values, noi['r'].values])
                rphi_phi = np.concatenate([sig['phi'].values, noi['phi'].values])
                rphi_grid = torch.tensor(
                    make_rphi_grid(rphi_r, rphi_phi, sx, sy)
                ).unsqueeze(0)

                with torch.no_grad():
                    s2.set_bs(1)
                    phi_probs = torch.sigmoid(s2(rphi_grid)).squeeze(0).numpy()

                phi_idx = np.clip(
                    ((sig['phi'].values - phi_min) / (phi_max - phi_min)
                     * SNN_PHI_BINS).astype(int),
                    0, SNN_PHI_BINS - 1
                )
                true_active = np.zeros(SNN_PHI_BINS, dtype=bool)
                true_active[phi_idx] = True

                all_phi_probs.append(phi_probs)
                all_true_active.append(true_active)

        print(f"  {event_id}: done  ({len(all_phi_probs)} cones so far)", flush=True)

    except Exception as e:
        print(f"  {event_id}: skipped — {e}")

all_phi_probs   = np.array(all_phi_probs,   dtype=np.float32)
all_true_active = np.array(all_true_active, dtype=bool)
n_cones         = len(all_phi_probs)
print(f"\nTotal signal cones evaluated: {n_cones}")
print(f"Mean true active bins: {all_true_active.sum(axis=1).mean():.1f}")

# ── THRESHOLD SWEEP ───────────────────────────────────────────────────────────
thresholds      = np.linspace(0, 1, 200)
detection_rates = []
precisions      = []
mean_pred_bins  = []

for thr in thresholds:
    pred_active = all_phi_probs > thr
    overlap     = pred_active & all_true_active
    detected    = overlap.any(axis=1).sum()
    detection_rates.append(detected / n_cones)
    n_pred    = pred_active.sum()
    n_correct = overlap.sum()
    precisions.append(n_correct / n_pred if n_pred > 0 else np.nan)
    mean_pred_bins.append(pred_active.sum(axis=1).mean())

detection_rates = np.array(detection_rates)
precisions      = np.array(precisions)
mean_pred_bins  = np.array(mean_pred_bins)

# ── PLOT 2: detection vs precision curve ──────────────────────────────────────
fig, ax0 = plt.subplots(figsize=(6, 5))
ax0.plot(precisions, detection_rates, color=C[0], linewidth=1.8,
         label='Stage 2 (8500 events, h=64/32)')
ax0.plot([0, 1], [0, 1], '--', color='grey', linewidth=0.8, alpha=0.5)
ax0.set_xlabel('Precision  (correct bins / predicted bins)')
ax0.set_ylabel('Detection rate  (cones with any overlap)')
ax0.set_xlim(0, 1)
ax0.set_ylim(0, 1)
for thr in [0.3, 0.5, 0.7, 0.9]:
    idx = np.argmin(np.abs(thresholds - thr))
    ax0.plot(precisions[idx], detection_rates[idx], 'o',
             color=C[1], markersize=6, zorder=5)
    ax0.annotate(f'p={thr}', (precisions[idx], detection_rates[idx]),
                 textcoords="offset points", xytext=(5, 5), fontsize=8)
ax0.legend()
ax0.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "s2_detection_precision.pdf"), dpi=150)
plt.show()

# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print(f"\n{'threshold':>10}  {'detection':>12}  {'precision':>10}  {'mean bins':>10}")
for thr in [0.3, 0.5, 0.6, 0.7, 0.8, 0.9]:
    idx = np.argmin(np.abs(thresholds - thr))
    print(f"{thr:>10.2f}  {detection_rates[idx]*100:>11.1f}%  "
          f"{precisions[idx]*100:>9.1f}%  {mean_pred_bins[idx]:>10.1f}")

# ── PLOT 3: example cone localisation ─────────────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(14, 6))
n_shown = 0

for event_id in test_ids:
    if n_shown >= 8:
        break
    try:
        h, _, p, t = load_event(os.path.join(data_dir, event_id))
        h["r"]   = np.hypot(h["x"], h["y"])
        h["phi"] = np.arctan2(h["y"], h["x"])
        p_cond, _, _ = preprocess_particles(10, 10, p, h)
        backr = h[np.random.rand(len(h)) < SAMPLE_FRACTION].copy()

        for sx, sy, sz in CONE_SIGNS:
            if n_shown >= 8:
                break
            sig, combined = get_cone(h, t, p_cond, sx, sy, sz, backr)
            if len(sig) == 0:
                continue
            noi = combined[~combined['hit_id'].isin(sig['hit_id'])]

            all_r = np.concatenate([sig['r'].values, noi['r'].values])
            all_z = np.concatenate([sig['z'].values, noi['z'].values])

            with torch.no_grad():
                s1_prob = torch.sigmoid(
                    s1(torch.tensor(make_rz_grid(all_r, all_z).T).unsqueeze(0))
                ).item()

            if s1_prob > THRESH_S1:
                phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
                rphi_r   = np.concatenate([sig['r'].values, noi['r'].values])
                rphi_phi = np.concatenate([sig['phi'].values, noi['phi'].values])
                rphi_grid = torch.tensor(
                    make_rphi_grid(rphi_r, rphi_phi, sx, sy)
                ).unsqueeze(0)

                with torch.no_grad():
                    s2.set_bs(1)
                    phi_probs = torch.sigmoid(s2(rphi_grid)).squeeze(0).numpy()

                phi_idx = np.clip(
                    ((sig['phi'].values - phi_min) / (phi_max - phi_min)
                     * SNN_PHI_BINS).astype(int),
                    0, SNN_PHI_BINS - 1
                )
                true_active = np.zeros(SNN_PHI_BINS, dtype=bool)
                true_active[phi_idx] = True

                ax = axes[n_shown // 4][n_shown % 4]
                ax.plot(phi_probs, color=C[0], linewidth=1.2, label='Predicted p')
                ax.fill_between(range(SNN_PHI_BINS),
                                0, true_active.astype(float),
                                alpha=0.3, color=C[1], label='True bins')
                ax.axhline(THRESH_S2, color='black', lw=0.8, ls='--')
                ax.set_ylim(0, 1)
                ax.set_xlabel('$\\phi$ bin')
                ax.set_ylabel('p')
                ax.set_title(f'Cone {n_shown+1}')
                if n_shown == 0:
                    ax.legend(fontsize=7)
                n_shown += 1

    except Exception as e:
        continue

plt.tight_layout()
plt.show()
