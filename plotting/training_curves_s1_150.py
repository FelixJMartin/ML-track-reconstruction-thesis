# ================================================================================
# training_curves_s1_150.py
# Training curves for the 150-bin Stage 1 model.
# Select noise level by uncommenting one line below.
# ================================================================================

import sys, os
import numpy as np
import torch
import scienceplots
import matplotlib.pyplot as plt

sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\SNN")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\training")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\build_datasets")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from trackml.dataset import load_event
from Functions import preprocess_particles
from build_data150_multinoise import make_rz_grid, get_cone, CONE_SIGNS
from train_s1_1000 import ConeFilterSNN

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "text.usetex":        False,
    "axes.spines.top":    True,
    "axes.spines.right":  True,
    "figure.facecolor":   "white",
    "axes.facecolor":     "white",
})

# ── SELECT MODEL ─────────────────────────────────────────────────────────────
# NOISE = 1   # 1% noise
NOISE = 2   # 2% noise
# NOISE = 5   # 5% noise
# NOISE = 10  # 10% noise

_BASE    = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\models_150"
S1_150_PATH = os.path.join(_BASE, f"s1_h512_256_lr0.001_dr0.1_noise{NOISE}pct.pt")

OUT_DIR  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Stage 1 — 150-bin loss & metrics ─────────────────────────────────────────
if os.path.exists(S1_150_PATH):
    ckpt         = torch.load(S1_150_PATH, map_location='cpu')
    train_losses = ckpt['train_losses']
    val_losses   = ckpt['val_losses']
    val_accs     = ckpt['val_accs']
    val_effs     = ckpt['val_effs']
    val_throws   = ckpt['val_throws']
    epochs       = range(1, len(train_losses) + 1)

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))

    ax_l.plot(epochs, train_losses, color='#0077BB', label='Train loss')
    ax_l.plot(epochs, val_losses,   color='#EE7733', label='Val loss')
    ax_l.set_xlabel('Epoch')
    ax_l.set_ylabel('BCE Loss')
    ax_l.legend()
    ax_l.grid(True, alpha=0.3, linestyle='--')

    ax_a.plot(epochs, [v * 100 for v in val_effs],   color='#0077BB', label='Efficiency %')
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
    print(f"Model not found: {S1_150_PATH}")

# ── Joint val accuracy: 1% vs 2% noise ───────────────────────────────────────
path_1pct  = os.path.join(_BASE, "s1_h512_256_lr0.001_dr0.1_noise1pct.pt")
path_2pct  = os.path.join(_BASE, "s1_h512_256_lr0.001_dr0.1_noise2pct.pt")
path_5pct  = os.path.join(_BASE, "s1_h512_256_lr0.001_dr0.1_noise5pct.pt")
path_10pct = os.path.join(_BASE, "s1_h512_256_lr0.001_dr0.1_noise10pct.pt")

fig, ax = plt.subplots(figsize=(7, 4))

for path, label, color in [
    (path_1pct,  '1% noise',  '#0077BB'),
    (path_2pct,  '2% noise',  '#EE7733'),
    (path_5pct,  '5% noise',  '#009988'),
    (path_10pct, '10% noise', '#AA3377'),
]:
    if os.path.exists(path):
        ck     = torch.load(path, map_location='cpu')
        accs   = ck['val_accs']
        epochs = range(1, len(accs) + 1)
        ax.plot(epochs, [v * 100 for v in accs], color=color, label=label)
    else:
        print(f"Skipping {label}: {path} not found")

ax.set_xlabel('Epoch')
ax.set_ylabel('Validation accuracy %')
ax.set_ylim(0, 105)
ax.legend()
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# ── Overlaid ROC: 1% vs 2% noise ─────────────────────────────────────────────
SNN_R_BINS = 150
SNN_Z_BINS = 150
data_dir   = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_1"
test_ids   = [f"event{i:09d}" for i in range(2800, 2820)]

MODELS = [
    (path_1pct,  '1% noise',  '#0077BB', 0.01),
    (path_2pct,  '2% noise',  '#EE7733', 0.02),
    (path_5pct,  '5% noise',  '#009988', 0.05),
    (path_10pct, '10% noise', '#AA3377', 0.10),
]

# MARKER_STYLES = {
#     0.5:  ('o', '$p=0.50$'),
#     0.8:  ('s', '$p=0.80$'),
#     0.9:  ('^', '$p=0.90$'),
#     0.95: ('D', '$p=0.95$'),
# }

# fig, ax_roc = plt.subplots(figsize=(6, 5))

# for model_path, label, color, sample_frac in MODELS:
#     if not os.path.exists(model_path):
#         print(f"Skipping ROC for {label}: model not found")
#         continue

#     model = ConeFilterSNN(SNN_R_BINS, h1=512, h2=256, bs=1)
#     ck    = torch.load(model_path, map_location='cpu')
#     model.load_state_dict(ck['model_state'])
#     model.eval(); model.set_bs(1)

#     all_probs, all_has_signal = [], []
#     print(f"\nEvaluating ROC for {label}...")
#     for event_id in test_ids:
#         try:
#             h, _, p, t = load_event(os.path.join(data_dir, event_id))
#             h["r"]   = np.hypot(h["x"], h["y"])
#             h["phi"] = np.arctan2(h["y"], h["x"])
#             p_cond, _, _ = preprocess_particles(10, 10, p, h)
#             backr = h[np.random.rand(len(h)) < sample_frac].copy()
#             for sx, sy, sz in CONE_SIGNS:
#                 sig, combined = get_cone(h, t, p_cond, sx, sy, sz, backr)
#                 noi = combined[~combined['hit_id'].isin(sig['hit_id'])]
#                 all_r = np.concatenate([sig['r'].values, noi['r'].values])
#                 all_z = np.concatenate([sig['z'].values, noi['z'].values])
#                 with torch.no_grad():
#                     prob = torch.sigmoid(
#                         model(torch.tensor(
#                             make_rz_grid(all_r, all_z, SNN_R_BINS, SNN_Z_BINS).T
#                         ).unsqueeze(0))
#                     ).item()
#                 all_probs.append(prob)
#                 all_has_signal.append(len(sig) > 0)
#         except Exception as e:
#             print(f"  {event_id}: skipped — {e}")

#     all_probs      = np.array(all_probs,      dtype=np.float32)
#     all_has_signal = np.array(all_has_signal, dtype=bool)

#     thresholds   = np.linspace(0, 1, 200)
#     efficiencies = []
#     throwrates   = []
#     for thr in thresholds:
#         pred = all_probs > thr
#         TP = ( pred &  all_has_signal).sum()
#         FN = (~pred &  all_has_signal).sum()
#         TN = (~pred & ~all_has_signal).sum()
#         FP = ( pred & ~all_has_signal).sum()
#         efficiencies.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
#         throwrates.append(  TN / (TN + FP) if (TN + FP) > 0 else 0)

#     ax_roc.plot(throwrates, efficiencies, color=color, linewidth=1.8, label=label)

#     for thr, (marker, mlabel) in MARKER_STYLES.items():
#         p_ = all_probs > thr
#         tp = ( p_ &  all_has_signal).sum()
#         fn = (~p_ &  all_has_signal).sum()
#         tn = (~p_ & ~all_has_signal).sum()
#         fp = ( p_ & ~all_has_signal).sum()
#         e  = tp / (tp + fn) if (tp + fn) > 0 else 0
#         tr = tn / (tn + fp) if (tn + fp) > 0 else 0
#         ax_roc.plot(tr, e, marker=marker, color='black', markersize=7, zorder=5)

# ax_roc.set_xlabel('Noise throwrate')
# ax_roc.set_ylabel('Signal efficiency')
# ax_roc.set_xlim(0, 1)
# ax_roc.set_ylim(0, 1)
# ax_roc.legend(loc='lower left', frameon=True, framealpha=0.7,
#               facecolor='white', edgecolor='#888888')
# ax_roc.grid(True, alpha=0.3, linestyle='--')
# plt.tight_layout()
# plt.show()

# ── Example matrices from matrices_150_5pct ───────────────────────────────────
mat_dir = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices_150_5pct"
X_rz   = np.load(os.path.join(mat_dir, "X_rz.npy"))
y_s1   = np.load(os.path.join(mat_dir, "y_stage1.npy"))

sig_idx = np.where(y_s1 == 1)[0]
noi_idx = np.where(y_s1 == 0)[0]
np.random.seed(42)
show_sig = np.random.choice(sig_idx, 4, replace=False)
show_noi = np.random.choice(noi_idx, 4, replace=False)

fig, axes = plt.subplots(2, 4, figsize=(13, 6))
fig.subplots_adjust(hspace=0.04, wspace=0.04)

for col, idx in enumerate(show_sig):
    ax = axes[0, col]
    ax.imshow(X_rz[idx].T, origin='lower', aspect='auto', cmap='Blues',
              extent=[0, 1100, 0, 3000])
    ax.set_xticks([])
    ax.set_yticks([])

for col, idx in enumerate(show_noi):
    ax = axes[1, col]
    ax.imshow(X_rz[idx].T, origin='lower', aspect='auto', cmap='Oranges',
              extent=[0, 1100, 0, 3000])
    ax.set_xticks([])
    ax.set_yticks([])

plt.tight_layout()
plt.show()


