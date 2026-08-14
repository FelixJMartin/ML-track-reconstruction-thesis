# ================================================================================
# train_s2_8500.py  — 10 epoch diagnostic, no stage 1 filter, 8500 events
# ================================================================================

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\SNN")

import os
import numpy as np
import torch
import torch.nn as nn
from snns import RadLIFLayer

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": False, "figure.dpi": 150,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": True, "axes.spines.right": True,
})

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── CONFIG ────────────────────────────────────────────────────────────────────
H1 = 64
H2 = 32
LR             = 1e-3
BATCH_SIZE     = 32
DROPOUT        = 0.2
EPOCHS         = 10
SNN_INPUT_SIZE = 101

TRAIN_DIR  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\train"
VAL_DIR    = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\val"
MODEL_PATH = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\s2_h128_64_test.pt"

# ── MODEL ─────────────────────────────────────────────────────────────────────
class PhiFinderSNN(nn.Module):
    def __init__(self, h1=H1, h2=H2, bs=BATCH_SIZE):
        super().__init__()
        self.lif1 = RadLIFLayer(SNN_INPUT_SIZE, h1, bs,
                                normalization='batchnorm', dropout=DROPOUT)
        self.lif2 = RadLIFLayer(h1, h2, bs,
                                normalization='batchnorm', dropout=DROPOUT)
        self.fc   = nn.Linear(h2, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.fc(self.lif2(self.lif1(x))).squeeze(-1)

    def set_bs(self, bs):
        self.lif1.batch_size = bs
        self.lif2.batch_size = bs


# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("Loading data...")
Xtr = np.load(os.path.join(TRAIN_DIR, "X_rphi.npy"))
ytr = np.load(os.path.join(TRAIN_DIR, "y_stage2.npy"))
Xv  = np.load(os.path.join(VAL_DIR,   "X_rphi.npy"))
yv  = np.load(os.path.join(VAL_DIR,   "y_stage2.npy"))

print(f"Train: {len(Xtr)} matrices  signal={int((ytr.sum(axis=1)>0).sum())}  "
      f"noise={int((ytr.sum(axis=1)==0).sum())}")
print(f"Val:   {len(Xv)} matrices  signal={int((yv.sum(axis=1)>0).sum())}  "
      f"noise={int((yv.sum(axis=1)==0).sum())}")

# ── SETUP ─────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model   = PhiFinderSNN(h1=H1, h2=H2, bs=BATCH_SIZE).to(device)
opt     = torch.optim.Adam(model.parameters(), lr=LR)
sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=3, factor=0.5)

n_pos      = int((ytr.sum(axis=1) > 0).sum())
n_neg      = int((ytr.sum(axis=1) == 0).sum())
pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)
loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print(f"pos_weight=1.0  signal_matrices={n_pos}  noise_matrices={n_neg}\n")

total_params = sum(p.numel() for p in model.parameters())
print(f"Model parameters: {total_params}")
print(f"Samples per parameter: {len(Xtr)/total_params:.2f}\n")

Xtr = torch.tensor(Xtr, dtype=torch.float32).to(device)
ytr = torch.tensor(ytr, dtype=torch.float32).to(device)
Xv  = torch.tensor(Xv,  dtype=torch.float32).to(device)
yv  = torch.tensor(yv,  dtype=torch.float32).to(device)
yv_sig_mask = yv.sum(dim=1) > 0

# ── TRAIN ─────────────────────────────────────────────────────────────────────
train_losses, val_losses, val_ious, val_overlaps = [], [], [], []
best_iou = 0.0

for ep in range(EPOCHS):
    model.train()
    idx = torch.randperm(len(Xtr))
    Xtr = Xtr[idx]
    ytr = ytr[idx]

    tl, nb = 0.0, 0
    for b in range(0, len(Xtr), BATCH_SIZE):
        xb = Xtr[b:b+BATCH_SIZE]
        yb = ytr[b:b+BATCH_SIZE]
        model.set_bs(len(xb))
        loss = loss_fn(model(xb), yb)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        tl += loss.item()
        nb += 1

    model.eval()
    model.set_bs(len(Xv))
    with torch.no_grad():
        vp  = model(Xv)
        vl  = loss_fn(vp, yv).item()
        iou = overlap = 0.0
        if yv_sig_mask.sum() > 0:
            pred_a = torch.sigmoid(vp[yv_sig_mask]) > 0.5
            true_a = yv[yv_sig_mask] > 0.5
            inter  = (pred_a & true_a).float().sum(dim=1)
            union  = (pred_a | true_a).float().sum(dim=1)
            iou     = (inter / union.clamp(min=1)).mean().item()
            overlap = (inter > 0).float().mean().item()

    train_losses.append(tl / nb)
    val_losses.append(vl)
    val_ious.append(iou)
    val_overlaps.append(overlap)
    model.set_bs(BATCH_SIZE)
    sched.step(vl)
    best_iou = max(best_iou, iou)

    print(f"  ep{ep+1:3d}/{EPOCHS}  loss={tl/nb:.4f}  val={vl:.4f}  "
          f"iou={iou*100:.1f}%  overlap={overlap*100:.1f}%  "
          f"best={best_iou*100:.1f}%", flush=True)

print(f"\nBest IoU: {best_iou*100:.1f}%")

torch.save({
    'model_state':  model.state_dict(),
    'train_losses': train_losses,
    'val_losses':   val_losses,
    'val_ious':     val_ious,
    'val_overlaps': val_overlaps,
    'config': {'H1': H1, 'H2': H2, 'LR': LR,
               'BATCH_SIZE': BATCH_SIZE, 'DROPOUT': DROPOUT}
}, MODEL_PATH)
print(f"Model saved to {MODEL_PATH}")

# ── PLOT ──────────────────────────────────────────────────────────────────────
epochs = range(1, EPOCHS + 1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(epochs, train_losses, color='#0077BB', label='Train loss')
ax1.plot(epochs, val_losses,   color='#EE7733', label='Val loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('BCE Loss')
ax1.legend()
ax1.grid(True, alpha=0.3, linestyle='--')

ax2.plot(epochs, [v * 100 for v in val_ious],
         color='#0077BB', label='IoU %')
ax2.plot(epochs, [v * 100 for v in val_overlaps],
         color='#EE7733', label='Overlap %')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('%')
ax2.set_ylim(0, 105)
ax2.legend()
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
# plt.show()