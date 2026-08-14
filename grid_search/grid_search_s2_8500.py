# ================================================================================
# grid_search_s2_8500.py  — 8500 events, h=64/32, no stage 1 filter
# ================================================================================

import sys
sys.path.append("/proj/mixed-precision/mixed-precision/felix/SNN")
sys.path.append("/proj/mixed-precision/mixed-precision/felix/code")


import os
import numpy as np
import torch
import torch.nn as nn
from snns import RadLIFLayer

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── DEVICE ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── CONFIG ────────────────────────────────────────────────────────────────────
H1             = 64
H2             = 32
LR             = 1e-3
BATCH_SIZE     = 32
DROPOUT        = 0.2
EPOCHS         = 200
SNN_INPUT_SIZE = 101

TRAIN_DIR  = "/scratch/felixm/matrices8500/train"
VAL_DIR    = "/scratch/felixm/matrices8500/val"
MODEL_DIR  = "/proj/mixed-precision/mixed-precision/felix/models/stage2_8500"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_PATH = os.path.join(MODEL_DIR, "s2_h64_32_200ep.pt")

print(f"\nConfig: H1={H1} H2={H2} LR={LR} BS={BATCH_SIZE} "
      f"DROPOUT={DROPOUT} EPOCHS={EPOCHS}")
print(f"Train dir: {TRAIN_DIR}")
print(f"Output:    {MODEL_PATH}\n")

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

print(f"Train: {len(Xtr)} matrices  "
      f"signal={int((ytr.sum(axis=1)>0).sum())}  "
      f"noise={int((ytr.sum(axis=1)==0).sum())}")
print(f"Val:   {len(Xv)} matrices  "
      f"signal={int((yv.sum(axis=1)>0).sum())}  "
      f"noise={int((yv.sum(axis=1)==0).sum())}")

total_params = sum(p.numel() for p in PhiFinderSNN().parameters())
print(f"\nModel parameters: {total_params}")
print(f"Samples per parameter: {len(Xtr)/total_params:.2f}\n")

# ── SETUP ─────────────────────────────────────────────────────────────────────
model   = PhiFinderSNN(h1=H1, h2=H2, bs=BATCH_SIZE).to(device)
opt     = torch.optim.Adam(model.parameters(), lr=LR)
sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5,
                                                      factor=0.5, verbose=True)

n_pos      = int((ytr.sum(axis=1) > 0).sum())
n_neg      = int((ytr.sum(axis=1) == 0).sum())
pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)
loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
print(f"pos_weight=1.0  signal_matrices={n_pos}  noise_matrices={n_neg}\n")

Xtr = torch.tensor(Xtr, dtype=torch.float32).to(device)
ytr = torch.tensor(ytr, dtype=torch.float32).to(device)
Xv  = torch.tensor(Xv,  dtype=torch.float32).to(device)
yv  = torch.tensor(yv,  dtype=torch.float32).to(device)
yv_sig_mask = yv.sum(dim=1) > 0

# ── TRAIN ─────────────────────────────────────────────────────────────────────
train_losses, val_losses, val_ious, val_overlaps = [], [], [], []
best_val = float('inf')

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

    if vl < best_val:
        best_val = vl
        torch.save({
            'model_state':  model.state_dict(),
            'train_losses': train_losses,
            'val_losses':   val_losses,
            'val_ious':     val_ious,
            'val_overlaps': val_overlaps,
            'best_epoch':   ep + 1,
            'config': {
                'H1': H1, 'H2': H2, 'LR': LR,
                'BATCH_SIZE': BATCH_SIZE,
                'DROPOUT': DROPOUT,
            }
        }, MODEL_PATH)

    print(f"  ep{ep+1:3d}/{EPOCHS}  "
          f"loss={tl/nb:.4f}  val={vl:.4f}  "
          f"iou={iou*100:.1f}%  overlap={overlap*100:.1f}%  "
          f"best_val={best_val:.4f}", flush=True)

print(f"\nTraining complete. Best val: {best_val:.4f}")
print(f"Model saved to: {MODEL_PATH}")