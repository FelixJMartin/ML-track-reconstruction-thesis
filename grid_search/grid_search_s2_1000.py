import sys
sys.path.append("/proj/uppmax2026-1-31/felix/SNN")
sys.path.append("/proj/uppmax2026-1-31/felix/code")

import os
import argparse
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

# ── ARGS ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--h1',         type=int,   default=512)
parser.add_argument('--h2',         type=int,   default=256)
parser.add_argument('--lr',         type=float, default=1e-3)
parser.add_argument('--dropout',    type=float, default=0.1)
parser.add_argument('--pos_weight', type=float, default=10.0)
parser.add_argument('--epochs',     type=int,   default=100)
parser.add_argument('--bs',         type=int,   default=32)
args = parser.parse_args()

EPOCHS         = args.epochs
LR             = args.lr
BATCH_SIZE     = args.bs
H1             = args.h1
H2             = args.h2
DROPOUT        = args.dropout
POS_WEIGHT_CAP = args.pos_weight
SNN_INPUT_SIZE = 101   # 100 r-bins + 1 occupancy row

# ── PATHS ─────────────────────────────────────────────────────────────────────
TRAIN_DIR  = "/proj/uppmax2026-1-31/felix/data/matrices1000/train2"
VAL_DIR    = "/proj/uppmax2026-1-31/felix/data/matrices1000/val2"
MODEL_DIR  = "/proj/uppmax2026-1-31/felix/models/grid_s2"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = f"s2_h{H1}_{H2}_lr{LR}_dr{DROPOUT}_pw{POS_WEIGHT_CAP}.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)

if os.path.exists(MODEL_PATH):
    print(f"Already exists: {MODEL_PATH} — skipping")
    sys.exit(0)

print(f"\nConfig: H1={H1} H2={H2} LR={LR} BS={BATCH_SIZE} "
      f"DROPOUT={DROPOUT} POS_WEIGHT={POS_WEIGHT_CAP}")
print(f"Output: {MODEL_PATH}\n")

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


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train():
    print("Loading Stage 2 filtered data...")
    Xtr = np.load(os.path.join(TRAIN_DIR, "X_rphi.npy"))
    ytr = np.load(os.path.join(TRAIN_DIR, "y_stage2.npy"))
    Xv  = np.load(os.path.join(VAL_DIR,   "X_rphi.npy"))
    yv  = np.load(os.path.join(VAL_DIR,   "y_stage2.npy"))

    print(f"Train: {len(Xtr)} cones  "
          f"signal={int((ytr.sum(axis=1)>0).sum())}  "
          f"false_pos={int((ytr.sum(axis=1)==0).sum())}")
    print(f"Val:   {len(Xv)} cones  "
          f"signal={int((yv.sum(axis=1)>0).sum())}  "
          f"false_pos={int((yv.sum(axis=1)==0).sum())}\n")

    model = PhiFinderSNN(h1=H1, h2=H2, bs=BATCH_SIZE).to(device)
    opt   = torch.optim.Adam(model.parameters(), lr=LR)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5,
                                                        factor=0.5)

    n_pos      = ytr.sum()
    n_neg      = ytr.size - n_pos
    true_ratio = n_neg / max(n_pos, 1)
    pw         = min(true_ratio, POS_WEIGHT_CAP)
    loss_fn    = nn.BCEWithLogitsLoss(
                     pos_weight=torch.tensor([pw], dtype=torch.float32).to(device))
    print(f"pos_weight={pw:.1f}  n_pos={int(n_pos)}  n_neg={int(n_neg)}\n")

    Xtr = torch.tensor(Xtr, dtype=torch.float32).to(device)
    ytr = torch.tensor(ytr, dtype=torch.float32).to(device)
    Xv  = torch.tensor(Xv,  dtype=torch.float32).to(device)
    yv  = torch.tensor(yv,  dtype=torch.float32).to(device)
    yv_sig_mask = yv.sum(dim=1) > 0

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
                pred_a  = torch.sigmoid(vp[yv_sig_mask]) > 0.5
                true_a  = yv[yv_sig_mask] > 0.5
                inter   = (pred_a & true_a).float().sum(dim=1)
                union   = (pred_a | true_a).float().sum(dim=1)
                iou     = (inter / union.clamp(min=1)).mean().item()
                overlap = (inter > 0).float().mean().item()

        train_losses.append(tl / nb)
        val_losses.append(vl)
        val_ious.append(iou)
        val_overlaps.append(overlap)
        model.set_bs(BATCH_SIZE)
        sched.step(vl)

        if iou > best_iou:
            best_iou = iou
            torch.save({
                'model_state':  model.state_dict(),
                'train_losses': train_losses,
                'val_losses':   val_losses,
                'val_ious':     val_ious,
                'val_overlaps': val_overlaps,
                'config': {
                    'H1': H1, 'H2': H2, 'LR': LR,
                    'BATCH_SIZE': BATCH_SIZE,
                    'DROPOUT': DROPOUT,
                    'POS_WEIGHT_CAP': POS_WEIGHT_CAP,
                }
            }, MODEL_PATH)

        print(f"  ep{ep+1:3d}/{EPOCHS}  "
              f"loss={tl/nb:.4f}  val={vl:.4f}  "
              f"iou={iou*100:.1f}%  overlap={overlap*100:.1f}%  "
              f"best={best_iou*100:.1f}%", flush=True)

    print(f"\nBest IoU: {best_iou*100:.1f}%")
    print(f"Saved: {MODEL_PATH}")


if __name__ == "__main__":
    train()