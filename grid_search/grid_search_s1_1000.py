## ================================================================================
# grid_search_s1_1000.py  —  UPPMAX grid search version
# Saves best model weights + full 100-epoch loss history for thesis plots.
# ================================================================================

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
parser.add_argument('--h1',      type=int,   default=256)
parser.add_argument('--h2',      type=int,   default=128)
parser.add_argument('--lr',      type=float, default=3e-4)
parser.add_argument('--dropout', type=float, default=0.1)
parser.add_argument('--epochs',  type=int,   default=100)
parser.add_argument('--bs',      type=int,   default=32)
args = parser.parse_args()

EPOCHS     = args.epochs
LR         = args.lr
BATCH_SIZE = args.bs
H1         = args.h1
H2         = args.h2
DROPOUT    = args.dropout
SNN_R_BINS = 100

# ── PATHS ─────────────────────────────────────────────────────────────────────
DATA_DIR  = "/proj/uppmax2026-1-31/felix/data/matrices1000"
MODEL_DIR = "/proj/uppmax2026-1-31/felix/models/grid_s1"
os.makedirs(MODEL_DIR, exist_ok=True)

train_dir  = os.path.join(DATA_DIR, "train")
val_dir    = os.path.join(DATA_DIR, "val")

MODEL_NAME = f"s1_h{H1}_{H2}_lr{LR}_dr{DROPOUT}.pt"
MODEL_PATH = os.path.join(MODEL_DIR, MODEL_NAME)


# ── MODEL ─────────────────────────────────────────────────────────────────────
class ConeFilterSNN(nn.Module):
    def __init__(self, input_size, h1, h2, bs):
        super().__init__()
        self.lif1 = RadLIFLayer(input_size, h1, bs,
                                normalization='batchnorm', dropout=DROPOUT)
        self.lif2 = RadLIFLayer(h1, h2, bs,
                                normalization='batchnorm', dropout=DROPOUT)
        self.fc   = nn.Linear(h2, 1)

    def forward(self, x):
        return self.fc(self.lif2(self.lif1(x)).mean(dim=1)).squeeze(-1)

    def set_bs(self, bs):
        self.lif1.batch_size = bs
        self.lif2.batch_size = bs


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train(X_train, y_train, X_val, y_val):
    model   = ConeFilterSNN(SNN_R_BINS, H1, H2, BATCH_SIZE).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr = torch.tensor(X_train).to(device)
    ytr = torch.tensor(y_train).to(device)
    Xv  = torch.tensor(X_val).to(device)
    yv  = torch.tensor(y_val).to(device)

    sig_mask   = yv == 1
    noise_mask = yv == 0

    print(f"  Train: {len(Xtr)} matrices  "
          f"signal={int(ytr.sum())}  noise={int((1-ytr).sum())}")
    print(f"  Val:   {len(Xv)} matrices  "
          f"signal={int(yv.sum())}  noise={int((1-yv).sum())}")

    train_losses, val_losses, val_accs = [], [], []
    val_effs, val_throws = [], []

    best_val   = float('inf')
    best_state = None

    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(Xtr))
        Xtr_s, ytr_s = Xtr[idx], ytr[idx]

        tl, nb = 0.0, 0
        for b in range(0, len(Xtr_s), BATCH_SIZE):
            xb, yb = Xtr_s[b:b+BATCH_SIZE], ytr_s[b:b+BATCH_SIZE]
            model.set_bs(len(xb))
            loss = loss_fn(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item(); nb += 1

        model.eval()
        model.set_bs(len(Xv))
        with torch.no_grad():
            vp    = model(Xv)
            vl    = loss_fn(vp, yv).item()
            pred  = (vp > 0.0).float()
            acc   = (pred == yv).float().mean().item()
            eff   = pred[sig_mask].mean().item()   if sig_mask.sum() > 0 else 0.0
            throw = 1 - pred[noise_mask].mean().item() if noise_mask.sum() > 0 else 0.0

        train_losses.append(tl / nb)
        val_losses.append(vl)
        val_accs.append(acc)
        val_effs.append(eff)
        val_throws.append(throw)

        # track best weights
        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        model.set_bs(BATCH_SIZE)
        print(f"  ep{ep+1:3d}/{EPOCHS}  loss={tl/nb:.4f}  val={vl:.4f}  "
              f"acc={acc*100:.1f}%  eff={eff*100:.1f}%  "
              f"throw={throw*100:.1f}%", flush=True)

    # save best weights + full loss history
    torch.save({
        'model_state':  best_state,       # best weights
        'train_losses': train_losses,     # full 100 epochs
        'val_losses':   val_losses,       # full 100 epochs
        'val_accs':     val_accs,
        'val_effs':     val_effs,
        'val_throws':   val_throws,
        'best_epoch':   val_losses.index(best_val) + 1,
        'config': {
            'H1': H1, 'H2': H2, 'LR': LR,
            'BATCH_SIZE': BATCH_SIZE,
            'DROPOUT': DROPOUT,
            'SNN_R_BINS': SNN_R_BINS,
        }
    }, MODEL_PATH)
    print(f"\n  Best val loss: {best_val:.4f} "
          f"at epoch {val_losses.index(best_val)+1}")
    print(f"  Model saved to {MODEL_PATH}")


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Config: H1={H1}  H2={H2}  LR={LR}  BS={BATCH_SIZE}  DROPOUT={DROPOUT}")
    print(f"Model:  {MODEL_PATH}\n")

    if os.path.exists(MODEL_PATH):
        print("Already exists — skipping.")
    else:
        print("Loading data...")
        X_train = np.load(os.path.join(train_dir, "X_rz.npy"))
        y_train = np.load(os.path.join(train_dir, "y_stage1.npy"))
        X_val   = np.load(os.path.join(val_dir,   "X_rz.npy"))
        y_val   = np.load(os.path.join(val_dir,   "y_stage1.npy"))

        train(X_train, y_train, X_val, y_val)