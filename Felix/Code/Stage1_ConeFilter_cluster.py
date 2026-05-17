import sys
sys.path.append("/proj/uppmax2026-1-31/felix/SNN")
sys.path.append("/proj/uppmax2026-1-31/felix/code")

import os
import argparse
import numpy as np
import torch
import torch.nn as nn
from snns import RadLIFLayer
from sklearn.model_selection import train_test_split

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── DEVICE ────────────────────────────────────────────────────────────────────
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")

# ── ARGS ──────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser()
parser.add_argument('--h1',       type=int,   default=512)
parser.add_argument('--h2',       type=int,   default=256)
parser.add_argument('--lr',       type=float, default=1e-3)
parser.add_argument('--dropout',  type=float, default=0.1)
parser.add_argument('--epochs',   type=int,   default=100)
parser.add_argument('--bs',       type=int,   default=32)
parser.add_argument('--noise',    type=int,   default=1,
                    help="Noise level in percent: 1, 2, 5, or 10")
args = parser.parse_args()

EPOCHS     = args.epochs
LR         = args.lr
BATCH_SIZE = args.bs
H1         = args.h1
H2         = args.h2
DROPOUT    = args.dropout
NOISE_PCT  = args.noise

SNN_R_BINS = 150  # updated from 100

# ── PATHS ─────────────────────────────────────────────────────────────────────
DATA_DIR  = f"/scratch/uppmax2026-1-31/felix/data/matrices_150_{NOISE_PCT}pct"
MODEL_DIR = f"/proj/uppmax2026-1-31/felix/models/robust_s1"
os.makedirs(MODEL_DIR, exist_ok=True)

MODEL_NAME = f"s1_h{H1}_{H2}_lr{LR}_dr{DROPOUT}_noise{NOISE_PCT}pct.pt"
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
def train(X_data, y_data):
    Xtr, Xv, ytr, yv = train_test_split(X_data, y_data, test_size=0.2,
                                         stratify=y_data, random_state=42)

    model   = ConeFilterSNN(SNN_R_BINS, H1, H2, BATCH_SIZE).to(device)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr = torch.tensor(Xtr).to(device)
    ytr = torch.tensor(ytr).to(device)
    Xv  = torch.tensor(Xv).to(device)
    yv  = torch.tensor(yv).to(device)

    sig_mask   = yv == 1
    noise_mask = yv == 0

    train_losses, val_losses, val_accs = [], [], []
    val_effs, val_throws = [], []
    best_acc = 0.0

    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(Xtr))
        Xtr, ytr = Xtr[idx], ytr[idx]

        tl, nb = 0.0, 0
        for b in range(0, len(Xtr), BATCH_SIZE):
            xb, yb = Xtr[b:b+BATCH_SIZE], ytr[b:b+BATCH_SIZE]
            model.set_bs(len(xb))
            loss = loss_fn(model(xb), yb)
            opt.zero_grad(); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()
            tl += loss.item(); nb += 1

        model.eval()
        model.set_bs(len(Xv))
        with torch.no_grad():
            vp   = model(Xv)
            vl   = loss_fn(vp, yv).item()
            pred = (vp > 0.0).float()
            acc  = (pred == yv).float().mean().item()
            eff  = pred[sig_mask].mean().item()
            throw= (1 - pred[noise_mask].mean().item())

        train_losses.append(tl / nb)
        val_losses.append(vl)
        val_accs.append(acc)
        val_effs.append(eff)
        val_throws.append(throw)

        model.set_bs(BATCH_SIZE)
        sched.step(vl)
        best_acc = max(best_acc, acc)
        print(f"  ep{ep+1:3d}/{EPOCHS}  loss={tl/nb:.4f}  val={vl:.4f}  "
              f"acc={acc*100:.1f}%  eff={eff*100:.1f}%  throw={throw*100:.1f}%  "
              f"best={best_acc*100:.1f}%", flush=True)

    print(f"\n  best_acc={best_acc*100:.1f}%\n")
    return model, train_losses, val_losses, val_accs, val_effs, val_throws


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Config: H1={H1}  H2={H2}  LR={LR}  BS={BATCH_SIZE}  DROPOUT={DROPOUT}  NOISE={NOISE_PCT}%")
    print(f"Data:   {DATA_DIR}")
    print(f"Model:  {MODEL_PATH}\n")

    if os.path.exists(MODEL_PATH):
        print("Already exists — skipping.")
    else:
        print("Loading data...")
        X_data = np.load(os.path.join(DATA_DIR, "X_rz.npy"))
        y_data = np.load(os.path.join(DATA_DIR, "y_stage1.npy"))
        print(f"  {len(X_data)} matrices  "
              f"signal={int(y_data.sum())}  noise={int((1-y_data).sum())}")

        trained_model, train_losses, val_losses, val_accs, val_effs, val_throws = \
            train(X_data, y_data)

        torch.save({
            'model_state':  trained_model.state_dict(),
            'train_losses': train_losses,
            'val_losses':   val_losses,
            'val_accs':     val_accs,
            'val_effs':     val_effs,
            'val_throws':   val_throws,
            'config': {
                'H1': H1, 'H2': H2, 'LR': LR,
                'BATCH_SIZE': BATCH_SIZE, 'DROPOUT': DROPOUT,
                'NOISE_PCT': NOISE_PCT, 'SNN_R_BINS': SNN_R_BINS
            }
        }, MODEL_PATH)
        print(f"Saved: {MODEL_PATH}")