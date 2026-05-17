# ================================================================================
# stage1_cone_filter.py
# Trains a RadLIF SNN to classify cones as signal (contains track) or noise.
# Input:  X_rz.npy      — r-z matrices (100x100)
#         y_stage1.npy  — binary labels (0/1)
# Output: model_stage1.pt
# ================================================================================

from trackml.dataset import load_event

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Felix\SNN")
import torch
import torch.nn as nn
from snns import RadLIFLayer
from sklearn.model_selection import train_test_split

import os
import numpy as np
from Functions import preprocess_particles

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── CONFIG ────────────────────────────────────────────────────────────────────
EPOCHS     = 60
LR         = 1e-3
BATCH_SIZE = 32

data_dir = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_100_events"
Save_dir = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices100"

from build_data100 import (make_rz_grid, get_cone, CONE_SIGNS, SAMPLE_FRACTION,
                           SNN_R_BINS, SNN_Z_BINS, SNN_R_MAX, SNN_Z_MAX)


# ── MODEL ─────────────────────────────────────────────────────────────────────
class ConeFilterSNN(nn.Module):
    def __init__(self, input_size, h1=256, h2=128, bs=32):
        super().__init__()
        self.lif1 = RadLIFLayer(input_size, h1, bs, normalization='batchnorm', dropout=0.1)
        self.lif2 = RadLIFLayer(h1, h2,     bs, normalization='batchnorm', dropout=0.1)
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

    model   = ConeFilterSNN(SNN_R_BINS, bs=BATCH_SIZE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr, ytr, Xv, yv = map(torch.tensor, [Xtr, ytr, Xv, yv])

    train_losses, val_losses, val_accs = [], [], []
    best = 0.0

    for ep in range(EPOCHS):
        model.train()
        idx = torch.randperm(len(Xtr))
        Xtr, ytr = Xtr[idx], ytr[idx]

        tl, nb = 0.0, 0
        for b in range(0, len(Xtr), BATCH_SIZE):
            xb, yb = Xtr[b:b+BATCH_SIZE], ytr[b:b+BATCH_SIZE]
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
            vp  = model(Xv)
            vl  = loss_fn(vp, yv).item()
            acc = ((vp > 0.0).float() == yv).float().mean().item()

        train_losses.append(tl / nb)
        val_losses.append(vl)
        val_accs.append(acc)

        model.set_bs(BATCH_SIZE)
        sched.step(vl)
        best = max(best, acc)
        print(f"  ep{ep+1:3d}/{EPOCHS}  loss={tl/nb:.4f}  val={vl:.4f}  "
              f"acc={acc*100:.1f}%  best={best*100:.1f}%")

    print(f"\n  best_acc={best*100:.1f}%\n")
    return model, train_losses, val_losses, val_accs


# ── SCAN EVENT ────────────────────────────────────────────────────────────────
def scan_event(event_path, model, threshold=0.5):
    h, _, p, t = load_event(event_path)
    h["r"]   = np.hypot(h["x"], h["y"])
    h["phi"] = np.arctan2(h["y"], h["x"])
    p_cond, _, _ = preprocess_particles(10, 10, p, h)
    backr = h[np.random.rand(len(h)) < SAMPLE_FRACTION].copy()

    model.eval()
    model.set_bs(1)
    correct = 0

    print(f"\n  Scanning {os.path.basename(event_path)}")
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
        ok = (prob > threshold) == has_signal
        correct += ok
        print(f"  cone({sx:+d},{sy:+d},{sz:+d})  conf={prob*100:.1f}%  "
              f"signal={has_signal}  {'✓' if ok else '✗'}")

    print(f"\n  Correct: {correct}/8")


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading training data...")
    X_data = np.load(os.path.join(Save_dir, "X_rz.npy"))
    y_data = np.load(os.path.join(Save_dir, "y_stage1.npy"))
    print(f"  {len(X_data)} matrices  signal={int(y_data.sum())}  noise={int((1-y_data).sum())}")

    model_path = os.path.join(Save_dir, "model_stage1.pt")
    if os.path.exists(model_path):
        print("Loading trained model from disk...")
        checkpoint    = torch.load(model_path)
        trained_model = ConeFilterSNN(SNN_R_BINS, bs=BATCH_SIZE)
        trained_model.load_state_dict(checkpoint['model_state'])
        trained_model.eval()
        print("Model loaded.")
    else:
        print("Training stage 1 model...")
        trained_model, train_losses, val_losses, val_accs = train(X_data, y_data)
        torch.save({
            'model_state':  trained_model.state_dict(),
            'train_losses': train_losses,
            'val_losses':   val_losses,
            'val_accs':     val_accs,
        }, model_path)
        print(f"Model saved to {model_path}")

        import matplotlib.pyplot as plt
        import scienceplots
        plt.style.use(["science", "no-latex"])
        plt.rcParams.update({"text.usetex": False, "figure.dpi": 150,
                             "figure.facecolor": "white", "axes.facecolor": "white"})
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(train_losses, label='train')
        ax1.plot(val_losses,   label='val')
        ax1.set_xlabel('epoch'); ax1.set_ylabel('BCE loss')
        ax1.set_title('Stage 1 loss'); ax1.legend()
        ax2.plot([a * 100 for a in val_accs], color='#0077BB')
        ax2.set_xlabel('epoch'); ax2.set_ylabel('accuracy %')
        ax2.set_title('Stage 1 validation accuracy')
        plt.tight_layout()
        plt.show()

    test_event = os.path.join(data_dir, "event000001080")
    scan_event(test_event, trained_model, threshold=0.5)