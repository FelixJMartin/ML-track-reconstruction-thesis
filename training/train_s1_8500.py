# ================================================================================
# train_s1_8500.py
# Trains a small RadLIF SNN on the 8500-event dataset.
# Smaller network (h1=64, h2=32) to match data size and avoid overfitting.
# Input:  matrices8500/train/X_rz.npy
#         matrices8500/train/y_stage1.npy
# ================================================================================

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\SNN")

import torch
import torch.nn as nn
from snns import RadLIFLayer

import os
import numpy as np

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── CONFIG ────────────────────────────────────────────────────────────────────
H1         = 64
H2         = 32
EPOCHS     = 60
LR         = 1e-3
BATCH_SIZE = 32
DROPOUT    = 0.2
SNN_R_BINS = 100   # input feature size (100 cols per row)

train_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\train"
val_dir    = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\val"
model_path = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\model_s1_h64_32.pt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


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


# ── PARAM COUNT ───────────────────────────────────────────────────────────────
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train(X_train, y_train, X_val, y_val):
    model   = ConeFilterSNN(SNN_R_BINS, H1, H2, BATCH_SIZE).to(device)
    print(f"  Parameters: {count_params(model):,}")
    print(f"  Data/param ratio: {len(X_train)/count_params(model):.2f}x")

    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=10, factor=0.5)
    loss_fn = nn.BCEWithLogitsLoss()

    Xtr = torch.tensor(X_train).to(device)
    ytr = torch.tensor(y_train).to(device)
    Xv  = torch.tensor(X_val).to(device)
    yv  = torch.tensor(y_val).to(device)

    sig_mask   = yv == 1
    noise_mask = yv == 0

    print(f"  Train: {len(Xtr)}  signal={int(ytr.sum())}  noise={int((1-ytr).sum())}")
    print(f"  Val:   {len(Xv)}  signal={int(yv.sum())}  noise={int((1-yv).sum())}")

    train_losses, val_losses, val_accs = [], [], []
    val_effs, val_throws = [], []
    best_val  = float('inf')
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
            eff   = pred[sig_mask].mean().item()   if sig_mask.sum()   > 0 else 0.0
            throw = 1 - pred[noise_mask].mean().item() if noise_mask.sum() > 0 else 0.0

        train_losses.append(tl / nb)
        val_losses.append(vl)
        val_accs.append(acc)
        val_effs.append(eff)
        val_throws.append(throw)

        model.set_bs(BATCH_SIZE)
        sched.step(vl)

        if vl < best_val:
            best_val   = vl
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        print(f"  ep{ep+1:3d}/{EPOCHS}  loss={tl/nb:.4f}  val={vl:.4f}  "
              f"acc={acc*100:.1f}%  eff={eff*100:.1f}%  throw={throw*100:.1f}%  "
              f"best_val={best_val:.4f}", flush=True)

    torch.save({
        'model_state':  best_state,
        'train_losses': train_losses,
        'val_losses':   val_losses,
        'val_accs':     val_accs,
        'val_effs':     val_effs,
        'val_throws':   val_throws,
        'best_epoch':   val_losses.index(best_val) + 1,
        'config': {
            'H1': H1, 'H2': H2, 'LR': LR,
            'BATCH_SIZE': BATCH_SIZE, 'DROPOUT': DROPOUT,
            'SNN_R_BINS': SNN_R_BINS,
        }
    }, model_path)

    print(f"\n  Best val loss: {best_val:.4f} at epoch {val_losses.index(best_val)+1}")
    print(f"  Model saved to {model_path}")
    return train_losses, val_losses, val_accs, val_effs, val_throws


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Config: H1={H1}  H2={H2}  LR={LR}  BS={BATCH_SIZE}  DROPOUT={DROPOUT}  EPOCHS={EPOCHS}")
    print(f"Train:  {train_dir}")
    print(f"Val:    {val_dir}\n")

    if os.path.exists(model_path):
        print("Model already exists — loading results for plotting.")
        ckpt = torch.load(model_path)
        train_losses = ckpt['train_losses']
        val_losses   = ckpt['val_losses']
        val_accs     = ckpt['val_accs']
        val_effs     = ckpt['val_effs']
        val_throws   = ckpt['val_throws']
        print(f"  Best epoch: {ckpt['best_epoch']}  Best val: {min(val_losses):.4f}")
    else:
        print("Loading data...")
        X_train = np.load(os.path.join(train_dir, "X_rz.npy"))
        y_train = np.load(os.path.join(train_dir, "y_stage1.npy"))
        X_val   = np.load(os.path.join(val_dir,   "X_rz.npy"))
        y_val   = np.load(os.path.join(val_dir,   "y_stage1.npy"))
        print(f"  Train: {X_train.shape}  Val: {X_val.shape}")

        train_losses, val_losses, val_accs, val_effs, val_throws = train(
            X_train, y_train, X_val, y_val
        )

    # ── PLOT ──────────────────────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    try:
        import scienceplots
        plt.style.use(["science", "no-latex"])
    except ImportError:
        pass
    plt.rcParams.update({"figure.dpi": 150, "figure.facecolor": "white",
                          "axes.facecolor": "white"})

    best_ep = val_losses.index(min(val_losses)) + 1
    epochs  = range(1, len(train_losses) + 1)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    axes[0].plot(epochs, train_losses, label='train')
    axes[0].plot(epochs, val_losses,   label='val')
    axes[0].axvline(best_ep, color='gray', linestyle='--', alpha=0.5, label=f'best ep{best_ep}')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
    axes[0].set_title('Stage 1 Loss (8500 events, h=64/32)')
    axes[0].legend()

    axes[1].plot(epochs, [a * 100 for a in val_accs], color='#0077BB')
    axes[1].axvline(best_ep, color='gray', linestyle='--', alpha=0.5)
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy %')
    axes[1].set_title('Validation Accuracy')

    axes[2].plot(epochs, [e * 100 for e in val_effs],   label='Signal efficiency')
    axes[2].plot(epochs, [t * 100 for t in val_throws], label='Noise throwrate')
    axes[2].axvline(best_ep, color='gray', linestyle='--', alpha=0.5)
    axes[2].set_xlabel('Epoch'); axes[2].set_ylabel('%')
    axes[2].set_title('Efficiency & Throwrate')
    axes[2].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(os.path.dirname(model_path), "stage1_8500_results.png"), dpi=150)
    plt.show()
    print("Plot saved.")