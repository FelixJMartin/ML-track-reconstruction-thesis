# ================================================================================
# train_s2_100.py
# ================================================================================

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\SNN")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\build_datasets")

import os
import numpy as np
import torch
import torch.nn as nn
from snns import RadLIFLayer
from trackml.dataset import load_event
from Functions import preprocess_particles

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": False, "figure.dpi": 150,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from build_data100 import (make_rz_grid, make_rphi_grid, get_cone, CONE_SIGNS,
                           SNN_R_BINS, SNN_PHI_BINS, SNN_R_MAX,
                           SAMPLE_FRACTION, CONE_PHI_RANGES)
from train_s1_1000 import ConeFilterSNN

# ── CONFIG ────────────────────────────────────────────────────────────────────
EPOCHS         = 100
LR             = 1e-3
BATCH_SIZE     = 32
SNN_INPUT_SIZE = SNN_R_BINS + 1
THRESH_S1      = 0.5
THRESH_S2      = 0.6
data_dir       = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_100_events"
Save_dir       = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices100"

# Subdirectories produced by build_data100.py event-level split
train_dir = os.path.join(Save_dir, "train")
val_dir   = os.path.join(Save_dir, "val")


# ── MODEL ─────────────────────────────────────────────────────────────────────
class PhiFinderSNN(nn.Module):
    def __init__(self, h1=256, h2=128, bs=32):
        super().__init__()
        self.lif1 = RadLIFLayer(SNN_INPUT_SIZE, h1, bs,
                                normalization='batchnorm', dropout=0.1)
        self.lif2 = RadLIFLayer(h1, h2, bs,
                                normalization='batchnorm', dropout=0.1)
        self.fc   = nn.Linear(h2, 1)

    def forward(self, x):
        x = x.transpose(1, 2)
        return self.fc(self.lif2(self.lif1(x))).squeeze(-1)

    def set_bs(self, bs):
        self.lif1.batch_size = bs
        self.lif2.batch_size = bs


# ── TRAIN ─────────────────────────────────────────────────────────────────────
def train_stage2(X_train, y_train, X_val, y_val):
    """
    Train Stage 2 using a pre-split train/val from event-level splitting.
    No matrix-level train_test_split — that would cause data leakage.
    """
    model  = PhiFinderSNN(bs=BATCH_SIZE)
    opt    = torch.optim.Adam(model.parameters(), lr=LR)
    sched  = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    n_pos      = y_train.sum()
    n_neg      = y_train.size - n_pos
    pos_weight = torch.tensor([min(n_neg / n_pos, 10.0)], dtype=torch.float32)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  pos_weight={pos_weight.item():.1f}  "
          f"n_pos={int(n_pos)}  n_neg={int(n_neg)}")

    Xtr = torch.tensor(X_train)
    ytr = torch.tensor(y_train)
    Xv  = torch.tensor(X_val)
    yv  = torch.tensor(y_val)
    yv_sig_mask = yv.sum(dim=1) > 0

    train_losses, val_losses, val_ious, val_overlaps = [], [], [], []
    best = 0.0

    for ep in range(EPOCHS):
        model.train()
        idx  = torch.randperm(len(Xtr))
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
            vp  = model(Xv)
            vl  = loss_fn(vp, yv).item()
            iou = any_overlap = 0.0
            if yv_sig_mask.sum() > 0:
                pred_a = torch.sigmoid(vp[yv_sig_mask]) > 0.5
                true_a = yv[yv_sig_mask] > 0.5
                inter  = (pred_a & true_a).float().sum(dim=1)
                union  = (pred_a | true_a).float().sum(dim=1)
                iou         = (inter / union.clamp(min=1)).mean().item()
                any_overlap = (inter > 0).float().mean().item()

        train_losses.append(tl / nb)
        val_losses.append(vl)
        val_ious.append(iou)
        val_overlaps.append(any_overlap)
        model.set_bs(BATCH_SIZE)
        sched.step(vl)
        best = max(best, iou)
        print(f"  ep{ep+1:3d}/{EPOCHS}  loss={tl/nb:.4f}  val={vl:.4f}  "
              f"iou={iou*100:.1f}%  overlap={any_overlap*100:.1f}%  best={best*100:.1f}%")

    print(f"\n  best_iou={best*100:.1f}%\n")
    return model, train_losses, val_losses, val_ious, val_overlaps


# ── FULL PIPELINE SCAN ────────────────────────────────────────────────────────
def scan_event_full(event_path, model_s1, model_s2,
                    thresh_s1=0.5, thresh_s2=0.6, plot=True):
    h, _, p, t = load_event(event_path)
    h["r"]   = np.hypot(h["x"], h["y"])
    h["phi"] = np.arctan2(h["y"], h["x"])
    p_cond, _, _ = preprocess_particles(10, 10, p, h)
    backr = h[np.random.rand(len(h)) < SAMPLE_FRACTION].copy()

    model_s1.eval(); model_s1.set_bs(1)
    model_s2.eval(); model_s2.set_bs(1)

    results = []
    print(f"\n{'─'*70}")
    print(f"  Full pipeline: {os.path.basename(event_path)}")
    print(f"{'─'*70}")

    for sx, sy, sz in CONE_SIGNS:
        sig, combined = get_cone(h, t, p_cond, sx, sy, sz, backr)
        noi = combined[~combined['hit_id'].isin(sig['hit_id'])]

        all_r = np.concatenate([sig['r'].values, noi['r'].values])
        all_z = np.concatenate([sig['z'].values, noi['z'].values])
        with torch.no_grad():
            s1_prob = torch.sigmoid(
                model_s1(torch.tensor(make_rz_grid(all_r, all_z).T).unsqueeze(0))
            ).item()

        has_signal = len(sig) > 0
        s1_pass    = s1_prob > thresh_s1
        print(f"  cone({sx:+d},{sy:+d},{sz:+d})  s1={s1_prob*100:.1f}%  "
              f"{'✓' if s1_pass == has_signal else '✗'}  signal={has_signal}  ",
              end="")

        phi_probs, active_bins, true_bins = None, [], []

        if s1_pass and len(sig) > 0:
            phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
            all_rphi_r   = np.concatenate([sig['r'].values,   noi['r'].values])
            all_rphi_phi = np.concatenate([sig['phi'].values, noi['phi'].values])
            rphi_grid    = torch.tensor(
                make_rphi_grid(all_rphi_r, all_rphi_phi, sx, sy)
            ).unsqueeze(0)

            with torch.no_grad():
                phi_probs = torch.sigmoid(
                    model_s2(rphi_grid)
                ).squeeze(0).numpy()

            active_bins = np.where(phi_probs > thresh_s2)[0].tolist()
            phi_idx     = np.clip(
                ((sig['phi'].values - phi_min) / (phi_max - phi_min)
                 * SNN_PHI_BINS).astype(int),
                0, SNN_PHI_BINS - 1
            )
            true_bins = sorted(set(phi_idx.tolist()))
            overlap   = len(set(active_bins) & set(true_bins))
            print(f"s2_active={len(active_bins)}  true={len(true_bins)}  overlap={overlap}")
        else:
            print("(stage 2 skipped)")

        results.append(dict(
            cone=(sx, sy, sz), s1_prob=s1_prob, s1_pass=s1_pass,
            has_signal=has_signal, phi_probs=phi_probs,
            active_bins=active_bins, true_bins=true_bins
        ))

    s1_correct = sum(r['s1_pass'] == r['has_signal'] for r in results)
    print(f"\n  Stage 1 correct: {s1_correct}/8")

    if plot:
        plottable = [r for r in results
                     if r['s1_pass'] and r['phi_probs'] is not None]
        if plottable:
            n = len(plottable)
            fig, axes = plt.subplots(1, n,
                                     figsize=(n * 4.2, 3.8),
                                     squeeze=False)
            fig.subplots_adjust(wspace=0.12)

            for col, (ax, r) in enumerate(zip(axes[0], plottable)):
                ax.grid(axis='y', color='grey', alpha=0.2, zorder=0)
                bars = ax.bar(range(SNN_PHI_BINS), r['phi_probs'],
                              width=1, color='#cce0f5', linewidth=0, zorder=2)
                for b in r['active_bins']:
                    bars[b].set_color('#1a6bb5')
                for b in r['true_bins']:
                    ax.axvline(b, color='#c0392b', lw=1.5, alpha=0.85, zorder=3)
                ax.axhline(thresh_s2, color='gray', lw=0.7, ls='--', alpha=0.5, zorder=3)

                overlap = len(set(r['active_bins']) & set(r['true_bins']))
                info = (f"pred={len(r['active_bins'])}\n"
                        f"true={len(r['true_bins'])}\n"
                        f"overlap={overlap}")
                ax.text(0.03, 0.97, info,
                        transform=ax.transAxes, fontsize=7.5,
                        va='top', ha='left',
                        bbox=dict(boxstyle='round,pad=0.3', fc='white',
                                  ec='grey', lw=0.6, alpha=0.85))

                ax.set_xlabel('phi bin', fontsize=8)
                ax.set_ylabel('P(track)' if col == 0 else '', fontsize=8)
                ax.set_xlim(0, SNN_PHI_BINS)
                ax.set_ylim(0, 1.05)

            fig.savefig('stage2_phi_predictions.pdf',
                        bbox_inches='tight', dpi=300)
            plt.show()

    return results


# ── MAIN ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    # ── load event-level split data ───────────────────────────────────────────
    # These are produced by build_data100.py which splits by event ID.
    # Train and val matrices never share the same event — no leakage.
    print("Loading train data...")
    X_rphi_tr = np.load(os.path.join(train_dir, "X_rphi.npy"))
    y_s2_tr   = np.load(os.path.join(train_dir, "y_stage2.npy"))

    print("Loading val data...")
    X_rphi_v  = np.load(os.path.join(val_dir, "X_rphi.npy"))
    y_s2_v    = np.load(os.path.join(val_dir, "y_stage2.npy"))

    sig_rows = (y_s2_tr.sum(axis=1) > 0).sum()
    fp_rows  = (y_s2_tr.sum(axis=1) == 0).sum()
    print(f"\nStage 2 training: {len(X_rphi_tr)} total rows  "
          f"signal={sig_rows}  false_pos={fp_rows}")
    print(f"Stage 2 val:      {len(X_rphi_v)} total rows")

    # ── train or load stage 2 ─────────────────────────────────────────────────
    model_path = os.path.join(Save_dir, "model_stage2.pt")

    if os.path.exists(model_path):
        print("Loading stage 2 model...")
        checkpoint    = torch.load(model_path)
        trained_model = PhiFinderSNN(bs=BATCH_SIZE)
        trained_model.load_state_dict(checkpoint['model_state'])
        trained_model.eval()
    else:
        print("Training stage 2 model...")
        trained_model, train_losses, val_losses, val_ious, val_overlaps = \
            train_stage2(X_rphi_tr, y_s2_tr, X_rphi_v, y_s2_v)
        torch.save({'model_state': trained_model.state_dict()}, model_path)

    # ── load stage 1 and scan ─────────────────────────────────────────────────
    print("Loading stage 1 model...")
    model_s1 = ConeFilterSNN(SNN_R_BINS, bs=1)
    model_s1.load_state_dict(
        torch.load(os.path.join(Save_dir, "model_stage1.pt"))['model_state']
    )
    model_s1.eval()

    test_event = os.path.join(data_dir, "event000001080")
    scan_event_full(test_event, model_s1, trained_model,
                    thresh_s1=THRESH_S1, thresh_s2=THRESH_S2, plot=True)