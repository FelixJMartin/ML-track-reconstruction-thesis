# ================================================================================
# stage2_phi_localiser.py
# ================================================================================

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Felix\SNN")

import os
import numpy as np
import torch
import torch.nn as nn
from snns import RadLIFLayer
from sklearn.model_selection import train_test_split
from trackml.dataset import load_event
from Functions import preprocess_particles

import matplotlib.pyplot as plt
plt.rcParams.update({
    "text.usetex": False, "figure.dpi": 150,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
})

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from build_data10 import (make_rz_grid, make_rphi_grid, get_cone, CONE_SIGNS,
                           SNN_R_BINS, SNN_PHI_BINS, SNN_R_MAX,
                           SAMPLE_FRACTION, CONE_PHI_RANGES)
from Stage1_ConeFilter10 import ConeFilterSNN

# ── CONFIG ────────────────────────────────────────────────────────────────────
EPOCHS         = 60
LR             = 1e-3
BATCH_SIZE     = 32
SNN_INPUT_SIZE = SNN_R_BINS
THRESH_S1      = 0.5
THRESH_S2      = 0.6
data_dir       = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_100_events"
Save_dir       = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices10"


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
def train_stage2(X, y):
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42)

    model   = PhiFinderSNN(bs=BATCH_SIZE)
    opt     = torch.optim.Adam(model.parameters(), lr=LR)
    sched   = torch.optim.lr_scheduler.ReduceLROnPlateau(opt, patience=5, factor=0.5)

    n_pos      = ytr.sum()
    n_neg      = ytr.size - n_pos
    pos_weight = torch.tensor([min(n_neg / n_pos, 100)], dtype=torch.float32)
    loss_fn    = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    print(f"  pos_weight={pos_weight.item():.1f}  "
          f"n_pos={int(n_pos)}  n_neg={int(n_neg)}")

    Xtr, ytr    = torch.tensor(Xtr), torch.tensor(ytr)
    Xv,  yv     = torch.tensor(Xv),  torch.tensor(yv)
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
                sx, sy, sz = r['cone']

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

    # ── load data ─────────────────────────────────────────────────────────────
    # X_rphi has no augmentation (1 entry per track); X_rz has N_AUGMENT entries
    # per track — different sizes, so no stage-1 index filter here.
    X_rphi = np.load(os.path.join(Save_dir, "X_rphi.npy"))
    y_s2   = np.load(os.path.join(Save_dir, "y_stage2.npy"))

    sig_rows = (y_s2.sum(axis=1) > 0).sum()
    fp_rows  = (y_s2.sum(axis=1) == 0).sum()
    print(f"Stage 2 dataset: {len(X_rphi)} total rows  "
          f"signal={sig_rows}  noise={fp_rows}")

    X_rphi_s2_train = X_rphi
    y_s2_train      = y_s2

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
            train_stage2(X_rphi_s2_train, y_s2_train)
        torch.save({
            'model_state':  trained_model.state_dict(),
            'train_losses': train_losses,
            'val_losses':   val_losses,
            'val_ious':     val_ious,
            'val_overlaps': val_overlaps,
        }, model_path)
        print(f"Model saved to {model_path}")

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(train_losses, label='train')
        ax1.plot(val_losses,   label='val')
        ax1.set_xlabel('epoch'); ax1.set_ylabel('BCE loss')
        ax1.set_title('Stage 2 loss'); ax1.legend()
        ax2.plot([v * 100 for v in val_ious],     label='IoU %',     color='#0077BB')
        ax2.plot([v * 100 for v in val_overlaps], label='Overlap %', color='#EE7733')
        ax2.set_xlabel('epoch'); ax2.set_ylabel('%')
        ax2.set_title('Stage 2 metrics'); ax2.legend()
        plt.tight_layout()
        plt.show()

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