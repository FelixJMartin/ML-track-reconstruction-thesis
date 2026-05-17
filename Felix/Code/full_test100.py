# ================================================================================
# pipeline_eval.py  — 100-event set, fixed baseline + alpha=0.8
# ================================================================================

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Felix\SNN")

import os
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D

from trackml.dataset import load_event
from Functions import preprocess_particles
from build_data100 import (
    make_rz_grid, make_rphi_grid, get_cone, CONE_SIGNS,
    SNN_R_BINS, SNN_PHI_BINS, SNN_R_MAX, SNN_Z_MAX,
    SAMPLE_FRACTION, CONE_PHI_RANGES
)
from Stage1_ConeFilter100 import ConeFilterSNN
from Stage2_PhiFilter100 import PhiFinderSNN

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.rcParams.update({
    "text.usetex": False, "figure.dpi": 150,
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 10,
})
C = ["#0077BB", "#EE7733", "#009988", "#CC3311"]

# ── CONFIG ────────────────────────────────────────────────────────────────────
THRESH_S1  = 0.5
THRESH_S2  = 0.6
MIN_PT     = 10
MIN_NHITS  = 10

data_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_100_events"
Save_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices100"
test_ids  = [f"event{i:09d}" for i in range(1080, 1100)]
VIS_EVENT = "event000001080"

# ── LOAD MODELS ───────────────────────────────────────────────────────────────
print("Loading models...")
model_s1 = ConeFilterSNN(SNN_R_BINS, bs=1)
model_s1.load_state_dict(
    torch.load(os.path.join(Save_dir, "model_stage1.pt"))["model_state"]
)
model_s1.eval(); model_s1.set_bs(1)

model_s2 = PhiFinderSNN(bs=1)
model_s2.load_state_dict(
    torch.load(os.path.join(Save_dir, "model_stage2.pt"))["model_state"]
)
model_s2.eval(); model_s2.set_bs(1)

# ── ACCUMULATORS ──────────────────────────────────────────────────────────────
hits_baseline_all  = []   # hits entering the pipeline (post-subsampling + all signal)
hits_after_s1_all  = []
hits_after_s2_all  = []
s1_tp, s1_fn, s1_tn, s1_fp = 0, 0, 0, 0
s2_detected, s2_total_sig   = 0, 0
s2_total_pred, s2_total_correct = 0, 0
tracks_total, tracks_survived   = 0, 0

vis_hits_before   = None
vis_hits_after_s1 = None
vis_hits_after_s2 = None

print(f"\nEvaluating {len(test_ids)} events...")
print(f"{'─'*70}")

for event_id in test_ids:
    try:
        h, _, p, t = load_event(os.path.join(data_dir, event_id))
        h["r"]   = np.hypot(h["x"], h["y"])
        h["phi"] = np.arctan2(h["y"], h["x"])

        p_cond, _, _ = preprocess_particles(MIN_PT, MIN_NHITS, p, h)
        good_pids     = set(p_cond["particle_id"].values)

        backr = h[np.random.rand(len(h)) < SAMPLE_FRACTION].copy()

        merged = h.merge(t[["hit_id", "particle_id"]], on="hit_id")

        # ── FIXED BASELINE: hits that actually enter the pipeline ─────────────
        sig_hit_ids      = set(merged[merged["particle_id"].isin(good_pids)]["hit_id"].values)
        pipeline_hit_ids = sig_hit_ids | set(backr["hit_id"].values)
        hits_baseline    = len(pipeline_hit_ids)
        hits_baseline_all.append(hits_baseline)
        # ──────────────────────────────────────────────────────────────────────

        pids_in_event    = set()
        pids_survived_s2 = set()
        hits_after_s1    = 0
        hits_after_s2    = 0

        surviving_hit_ids_s1 = set()
        surviving_hit_ids_s2 = set()

        for sx, sy, sz in CONE_SIGNS:
            sig, combined = get_cone(h, t, p_cond, sx, sy, sz, backr)
            noi = combined[~combined['hit_id'].isin(sig['hit_id'])]
            has_signal = len(sig) > 0

            if has_signal:
                for pid in sig['particle_id'].unique():
                    pids_in_event.add(pid)

            all_r = np.concatenate([sig["r"].values, noi["r"].values])
            all_z = np.concatenate([sig["z"].values, noi["z"].values])
            with torch.no_grad():
                s1_prob = torch.sigmoid(
                    model_s1(torch.tensor(make_rz_grid(all_r, all_z).T).unsqueeze(0))
                ).item()

            s1_pass = s1_prob > THRESH_S1

            if has_signal and s1_pass:         s1_tp += 1
            if has_signal and not s1_pass:     s1_fn += 1
            if not has_signal and not s1_pass: s1_tn += 1
            if not has_signal and s1_pass:     s1_fp += 1

            if s1_pass:
                hits_after_s1 += len(combined)
                surviving_hit_ids_s1.update(combined['hit_id'].values)

            if s1_pass and len(sig) > 0:
                phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
                rphi_r   = np.concatenate([sig["r"].values,   noi["r"].values])
                rphi_phi = np.concatenate([sig["phi"].values, noi["phi"].values])
                rphi_grid = torch.tensor(
                    make_rphi_grid(rphi_r, rphi_phi, sx, sy)
                ).unsqueeze(0)

                with torch.no_grad():
                    phi_probs = torch.sigmoid(
                        model_s2(rphi_grid)
                    ).squeeze(0).numpy()

                active_bins = np.where(phi_probs > THRESH_S2)[0]
                phi_idx = np.clip(
                    ((sig["phi"].values - phi_min) / (phi_max - phi_min)
                     * SNN_PHI_BINS).astype(int),
                    0, SNN_PHI_BINS - 1
                )
                true_bins  = set(phi_idx.tolist())
                active_set = set(active_bins.tolist())
                overlap    = len(active_set & true_bins)

                s2_total_sig     += 1
                s2_total_pred    += len(active_bins)
                s2_total_correct += overlap

                if overlap > 0:
                    s2_detected += 1
                    for pid in sig['particle_id'].unique():
                        pids_survived_s2.add(pid)

                for b in active_bins:
                    phi_lo = phi_min + b       * (phi_max - phi_min) / SNN_PHI_BINS
                    phi_hi = phi_min + (b + 1) * (phi_max - phi_min) / SNN_PHI_BINS
                    in_bin = combined[
                        (combined['phi'] >= phi_lo) &
                        (combined['phi'] <  phi_hi)
                    ]
                    hits_after_s2 += len(in_bin)
                    surviving_hit_ids_s2.update(in_bin['hit_id'].values)

        hits_after_s1_all.append(hits_after_s1)
        hits_after_s2_all.append(hits_after_s2)
        tracks_total    += len(pids_in_event)
        tracks_survived += len(pids_survived_s2)

        if event_id == VIS_EVENT:
            vis_hits_before   = h[h['hit_id'].isin(pipeline_hit_ids)].copy()
            vis_hits_after_s1 = h[h['hit_id'].isin(surviving_hit_ids_s1)].copy()
            vis_hits_after_s2 = h[h['hit_id'].isin(surviving_hit_ids_s2)].copy()

        pct_s1 = 100 * (1 - hits_after_s1 / max(hits_baseline, 1))
        pct_s2 = 100 * (1 - hits_after_s2 / max(hits_baseline, 1))
        print(f"  {event_id}: {hits_baseline:,} → {hits_after_s1:,} → {hits_after_s2:,}  "
              f"({pct_s1:.0f}%/{pct_s2:.0f}% removed)  "
              f"tracks: {len(pids_survived_s2)}/{len(pids_in_event)}")

    except Exception as e:
        print(f"  {event_id}: skipped — {e}")

# ── AGGREGATE STATS ───────────────────────────────────────────────────────────
hits_baseline_all  = np.array(hits_baseline_all)
hits_after_s1_all  = np.array(hits_after_s1_all)
hits_after_s2_all  = np.array(hits_after_s2_all)

s1_eff      = s1_tp / (s1_tp + s1_fn) if (s1_tp + s1_fn) > 0 else 0
s1_throw    = s1_tn / (s1_tn + s1_fp) if (s1_tn + s1_fp) > 0 else 0
s2_det_rate = s2_detected / s2_total_sig if s2_total_sig > 0 else 0
s2_prec     = s2_total_correct / s2_total_pred if s2_total_pred > 0 else 0
full_eff    = tracks_survived / tracks_total if tracks_total > 0 else 0

print(f"\n{'═'*60}")
print(f"  PIPELINE SUMMARY — {len(test_ids)} events")
print(f"{'═'*60}")
print(f"  Hit reduction (baseline = post-subsampling input to pipeline):")
print(f"    Baseline : {hits_baseline_all.mean():>10,.0f} hits/event  (signal + {SAMPLE_FRACTION*100:.0f}% bg sample)")
print(f"    Stage 1  : {hits_after_s1_all.mean():>10,.0f} hits/event  "
      f"({100*(1-hits_after_s1_all.mean()/hits_baseline_all.mean()):.1f}% removed)")
print(f"    Stage 2  : {hits_after_s2_all.mean():>10,.0f} hits/event  "
      f"({100*(1-hits_after_s2_all.mean()/hits_baseline_all.mean()):.1f}% removed)")
print(f"  Stage 1: efficiency={s1_eff*100:.1f}%  throwrate={s1_throw*100:.1f}%")
print(f"  Stage 2: detection={s2_det_rate*100:.1f}%  precision={s2_prec*100:.1f}%")
print(f"  Track survival: {full_eff*100:.1f}%  ({tracks_survived}/{tracks_total})")
print(f"{'═'*60}")







# ── PLOT 1: pipeline summary ───────────────────────────────────────────────────
fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(14, 5))
fig.subplots_adjust(wspace=0.35)

x = np.arange(len(hits_baseline_all))
ax0.bar(x - 0.25, hits_baseline_all / 1000, width=0.25, color=C[0], alpha=0.8, label='Pipeline input')
ax0.bar(x,        hits_after_s1_all / 1000, width=0.25, color=C[1], alpha=0.8, label='After Stage 1')
ax0.bar(x + 0.25, hits_after_s2_all / 1000, width=0.25, color=C[2], alpha=0.8, label='After Stage 2')
ax0.set_xlabel('Event index')
ax0.set_ylabel('Hits (thousands)')
ax0.legend(fontsize=8)
ax0.set_xticks(x)
ax0.set_xticklabels([str(i) for i in range(len(hits_baseline_all))], fontsize=7)

metrics = ['S1 efficiency', 'S1 throwrate', 'S2 detection', 'S2 precision', 'Track survival']
values  = [s1_eff*100, s1_throw*100, s2_det_rate*100, s2_prec*100, full_eff*100]
colors  = [C[1], C[0], C[1], C[0], C[2]]
bars    = ax1.barh(metrics, values, color=colors, alpha=0.8)
ax1.set_xlim(0, 105)
ax1.set_xlabel('%')
for bar, val in zip(bars, values):
    ax1.text(val + 1, bar.get_y() + bar.get_height() / 2,
             f'{val:.1f}%', va='center', fontsize=9)

plt.tight_layout()
plt.show()

# ── PLOT 2: 3D before/after visualisation ─────────────────────────────────────
def plot_3d_panel(ax, hits, annotation, s=1):
    if 'volume_id' in hits.columns:
        volumes = sorted(hits.volume_id.unique())
        palette = sns.color_palette("rainbow", len(volumes))
        for i, volume in enumerate(volumes):
            v = hits[hits.volume_id == volume]
            ax.scatter(v.z, v.x, v.y, s=s, alpha=0.8, color=palette[i])
    else:
        ax.scatter(hits.z, hits.x, hits.y, s=s, alpha=0.8, color=C[0])
    ax.set_xlim(-3000, 3000)
    ax.set_ylim(-1200, 1200)
    ax.set_zlim(-1200, 1200)
    ax.tick_params(labelsize=6, colors=(0, 0, 0, 0.4))
    ax.view_init(elev=20, azim=130)

    # ── fade pane walls and grid ──────────────────────────────────────────────
    for pane in [ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane]:
        pane.fill = False
        pane.set_edgecolor((0, 0, 0, 0.15))
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis._axinfo['grid']['color'] = (0, 0, 0, 0.12)
        axis.line.set_color((0, 0, 0, 0.15))
        axis._axinfo['tick']['color'] = (0, 0, 0, 0.15)

    ax.xaxis.label.set_alpha(0.4)
    ax.yaxis.label.set_alpha(0.4)
    ax.zaxis.label.set_alpha(0.4)

    # ── annotation box ────────────────────────────────────────────────────────
    ax.text2D(0.03, 0.05, annotation,
              transform=ax.transAxes, fontsize=8,
              verticalalignment='bottom',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        edgecolor='grey', alpha=0.7))

if vis_hits_before is not None:
    pct_s1 = 100 * (1 - len(vis_hits_after_s1) / len(vis_hits_before))
    pct_s2 = 100 * (1 - len(vis_hits_after_s2) / len(vis_hits_before))

    fig3d = plt.figure(figsize=(14, 4))
    ax_b  = fig3d.add_subplot(131, projection='3d')
    ax_s1 = fig3d.add_subplot(132, projection='3d')
    ax_s2 = fig3d.add_subplot(133, projection='3d')

    plot_3d_panel(ax_b,  vis_hits_before,
                  f'n = {len(vis_hits_before):,}')
    plot_3d_panel(ax_s1, vis_hits_after_s1,
                  f'n = {len(vis_hits_after_s1):,}  ({pct_s1:.0f}% removed)')
    plot_3d_panel(ax_s2, vis_hits_after_s2,
                  f'n = {len(vis_hits_after_s2):,}  ({pct_s2:.0f}% removed)')

    plt.tight_layout()

    # ── force axis line alpha after draw ─────────────────────────────────────
    for ax in [ax_b, ax_s1, ax_s2]:
        for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
            axis.line.set_alpha(0.15)

    plt.show()