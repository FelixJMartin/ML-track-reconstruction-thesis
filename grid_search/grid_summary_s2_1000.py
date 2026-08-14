# ── grid_summary_s2_1000.py ─────────────────────────────────────────────────
# Grid search summary for Stage 2 (1000-event dataset, pw=10).
# 24 configs: H1∈{256,512} × H2∈{128,256} × LR∈{1e-3,5e-4,3e-4} × dr∈{0.1,0.2}
# Baseline : matrices1000/stage2_s2_h512_256_lr1e-3_dr0.1_pw10.pt
# ──────────────────────────────────────────────────────────────────────────────
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401
from matplotlib.lines import Line2D

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "text.usetex":       False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
    "axes.spines.top":   True,
    "axes.spines.right": True,
    "figure.dpi":        150,
})

GRID_DIR      = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices1000\grid_s2"
BASELINE_PATH = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices1000\stage2_s2_h512_256_lr1e-3_dr0.1_pw10.pt"
OUT_DIR       = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\figures\grid_s2"
os.makedirs(OUT_DIR, exist_ok=True)

C = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#33BBEE", "#EE3377"]

# ── LOAD BASELINE ─────────────────────────────────────────────────────────────
baseline_ckpt    = torch.load(BASELINE_PATH, map_location='cpu')
baseline_iou     = max(v * 100 for v in baseline_ckpt['val_ious'])
baseline_overlap = max(v * 100 for v in baseline_ckpt['val_overlaps'])
print(f"Baseline  IoU={baseline_iou:.1f}%  Overlap={baseline_overlap:.1f}%")

# ── LOG PARSER ────────────────────────────────────────────────────────────────
import re

LOG_DIR = os.path.join(GRID_DIR, "logs")

_lr_map = {'1e-3': 0.001, '5e-4': 0.0005, '3e-4': 0.0003}
_ep_re  = re.compile(
    r'ep\s*(\d+)/\d+\s+loss=([\d.]+)\s+val=([\d.]+)\s+iou=([\d.]+)%\s+overlap=([\d.]+)%'
)

def parse_log(path):
    train_l, val_l, ious, overlaps = [], [], [], []
    with open(path) as f:
        for line in f:
            m = _ep_re.search(line)
            if m:
                train_l.append(float(m.group(2)))
                val_l.append(float(m.group(3)))
                ious.append(float(m.group(4)))
                overlaps.append(float(m.group(5)))
    return train_l, val_l, ious, overlaps

def log_name_to_pt_name(log_fname):
    """Convert log_s2_h512_256_lr1e-3_dr0.1_JOBID.log → h512_256_lr0.001_dr0.1_pw10.0"""
    s = log_fname.replace('log_s2_', '').rsplit('_', 1)[0]  # strip jobid
    for short, full in _lr_map.items():
        s = s.replace(f'lr{short}', f'lr{full}')
    return s + '_pw10.0'

# build log lookup: pt_name → (train_l, val_l, ious, overlaps)
log_lookup = {}
if os.path.isdir(LOG_DIR):
    for lf in os.listdir(LOG_DIR):
        if not lf.endswith('.log'):
            continue
        key = log_name_to_pt_name(lf)
        log_lookup[key] = parse_log(os.path.join(LOG_DIR, lf))
    print(f"Parsed {len(log_lookup)} log files from {LOG_DIR}")
else:
    print(f"No logs directory found at {LOG_DIR} — using checkpoint history only")

# ── LOAD ALL CHECKPOINTS ──────────────────────────────────────────────────────
results = []
for fname in sorted(os.listdir(GRID_DIR)):
    if not fname.endswith('.pt'):
        continue
    ckpt = torch.load(os.path.join(GRID_DIR, fname), map_location='cpu')
    cfg  = ckpt.get('config', {})

    pt_key = fname.replace('.pt', '').replace('s2_', '')
    if pt_key in log_lookup:
        train_l, val_l, ious, overlaps = log_lookup[pt_key]
    else:
        train_l  = ckpt['train_losses']
        val_l    = ckpt['val_losses']
        ious     = [v * 100 for v in ckpt['val_ious']]
        overlaps = [v * 100 for v in ckpt['val_overlaps']]

    results.append({
        'name':         pt_key,
        'H1':           cfg.get('H1', 256),
        'H2':           cfg.get('H2', 128),
        'LR':           cfg.get('LR', 0.001),
        'DROPOUT':      cfg.get('DROPOUT', 0.1),
        'best_val':     min(val_l),
        'best_iou':     max(ious),
        'best_overlap': max(overlaps),
        'val_ious':     ious,
        'val_overlaps': overlaps,
        'val_losses':   val_l,
        'train_losses': train_l,
    })

results_by_iou = sorted(results, key=lambda x: x['best_iou'], reverse=True)
results_by_val = sorted(results, key=lambda x: x['best_val'])

# ── PRINT SUMMARY TABLE ───────────────────────────────────────────────────────
print(f"\n{'='*80}")
print(f"  Stage 2 Grid Search — {len(results)} configs — ranked by best IoU")
print(f"{'='*80}")
print(f"  {'rank':<5} {'config':<38} {'IoU':>7} {'overlap':>9} {'val_loss':>10}")
print(f"  {'-'*78}")
for i, r in enumerate(results_by_iou):
    mark = ' ★' if i == 0 else '  '
    print(f"  {i+1:<5}{mark}{r['name']:<36} "
          f"{r['best_iou']:>6.1f}% {r['best_overlap']:>8.1f}% {r['best_val']:>10.4f}")
print(f"\n  {'  baseline':<43} {baseline_iou:>6.1f}% {baseline_overlap:>8.1f}%")
print(f"{'='*80}\n")


# ── PLOT 0 — SMALL-MULTIPLES GRID OVERVIEW ───────────────────────────────────
NCOLS, NROWS = 6, 4
fig, axes = plt.subplots(NROWS, NCOLS, figsize=(18, 11),
                         sharex=False, sharey=True)
fig.subplots_adjust(hspace=0.55, wspace=0.08, left=0.06, right=0.99,
                    top=0.92, bottom=0.06)

all_vals = [v for r in results for v in r['val_losses'] + r['train_losses']]
y_lo = max(0.0,  min(all_vals) - 0.02)
y_hi = min(1.5,  max(all_vals) + 0.08)

for idx, (r, ax) in enumerate(zip(results_by_iou, axes.flat)):
    is_winner = (idx == 0)
    epochs    = range(1, len(r['val_losses']) + 1)

    if is_winner:
        ax.set_facecolor('#F2FAF2')
        for spine in ax.spines.values():
            spine.set_edgecolor('#2E7D32')
            spine.set_linewidth(2.0)
        c_train, c_val, lw, a_train = C[0], C[1], 1.5, 0.55
    else:
        ax.set_facecolor('white')
        for spine in ax.spines.values():
            spine.set_edgecolor('#CCCCCC')
            spine.set_linewidth(0.7)
        c_train, c_val, lw, a_train = '#BBBBBB', '#888888', 0.9, 0.45

    ax.plot(epochs, r['train_losses'], color=c_train, lw=lw, alpha=a_train, ls='--')
    ax.plot(epochs, r['val_losses'],   color=c_val,   lw=lw)

    badge_color = '#2E7D32' if is_winner else '#777777'
    ax.text(0.06, 0.93, f'#{idx+1}', transform=ax.transAxes,
            fontsize=6.5, va='top', color=badge_color,
            fontweight='bold' if is_winner else 'normal')
    ax.text(0.97, 0.93, f'IoU {r["best_iou"]:.1f}%', transform=ax.transAxes,
            fontsize=6, va='top', ha='right', color='#444444')

    if is_winner:
        ax.text(0.5, 0.48, '★', transform=ax.transAxes,
                fontsize=28, ha='center', va='center',
                color='#2E7D32', alpha=0.12)

    label = f"h{r['H1']},{r['H2']}\nlr={r['LR']}  dr={r['DROPOUT']}"
    ax.set_title(label, fontsize=6, pad=3,
                 color='#2E7D32' if is_winner else '#444444',
                 fontweight='bold' if is_winner else 'normal',
                 linespacing=1.3)

    ax.set_xlim(1, 100)
    ax.set_ylim(y_lo, y_hi)
    ax.tick_params(labelsize=5.5, length=2)
    ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)

for ax in axes.flat[len(results_by_iou):]:
    ax.set_visible(False)

fig.text(0.5,  0.015, 'Epoch',    ha='center', fontsize=9)
fig.text(0.015, 0.5,  'BCE Loss', va='center', rotation='vertical', fontsize=9)

legend_handles = [
    Line2D([0], [0], color=C[0], lw=1.4, ls='--', alpha=0.7, label='Train loss'),
    Line2D([0], [0], color=C[1], lw=1.4,           label='Val loss'),
]
axes[0, -1].legend(handles=legend_handles, fontsize=7, loc='upper right',
                   framealpha=0.9, frameon=True, edgecolor='#999999')

plt.suptitle(
    f'Stage 2 grid search — {len(results)} configurations, sorted by best IoU  (winner top-left)',
    fontsize=10, y=0.97)
plt.savefig(os.path.join(OUT_DIR, 'grid_s2_overview.pdf'), bbox_inches='tight')
plt.show()


# ── PLOT 1 — RANKED BAR CHART (IoU + overlap) ────────────────────────────────
names    = [r['name'].replace('h', 'H').replace('_lr', '\nlr=').replace('_dr', '  dr=')
                     .replace('_pw10.0', '')
            for r in results_by_iou]
ious     = [r['best_iou']     for r in results_by_iou]
overlaps = [r['best_overlap'] for r in results_by_iou]
x        = np.arange(len(results_by_iou))
width    = 0.38

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - width/2, ious,     width, color=C[0], alpha=0.82, label='IoU %')
ax.bar(x + width/2, overlaps, width, color=C[1], alpha=0.82, label='Overlap %')

ax.bar(0 - width/2, ious[0],     width, color=C[0], alpha=1.0, edgecolor='#2E7D32', linewidth=1.5)
ax.bar(0 + width/2, overlaps[0], width, color=C[1], alpha=1.0, edgecolor='#2E7D32', linewidth=1.5)
ax.text(0, ious[0] + 0.4, '★', ha='center', fontsize=10, color='#2E7D32')

ax.axhline(baseline_iou,     color=C[0], lw=1.2, ls='--', alpha=0.6,
           label=f'Baseline IoU {baseline_iou:.1f}%')
ax.axhline(baseline_overlap, color=C[1], lw=1.2, ls='--', alpha=0.6,
           label=f'Baseline overlap {baseline_overlap:.1f}%')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=6.5)
ax.set_ylabel('%')
ax.set_ylim(30, 105)
ax.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.suptitle('Stage 2 — IoU & overlap, ranked', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'grid_s2_bar.pdf'), bbox_inches='tight')
plt.show()


# ── PLOT 2 — IoU vs OVERLAP SCATTER ──────────────────────────────────────────
h1_colors  = {256: C[0], 512: C[1]}
h2_markers = {128: 'o',  256: 's'}

fig, ax = plt.subplots(figsize=(7, 5))
for r in results:
    ax.scatter(r['best_overlap'], r['best_iou'],
               color=h1_colors.get(r['H1'], 'grey'),
               marker=h2_markers.get(r['H2'], 'o'),
               s=70, zorder=5, alpha=0.85)
    ax.annotate(f"lr={r['LR']}\ndr={r['DROPOUT']}",
                (r['best_overlap'], r['best_iou']),
                fontsize=5.5, textcoords='offset points', xytext=(4, 2),
                color='#444444')

w = results_by_iou[0]
ax.scatter(w['best_overlap'], w['best_iou'],
           color=h1_colors.get(w['H1'], 'grey'),
           marker=h2_markers.get(w['H2'], 'o'),
           s=130, zorder=6, edgecolors='#2E7D32', linewidths=2.0)
ax.text(w['best_overlap'], w['best_iou'] + 0.3, '★', ha='center',
        fontsize=11, color='#2E7D32', zorder=7)

ax.axvline(baseline_overlap, color='grey', lw=1.0, ls='--', alpha=0.6, label='Baseline overlap')
ax.axhline(baseline_iou,     color='grey', lw=1.0, ls=':',  alpha=0.6, label='Baseline IoU')

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C[0], markersize=8, label='H1=256'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C[1], markersize=8, label='H1=512'),
    Line2D([0], [0], marker='o', color='grey', markersize=8, label='H2=128'),
    Line2D([0], [0], marker='s', color='grey', markersize=8, label='H2=256'),
    Line2D([0], [0], color='grey', lw=1.0, ls='--', label='Baseline overlap'),
    Line2D([0], [0], color='grey', lw=1.0, ls=':',  label='Baseline IoU'),
]
ax.legend(handles=legend_elements, fontsize=8, frameon=True,
          edgecolor='black', fancybox=False, framealpha=1.0)
ax.set_xlabel('Best overlap %')
ax.set_ylabel('Best IoU %')
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'grid_s2_scatter.pdf'), bbox_inches='tight')
plt.show()


# ── PLOT 3 — WINNER vs BASELINE CURVES ───────────────────────────────────────
w        = results_by_iou[0]
epochs_w = range(1, len(w['val_losses']) + 1)
epochs_b = range(1, len(baseline_ckpt['val_losses']) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(epochs_b, baseline_ckpt['train_losses'],
         color=C[0], lw=1.2, ls='--', alpha=0.6, label='Baseline train')
ax1.plot(epochs_b, baseline_ckpt['val_losses'],
         color=C[1], lw=1.2, ls='--', alpha=0.6, label='Baseline val')
ax1.plot(epochs_w, w['train_losses'], color=C[0], lw=1.5, label='Winner train')
ax1.plot(epochs_w, w['val_losses'],   color=C[1], lw=1.2, label='Winner val')
ax1.set_xlim(1, 100)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('BCE Loss')
ax1.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)

ax2.plot(epochs_b, [v * 100 for v in baseline_ckpt['val_ious']],
         color=C[0], lw=1.2, ls='--', alpha=0.6, label=f'Baseline IoU (best {baseline_iou:.1f}%)')
ax2.plot(epochs_b, [v * 100 for v in baseline_ckpt['val_overlaps']],
         color=C[1], lw=1.2, ls='--', alpha=0.6, label=f'Baseline overlap (best {baseline_overlap:.1f}%)')
ax2.plot(epochs_w, w['val_ious'],     color=C[0], lw=1.5, label='Winner IoU')
ax2.plot(epochs_w, w['val_overlaps'], color=C[1], lw=1.2, label='Winner overlap')
ax2.set_xlim(1, 100)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('%')
ax2.set_ylim(0, 105)
ax2.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)

plt.tight_layout()
plt.show()

print(f"\n★ Winner: {w['name']}")
print(f"  best_iou={w['best_iou']:.1f}%  best_overlap={w['best_overlap']:.1f}%  best_val={w['best_val']:.4f}")
print(f"\n  Baseline IoU={baseline_iou:.1f}%  Overlap={baseline_overlap:.1f}%")
