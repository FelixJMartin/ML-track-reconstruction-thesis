# ── grid_s2_summary.py ────────────────────────────────────────────────────────
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

GRID_DIR     = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\models\grid_s2"
BASELINE_PATH= r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices1000\model_stage2_h256_128_lr0.001_bs32_dr0.1.pt"
OUT_DIR      = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\figures\grid_s2"
os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD ALL CHECKPOINTS ──────────────────────────────────────────────────────
results = []
for fname in sorted(os.listdir(GRID_DIR)):
    if not fname.endswith('.pt') or not fname.startswith('s2_'):
        continue
    ckpt = torch.load(os.path.join(GRID_DIR, fname), map_location='cpu')
    cfg  = ckpt['config']

    ious     = ckpt['val_ious']
    overlaps = ckpt['val_overlaps']
    losses   = ckpt['val_losses']

    results.append({
        'name':         fname.replace('.pt','').replace('s2_',''),
        'H1':           cfg['H1'],
        'H2':           cfg['H2'],
        'LR':           cfg['LR'],
        'PW':           cfg['POS_WEIGHT_CAP'],
        'best_iou':     max(ious)     * 100,
        'best_overlap': max(overlaps) * 100,
        'val_ious':     [v * 100 for v in ious],
        'val_overlaps': [v * 100 for v in overlaps],
        'val_losses':   losses,
        'train_losses': ckpt['train_losses'],
    })

results.sort(key=lambda x: x['best_iou'], reverse=True)

# ── PRINT SUMMARY TABLE ───────────────────────────────────────────────────────
print(f"\n{'='*70}")
print(f"  Stage 2 Grid Search Summary — ranked by IoU")
print(f"{'='*70}")
print(f"  {'config':<40} {'best_iou':>9} {'best_overlap':>13}")
print(f"  {'-'*40} {'-'*9} {'-'*13}")
for r in results:
    print(f"  {r['name']:<40} {r['best_iou']:>8.1f}%  {r['best_overlap']:>12.1f}%")
print(f"\n  {'baseline h256_128_lr1e-3_pw10':<40} {'70.0%':>9} {'95.0%':>13}")
print(f"{'='*70}\n")

# ── PLOT 1 — BAR CHART ranked by IoU ─────────────────────────────────────────
names  = [r['name'].replace('h','').replace('_lr','\nlr=')
           .replace('_dr0.1','').replace('_pw',' pw=')
          for r in results]
ious     = [r['best_iou']     for r in results]
overlaps = [r['best_overlap'] for r in results]
x        = np.arange(len(results))
width    = 0.35

fig, ax = plt.subplots(figsize=(14, 5))
ax.bar(x - width/2, ious,     width, color='#0077BB', alpha=0.8, label='IoU %')
ax.bar(x + width/2, overlaps, width, color='#EE7733', alpha=0.8, label='Overlap %')

ax.axhline(70.0, color='#0077BB', lw=1.2, ls='--', alpha=0.5, label='Baseline IoU')
ax.axhline(95.0, color='#EE7733', lw=1.2, ls='--', alpha=0.5, label='Baseline overlap')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=6.5)
ax.set_ylabel('%')
ax.set_ylim(30, 105)
ax.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'grid_s2_bar.pdf'), bbox_inches='tight')
plt.show()

# ── PLOT 2 — pos_weight effect ────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

colors  = {'256': '#0077BB', '512': '#EE7733'}
markers = {128: 'o', 256: 's'}
pws     = [10.0, 30.0, 68.0]

for r in results:
    c = colors[str(r['H1'])]
    m = markers[r['H2']]
    ax.scatter(r['PW'], r['best_iou'], color=c, marker=m, s=80, zorder=5)

ax.axhline(70.0, color='black', lw=1.2, ls='--', alpha=0.5, label='Baseline IoU')

from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#0077BB', markersize=8, label='H1=256'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#EE7733', markersize=8, label='H1=512'),
    Line2D([0], [0], marker='o', color='grey', markersize=8, label='H2=128'),
    Line2D([0], [0], marker='s', color='grey', markersize=8, label='H2=256'),
    Line2D([0], [0], color='black', lw=1.2, ls='--', label='Baseline IoU'),
]
ax.legend(handles=legend_elements, fontsize=8, frameon=True,
          edgecolor='black', fancybox=False, framealpha=1.0)

ax.set_xlabel('pos_weight cap')
ax.set_ylabel('Best IoU %')
ax.set_xticks([10, 30, 68])
ax.set_xticklabels(['10', '30', '68'])
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# ── PLOT 3 — WINNER VS BASELINE LOSS + IoU CURVES ────────────────────────────
baseline_ckpt = torch.load(BASELINE_PATH, map_location='cpu')

winner   = results[0]
epochs_w = range(1, len(winner['val_losses']) + 1)
epochs_b = range(1, len(baseline_ckpt['val_losses']) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(epochs_b, baseline_ckpt['train_losses'],
         color='#0077BB', lw=1.2, ls='--', alpha=0.6, label='Baseline train')
ax1.plot(epochs_b, baseline_ckpt['val_losses'],
         color='#EE7733', lw=1.2, ls='--', alpha=0.6, label='Baseline val')
ax1.plot(epochs_w, winner['train_losses'],
         color='#0077BB', lw=1.5, label='Winner train')
ax1.plot(epochs_w, winner['val_losses'],
         color='#EE7733', lw=1.5, label='Winner val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('BCE Loss')
ax1.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
ax1.grid(True, alpha=0.3, linestyle='--')

ax2.plot(epochs_b, [v * 100 for v in baseline_ckpt['val_ious']],
         color='#CC3311', lw=1.2, ls='--', alpha=0.6, label='Baseline IoU')
ax2.plot(epochs_w, winner['val_ious'],
         color='#009988', lw=1.5, label='Winner IoU')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('IoU %')
ax2.set_ylim(0, 100)
ax2.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

print(f"\nWinner: {results[0]['name']}")
print(f"  best_iou     = {results[0]['best_iou']:.1f}%")
print(f"  best_overlap = {results[0]['best_overlap']:.1f}%")