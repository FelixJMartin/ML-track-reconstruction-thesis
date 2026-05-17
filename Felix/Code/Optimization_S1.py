# ── grid_s1_summary.py ────────────────────────────────────────────────────────
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

GRID_DIR = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\models\grid_s1"
OUT_DIR  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\figures\grid_s1"
os.makedirs(OUT_DIR, exist_ok=True)

# ── LOAD ALL CHECKPOINTS ──────────────────────────────────────────────────────
results = []
for fname in sorted(os.listdir(GRID_DIR)):
    if not fname.endswith('.pt'):
        continue
    ckpt = torch.load(os.path.join(GRID_DIR, fname), map_location='cpu')
    cfg  = ckpt['config']
    
    effs   = ckpt['val_effs']
    throws = ckpt['val_throws']
    accs   = ckpt['val_accs']
    losses = ckpt['val_losses']
    
    results.append({
        'name':       fname.replace('.pt','').replace('s1_',''),
        'H1':         cfg['H1'],
        'H2':         cfg['H2'],
        'LR':         cfg['LR'],
        'DROPOUT':    cfg['DROPOUT'],
        'best_acc':   max(accs)   * 100,
        'best_eff':   max(effs)   * 100,
        'best_throw': max(throws) * 100,
        'val_accs':   [v * 100 for v in accs],
        'val_effs':   [v * 100 for v in effs],
        'val_throws': [v * 100 for v in throws],
        'val_losses': losses,
        'train_losses': ckpt['train_losses'],
    })

# sort by best_eff (your priority)
results.sort(key=lambda x: x['best_eff'], reverse=True)

# ── PRINT SUMMARY TABLE ───────────────────────────────────────────────────────
print(f"\n{'='*75}")
print(f"  Stage 1 Grid Search Summary — ranked by signal efficiency")
print(f"{'='*75}")
print(f"  {'config':<35} {'best_acc':>9} {'best_eff':>9} {'best_throw':>11}")
print(f"  {'-'*35} {'-'*9} {'-'*9} {'-'*11}")
for r in results:
    print(f"  {r['name']:<35} {r['best_acc']:>8.1f}%  {r['best_eff']:>8.1f}%  {r['best_throw']:>10.1f}%")

# baseline for comparison
print(f"\n  {'baseline h256_128_lr1e-3':<35} {'83.0%':>9} {'82.7%':>9} {'55.6%':>11}")
print(f"{'='*75}\n")

# ── PLOT 1 — SUMMARY BAR CHART ────────────────────────────────────────────────
names      = [r['name'].replace('h','').replace('_lr','\nlr=').replace('_dr',' dr=') 
              for r in results]
effs       = [r['best_eff']   for r in results]
throws     = [r['best_throw'] for r in results]
x          = np.arange(len(results))
width      = 0.35

fig, ax = plt.subplots(figsize=(13, 5))
bars1 = ax.bar(x - width/2, effs,   width, color='#0077BB', alpha=0.8, label='Signal efficiency %')
bars2 = ax.bar(x + width/2, throws, width, color='#EE7733', alpha=0.8, label='Noise throwrate %')

ax.axhline(82.7, color='#0077BB', lw=1.2, ls='--', alpha=0.5, label='Baseline efficiency')
ax.axhline(55.6, color='#EE7733', lw=1.2, ls='--', alpha=0.5, label='Baseline throwrate')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=7)
ax.set_ylabel('%')
ax.set_ylim(50, 100)
ax.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
plt.tight_layout()
plt.show()

# ── PLOT 2 — EFFICIENCY vs THROWRATE SCATTER ─────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 5))

colors = {'256': '#0077BB', '512': '#EE7733'}
markers = {128: 'o', 256: 's'}

for r in results:
    c = colors[str(r['H1'])]
    m = markers[r['H2']]
    ax.scatter(r['best_throw'], r['best_eff'],
               color=c, marker=m, s=80, zorder=5)
    ax.annotate(f"lr={r['LR']}\ndr={r['DROPOUT']}",
                (r['best_throw'], r['best_eff']),
                fontsize=6, textcoords='offset points', xytext=(5, 3))

# baseline
ax.scatter(55.6, 82.7, color='black', marker='*', s=150, zorder=6, label='Baseline')
ax.annotate('baseline', (55.6, 82.7), fontsize=7,
            textcoords='offset points', xytext=(5, -10))

# legend for architecture
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#0077BB', markersize=8, label='H1=256'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#EE7733', markersize=8, label='H1=512'),
    Line2D([0], [0], marker='o', color='grey', markersize=8, label='H2=128'),
    Line2D([0], [0], marker='s', color='grey', markersize=8, label='H2=256'),
]
ax.legend(handles=legend_elements, fontsize=8, frameon=True,
          edgecolor='black', fancybox=False, framealpha=1.0)

ax.set_xlabel('Noise throwrate %')
ax.set_ylabel('Signal efficiency %')
ax.grid(True, alpha=0.3, linestyle='--')
plt.tight_layout()
plt.show()

# ── PLOT 3 — WINNER VS BASELINE LOSS CURVES ───────────────────────────────────
# load baseline
baseline_path = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices1000\model_h256_128_lr0.001_bs32_dr0.1.pt"
baseline_ckpt = torch.load(baseline_path, map_location='cpu')

winner = results[0]  # already sorted by best_eff
epochs_w = range(1, len(winner['val_losses']) + 1)
epochs_b = range(1, len(baseline_ckpt['val_losses']) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(epochs_b, baseline_ckpt['train_losses'], color='#0077BB', lw=1.2, ls='--', alpha=0.6, label='Baseline train')
ax1.plot(epochs_b, baseline_ckpt['val_losses'],   color='#EE7733', lw=1.2, ls='--', alpha=0.6, label='Baseline val')
ax1.plot(epochs_w, winner['train_losses'], color='#0077BB', lw=1.5, label='Winner train')
ax1.plot(epochs_w, winner['val_losses'],   color='#EE7733', lw=1.5, label='Winner val')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('BCE Loss')
ax1.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
ax1.grid(True, alpha=0.3, linestyle='--')

ax2.plot(epochs_b, [v * 100 for v in baseline_ckpt['val_accs']], color='#CC3311', lw=1.2, ls='--', alpha=0.6, label='Baseline acc')
ax2.plot(epochs_w, winner['val_accs'],color='#009988', lw=1.5, label='Winner acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('%')
ax2.set_ylim(50, 100)
ax2.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
ax2.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()

print(f"\nWinner: {results[0]['name']}")
print(f"  best_acc={results[0]['best_acc']:.1f}%")
print(f"  best_eff={results[0]['best_eff']:.1f}%")
print(f"  best_throw={results[0]['best_throw']:.1f}%")




