# ── grid_summary_s1_150.py ──────────────────────────────────────────────────
# Grid search summary for Stage 1 (150-bin, 100-event dataset).
# 24 configs: H1∈{256,512} × H2∈{128,256} × LR∈{1e-3,5e-4,3e-4} × dr∈{0.1,0.2}
# Models stored in: data/matrices150/
# ──────────────────────────────────────────────────────────────────────────────
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots  # noqa: F401 — registers the "science" style
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

GRID_DIR = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices1000\grid_s1"
OUT_DIR  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\figures\grid_s1"
os.makedirs(OUT_DIR, exist_ok=True)

C = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#33BBEE", "#EE3377"]

# ── LOAD ALL CHECKPOINTS ──────────────────────────────────────────────────────
results = []
for fname in sorted(os.listdir(GRID_DIR)):
    if not fname.endswith('.pt') or fname == 'model_stage1.pt':
        continue
    path = os.path.join(GRID_DIR, fname)
    ckpt = torch.load(path, map_location='cpu')
    cfg  = ckpt.get('config', {})

    train_l = ckpt['train_losses']
    val_l   = ckpt['val_losses']
    accs    = ckpt['val_accs']
    effs    = ckpt.get('val_effs',   [0.0])
    throws  = ckpt.get('val_throws', [0.0])

    results.append({
        'name':         fname.replace('.pt', '').replace('s1_', ''),
        'H1':           cfg.get('H1',      int(fname.split('_h')[1].split('_')[0].split('_')[0])),
        'H2':           cfg.get('H2',      128),
        'LR':           cfg.get('LR',      0.001),
        'DROPOUT':      cfg.get('DROPOUT', 0.1),
        'best_val':     min(val_l),
        'best_acc':     max(accs)   * 100,
        'best_eff':     max(effs)   * 100,
        'best_throw':   max(throws) * 100,
        'val_accs':     [v * 100 for v in accs],
        'val_effs':     [v * 100 for v in effs],
        'val_throws':   [v * 100 for v in throws],
        'val_losses':   val_l,
        'train_losses': train_l,
    })

# parse H1/H2 from filename if config missing
for r in results:
    if isinstance(r['H1'], int):
        continue
    parts = r['name'].split('_')
    h_part = next((p for p in parts if p.startswith('h')), None)
    if h_part:
        nums = h_part[1:].split('_')
        r['H1'] = int(nums[0]) if len(nums) > 0 else 256
        r['H2'] = int(nums[1]) if len(nums) > 1 else 128

# ── PRINT SUMMARY TABLE ───────────────────────────────────────────────────────
results_by_val  = sorted(results, key=lambda x: x['best_val'])
results_by_eff  = sorted(results, key=lambda x: x['best_eff'], reverse=True)

print(f"\n{'='*80}")
print(f"  Stage 1 Grid Search — {len(results)} configs — ranked by best val loss")
print(f"{'='*80}")
print(f"  {'rank':<5} {'config':<38} {'val_loss':>9} {'acc':>7} {'eff':>7} {'throw':>7}")
print(f"  {'-'*78}")
for i, r in enumerate(results_by_val):
    marker = ' ★' if i == 0 else '  '
    print(f"  {i+1:<5}{marker}{r['name']:<36} "
          f"{r['best_val']:>9.4f} {r['best_acc']:>6.1f}% {r['best_eff']:>6.1f}% {r['best_throw']:>6.1f}%")
print(f"{'='*80}\n")


# ── PLOT 0 — OVERLAY: all 24 configs, winner highlighted ─────────────────────
all_vals = [v for r in results for v in r['val_losses'] + r['train_losses']]
y_lo = max(0.45, min(all_vals) - 0.04)
y_hi = min(2.0,  max(all_vals) + 0.10)

fig, ax = plt.subplots(figsize=(8, 5))

for idx, r in enumerate(results_by_val):
    epochs = range(1, len(r['val_losses']) + 1)
    is_winner = (idx == 0)

    if is_winner:
        # draw train first (underneath), then val on top
        ax.plot(epochs, r['train_losses'],
                color=C[0], lw=1.6, ls='--', alpha=0.75,
                label='Winner — train loss', zorder=5)
        ax.plot(epochs, r['val_losses'],
                color=C[1], lw=1.2,
                label='Winner — val loss', zorder=6)
    else:
        ax.plot(epochs, r['train_losses'],
                color='#CCCCCC', lw=0.7, ls='--', alpha=0.4, zorder=2)
        ax.plot(epochs, r['val_losses'],
                color='#AAAAAA', lw=0.7, alpha=0.35, zorder=2)

# ghost legend entries for the losers
from matplotlib.lines import Line2D
ghost_train = Line2D([0], [0], color='#CCCCCC', lw=1.0, ls='--', alpha=0.6,
                     label='Other configs — train loss')
ghost_val   = Line2D([0], [0], color='#AAAAAA', lw=1.0, alpha=0.6,
                     label='Other configs — val loss')



handles, labels = ax.get_legend_handles_labels()
ax.legend(
    handles + [ghost_train, ghost_val],
    labels  + [ghost_train.get_label(), ghost_val.get_label()],
    fontsize=7.5, frameon=True, edgecolor='#999999',
    framealpha=0.95, loc='upper left'
)

ax.set_xlim(1, 100)
ax.set_ylim(y_lo, y_hi)
ax.set_xlabel('Epoch', fontsize=9)
ax.set_ylabel('BCE Loss', fontsize=9)
ax.grid(True, alpha=0.25, linestyle='--', linewidth=0.5)
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
ax.tick_params(labelsize=8)


plt.tight_layout()
plt.show()


# ── PLOT 1 — RANKED BAR CHART (eff + throwrate) ──────────────────────────────
names  = [r['name'].replace('h', 'H').replace('_lr', '\nlr=').replace('_dr', '  dr=')
          for r in results_by_eff]
effs   = [r['best_eff']   for r in results_by_eff]
throws = [r['best_throw'] for r in results_by_eff]
x      = np.arange(len(results_by_eff))
width  = 0.38

fig, ax = plt.subplots(figsize=(14, 5))
bars1 = ax.bar(x - width/2, effs,   width, color=C[0], alpha=0.82, label='Signal efficiency %')
bars2 = ax.bar(x + width/2, throws, width, color=C[1], alpha=0.82, label='Noise throwrate %')

# highlight winner bar
ax.bar(0 - width/2, effs[0],   width, color=C[0], alpha=1.0, edgecolor='#2E7D32', linewidth=1.5)
ax.bar(0 + width/2, throws[0], width, color=C[1], alpha=1.0, edgecolor='#2E7D32', linewidth=1.5)
ax.text(0, effs[0] + 0.4, '★', ha='center', fontsize=10, color='#2E7D32')

ax.set_xticks(x)
ax.set_xticklabels(names, fontsize=6.5)
ax.set_ylabel('%')
ax.set_ylim(50, 102)
ax.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
ax.grid(True, alpha=0.3, linestyle='--', axis='y')
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.suptitle('Stage 1 — signal efficiency & noise throwrate, ranked', fontsize=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'grid_s1_bar.pdf'), bbox_inches='tight')
plt.show()


# ── PLOT 2 — EFFICIENCY vs THROWRATE SCATTER ─────────────────────────────────
h1_colors  = {256: C[0], 512: C[1]}
h2_markers = {128: 'o',  256: 's'}

fig, ax = plt.subplots(figsize=(7, 5))
for r in results:
    ax.scatter(r['best_throw'], r['best_eff'],
               color=h1_colors.get(r['H1'], 'grey'),
               marker=h2_markers.get(r['H2'], 'o'),
               s=70, zorder=5, alpha=0.85)
    ax.annotate(f"lr={r['LR']}\ndr={r['DROPOUT']}",
                (r['best_throw'], r['best_eff']),
                fontsize=5.5, textcoords='offset points', xytext=(4, 2),
                color='#444444')

# mark winner
w = results_by_eff[0]
ax.scatter(w['best_throw'], w['best_eff'],
           color=h1_colors.get(w['H1'], 'grey'),
           marker=h2_markers.get(w['H2'], 'o'),
           s=130, zorder=6, edgecolors='#2E7D32', linewidths=2.0)
ax.text(w['best_throw'], w['best_eff'] + 0.4, '★', ha='center',
        fontsize=11, color='#2E7D32', zorder=7)

legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C[0], markersize=8, label='H1=256'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C[1], markersize=8, label='H1=512'),
    Line2D([0], [0], marker='o', color='grey', markersize=8,  label='H2=128'),
    Line2D([0], [0], marker='s', color='grey', markersize=8,  label='H2=256'),
]
ax.legend(handles=legend_elements, fontsize=8, frameon=True,
          edgecolor='black', fancybox=False, framealpha=1.0)
ax.set_xlabel('Noise throwrate %')
ax.set_ylabel('Signal efficiency %')
ax.grid(True, alpha=0.3, linestyle='--')
ax.spines['top'].set_visible(True)
ax.spines['right'].set_visible(True)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'grid_s1_scatter.pdf'), bbox_inches='tight')
plt.show()


# ── PLOT 3 — WINNER TRAINING CURVES (all metrics) ────────────────────────────
w      = results_by_eff[0]
epochs = range(1, len(w['val_losses']) + 1)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

ax1.plot(epochs, w['train_losses'], color=C[0], lw=1.4, ls='--', alpha=0.7, label='Train loss')
ax1.plot(epochs, w['val_losses'],   color=C[1], lw=1.6, label='Val loss')
ax1.set_xlim(1, 100)
ax1.set_xlabel('Epoch')
ax1.set_ylabel('BCE Loss')
ax1.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
ax1.grid(True, alpha=0.3, linestyle='--')
ax1.spines['top'].set_visible(True)
ax1.spines['right'].set_visible(True)

ax2.plot(epochs, w['val_effs'],   color=C[0], lw=1.5, label='Efficiency %')
ax2.plot(epochs, w['val_throws'], color=C[2], lw=1.5, label='Throwrate %')
ax2.plot(epochs, w['val_accs'],   color=C[1], lw=1.2, ls='--', alpha=0.7, label='Accuracy %')
ax2.set_xlim(1, 100)
ax2.set_xlabel('Epoch')
ax2.set_ylabel('%')
ax2.set_ylim(0, 105)
ax2.legend(fontsize=8, frameon=True, edgecolor='black', fancybox=False, framealpha=1.0)
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.spines['top'].set_visible(True)
ax2.spines['right'].set_visible(True)

plt.suptitle(f'★ Winner: {w["name"]}  '
             f'(eff={w["best_eff"]:.1f}%  throw={w["best_throw"]:.1f}%  acc={w["best_acc"]:.1f}%)',
             fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'grid_s1_winner.pdf'), bbox_inches='tight')
plt.show()

print(f"\n★ Winner: {w['name']}")
print(f"  best_val={w['best_val']:.4f}  acc={w['best_acc']:.1f}%  "
      f"eff={w['best_eff']:.1f}%  throw={w['best_throw']:.1f}%")
