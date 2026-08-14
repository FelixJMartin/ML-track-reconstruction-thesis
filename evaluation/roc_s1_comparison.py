# ================================================================================
# roc_s1_comparison.py
# Compares stage 1 models on held-out test events (9500-9999, train_5):
#   - 1800-event winner: h=512/128, early exit
#   - 8500-event small:  h=64/32,   60 epochs
# ================================================================================

from trackml.dataset import load_event

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\SNN")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\build_datasets")
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np

from Functions import preprocess_particles
from build_data8500 import (make_rz_grid, get_cone, CONE_SIGNS,
                             SNN_R_BINS as R_BINS_8500,
                             SNN_R_MAX, SNN_Z_MAX, SAMPLE_FRACTION)

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

try:
    import scienceplots
    plt.style.use(["science", "no-latex"])
except ImportError:
    pass

plt.rcParams.update({
    "text.usetex":       False,
    "figure.dpi":        150,
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   9,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})

C = ["#0077BB", "#EE7733", "#009988", "#CC3311"]

# ── CONFIG ────────────────────────────────────────────────────────────────────
data_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_5"
test_ids  = [f"event{i:09d}" for i in range(9500, 10000)]

MODELS = {
    "1800 events, h=512/128 (winner)": {
        "path":     r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices1000\grid_s1\s1_h512_128_lr0.001_dr0.1.pt",
        "h1":       512,
        "h2":       128,
        "r_bins":   100,
        "dropout":  0.1,
        "color":    C[1],
        "ls":       "--",
    },
    "8500 events, h=64/32 (ours)": {
        "path":     r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\model_s1_h64_32.pt",
        "h1":       64,
        "h2":       32,
        "r_bins":   100,
        "dropout":  0.2,
        "color":    C[0],
        "ls":       "-",
    },
}

# ── MODEL ─────────────────────────────────────────────────────────────────────
class ConeFilterSNN(nn.Module):
    def __init__(self, input_size, h1, h2, bs, dropout):
        super().__init__()
        from snns import RadLIFLayer
        self.lif1 = RadLIFLayer(input_size, h1, bs,
                                normalization='batchnorm', dropout=dropout)
        self.lif2 = RadLIFLayer(h1, h2, bs,
                                normalization='batchnorm', dropout=dropout)
        self.fc   = nn.Linear(h2, 1)

    def forward(self, x):
        return self.fc(self.lif2(self.lif1(x)).mean(dim=1)).squeeze(-1)

    def set_bs(self, bs):
        self.lif1.batch_size = bs
        self.lif2.batch_size = bs


# ── EVALUATE ONE MODEL ────────────────────────────────────────────────────────
def evaluate(model, r_bins, test_ids, data_dir):
    all_probs      = []
    all_has_signal = []

    print(f"  Evaluating {len(test_ids)} events...")
    for event_id in test_ids:
        try:
            h, _, p, t = load_event(os.path.join(data_dir, event_id))
            h["r"]   = np.hypot(h["x"], h["y"])
            h["phi"] = np.arctan2(h["y"], h["x"])
            p_cond, _, _ = preprocess_particles(10, 10, p, h)
            backr = h[np.random.rand(len(h)) < SAMPLE_FRACTION].copy()

            for sx, sy, sz in CONE_SIGNS:
                sig, combined = get_cone(h, t, p_cond, sx, sy, sz, backr)
                noi = combined[~combined['hit_id'].isin(sig['hit_id'])]

                all_r = np.concatenate([sig['r'].values, noi['r'].values])
                all_z = np.concatenate([sig['z'].values, noi['z'].values])

                with torch.no_grad():
                    prob = torch.sigmoid(
                        model(torch.tensor(make_rz_grid(all_r, all_z).T).unsqueeze(0))
                    ).item()

                all_probs.append(prob)
                all_has_signal.append(len(sig) > 0)

        except Exception as e:
            print(f"    {event_id}: skipped — {e}")

    return (np.array(all_probs,      dtype=np.float32),
            np.array(all_has_signal, dtype=bool))


# ── ROC CURVE ─────────────────────────────────────────────────────────────────
def compute_roc(probs, labels):
    thresholds   = np.linspace(0, 1, 300)
    efficiencies = []
    throwrates   = []
    for thr in thresholds:
        pred = probs > thr
        TP = ( pred &  labels).sum()
        FN = (~pred &  labels).sum()
        TN = (~pred & ~labels).sum()
        FP = ( pred & ~labels).sum()
        efficiencies.append(TP / (TP + FN) if (TP + FN) > 0 else 0)
        throwrates.append(  TN / (TN + FP) if (TN + FP) > 0 else 0)
    return np.array(efficiencies), np.array(throwrates)


# ── MAIN ──────────────────────────────────────────────────────────────────────
results = {}

for name, cfg in MODELS.items():
    print(f"\nLoading: {name}")
    model = ConeFilterSNN(cfg['r_bins'], cfg['h1'], cfg['h2'], bs=1, dropout=cfg['dropout'])
    ckpt  = torch.load(cfg['path'], map_location='cpu')
    model.load_state_dict(ckpt['model_state'])
    model.eval(); model.set_bs(1)

    probs, labels = evaluate(model, cfg['r_bins'], test_ids, data_dir)
    eff, thr      = compute_roc(probs, labels)

    # stats at p=0.5
    pred  = probs > 0.5
    TP = ( pred &  labels).sum()
    FN = (~pred &  labels).sum()
    TN = (~pred & ~labels).sum()
    FP = ( pred & ~labels).sum()
    acc   = (TP + TN) / len(probs)
    e05   = TP / (TP + FN) if (TP + FN) > 0 else 0
    t05   = TN / (TN + FP) if (TN + FP) > 0 else 0

    results[name] = {
        "probs": probs, "labels": labels,
        "eff": eff, "thr": thr,
        "acc": acc, "e05": e05, "t05": t05,
        "color": cfg['color'], "ls": cfg['ls'],
    }

    print(f"  p=0.5 → acc={acc*100:.1f}%  eff={e05*100:.1f}%  throwrate={t05*100:.1f}%")


# ── PLOT 1: ROC comparison ────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(7, 6))

for name, r in results.items():
    ax.plot(r['thr'], r['eff'], color=r['color'], ls=r['ls'],
            linewidth=2.0, label=name)
    # mark p=0.5
    ax.plot(r['t05'], r['e05'], 'o', color=r['color'], markersize=8, zorder=5)

ax.set_xlabel(f'Noise throwrate')
ax.set_ylabel(f'Signal efficiency')
ax.set_xlim(0, 1); ax.set_ylim(0, 1)
ax.legend(loc='lower left', framealpha=0.9)
ax.grid(True, alpha=0.3, linestyle='--')
ax.set_title('Stage 1 ROC: 1800-event winner vs 8500-event model\n(evaluated on 500 held-out test events)')
plt.tight_layout()
plt.savefig(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\ROC_comparison.png", dpi=150)
plt.show()
print("ROC comparison saved.")


# ── PLOT 2: Confidence histograms side by side ────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, (name, r) in zip(axes, results.items()):
    n_sig = r['labels'].sum()
    n_noi = (~r['labels']).sum()
    ax.hist(r['probs'][ r['labels']], bins=40, alpha=0.65,
            color=C[1], label=f'Signal (n={n_sig})', density=True)
    ax.hist(r['probs'][~r['labels']], bins=40, alpha=0.65,
            color=C[0], label=f'Noise (n={n_noi})',  density=True)
    ax.axvline(0.5, color='black', lw=1.0, ls='--', label='p=0.5')
    ax.set_xlabel('Model confidence p')
    ax.set_ylabel('Density')
    ax.set_title(name)
    ax.legend()

plt.tight_layout()
plt.savefig(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\confidence_comparison.png", dpi=150)
plt.show()
print("Confidence comparison saved.")


# ── SUMMARY TABLE ─────────────────────────────────────────────────────────────
print(f"\n{'Model':<40} {'Acc':>6} {'Eff@0.5':>9} {'Throw@0.5':>11}")
print("-" * 70)
for name, r in results.items():
    print(f"{name:<40} {r['acc']*100:>5.1f}%  {r['e05']*100:>8.1f}%  {r['t05']*100:>10.1f}%")

print(f"\nThreshold sweep:")
print(f"\n{'threshold':>10}  {'':^35}  {'':^35}")
print(f"{'':>10}  {'1800 winner':^35}  {'8500 model':^35}")
print(f"{'':>10}  {'eff':>10}  {'throw':>10}  {'':>10}  {'eff':>10}  {'throw':>10}")
print("-" * 85)

r1 = list(results.values())[0]
r2 = list(results.values())[1]

for thr in [0.3, 0.5, 0.7, 0.8, 0.9]:
    stats = []
    for r in [r1, r2]:
        p_  = r['probs'] > thr
        tp  = ( p_ &  r['labels']).sum()
        fn  = (~p_ &  r['labels']).sum()
        tn  = (~p_ & ~r['labels']).sum()
        fp  = ( p_ & ~r['labels']).sum()
        e   = tp / (tp + fn) if (tp + fn) > 0 else 0
        tr  = tn / (tn + fp) if (tn + fp) > 0 else 0
        stats.append((e, tr))
    print(f"{thr:>10.2f}  {stats[0][0]*100:>9.1f}%  {stats[0][1]*100:>9.1f}%"
          f"  {'':>10}  {stats[1][0]*100:>9.1f}%  {stats[1][1]*100:>9.1f}%")
    