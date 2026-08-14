# pipeline_walkthrough.py — pipeline visualisation for a single held-out event

import sys, os
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\SNN")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\build_datasets")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots  # noqa: F401 — registers "science" style
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
from trackml.dataset import load_event
from snns import RadLIFLayer
from Functions import preprocess_particles
from build_data8500 import (
    make_rz_grid, make_rphi_grid,
    SNN_R_BINS, SNN_PHI_BINS, CONE_PHI_RANGES, SAMPLE_FRACTION,
)

plt.style.use(["science", "no-latex"])
plt.rcParams.update({"text.usetex": False, "figure.dpi": 150,
                     "figure.facecolor": "white", "axes.facecolor": "white"})

# ── CONFIG ────────────────────────────────────────────────────────────────────
EVENT_ID  = "event000009999"
DATA_DIR  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_5"
S1_PATH   = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\s1_8500_ep115.pt"
S2_PATH   = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\s2_h64_32_final.pt"
THRESH_S1 = 0.3
THRESH_S2 = 0.09
SNN_INPUT_SIZE = SNN_R_BINS + 1

OCTANTS = [
    (1, "oct1", ( 1,  1,  1)),
    (2, "oct2", ( 1,  1, -1)),
    (3, "oct3", ( 1, -1,  1)),
    (4, "oct4", ( 1, -1, -1)),
    (5, "oct5", (-1,  1,  1)),
    (6, "oct6", (-1,  1, -1)),
    (7, "oct7", (-1, -1,  1)),
    (8, "oct8", (-1, -1, -1)),
]
PALETTE = sns.color_palette("tab10", 8)

# ── HELPERS ───────────────────────────────────────────────────────────────────
def split_cones(hits):
    cones = []
    for _, _, (sx, sy, sz) in OCTANTS:
        mask = (
            ((hits["x"] > 0) if sx > 0 else (hits["x"] < 0)) &
            ((hits["y"] > 0) if sy > 0 else (hits["y"] < 0)) &
            ((hits["z"] > 0) if sz > 0 else (hits["z"] < 0))
        )
        cones.append(hits[mask])
    return cones

def signal_hits_in_cone(cone, truth, particles, min_hits=10):
    good_pids = preprocess_particles(10, 10, particles, cone)[0]["particle_id"]
    sig_ids   = truth[truth["particle_id"].isin(good_pids)]["hit_id"]
    sig       = cone[cone["hit_id"].isin(sig_ids)]
    return sig if len(sig) >= min_hits else sig.iloc[0:0]

def model_input(noise_cone, sig):
    if len(sig) > 0:
        noise_only = noise_cone[~noise_cone["hit_id"].isin(sig["hit_id"])]
        return pd.concat([noise_only, sig])
    return noise_cone

def phi_bin_idx(hits, phi_min, phi_max):
    return np.clip(
        ((hits["phi"].values - phi_min) / (phi_max - phi_min) * SNN_PHI_BINS
         ).astype(int),
        0, SNN_PHI_BINS - 1,
    )

def style_3d(ax):
    ax.set_xlim(-1200, 1200); ax.set_ylim(-1200, 1200); ax.set_zlim(-1200, 1200)
    ax.set_xlabel("Z", labelpad=6, color="#aaaaaa", fontsize=8)
    ax.set_ylabel("X", labelpad=6, color="#aaaaaa", fontsize=8)
    ax.set_zlabel("Y", labelpad=6, color="#aaaaaa", fontsize=8)
    ax.tick_params(colors="#cccccc", labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.97, 0.97, 0.97, 0.1))
        pane.set_edgecolor("#dddddd")
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo["grid"].update(color="#dddddd", linewidth=0.4, linestyle="--")

def new_3d():
    fig = plt.figure(figsize=(11, 7))
    ax  = fig.add_subplot(111, projection="3d")
    ax.view_init(elev=20, azim=130)
    return fig, ax

# r-z matrix colourmap: background=black, noise=red, signal=white
_rz_cmap = ListedColormap(["#0d0d0d", "#cc2200", "#ffffff"])
_rz_norm = BoundaryNorm([-0.5, 0.5, 1.5, 3.5], _rz_cmap.N)

def make_rz_matrix(noise, sig, r_bins=100, z_bins=100):
    r_min, r_max = noise["r"].min(), noise["r"].max()
    z_min, z_max = noise["z"].abs().min(), noise["z"].abs().max()
    def to_idx(r, z):
        ri = ((r - r_min) / (r_max - r_min) * (r_bins - 1)).astype(int).clip(0, r_bins - 1)
        zi = ((z - z_min) / (z_max - z_min) * (z_bins - 1)).astype(int).clip(0, z_bins - 1)
        return ri, zi
    m = np.zeros((r_bins, z_bins))
    ri, zi = to_idx(noise["r"].values, noise["z"].abs().values)
    m[ri, zi] = 1
    if len(sig) > 0:
        sri, szi = to_idx(sig["r"].values, sig["z"].abs().values)
        m[sri, szi] = 3
    return m

# ── MODELS ────────────────────────────────────────────────────────────────────
class ConeFilterSNN(nn.Module):
    def __init__(self, input_size=100, h1=64, h2=32, bs=1):
        super().__init__()
        self.lif1 = RadLIFLayer(input_size, h1, bs, normalization="batchnorm", dropout=0.2)
        self.lif2 = RadLIFLayer(h1, h2,     bs, normalization="batchnorm", dropout=0.2)
        self.fc   = nn.Linear(h2, 1)
    def forward(self, x):
        return self.fc(self.lif2(self.lif1(x)).mean(dim=1)).squeeze(-1)

class PhiFinderSNN(nn.Module):
    def __init__(self, h1=64, h2=32, bs=1):
        super().__init__()
        self.lif1 = RadLIFLayer(SNN_INPUT_SIZE, h1, bs, normalization="batchnorm", dropout=0.2)
        self.lif2 = RadLIFLayer(h1, h2,         bs, normalization="batchnorm", dropout=0.2)
        self.fc   = nn.Linear(h2, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.fc(self.lif2(self.lif1(x))).squeeze(-1)

def load_model(cls, path, **kwargs):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = cls(**kwargs)
    m.load_state_dict(ckpt["model_state"])
    return m.eval()

# ── LOAD EVENT + PREPARE CONES ────────────────────────────────────────────────
h, _, p, t = load_event(os.path.join(DATA_DIR, EVENT_ID))
h["r"]   = np.hypot(h["x"], h["y"])
h["phi"] = np.arctan2(h["y"], h["x"])
p_cond, _, _ = preprocess_particles(10, 10, p, h)
print(f"Loaded {EVENT_ID}: {len(h)} hits, {len(p_cond)} signal particles")
sig_hit_ids = set(t[t["particle_id"].isin(p_cond["particle_id"])]["hit_id"].values)

hits_cones_full = split_cones(h)
hits_cones_disp = split_cones(h[np.random.rand(len(h)) < SAMPLE_FRACTION].copy())
sig_cones       = [signal_hits_in_cone(c, t, p) for c in hits_cones_full]
# what the model actually sees: 1% noise + all signal hits (matches training)
baseline_parts  = [model_input(hits_cones_disp[i], sig_cones[i]) for i in range(8)]


model_s1 = load_model(ConeFilterSNN, S1_PATH)
model_s2 = load_model(PhiFinderSNN,  S2_PATH)



def plot_3d(hits, sig_ids, s=1):
    volumes = sorted(hits.volume_id.unique())
    palette = sns.color_palette("rainbow", len(volumes))
    for i, volume in enumerate(volumes):
        v = hits[(hits.volume_id == volume) & (~hits.hit_id.isin(sig_ids))]
        ax.scatter(v.z, v.x, v.y, s=s, alpha=0.9,
                   label=f"volume {volume}", color=palette[i])
    sig = hits[hits.hit_id.isin(sig_ids)]
    ax.scatter(sig.z, sig.x, sig.y, s=s*3, alpha=1,
               color='red', zorder=5, label='signal')
    ax.set_xlim(-1200, 1200); ax.set_ylim(-1200, 1200); ax.set_zlim(-1200, 1200)
    ax.set_xlabel('Z (mm)', labelpad=6, color='#aaaaaa', fontsize=8)
    ax.set_ylabel('Y (mm)', labelpad=6, color='#aaaaaa', fontsize=8)
    ax.set_zlabel('X (mm)', labelpad=6, color='#aaaaaa', fontsize=8)
    ax.tick_params(colors='#cccccc', labelsize=6)
    for pane in (ax.xaxis.pane, ax.yaxis.pane, ax.zaxis.pane):
        pane.set_facecolor((0.97, 0.97, 0.97, 0.1))
        pane.set_edgecolor('#dddddd')
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis._axinfo['grid'].update(color='#dddddd', linewidth=0.4, linestyle='--')

noise_hits    = h[~h.hit_id.isin(sig_hit_ids)]
noise_sampled = noise_hits[np.random.rand(len(noise_hits)) < SAMPLE_FRACTION].copy()
disp          = pd.concat([noise_sampled, h[h.hit_id.isin(sig_hit_ids)]])



# ── PLOT: pipeline summary (hardcoded 500-event results) ──────────────────────
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# ── DATA ──────────────────────────────────────────────────────────────────────
s1_data = {
    'S1 Precision': {'mean': 0.346, 'err': 0.015},
    'S1 Recall':    {'mean': 0.904, 'err': 0.019},
}

s2_data = {
    'S2 Precision': {'mean': 0.653, 'err': 0.018},
    'S2 Recall':    {'mean': 0.768, 'err': 0.021},
}

e2e_data = {
    'Precision':     {'mean': 0.653, 'err': 0.018},
    'Recall':        {'mean': 0.768, 'err': 0.021},
    'Hit reduction': {'mean': 0.946, 'err': 0.003},
}

# ── STYLE ─────────────────────────────────────────────────────────────────────
PREC_COLOR = '#2d5c40'   # dark muted green
REC_COLOR  = '#2b4a72'   # dark muted slate blue
HIT_COLOR  = '#5c3f6b'   # dark muted purple
ERR_COLOR  = 'black'
BAR_HEIGHT = 0.6
GAP        = 0.3

def _bar_color(label):
    l = label.lower()
    if 'precision' in l:
        return PREC_COLOR
    if 'recall' in l:
        return REC_COLOR
    return HIT_COLOR

# ── PANEL a: grouped S1 + S2 ─────────────────────────────────────────────────
def draw_grouped_panel(ax, data_bottom, data_top):
    labels_b = list(data_bottom.keys())
    means_b  = [data_bottom[k]['mean'] for k in labels_b]
    errs_b   = [data_bottom[k]['err']  for k in labels_b]
    y_b      = np.arange(len(labels_b))

    labels_t = list(data_top.keys())
    means_t  = [data_top[k]['mean'] for k in labels_t]
    errs_t   = [data_top[k]['err']  for k in labels_t]
    y_t      = y_b[-1] + 1 + GAP + np.arange(len(labels_t))

    for y_pos, means, errs, labels in [
        (y_b, means_b, errs_b, labels_b),
        (y_t, means_t, errs_t, labels_t),
    ]:
        colors = [_bar_color(l) for l in labels]
        ax.barh(
            y_pos, means,
            xerr=errs,
            height=BAR_HEIGHT,
            color=colors,
            error_kw=dict(ecolor=ERR_COLOR, elinewidth=0.8,
                          capsize=3, capthick=0.8),
            zorder=3,
        )
        for yp, m, e in zip(y_pos, means, errs):
            ax.text(m + e + 0.010, yp, f'{m:.3f}',
                    va='center', ha='left', fontsize=8)

    all_y      = np.concatenate([y_b, y_t])
    all_labels = labels_b + labels_t
    ax.set_yticks(all_y)
    ax.set_yticklabels(all_labels, fontsize=9, ha='left')
    ax.yaxis.set_tick_params(pad=70)

    sep_y = (y_b[-1] + y_t[0]) / 2
    ax.axhline(sep_y, xmin=0, xmax=0.75,
               color='grey', linewidth=0.6,
               linestyle='--', alpha=0.5, zorder=2)

    ax.set_xlim(0, 1.18)
    ax.set_ylim(-0.6, all_y[-1] + 0.6)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis='x', labelsize=8)
    ax.tick_params(top=False, right=False, which='both')
    ax.set_xlabel('Score', fontsize=9)
    ax.grid(axis='x', color='grey', alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── PANEL b: end-to-end ───────────────────────────────────────────────────────
def draw_panel(ax, data):
    labels = list(data.keys())
    means  = [data[k]['mean'] for k in labels]
    errs   = [data[k]['err']  for k in labels]
    y      = np.arange(len(labels))
    colors = [_bar_color(l) for l in labels]

    ax.barh(
        y, means,
        xerr=errs,
        height=BAR_HEIGHT,
        color=colors,
        error_kw=dict(ecolor=ERR_COLOR, elinewidth=0.8,
                      capsize=3, capthick=0.8),
        zorder=3,
    )
    for yp, m, e in zip(y, means, errs):
        ax.text(m + e + 0.010, yp, f'{m:.3f}',
                va='center', ha='left', fontsize=8)

    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9, ha='left')
    ax.yaxis.set_tick_params(pad=70)
    ax.set_xlim(0, 1.18)
    ax.set_ylim(-0.6, y[-1] + 0.6)
    ax.tick_params(top=False, right=False)
    ax.tick_params(which='minor', top=False, right=False)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.2))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))
    ax.tick_params(axis='x', labelsize=8)
    ax.set_xlabel('Score', fontsize=9)
    ax.grid(axis='x', color='grey', alpha=0.25, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)


# ── FIGURE ────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(9, 2.5))
fig.subplots_adjust(wspace=0.55, top=0.95, bottom=0.15)

draw_grouped_panel(axes[0], s2_data, s1_data)
draw_panel(axes[1], e2e_data)

plt.show()








# # ── PANEL 2: 8 r-z matrices ───────────────────────────────────────────────────
# fig, axes = plt.subplots(2, 4, figsize=(16, 8))
# for i, (noise, sig, ax) in enumerate(zip(hits_cones_disp, sig_cones, axes.flat)):
#     ax.imshow(make_rz_matrix(noise, sig), aspect="auto", origin="lower",
#               cmap=_rz_cmap, norm=_rz_norm)
#     ax.tick_params(labelsize=6)
# fig.legend(
#     handles=[Patch(facecolor="#555555", label="noise"),
#              Patch(facecolor="#cc2200", label="signal")],
#     loc="lower center", ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.02),
# )
# plt.tight_layout(); plt.show()









# ── STAGE 1: classify each octant ────────────────────────────────────────────
print(f"\nStage 1 — {EVENT_ID}\n" + "─" * 46)
predicted_signal_idx = []
for i, ((num, lbl, _), sig) in enumerate(zip(OCTANTS, sig_cones)):
    inp  = model_input(hits_cones_disp[i], sig)
    grid = torch.tensor(make_rz_grid(inp["r"].values, inp["z"].values),
                        dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        conf = torch.sigmoid(model_s1(grid)).item()
    flag  = conf > THRESH_S1
    truth = "(sig)" if len(sig) > 0 else "(noi)"
    print(f"  Oct {num}  {conf*100:5.1f}%  {'SIGNAL' if flag else 'noise ':6s}  {truth}")
    if flag:
        predicted_signal_idx.append(i)





# ── PANEL 2.5: r-z matrices with S1 confidence ───────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
for i, (noise, sig, ax) in enumerate(zip(hits_cones_disp, sig_cones, axes.flat)):
    inp  = model_input(hits_cones_disp[i], sig)
    grid = torch.tensor(make_rz_grid(inp["r"].values, inp["z"].values),
                        dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        conf = torch.sigmoid(model_s1(grid)).item()
    passed = conf > THRESH_S1
    ax.imshow(make_rz_matrix(noise, sig), aspect="auto", origin="lower",
              cmap=_rz_cmap, norm=_rz_norm)
    for spine in ax.spines.values():
        spine.set_edgecolor("#44aa44" if passed else "#cc2200")
        spine.set_linewidth(3)
    ax.set_title(f"{conf*100:.0f}%  {'✓' if passed else '✗'}",
                 fontsize=9, color="#44aa44" if passed else "#cc2200")
    ax.tick_params(labelsize=6)
plt.tight_layout()
plt.show()




# ── PANEL 3: surviving octants after S1, coloured by volume, signal in red ────
surviving_hits = pd.concat([baseline_parts[i] for i in predicted_signal_idx])

fig, ax = new_3d()
plot_3d(surviving_hits, sig_hit_ids, s=0.2)
plt.tight_layout()
plt.show()




# ── STAGE 2: predict phi bins for each passing octant ────────────────────────
print(f"\nStage 2 — {EVENT_ID}\n" + "─" * 46)
octant_phi_results = {}
all_phi_probs_plot = {}

for i in predicted_signal_idx:
    num, _, (sx, sy, sz) = OCTANTS[i]
    inp = model_input(hits_cones_disp[i], sig_cones[i])
    rphi_grid = torch.tensor(
        make_rphi_grid(inp["r"].values, inp["phi"].values, sx, sy),
        dtype=torch.float32,
    ).unsqueeze(0)
    with torch.no_grad():
        phi_probs = torch.sigmoid(model_s2(rphi_grid)).squeeze(0).numpy()
    pred_bins = np.where(phi_probs > THRESH_S2)[0]
    phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
    angles_deg = np.degrees(
        phi_min + (pred_bins + 0.5) * (phi_max - phi_min) / SNN_PHI_BINS
    )
    octant_phi_results[i]  = {"bins": pred_bins, "probs": phi_probs[pred_bins]}
    all_phi_probs_plot[i]  = {"full_probs": phi_probs, "phi_min": phi_min,
                               "phi_max": phi_max, "num": num}
    print(f"  Oct {num}  [{np.degrees(phi_min):.0f}°,{np.degrees(phi_max):.0f}°]  "
          f"{len(pred_bins)}/{SNN_PHI_BINS} bins signal")
    for b, ang, pr in zip(pred_bins, angles_deg, phi_probs[pred_bins]):
        print(f"    bin {b:3d}  {ang:+7.2f}°  {pr*100:.1f}%")

# ── PANEL 3.5: phi predictions in 4x4 grid ───────────────────────────────────
fig, axes = plt.subplots(2, 4, figsize=(16, 8))
axes_flat = axes.flat

for idx, (i, data) in enumerate(all_phi_probs_plot.items()):
    ax = axes_flat[idx]
    phi_probs = data["full_probs"]
    phi_min   = data["phi_min"]
    phi_max   = data["phi_max"]
    num       = data["num"]
    _, _, (sx, sy, _) = OCTANTS[i]

    # true active bins
    true_active = np.zeros(SNN_PHI_BINS, dtype=bool)
    if len(sig_cones[i]) > 0:
        idx_bins = phi_bin_idx(sig_cones[i], phi_min, phi_max)
        true_active[idx_bins] = True

    ax.fill_between(range(SNN_PHI_BINS), 0, true_active.astype(float),
                    alpha=0.3, color="#EE7733", label="True bins")
    ax.plot(phi_probs, color="#0077BB", linewidth=1.2, label="Predicted p")
    ax.axhline(THRESH_S2, color="black", lw=0.8, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xlabel("$\\phi$ bin", fontsize=8)
    ax.set_ylabel("p", fontsize=8)
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.tick_params(labelsize=7)
    if idx == 0:
        ax.legend(fontsize=7)

# hide unused axes
for idx in range(len(all_phi_probs_plot), 8):
    axes_flat[idx].set_visible(False)

plt.tight_layout()
plt.show()



# ── ACCURACY CHECK — baseline matches training: 1% noise + all signal hits ────
true_sig_ids = set(t[t["particle_id"].isin(p_cond["particle_id"])]["hit_id"].values)

baseline_all = pd.concat(baseline_parts)
n_baseline     = len(baseline_all)
n_true         = baseline_all["hit_id"].isin(true_sig_ids).sum()

s1_hits  = pd.concat([baseline_parts[i] for i in predicted_signal_idx])
n_s1_sig = s1_hits["hit_id"].isin(true_sig_ids).sum()

s2_parts = []
for i in predicted_signal_idx:
    _, _, (sx, sy, _) = OCTANTS[i]
    cone     = baseline_parts[i]
    bins     = octant_phi_results[i]["bins"]
    if not len(bins):
        continue
    phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
    s2_parts.append(cone[np.isin(phi_bin_idx(cone, phi_min, phi_max), bins)])

s2_hits  = pd.concat(s2_parts) if s2_parts else pd.DataFrame(columns=h.columns)
n_s2_sig = s2_hits["hit_id"].isin(true_sig_ids).sum() if len(s2_hits) else 0

def _row(label, total, sig):
    recall = f"{100*sig/n_true:.1f}%" if n_true > 0 else "n/a"
    prec   = f"{100*sig/total:.1f}%"  if total  > 0 else "n/a"
    return f"  {label}  kept {total:6d}  signal {sig}/{n_true} ({recall} recall)  prec {prec}"

print(f"\nAccuracy (1% sample) — {EVENT_ID}\n" + "─" * 46)
if n_true == 0:
    print(f"  Sample  kept {n_baseline:6d}  signal 0 (none in 1% sample — chance miss)")
else:
    print(f"  Sample  kept {n_baseline:6d}  signal {n_true} (baseline)")
print(_row("S1", len(s1_hits), n_s1_sig))
print(_row("S2", len(s2_hits), n_s2_sig))





# ── PANEL 4: phi-filtered hits, coloured by volume, signal in red ─────────────
all_kept_parts = []
for i in predicted_signal_idx:
    _, _, (sx, sy, _) = OCTANTS[i]
    cone     = baseline_parts[i]
    bins     = octant_phi_results[i]["bins"]
    if not len(bins):
        continue
    phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
    kept = cone[np.isin(phi_bin_idx(cone, phi_min, phi_max), bins)]
    all_kept_parts.append(kept)

final_hits = pd.concat(all_kept_parts) if all_kept_parts else pd.DataFrame(columns=h.columns)

fig, ax = new_3d()
plot_3d(final_hits, sig_hit_ids, s=0.5)
plt.tight_layout()
plt.show()


# ── FINAL SUMMARY ─────────────────────────────────────────────────────────────
n_total_hits = n_baseline
n_final_hits = len(s2_hits)
hit_reduction = 100 * (1 - n_final_hits / n_total_hits) if n_total_hits > 0 else 0

# Stage 1 cone-level metrics
n_sig_cones   = sum(1 for s in sig_cones if len(s) > 0)
n_pass_cones  = len(predicted_signal_idx)
n_sig_passed  = sum(1 for i in predicted_signal_idx if len(sig_cones[i]) > 0)
n_noise_cones = 8 - n_sig_cones
n_noise_pass  = n_pass_cones - n_sig_passed
s1_efficiency = 100 * n_sig_passed / n_sig_cones if n_sig_cones > 0 else 0
s1_throwrate  = 100 * (1 - n_noise_pass / n_noise_cones) if n_noise_cones > 0 else 0

# Stage 2 IoU across passing signal cones
ious = []
for i in predicted_signal_idx:
    if len(sig_cones[i]) == 0:
        continue
    _, _, (sx, sy, _) = OCTANTS[i]
    phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
    true_active = np.zeros(SNN_PHI_BINS, dtype=bool)
    true_active[phi_bin_idx(sig_cones[i], phi_min, phi_max)] = True
    pred_active = np.zeros(SNN_PHI_BINS, dtype=bool)
    pred_active[octant_phi_results[i]["bins"]] = True
    intersection = (true_active & pred_active).sum()
    union        = (true_active | pred_active).sum()
    if union > 0:
        ious.append(intersection / union)

mean_iou = 100 * np.mean(ious) if ious else 0

# S2 precision across all predicted bins
total_pred    = sum(len(octant_phi_results[i]["bins"]) for i in predicted_signal_idx)
total_correct = 0
for i in predicted_signal_idx:
    if len(sig_cones[i]) == 0:
        continue
    _, _, (sx, sy, _) = OCTANTS[i]
    phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
    true_active = np.zeros(SNN_PHI_BINS, dtype=bool)
    true_active[phi_bin_idx(sig_cones[i], phi_min, phi_max)] = True
    total_correct += true_active[octant_phi_results[i]["bins"]].sum()
s2_precision = 100 * total_correct / total_pred if total_pred > 0 else 0

print(f"\n{'='*50}")
print(f"PIPELINE SUMMARY — {EVENT_ID}")
print(f"{'='*50}")
print(f"  Total hits (baseline)    : {n_total_hits}")
print(f"  Final hits (after S2)    : {n_final_hits}")
print(f"  Hit reduction            : {hit_reduction:.1f}%")
print(f"{'─'*50}")
print(f"  S1 signal efficiency     : {s1_efficiency:.1f}%  ({n_sig_passed}/{n_sig_cones} signal cones)")
print(f"  S1 noise throwrate       : {s1_throwrate:.1f}%  ({n_noise_cones-n_noise_pass}/{n_noise_cones} noise cones rejected)")
print(f"{'─'*50}")
print(f"  S2 mean IoU              : {mean_iou:.1f}%")
print(f"  S2 precision             : {s2_precision:.1f}%")
print(f"{'─'*50}")
print(f"  Signal hit recall (S1)   : {100*n_s1_sig/n_true:.1f}%  ({n_s1_sig}/{n_true})")
print(f"  Signal hit recall (S2)   : {100*n_s2_sig/n_true:.1f}%  ({n_s2_sig}/{n_true})")
print(f"{'='*50}")

print(f"\nEND-TO-END PIPELINE")
print(f"{'─'*50}")
print(f"  Input hits               : {n_total_hits}")
print(f"  After Stage 1            : {len(s1_hits)}  ({100*len(s1_hits)/n_total_hits:.1f}% of input)")
print(f"  After Stage 2            : {n_final_hits}  ({100*n_final_hits/n_total_hits:.1f}% of input)")
print(f"  Signal hits in           : {n_true}")
print(f"  Signal hits out          : {n_s2_sig}  ({100*n_s2_sig/n_true:.1f}% recall)")
print(f"  Noise hits in            : {n_total_hits - n_true}")
print(f"  Noise hits out           : {n_final_hits - n_s2_sig}")
print(f"  Hit reduction            : {hit_reduction:.1f}%")
print(f"  Signal hit recall        : {100*n_s2_sig/n_true:.1f}%")
print(f"  Final precision          : {100*n_s2_sig/n_final_hits:.1f}%")
print(f"{'='*50}")


import time

# ── TIMING START ──────────────────────────────────────────────────────────────
tik = time.perf_counter()

# ── STAGE 1: classify each octant ────────────────────────────────────────────
predicted_signal_idx = []
for i, ((num, lbl, _), sig) in enumerate(zip(OCTANTS, sig_cones)):
    inp  = model_input(hits_cones_disp[i], sig)
    grid = torch.tensor(make_rz_grid(inp["r"].values, inp["z"].values),
                        dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        conf = torch.sigmoid(model_s1(grid)).item()
    if conf > THRESH_S1:
        predicted_signal_idx.append(i)

# ── STAGE 2: predict phi bins for each passing octant ────────────────────────
octant_phi_results = {}
s2_parts = []
for i in predicted_signal_idx:
    _, _, (sx, sy, _) = OCTANTS[i]
    inp = model_input(hits_cones_disp[i], sig_cones[i])
    rphi_grid = torch.tensor(
        make_rphi_grid(inp["r"].values, inp["phi"].values, sx, sy),
        dtype=torch.float32,
    ).unsqueeze(0)
    with torch.no_grad():
        phi_probs = torch.sigmoid(model_s2(rphi_grid)).squeeze(0).numpy()
    pred_bins = np.where(phi_probs > THRESH_S2)[0]
    octant_phi_results[i] = {"bins": pred_bins, "probs": phi_probs[pred_bins]}
    phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
    cone = hits_cones_disp[i]
    if len(pred_bins):
        s2_parts.append(cone[np.isin(phi_bin_idx(cone, phi_min, phi_max), pred_bins)])

final_hits = pd.concat(s2_parts) if s2_parts else pd.DataFrame(columns=h.columns)

# ── TIMING END ────────────────────────────────────────────────────────────────
tok = time.perf_counter()

print(f"\n{'='*50}")
print(f"PIPELINE TIMING — {EVENT_ID}")
print(f"{'='*50}")
print(f"  Total inference time     : {tok - tik:.4f} s")
print(f"  Input hits               : {len(hits_cones_disp[0]) * 8}")
print(f"  Output hits              : {len(final_hits)}")
print(f"{'='*50}")