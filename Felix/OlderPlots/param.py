import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Felix\SNN")
import torch
import numpy as np
from snns import RadLIFLayer
import torch.nn as nn
import matplotlib.pyplot as plt
import os

import scienceplots  # must be imported to register the "science" style
plt.style.use(["science", "no-latex"])

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

plt.style.use(["science", "no-latex"])
plt.rcParams.update({
    "text.usetex":       False,
    "figure.dpi":        150,
    "font.size":         10,
    "axes.titlesize":    11,
    "axes.labelsize":    10,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "legend.fontsize":   8,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "figure.facecolor":  "white",
    "axes.facecolor":    "white",
})
C = ["#0077BB", "#EE7733", "#009988", "#CC3311", "#33BBEE", "#EE3377"]

# ── Paths ─────────────────────────────────────────────────────────────────────
Save_dir = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices100"

# ── Load checkpoint ───────────────────────────────────────────────────────────
checkpoint = torch.load(os.path.join(Save_dir, "model_stage1.pt"))
state      = checkpoint['model_state']

# ── Model definition (must match training) ────────────────────────────────────
class TrackFinderSNN(nn.Module):
    def __init__(self, input_size, h1=256, h2=128, bs=1):
        super().__init__()
        self.lif1 = RadLIFLayer(input_size, h1, bs, normalization='batchnorm', dropout=0.0)
        self.lif2 = RadLIFLayer(h1, h2,     bs, normalization='batchnorm', dropout=0.0)
        self.fc   = nn.Linear(h2, 1)

    def forward(self, x):
        return self.fc(self.lif2(self.lif1(x)).mean(dim=1)).squeeze(-1)

    def set_bs(self, bs):
        self.lif1.batch_size = bs
        self.lif2.batch_size = bs

model = TrackFinderSNN(100, bs=1)
model.load_state_dict(state)
model.eval()
model.set_bs(1)

# ── Load data ─────────────────────────────────────────────────────────────────
X = np.load(os.path.join(Save_dir, "X_rz.npy"))       # stage 1 input
y = np.load(os.path.join(Save_dir, "y_stage1.npy"))   # stage 1 labels

# ── 1. Learned parameter distributions ───────────────────────────────────────
params_to_plot = {
    'lif1.alpha': r'Layer 1 — membrane decay $\alpha$',
    'lif1.beta':  r'Layer 1 — adapt. decay $\beta$',
    'lif1.a':     r'Layer 1 — coupling $a$',
    'lif1.b':     r'Layer 1 — spike adapt. $b$',
    'lif2.alpha': r'Layer 2 — membrane decay $\alpha$',
    'lif2.beta':  r'Layer 2 — adapt. decay $\beta$',
    'lif2.a':     r'Layer 2 — coupling $a$',
    'lif2.b':     r'Layer 2 — spike adapt. $b$',
}

fig, axes = plt.subplots(2, 4, figsize=(14, 5))
for ax, (key, title) in zip(axes.flat, params_to_plot.items()):
    values = state[key].numpy()
    ax.hist(values, bins=30, color=C[0], edgecolor="white", linewidth=0.4)
    ax.axvline(values.mean(),   color=C[1], linestyle="--", linewidth=1.2,
               label=f"mean={values.mean():.3f}")
    ax.axvline(values.median() if hasattr(values, 'median') else np.median(values),
               color=C[2], linestyle=":", linewidth=1.2,
               label=f"med={np.median(values):.3f}")
    ax.set_title(title)
    ax.set_xlabel("value")
    ax.set_ylabel("count")
    ax.legend(fontsize=7)

plt.suptitle("Learned RadLIF parameters — Stage 1 (1000 events)", fontsize=12)
plt.tight_layout()
plt.show()

# ── 2. Parameter interpretation printout ─────────────────────────────────────
print("\n=== RadLIF Parameter Summary ===")
print(f"{'Parameter':<15} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}  Interpretation")
print("-" * 80)
interp = {
    'lif1.alpha': "Memory: how much past potential is retained (0=none, 1=full)",
    'lif1.beta':  "Adapt. memory: how long threshold adaptation persists",
    'lif1.a':     "Subthreshold coupling: u feeds into adapt. variable w",
    'lif1.b':     "Spike-triggered adapt.: how much each spike raises threshold",
    'lif2.alpha': "Layer 2 membrane decay",
    'lif2.beta':  "Layer 2 adaptation decay",
    'lif2.a':     "Layer 2 subthreshold coupling",
    'lif2.b':     "Layer 2 spike adaptation",
}
for key, label in interp.items():
    v = state[key].numpy()
    print(f"{key:<15} {v.mean():>8.3f} {v.std():>8.3f} {v.min():>8.3f} {v.max():>8.3f}  {label}")

# ── 3. Forward pass — pick one signal and one background example ──────────────
sig_idx  = np.where(y == 1.0)[0][0]
bg_idx   = np.where(y == 0.0)[0][0]

def run_layer1(model, x_tensor):
    """Manually unroll layer 1 to record membrane potential and spikes."""
    with torch.no_grad():
        Wx = model.lif1.W(x_tensor)
        if model.lif1.normalize:
            _Wx = model.lif1.norm(Wx.reshape(Wx.shape[0]*Wx.shape[1], Wx.shape[2]))
            Wx  = _Wx.reshape(Wx.shape)
        ut = torch.zeros(1, model.lif1.hidden_size)
        wt = torch.zeros(1, model.lif1.hidden_size)
        st = torch.zeros(1, model.lif1.hidden_size)
        alpha = torch.clamp(model.lif1.alpha, *model.lif1.alpha_lim)
        beta  = torch.clamp(model.lif1.beta,  *model.lif1.beta_lim)
        a     = torch.clamp(model.lif1.a,     *model.lif1.a_lim)
        b     = torch.clamp(model.lif1.b,     *model.lif1.b_lim)
        V     = model.lif1.V.weight.clone().fill_diagonal_(0)
        ut_hist, st_hist, wt_hist = [], [], []
        for t in range(Wx.shape[1]):
            wt = beta * wt + a * ut + b * st
            ut = alpha * (ut - st) + (1 - alpha) * (Wx[:, t, :] + torch.matmul(st, V) - wt)
            st = (ut > model.lif1.threshold).float()
            ut_hist.append(ut.squeeze(0).numpy())
            st_hist.append(st.squeeze(0).numpy())
            wt_hist.append(wt.squeeze(0).numpy())
    return np.array(ut_hist), np.array(st_hist), np.array(wt_hist)

x_sig = torch.tensor(X[sig_idx]).unsqueeze(0)
x_bg  = torch.tensor(X[bg_idx]).unsqueeze(0)

ut_sig, st_sig, wt_sig = run_layer1(model, x_sig)
ut_bg,  st_bg,  wt_bg  = run_layer1(model, x_bg)



# ── 5. Membrane potential for a few selective neurons ─────────────────────────
def plot_membrane(ut_hist, st_hist, model, title_prefix, color_offset=0):
    spike_counts   = st_hist.sum(axis=0)
    sparse_mask    = (spike_counts >= 1) & (spike_counts <= 5)
    sparse_neurons = np.where(sparse_mask)[0]
    if len(sparse_neurons) == 0:
        print(f"No sparse neurons found for {title_prefix}")
        return
    chosen = sparse_neurons[np.linspace(0, len(sparse_neurons)-1,
                                         min(5, len(sparse_neurons)),
                                         dtype=int)]
    T_steps   = ut_hist.shape[0]
    t_ax      = np.arange(T_steps)
    threshold = model.lif1.threshold

    fig, axes = plt.subplots(2, 1, figsize=(12, 5), sharex=True,
                             gridspec_kw={'height_ratios': [2, 1]})

    # ── subtle blue-grey background, black frame ──────────────────────────────
    for ax in axes:
        ax.set_facecolor('#F0F4F8')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color('black')
            spine.set_linewidth(0.8)

    # ── membrane potential panel ──────────────────────────────────────────────
    axes[0].grid(True, alpha=0.5, linestyle='--', linewidth=0.5,
                 color='#C0C8D0', zorder=0)
    for col, nidx in enumerate(chosen):
        axes[0].plot(t_ax, ut_hist[:, nidx],
                     color=C[(col+color_offset) % len(C)],
                     linewidth=1.2, label=f'n{nidx}', zorder=2)
    axes[0].axhline(threshold, color='black', linestyle='--',
                    linewidth=0.8, alpha=0.6, label='threshold', zorder=3)
    axes[0].set_ylabel('u(t)')
    axes[0].legend(loc='upper right', ncol=len(chosen)+1)

    # ── spike raster panel ────────────────────────────────────────────────────
    axes[1].grid(True, alpha=0.5, linestyle='--', linewidth=0.5,
                 color='#C0C8D0', zorder=0)
    for col, nidx in enumerate(chosen):
        spike_times = np.where(st_hist[:, nidx] == 1)[0]
        axes[1].scatter(spike_times, np.full_like(spike_times, col),
                        marker='|', s=250, linewidths=2.0,
                        color=C[(col+color_offset) % len(C)], zorder=2)
    axes[1].set_yticks(range(len(chosen)))
    axes[1].set_yticklabels([f'n{nidx}' for nidx in chosen])
    axes[1].set_ylabel('Neuron')
    axes[1].set_xlabel('r timestep')
    axes[1].set_ylim(-0.5, len(chosen)-0.5)

    plt.tight_layout()
    plt.show()

plot_membrane(ut_sig, st_sig, model, "Signal")

