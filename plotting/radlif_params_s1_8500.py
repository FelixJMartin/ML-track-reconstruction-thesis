# radlif_params_s1_8500.py
# Visualize learned parameters of model_s1_h64_32.pt

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import scienceplots

plt.style.use(["science", "no-latex"])
plt.rcParams["text.usetex"] = False

MODEL_PATH = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\s1_8500_ep115.pt"

# ── Load ────────────────────────────────────────────────────────────────────
ckpt = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
sd   = ckpt["model_state"]
cfg  = ckpt["config"]

train_losses = np.array(ckpt["train_losses"])
val_losses   = np.array(ckpt["val_losses"])
val_accs     = np.array(ckpt["val_accs"])
best_ep      = ckpt["best_epoch"]
epochs       = np.arange(1, len(train_losses) + 1)

# Per-neuron learnable scalars (sigmoid-ed in sparch — values are raw logits)
# Plotting raw values directly; note sparch applies sigmoid to alpha/beta in forward
p = {
    "alpha1": sd["lif1.alpha"].numpy(),
    "beta1":  sd["lif1.beta"].numpy(),
    "a1":     sd["lif1.a"].numpy(),
    "b1":     sd["lif1.b"].numpy(),
    "alpha2": sd["lif2.alpha"].numpy(),
    "beta2":  sd["lif2.beta"].numpy(),
    "a2":     sd["lif2.a"].numpy(),
    "b2":     sd["lif2.b"].numpy(),
}

W1 = sd["lif1.W.weight"].numpy()   # (64, 100)
V1 = sd["lif1.V.weight"].numpy()   # (64, 64)
W2 = sd["lif2.W.weight"].numpy()   # (32, 64)
V2 = sd["lif2.V.weight"].numpy()   # (32, 32)
fc = sd["fc.weight"].numpy()[0]    # (32,)


def plot_params():
    """2×4 grid: rows = layers, columns = α β a b. Matches PARAM.pdf style."""
    param_meta = [
        ("alpha", r"membrane decay $\alpha$"),
        ("beta",  r"adapt. decay $\beta$"),
        ("a",     r"coupling $a$"),
        ("b",     r"spike adapt. $b$"),
    ]
    layers = [
        ("lif1", "Layer 1"),
        ("lif2", "Layer 2"),
    ]

    fig, axes = plt.subplots(
        2, 4,
        figsize=(16, 8),
        gridspec_kw={"hspace": 0.38, "wspace": 0.28},
    )

    for row, _ in enumerate(layers):
        for col, (pkey, plabel) in enumerate(param_meta):
            ax  = axes[row, col]
            v   = p[f"{pkey}{row + 1}"]
            mn  = v.mean()
            med = np.median(v)

            ax.hist(v, bins=30, color="steelblue", alpha=0.75, edgecolor="none")
            ax.axvline(mn,  color="black", lw=1.0, linestyle="--",
                       label=f"mean={mn:.3f}")
            ax.axvline(med, color="#1a7a1a", lw=1.0, linestyle=":",
                       label=f"med={med:.3f}")
            ax.legend(fontsize=6.5, frameon=False, handlelength=1.5)
            ax.grid(True, color="#cccccc", linewidth=0.6, linestyle="--", zorder=0)
            ax.set_axisbelow(True)

            # column titles only on top row
            if row == 0:
                ax.set_title(plabel, fontsize=10)

            # x label only on bottom row
            if row == 1:
                ax.set_xlabel("value", fontsize=8)

            ax.set_ylabel("count", fontsize=8)


    plt.show()


def plot_model():
    plot_params()


if __name__ == "__main__":
    plot_model()
