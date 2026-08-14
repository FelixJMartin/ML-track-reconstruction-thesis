# lif_vs_radlif_theory.py
# Theory visualizations using exact sparch RadLIF dynamics

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import scienceplots

plt.style.use(["science", "no-latex"])
plt.rcParams["text.usetex"] = False

# ── Parameters ─────────────────────────────────────────────────────────────
alpha     = 0.9
beta      = 0.85
a         = 0.1
b         = 1.5
v_rec     = 0.3
threshold = 1.0
T         = 300

# ── Input ───────────────────────────────────────────────────────────────────
I = torch.zeros(T)
I[20:70]   = 0.3
I[100:150] = 0.3
I[190:240] = 1.8

def run_lif(I, alpha, threshold):
    ut, st = 0.0, 0.0
    mem, spk = [], []
    for t in range(len(I)):
        ut = alpha * (ut - st) + (1 - alpha) * I[t].item()
        st = float(ut >= threshold)
        mem.append(ut)
        spk.append(st)
    return np.array(mem), np.array(spk)

def run_radlif(I, alpha, beta, a, b, v_rec, threshold):
    ut, wt, st = 0.0, 0.0, 0.0
    mem, spk, adapt = [], [], []
    for t in range(len(I)):
        wt = beta * wt + a * ut + b * st
        ut = alpha * (ut - st) + (1 - alpha) * (
            I[t].item() + v_rec * st - wt
        )
        st = float(ut >= threshold)
        mem.append(ut)
        spk.append(st)
        adapt.append(wt)
    return np.array(mem), np.array(spk), np.array(adapt)

def plot_lif_vs_radlif():
    mem_lif, spk_lif            = run_lif(I, alpha, threshold)
    mem_rad, spk_rad, adapt_rad = run_radlif(I, alpha, beta, a, b, v_rec, threshold)
    t = np.arange(T)

    fig = plt.figure(figsize=(16, 9))
    gs  = gridspec.GridSpec(3, 2, hspace=0.12, wspace=0.4,
                            height_ratios=[0.6, 2.5, 0.8])

    # ── Row 0: Input current ──────────────────────────────────────────────
    for col in range(2):
        ax = fig.add_subplot(gs[0, col])
        ax.plot(t, I.numpy(), color='tab:orange', lw=1.2)
        ax.set_ylabel('$I_{in}$')
        ax.set_xlim([0, T])
        ax.set_ylim([-0.05, 0.65])
        ax.tick_params(labelbottom=False)
        ax.set_title('LIF' if col == 0 else 'RadLIF',
                     fontsize=12, fontweight='bold')

        # annotate the three pulses on input (same for both)
        if col == 0:
            ax.text(35, 0.52, 'weak', fontsize=7, ha='center', color='grey')
            ax.text(120, 0.52, 'weak', fontsize=7, ha='center', color='grey')
            ax.text(212, 0.52, 'strong', fontsize=7, ha='center', color='tomato')

    # ── Row 1: Membrane potential ─────────────────────────────────────────
    # LIF
    ax_lif = fig.add_subplot(gs[1, 0])
    ax_lif.plot(t, mem_lif, color='steelblue', lw=1.2)
    ax_lif.axhline(threshold, color='black', linestyle='--', alpha=0.35, lw=1.2)
    ax_lif.set_ylabel('$U_{mem}$')
    ax_lif.set_xlim([0, T])
    ax_lif.tick_params(labelbottom=False)

    # LIF annotations
    ax_lif.annotate(r'$\alpha$: membrane decay',
                    xy=(67, 0.18), xytext=(80, 0.5),
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1),
                    fontsize=8, color='steelblue')

    ax_lif.annotate(r'$\vartheta$ = threshold',
                    xy=(5, threshold), xytext=(5, 1.08),
                    fontsize=8, color='dimgrey')

    ax_lif.annotate('sub-threshold\n(no spike)',
                    xy=(45, 0.30), xytext=(45, 0.65),
                    arrowprops=dict(arrowstyle='->', color='grey', lw=0.8),
                    fontsize=7, color='grey', ha='center')

    ax_lif.annotate('fires!',
                    xy=(200, 1.02), xytext=(200, 1.25),
                    arrowprops=dict(arrowstyle='->', color='tomato', lw=1),
                    fontsize=8, color='tomato', ha='center')

    # RadLIF
    ax_rad = fig.add_subplot(gs[1, 1])
    ax_rad.plot(t, mem_rad, color='steelblue', lw=1.2, label='$u[t]$')
    ax_rad.plot(t, adapt_rad, color='tomato', lw=0.9,
                linestyle='--', alpha=0.8, label='$w[t]$')
    ax_rad.axhline(threshold, color='black', linestyle='--', alpha=0.35, lw=1.2)
    ax_rad.set_ylabel('$U_{mem}$')
    ax_rad.set_xlim([0, T])
    ax_rad.tick_params(labelbottom=False)
    ax_rad.legend(fontsize=8, frameon=False, loc='upper left')

    # RadLIF annotations
    ax_rad.annotate(r'$\alpha$: membrane decay',
                    xy=(67, 0.15), xytext=(75, 0.5),
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1),
                    fontsize=8, color='steelblue')

    ax_rad.annotate(r'$\beta$: adaptation decay',
                    xy=(155, adapt_rad[155]), xytext=(160, 0.6),
                    arrowprops=dict(arrowstyle='->', color='tomato', lw=1),
                    fontsize=8, color='tomato')

    ax_rad.annotate(r'$b$: spike-triggered increment',
                    xy=(200, adapt_rad[200]), xytext=(210, 1.6),
                    arrowprops=dict(arrowstyle='->', color='tomato', lw=1),
                    fontsize=8, color='tomato')

    ax_rad.annotate(r'$a$: subthreshold coupling',
                    xy=(120, adapt_rad[120]), xytext=(30, 0.35),
                    arrowprops=dict(arrowstyle='->', color='tomato', lw=1),
                    fontsize=8, color='tomato')

    ax_rad.annotate('adaptation\nsuppresses firing',
                    xy=(220, 0.3), xytext=(240, 0.8),
                    arrowprops=dict(arrowstyle='->', color='grey', lw=0.8),
                    fontsize=7, color='grey', ha='center')

    ax_rad.annotate(r'$V$: recurrent weights',
                    xy=(195, mem_rad[195]), xytext=(150, -0.3),
                    arrowprops=dict(arrowstyle='->', color='steelblue', lw=1),
                    fontsize=8, color='steelblue')

    # ── Row 2: Output spikes ──────────────────────────────────────────────
    for col, spk in enumerate([spk_lif, spk_rad]):
        ax = fig.add_subplot(gs[2, col])
        spike_times = np.where(spk > 0)[0]
        ax.vlines(spike_times, 0, 1, color='black', lw=1.5)
        ax.set_xlim([0, T])
        ax.set_ylim([0, 1.5])
        ax.set_ylabel('Output spikes')
        ax.set_yticks([])
        ax.set_xlabel('Time step')

        if col == 0:
            ax.text(210, 1.1, 'many spikes\n(no fatigue)',
                    fontsize=7, color='grey', ha='center')
        else:
            ax.text(215, 1.1, 'fewer spikes\n(adapted)',
                    fontsize=7, color='grey', ha='center')

    plt.savefig('lif_vs_radlif.pdf', bbox_inches='tight', dpi=300)
    plt.savefig('lif_vs_radlif.svg', bbox_inches='tight')  # for Figma
    plt.show()

if __name__ == "__main__":
    plot_lif_vs_radlif()