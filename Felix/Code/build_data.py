# ================================================================================
# build_data.py
# Builds all training datasets for stage 1 (r-z) and stage 2 (r-phi).
# Run this once to generate all .npy files, then run stage1_train.py / stage2_train.py
# ================================================================================

from trackml.dataset import load_event
import os
import numpy as np
import pandas as pd
from Functions import preprocess_particles

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── CONFIG ───────────────────────────────────────────────────────────────────
SAMPLE_FRACTION = 0.01
N_AUGMENT       = 2      # noise shuffle augmentations per track — keeps 50/50 naturally
SNN_R_BINS      = 100
SNN_Z_BINS      = 100
SNN_PHI_BINS    = 100
SNN_R_MAX       = 1100.0
SNN_Z_MAX       = 3000.0

data_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_1"
Save_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices1000"
event_ids = [f"event{i:09d}" for i in range(1000, 2800)]  # 1800 train events

# ── CONE DEFINITIONS ─────────────────────────────────────────────────────────
CONE_SIGNS = [
    (+1,+1,+1), (+1,+1,-1), (+1,-1,+1), (+1,-1,-1),
    (-1,+1,+1), (-1,+1,-1), (-1,-1,+1), (-1,-1,-1)
]

CONE_PHI_RANGES = {
    (+1,+1): (0,          np.pi/2),
    (+1,-1): (-np.pi/2,   0),
    (-1,+1): (np.pi/2,    np.pi),
    (-1,-1): (-np.pi,     -np.pi/2),
}


# ── GRID FUNCTIONS ───────────────────────────────────────────────────────────
def make_rz_grid(hit_r, hit_z):
    """R-z projection: rows=r, cols=z. Uses measured x,y,z positions."""
    spikematrix = np.zeros((SNN_R_BINS, SNN_Z_BINS), dtype=np.float32)
    ri = np.clip((hit_r / SNN_R_MAX * SNN_R_BINS).astype(int), 0, SNN_R_BINS - 1)
    zi = np.clip((np.abs(hit_z) / SNN_Z_MAX * SNN_Z_BINS).astype(int), 0, SNN_Z_BINS - 1)
    spikematrix[ri, zi] = 1.0
    return spikematrix


def make_rphi_grid(hit_r, hit_phi, sx, sy):
    """
    R-phi occupancy grid with one extra summary timestep appended.
    Rows 0..SNN_R_BINS-1 : binary occupancy — 1 if any hit in (r, phi) bin
    Row  SNN_R_BINS      : column occupancy fraction — how many r-bins
                           are occupied per phi bin, normalised by SNN_R_BINS.
                           This gives the SNN an explicit summary of r-layer
                           continuity per phi bin, making tracks easier to
                           distinguish from noise.
    Output shape: (SNN_R_BINS + 1, SNN_PHI_BINS)
    """
    phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]

    g    = np.zeros((SNN_R_BINS, SNN_PHI_BINS), dtype=np.float32)
    ri   = np.clip((hit_r / SNN_R_MAX * SNN_R_BINS).astype(int),
                   0, SNN_R_BINS - 1)
    phii = np.clip(
        ((hit_phi - phi_min) / (phi_max - phi_min) * SNN_PHI_BINS).astype(int),
        0, SNN_PHI_BINS - 1
    )
    g[ri, phii] = 1.0
    
    # column occupancy: fraction of r-layers occupied per phi bin
    col_occ = (g.sum(axis=0) / SNN_R_BINS)[np.newaxis, :]  # (1, PHI_bins)

    return np.concatenate([g, col_occ], axis=0)             # (R_bins+1, PHI_bins)



# ── CONE FILTER ──────────────────────────────────────────────────────────────
def get_cone(hits, truth, particles_conditioned, sx, sy, sz, backr):
    """
    Returns:
      sig      — signal hits in this cone with 8+ hits per particle
      combined — sig + background sample, duplicates removed by hit_id
    """
    good_pids = particles_conditioned['particle_id']
    merged    = hits.merge(truth[['hit_id', 'particle_id']], on='hit_id')

    cone = merged[
        ((merged['x'] > 0) if sx > 0 else (merged['x'] < 0)) &
        ((merged['y'] > 0) if sy > 0 else (merged['y'] < 0)) &
        ((merged['z'] > 0) if sz > 0 else (merged['z'] < 0))
    ]

    # only count a particle as signal if it has 8+ hits in THIS cone
    cone_pid_counts = cone[cone['particle_id'].isin(good_pids)].groupby('particle_id').size()
    valid_pids      = cone_pid_counts[cone_pid_counts >= 8].index
    sig = cone[cone['particle_id'].isin(valid_pids)]

    noi = backr[
        ((backr['x'] > 0) if sx > 0 else (backr['x'] < 0)) &
        ((backr['y'] > 0) if sy > 0 else (backr['y'] < 0)) &
        ((backr['z'] > 0) if sz > 0 else (backr['z'] < 0))
    ]

    combined = pd.concat([sig, noi]).drop_duplicates(subset='hit_id')
    return sig, combined


# ── DATASET BUILDER ──────────────────────────────────────────────────────────
def build_dataset(data_dir, event_ids, save_dir, sample_fraction):
    """
    For each track in each cone in each event, builds:
      X_rz          — r-z matrix (signal + background)   → stage 1 input
      X_rphi        — r-phi matrix (signal + background)  → stage 2 input
      X_rphi_signal — r-phi matrix (signal hits only)     → stage 2 label source
      y_stage1      — binary 0/1                          → stage 1 label

    All positions use measured x, y, z from the hits file only.
    Truth file is used solely to identify which particle generated each hit.
    N_AUGMENT shuffles noise differently each time so model learns that
    track pattern is consistent while noise is random — keeps 50/50 naturally.
    """
    X_rz_all       = []
    X_rphi_all     = []
    X_rphi_sig_all = []
    y_all          = []
    total          = 0

    for event_id in event_ids:
        try:
            h, c, p, t = load_event(os.path.join(data_dir, event_id))

            h['r']   = np.hypot(h['x'], h['y'])
            h['phi'] = np.arctan2(h['y'], h['x'])

            p_cond, h_cond, naff = preprocess_particles(10, 10, p, h)
            backr = h[np.random.rand(len(h)) < sample_fraction].copy()

            for sx, sy, sz in CONE_SIGNS:
                sig, combined = get_cone(h, t, p_cond, sx, sy, sz, backr)
                noi = combined[~combined['hit_id'].isin(sig['hit_id'])]

                if len(sig) == 0 or len(noi) == 0:
                    continue

                for pid in sig['particle_id'].unique():
                    track = sig[sig['particle_id'] == pid].copy()

                    if len(track) < 8:
                        continue

                    for _ in range(N_AUGMENT):
                        # shuffle noise differently each augmentation
                        ns = noi.sample(len(noi), replace=False)

                        # ── signal matrix (track + shuffled background) ────
                        all_r   = np.concatenate([track['r'].values,   ns['r'].values])
                        all_z   = np.concatenate([track['z'].values,   ns['z'].values])
                        all_phi = np.concatenate([track['phi'].values, ns['phi'].values])

                        X_rz_all.append(make_rz_grid(all_r, all_z).T)
                        X_rphi_all.append(make_rphi_grid(all_r, all_phi, sx, sy))
                        
                        # signal-only rphi grid for generating y_stage2 labels
                        X_rphi_sig_all.append(make_rphi_grid(
                            track['r'].values, track['phi'].values, sx, sy
                        ))
                        y_all.append(1.0)

                        # ── noise matrix — one per augmentation = 50/50 ratio
                        nn_ = noi.sample(len(noi), replace=False)
                        X_rz_all.append(make_rz_grid(nn_['r'].values, nn_['z'].values).T)
                        X_rphi_all.append(make_rphi_grid(nn_['r'].values, nn_['phi'].values, sx, sy))
                        
                        # Add SNN_R_BINS + 1 to properly match shapes!
                        X_rphi_sig_all.append(np.zeros((SNN_R_BINS + 1, SNN_PHI_BINS), dtype=np.float32))
                        y_all.append(0.0)

                    total += 1

            print(f"  {event_id}: tracks={total}  matrices={len(X_rz_all)}")

        except Exception as e:
            print(f"  {event_id}: skipped — {e}")

    # ── stack ─────────────────────────────────────────────────────────────────
    X_rz       = np.stack(X_rz_all)
    X_rphi     = np.stack(X_rphi_all)
    X_rphi_sig = np.stack(X_rphi_sig_all)
    y_stage1   = np.array(y_all, dtype=np.float32)
    
    # y_stage2: which phi bins contain signal hits
    # use only the binary rows (0..SNN_R_BINS-1), not the summary row
    y_stage2 = (X_rphi_sig[:, :SNN_R_BINS, :].sum(axis=1) > 0).astype(np.float32)

    # ── save ──────────────────────────────────────────────────────────────────
    np.save(os.path.join(save_dir, "X_rz.npy"),     X_rz)
    np.save(os.path.join(save_dir, "X_rphi.npy"),   X_rphi)
    np.save(os.path.join(save_dir, "y_stage1.npy"), y_stage1)
    np.save(os.path.join(save_dir, "y_stage2.npy"), y_stage2)

    print(f"\nSaved {len(X_rz)} matrix sets to {save_dir}")
    print(f"  X_rz shape:     {X_rz.shape}")
    print(f"  X_rphi shape:   {X_rphi.shape}")
    print(f"  y_stage1 shape: {y_stage1.shape}  signal={int(y_stage1.sum())}  noise={int((1-y_stage1).sum())}")
    print(f"  y_stage2 shape: {y_stage2.shape}  mean active phi bins={y_stage2.sum(axis=1).mean():.1f}")

    return X_rz, X_rphi, y_stage1, y_stage2


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":

    if os.path.exists(os.path.join(Save_dir, "X_rz.npy")):
        print("Datasets already exist — delete them to rebuild.")
        for f in ["X_rz.npy", "X_rphi.npy", "y_stage1.npy", "y_stage2.npy"]:
            path = os.path.join(Save_dir, f)
            if os.path.exists(path):
                arr = np.load(path)
                print(f"  {f}: {arr.shape}")
    else:
        print("Building datasets...")
        build_dataset(data_dir, event_ids, Save_dir, SAMPLE_FRACTION)

    # ── VISUALISE EVENT 1000 — 8 CONES IN R-Z AND R-PHI ──────────────────────
    import matplotlib.pyplot as plt
    import scienceplots
    plt.style.use(["science", "no-latex"])
    plt.rcParams.update({"text.usetex": False, "figure.dpi": 150,
                         "figure.facecolor": "white", "axes.facecolor": "white"})

    print("\nVisualising event 1000...")
    h, _, p, t = load_event(os.path.join(data_dir, "event000001000"))
    h['r']   = np.hypot(h['x'], h['y'])
    h['phi'] = np.arctan2(h['y'], h['x'])
    p_cond, _, _ = preprocess_particles(10, 10, p, h)
    backr = h[np.random.rand(len(h)) < SAMPLE_FRACTION].copy()

    # ── R-Z plot ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (sx, sy, sz) in zip(axes.flat, CONE_SIGNS):
        sig, combined = get_cone(h, t, p_cond, sx, sy, sz, backr)
        noi = combined[~combined['hit_id'].isin(sig['hit_id'])]

        img = np.zeros((SNN_R_BINS, SNN_Z_BINS, 3), dtype=np.float32)
        if len(noi) > 0:
            ri = np.clip((noi['r'].values / SNN_R_MAX * SNN_R_BINS).astype(int), 0, SNN_R_BINS - 1)
            zi = np.clip((np.abs(noi['z'].values) / SNN_Z_MAX * SNN_Z_BINS).astype(int), 0, SNN_Z_BINS - 1)
            img[ri, zi] = [1.0, 0.2, 0.2]
        if len(sig) > 0:
            ri = np.clip((sig['r'].values / SNN_R_MAX * SNN_R_BINS).astype(int), 0, SNN_R_BINS - 1)
            zi = np.clip((np.abs(sig['z'].values) / SNN_Z_MAX * SNN_Z_BINS).astype(int), 0, SNN_Z_BINS - 1)
            img[ri, zi] = [1.0, 1.0, 1.0]

        ax.imshow(img, aspect='auto', origin='lower')
        ax.set_title(f'sig={len(sig)}')
        ax.set_xlabel('z bin')
        ax.set_ylabel('r bin')

    plt.suptitle('Event 1000 — R-Z projection  (white=signal, red=background)', fontsize=12)
    plt.tight_layout()
    plt.show()

    # ── R-PHI plot ────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    for ax, (sx, sy, sz) in zip(axes.flat, CONE_SIGNS):
        sig, combined = get_cone(h, t, p_cond, sx, sy, sz, backr)
        noi = combined[~combined['hit_id'].isin(sig['hit_id'])]
        phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]

        img = np.zeros((SNN_R_BINS, SNN_PHI_BINS, 3), dtype=np.float32)
        if len(noi) > 0:
            ri   = np.clip((noi['r'].values / SNN_R_MAX * SNN_R_BINS).astype(int), 0, SNN_R_BINS - 1)
            phii = np.clip(
                ((noi['phi'].values - phi_min) / (phi_max - phi_min) * SNN_PHI_BINS).astype(int),
                0, SNN_PHI_BINS - 1
            )
            img[ri, phii] = [1.0, 0.2, 0.2]
        if len(sig) > 0:
            ri   = np.clip((sig['r'].values / SNN_R_MAX * SNN_R_BINS).astype(int), 0, SNN_R_BINS - 1)
            phii = np.clip(
                ((sig['phi'].values - phi_min) / (phi_max - phi_min) * SNN_PHI_BINS).astype(int),
                0, SNN_PHI_BINS - 1
            )
            img[ri, phii] = [1.0, 1.0, 1.0]

        ax.imshow(img, aspect='auto', origin='lower')
        ax.set_title(f'sig={len(sig)}')
        ax.set_xlabel('phi bin (normalised)')
        ax.set_ylabel('r bin')

    plt.suptitle('Event 1000 — R-PHI projection  (white=signal, red=background)', fontsize=12)
    plt.tight_layout()
    plt.show()

    # ── VISUALISE SAVED LABELS ────────────────────────────────────────────────
    import matplotlib.gridspec as gridspec

    X_rz     = np.load(os.path.join(Save_dir, "X_rz.npy"))
    X_rphi   = np.load(os.path.join(Save_dir, "X_rphi.npy"))
    y_stage1 = np.load(os.path.join(Save_dir, "y_stage1.npy"))
    y_stage2 = np.load(os.path.join(Save_dir, "y_stage2.npy"))

    N_SHOW  = 6
    sig_idx = np.where(y_stage1 == 1.0)[0]
    noi_idx = np.where(y_stage1 == 0.0)[0]
    half    = N_SHOW // 2
    idxs    = np.concatenate([
        np.random.choice(sig_idx, half, replace=False),
        np.random.choice(noi_idx, half, replace=False),
    ])
    np.random.shuffle(idxs)

    fig = plt.figure(figsize=(N_SHOW * 3.2, 10))
    fig.suptitle("Samples: X_rz (top) | X_rphi (middle) | y_stage2 phi vector (bottom)\n"
                 "y_stage1 shown in title  —  blue=active phi bins",
                 fontsize=11, y=1.01)

    for col, idx in enumerate(idxs):
        gs = gridspec.GridSpec(3, N_SHOW, figure=fig,
                               hspace=0.55, wspace=0.3,
                               left=0.04, right=0.98,
                               top=0.93, bottom=0.05)

        ax_rz = fig.add_subplot(gs[0, col])
        ax_rz.imshow(X_rz[idx], aspect="auto", origin="lower",
                     cmap="Blues", interpolation="nearest")
        label_str = "SIGNAL" if y_stage1[idx] == 1.0 else "NOISE"
        color      = "#1a6bb5" if y_stage1[idx] == 1.0 else "#c0392b"
        ax_rz.set_title(f"y₁={int(y_stage1[idx])}  [{label_str}]",
                        fontsize=8, color=color, pad=3)
        ax_rz.set_xlabel("z bin", fontsize=7)
        ax_rz.set_ylabel("r bin", fontsize=7)
        ax_rz.tick_params(labelsize=6)

        ax_rp = fig.add_subplot(gs[1, col])
        ax_rp.imshow(X_rphi[idx], aspect="auto", origin="lower",
                     cmap="Blues", interpolation="nearest")
        ax_rp.set_xlabel("phi bin", fontsize=7)
        ax_rp.set_ylabel("r bin",   fontsize=7)
        ax_rp.tick_params(labelsize=6)
        if col == 0:
            ax_rp.set_title("X_rphi", fontsize=8, pad=3)

        ax_y2   = fig.add_subplot(gs[2, col])
        phi_vec = y_stage2[idx]
        active  = np.where(phi_vec == 1)[0]
        ax_y2.bar(range(len(phi_vec)), phi_vec,
                  color=["#1a6bb5" if v else "#d0dce8" for v in phi_vec],
                  width=1.0, linewidth=0)
        ax_y2.set_ylim(0, 1.35)
        ax_y2.set_yticks([0, 1])
        ax_y2.set_yticklabels(["0", "1"], fontsize=6)
        ax_y2.set_xlabel("phi bin", fontsize=7)
        ax_y2.tick_params(axis="x", labelsize=6)
        n_active = int(phi_vec.sum())
        ax_y2.set_title(f"y₂  ({n_active} active bins)", fontsize=8, pad=3)
        if n_active > 0:
            ax_y2.axvspan(active[0] - 0.5, active[-1] + 0.5,
                          alpha=0.12, color="#1a6bb5", zorder=0)

    plt.show()