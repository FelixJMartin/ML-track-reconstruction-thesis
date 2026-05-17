# ================================================================================
# build_data.py
# Builds all training datasets for stage 1 (r-z) and stage 2 (r-phi).
# 100 event test with measured positions and N_AUGMENT=3
# ================================================================================

from trackml.dataset import load_event
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import preprocess_particles

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── CONFIG ───────────────────────────────────────────────────────────────────
SAMPLE_FRACTION = 0.01
N_AUGMENT       = 3
SNN_R_BINS      = 100
SNN_Z_BINS      = 100
SNN_PHI_BINS    = 100
SNN_R_MAX       = 1100.0
SNN_Z_MAX       = 3000.0

data_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_100_events"
Save_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices100"
event_ids = [f"event{i:09d}" for i in range(1000, 1100)]  # 100 events

PLOT_ONE        = True           # set True to just plot matrices for one event
PLOT_EVENT_ID   = "event000001000"
PLOT_CONE       = (+1, +1, +1)   # (sx, sy, sz)


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
    spikematrix = np.zeros((SNN_R_BINS, SNN_Z_BINS), dtype=np.float32)
    ri = np.clip((hit_r / SNN_R_MAX * SNN_R_BINS).astype(int), 0, SNN_R_BINS - 1)
    zi = np.clip((np.abs(hit_z) / SNN_Z_MAX * SNN_Z_BINS).astype(int), 0, SNN_Z_BINS - 1)
    spikematrix[ri, zi] = 1.0
    return spikematrix


def make_rphi_grid(hit_r, hit_phi, sx, sy):
    """
    R-phi occupancy grid with one extra summary timestep appended.
    Rows 0..SNN_R_BINS-1 : binary occupancy — 1 if any hit in (r, phi) bin
    Row  SNN_R_BINS       : column occupancy fraction — how many r-bins
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
                        ns = noi.sample(len(noi), replace=False)

                        # signal matrix
                        all_r   = np.concatenate([track['r'].values,   ns['r'].values])
                        all_z   = np.concatenate([track['z'].values,   ns['z'].values])
                        all_phi = np.concatenate([track['phi'].values, ns['phi'].values])

                        X_rz_all.append(make_rz_grid(all_r, all_z).T)
                        X_rphi_all.append(make_rphi_grid(all_r, all_phi, sx, sy))

                        # signal-only rphi grid for generating y_stage2 labels
                        # note: signal-only grid also has the extra occupancy row
                        X_rphi_sig_all.append(make_rphi_grid(
                            track['r'].values, track['phi'].values, sx, sy
                        ))
                        y_all.append(1.0)

                        # noise-only matrix
                        nn_ = noi.sample(len(noi), replace=False)
                        X_rz_all.append(make_rz_grid(nn_['r'].values, nn_['z'].values).T)
                        X_rphi_all.append(make_rphi_grid(
                            nn_['r'].values, nn_['phi'].values, sx, sy
                        ))
                        X_rphi_sig_all.append(
                            np.zeros((SNN_R_BINS + 1, SNN_PHI_BINS), dtype=np.float32)
                        )
                        y_all.append(0.0)

                    total += 1

            print(f"  {event_id}: tracks={total}  matrices={len(X_rz_all)}")

        except Exception as e:
            print(f"  {event_id}: skipped — {e}")

    X_rz       = np.stack(X_rz_all)
    X_rphi     = np.stack(X_rphi_all)
    X_rphi_sig = np.stack(X_rphi_sig_all)
    y_stage1   = np.array(y_all, dtype=np.float32)

    # y_stage2: which phi bins contain signal hits
    # use only the binary rows (0..SNN_R_BINS-1), not the summary row
    y_stage2 = (X_rphi_sig[:, :SNN_R_BINS, :].sum(axis=1) > 0).astype(np.float32)

    np.save(os.path.join(save_dir, "X_rz.npy"),     X_rz)
    np.save(os.path.join(save_dir, "X_rphi.npy"),   X_rphi)
    np.save(os.path.join(save_dir, "y_stage1.npy"), y_stage1)
    np.save(os.path.join(save_dir, "y_stage2.npy"), y_stage2)

    print(f"\nSaved {len(X_rz)} matrix sets to {save_dir}")
    print(f"  X_rz shape:     {X_rz.shape}")
    print(f"  X_rphi shape:   {X_rphi.shape}  — rows 0..{SNN_R_BINS-1}=binary, row {SNN_R_BINS}=col_occ")
    print(f"  y_stage1 shape: {y_stage1.shape}  signal={int(y_stage1.sum())}  noise={int((1-y_stage1).sum())}")
    print(f"  y_stage2 shape: {y_stage2.shape}  mean active phi bins={y_stage2.sum(axis=1).mean():.1f}")

    return X_rz, X_rphi, y_stage1, y_stage2


# ── SINGLE-EVENT PREVIEW ─────────────────────────────────────────────────────
def plot_one_event(event_id, sx, sy, sz):
    h, _, p, t = load_event(os.path.join(data_dir, event_id))
    h['r']   = np.hypot(h['x'], h['y'])
    h['phi'] = np.arctan2(h['y'], h['x'])

    p_cond, _, _ = preprocess_particles(10, 10, p, h)
    backr        = h[np.random.rand(len(h)) < SAMPLE_FRACTION].copy()
    sig, combined = get_cone(h, t, p_cond, sx, sy, sz, backr)
    noi           = combined[~combined['hit_id'].isin(sig['hit_id'])]

    all_r   = np.concatenate([sig['r'].values,   noi['r'].values])
    all_z   = np.concatenate([sig['z'].values,   noi['z'].values])
    all_phi = np.concatenate([sig['phi'].values, noi['phi'].values])

    rz_grid   = make_rz_grid(all_r, all_z).T
    rphi_grid = make_rphi_grid(all_r, all_phi, sx, sy)

    cone_label = f"({'+'if sx>0 else'-'}x, {'+'if sy>0 else'-'}y, {'+'if sz>0 else'-'}z)"
    has_signal = len(sig) > 0

    _, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(rz_grid, aspect='auto', origin='lower', cmap='hot')
    axes[0].set_title(f'r-z matrix  —  {event_id}  cone {cone_label}\n'
                      f'signal hits={len(sig)}  noise hits={len(noi)}  label={"signal" if has_signal else "noise"}')
    axes[0].set_xlabel('z bins'); axes[0].set_ylabel('r bins')

    axes[1].imshow(rphi_grid[:SNN_R_BINS], aspect='auto', origin='lower', cmap='hot')
    axes[1].set_title(f'r-phi matrix  (binary rows only)')
    axes[1].set_xlabel('phi bins'); axes[1].set_ylabel('r bins')

    plt.tight_layout()
    plt.show()


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