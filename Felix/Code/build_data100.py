# ================================================================================
# build_data100.py
# Builds all training datasets for stage 1 (r-z) and stage 2 (r-phi).
# 100 event test with measured positions and N_AUGMENT=2
# EVENT-LEVEL train/val split to prevent data leakage.
# ================================================================================

from trackml.dataset import load_event
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from Functions import preprocess_particles

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── CONFIG ───────────────────────────────────────────────────────────────────
SAMPLE_FRACTION = 0.01
N_AUGMENT       = 2
SNN_R_BINS      = 100
SNN_Z_BINS      = 100
SNN_PHI_BINS    = 100
SNN_R_MAX       = 1100.0
SNN_Z_MAX       = 3000.0

data_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_100_events"
Save_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices100"

# ── EVENT-LEVEL SPLIT ────────────────────────────────────────────────────────
# Split event IDs BEFORE building any matrices to prevent leakage.
# Augmented copies of the same event will only appear in one split.
all_event_ids = [f"event{i:09d}" for i in range(1000, 1100)]
train_ids, val_ids = train_test_split(all_event_ids, test_size=0.2, random_state=42)

print(f"Train events: {len(train_ids)}  Val events: {len(val_ids)}")

PLOT_ONE        = False
PLOT_EVENT_ID   = "event000001000"
PLOT_CONE       = (+1, +1, +1)


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
    R-phi occupancy grid with one extra summary row appended.
    Rows 0..SNN_R_BINS-1 : binary occupancy
    Row  SNN_R_BINS       : column occupancy fraction per phi bin
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

    col_occ = (g.sum(axis=0) / SNN_R_BINS)[np.newaxis, :]
    return np.concatenate([g, col_occ], axis=0)


# ── CONE FILTER ──────────────────────────────────────────────────────────────
def get_cone(hits, truth, particles_conditioned, sx, sy, sz, backr):
    good_pids = particles_conditioned['particle_id']
    merged    = hits.merge(truth[['hit_id', 'particle_id']], on='hit_id')

    cone = merged[
        ((merged['x'] > 0) if sx > 0 else (merged['x'] < 0)) &
        ((merged['y'] > 0) if sy > 0 else (merged['y'] < 0)) &
        ((merged['z'] > 0) if sz > 0 else (merged['z'] < 0))
    ]

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
    Builds and saves matrix datasets for a given list of event IDs.
    All augmented copies of an event stay in this split only.
    """
    os.makedirs(save_dir, exist_ok=True)

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
                        X_rphi_sig_all.append(make_rphi_grid(
                            track['r'].values, track['phi'].values, sx, sy
                        ))
                        y_all.append(1.0)

                        # noise-only matrix (same event, signal removed)
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
    y_stage2   = (X_rphi_sig[:, :SNN_R_BINS, :].sum(axis=1) > 0).astype(np.float32)

    np.save(os.path.join(save_dir, "X_rz.npy"),     X_rz)
    np.save(os.path.join(save_dir, "X_rphi.npy"),   X_rphi)
    np.save(os.path.join(save_dir, "y_stage1.npy"), y_stage1)
    np.save(os.path.join(save_dir, "y_stage2.npy"), y_stage2)

    print(f"\nSaved {len(X_rz)} matrices to {save_dir}")
    print(f"  X_rz shape:     {X_rz.shape}")
    print(f"  X_rphi shape:   {X_rphi.shape}")
    print(f"  y_stage1:       signal={int(y_stage1.sum())}  noise={int((1-y_stage1).sum())}")
    print(f"  y_stage2:       mean active phi bins={y_stage2.sum(axis=1).mean():.1f}")

    return X_rz, X_rphi, y_stage1, y_stage2


# ── MAIN ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train_save = os.path.join(Save_dir, "train")
    val_save   = os.path.join(Save_dir, "val")

    if os.path.exists(os.path.join(train_save, "X_rz.npy")):
        print("Datasets already exist — delete them to rebuild.")
        for split, d in [("train", train_save), ("val", val_save)]:
            print(f"\n{split}:")
            for f in ["X_rz.npy", "X_rphi.npy", "y_stage1.npy", "y_stage2.npy"]:
                path = os.path.join(d, f)
                if os.path.exists(path):
                    arr = np.load(path)
                    print(f"  {f}: {arr.shape}")
    else:
        print("Building train dataset...")
        build_dataset(data_dir, train_ids, train_save, SAMPLE_FRACTION)

        print("\nBuilding val dataset...")
        build_dataset(data_dir, val_ids, val_save, SAMPLE_FRACTION)