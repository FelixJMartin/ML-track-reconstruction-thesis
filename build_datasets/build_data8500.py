# ================================================================================
# build_data8500.py
# Builds training datasets from train_1 through train_5 (~8500 events total).
# EVENT-LEVEL train/val split to prevent data leakage.
# ================================================================================

from trackml.dataset import load_event
from sklearn.model_selection import train_test_split
import os
import numpy as np
import pandas as pd
import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE")
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

# Each train folder and its event ID range (1000 events each = 1000-1999, 2000-2999, etc.)
TRAIN_FOLDERS = {
    r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_1": range(1000, 2820),
    r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_2": range(2820, 4590),
    r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_3": range(4590, 6410),
    r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_4": range(6410, 8180),
    r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_5": range(8180, 9500),
}

Save_dir = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500"

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
    phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
    g    = np.zeros((SNN_R_BINS, SNN_PHI_BINS), dtype=np.float32)
    ri   = np.clip((hit_r / SNN_R_MAX * SNN_R_BINS).astype(int), 0, SNN_R_BINS - 1)
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


# ── COLLECT ALL EVENT IDs ACROSS FOLDERS ─────────────────────────────────────
def collect_all_events(train_folders):
    """Returns list of (data_dir, event_id) tuples for all available events."""
    all_events = []
    for folder, id_range in train_folders.items():
        if not os.path.exists(folder):
            print(f"  WARNING: folder not found, skipping — {folder}")
            continue
        for i in id_range:
            event_id = f"event{i:09d}"
            # Check at least the hits file exists
            if os.path.exists(os.path.join(folder, f"{event_id}-hits.csv")):
                all_events.append((folder, event_id))
    return all_events


# ── DATASET BUILDER ──────────────────────────────────────────────────────────
def build_dataset(event_list, save_dir, sample_fraction):
    """
    Builds and saves matrix datasets for a given list of (data_dir, event_id) tuples.
    """
    os.makedirs(save_dir, exist_ok=True)

    X_rz_all       = []
    X_rphi_all     = []
    X_rphi_sig_all = []
    y_all          = []
    total_tracks   = 0

    for data_dir, event_id in event_list:
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

                        all_r   = np.concatenate([track['r'].values,   ns['r'].values])
                        all_z   = np.concatenate([track['z'].values,   ns['z'].values])
                        all_phi = np.concatenate([track['phi'].values, ns['phi'].values])

                        X_rz_all.append(make_rz_grid(all_r, all_z).T)
                        X_rphi_all.append(make_rphi_grid(all_r, all_phi, sx, sy))
                        X_rphi_sig_all.append(make_rphi_grid(
                            track['r'].values, track['phi'].values, sx, sy
                        ))
                        y_all.append(1.0)

                        nn_ = noi.sample(len(noi), replace=False)
                        X_rz_all.append(make_rz_grid(nn_['r'].values, nn_['z'].values).T)
                        X_rphi_all.append(make_rphi_grid(
                            nn_['r'].values, nn_['phi'].values, sx, sy
                        ))
                        X_rphi_sig_all.append(
                            np.zeros((SNN_R_BINS + 1, SNN_PHI_BINS), dtype=np.float32)
                        )
                        y_all.append(0.0)

                    total_tracks += 1

            print(f"  {event_id}: tracks={total_tracks}  matrices={len(X_rz_all)}", flush=True)

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
        exit()

    # Collect all events across all folders
    print("Scanning folders for events...")
    all_events = collect_all_events(TRAIN_FOLDERS)
    print(f"Found {len(all_events)} events total across all train folders")

    # Event-level split BEFORE building any matrices
    train_events, val_events = train_test_split(all_events, test_size=0.2, random_state=42)
    print(f"Train events: {len(train_events)}  Val events: {len(val_events)}\n")

    print("Building train dataset...")
    build_dataset(train_events, train_save, SAMPLE_FRACTION)

    print("\nBuilding val dataset...")
    build_dataset(val_events, val_save, SAMPLE_FRACTION)