# ================================================================================
# build_data150_multinoise.py
# Builds all 150x150 datasets at 1%, 2%, 5%, 10% noise in one run.
# Just set BINS and NOISE_LEVELS below and run:
#   python build_data150_multinoise.py
# ================================================================================

import os
import numpy as np
import pandas as pd
from trackml.dataset import load_event
from sklearn.model_selection import train_test_split
import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE")
from Functions import preprocess_particles

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── CONFIG ───────────────────────────────────────────────────────────────────
BINS         = 150
NOISE_LEVELS = [0.01, 0.02, 0.05, 0.10]
N_AUGMENT    = 2
SNN_R_MAX    = 1100.0
SNN_Z_MAX    = 3000.0

data_dir  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_1"
base_save = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data"
# ── EVENT-LEVEL SPLIT ────────────────────────────────────────────────────────
# Split BEFORE building any matrices so augmented copies of the same event
# never appear in both train and val.
all_event_ids = [f"event{i:09d}" for i in range(1000, 2800)]
train_ids, val_ids = train_test_split(
    all_event_ids, test_size=0.2, random_state=42
)
print(f"Train events: {len(train_ids)}  Val events: {len(val_ids)}")

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


# ── GRID FUNCTIONS (bins passed explicitly) ───────────────────────────────────
def make_rz_grid(hit_r, hit_z, r_bins, z_bins):
    spikematrix = np.zeros((r_bins, z_bins), dtype=np.float32)
    ri = np.clip((hit_r / SNN_R_MAX * r_bins).astype(int), 0, r_bins - 1)
    zi = np.clip((np.abs(hit_z) / SNN_Z_MAX * z_bins).astype(int), 0, z_bins - 1)
    spikematrix[ri, zi] = 1.0
    return spikematrix


def make_rphi_grid(hit_r, hit_phi, sx, sy, r_bins, phi_bins):
    phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]

    g    = np.zeros((r_bins, phi_bins), dtype=np.float32)
    ri   = np.clip((hit_r / SNN_R_MAX * r_bins).astype(int), 0, r_bins - 1)
    phii = np.clip(
        ((hit_phi - phi_min) / (phi_max - phi_min) * phi_bins).astype(int),
        0, phi_bins - 1
    )
    g[ri, phii] = 1.0

    col_occ = (g.sum(axis=0) / r_bins)[np.newaxis, :]
    return np.concatenate([g, col_occ], axis=0)  # (r_bins+1, phi_bins)


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
def build_dataset(data_dir, event_ids, save_dir, sample_fraction, bins):
    r_bins   = bins
    z_bins   = bins
    phi_bins = bins

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

                        all_r   = np.concatenate([track['r'].values,   ns['r'].values])
                        all_z   = np.concatenate([track['z'].values,   ns['z'].values])
                        all_phi = np.concatenate([track['phi'].values, ns['phi'].values])

                        X_rz_all.append(make_rz_grid(all_r, all_z, r_bins, z_bins).T)
                        X_rphi_all.append(make_rphi_grid(all_r, all_phi, sx, sy, r_bins, phi_bins))
                        X_rphi_sig_all.append(make_rphi_grid(
                            track['r'].values, track['phi'].values, sx, sy, r_bins, phi_bins
                        ))
                        y_all.append(1.0)

                        nn_ = noi.sample(len(noi), replace=False)
                        X_rz_all.append(make_rz_grid(nn_['r'].values, nn_['z'].values, r_bins, z_bins).T)
                        X_rphi_all.append(make_rphi_grid(nn_['r'].values, nn_['phi'].values, sx, sy, r_bins, phi_bins))
                        X_rphi_sig_all.append(np.zeros((r_bins + 1, phi_bins), dtype=np.float32))
                        y_all.append(0.0)

                    total += 1

            print(f"  {event_id}: tracks={total}  matrices={len(X_rz_all)}")

        except Exception as e:
            print(f"  {event_id}: skipped — {e}")

    X_rz       = np.stack(X_rz_all)
    X_rphi     = np.stack(X_rphi_all)
    X_rphi_sig = np.stack(X_rphi_sig_all)
    y_stage1   = np.array(y_all, dtype=np.float32)
    y_stage2   = (X_rphi_sig[:, :r_bins, :].sum(axis=1) > 0).astype(np.float32)

    np.save(os.path.join(save_dir, "X_rz.npy"),     X_rz)
    np.save(os.path.join(save_dir, "X_rphi.npy"),   X_rphi)
    np.save(os.path.join(save_dir, "y_stage1.npy"), y_stage1)
    np.save(os.path.join(save_dir, "y_stage2.npy"), y_stage2)

    print(f"\nSaved {len(X_rz)} matrix sets to {save_dir}")
    print(f"  X_rz shape:     {X_rz.shape}")
    print(f"  X_rphi shape:   {X_rphi.shape}")
    print(f"  y_stage1 shape: {y_stage1.shape}  signal={int(y_stage1.sum())}  noise={int((1-y_stage1).sum())}")
    print(f"  y_stage2 shape: {y_stage2.shape}  mean active phi bins={y_stage2.sum(axis=1).mean():.1f}")


# ── MAIN — loop over all noise levels ────────────────────────────────────────
if __name__ == "__main__":

    for noise in NOISE_LEVELS:
        noise_pct  = int(round(noise * 100))
        noise_dir  = os.path.join(base_save, f"matrices_{BINS}_{noise_pct}pct")
        train_save = os.path.join(noise_dir, "train")
        val_save   = os.path.join(noise_dir, "val")

        print(f"\n{'='*60}")
        print(f"  Building {BINS}x{BINS} at {noise_pct}% noise")
        print(f"  Output: {noise_dir}")
        print(f"{'='*60}")

        if os.path.exists(os.path.join(train_save, "X_rz.npy")):
            print("  Already exists — skipping. Delete train/ to rebuild.")
            for split, d in [("train", train_save), ("val", val_save)]:
                print(f"\n  {split}:")
                for f in ["X_rz.npy", "X_rphi.npy", "y_stage1.npy", "y_stage2.npy"]:
                    path = os.path.join(d, f)
                    if os.path.exists(path):
                        arr = np.load(path)
                        print(f"    {f}: {arr.shape}")
        else:
            print("  Building train split...")
            build_dataset(data_dir, train_ids, train_save, noise, BINS)

            print("\n  Building val split...")
            build_dataset(data_dir, val_ids, val_save, noise, BINS)

    print("\n" + "="*60)
    print("  All datasets complete!")
    print("="*60)