# ================================================================================
# ================= IMPORTS ======================================================

from trackml.dataset import load_event
import matplotlib.pyplot as plt
import seaborn as sns
import scienceplots
import sys, os
import numpy as np
import pandas as pd
import torch

# ── Model imports ─────────────────────────────────────────────────────────────
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Felix\SNN")
from Stage1_ConeFilter import ConeFilterSNN
from Stage2_PhiFilter100 import PhiFinderSNN, scan_event_full
from build_data100 import SNN_R_BINS as _SNN_R_BINS
from Functions import preprocess_particles, choice_particels

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
plt.style.use(["science", "no-latex"])
plt.rcParams["text.usetex"]      = False
plt.rcParams["axes.spines.top"]  = True
plt.rcParams["axes.spines.right"] = True

# ================================================================================
# ================= DIRECTORIES ==================================================

BASE_DIR     = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data"
DATA_DIR     = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_100_events"
DIR_100      = os.path.join(BASE_DIR, "matrices100")
DIR_1000     = os.path.join(BASE_DIR, "matrices1000")
OUT_DIR      = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\figures"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Model paths ───────────────────────────────────────────────────────────────
S1_100_PATH  = os.path.join(DIR_100,  "model_stage1.pt")
S2_100_PATH  = os.path.join(DIR_100,  "model_stage2.pt")
S1_1800_PATH = os.path.join(DIR_1000, "model_h256_128_lr0.001_bs32_dr0.1.pt")
S2_1800_PATH = os.path.join(DIR_1000, "model_stage2_h256_128_lr0.001_bs32_dr0.1.pt")



# ── Data paths ────────────────────────────────────────────────────────────────
X_TRAIN_PATH = os.path.join(DIR_100, "X_train.npy")
Y_TRAIN_PATH = os.path.join(DIR_100, "y_train.npy")

# ================================================================================
# ================= LOAD EVENT ===================================================

EVENT_PREFIX = "event000001000"
hits, cells, particles, truth = load_event(os.path.join(DATA_DIR, EVENT_PREFIX))

hits["r"]   = np.hypot(hits["x"], hits["y"])
hits["phi"] = np.arctan2(hits["y"], hits["x"])

particle_conditions,  hits_cond,        n_aff_full      = preprocess_particles(10, 10, particles, hits)
particles_conditioned, hits_conditioned, n_afferent_trim = preprocess_particles(10, 10, particles, hits)

first_particle_ind = particles_conditioned.index
n                  = first_particle_ind[0]
signal, particle_id = choice_particels(hits, particles_conditioned, n, truth)

SAMPLE_FRACTION = 0.01
Kagg_n = hits[np.random.rand(len(hits)) < SAMPLE_FRACTION].copy()



# ================================================================================
# ================= PLOTTING FUNCTIONS ===========================================
 
def plot_3d(hits, s=1):
    volumes = sorted(hits.volume_id.unique())
    palette = sns.color_palette("rainbow", len(volumes))
    for i, volume in enumerate(volumes):
        v = hits[hits.volume_id == volume]
        ax.scatter(v.z, v.x, v.y, s=s, alpha=0.3,
                   label=f"volume {volume}", color=palette[i])
    ax.set_xlim(-1200, 1200); ax.set_ylim(-1200, 1200); ax.set_zlim(-1200, 1200)
    ax.set_xlabel('Z (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('X (mm)')
 
 
OCTANTS = [
    (1, '+x+y+z', (+1,+1,+1)),
    (2, '+x+y-z', (+1,+1,-1)),
    (3, '+x-y+z', (+1,-1,+1)),
    (4, '+x-y-z', (+1,-1,-1)),
    (5, '-x+y+z', (-1,+1,+1)),
    (6, '-x+y-z', (-1,+1,-1)),
    (7, '-x-y+z', (-1,-1,+1)),
    (8, '-x-y-z', (-1,-1,-1)),
]

def plot_3d_octants(hits, s=1):
    palette = sns.color_palette("tab10", 8)
    for (num, label, (sx, sy, sz)), color in zip(OCTANTS, palette):
        mask = (
            (hits["x"] > 0 if sx > 0 else hits["x"] < 0) &
            (hits["y"] > 0 if sy > 0 else hits["y"] < 0) &
            (hits["z"] > 0 if sz > 0 else hits["z"] < 0)
        )
        v = hits[mask]
        ax.scatter(v.z, v.x, v.y, s=s, alpha=0.3,
                   label=f"octant {num}", color=color)
    ax.set_xlim(-1200, 1200); ax.set_ylim(-1200, 1200); ax.set_zlim(-1200, 1200)
    ax.set_xlabel('Z (mm)'); ax.set_ylabel('X (mm)'); ax.set_zlabel('Y (mm)')


def plot_signal(hits, s=1):
    volumes = sorted(hits.volume_id.unique())
    for volume in volumes:
        v = hits[hits.volume_id == volume]
        ax.scatter(v.z, v.x, v.y, s=s, alpha=1.0, color='#ff0033', zorder=5)
    ax.set_xlim(-1200, 1200); ax.set_ylim(-1200, 1200); ax.set_zlim(-1200, 1200)
    ax.set_xlabel('X (mm)'); ax.set_ylabel('Y (mm)'); ax.set_zlabel('Z (mm)')
 
 
def plot_rz(hits, color="rainbow"):
    volumes = sorted(hits.volume_id.unique())
    palette = sns.color_palette(color, len(volumes))
    for i, volume in enumerate(volumes):
        v = hits[hits.volume_id == volume]
        plt.scatter(v.z, v.r, s=3, alpha=0.3,
                    label=f"volume {volume}", color=palette[i])
    plt.xlabel('z (mm)'); plt.ylabel('r (mm)')
    plt.xlim(-1500, 1500); plt.ylim(-1500, 1500)
 
 
def vector(ax, h, length=1000):
    x, y, z = h["x"], h["y"], h["z"]
    R = np.sqrt(x**2 + y**2 + z**2)
    ax.quiver(0, 0, 0, z/R*length, x/R*length, y/R*length)
 
 
def noise_tuner(hits, lvl):
    acceptance = np.random.rand(len(hits)) < lvl
    return hits[acceptance]
 
 
def signal_finder(cone, truth, particles, min_hits=10):
    good_particles = preprocess_particles(10, 10, particles, cone)[0]
    signal_hit_ids = truth[truth["particle_id"].isin(good_particles["particle_id"])]["hit_id"]
    signal_hits    = cone[cone["hit_id"].isin(signal_hit_ids)]
    if len(signal_hits) < min_hits:
        return signal_hits.iloc[0:0]
    return signal_hits
 
 
def split_cones(hits):
    signs = [(+1,+1,+1),(+1,+1,-1),(+1,-1,+1),(+1,-1,-1),
             (-1,+1,+1),(-1,+1,-1),(-1,-1,+1),(-1,-1,-1)]
    cones = []
    for sx, sy, sz in signs:
        mask = (
            ((hits["x"] > 0) if sx > 0 else (hits["x"] < 0)) &
            ((hits["y"] > 0) if sy > 0 else (hits["y"] < 0)) &
            ((hits["z"] > 0) if sz > 0 else (hits["z"] < 0))
        )
        cones.append(hits[mask])
    return cones
 
 
def make_rz_matrix(noise, sig, r_bins=100, z_bins=100):
    r_min, r_max = noise["r"].min(), noise["r"].max()
    z_min, z_max = noise["z"].abs().min(), noise["z"].abs().max()

    def to_idx(r, z):
        ri = ((r - r_min) / (r_max - r_min) * (r_bins-1)).astype(int).clip(0, r_bins-1)
        zi = ((z - z_min) / (z_max - z_min) * (z_bins-1)).astype(int).clip(0, z_bins-1)
        return ri, zi

    matrix       = np.zeros((r_bins, z_bins))
    ri, zi        = to_idx(noise["r"], noise["z"].abs())
    matrix[ri, zi] = 1
    if len(sig) > 0:
        sri, szi        = to_idx(sig["r"], sig["z"].abs())
        matrix[sri, szi] = 3
    return matrix


CONE_PHI_RANGES = {
    (+1,+1): (0,          np.pi/2),
    (+1,-1): (-np.pi/2,   0),
    (-1,+1): (np.pi/2,    np.pi),
    (-1,-1): (-np.pi,    -np.pi/2),
}

SNN_R_BINS   = 100
SNN_PHI_BINS = 100
SNN_R_MAX    = 1100.0

def make_rphi_matrix(noise, sig, sx, sy, r_bins=100, phi_bins=100):
    r_max            = noise["r"].max()
    phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]

    def to_idx(r, phi):
        ri   = (r / r_max * (r_bins-1)).astype(int).clip(0, r_bins-1)
        phii = ((phi - phi_min) / (phi_max - phi_min) * (phi_bins-1)).astype(int).clip(0, phi_bins-1)
        return ri, phii

    matrix = np.zeros((r_bins, phi_bins))
    ri, phii = to_idx(noise["r"], noise["phi"])
    matrix[ri, phii] = 1
    if len(sig) > 0:
        sri, sphii = to_idx(sig["r"], sig["phi"])
        matrix[sri, sphii] = 3
    return matrix


def make_rphi_extended_matrix(noise, sig, sx, sy):
    """Same colour encoding as make_rphi_matrix but appends the column-occupancy summary row."""
    phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]

    def to_idx(r, phi):
        ri   = (r / SNN_R_MAX * (SNN_R_BINS-1)).astype(int).clip(0, SNN_R_BINS-1)
        phii = ((phi - phi_min) / (phi_max - phi_min) * (SNN_PHI_BINS-1)).astype(int).clip(0, SNN_PHI_BINS-1)
        return ri, phii

    matrix = np.zeros((SNN_R_BINS, SNN_PHI_BINS))
    ri, phii = to_idx(noise["r"], noise["phi"])
    matrix[ri, phii] = 1
    if len(sig) > 0:
        sri, sphii = to_idx(sig["r"], sig["phi"])
        matrix[sri, sphii] = 3

    # fraction of r-bins occupied per phi bin (any hit, signal or noise)
    col_occ = ((matrix > 0).sum(axis=0) / SNN_R_BINS)[np.newaxis, :]
    return np.concatenate([matrix, col_occ], axis=0)   # (R_BINS+1, PHI_BINS)




# ================================================================================
# ================= PLOT 1 — Full detector 3D ====================================

# fig = plt.figure(figsize=(12, 8))
# ax  = fig.add_subplot(111, projection='3d')
# ax.view_init(elev=20, azim=130)
# plot_3d(hits, s=0.2)
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, 'full_detector_3d.pdf'), bbox_inches='tight')
# plt.show()

# ================================================================================
# ================= PLOT 1b — Full detector coloured by octant ===================

# fig = plt.figure(figsize=(12, 8))
# ax  = fig.add_subplot(111, projection='3d')
# ax.view_init(elev=20, azim=130)
# plot_3d_octants(hits, s=0.2)
# ax.legend(loc='upper left', fontsize=8, markerscale=4,
#           framealpha=0.8, ncol=2)
# plt.tight_layout()
# plt.savefig(os.path.join(OUT_DIR, 'octants_3d.pdf'), bbox_inches='tight')
# plt.show()

# ================================================================================
# ================= PLOT 4 — All 8 cones r-z matrices ============================

# Kagg_n      = noise_tuner(Kagg_n, 1)
# hits_cones  = split_cones(hits)
# noise_cones = split_cones(Kagg_n)
# sig_cones   = [signal_finder(c, truth, particles) for c in hits_cones]

# cone_signs_list = [(+1,+1,+1),(+1,+1,-1),(+1,-1,+1),(+1,-1,-1),
#                    (-1,+1,+1),(-1,+1,-1),(-1,-1,+1),(-1,-1,-1)]

# fig, axes = plt.subplots(2, 4, figsize=(20, 10))
# for i, (noise, sig, ax) in enumerate(zip(noise_cones, sig_cones, axes.flat)):
#     matrix = make_rz_matrix(noise, sig, r_bins=100, z_bins=100)
#     ax.imshow(matrix, aspect='auto', cmap='hot', vmax=3)
#     ax.set_xlabel('z bins')
#     ax.set_ylabel('r bins')
#     ax.set_title(f'Cone {i+1}', fontsize=8)
# plt.tight_layout()
# plt.show()

# ================================================================================
# ================= PLOT 4b — All 8 cones r-phi matrices =========================

# fig, axes = plt.subplots(2, 4, figsize=(20, 10))
# for i, ((sx, sy, sz), noise, sig, ax) in enumerate(
#         zip(cone_signs_list, noise_cones, sig_cones, axes.flat)):
#     matrix = make_rphi_matrix(noise, sig, sx, sy, r_bins=100, phi_bins=100)
#     ax.imshow(matrix, aspect='auto', origin='lower', cmap='hot', vmax=3)
#     ax.set_xlabel('phi bins')
#     ax.set_ylabel('r bins')
# plt.tight_layout()
# plt.show()

# ================================================================================
# ================= PLOT 4c — All 8 cones r-phi + occupancy strip ================

# fig = plt.figure(figsize=(22, 13))
# # 5 rows: matrix | occ-strip | spacer | matrix | occ-strip
# gs = fig.add_gridspec(5, 5,
#                       height_ratios=[8, 1, 1.2, 8, 1],
#                       width_ratios=[1, 1, 1, 1, 0.04],
#                       hspace=0.08, wspace=0.35)

# # map group index (0=top, 1=bottom) to grid rows
# _bin_rows = [0, 3]
# _occ_rows = [1, 4]

# last_occ_im = None
# for i, ((sx, sy, sz), noise, sig) in enumerate(
#         zip(cone_signs_list, noise_cones, sig_cones)):
#     matrix = make_rphi_extended_matrix(noise, sig, sx, sy)
#     binary = matrix[:SNN_R_BINS]
#     occ    = matrix[SNN_R_BINS:SNN_R_BINS + 1]

#     group  = i // 4
#     gs_col = i % 4

#     ax_bin = fig.add_subplot(gs[_bin_rows[group], gs_col])
#     ax_bin.imshow(binary, aspect='auto', origin='lower', cmap='hot', vmax=3)
#     ax_bin.set_ylabel('r bins')
#     ax_bin.set_xticks([])

#     ax_occ = fig.add_subplot(gs[_occ_rows[group], gs_col])
#     last_occ_im = ax_occ.imshow(occ, aspect='auto', origin='lower',
#                                 cmap='plasma', vmin=0, vmax=1)
#     ax_occ.set_yticks([])
#     ax_occ.set_xlabel('phi bins')

# cbar_ax = fig.add_subplot(gs[:, 4])
# fig.colorbar(last_occ_im, cax=cbar_ax, label='column occupancy fraction')
# plt.tight_layout()
# plt.show()


# ================================================================================
# ================= PLOT 4d — Single cone r-phi + occupancy strip ================

# _i  = 1   # which cone to show (0–7)
# _sx, _sy, _ = cone_signs_list[_i]
# _matrix = make_rphi_extended_matrix(noise_cones[_i], sig_cones[_i], _sx, _sy)
# _binary = _matrix[:SNN_R_BINS]
# _occ    = _matrix[SNN_R_BINS:SNN_R_BINS + 1]

# fig = plt.figure(figsize=(7, 8))
# gs  = fig.add_gridspec(2, 2,
#                        height_ratios=[10, 1],
#                        width_ratios=[1, 0.04],
#                        hspace=0.05, wspace=0.08)

# ax_bin = fig.add_subplot(gs[0, 0])
# ax_bin.imshow(_binary, aspect='auto', origin='lower', cmap='hot', vmax=3)
# ax_bin.set_ylabel('r bins')
# ax_bin.set_xticks([])

# ax_occ = fig.add_subplot(gs[1, 0])
# occ_im = ax_occ.imshow(_occ, aspect='auto', origin='lower',
#                         cmap='plasma', vmin=0, vmax=1)
# ax_occ.set_yticks([])
# ax_occ.set_xlabel('phi bins')

# cbar_ax = fig.add_subplot(gs[:, 1])
# fig.colorbar(occ_im, cax=cbar_ax, label='column occupancy fraction')
# plt.tight_layout()
# plt.show()



if os.path.exists(S1_100_PATH):
    ckpt         = torch.load(S1_100_PATH, map_location='cpu')
    train_losses = ckpt['train_losses']
    val_losses   = ckpt['val_losses']
    val_accs     = ckpt['val_accs']
    epochs       = range(1, len(train_losses) + 1)

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))
    ax_l.plot(epochs, train_losses, color='#0077BB', label='Train loss')
    ax_l.plot(epochs, val_losses,   color='#EE7733', label='Val loss')
    ax_l.set_xlabel('Epoch')
    ax_l.set_ylabel('BCE Loss')
    ax_l.legend()
    ax_l.grid(True, alpha=0.3, linestyle='--')
    ax_a.plot(epochs, [v * 100 for v in val_accs], color='#009988')
    ax_a.set_xlabel('Epoch')
    ax_a.set_ylabel('Validation accuracy (%)')
    ax_a.set_ylim(0, 105)
    ax_a.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'loss_100events.pdf'), bbox_inches='tight')
    plt.show()
else:
    print(f"Stage 1 (100-event) model not found — skipping Plot 7.")

# ================================================================================
# ================= PLOT 8 — Training loss (1800-event model) ====================

if os.path.exists(S1_1800_PATH):
    ckpt         = torch.load(S1_1800_PATH, map_location='cpu')
    train_losses = ckpt['train_losses']
    val_losses   = ckpt['val_losses']
    val_accs     = ckpt['val_accs']
    epochs       = range(1, len(train_losses) + 1)

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))
    ax_l.plot(epochs, train_losses, color='#0077BB', label='Train loss')
    ax_l.plot(epochs, val_losses,   color='#EE7733', label='Val loss')
    ax_l.set_xlabel('Epoch')
    ax_l.set_ylabel('BCE Loss')
    ax_l.legend()
    ax_l.grid(True, alpha=0.3, linestyle='--')
    ax_a.plot(epochs, [v * 100 for v in val_accs], color='#009988')
    ax_a.set_xlabel('Epoch')
    ax_a.set_ylabel('Validation accuracy (%)')
    ax_a.set_ylim(0, 105)
    ax_a.grid(True, alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()
else:
    print(f"Stage 1 (1800-event) model not found — skipping Plot 8.")



# ================================================================================
# ================= PLOT 9 — Stage 2 training loss (100-event model) =============

if os.path.exists(S2_100_PATH):
    ckpt         = torch.load(S2_100_PATH, map_location='cpu')
    train_losses = ckpt['train_losses']
    val_losses   = ckpt['val_losses']
    val_ious     = ckpt['val_ious']
    val_overlaps = ckpt['val_overlaps']
    epochs       = range(1, len(train_losses) + 1)

    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))

    ax_l.plot(epochs, train_losses, color='#0077BB', label='train')
    ax_l.plot(epochs, val_losses,   color='#009988', label='val')
    ax_l.set_xlabel('epoch')
    ax_l.set_ylabel('BCE loss')
    ax_l.legend()
    ax_l.grid(True, alpha=0.3, linestyle='--')

    ax_a.plot(epochs, [v * 100 for v in val_ious],     color='#0077BB', label='IoU %')
    ax_a.plot(epochs, [v * 100 for v in val_overlaps], color='#EE7733', label='Overlap %')
    ax_a.set_xlabel('epoch')
    ax_a.set_ylabel('%')
    ax_a.set_ylim(0, 105)
    ax_a.legend()
    ax_a.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'loss_stage2_100events.pdf'), bbox_inches='tight')
    plt.show()
else:
    print(f"Stage 2 (100-event) model not found — skipping Plot 9.")

# ================================================================================
# ================= PLOT 10 — Stage 2 training loss (1800-event model) ===========


if os.path.exists(S2_1800_PATH):
    ckpt         = torch.load(S2_1800_PATH, map_location='cpu')
    ckpt = torch.load(S2_1800_PATH, map_location='cpu')
    print(ckpt.keys())  
    train_losses = ckpt['train_losses']
    val_losses   = ckpt['val_losses']
    val_ious     = ckpt['val_ious']
    val_overlaps = ckpt['val_overlaps']
    epochs       = range(1, len(train_losses) + 1)


    fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))

    ax_l.plot(epochs, train_losses, color='#0077BB', label='train')
    ax_l.plot(epochs, val_losses,   color='#009988', label='val')
    ax_l.set_xlabel('epoch')
    ax_l.set_ylabel('BCE loss')
    ax_l.legend()
    ax_l.grid(True, alpha=0.3, linestyle='--')

    ax_a.plot(epochs, [v * 100 for v in val_ious],     color='#0077BB', label='IoU %')
    ax_a.plot(epochs, [v * 100 for v in val_overlaps], color='#EE7733', label='Overlap %', linestyle='--')
    ax_a.set_xlabel('epoch')
    ax_a.set_ylabel('%')
    ax_a.set_ylim(0, 101) 
    ax_a.legend()
    ax_a.grid(True, alpha=0.3, linestyle='--')

    plt.tight_layout()
    plt.show()
else:
    print(f"Stage 2 (1800-event) model not found — skipping Plot 10.")




seeds = [0, 1, 2, 3, 4]
ious  = [65.6, 62.1, 60.1, 60.7, 64.9]
mean  = np.mean(ious)
std   = np.std(ious)

fig, ax = plt.subplots(figsize=(7, 4))

ax.bar(seeds, ious, color='#0077BB', alpha=0.7, label='Random subset (1298 samples)')
ax.axhline(mean, color='#0077BB', lw=1.5, ls='--', label=f'Subset mean {mean:.1f}%')
ax.axhline(90,   color='#CC3311', lw=1.5, ls='--', label='Original 100-event result (90%)')
ax.axhline(70,   color='#009988', lw=1.5, ls='--', label='1800-event model (70%)')

ax.fill_between([-0.5, 4.5], mean - std, mean + std,
                alpha=0.15, color='#0077BB', label=f'$\pm$1 std ({std:.1f}%)')

ax.set_xticks(seeds)
ax.set_xticklabels([f'Seed {s}' for s in seeds])
ax.set_ylabel('Best IoU %')
ax.set_ylim(50, 100)
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3, linestyle='--')

plt.tight_layout()
plt.show()



if os.path.exists(S2_100_PATH) and os.path.exists(S2_1800_PATH):

    ckpt_100  = torch.load(S2_100_PATH,  map_location='cpu')
    ckpt_1800 = torch.load(S2_1800_PATH, map_location='cpu')

    iou_100  = [v * 100 for v in ckpt_100['val_ious']]
    iou_1800 = [v * 100 for v in ckpt_1800['val_ious']]

    epochs_100  = np.arange(1, len(iou_100)  + 1)
    epochs_1800 = np.arange(1, len(iou_1800) + 1)
    epochs_var  = np.arange(1, 51)

    all_ious = np.array([
        [0.0, 0.0, 2.2, 53.2, 54.5, 54.9, 55.9, 56.9, 56.1, 57.0, 60.2, 57.2, 58.8, 57.0, 60.5, 51.9, 57.9, 58.7, 60.6, 57.9, 58.6, 61.3, 58.0, 61.7, 60.3, 59.7, 63.4, 61.2, 61.3, 61.0, 62.0, 61.5, 58.5, 60.5, 60.9, 59.3, 61.5, 61.1, 59.9, 62.7, 60.2, 60.2, 62.7, 60.9, 60.5, 61.7, 62.0, 61.1, 61.0, 65.6],
        [1.5, 0.6, 3.5, 48.4, 52.3, 53.9, 57.3, 54.4, 58.7, 57.3, 58.3, 50.8, 55.9, 56.9, 56.8, 57.2, 57.2, 60.4, 61.5, 58.4, 60.2, 61.6, 58.9, 59.5, 58.7, 61.1, 61.3, 60.5, 61.9, 59.7, 58.2, 57.6, 61.0, 62.1, 61.2, 61.2, 59.0, 59.1, 61.4, 59.1, 60.6, 60.6, 61.6, 60.1, 60.0, 59.5, 61.6, 59.4, 61.0, 58.7],
        [0.0, 0.0, 4.3, 49.8, 55.0, 53.8, 54.6, 55.3, 54.6, 53.9, 55.1, 50.0, 52.4, 55.4, 55.4, 55.5, 53.8, 55.0, 56.0, 52.1, 52.2, 55.1, 55.3, 54.0, 55.0, 53.5, 52.4, 53.4, 55.2, 54.6, 55.9, 55.8, 57.0, 58.0, 55.1, 57.4, 54.7, 56.2, 56.6, 58.0, 55.8, 54.6, 55.6, 57.4, 56.7, 56.7, 57.1, 58.5, 60.1, 55.2],
        [0.0, 0.0, 0.3, 52.4, 54.2, 54.5, 57.4, 58.1, 54.6, 60.7, 51.5, 52.9, 57.5, 51.1, 55.1, 54.3, 51.6, 55.4, 54.8, 56.3, 52.6, 56.0, 53.7, 56.3, 52.5, 55.4, 56.2, 52.5, 56.4, 54.6, 54.6, 56.0, 55.0, 55.8, 58.7, 54.6, 56.0, 54.3, 56.6, 55.2, 55.7, 53.8, 56.4, 55.5, 56.9, 54.1, 58.6, 55.5, 55.4, 56.2],
        [2.6, 0.0, 16.4, 54.7, 53.5, 56.0, 56.3, 57.0, 57.6, 56.3, 58.1, 56.7, 60.7, 59.1, 57.4, 60.7, 56.7, 58.1, 60.3, 62.7, 60.3, 61.0, 60.6, 61.4, 61.2, 61.6, 60.5, 64.0, 61.1, 61.1, 60.3, 61.1, 61.3, 62.9, 61.6, 61.7, 63.1, 60.0, 58.4, 62.5, 60.0, 64.9, 61.1, 60.7, 60.5, 62.1, 60.1, 61.5, 60.0, 62.4],
    ])
    mean_iou = all_ious.mean(axis=0)

    fig, ax = plt.subplots(figsize=(9, 5))

    # variance test curves
    for iou in all_ious:
        ax.plot(epochs_var, iou, color='#0077BB', alpha=0.35, lw=0.8)
    ax.plot(epochs_var, mean_iou, color='#0077BB', lw=1.5,
            label=f'Variance test mean (1298 samples, 5 seeds)')

    # 100-event and 1800-event model curves
    ax.plot(epochs_100,  iou_100,  color='#CC3311', lw=1.5,
            label='100-event model')
    ax.plot(epochs_1800, iou_1800, color='#009988', lw=1.5,
            label='1800-event model')

    ax.set_xlabel('Epoch')
    ax.set_ylabel('IoU %')
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(fontsize=8, frameon=True, edgecolor='black',
              fancybox=False, framealpha=1.0)

    plt.tight_layout()
    plt.show()

else:
    print("One or both model checkpoints not found — skipping variance curve plot.")



















# ================================================================================
# ================= PLOT 11 — Stage 2 training loss (matrices10, no-aug rphi) ====

# S2_10_PATH = os.path.join(BASE_DIR, "matrices10", "model_stage2.pt")

# if os.path.exists(S2_10_PATH):
#     ckpt         = torch.load(S2_10_PATH, map_location='cpu')
#     train_losses = ckpt['train_losses']
#     val_losses   = ckpt['val_losses']
#     val_ious     = ckpt['val_ious']
#     val_overlaps = ckpt['val_overlaps']
#     epochs       = range(1, len(train_losses) + 1)

#     fig, (ax_l, ax_a) = plt.subplots(1, 2, figsize=(11, 4))

#     ax_l.plot(epochs, train_losses, color='#0077BB', label='train')
#     ax_l.plot(epochs, val_losses,   color='#009988', label='val')
#     ax_l.set_xlabel('epoch')
#     ax_l.set_ylabel('BCE loss')
#     ax_l.legend()
#     ax_l.grid(True, alpha=0.3, linestyle='--')

#     ax_a.plot(epochs, [v * 100 for v in val_ious],     color='#0077BB', label='IoU %')
#     ax_a.plot(epochs, [v * 100 for v in val_overlaps], color='#EE7733', label='Overlap %', linestyle='--')
#     ax_a.set_xlabel('epoch')
#     ax_a.set_ylabel('%')
#     ax_a.set_ylim(0, 105)
#     ax_a.legend()
#     ax_a.grid(True, alpha=0.3, linestyle='--')

#     plt.tight_layout()
#     plt.savefig(os.path.join(OUT_DIR, 'loss_stage2_matrices10.pdf'), bbox_inches='tight')
#     plt.show()
# else:
#     print(f"Stage 2 (matrices10) model not found — skipping Plot 11.")






# # ================================================================================
# # ================= PLOT 6 — Stage 2 phi predictions on one event ================

# if os.path.exists(S1_100_PATH) and os.path.exists(S2_100_PATH):
#     model_s1 = ConeFilterSNN(_SNN_R_BINS, bs=1)
#     model_s1.load_state_dict(torch.load(S1_100_PATH)['model_state'])
#     model_s1.eval()

#     model_s2 = PhiFinderSNN(bs=1)
#     model_s2.load_state_dict(torch.load(S2_100_PATH)['model_state'])
#     model_s2.eval()

#     event_path = os.path.join(DATA_DIR, EVENT_PREFIX)
#     scan_event_full(event_path, model_s1, model_s2,
#                     thresh_s1=0.5, thresh_s2=0.6, plot=True)
# else:
#     print(f"Stage 1/2 models not found — skipping Plot 6.")

# # ================================================================================
# # ================= PLOT 6b — Stage 2 phi predictions (1800-event model) =========

# if os.path.exists(S1_1800_PATH) and os.path.exists(S2_1800_PATH):
#     model_s1 = ConeFilterSNN(_SNN_R_BINS, bs=1)
#     model_s1.load_state_dict(torch.load(S1_1800_PATH, map_location='cpu')['model_state'])
#     model_s1.eval()

#     model_s2 = PhiFinderSNN(bs=1)
#     model_s2.load_state_dict(torch.load(S2_1800_PATH)['model_state'])
#     model_s2.eval()

#     event_path = os.path.join(DATA_DIR, EVENT_PREFIX)
#     scan_event_full(event_path, model_s1, model_s2,
#                     thresh_s1=0.5, thresh_s2=0.6, plot=True)
# else:
#     print(f"Stage 1/2 (1800-event) models not found — skipping Plot 6b.")

# # ================================================================================
# # ================= PLOT 7 — Training loss (100-event model) =====================