# ================================================================================
# pipeline_eval_500.py
# Evaluates the full two-stage pipeline on 500 held-out test events.
# Computes per-event metrics and reports mean ± 95% CI across events.
# ================================================================================

import sys, os
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\SNN")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\build_datasets")
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy import stats
from trackml.dataset import load_event
from snns import RadLIFLayer
from Functions import preprocess_particles
from build_data8500 import (
    make_rz_grid, make_rphi_grid,
    SNN_R_BINS, SNN_PHI_BINS, CONE_PHI_RANGES, SAMPLE_FRACTION,
)

# ── CONFIG ────────────────────────────────────────────────────────────────────
DATA_DIR  = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\Data\train_5"
S1_PATH   = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\s1_8500_ep115.pt"
S2_PATH   = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices8500\s2_h64_32_final.pt"
THRESH_S1 = 0.3
THRESH_S2 = 0.1
SNN_INPUT_SIZE = SNN_R_BINS + 1
test_ids  = [f"event{i:09d}" for i in range(9500, 10000)]

np.random.seed(42)   # fixed — one run, variance comes from events not sampling

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

def ci95(values):
    n  = len(values)
    m  = np.mean(values)
    se = stats.sem(values)
    h  = se * stats.t.ppf(0.975, df=n - 1)
    return float(m), float(h)

# ── MODELS ────────────────────────────────────────────────────────────────────
class ConeFilterSNN(nn.Module):
    def __init__(self, input_size=100, h1=64, h2=32, bs=1):
        super().__init__()
        self.lif1 = RadLIFLayer(input_size, h1, bs, normalization="batchnorm", dropout=0.2)
        self.lif2 = RadLIFLayer(h1, h2,     bs, normalization="batchnorm", dropout=0.2)
        self.fc   = nn.Linear(h2, 1)
    def forward(self, x):
        return self.fc(self.lif2(self.lif1(x)).mean(dim=1)).squeeze(-1)
    def set_bs(self, bs):
        self.lif1.batch_size = bs
        self.lif2.batch_size = bs

class PhiFinderSNN(nn.Module):
    def __init__(self, h1=64, h2=32, bs=1):
        super().__init__()
        self.lif1 = RadLIFLayer(SNN_INPUT_SIZE, h1, bs, normalization="batchnorm", dropout=0.2)
        self.lif2 = RadLIFLayer(h1, h2,         bs, normalization="batchnorm", dropout=0.2)
        self.fc   = nn.Linear(h2, 1)
    def forward(self, x):
        x = x.transpose(1, 2)
        return self.fc(self.lif2(self.lif1(x))).squeeze(-1)
    def set_bs(self, bs):
        self.lif1.batch_size = bs
        self.lif2.batch_size = bs

def load_model(cls, path, **kwargs):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    m = cls(**kwargs)
    m.load_state_dict(ckpt["model_state"])
    return m.eval()

model_s1 = load_model(ConeFilterSNN, S1_PATH)
model_s2 = load_model(PhiFinderSNN,  S2_PATH)
print("Models loaded.")

# ── PER-EVENT COLLECTORS ──────────────────────────────────────────────────────
per_event = {
    's1_recall':      [],
    's1_specificity': [],
    's1_precision':   [],
    's2_precision':   [],
    's2_recall':      [],
    's2_iou':         [],
    'hit_reduction':  [],
}

# global totals for the summary printout (unchanged from original)
total_baseline_hits  = 0
total_s2_hits        = 0
total_signal_in      = 0
total_signal_s2      = 0
total_sig_cones      = 0
total_sig_cones_pass = 0
total_noi_cones      = 0
total_noi_cones_rej  = 0
total_pred_bins      = 0
total_correct_bins   = 0
all_ious             = []
n_processed          = 0
n_skipped            = 0

# ── MAIN LOOP ─────────────────────────────────────────────────────────────────
for event_id in test_ids:
    try:
        h, _, p, t = load_event(os.path.join(DATA_DIR, event_id))
        h["r"]   = np.hypot(h["x"], h["y"])
        h["phi"] = np.arctan2(h["y"], h["x"])
        p_cond, _, _ = preprocess_particles(10, 10, p, h)

        sig_hit_ids = set(t[t["particle_id"].isin(
            p_cond["particle_id"])]["hit_id"].values)

        noise_hits    = h[~h.hit_id.isin(sig_hit_ids)]
        noise_sampled = noise_hits[np.random.rand(len(noise_hits)) < SAMPLE_FRACTION].copy()
        h_sampled     = pd.concat([noise_sampled, h[h.hit_id.isin(sig_hit_ids)]])

        hits_cones_full = split_cones(h)
        hits_cones_disp = split_cones(h_sampled)
        sig_cones       = [signal_hits_in_cone(c, t, p) for c in hits_cones_full]
        baseline_parts  = [model_input(hits_cones_disp[i], sig_cones[i]) for i in range(8)]

        n_sig_in = int(h_sampled["hit_id"].isin(sig_hit_ids).sum())
        total_baseline_hits += len(h_sampled)
        total_signal_in     += n_sig_in

        # ── Stage 1 ──
        predicted_signal_idx = []

        e_sig_cones      = 0
        e_sig_cones_pass = 0
        e_noi_cones      = 0
        e_noi_cones_rej  = 0
        e_noi_cones_pass = 0

        for i, ((_, _, _), sig) in enumerate(zip(OCTANTS, sig_cones)):
            inp  = model_input(hits_cones_disp[i], sig)
            grid = torch.tensor(make_rz_grid(inp["r"].values, inp["z"].values),
                                dtype=torch.float32).unsqueeze(0)
            with torch.no_grad():
                conf = torch.sigmoid(model_s1(grid)).item()
            is_sig_cone = len(sig) > 0
            passed      = conf > THRESH_S1

            if is_sig_cone:
                e_sig_cones += 1
                if passed:
                    e_sig_cones_pass += 1
            else:
                e_noi_cones += 1
                if not passed:
                    e_noi_cones_rej += 1
                else:
                    e_noi_cones_pass += 1

            if passed:
                predicted_signal_idx.append(i)

        # global S1 totals
        total_sig_cones      += e_sig_cones
        total_sig_cones_pass += e_sig_cones_pass
        total_noi_cones      += e_noi_cones
        total_noi_cones_rej  += e_noi_cones_rej

        # per-event S1 metrics
        if e_sig_cones > 0:
            per_event['s1_recall'].append(e_sig_cones_pass / e_sig_cones)
        if e_noi_cones > 0:
            per_event['s1_specificity'].append(e_noi_cones_rej / e_noi_cones)
        denom = e_sig_cones_pass + e_noi_cones_pass
        if denom > 0:
            per_event['s1_precision'].append(e_sig_cones_pass / denom)

        # ── Stage 2 ──
        octant_phi_results = {}
        event_ious         = []

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
            octant_phi_results[i] = pred_bins

            if len(sig_cones[i]) > 0:
                phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
                true_active = np.zeros(SNN_PHI_BINS, dtype=bool)
                true_active[phi_bin_idx(sig_cones[i], phi_min, phi_max)] = True
                pred_active = np.zeros(SNN_PHI_BINS, dtype=bool)
                pred_active[pred_bins] = True
                intersection = (true_active & pred_active).sum()
                union        = (true_active | pred_active).sum()
                if union > 0:
                    iou = intersection / union
                    event_ious.append(iou)
                    all_ious.append(iou)
                total_pred_bins    += pred_active.sum()
                total_correct_bins += intersection

        s2_parts = []
        for i in predicted_signal_idx:
            _, _, (sx, sy, _) = OCTANTS[i]
            cone = baseline_parts[i]
            bins = octant_phi_results[i]
            if not len(bins):
                continue
            phi_min, phi_max = CONE_PHI_RANGES[(sx, sy)]
            s2_parts.append(cone[np.isin(phi_bin_idx(cone, phi_min, phi_max), bins)])

        s2_hits  = pd.concat(s2_parts) if s2_parts else pd.DataFrame(columns=h.columns)
        n_s2     = len(s2_hits)
        n_s2_sig = int(s2_hits["hit_id"].isin(sig_hit_ids).sum()) if n_s2 > 0 else 0
        total_s2_hits   += n_s2
        total_signal_s2 += n_s2_sig
        
        if len(h_sampled) > 0:
            per_event['hit_reduction'].append(1 - n_s2 / len(h_sampled))
        
        # per-event S2 metrics
        if n_s2 > 0:
            per_event['s2_precision'].append(n_s2_sig / n_s2)
        if n_sig_in > 0:
            per_event['s2_recall'].append(n_s2_sig / n_sig_in)
        if event_ious:
            per_event['s2_iou'].append(float(np.mean(event_ious)))

        n_processed += 1
        if n_processed % 50 == 0:
            print(f"  {n_processed}/500 events done...", flush=True)

    except Exception as e:
        print(f"  {event_id}: skipped — {e}")
        n_skipped += 1

# ── GLOBAL SUMMARY ────────────────────────────────────────────────────────────
s1_efficiency = 100 * total_sig_cones_pass / total_sig_cones  if total_sig_cones  > 0 else 0
s1_throwrate  = 100 * total_noi_cones_rej  / total_noi_cones  if total_noi_cones  > 0 else 0
s2_hit_recall = 100 * total_signal_s2      / total_signal_in  if total_signal_in  > 0 else 0
s2_bin_prec   = 100 * total_correct_bins   / total_pred_bins  if total_pred_bins  > 0 else 0
mean_iou      = 100 * np.mean(all_ious)                       if all_ious         else 0
hit_reduction = 100 * (1 - total_s2_hits   / total_baseline_hits) if total_baseline_hits > 0 else 0
final_prec    = 100 * total_signal_s2      / total_s2_hits    if total_s2_hits    > 0 else 0

# per-event 1-sigma errors
def sigma1(vals):
    return float(np.std(vals, ddof=1) / np.sqrt(len(vals))) if vals else 0.0

s1_eff_err  = 100 * sigma1(per_event['s1_recall'])
s1_thr_err  = 100 * sigma1(per_event['s1_specificity'])
iou_err     = 100 * sigma1(per_event['s2_iou'])
s2_prec_err = 100 * sigma1(per_event['s2_precision'])
recall_err  = 100 * sigma1(per_event['s2_recall'])
hitr_err    = 100 * sigma1(per_event['hit_reduction'])
prec_err    = 100 * sigma1(per_event['s2_precision'])

print(f"\n{'='*55}")
print(f"PIPELINE EVALUATION — {n_processed} events  ({n_skipped} skipped)")
print(f"  Thresholds: S1={THRESH_S1}  S2={THRESH_S2}")
print(f"{'='*55}")
print(f"  S1 signal efficiency     : {s1_efficiency:.1f} ± {s1_eff_err:.1f}%")
print(f"  S1 noise throwrate       : {s1_throwrate:.1f} ± {s1_thr_err:.1f}%")
print(f"  S2 mean IoU              : {mean_iou:.1f} ± {iou_err:.1f}%")
print(f"  S2 bin precision         : {s2_bin_prec:.1f} ± {s2_prec_err:.1f}%")
print(f"  End-to-end signal recall : {s2_hit_recall:.1f} ± {recall_err:.1f}%")
print(f"  End-to-end hit precision : {final_prec:.1f} ± {prec_err:.1f}%")
print(f"  Hit reduction            : {hit_reduction:.1f} ± {hitr_err:.1f}%")
print(f"  Hits in → out            : {total_baseline_hits:,} → {total_s2_hits:,}")

# ── PER-EVENT STATS ───────────────────────────────────────────────────────────
print(f"\n{'='*55}")
print(f"PER-EVENT METRICS  (mean ± 95% CI half-width,  n events per metric)")
print(f"{'='*55}")

metrics_summary = {}
for key, vals in per_event.items():
    mean, err = ci95(vals)
    metrics_summary[key] = {'mean': mean, 'err': err, 'n': len(vals)}
    print(f"  {key:<20s}: {mean:.3f} ± {err:.3f}  (n={len(vals)})")

print(f"\nmetrics_summary = {metrics_summary}")