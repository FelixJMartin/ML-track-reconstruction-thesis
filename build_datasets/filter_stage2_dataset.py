# ================================================================================
# filter_stage2_dataset.py
# Runs Stage 1 filter to produce X_rphi_s2 and y_s2_filt for 1% noise, buffer 150
# ================================================================================

import sys
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\SNN")
sys.path.append(r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\training")

import os
import numpy as np
import torch

from train_s1_1000 import ConeFilterSNN

SNN_R_BINS = 150

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# ── CONFIG ────────────────────────────────────────────────────────────────────
THRESH_S1    = 0.5
matrices_dir = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\matrices_150_1pct"
model_path   = r"C:\Users\felix\Downloads\Skola\Kand\BACHELOR_CODE\data\models_150\s1_h512_256_lr0.001_dr0.1_noise1pct.pt"

# ── LOAD MATRICES ─────────────────────────────────────────────────────────────
print("Loading matrices...")
X_rz   = np.load(os.path.join(matrices_dir, "X_rz.npy"))
X_rphi = np.load(os.path.join(matrices_dir, "X_rphi.npy"))
y_s1   = np.load(os.path.join(matrices_dir, "y_stage1.npy"))
y_s2   = np.load(os.path.join(matrices_dir, "y_stage2.npy"))
print(f"Loaded: {len(X_rz)} cones  signal={int(y_s1.sum())}  noise={int((1-y_s1).sum())}")

# ── LOAD STAGE 1 MODEL ────────────────────────────────────────────────────────
print("Loading Stage 1 model...")
model_s1 = ConeFilterSNN(SNN_R_BINS, h1=512, h2=256, bs=1)
model_s1.load_state_dict(
    torch.load(model_path, map_location=torch.device('cpu'))["model_state"]
)
model_s1.eval()
model_s1.set_bs(1)

# ── RUN STAGE 1 FILTER ────────────────────────────────────────────────────────
print("Running Stage 1 filter...")
probs = []
with torch.no_grad():
    for i in range(len(X_rz)):
        p = torch.sigmoid(
            model_s1(torch.tensor(X_rz[i]).unsqueeze(0))
        ).item()
        probs.append(p)
        if i % 500 == 0:
            print(f"  {i}/{len(X_rz)}...")

probs       = np.array(probs)
passed_mask = probs > THRESH_S1

X_rphi_s2 = X_rphi[passed_mask]
y_s2_filt = y_s2[passed_mask]

# ── STATS ─────────────────────────────────────────────────────────────────────
n_passed  = passed_mask.sum()
n_fp      = (y_s1[passed_mask] == 0).sum()
sig_rows  = (y_s2_filt.sum(axis=1) > 0).sum()
fp_rows   = (y_s2_filt.sum(axis=1) == 0).sum()

print(f"\nStage 1 passed : {n_passed}/{len(X_rz)} cones")
print(f"Contamination  : {100*n_fp/n_passed:.1f}% false positives")
print(f"Stage 2 dataset: {len(X_rphi_s2)} rows  signal={sig_rows}  false_pos={fp_rows}")

# ── SAVE ──────────────────────────────────────────────────────────────────────
out_rphi = os.path.join(matrices_dir, "X_rphi_s2.npy")
out_y    = os.path.join(matrices_dir, "y_s2_filt.npy")
np.save(out_rphi, X_rphi_s2)
np.save(out_y,    y_s2_filt)
print(f"\nSaved to:")
print(f"  {out_rphi}")
print(f"  {out_y}")