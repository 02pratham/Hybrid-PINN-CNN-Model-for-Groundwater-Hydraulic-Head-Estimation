#!/usr/bin/env python3
"""
data_prep_cnn.py

Scans a directory of .h5 PINN outputs and writes per-sample .npz files containing:
 - input: fields[3] (log10(k)) -> saved as `logk` (float32)
 - target: h_pinn (PINN prediction) -> saved as `hpinn` (float32)

Outputs (default):
 - ./data/cnn_npz/*.npz (one file per sample)
 - index file: all_list.txt listing all generated .npz file paths (used by training script to split)

Usage:
    python data_prep_cnn.py

Adjust DATA_DIR and OUT_DIR below as needed.
"""
import os
import glob
import h5py
import numpy as np

# -------------------------
# User settings
# -------------------------
DATA_DIR = "./data/pinn_train_data"          # where your .h5 PINN outputs are
OUT_DIR = "./data/cnn_train_data"            # where per-sample npz files and lists will be saved
FORCE_OVERWRITE = False
# -------------------------

os.makedirs(OUT_DIR, exist_ok=True)

pattern = os.path.join(DATA_DIR, "*.h5")
files = sorted(glob.glob(pattern))
if len(files) == 0:
    raise SystemExit(f"No .h5 files found in {DATA_DIR}")

print(f"Found {len(files)} h5 samples in {DATA_DIR}")

# Helper to extract arrays and sanity-check shapes
def extract_from_h5(path):
    with h5py.File(path, 'r') as f:
        # log_k is stored in fields[3]
        fields = f['fields'][:]
        if fields.shape[0] < 4:
            raise ValueError(f"fields has unexpected channel count in {path}")
        logk = fields[3].astype(np.float32)
        # h_pinn should already be written by your PINN script
        if 'h_pinn' not in f:
            raise ValueError(f"h_pinn not present in {path}; run PINN training first")
        hpinn = f['h_pinn'][:].astype(np.float32)
        # ensure shapes match
        if logk.shape != hpinn.shape:
            raise ValueError(f"shape mismatch in {path}: logk {logk.shape} vs hpinn {hpinn.shape}")
    return logk, hpinn

# Save function

def save_npz(out_path, logk, hpinn):
    # save arrays with channels first shape (C,H,W) -> (1,H,W)
    np.savez_compressed(out_path, logk=logk[None, :, :], hpinn=hpinn[None, :, :])

# Process all files and write per-sample npz into OUT_DIR
npz_paths = []
for p in files:
    name = os.path.splitext(os.path.basename(p))[0]
    out_file = os.path.join(OUT_DIR, name + '.npz')
    if os.path.exists(out_file) and not FORCE_OVERWRITE:
        npz_paths.append(out_file)
        continue
    try:
        logk, hpinn = extract_from_h5(p)
        save_npz(out_file, logk, hpinn)
        npz_paths.append(out_file)
    except Exception as e:
        print(f"Warning: failed on {p}: {e}")

# Write a single index file listing all .npz files (training script will handle splitting)
index_file = os.path.join(OUT_DIR, 'all_list.txt')
with open(index_file, 'w') as f:
    for p in npz_paths:
        f.write(p + '')

print('Data preparation complete. Wrote', len(npz_paths), 'npz files to', OUT_DIR)
print('Index saved to', index_file)