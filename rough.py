# import torch
# print(torch.version.cuda)
# print(torch.cuda.is_available())

# import h5py

# path = "./data/pinn_train_data/sample_00000002.h5"  # change path

# with h5py.File(path, "r") as f:
#     print("\n=== DATASETS ===")
#     for key in f.keys():
#         obj = f[key]
#         if isinstance(obj, h5py.Dataset):
#             print(f"{key:20s} shape={obj.shape} dtype={obj.dtype}")
#         else:
#             print(f"{key:20s} (unknown type)")

#     print("\n=== ATTRIBUTES ===")
#     for attr in f.attrs:
#         print(f"{attr}: {f.attrs[attr]}")

# quick_check_split.py
# import os, random

# master = "./data/cnn_train_data/all_list.txt"   # adjust if different
# with open(master,'r') as f:
#     all_paths = [l.strip() for l in f if l.strip()]
# print("Total listed samples (N):", len(all_paths))
# print("First 5 entries:", all_paths[:5])

# # print the fractions you pass to train script (example values)
# val_frac = 0.15
# test_frac = 0.15
# N = len(all_paths)
# n_test = int(round(N * test_frac))
# n_val  = int(round(N * val_frac))
# n_train = N - n_val - n_test
# print("Computed:", "n_train", n_train, "n_val", n_val, "n_test", n_test)



# fix_all_list_from_folder.py
import os, glob

# EDIT these if your paths are different
NPZ_DIR = "./data/cnn_train_data"   # folder that contains sample_000000XX.npz
OUT_INDEX = os.path.join(NPZ_DIR, "all_list.txt")  # where to write

# find npz files (sorted)
files = sorted(glob.glob(os.path.join(NPZ_DIR, "*.npz")))
if not files:
    raise SystemExit(f"No .npz files found in {NPZ_DIR}")

# write one per line
with open(OUT_INDEX, "w") as f:
    for p in files:
        f.write(p.replace("\\", "/") + "\n")   # normalize slashes for portability

print(f"Wrote {len(files)} entries to {OUT_INDEX}")
print("First 10 entries:")
for p in files[:10]:
    print(" ", p)

