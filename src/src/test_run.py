# #!/usr/bin/env python3
# """
# train_all_pinn_gpu_opt.py

# Same training pipeline as your provided script, with GPU/CPU auto-selection,
# cuDNN autotuner and mixed-precision (AMP) during Adam to improve GPU utilization.

# Behavior:
#  - If CUDA is available and DEVICE is set to "cuda", runs on GPU with AMP.
#  - If CUDA is not available or DEVICE="cpu", runs on CPU.
#  - Prints device info at start and for each sample.
#  - Frees GPU cache after each sample.

# Other training logic (losses, L-BFGS, saving) unchanged.
# """
# import os
# import time
# import json
# import traceback
# import glob

# import h5py
# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.interpolate import RegularGridInterpolator

# import torch
# import torch.nn as nn
# import torch.optim as optim

# # -------------------------
# # USER SETTINGS (edit if needed)
# # -------------------------
# DATA_DIR = "./data/pinn_train_data"        # directory containing .h5 samples
# OUT_ROOT = "./results/pinn_results"        # where per-sample outputs are written
# DEVICE = "cuda"                            # "cpu" or "cuda" (script will fallback if cuda not available)
# STEPS_ADAM = 5000
# LR_ADAM = 1e-3
# LBFGS_STEPS = 60
# HIDDEN = 64
# NLAYERS = 6
# LAMBDA_PDE = 1.0
# LAMBDA_BC = 1.0
# LAMBDA_SUP = 1.0   # set 0.0 for physics-only PINN labeling
# VERBOSE = True
# # -------------------------

# # -------------------------
# # Utilities: load / normalize
# # -------------------------
# def load_sample(h5_path):
#     with h5py.File(h5_path, "r") as f:
#         fields = f['fields'][:]               # (4, nz, nx)
#         x = f['x'][:].astype(float)           # (nx,)
#         z = f['z'][:].astype(float)           # (nz,)
#         coll_x = f['collocation/x'][:]
#         coll_z = f['collocation/z'][:]
#         collocation = np.column_stack([coll_x, coll_z])  # (N,2) in (x,z)
#         collocation_k = f['collocation/k_at_collocation'][:]
#         bc_x = f['bc/x'][:]
#         bc_z = f['bc/z'][:]
#         bc_vals = f['bc/h'][:]
#         bc_pts = np.column_stack([bc_x, bc_z])
#         h_ref = f.get('h_ref')
#         if h_ref is not None:
#             h_ref = h_ref[:]
#         meta = {}
#         try:
#             meta = json.loads(f.attrs.get('meta', "{}"))
#         except Exception:
#             try:
#                 meta = eval(f.attrs.get('meta', "{}"))
#             except Exception:
#                 meta = {}
#     # normalization constants
#     x_mean = 0.5*(x.min() + x.max())
#     x_scale = 0.5*(x.max() - x.min()) if (x.max() - x.min()) != 0 else 1.0
#     z_mean = 0.5*(z.min() + z.max())
#     z_scale = 0.5*(z.max() - z.min()) if (z.max() - z.min()) != 0 else 1.0
#     k_grid = fields[3]   # log10(k)
#     k_mean = float(np.mean(k_grid))
#     k_std  = float(np.std(k_grid)) + 1e-12
#     interp_k = RegularGridInterpolator((z, x), k_grid, bounds_error=False, fill_value=None)
#     return {
#         'fields': fields, 'x': x, 'z': z,
#         'collocation': collocation, 'collocation_k': collocation_k,
#         'bc_pts': bc_pts, 'bc_vals': bc_vals,
#         'h_ref': h_ref, 'meta': meta,
#         'norm': {'x_mean': x_mean, 'x_scale': x_scale, 'z_mean': z_mean, 'z_scale': z_scale, 'k_mean': k_mean, 'k_std': k_std},
#         'interp_k': interp_k
#     }

# def normalize_inputs(x, z, k, norm):
#     x_n = (x - norm['x_mean']) / (norm['x_scale'] + 1e-12)
#     z_n = (z - norm['z_mean']) / (norm['z_scale'] + 1e-12)
#     k_n = (k - norm['k_mean']) / (norm['k_std'] + 1e-12)
#     return np.stack([x_n, z_n, k_n], axis=1)

# def to_torch(arr, device, requires_grad=False, dtype=torch.float32):
#     t = torch.tensor(arr, device=device, dtype=dtype)
#     t.requires_grad = requires_grad
#     return t

# # -------------------------
# # Model & PDE residual
# # -------------------------
# class PINN_MLP(nn.Module):
#     def __init__(self, in_dim=3, hidden=64, n_layers=6, activation='tanh'):
#         super().__init__()
#         act = nn.Tanh if activation == 'tanh' else nn.ReLU
#         layers = [nn.Linear(in_dim, hidden), act()]
#         for _ in range(n_layers-1):
#             layers += [nn.Linear(hidden, hidden), act()]
#         layers.append(nn.Linear(hidden, 1))
#         self.net = nn.Sequential(*layers)
#         for m in self.net:
#             if isinstance(m, nn.Linear):
#                 nn.init.xavier_normal_(m.weight)
#                 nn.init.zeros_(m.bias)
#     def forward(self, x):
#         return self.net(x).squeeze(-1)

# def pde_residual(model, x_t, z_t, k_t):
#     inp = torch.stack([x_t, z_t, k_t], dim=1)
#     inp.requires_grad_(True)
#     h_pred = model(inp)
#     grads = torch.autograd.grad(h_pred, inp, grad_outputs=torch.ones_like(h_pred), create_graph=True)[0]
#     dh_dx = grads[:,0]; dh_dz = grads[:,1]
#     k = k_t
#     kdh_dx = k * dh_dx
#     kdh_dz = k * dh_dz
#     grad1 = torch.autograd.grad(kdh_dx, inp, grad_outputs=torch.ones_like(kdh_dx), create_graph=True)[0]
#     grad2 = torch.autograd.grad(kdh_dz, inp, grad_outputs=torch.ones_like(kdh_dz), create_graph=True)[0]
#     return grad1[:,0] + grad2[:,1]

# # -------------------------
# # Single-sample trainer
# # -------------------------
# def train_single_sample(h5_path, out_dir,
#                         device='cpu',
#                         hidden=HIDDEN, n_layers=NLAYERS,
#                         steps_adam=STEPS_ADAM, lr_adam=LR_ADAM,
#                         lbfgs_steps=LBFGS_STEPS,
#                         lambda_pde=LAMBDA_PDE, lambda_bc=LAMBDA_BC, lambda_sup=LAMBDA_SUP):
#     os.makedirs(out_dir, exist_ok=True)
#     if VERBOSE:
#         print(f"\n--- Training sample: {h5_path}")
#     data = load_sample(h5_path)
#     fields = data['fields']; x_grid = data['x']; z_grid = data['z']
#     collocation = data['collocation']; collocation_k = data['collocation_k']
#     bc_pts = data['bc_pts']; bc_vals = data['bc_vals']
#     h_ref = data['h_ref']; norm = data['norm']; interp_k = data['interp_k']

#     device = torch.device(device)

#     # collocation inputs
#     Xc = collocation[:,0]; Zc = collocation[:,1]; Kc = collocation_k
#     inp_coll = normalize_inputs(Xc, Zc, Kc, norm)
#     x_coll_t = to_torch(inp_coll[:,0], device, requires_grad=True)
#     z_coll_t = to_torch(inp_coll[:,1], device, requires_grad=True)
#     k_coll_t = to_torch(inp_coll[:,2], device, requires_grad=True)

#     # BC inputs
#     Xb = bc_pts[:,0]; Zb = bc_pts[:,1]
#     kb = interp_k(np.column_stack([Zb, Xb]))
#     inp_bc = normalize_inputs(Xb, Zb, kb, norm)
#     x_bc_t = to_torch(inp_bc[:,0], device, requires_grad=True)
#     z_bc_t = to_torch(inp_bc[:,1], device, requires_grad=True)
#     k_bc_t = to_torch(inp_bc[:,2], device, requires_grad=True)
#     bc_vals_t = to_torch(bc_vals.astype(np.float32), device, requires_grad=False)

#     # supervised grid
#     if (h_ref is not None) and (lambda_sup > 0.0):
#         Xg, Zg = np.meshgrid(x_grid, z_grid)
#         Xg_flat = Xg.ravel(); Zg_flat = Zg.ravel()
#         kg_flat = interp_k(np.column_stack([Zg_flat, Xg_flat]))
#         inp_grid = normalize_inputs(Xg_flat, Zg_flat, kg_flat, norm)
#         xg_t = to_torch(inp_grid[:,0], device, requires_grad=True)
#         zg_t = to_torch(inp_grid[:,1], device, requires_grad=True)
#         kg_t = to_torch(inp_grid[:,2], device, requires_grad=True)
#         h_ref_flat = h_ref.ravel().astype(np.float32)
#         h_ref_t = to_torch(h_ref_flat, device, requires_grad=False)
#     else:
#         xg_t = zg_t = kg_t = h_ref_t = None

#     # model + optimizer
#     model = PINN_MLP(in_dim=3, hidden=hidden, n_layers=n_layers, activation='tanh').to(device)
#     optimizer = optim.Adam(model.parameters(), lr=lr_adam)
#     mse = nn.MSELoss()

#     # Mixed precision (AMP) setup if using CUDA
#     use_amp = (device.type == "cuda")
#     scaler = None
#     if use_amp:
#         try:
#             from torch.cuda.amp import autocast, GradScaler
#             scaler = GradScaler()
#             if VERBOSE:
#                 print("  AMP enabled (mixed precision) on CUDA.")
#         except Exception:
#             scaler = None
#             if VERBOSE:
#                 print("  AMP not available; running in full precision.")

#     if VERBOSE:
#         print("  Starting Adam...")
#     for it in range(steps_adam):
#         optimizer.zero_grad()
#         if use_amp and scaler is not None:
#             with autocast():
#                 res = pde_residual(model, x_coll_t, z_coll_t, k_coll_t)
#                 loss_pde = torch.mean(res**2)
#                 inp_bc_tensor = torch.stack([x_bc_t, z_bc_t, k_bc_t], dim=1)
#                 hbc_pred = model(inp_bc_tensor)
#                 loss_bc = mse(hbc_pred, bc_vals_t)
#                 loss_sup = torch.tensor(0.0, device=device)
#                 if xg_t is not None:
#                     inp_grid = torch.stack([xg_t, zg_t, kg_t], dim=1)
#                     hgrid_pred = model(inp_grid)
#                     loss_sup = mse(hgrid_pred, h_ref_t)
#                 loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_sup * loss_sup
#             scaler.scale(loss).backward()
#             scaler.step(optimizer)
#             scaler.update()
#         else:
#             res = pde_residual(model, x_coll_t, z_coll_t, k_coll_t)
#             loss_pde = torch.mean(res**2)
#             inp_bc_tensor = torch.stack([x_bc_t, z_bc_t, k_bc_t], dim=1)
#             hbc_pred = model(inp_bc_tensor)
#             loss_bc = mse(hbc_pred, bc_vals_t)
#             loss_sup = torch.tensor(0.0, device=device)
#             if xg_t is not None:
#                 inp_grid = torch.stack([xg_t, zg_t, kg_t], dim=1)
#                 hgrid_pred = model(inp_grid)
#                 loss_sup = mse(hgrid_pred, h_ref_t)
#             loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_sup * loss_sup
#             loss.backward()
#             optimizer.step()

#         if VERBOSE and ((it+1) % 500 == 0 or it == 0):
#             l_pde = loss_pde.item() if 'loss_pde' in locals() else 0.0
#             l_bc = loss_bc.item() if 'loss_bc' in locals() else 0.0
#             l_sup = loss_sup.item() if isinstance(loss_sup, torch.Tensor) else 0.0
#             total = loss.item() if 'loss' in locals() else 0.0
#             print(f"    Adam iter {it+1}/{steps_adam}: total={total:.3e} pde={l_pde:.3e} bc={l_bc:.3e} sup={l_sup:.3e}")

#     # L-BFGS fine-tune
#     if VERBOSE:
#         print("  Starting L-BFGS...")
#     optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=50, line_search_fn='strong_wolfe')
#     def closure():
#         optimizer_lbfgs.zero_grad()
#         res = pde_residual(model, x_coll_t, z_coll_t, k_coll_t)
#         loss_pde = torch.mean(res**2)
#         inp_bc_tensor = torch.stack([x_bc_t, z_bc_t, k_bc_t], dim=1)
#         hbc_pred = model(inp_bc_tensor)
#         loss_bc = mse(hbc_pred, bc_vals_t)
#         loss_sup = torch.tensor(0.0, device=device)
#         if xg_t is not None:
#             inp_grid = torch.stack([xg_t, zg_t, kg_t], dim=1)
#             hgrid_pred = model(inp_grid)
#             loss_sup = mse(hgrid_pred, h_ref_t)
#         loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_sup * loss_sup
#         loss.backward()
#         closure.last = (loss.item(), loss_pde.item(), loss_bc.item(), loss_sup.item() if isinstance(loss_sup, torch.Tensor) else 0.0)
#         return loss
#     for i in range(lbfgs_steps):
#         optimizer_lbfgs.step(closure)
#         if VERBOSE and ((i+1) % 10 == 0 or i == 0):
#             print(f"    L-BFGS step {i+1}/{lbfgs_steps}: loss={closure.last[0]:.3e} pde={closure.last[1]:.3e} bc={closure.last[2]:.3e} sup={closure.last[3]:.3e}")

#     # evaluate on full grid
#     Xg, Zg = np.meshgrid(x_grid, z_grid)
#     Xg_flat = Xg.ravel(); Zg_flat = Zg.ravel()
#     kg_flat = interp_k(np.column_stack([Zg_flat, Xg_flat]))
#     inp_grid = normalize_inputs(Xg_flat, Zg_flat, kg_flat, norm)
#     model.eval()
#     with torch.no_grad():
#         inp_grid_t = torch.tensor(inp_grid, device=device, dtype=torch.float32)
#         h_pred_flat = model(inp_grid_t).cpu().numpy()
#     h_pred = h_pred_flat.reshape(Zg.shape)

#     # save h_pinn back to hdf5
#     try:
#         with h5py.File(h5_path, 'a') as f:
#             if 'h_pinn' in f:
#                 del f['h_pinn']
#             f.create_dataset('h_pinn', data=h_pred.astype(np.float32), compression='gzip')
#             f.attrs['pinn_info'] = json.dumps({'hidden': hidden, 'n_layers': n_layers, 'lambda_pde': lambda_pde, 'lambda_bc': lambda_bc, 'lambda_sup': lambda_sup})
#     except Exception as e:
#         print("  Warning: failed to write h_pinn:", e)

#     # save model + diagnostic image
#     model_file = os.path.join(out_dir, 'pinn_model.pt')
#     torch.save(model.state_dict(), model_file)
#     plot_file = os.path.join(out_dir, 'pinn_diag.png')
#     plt.figure(figsize=(12,4))
#     plt.subplot(1,3,1)
#     if h_ref is not None:
#         plt.imshow(h_ref, origin='lower', aspect='auto'); plt.title('h_ref'); plt.colorbar()
#     else:
#         plt.text(0.2,0.5,'No h_ref', fontsize=12)
#     plt.subplot(1,3,2)
#     plt.imshow(h_pred, origin='lower', aspect='auto'); plt.title('h_pinn'); plt.colorbar()
#     plt.subplot(1,3,3)
#     if h_ref is not None:
#         plt.imshow(np.abs(h_pred - h_ref), origin='lower', aspect='auto'); plt.title('abs error'); plt.colorbar()
#     else:
#         plt.imshow(np.zeros_like(h_pred), origin='lower', aspect='auto'); plt.title('abs error (no h_ref)')
#     plt.tight_layout(); plt.savefig(plot_file, dpi=150); plt.close()

#     if h_ref is not None:
#         mae = np.mean(np.abs(h_pred - h_ref)); rmse = np.sqrt(np.mean((h_pred - h_ref)**2))
#         if VERBOSE:
#             print(f"  Finished: MAE={mae:.6e}, RMSE={rmse:.6e}")
#     if VERBOSE:
#         print(f"  Saved model: {model_file}, plot: {plot_file}")

#     # free GPU memory if used
#     try:
#         if device.type == "cuda":
#             torch.cuda.empty_cache()
#     except Exception:
#         pass

#     return {'model': model_file, 'plot': plot_file, 'h_pinn_path': h5_path}

# # -------------------------
# # Main: iterate folder
# # -------------------------
# if __name__ == "__main__":
#     # auto device selection
#     chosen_device = DEVICE
#     if DEVICE == "cuda" and not torch.cuda.is_available():
#         print("CUDA requested but not available; falling back to CPU.")
#         chosen_device = "cpu"
#     device_obj = torch.device(chosen_device)
#     # enable cuDNN autotuner when using CUDA for improved performance
#     if device_obj.type == "cuda":
#         torch.backends.cudnn.benchmark = True
#         print("Using CUDA device for training. cuDNN benchmark = True")
#     else:
#         print("Using CPU for training.")

#     os.makedirs(OUT_ROOT, exist_ok=True)
#     pattern = os.path.join(DATA_DIR, "*.h5")
#     files = sorted(glob.glob(pattern))
#     if len(files) == 0:
#         print("No .h5 files found in DATA_DIR:", DATA_DIR)
#         raise SystemExit(1)
#     print(f"Found {len(files)} samples in {DATA_DIR}. Training sequentially on device={chosen_device} ...")

#     failures = []
#     for fpath in files:
#         sample_name = os.path.splitext(os.path.basename(fpath))[0]
#         out_dir = os.path.join(OUT_ROOT, sample_name)
#         os.makedirs(out_dir, exist_ok=True)
#         try:
#             t0 = time.time()
#             res = train_single_sample(fpath, out_dir,
#                                       device=chosen_device,
#                                       hidden=HIDDEN, n_layers=NLAYERS,
#                                       steps_adam=STEPS_ADAM, lr_adam=LR_ADAM,
#                                       lbfgs_steps=LBFGS_STEPS,
#                                       lambda_pde=LAMBDA_PDE, lambda_bc=LAMBDA_BC, lambda_sup=LAMBDA_SUP)
#             t1 = time.time()
#             print(f"Sample {sample_name} done in {t1-t0:.1f}s")
#         except Exception:
#             print(f"Sample {sample_name} failed, continuing. Traceback:")
#             traceback.print_exc()
#             failures.append(fpath)

#     print("\nAll finished. Failures:", len(failures))
#     if failures:
#         for p in failures:
#             print(" -", p)

#!/usr/bin/env python3
"""
train_all_pinn_gpu_opt.py

Same training pipeline as your provided script, with GPU/CPU auto-selection,
cuDNN autotuner and mixed-precision (AMP) during Adam to improve GPU utilization.

Fixed:
 - Use new torch.amp API (autocast, GradScaler) to avoid deprecation warnings
 - Ensure CUDA device is set early to initialize CUDA context (avoid cuBLAS "no current CUDA context" message)

Other training logic (losses, L-BFGS, saving) unchanged.
"""
import os
import time
import json
import traceback
import glob

import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

import torch
import torch.nn as nn
import torch.optim as optim

# -------------------------
# USER SETTINGS (edit if needed)
# -------------------------
DATA_DIR = "./data/pinn_train_data"        # directory containing .h5 samples
OUT_ROOT = "./results/pinn_results"        # where per-sample outputs are written
DEVICE = "cuda"                            # "cpu" or "cuda" (script will fallback if cuda not available)
STEPS_ADAM = 5000
LR_ADAM = 1e-3
LBFGS_STEPS = 60
HIDDEN = 64
NLAYERS = 6
LAMBDA_PDE = 1.0
LAMBDA_BC = 1.0
LAMBDA_SUP = 1.0   # set 0.0 for physics-only PINN labeling
VERBOSE = True
# -------------------------

# -------------------------
# Utilities: load / normalize
# -------------------------
def load_sample(h5_path):
    with h5py.File(h5_path, "r") as f:
        fields = f['fields'][:]               # (4, nz, nx)
        x = f['x'][:].astype(float)           # (nx,)
        z = f['z'][:].astype(float)           # (nz,)
        coll_x = f['collocation/x'][:]
        coll_z = f['collocation/z'][:]
        collocation = np.column_stack([coll_x, coll_z])  # (N,2) in (x,z)
        collocation_k = f['collocation/k_at_collocation'][:]
        bc_x = f['bc/x'][:]
        bc_z = f['bc/z'][:]
        bc_vals = f['bc/h'][:]
        bc_pts = np.column_stack([bc_x, bc_z])
        h_ref = f.get('h_ref')
        if h_ref is not None:
            h_ref = h_ref[:]
        meta = {}
        try:
            meta = json.loads(f.attrs.get('meta', "{}"))
        except Exception:
            try:
                meta = eval(f.attrs.get('meta', "{}"))
            except Exception:
                meta = {}
    # normalization constants
    x_mean = 0.5*(x.min() + x.max())
    x_scale = 0.5*(x.max() - x.min()) if (x.max() - x.min()) != 0 else 1.0
    z_mean = 0.5*(z.min() + z.max())
    z_scale = 0.5*(z.max() - z.min()) if (z.max() - z.min()) != 0 else 1.0
    k_grid = fields[3]   # log10(k)
    k_mean = float(np.mean(k_grid))
    k_std  = float(np.std(k_grid)) + 1e-12
    interp_k = RegularGridInterpolator((z, x), k_grid, bounds_error=False, fill_value=None)
    return {
        'fields': fields, 'x': x, 'z': z,
        'collocation': collocation, 'collocation_k': collocation_k,
        'bc_pts': bc_pts, 'bc_vals': bc_vals,
        'h_ref': h_ref, 'meta': meta,
        'norm': {'x_mean': x_mean, 'x_scale': x_scale, 'z_mean': z_mean, 'z_scale': z_scale, 'k_mean': k_mean, 'k_std': k_std},
        'interp_k': interp_k
    }

def normalize_inputs(x, z, k, norm):
    x_n = (x - norm['x_mean']) / (norm['x_scale'] + 1e-12)
    z_n = (z - norm['z_mean']) / (norm['z_scale'] + 1e-12)
    k_n = (k - norm['k_mean']) / (norm['k_std'] + 1e-12)
    return np.stack([x_n, z_n, k_n], axis=1)

def to_torch(arr, device, requires_grad=False, dtype=torch.float32):
    t = torch.tensor(arr, device=device, dtype=dtype)
    t.requires_grad = requires_grad
    return t

# -------------------------
# Model & PDE residual
# -------------------------
class PINN_MLP(nn.Module):
    def __init__(self, in_dim=3, hidden=64, n_layers=6, activation='tanh'):
        super().__init__()
        act = nn.Tanh if activation == 'tanh' else nn.ReLU
        layers = [nn.Linear(in_dim, hidden), act()]
        for _ in range(n_layers-1):
            layers += [nn.Linear(hidden, hidden), act()]
        layers.append(nn.Linear(hidden, 1))
        self.net = nn.Sequential(*layers)
        for m in self.net:
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.zeros_(m.bias)
    def forward(self, x):
        return self.net(x).squeeze(-1)

def pde_residual(model, x_t, z_t, k_t):
    inp = torch.stack([x_t, z_t, k_t], dim=1)
    inp.requires_grad_(True)
    h_pred = model(inp)
    grads = torch.autograd.grad(h_pred, inp, grad_outputs=torch.ones_like(h_pred), create_graph=True)[0]
    dh_dx = grads[:,0]; dh_dz = grads[:,1]
    k = k_t
    kdh_dx = k * dh_dx
    kdh_dz = k * dh_dz
    grad1 = torch.autograd.grad(kdh_dx, inp, grad_outputs=torch.ones_like(kdh_dx), create_graph=True)[0]
    grad2 = torch.autograd.grad(kdh_dz, inp, grad_outputs=torch.ones_like(kdh_dz), create_graph=True)[0]
    return grad1[:,0] + grad2[:,1]

# -------------------------
# Single-sample trainer
# -------------------------
def train_single_sample(h5_path, out_dir,
                        device='cpu',
                        hidden=HIDDEN, n_layers=NLAYERS,
                        steps_adam=STEPS_ADAM, lr_adam=LR_ADAM,
                        lbfgs_steps=LBFGS_STEPS,
                        lambda_pde=LAMBDA_PDE, lambda_bc=LAMBDA_BC, lambda_sup=LAMBDA_SUP):
    os.makedirs(out_dir, exist_ok=True)
    if VERBOSE:
        print(f"\n--- Training sample: {h5_path}")
    data = load_sample(h5_path)
    fields = data['fields']; x_grid = data['x']; z_grid = data['z']
    collocation = data['collocation']; collocation_k = data['collocation_k']
    bc_pts = data['bc_pts']; bc_vals = data['bc_vals']
    h_ref = data['h_ref']; norm = data['norm']; interp_k = data['interp_k']

    device = torch.device(device)

    # collocation inputs
    Xc = collocation[:,0]; Zc = collocation[:,1]; Kc = collocation_k
    inp_coll = normalize_inputs(Xc, Zc, Kc, norm)
    x_coll_t = to_torch(inp_coll[:,0], device, requires_grad=True)
    z_coll_t = to_torch(inp_coll[:,1], device, requires_grad=True)
    k_coll_t = to_torch(inp_coll[:,2], device, requires_grad=True)

    # BC inputs
    Xb = bc_pts[:,0]; Zb = bc_pts[:,1]
    kb = interp_k(np.column_stack([Zb, Xb]))
    inp_bc = normalize_inputs(Xb, Zb, kb, norm)
    x_bc_t = to_torch(inp_bc[:,0], device, requires_grad=True)
    z_bc_t = to_torch(inp_bc[:,1], device, requires_grad=True)
    k_bc_t = to_torch(inp_bc[:,2], device, requires_grad=True)
    bc_vals_t = to_torch(bc_vals.astype(np.float32), device, requires_grad=False)

    # supervised grid
    if (h_ref is not None) and (lambda_sup > 0.0):
        Xg, Zg = np.meshgrid(x_grid, z_grid)
        Xg_flat = Xg.ravel(); Zg_flat = Zg.ravel()
        kg_flat = interp_k(np.column_stack([Zg_flat, Xg_flat]))
        inp_grid = normalize_inputs(Xg_flat, Zg_flat, kg_flat, norm)
        xg_t = to_torch(inp_grid[:,0], device, requires_grad=True)
        zg_t = to_torch(inp_grid[:,1], device, requires_grad=True)
        kg_t = to_torch(inp_grid[:,2], device, requires_grad=True)
        h_ref_flat = h_ref.ravel().astype(np.float32)
        h_ref_t = to_torch(h_ref_flat, device, requires_grad=False)
    else:
        xg_t = zg_t = kg_t = h_ref_t = None

    # model + optimizer
    model = PINN_MLP(in_dim=3, hidden=hidden, n_layers=n_layers, activation='tanh').to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr_adam)
    mse = nn.MSELoss()

    # Mixed precision (AMP) setup if using CUDA
    use_amp = (device.type == "cuda")
    scaler = None
    autocast = None
    if use_amp:
        try:
            # use new torch.amp API to avoid deprecation warnings
            from torch.amp import autocast, GradScaler
            scaler = GradScaler(device_type="cuda")
            if VERBOSE:
                print("  AMP enabled (mixed precision) on CUDA using torch.amp.")
        except Exception:
            scaler = None
            autocast = None
            if VERBOSE:
                print("  torch.amp not available; running in full precision.")
    # if not using amp, ensure autocast is None (we'll branch later)
    # Note: don't import torch.cuda.amp.autocast to avoid deprecation warnings.

    if VERBOSE:
        print("  Starting Adam...")
    for it in range(steps_adam):
        optimizer.zero_grad()
        if use_amp and scaler is not None:
            # use autocast with explicit device string to avoid deprecation warnings
            with autocast("cuda"):
                res = pde_residual(model, x_coll_t, z_coll_t, k_coll_t)
                loss_pde = torch.mean(res**2)
                inp_bc_tensor = torch.stack([x_bc_t, z_bc_t, k_bc_t], dim=1)
                hbc_pred = model(inp_bc_tensor)
                loss_bc = mse(hbc_pred, bc_vals_t)
                loss_sup = torch.tensor(0.0, device=device)
                if xg_t is not None:
                    inp_grid = torch.stack([xg_t, zg_t, kg_t], dim=1)
                    hgrid_pred = model(inp_grid)
                    loss_sup = mse(hgrid_pred, h_ref_t)
                loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_sup * loss_sup
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            res = pde_residual(model, x_coll_t, z_coll_t, k_coll_t)
            loss_pde = torch.mean(res**2)
            inp_bc_tensor = torch.stack([x_bc_t, z_bc_t, k_bc_t], dim=1)
            hbc_pred = model(inp_bc_tensor)
            loss_bc = mse(hbc_pred, bc_vals_t)
            loss_sup = torch.tensor(0.0, device=device)
            if xg_t is not None:
                inp_grid = torch.stack([xg_t, zg_t, kg_t], dim=1)
                hgrid_pred = model(inp_grid)
                loss_sup = mse(hgrid_pred, h_ref_t)
            loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_sup * loss_sup
            loss.backward()
            optimizer.step()

        if VERBOSE and ((it+1) % 500 == 0 or it == 0):
            l_pde = loss_pde.item() if 'loss_pde' in locals() else 0.0
            l_bc = loss_bc.item() if 'loss_bc' in locals() else 0.0
            l_sup = loss_sup.item() if isinstance(loss_sup, torch.Tensor) else 0.0
            total = loss.item() if 'loss' in locals() else 0.0
            print(f"    Adam iter {it+1}/{steps_adam}: total={total:.3e} pde={l_pde:.3e} bc={l_bc:.3e} sup={l_sup:.3e}")

    # L-BFGS fine-tune
    if VERBOSE:
        print("  Starting L-BFGS...")
    optimizer_lbfgs = optim.LBFGS(model.parameters(), lr=1.0, max_iter=20, history_size=50, line_search_fn='strong_wolfe')
    def closure():
        optimizer_lbfgs.zero_grad()
        res = pde_residual(model, x_coll_t, z_coll_t, k_coll_t)
        loss_pde = torch.mean(res**2)
        inp_bc_tensor = torch.stack([x_bc_t, z_bc_t, k_bc_t], dim=1)
        hbc_pred = model(inp_bc_tensor)
        loss_bc = mse(hbc_pred, bc_vals_t)
        loss_sup = torch.tensor(0.0, device=device)
        if xg_t is not None:
            inp_grid = torch.stack([xg_t, zg_t, kg_t], dim=1)
            hgrid_pred = model(inp_grid)
            loss_sup = mse(hgrid_pred, h_ref_t)
        loss = lambda_pde * loss_pde + lambda_bc * loss_bc + lambda_sup * loss_sup
        loss.backward()
        closure.last = (loss.item(), loss_pde.item(), loss_bc.item(), loss_sup.item() if isinstance(loss_sup, torch.Tensor) else 0.0)
        return loss
    for i in range(lbfgs_steps):
        optimizer_lbfgs.step(closure)
        if VERBOSE and ((i+1) % 10 == 0 or i == 0):
            print(f"    L-BFGS step {i+1}/{lbfgs_steps}: loss={closure.last[0]:.3e} pde={closure.last[1]:.3e} bc={closure.last[2]:.3e} sup={closure.last[3]:.3e}")

    # evaluate on full grid
    Xg, Zg = np.meshgrid(x_grid, z_grid)
    Xg_flat = Xg.ravel(); Zg_flat = Zg.ravel()
    kg_flat = interp_k(np.column_stack([Zg_flat, Xg_flat]))
    inp_grid = normalize_inputs(Xg_flat, Zg_flat, kg_flat, norm)
    model.eval()
    with torch.no_grad():
        inp_grid_t = torch.tensor(inp_grid, device=device, dtype=torch.float32)
        h_pred_flat = model(inp_grid_t).cpu().numpy()
    h_pred = h_pred_flat.reshape(Zg.shape)

    # save h_pinn back to hdf5
    try:
        with h5py.File(h5_path, 'a') as f:
            if 'h_pinn' in f:
                del f['h_pinn']
            f.create_dataset('h_pinn', data=h_pred.astype(np.float32), compression='gzip')
            f.attrs['pinn_info'] = json.dumps({'hidden': hidden, 'n_layers': n_layers, 'lambda_pde': lambda_pde, 'lambda_bc': lambda_bc, 'lambda_sup': lambda_sup})
    except Exception as e:
        print("  Warning: failed to write h_pinn:", e)

    # save model + diagnostic image
    model_file = os.path.join(out_dir, 'pinn_model.pt')
    torch.save(model.state_dict(), model_file)
    plot_file = os.path.join(out_dir, 'pinn_diag.png')
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    if h_ref is not None:
        plt.imshow(h_ref, origin='lower', aspect='auto'); plt.title('h_ref'); plt.colorbar()
    else:
        plt.text(0.2,0.5,'No h_ref', fontsize=12)
    plt.subplot(1,3,2)
    plt.imshow(h_pred, origin='lower', aspect='auto'); plt.title('h_pinn'); plt.colorbar()
    plt.subplot(1,3,3)
    if h_ref is not None:
        plt.imshow(np.abs(h_pred - h_ref), origin='lower', aspect='auto'); plt.title('abs error'); plt.colorbar()
    else:
        plt.imshow(np.zeros_like(h_pred), origin='lower', aspect='auto'); plt.title('abs error (no h_ref)')
    plt.tight_layout(); plt.savefig(plot_file, dpi=150); plt.close()

    if h_ref is not None:
        mae = np.mean(np.abs(h_pred - h_ref)); rmse = np.sqrt(np.mean((h_pred - h_ref)**2))
        if VERBOSE:
            print(f"  Finished: MAE={mae:.6e}, RMSE={rmse:.6e}")
    if VERBOSE:
        print(f"  Saved model: {model_file}, plot: {plot_file}")

    # free GPU memory if used
    try:
        if device.type == "cuda":
            torch.cuda.empty_cache()
    except Exception:
        pass

    return {'model': model_file, 'plot': plot_file, 'h_pinn_path': h5_path}

# -------------------------
# Main: iterate folder
# -------------------------
if __name__ == "__main__":
    # auto device selection
    chosen_device = DEVICE
    if DEVICE == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available; falling back to CPU.")
        chosen_device = "cpu"
    device_obj = torch.device(chosen_device)

    # If using CUDA, set current device explicitly to initialize CUDA context early.
    if device_obj.type == "cuda":
        try:
            # set_device accepts an int or a torch.device; use the current device index
            idx = device_obj.index if device_obj.index is not None else 0
            torch.cuda.set_device(idx)
        except Exception:
            # best-effort; ignore if it fails
            pass

    # enable cuDNN autotuner when using CUDA for improved performance
    if device_obj.type == "cuda":
        torch.backends.cudnn.benchmark = True
        print("Using CUDA device for training. cuDNN benchmark = True")
    else:
        print("Using CPU for training.")

    os.makedirs(OUT_ROOT, exist_ok=True)
    pattern = os.path.join(DATA_DIR, "*.h5")
    files = sorted(glob.glob(pattern))
    if len(files) == 0:
        print("No .h5 files found in DATA_DIR:", DATA_DIR)
        raise SystemExit(1)
    print(f"Found {len(files)} samples in {DATA_DIR}. Training sequentially on device={chosen_device} ...")

    failures = []
    for fpath in files:
        sample_name = os.path.splitext(os.path.basename(fpath))[0]
        out_dir = os.path.join(OUT_ROOT, sample_name)
        os.makedirs(out_dir, exist_ok=True)
        try:
            t0 = time.time()
            res = train_single_sample(fpath, out_dir,
                                      device=chosen_device,
                                      hidden=HIDDEN, n_layers=NLAYERS,
                                      steps_adam=STEPS_ADAM, lr_adam=LR_ADAM,
                                      lbfgs_steps=LBFGS_STEPS,
                                      lambda_pde=LAMBDA_PDE, lambda_bc=LAMBDA_BC, lambda_sup=LAMBDA_SUP)
            t1 = time.time()
            print(f"Sample {sample_name} done in {t1-t0:.1f}s")
        except Exception:
            print(f"Sample {sample_name} failed, continuing. Traceback:")
            traceback.print_exc()
            failures.append(fpath)

    print("\nAll finished. Failures:", len(failures))
    if failures:
        for p in failures:
            print(" -", p)

