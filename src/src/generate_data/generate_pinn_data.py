#!/usr/bin/env python3
"""
generate_pinn_data.py

Generates synthetic random-field samples and optional hydraulic head (h_ref)
via a simple finite-difference solver. Saves per-sample HDF5 files ready
for PINN training.

Usage:
    python generate_pinn_data.py --out_dir data --n_samples 100
"""
import os
import argparse
import json
from tqdm import tqdm
import numpy as np
import h5py
from scipy.ndimage import gaussian_filter
from scipy.interpolate import RegularGridInterpolator
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import spsolve

# optional gstools
try:
    import gstools as gs
    _HAS_GSTOOLS = True
except Exception:
    _HAS_GSTOOLS = False

#---------------------
# Pinn Data Utils
#-----------------------

def _sample_with_gstools(model, seed, nz, nx):
    """Return a single gaussian field sampled with gstools SRF. Output shape (nz, nx)."""
    srf = gs.SRF(model, seed=int(seed))
    # gstools' .structured expects the shape as [nx, ny] or [nx, ny] depending on version.
    # Using (nz, nx) here but call with [nz, nx] consistent with earlier code.
    return srf.structured([nz, nx])

def _fallback_gaussian_filtered(seed, nz, nx, sigma_physical=6.0):
    """Fallback sampler: white noise smoothed with gaussian_filter (not exact FFT-MA but OK for tests)."""
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(size=(nz, nx))
    # sigma_physical is in grid points; adjust as needed
    arr = gaussian_filter(arr, sigma=sigma_physical, mode='reflect')
    arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-12)
    return arr

def sample_multivariate_fields(nx=128, nz=128, Lx=20.0, Lz=10.0,
                               means=(25.0, 30.0, 18.0, 1e-5),
                               covs=(0.3, 0.1, 0.05, 0.5),
                               corr_lengths=(2.0, 1.0),
                               cross_corr=None,
                               seed=0,
                               model_type='exponential'):
    """
    Returns fields array shape (4, nz, nx):
      channels: [c (kPa), phi_deg, gamma (kN/m3), log10(k (m/s))]
    Uses gstools.SRF if available; otherwise uses gaussian-filtered noise as fallback.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, Lx, nx)
    z = np.linspace(0, Lz, nz)

    # choose covariance model for gstools if available
    if _HAS_GSTOOLS:
        if model_type.lower() == 'exponential':
            model = gs.Exponential(dim=2, var=1.0, len_scale=[corr_lengths[0], corr_lengths[1]])
        else:
            model = gs.SquaredExp(dim=2, var=1.0, len_scale=[corr_lengths[0], corr_lengths[1]])

    # sample independent Gaussian fields (zero-mean, unit var)
    gaussian_fields = np.zeros((4, nz, nx))
    for i in range(4):
        if _HAS_GSTOOLS:
            try:
                field = _sample_with_gstools(model, seed + i, nz, nx)
                # ensure zero mean, unit var
                field = (field - np.mean(field)) / (np.std(field) + 1e-12)
            except Exception:
                # fallback
                field = _fallback_gaussian_filtered(seed + i, nz, nx, sigma_physical=max(1, int(corr_lengths[0])))
        else:
            field = _fallback_gaussian_filtered(seed + i, nz, nx, sigma_physical=max(1, int(corr_lengths[0])))
        gaussian_fields[i] = field

    # apply cross-correlation across channels if provided
    if cross_corr is not None:
        L = np.linalg.cholesky(cross_corr)
        flat = gaussian_fields.reshape(4, -1)
        mixed = L @ flat
        gaussian_fields = mixed.reshape(4, nz, nx)

    # map Gaussian -> target marginal (lognormal for positively-valued channels)
    fields = np.zeros_like(gaussian_fields)
    for i in range(4):
        mean_i = means[i]
        cov_i = covs[i]
        if i == 1:
            # friction angle: map gaussian -> normal around mean and clip to realistic bounds
            sigma = cov_i * mean_i
            fields[i] = np.clip(mean_i + sigma * gaussian_fields[i], 12.0, 50.0)
        elif i == 2:
            # unit weight: small variations, treat as normal
            sigma = cov_i * mean_i
            fields[i] = np.clip(mean_i + sigma * gaussian_fields[i], 12.0, 25.0)
        elif i == 3:
            # permeability: lognormal. mean_i is physical mean in m/s
            cov = cov_i
            sigma_ln = np.sqrt(np.log(1 + cov**2))
            mu_ln = np.log(mean_i) - 0.5 * sigma_ln**2
            ln_field = mu_ln + sigma_ln * gaussian_fields[i]
            k_field = np.exp(ln_field)  # in m/s
            fields[i] = np.log10(k_field)  # store log10(k)
        else:
            # cohesion c (kPa): lognormal positive
            cov = cov_i
            sigma_ln = np.sqrt(np.log(1 + cov**2))
            mu_ln = np.log(mean_i) - 0.5 * sigma_ln**2
            fields[i] = np.exp(mu_ln + sigma_ln * gaussian_fields[i])

    return fields, x, z


def save_sample_h5(path, fields, x, z, collocation, collocation_k,
                   bc_pts, bc_values, meta, h_ref=None):
    """
    Save sample to HDF5. fields shape (4, nz, nx). collocation arrays shaped (N,).
    """
    with h5py.File(path, 'w') as f:
        f.create_dataset('fields', data=fields.astype('float32'), compression='gzip')
        f.create_dataset('x', data=x.astype('float32'))
        f.create_dataset('z', data=z.astype('float32'))

        grp = f.create_group('collocation')
        grp.create_dataset('x', data=collocation[:,0].astype('float32'))
        grp.create_dataset('z', data=collocation[:,1].astype('float32'))
        grp.create_dataset('k_at_collocation', data=collocation_k.astype('float32'))

        grp2 = f.create_group('bc')
        grp2.create_dataset('x', data=bc_pts[:,0].astype('float32'))
        grp2.create_dataset('z', data=bc_pts[:,1].astype('float32'))
        grp2.create_dataset('h', data=bc_values.astype('float32'))

        if h_ref is not None:
            f.create_dataset('h_ref', data=h_ref.astype('float32'), compression='gzip')

        f.attrs['meta'] = json.dumps(meta)

#------------------------
# fd Seepage Solver
# -----------------------

def solve_variable_k_dirichlet(k_grid_log10, x_grid, z_grid, bc_mask, bc_values):
    """
    Finite-difference solver for steady Darcy: div( k grad h ) = 0
    - k_grid_log10: array (nz, nx) of log10(k) values (k in m/s stored as log10)
    - x_grid: 1d array (nx), z_grid: 1d array (nz)
    - bc_mask: boolean array (nz, nx) True where Dirichlet BCs apply
    - bc_values: array (nz, nx) of Dirichlet values (only valid where bc_mask True)
    Returns:
      h: (nz, nx) hydraulic head approximate (same spacing)
    Notes:
      Uses cell-centered finite difference with harmonic averaging of k at interfaces.
    """
    nz, nx = k_grid_log10.shape
    dx = x_grid[1] - x_grid[0]
    dz = z_grid[1] - z_grid[0]

    k_grid = 10.0 ** k_grid_log10

    N = nx * nz
    A = lil_matrix((N, N))
    b = np.zeros(N)

    def idx(i, j):
        return i * nx + j

    for i in range(nz):
        for j in range(nx):
            p = idx(i, j)
            if bc_mask[i, j]:
                A[p, p] = 1.0
                b[p] = bc_values[i, j]
                continue

            # neighbors: left,right,up,down (with Neumann=zero flux if at boundary without BC)
            coeff_center = 0.0

            # Left interface
            if j - 1 >= 0:
                kL = harmonic_mean(k_grid[i, j], k_grid[i, j-1])
                w = kL / dx**2
                A[p, idx(i, j-1)] = -w
                coeff_center += w
            else:
                # Neumann zero flux -> no contribution
                pass

            # Right
            if j + 1 < nx:
                kR = harmonic_mean(k_grid[i, j], k_grid[i, j+1])
                w = kR / dx**2
                A[p, idx(i, j+1)] = -w
                coeff_center += w

            # Up (i-1)  (z decreases upward)
            if i - 1 >= 0:
                kU = harmonic_mean(k_grid[i, j], k_grid[i-1, j])
                w = kU / dz**2
                A[p, idx(i-1, j)] = -w
                coeff_center += w

            # Down (i+1)
            if i + 1 < nz:
                kD = harmonic_mean(k_grid[i, j], k_grid[i+1, j])
                w = kD / dz**2
                A[p, idx(i+1, j)] = -w
                coeff_center += w

            A[p, p] = coeff_center
            b[p] = 0.0

    A = csr_matrix(A)
    h_vec = spsolve(A, b)
    h = h_vec.reshape(nz, nx)
    return h

def harmonic_mean(a, b, eps=1e-16):
    return 2.0 * a * b / (a + b + eps)

# -------------------------
# Random-field sampler
# -------------------------
def _sample_with_gstools(model, seed, nz, nx):
    srf = gs.SRF(model, seed=int(seed))
    return srf.structured([nz, nx])

def _fallback_gaussian_filtered(seed, nz, nx, sigma_physical=6.0):
    rng = np.random.default_rng(seed)
    arr = rng.standard_normal(size=(nz, nx))
    arr = gaussian_filter(arr, sigma=sigma_physical, mode='reflect')
    arr = (arr - np.mean(arr)) / (np.std(arr) + 1e-12)
    return arr

def sample_multivariate_fields(nx=128, nz=128, Lx=20.0, Lz=10.0,
                               means=(25.0, 30.0, 18.0, 1e-5),
                               covs=(0.3, 0.1, 0.05, 0.5),
                               corr_lengths=(2.0, 1.0),
                               cross_corr=None,
                               seed=0,
                               model_type='exponential'):
    """
    Returns fields array shape (4, nz, nx):
      channels: [c (kPa), phi_deg, gamma (kN/m3), log10(k (m/s))]
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(0, Lx, nx)
    z = np.linspace(0, Lz, nz)

    if _HAS_GSTOOLS:
        if model_type.lower() == 'exponential':
            model = gs.Exponential(dim=2, var=1.0, len_scale=[corr_lengths[0], corr_lengths[1]])
        else:
            model = gs.SquaredExp(dim=2, var=1.0, len_scale=[corr_lengths[0], corr_lengths[1]])

    gaussian_fields = np.zeros((4, nz, nx))
    for i in range(4):
        if _HAS_GSTOOLS:
            try:
                field = _sample_with_gstools(model, seed + i, nz, nx)
                field = (field - np.mean(field)) / (np.std(field) + 1e-12)
            except Exception:
                field = _fallback_gaussian_filtered(seed + i, nz, nx, sigma_physical=max(1, int(corr_lengths[0])))
        else:
            field = _fallback_gaussian_filtered(seed + i, nz, nx, sigma_physical=max(1, int(corr_lengths[0])))
        gaussian_fields[i] = field

    if cross_corr is not None:
        L = np.linalg.cholesky(cross_corr)
        flat = gaussian_fields.reshape(4, -1)
        mixed = L @ flat
        gaussian_fields = mixed.reshape(4, nz, nx)

    fields = np.zeros_like(gaussian_fields)
    for i in range(4):
        mean_i = means[i]
        cov_i = covs[i]
        if i == 1:
            sigma = cov_i * mean_i
            fields[i] = np.clip(mean_i + sigma * gaussian_fields[i], 12.0, 50.0)
        elif i == 2:
            sigma = cov_i * mean_i
            fields[i] = np.clip(mean_i + sigma * gaussian_fields[i], 12.0, 25.0)
        elif i == 3:
            cov = cov_i
            sigma_ln = np.sqrt(np.log(1 + cov**2))
            mu_ln = np.log(mean_i) - 0.5 * sigma_ln**2
            ln_field = mu_ln + sigma_ln * gaussian_fields[i]
            k_field = np.exp(ln_field)  # m/s
            fields[i] = np.log10(k_field)
        else:
            cov = cov_i
            sigma_ln = np.sqrt(np.log(1 + cov**2))
            mu_ln = np.log(mean_i) - 0.5 * sigma_ln**2
            fields[i] = np.exp(mu_ln + sigma_ln * gaussian_fields[i])

    return fields, x, z

# -------------------------
# FD solver for div(k grad h) = 0 (Dirichlet)
# -------------------------
def harmonic_mean(a, b, eps=1e-16):
    return 2.0 * a * b / (a + b + eps)

def solve_variable_k_dirichlet(k_grid_log10, x_grid, z_grid, bc_mask, bc_values):
    """
    Finite-difference solver for steady Darcy: div( k grad h ) = 0
    - k_grid_log10: array (nz, nx) of log10(k) values
    - x_grid: 1d array (nx), z_grid: 1d array (nz)
    - bc_mask: boolean array (nz, nx) True where Dirichlet BCs apply
    - bc_values: array (nz, nx) of Dirichlet values (only valid where bc_mask True)
    Returns h (nz, nx)
    """
    nz, nx = k_grid_log10.shape
    dx = x_grid[1] - x_grid[0]
    dz = z_grid[1] - z_grid[0]

    k_grid = 10.0 ** k_grid_log10

    N = nx * nz
    A = lil_matrix((N, N))
    b = np.zeros(N)

    def idx(i, j):
        return i * nx + j

    for i in range(nz):
        for j in range(nx):
            p = idx(i, j)
            if bc_mask[i, j]:
                A[p, p] = 1.0
                b[p] = bc_values[i, j]
                continue

            coeff_center = 0.0

            # Left
            if j - 1 >= 0:
                kL = harmonic_mean(k_grid[i, j], k_grid[i, j-1])
                w = kL / dx**2
                A[p, idx(i, j-1)] = -w
                coeff_center += w

            # Right
            if j + 1 < nx:
                kR = harmonic_mean(k_grid[i, j], k_grid[i, j+1])
                w = kR / dx**2
                A[p, idx(i, j+1)] = -w
                coeff_center += w

            # Up (i-1)
            if i - 1 >= 0:
                kU = harmonic_mean(k_grid[i, j], k_grid[i-1, j])
                w = kU / dz**2
                A[p, idx(i-1, j)] = -w
                coeff_center += w

            # Down (i+1)
            if i + 1 < nz:
                kD = harmonic_mean(k_grid[i, j], k_grid[i+1, j])
                w = kD / dz**2
                A[p, idx(i+1, j)] = -w
                coeff_center += w

            A[p, p] = coeff_center
            b[p] = 0.0

    A = csr_matrix(A)
    h_vec = spsolve(A, b)
    h = h_vec.reshape(nz, nx)
    return h

# -------------------------
# Collocation & BC helpers
# -------------------------
def sample_collocation_points(N_pde, Lx, Lz, seed):
    rng = np.random.default_rng(seed)
    margin = 1e-10
    xs = rng.uniform(margin, Lx - margin, size=(N_pde,))
    zs = rng.uniform(margin, Lz - margin, size=(N_pde,))
    pts = np.column_stack([xs, zs])  # (x, z)
    return pts

def create_bc_on_top_and_bottom(x, z, top_h=1.0, bottom_h=0.0, N_bc_each=500):
    nx = x.size
    nz = z.size
    X, Z = np.meshgrid(x, z)
    bc_mask = np.zeros((nz, nx), dtype=bool)
    bc_values = np.zeros((nz, nx), dtype=float)

    # top row (z index 0)
    bc_mask[0, :] = True
    bc_values[0, :] = top_h
    # bottom row (z index -1)
    bc_mask[-1, :] = True
    bc_values[-1, :] = bottom_h

    # create BC point list (x,z) and values for saving (subset)
    xs_bc = np.linspace(x.min(), x.max(), N_bc_each)
    zs_top = np.full_like(xs_bc, z.min())
    zs_bot = np.full_like(xs_bc, z.max())
    bc_pts = np.vstack([np.column_stack([xs_bc, zs_top]), np.column_stack([xs_bc, zs_bot])])
    bc_vals = np.hstack([np.full(xs_bc.shape, top_h), np.full(xs_bc.shape, bottom_h)])
    return bc_mask, bc_values, bc_pts, bc_vals

# -------------------------
# HDF5 writer
# -------------------------
def save_sample_h5(path, fields, x, z, collocation, collocation_k,
                   bc_pts, bc_values, meta, h_ref=None):
    with h5py.File(path, 'w') as f:
        f.create_dataset('fields', data=fields.astype('float32'), compression='gzip')
        f.create_dataset('x', data=x.astype('float32'))
        f.create_dataset('z', data=z.astype('float32'))

        grp = f.create_group('collocation')
        grp.create_dataset('x', data=collocation[:,0].astype('float32'))
        grp.create_dataset('z', data=collocation[:,1].astype('float32'))
        grp.create_dataset('k_at_collocation', data=collocation_k.astype('float32'))

        grp2 = f.create_group('bc')
        grp2.create_dataset('x', data=bc_pts[:,0].astype('float32'))
        grp2.create_dataset('z', data=bc_pts[:,1].astype('float32'))
        grp2.create_dataset('h', data=bc_values.astype('float32'))

        if h_ref is not None:
            f.create_dataset('h_ref', data=h_ref.astype('float32'), compression='gzip')

        f.attrs['meta'] = json.dumps(meta)

# -------------------------
# Main orchestration
# -------------------------
def main(out_dir='data', n_samples=100, nx=128, nz=128, Lx=20.0, Lz=10.0,
         N_pde=10000, N_bc=1000, seed_base=0):
    os.makedirs(out_dir, exist_ok=True)

    # default cross-correlation (example SPD)
    R = np.eye(4)
    R[0,1] = 0.3; R[1,0] = 0.3
    R[0,3] = 0.2; R[3,0] = 0.2

    for ii in tqdm(range(n_samples), desc="samples"):
        seed = seed_base + ii
        fields, x, z = sample_multivariate_fields(nx=nx, nz=nz, Lx=Lx, Lz=Lz,
                                                  means=(25.0, 30.0, 18.0, 1e-5),
                                                  covs=(0.3, 0.1, 0.02, 0.5),
                                                  corr_lengths=(2.0, 1.0),
                                                  cross_corr=R,
                                                  seed=seed)
        # fields: (4, nz, nx), channel3 is log10(k)
        meta = {
            'seed': int(seed),
            'nx': int(nx),'nz':int(nz),
            'Lx': float(Lx),'Lz':float(Lz),
            'means': [25.0,30.0,18.0,1e-5],
            'covs': [0.3,0.1,0.02,0.5],
            'corr_lengths': [2.0,1.0]
        }

        # collocation points (x,z)
        collocation = sample_collocation_points(N_pde, Lx, Lz, seed + 12345)
        # create BC masks and point lists
        bc_mask, bc_grid_values, bc_pts, bc_vals = create_bc_on_top_and_bottom(x, z, top_h=1.0, bottom_h=0.0, N_bc_each=N_bc//2)

        # Interpolate k at collocation points.
        # Important: RegularGridInterpolator grid order = (z, x) for array shaped (nz, nx).
        # Query points must be provided in the same ordering: (z, x).
        eps = 1e-12
        coll_x = np.clip(collocation[:, 0], x.min() + eps, x.max() - eps)
        coll_z = np.clip(collocation[:, 1], z.min() + eps, z.max() - eps)
        collocation_for_interp = np.column_stack([coll_z, coll_x])   # (z, x) order

        interp_k = RegularGridInterpolator((z, x), fields[3], bounds_error=False, fill_value=None)
        collocation_k = interp_k(collocation_for_interp)

        # Solve for h_ref using FD solver on the whole grid
        h_ref = solve_variable_k_dirichlet(k_grid_log10=fields[3], x_grid=x, z_grid=z,
                                           bc_mask=bc_mask, bc_values=bc_grid_values)

        # Save sample file
        out_path = os.path.join(out_dir, f"sample_{ii+1:08d}.h5")
        save_sample_h5(out_path, fields, x, z, collocation, collocation_k, bc_pts, bc_vals, meta, h_ref=h_ref)

    print("Saved", n_samples, "samples to", out_dir)

# -------------------------
# CLI
# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='data\pinn_train_data')
    parser.add_argument('--n_samples', type=int, default=10)
    parser.add_argument('--nx', type=int, default=128)
    parser.add_argument('--nz', type=int, default=128)
    parser.add_argument('--Lx', type=float, default=20.0)
    parser.add_argument('--Lz', type=float, default=10.0)
    parser.add_argument('--N_pde', type=int, default=10000)
    parser.add_argument('--N_bc', type=int, default=1000)
    parser.add_argument('--seed_base', type=int, default=0)
    args = parser.parse_args()
    main(out_dir=args.out_dir, n_samples=args.n_samples, nx=args.nx, nz=args.nz,
         Lx=args.Lx, Lz=args.Lz, N_pde=args.N_pde, N_bc=args.N_bc, seed_base=args.seed_base)
