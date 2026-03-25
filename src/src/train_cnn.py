#!/usr/bin/env python3
"""
train_cnn.py

Trains a U-Net CNN to map log_k -> h (Supervised labels = h_pinn).
Reads `all_list.txt` created by data_prep_cnn.py, performs train/val/test split
INTERNALLY (no splitting in data prep).

Outputs:
 - train/val/test_list.txt in out_dir
 - best_model.pt
 - checkpoints per epoch
 - diagnostic images

Fully GPU-optimized (mixed precision) if CUDA available.
"""

import os
import argparse
import time
import random
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
import contextlib

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# -------------------------
# UNet Model
# -------------------------
class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),

            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(True),
        )
    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, in_ch=1, out_ch=1, base=64):
        super().__init__()
        # Encoder
        self.enc1 = ConvBlock(in_ch, base)
        self.enc2 = ConvBlock(base, base*2)
        self.enc3 = ConvBlock(base*2, base*4)
        self.enc4 = ConvBlock(base*4, base*8)

        # Pools
        self.pool = nn.MaxPool2d(2)

        # Up-sampling
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        # Decoder
        self.dec4 = ConvBlock(base*8 + base*4, base*4)
        self.dec3 = ConvBlock(base*4 + base*2, base*2)
        self.dec2 = ConvBlock(base*2 + base, base)

        self.final = nn.Conv2d(base, out_ch, 1)

    def forward(self, x):
        e1 = self.enc1(x)
        e2 = self.enc2(self.pool(e1))
        e3 = self.enc3(self.pool(e2))
        e4 = self.enc4(self.pool(e3))

        d4 = self.up(e4)
        d4 = self.dec4(torch.cat([d4, e3], dim=1))

        d3 = self.up(d4)
        d3 = self.dec3(torch.cat([d3, e2], dim=1))

        d2 = self.up(d3)
        d2 = self.dec2(torch.cat([d2, e1], dim=1))

        return self.final(d2)

# -------------------------
# Dataset
# -------------------------
class NPZDataset(Dataset):
    def __init__(self, list_file):
        with open(list_file, "r") as f:
            self.paths = [l.strip() for l in f.readlines() if l.strip()]
        if len(self.paths) == 0:
            raise RuntimeError("No samples in " + list_file)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        d = np.load(path)
        logk = d["logk"].astype(np.float32)       # shape (1,H,W)
        hpinn = d["hpinn"].astype(np.float32)     # shape (1,H,W)
        return torch.from_numpy(logk), torch.from_numpy(hpinn)

# -------------------------
# Visualization
# -------------------------
def plot_sample(inp, targ, pred, save_path):
    inp, targ, pred = inp[0], targ[0], pred[0]
    plt.figure(figsize=(12,4))
    plt.subplot(1,3,1)
    plt.imshow(inp, origin='lower'); plt.title("log(k)"); plt.colorbar()

    plt.subplot(1,3,2)
    plt.imshow(targ, origin='lower'); plt.title("h_pinn"); plt.colorbar()

    plt.subplot(1,3,3)
    plt.imshow(pred, origin='lower'); plt.title("h_pred"); plt.colorbar()

    plt.tight_layout()
    plt.savefig(save_path, dpi=120)
    plt.close()

# -------------------------
# TRAINING
# -------------------------
def train(args):

    # ---------- DEVICE ----------
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu")
    print(f"Using device: {device}")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        # initialize CUDA context early to prevent warnings
        _ = torch.tensor(0.0, device=device)
        torch.cuda.synchronize()

    # ---------- LOAD MASTER INDEX ----------
    master = os.path.join(args.data_dir, "all_list.txt")
    if not os.path.exists(master):
        raise RuntimeError("Missing all_list.txt → Run data_prep_cnn.py first.")

    with open(master, "r") as f:
        all_paths = [l.strip() for l in f.readlines() if l.strip()]

    if len(all_paths) == 0:
        raise RuntimeError("all_list.txt is empty.")

    # ---------- INTERNAL SPLIT ----------
    random.seed(args.seed)
    random.shuffle(all_paths)

    N = len(all_paths)
    N_test = int(N * args.test_frac)
    N_val  = int(N * args.val_frac)
    N_train = N - N_test - N_val

    train_paths = all_paths[:N_train]
    val_paths   = all_paths[N_train:N_train+N_val]
    test_paths  = all_paths[N_train+N_val:]

    print(f"TOTAL: {N} | train={len(train_paths)} | val={len(val_paths)} | test={len(test_paths)}")

    os.makedirs(args.out_dir, exist_ok=True)

    # save split lists
    def dump(lst, name):
        with open(os.path.join(args.out_dir, name), "w") as f:
            for p in lst: f.write(p + "\n")

    dump(train_paths, "train_list.txt")
    dump(val_paths, "val_list.txt")
    dump(test_paths, "test_list.txt")

    # ---------- Dataset + DataLoader ----------
    train_ds = NPZDataset(os.path.join(args.out_dir, "train_list.txt"))
    val_ds   = NPZDataset(os.path.join(args.out_dir, "val_list.txt"))
    test_ds  = NPZDataset(os.path.join(args.out_dir, "test_list.txt"))

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              num_workers=args.num_workers, pin_memory=True)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=max(1, args.num_workers//2), pin_memory=True)

    # ---------- Model ----------
    model = UNet(in_ch=1, out_ch=1, base=args.base_channels).to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.MSELoss()

    # ---------- AMP (Mixed Precision) ----------
    use_amp = (device.type == "cuda")
    if use_amp:
        try:
            from torch.amp import autocast, GradScaler
            scaler = GradScaler(device_type="cuda")
            autocast_ctx = lambda: autocast("cuda")
        except:
            from torch.cuda.amp import autocast, GradScaler
            scaler = GradScaler()
            autocast_ctx = lambda: autocast()
    else:
        scaler = None
        autocast_ctx = lambda: contextlib.nullcontext()

    best_val = 1e9
    best_ckpt = os.path.join(args.out_dir, "best_model.pt")

    # -------------------------
    # TRAINING LOOP
    # -------------------------
    for epoch in range(1, args.epochs+1):
        t0 = time.time()
        model.train()
        train_loss = 0.0

        for k_batch, h_batch in train_loader:
            k_batch, h_batch = k_batch.to(device), h_batch.to(device)
            optimizer.zero_grad()

            with autocast_ctx():
                pred = model(k_batch)
                loss = criterion(pred, h_batch)

            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

            train_loss += loss.item() * k_batch.size(0)

        train_loss /= len(train_loader.dataset)

        # ----- Validation -----
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for k_batch, h_batch in val_loader:
                k_batch, h_batch = k_batch.to(device), h_batch.to(device)

                with autocast_ctx():
                    pred = model(k_batch)
                    loss = criterion(pred, h_batch)

                val_loss += loss.item() * k_batch.size(0)

        val_loss /= len(val_loader.dataset)

        print(f"Epoch {epoch}/{args.epochs} | train={train_loss:.4e} | val={val_loss:.4e} | time={time.time()-t0:.1f}s")

        # Save best model
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict()}, best_ckpt)
            print("  Saved best model.")

        # Save diagnostic image
        try:
            sample_path = val_paths[0]
            d = np.load(sample_path)
            kin = d["logk"]
            hin = d["hpinn"]

            model.eval()
            with torch.no_grad():
                x = torch.from_numpy(kin).unsqueeze(0).to(device)
                pred = model(x).cpu().numpy()

            plot_sample(kin, hin, pred[0], os.path.join(args.out_dir, f"diag_epoch_{epoch:03d}.png"))
        except Exception as e:
            print("Diagnostic plot failed:", e)

    # -------------------------
    # FINAL TEST EVALUATION
    # -------------------------
    print("\nEvaluating best model on test set...")
    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    test_loss = 0.0
    with torch.no_grad():
        for k_batch, h_batch in test_loader:
            k_batch, h_batch = k_batch.to(device), h_batch.to(device)
            with autocast_ctx():
                pred = model(k_batch)
                loss = criterion(pred, h_batch)
            test_loss += loss.item() * k_batch.size(0)

    test_loss /= len(test_loader.dataset)
    print(f"TEST LOSS = {test_loss:.4e}")
    print("Best model:", best_ckpt)

# -------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", default="./data/cnn_train_data")
    parser.add_argument("--out_dir", default="./results/cnn_results")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--batch_size", type=int, default=12)
    parser.add_argument("--base_channels", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test_frac", type=float, default=0.15)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--num_workers", type=int, default=4)

    args = parser.parse_args()
    train(args)