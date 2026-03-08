"""
Quick diagnosis - check what the dataset actually looks like
and what the model is outputting internally.
"""
import os, sys
import numpy as np
import pandas as pd
import torch

backend_dir = os.path.dirname(os.path.abspath(__file__))

# ── 1. Check dataset balance ──────────────────────────────────────────────────
csv_path = os.path.join(backend_dir, "dataset_phishing.csv")
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    col_vals = df["status"].astype(str).str.strip().str.lower()
    phish = (col_vals == "phishing").sum()
    legit = (col_vals == "legitimate").sum()
    total = len(df)
    print(f"Dataset balance:")
    print(f"  Total     : {total}")
    print(f"  Phishing  : {phish} ({phish/total*100:.1f}%)")
    print(f"  Legitimate: {legit} ({legit/total*100:.1f}%)")
    print(f"  Ratio     : {phish/max(legit,1):.2f}x more phishing than legit")
else:
    print("CSV not found")

# ── 2. Check what model outputs for known safe/phishing features ──────────────
ck_path = os.path.join(backend_dir, "model.pth")
if os.path.exists(ck_path):
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    print(f"\nCheckpoint keys  : {list(ck.keys())}")
    print(f"input_dim        : {ck.get('input_dim')}")
    print(f"threshold        : {ck.get('threshold')}")
    print(f"version          : {ck.get('version')}")

    if "scaler_mean" in ck:
        mean  = np.array(ck["scaler_mean"])
        scale = np.array(ck["scaler_scale"])
        print(f"\nScaler mean  (first 10): {mean[:10].round(3).tolist()}")
        print(f"Scaler scale (first 10): {scale[:10].round(3).tolist()}")

        # Simulate what a safe URL looks like after scaling
        # google.com: length_url~22, length_hostname~10, ip=0, nb_dots=1, ...
        safe_raw = np.zeros((1, len(mean)), dtype=np.float32)
        safe_raw[0, 0] = 22   # length_url
        safe_raw[0, 1] = 10   # length_hostname
        safe_raw[0, 3] = 1    # nb_dots
        safe_raw[0, 13] = 3   # nb_slash
        safe_raw_scaled = np.clip((safe_raw - mean) / (scale + 1e-8), -5, 5)
        print(f"\nScaled safe features (first 10): {safe_raw_scaled[0,:10].round(3).tolist()}")

        # Check if scaler is producing extreme values
        max_val = np.abs(safe_raw_scaled).max()
        print(f"Max absolute scaled value: {max_val:.3f}")
        if max_val > 4:
            print("WARNING: Extreme scaled values — scaler may be broken!")
