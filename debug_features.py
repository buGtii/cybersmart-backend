"""
Debug: Print what features your extractor actually outputs
for known safe vs phishing URLs, then compare to dataset values.
"""
import os, sys, numpy as np
import torch

backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

from feature_extractor import FeatureExtractor

fe = FeatureExtractor(timeout=3)

urls = [
    ("https://www.google.com",                              "SAFE"),
    ("https://www.github.com",                              "SAFE"),
    ("http://paypal-secure-login.xyz/verify?user=admin",    "PHISHING"),
]

print("=" * 80)
print("RAW FEATURE VALUES (before scaling)")
print("=" * 80)

for url, label in urls:
    feats = fe.extract(url)
    names = fe.FEATURE_NAMES
    print(f"\n[{label}] {url[:60]}")
    print(f"  Total features: {len(feats)}")
    for i, (n, v) in enumerate(zip(names, feats)):
        print(f"  {i+1:2d}. {n:<35} = {v:.4f}")

# Now load scaler from checkpoint and show what scaled values look like
ck_path = os.path.join(backend_dir, "model.pth")
if os.path.exists(ck_path):
    ck = torch.load(ck_path, map_location="cpu", weights_only=True)
    mean  = np.array(ck["scaler_mean"],  dtype=np.float32)
    scale = np.array(ck["scaler_scale"], dtype=np.float32)
    threshold = ck.get("threshold", 0.5)
    input_dim = ck.get("input_dim", 87)

    print(f"\n{'='*80}")
    print(f"SCALED VALUES (what model actually sees) | threshold={threshold}")
    print(f"{'='*80}")

    for url, label in urls:
        feats = fe.extract(url)
        X = np.array([feats], dtype=np.float32)

        # Align to model input_dim
        if X.shape[1] < input_dim:
            X = np.hstack([X, np.zeros((1, input_dim - X.shape[1]))])
        elif X.shape[1] > input_dim:
            X = X[:, :input_dim]

        X_scaled = np.clip((X - mean) / (scale + 1e-8), -5, 5)
        print(f"\n[{label}] {url[:55]}")
        print(f"  Raw    first10: {[round(float(v),3) for v in X[0,:10]]}")
        print(f"  Scaled first10: {[round(float(v),3) for v in X_scaled[0,:10]]}")
        print(f"  Raw    mean={X[0].mean():.3f}  max={X[0].max():.3f}")
        print(f"  Scaled mean={X_scaled[0].mean():.3f}  max={X_scaled[0].max():.3f}")

        # Check dataset mean for comparison
        print(f"  Dataset mean (first10): {mean[:10].round(3).tolist()}")
        print(f"  Difference from dataset mean: {(X[0,:10] - mean[:10]).round(3).tolist()}")
