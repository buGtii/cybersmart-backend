"""
CyberSmart - PyTorch Model v3
Reads threshold, input_dim, and scaler from model.pth checkpoint.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, TensorDataset

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model.pth")


class PhishingDetector(nn.Module):
    def __init__(self, input_dim=56, dropout=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(128, 64),        nn.BatchNorm1d(64),  nn.ReLU(), nn.Dropout(dropout),
            nn.Linear(64,  32),        nn.BatchNorm1d(32),  nn.ReLU(), nn.Dropout(dropout/2),
            nn.Linear(32,  16),        nn.ReLU(),
            nn.Linear(16,  1),         nn.Sigmoid()
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


class TorchScaler:
    def __init__(self, mean, scale):
        self.mean  = np.array(mean,  dtype=np.float32)
        self.scale = np.array(scale, dtype=np.float32)

    def transform(self, X):
        X = np.array(X, dtype=np.float32)
        return np.clip((X - self.mean) / (self.scale + 1e-8), -5, 5)


class ModelManager:
    def __init__(self, model_path=MODEL_PATH, device=None):
        self.model_path = model_path
        self.device     = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.scaler     = None
        self.input_dim  = None
        self.model      = None
        self.threshold  = 0.5   # default, overridden from checkpoint

        if os.path.exists(model_path):
            self._load()
            print(f"[Model] Loaded | input_dim={self.input_dim} | threshold={self.threshold}")
        else:
            print("[Model] No model.pth — bootstrapping with synthetic data...")
            self._bootstrap()

    def _load(self):
        ck = torch.load(self.model_path, map_location=self.device, weights_only=True)

        self.input_dim = ck.get("input_dim", 56)
        self.threshold = float(ck.get("threshold", 0.5))

        self.model = PhishingDetector(input_dim=self.input_dim).to(self.device)
        self.model.load_state_dict(ck["model_state_dict"])
        self.model.eval()

        if "scaler_mean" in ck and "scaler_scale" in ck:
            self.scaler = TorchScaler(ck["scaler_mean"], ck["scaler_scale"])
            print("[Model] Scaler loaded OK")
        else:
            print("[Model] WARNING: no scaler in checkpoint")

    def predict(self, features):
        self.model.eval()
        X = np.array([features], dtype=np.float32)

        if self.scaler is not None:
            if X.shape[1] != self.input_dim:
                X = self._align(X)
            X = self.scaler.transform(X)
        else:
            if X.shape[1] != self.input_dim:
                X = self._align(X)
            X = np.clip(X, 0, 1)

        with torch.no_grad():
            prob = self.model(torch.tensor(X).to(self.device)).item()

        label      = "phishing" if prob > self.threshold else "safe"
        confidence = prob if prob > self.threshold else (1 - prob)

        return {
            "probability": round(prob, 4),
            "label":       label,
            "confidence":  round(confidence * 100, 2),
            "risk_level":  self._risk(prob)
        }

    def _align(self, X):
        c = X.shape[1]
        if c < self.input_dim:
            X = np.hstack([X, np.zeros((1, self.input_dim - c), dtype=np.float32)])
        elif c > self.input_dim:
            X = X[:, :self.input_dim]
        return X

    def _risk(self, prob):
        t = self.threshold
        if prob < t * 0.6:           return "LOW"
        if prob < t:                  return "MEDIUM"
        if prob < t + (1-t) * 0.5:   return "HIGH"
        return "CRITICAL"

    def _bootstrap(self):
        self.input_dim = 56
        self.threshold = 0.5
        np.random.seed(42)
        n = 3000
        X = np.vstack([np.random.beta(1, 5, (n, self.input_dim)),
                       np.random.beta(5, 1, (n, self.input_dim))]).astype(np.float32)
        y = np.array([0]*n + [1]*n, dtype=np.float32)
        idx = np.random.permutation(len(X))
        X, y = X[idx], y[idx]
        self.model = PhishingDetector(self.input_dim).to(self.device)
        Xt = torch.tensor(X); yt = torch.tensor(y).unsqueeze(1)
        loader = DataLoader(TensorDataset(Xt, yt), batch_size=64, shuffle=True)
        opt = optim.Adam(self.model.parameters(), lr=0.001)
        self.model.train()
        for ep in range(30):
            for bX, by in loader:
                bX, by = bX.to(self.device), by.to(self.device)
                opt.zero_grad()
                nn.BCELoss()(self.model(bX), by).backward()
                opt.step()
            if (ep+1) % 10 == 0: print(f"  [Bootstrap] Epoch {ep+1}/30")
        self.model.eval()
        torch.save({"model_state_dict": self.model.state_dict(),
                    "input_dim": self.input_dim, "threshold": self.threshold,
                    "version": "bootstrap"}, self.model_path)
        print("[Model] Bootstrap complete. Run train_on_dataset.py for real accuracy!")


if __name__ == "__main__":
    m = ModelManager()
    print(f"input_dim={m.input_dim}  threshold={m.threshold}")
    print("Safe  :", m.predict([0.0]*m.input_dim))
    print("Phish :", m.predict([1.0]*m.input_dim))
