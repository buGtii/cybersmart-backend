"""
CyberSmart - Trainer v5 (FINAL FIX)

Root cause identified:
  The dataset page-level features (nb_hyperlinks, web_traffic, whois etc.)
  were computed by the dataset authors by actually crawling each URL.
  Our live extractor cannot reliably replicate those values in real-time,
  causing massive distribution mismatch (dataset mean length_url=60,
  our extractor gets 22 for google.com).

Solution:
  Train ONLY on the 56 URL-structure features (columns 1-56).
  These are pure string/regex features we can compute perfectly from any URL.
  No page fetching, no WHOIS, no DNS needed — instant and accurate.

  Features 1-56:
    length_url, length_hostname, ip, nb_dots, nb_hyphens, nb_at,
    nb_qm, nb_and, nb_or, nb_eq, nb_underscore, nb_tilde, nb_percent,
    nb_slash, nb_star, nb_colon, nb_comma, nb_semicolumn, nb_dollar,
    nb_space, nb_www, nb_com, nb_dslash, http_in_path, https_token,
    ratio_digits_url, ratio_digits_host, punycode, port, tld_in_path,
    tld_in_subdomain, abnormal_subdomain, nb_subdomains, prefix_suffix,
    random_domain, shortening_service, path_extension, nb_redirection,
    nb_external_redirection, length_words_raw, char_repeat,
    shortest_words_raw, shortest_word_host, shortest_word_path,
    longest_words_raw, longest_word_host, longest_word_path,
    avg_words_raw, avg_word_host, avg_word_path, phish_hints,
    domain_in_brand, brand_in_subdomain, brand_in_path,
    suspecious_tld, statistical_report
"""

import os, sys
import numpy as np

try:
    import torch, torch.nn as nn, torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:
    print("pip install torch"); sys.exit(1)

try:
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler
    from sklearn.metrics         import (classification_report,
                                         confusion_matrix,
                                         roc_auc_score, f1_score)
except ImportError:
    os.system("pip install scikit-learn")
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing   import StandardScaler
    from sklearn.metrics         import (classification_report,
                                         confusion_matrix,
                                         roc_auc_score, f1_score)

try:
    import pandas as pd
except ImportError:
    os.system("pip install pandas"); import pandas as pd


# ── Only use these 56 URL-structure features ─────────────────────────────────
URL_FEATURES = [
    "length_url", "length_hostname", "ip", "nb_dots", "nb_hyphens",
    "nb_at", "nb_qm", "nb_and", "nb_or", "nb_eq",
    "nb_underscore", "nb_tilde", "nb_percent", "nb_slash", "nb_star",
    "nb_colon", "nb_comma", "nb_semicolumn", "nb_dollar", "nb_space",
    "nb_www", "nb_com", "nb_dslash", "http_in_path", "https_token",
    "ratio_digits_url", "ratio_digits_host", "punycode", "port",
    "tld_in_path", "tld_in_subdomain", "abnormal_subdomain",
    "nb_subdomains", "prefix_suffix", "random_domain", "shortening_service",
    "path_extension", "nb_redirection", "nb_external_redirection",
    "length_words_raw", "char_repeat", "shortest_words_raw",
    "shortest_word_host", "shortest_word_path", "longest_words_raw",
    "longest_word_host", "longest_word_path", "avg_words_raw",
    "avg_word_host", "avg_word_path", "phish_hints",
    "domain_in_brand", "brand_in_subdomain", "brand_in_path",
    "suspecious_tld", "statistical_report",
]
N_FEATURES = len(URL_FEATURES)  # 56


# ─────────────────────────────────────────────────────────────────────────────
#  LOADERS
# ─────────────────────────────────────────────────────────────────────────────

def load_kaggle_csv(filepath):
    print(f"[Loader] Kaggle CSV: {filepath}")
    df = pd.read_csv(filepath)

    label_col = next((c for c in ["status","label","phishing","Result","class"]
                      if c in df.columns), df.columns[-1])

    col_vals = df[label_col].astype(str).str.strip().str.lower()
    label_map = {"phishing":1,"phish":1,"1":1,"1.0":1,"true":1,
                 "legitimate":0,"legit":0,"safe":0,"0":0,"0.0":0,"-1":1}
    df["_y"] = col_vals.map(label_map)
    df = df[df["_y"].notna()].copy()
    y = df["_y"].values.astype(np.float32)

    # Select only URL-structure features that exist in this CSV
    # Handle the nb_semicolumn vs nb_semicolon spelling difference
    available = []
    for feat in URL_FEATURES:
        if feat in df.columns:
            available.append(feat)
        elif feat == "nb_semicolumn" and "nb_semicolon" in df.columns:
            df["nb_semicolumn"] = df["nb_semicolon"]
            available.append(feat)
        else:
            print(f"[Loader] WARNING: feature '{feat}' not in CSV, filling with 0")
            df[feat] = 0.0
            available.append(feat)

    X = df[URL_FEATURES].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=1.0, neginf=0.0)

    n_phish = int(y.sum())
    n_safe  = int((y==0).sum())
    print(f"[Loader] {len(X)} samples | {N_FEATURES} URL features only")
    print(f"[Loader] Phishing: {n_phish} ({n_phish/len(y)*100:.1f}%) | "
          f"Safe: {n_safe} ({n_safe/len(y)*100:.1f}%)")
    return X, y


def load_dataset_auto(backend_dir):
    search = [backend_dir, os.path.dirname(backend_dir), os.getcwd()]
    for d in search:
        for name in ["dataset_phishing.csv","phishing.csv","phishing_dataset.csv"]:
            p = os.path.join(d, name)
            if os.path.exists(p): return load_kaggle_csv(p)
    print("ERROR: dataset_phishing.csv not found in backend folder")
    sys.exit(1)


# ─────────────────────────────────────────────────────────────────────────────
#  MODEL
# ─────────────────────────────────────────────────────────────────────────────

class PhishingNet(nn.Module):
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

    def forward(self, x): return self.net(x)


# ─────────────────────────────────────────────────────────────────────────────
#  TRAINER
# ─────────────────────────────────────────────────────────────────────────────

class Trainer:
    def __init__(self, save_path):
        self.save_path = save_path
        self.device    = "cuda" if torch.cuda.is_available() else "cpu"
        self.scaler    = StandardScaler()
        self.threshold = 0.5
        self.model     = None
        print(f"[Trainer] Device: {self.device}")

    def prepare(self, X, y):
        X_tmp, X_test, y_tmp, y_test = train_test_split(
            X, y, test_size=0.15, random_state=42, stratify=y)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tmp, y_tmp, test_size=0.15/(1-0.15), random_state=42, stratify=y_tmp)

        # Balance training set
        idx_p = np.where(y_train==1)[0]
        idx_s = np.where(y_train==0)[0]
        n = min(len(idx_p), len(idx_s))
        rng = np.random.default_rng(42)
        idx = np.concatenate([rng.choice(idx_p,n,replace=False),
                               rng.choice(idx_s,n,replace=False)])
        rng.shuffle(idx)
        X_train, y_train = X_train[idx], y_train[idx]

        X_train = np.clip(self.scaler.fit_transform(X_train), -5, 5).astype(np.float32)
        X_val   = np.clip(self.scaler.transform(X_val),       -5, 5).astype(np.float32)
        X_test  = np.clip(self.scaler.transform(X_test),      -5, 5).astype(np.float32)

        print(f"[Trainer] Train: {len(X_train)} (balanced 50/50) | "
              f"Val: {len(X_val)} | Test: {len(X_test)}")
        return X_train, X_val, X_test, y_train, y_val, y_test

    def train(self, X_train, y_train, X_val, y_val,
              epochs=150, lr=0.001, batch_size=256, patience=20):

        self.model = PhishingNet(X_train.shape[1]).to(self.device)
        loader = DataLoader(
            TensorDataset(torch.tensor(X_train), torch.tensor(y_train).unsqueeze(1)),
            batch_size=batch_size, shuffle=True)

        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=10, min_lr=1e-6)

        Xv = torch.tensor(X_val).to(self.device)
        yv = torch.tensor(y_val).unsqueeze(1).to(self.device)

        print(f"\n[Trainer] Training {epochs} epochs | lr={lr} | batch={batch_size}")
        print(f"{'Ep':<6}{'TrLoss':<10}{'VlLoss':<10}{'TrAcc':<10}{'VlAcc':<10}{'LR'}")
        print("-" * 55)

        best_vl, best_w, no_imp = float("inf"), None, 0

        for ep in range(1, epochs+1):
            self.model.train()
            tl, tc, tt = 0.0, 0, 0
            for bX, by in loader:
                bX, by = bX.to(self.device), by.to(self.device)
                optimizer.zero_grad()
                out  = self.model(bX)
                loss = criterion(out, by)
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                tl += loss.item(); tc += ((out>0.5)==by).sum().item(); tt += by.size(0)

            self.model.eval()
            with torch.no_grad():
                vo  = self.model(Xv)
                vl  = criterion(vo, yv).item()
                va  = ((vo>0.5)==yv).float().mean().item()*100

            scheduler.step(vl)
            cur_lr = optimizer.param_groups[0]["lr"]

            if vl < best_vl - 1e-4:
                best_vl = vl
                best_w  = {k:v.clone() for k,v in self.model.state_dict().items()}
                no_imp  = 0
            else:
                no_imp += 1

            if ep%20==0 or ep==1:
                print(f"  {ep:<4} {tl/len(loader):<10.4f}{vl:<10.4f}"
                      f"{tc/tt*100:<10.1f}{va:<10.1f}{cur_lr:.2e}")

            if no_imp >= patience:
                print(f"\n[Trainer] Early stop at epoch {ep}")
                break

        self.model.load_state_dict(best_w)
        print(f"[Trainer] Best val loss: {best_vl:.4f}")

    def find_threshold(self, X_val, y_val):
        self.model.eval()
        with torch.no_grad():
            probs = self.model(torch.tensor(X_val).to(self.device)).cpu().numpy().flatten()

        print("\n[Threshold] Searching best F1...")
        print(f"  {'T':<6}{'F1':<8}{'Prec%':<9}{'Rec%':<9}{'SafeAcc%':<10}{'Acc%'}")
        print("  " + "-" * 50)

        best_t, best_f1 = 0.5, 0.0
        for t in np.arange(0.05, 0.96, 0.05):
            preds   = (probs > t).astype(int)
            f1      = f1_score(y_val, preds, zero_division=0)
            rpt     = classification_report(y_val, preds, output_dict=True, zero_division=0)
            prec    = rpt.get("1",{}).get("precision",0)*100
            rec     = rpt.get("1",{}).get("recall",0)*100
            safe_r  = rpt.get("0",{}).get("recall",0)*100
            acc     = rpt.get("accuracy",0)*100
            mark    = " ◄" if f1 > best_f1 else ""
            print(f"  {t:<6.2f}{f1:<8.4f}{prec:<9.1f}{rec:<9.1f}{safe_r:<10.1f}{acc:.1f}{mark}")
            if f1 > best_f1: best_f1, best_t = f1, float(t)

        self.threshold = round(best_t, 2)
        print(f"\n[Threshold] Best = {self.threshold}  (F1={best_f1:.4f})")
        return self.threshold

    def evaluate(self, X_test, y_test):
        self.model.eval()
        with torch.no_grad():
            probs = self.model(torch.tensor(X_test).to(self.device)).cpu().numpy().flatten()

        preds = (probs > self.threshold).astype(int)
        rpt   = classification_report(y_test, preds,
                                      target_names=["Safe","Phishing"],
                                      output_dict=True)
        cm  = confusion_matrix(y_test, preds)
        auc = roc_auc_score(y_test, probs)

        acc      = rpt["accuracy"]*100
        prec     = rpt["Phishing"]["precision"]*100
        rec      = rpt["Phishing"]["recall"]*100
        f1       = rpt["Phishing"]["f1-score"]*100
        safe_acc = rpt["Safe"]["recall"]*100

        print("\n" + "="*58)
        print(f"  FINAL TEST RESULTS  (threshold={self.threshold})")
        print("="*58)
        print(f"  Overall Accuracy : {acc:.2f}%")
        print(f"  Safe Accuracy    : {safe_acc:.2f}%  ← key metric")
        print(f"  Phishing Recall  : {rec:.2f}%")
        print(f"  Phishing Prec    : {prec:.2f}%")
        print(f"  F1-Score         : {f1:.2f}%")
        print(f"  AUC-ROC          : {auc:.4f}")
        print(f"\n  Confusion Matrix (T=test set):")
        print(f"  ┌──────────────────────────────────┐")
        print(f"  │              Predicted             │")
        print(f"  │          Safe     Phishing         │")
        print(f"  │  Safe    {cm[0][0]:<9} {cm[0][1]:<6}  FP={cm[0][1]}   │")
        print(f"  │  Phish   {cm[1][0]:<9} {cm[1][1]:<6}  FN={cm[1][0]}   │")
        print(f"  └──────────────────────────────────┘")
        print("="*58)
        return {"accuracy":acc,"safe_acc":safe_acc,"recall":rec,
                "precision":prec,"f1":f1,"auc":auc}

    def save(self):
        torch.save({
            "model_state_dict": self.model.state_dict(),
            "model_class":      "PhishingDetector",
            "input_dim":        N_FEATURES,
            "n_features":       N_FEATURES,
            "feature_names":    URL_FEATURES,
            "version":          "5.0.0-url-features-only",
            "threshold":        self.threshold,
            "scaler_mean":      self.scaler.mean_.tolist(),
            "scaler_scale":     self.scaler.scale_.tolist(),
        }, self.save_path)
        print(f"\n[Trainer] Saved -> {self.save_path}")
        print(f"[Trainer] input_dim={N_FEATURES}  threshold={self.threshold}")


# ─────────────────────────────────────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    backend_dir = os.path.dirname(os.path.abspath(__file__))

    print("""
╔══════════════════════════════════════════════════╗
║  CyberSmart - Trainer v5  (URL Features Only)    ║
║  56 pure URL-structure features, no page fetch   ║
╚══════════════════════════════════════════════════╝
""")

    X, y = load_dataset_auto(backend_dir)
    print(f"\n[Main] Samples={len(X)}  Features={X.shape[1]}")

    trainer = Trainer(save_path=os.path.join(backend_dir, "model.pth"))
    X_train, X_val, X_test, y_train, y_val, y_test = trainer.prepare(X, y)
    trainer.train(X_train, y_train, X_val, y_val,
                  epochs=150, lr=0.001, batch_size=256, patience=20)
    trainer.find_threshold(X_val, y_val)
    metrics = trainer.evaluate(X_test, y_test)
    trainer.save()

    print(f"""
✅ TRAINING COMPLETE!

   Overall Accuracy : {metrics['accuracy']:.1f}%
   Safe Accuracy    : {metrics['safe_acc']:.1f}%
   Phishing Recall  : {metrics['recall']:.1f}%
   AUC-ROC          : {metrics['auc']:.4f}
   Threshold        : {trainer.threshold}
   Features used    : {N_FEATURES} (URL-structure only)

   Now run:
   python app.py
   python bulk_test.py
""")


if __name__ == "__main__":
    main()
