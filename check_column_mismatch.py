"""
Critical check: Are our feature extractor column names in the SAME ORDER
as the Kaggle dataset columns? If not, the model is getting wrong features.
"""
import os, sys
import pandas as pd
import numpy as np

backend_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, backend_dir)

csv_path = os.path.join(backend_dir, "dataset_phishing.csv")
df = pd.read_csv(csv_path)

# Get dataset columns (minus url and status)
dataset_cols = [c for c in df.columns if c not in ["url","status","URL","label"]]
print(f"Dataset columns ({len(dataset_cols)}):")
for i, c in enumerate(dataset_cols):
    print(f"  {i+1:2d}. {c}")

print()

from feature_extractor import FeatureExtractor
extractor_cols = FeatureExtractor.FEATURE_NAMES
print(f"\nExtractor columns ({len(extractor_cols)}):")
for i, c in enumerate(extractor_cols):
    print(f"  {i+1:2d}. {c}")

print(f"\n{'='*60}")
print("COLUMN COMPARISON:")
print(f"{'='*60}")

max_len = max(len(dataset_cols), len(extractor_cols))
mismatches = 0
for i in range(max_len):
    d = dataset_cols[i] if i < len(dataset_cols) else "MISSING"
    e = extractor_cols[i] if i < len(extractor_cols) else "MISSING"
    match = "✅" if d == e else "❌"
    if d != e:
        mismatches += 1
    print(f"  {i+1:2d}. Dataset: {d:<35} Extractor: {e:<35} {match}")

print(f"\nTotal mismatches: {mismatches}")
print(f"Dataset cols: {len(dataset_cols)}, Extractor cols: {len(extractor_cols)}")

# Also show sample values from dataset for a few rows
print(f"\n{'='*60}")
print("DATASET SAMPLE VALUES (first 3 rows, first 10 cols):")
print(f"{'='*60}")
drop = [c for c in ["url","status","URL","label"] if c in df.columns]
X_df = df.drop(columns=drop).select_dtypes(include=[np.number])
print(f"Numeric columns: {X_df.shape[1]}")
print(X_df.head(3).iloc[:, :10].to_string())
