"""
generate_models.py
──────────────────
Run this ONCE after training in the Jupyter notebook to produce
all the .pkl files the Streamlit app needs.

Usage:
  python generate_models.py

Prerequisites:
  1. Run the full Jupyter notebook first (all cells top to bottom).
  2. Confirm the notebook created a  models/  folder with .pkl files.
  3. Copy that  models/  folder next to this script.

If you already have the models/ folder from the notebook, you do NOT
need to run this script — the app.py loads from there directly.

This script is a FALLBACK that re-trains only the lightweight sklearn
models (no Keras) so you can quickly verify the pipeline without a GPU.
"""

import os
import numpy as np
import pandas as pd
import joblib
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ── CONFIG ───────────────────────────────────────────────────────────────────
DATA_PATH  = "data/raw/Crop_recommendation.csv"   # adjust if needed
MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)

# ── LOAD DATA ─────────────────────────────────────────────────────────────────
print("📂  Loading data …")
if not Path(DATA_PATH).exists():
    raise FileNotFoundError(
        f"\n❌  Could not find: {DATA_PATH}\n"
        "    Please place Crop_recommendation.csv in data/raw/ and re-run.\n"
        "    Download from: https://www.kaggle.com/datasets/atharvaingle/crop-recommendation-dataset"
    )

df = pd.read_csv(DATA_PATH)
df.columns = df.columns.str.strip()
df.dropna(inplace=True)
df.drop_duplicates(inplace=True)
print(f"   Shape: {df.shape}")

# ── ENCODE & SPLIT ────────────────────────────────────────────────────────────
le_crop = LabelEncoder()
df["crop_encoded"] = le_crop.fit_transform(df["label"])

X = df[["N", "P", "K", "temperature", "humidity", "ph", "rainfall"]].values
y = df["crop_encoded"].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# ── TRAIN MODELS ─────────────────────────────────────────────────────────────
print("\n🤖  Training models …")

models = {
    "rf_cls":  (RandomForestClassifier(n_estimators=100, random_state=42), False),
    "lr_cls":  (LogisticRegression(max_iter=1000, random_state=42),        True),
    "dt_cls":  (DecisionTreeClassifier(max_depth=10, random_state=42),     False),
    "knn_cls": (KNeighborsClassifier(n_neighbors=5),                       True),
    "svm_cls": (SVC(kernel="rbf", probability=True, random_state=42),      True),
}

for name, (mdl, needs_scale) in models.items():
    X_tr = X_train_sc if needs_scale else X_train
    X_te = X_test_sc  if needs_scale else X_test
    mdl.fit(X_tr, y_train)
    acc = accuracy_score(y_test, mdl.predict(X_te))
    print(f"   {name:<10}  accuracy = {acc*100:.2f}%")
    joblib.dump(mdl, MODELS_DIR / f"{name}.pkl")

# ── SAVE ENCODERS / SCALERS ──────────────────────────────────────────────────
joblib.dump(scaler,  MODELS_DIR / "scaler_cls.pkl")
joblib.dump(le_crop, MODELS_DIR / "le_crop.pkl")

# ── ALSO SAVE brain.pkl (all-in-one) ─────────────────────────────────────────
rf = models["rf_cls"][0]
joblib.dump({"model": rf, "scaler": scaler, "encoder": le_crop}, "brain.pkl")

# ── DONE ──────────────────────────────────────────────────────────────────────
print("\n✅  Saved to models/:")
for f in sorted(MODELS_DIR.iterdir()):
    size_kb = f.stat().st_size / 1024
    print(f"   {f.name:<25}  {size_kb:.1f} KB")
print("\n   brain.pkl  (all-in-one bundle)")
print("\n🚀  Now run:  streamlit run app.py")
