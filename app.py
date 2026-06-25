"""
Smart Agriculture Predictor — Streamlit Dashboard
Dark-theme GUI inspired by EcoSync dashboard design
"""

import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os
import json
from pathlib import Path

# ─── Page Config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroSync — Smart Agriculture",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS (EcoSync-inspired dark theme) ─────────────────────────────────
st.markdown("""
<style>
/* ---------- Global Reset & Font ---------- */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
    background-color: #0f1117;
    color: #e2e8f0;
}

/* ---------- Main background ---------- */
.stApp {
    background-color: #0f1117;
}

/* ---------- Sidebar ---------- */
[data-testid="stSidebar"] {
    background-color: #161b27 !important;
    border-right: 1px solid #1e2d40;
}
[data-testid="stSidebar"] .stMarkdown h1,
[data-testid="stSidebar"] .stMarkdown h2,
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #4ade80;
}

/* ---------- Headers ---------- */
h1 { color: #f8fafc; font-weight: 700; letter-spacing: -0.5px; }
h2 { color: #cbd5e1; font-weight: 600; }
h3 { color: #94a3b8; font-weight: 500; }

/* ---------- Metric Cards ---------- */
.metric-card {
    background: #161b27;
    border: 1px solid #1e2d40;
    border-radius: 12px;
    padding: 20px 24px;
    margin-bottom: 12px;
    transition: border-color 0.2s;
}
.metric-card:hover { border-color: #4ade80; }
.metric-card .label {
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1px;
    color: #64748b;
    margin-bottom: 6px;
}
.metric-card .value {
    font-size: 28px;
    font-weight: 700;
    color: #f8fafc;
    line-height: 1.1;
}
.metric-card .sub {
    font-size: 12px;
    color: #4ade80;
    margin-top: 4px;
}

/* ---------- Crop Recommendation Cards ---------- */
.crop-card {
    background: #161b27;
    border: 1px solid #1e2d40;
    border-radius: 14px;
    padding: 22px 24px;
    margin-bottom: 16px;
    position: relative;
    overflow: hidden;
}
.crop-card.rank-1 { border-left: 4px solid #4ade80; }
.crop-card.rank-2 { border-left: 4px solid #3b82f6; }
.crop-card.rank-3 { border-left: 4px solid #a78bfa; }

.crop-name {
    font-size: 22px;
    font-weight: 700;
    color: #f8fafc;
    text-transform: capitalize;
}
.confidence-badge {
    display: inline-block;
    background: #14532d;
    color: #4ade80;
    font-size: 12px;
    font-weight: 600;
    padding: 3px 10px;
    border-radius: 20px;
    margin-left: 10px;
}
.stat-row {
    display: flex;
    flex-wrap: wrap;
    gap: 16px;
    margin-top: 14px;
}
.stat-item {
    background: #0f1117;
    border-radius: 8px;
    padding: 10px 14px;
    min-width: 120px;
}
.stat-item .s-label {
    font-size: 10px;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    color: #64748b;
    margin-bottom: 4px;
}
.stat-item .s-val {
    font-size: 15px;
    font-weight: 600;
    color: #e2e8f0;
}

/* ---------- Section divider ---------- */
.section-header {
    display: flex;
    align-items: center;
    gap: 10px;
    margin: 28px 0 16px 0;
}
.section-header .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: #4ade80;
    flex-shrink: 0;
}
.section-title {
    font-size: 13px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 1.2px;
    color: #64748b;
}

/* ---------- Status badges ---------- */
.badge-green  { color: #4ade80; background: #14532d33; border-radius: 6px; padding: 2px 8px; font-size: 12px; }
.badge-blue   { color: #60a5fa; background: #1e3a5f33; border-radius: 6px; padding: 2px 8px; font-size: 12px; }
.badge-purple { color: #c084fc; background: #3b1f5533; border-radius: 6px; padding: 2px 8px; font-size: 12px; }

/* ---------- Top nav bar ---------- */
.topbar {
    background: #161b27;
    border: 1px solid #1e2d40;
    border-radius: 14px;
    padding: 16px 24px;
    display: flex;
    align-items: center;
    justify-content: space-between;
    margin-bottom: 24px;
}
.app-logo {
    font-size: 20px;
    font-weight: 700;
    color: #4ade80;
    letter-spacing: -0.5px;
}
.app-sub {
    font-size: 12px;
    color: #64748b;
}

/* ---------- Input styling ---------- */
.stNumberInput input, .stSelectbox select, .stSlider {
    background: #0f1117 !important;
    border: 1px solid #1e2d40 !important;
    color: #e2e8f0 !important;
    border-radius: 8px !important;
}
label[data-testid="stWidgetLabel"] {
    color: #94a3b8 !important;
    font-size: 13px !important;
    font-weight: 500 !important;
}

/* ---------- Buttons ---------- */
.stButton > button {
    background: #16a34a !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 12px 28px !important;
    width: 100%;
    transition: background 0.2s !important;
}
.stButton > button:hover {
    background: #15803d !important;
}

/* ---------- Alerts ---------- */
.stAlert { border-radius: 10px !important; }

/* ---------- Tabs ---------- */
.stTabs [data-baseweb="tab-list"] {
    background: #161b27;
    border-radius: 10px;
    padding: 4px;
    gap: 4px;
    border: 1px solid #1e2d40;
}
.stTabs [data-baseweb="tab"] {
    background: transparent;
    color: #64748b;
    border-radius: 8px;
    font-weight: 500;
    font-size: 13px;
}
.stTabs [aria-selected="true"] {
    background: #16a34a !important;
    color: #fff !important;
}

/* ---------- Footer ---------- */
.footer {
    text-align: center;
    color: #334155;
    font-size: 11px;
    margin-top: 40px;
    padding: 20px 0;
    border-top: 1px solid #1e2d40;
}
</style>
""", unsafe_allow_html=True)

# ─── Crop Economics & Requirements (embedded from notebook) ──────────────────
CROP_ECONOMICS = {
    "rice":        {"cost": 35000, "profit": 25000, "market": "₹18–22/kg"},
    "maize":       {"cost": 28000, "profit": 20000, "market": "₹15–18/kg"},
    "chickpea":    {"cost": 22000, "profit": 30000, "market": "₹45–60/kg"},
    "kidneybeans": {"cost": 24000, "profit": 28000, "market": "₹60–80/kg"},
    "pigeonpeas":  {"cost": 20000, "profit": 26000, "market": "₹50–65/kg"},
    "mothbeans":   {"cost": 18000, "profit": 22000, "market": "₹40–55/kg"},
    "mungbean":    {"cost": 20000, "profit": 25000, "market": "₹55–70/kg"},
    "blackgram":   {"cost": 19000, "profit": 24000, "market": "₹55–70/kg"},
    "lentil":      {"cost": 21000, "profit": 27000, "market": "₹50–65/kg"},
    "pomegranate": {"cost": 60000, "profit": 90000, "market": "₹50–80/kg"},
    "banana":      {"cost": 50000, "profit": 70000, "market": "₹15–30/kg"},
    "mango":       {"cost": 40000, "profit": 60000, "market": "₹30–80/kg"},
    "grapes":      {"cost": 80000, "profit": 120000, "market": "₹40–100/kg"},
    "watermelon":  {"cost": 30000, "profit": 35000, "market": "₹8–15/kg"},
    "muskmelon":   {"cost": 28000, "profit": 32000, "market": "₹10–20/kg"},
    "apple":       {"cost": 70000, "profit": 100000, "market": "₹60–150/kg"},
    "orange":      {"cost": 45000, "profit": 65000, "market": "₹20–50/kg"},
    "papaya":      {"cost": 35000, "profit": 50000, "market": "₹12–25/kg"},
    "coconut":     {"cost": 25000, "profit": 40000, "market": "₹15–30/piece"},
    "cotton":      {"cost": 35000, "profit": 30000, "market": "₹55–75/kg"},
    "jute":        {"cost": 20000, "profit": 18000, "market": "₹35–50/kg"},
    "coffee":      {"cost": 55000, "profit": 80000, "market": "₹80–150/kg"},
}

CROP_REQUIREMENTS = {
    "rice":        {"Water": "High (1200–2000 mm)",       "Soil": "Clay/Loamy",        "Season": "Kharif",            "Fertilizer": "Urea + DAP",       "Tools": "Tractor, Transplanter"},
    "maize":       {"Water": "Moderate (500–800 mm)",     "Soil": "Sandy Loam",         "Season": "Kharif/Rabi",       "Fertilizer": "17-17-17 NPK",     "Tools": "Seed Drill, Sprayer"},
    "chickpea":    {"Water": "Low (300–400 mm)",          "Soil": "Sandy/Loamy",        "Season": "Rabi",              "Fertilizer": "DAP + Rhizobium",  "Tools": "Seed Drill, Sprayer"},
    "kidneybeans": {"Water": "Moderate (600 mm)",         "Soil": "Loamy",              "Season": "Kharif",            "Fertilizer": "10-26-26 NPK",     "Tools": "Sprayer, Harvester"},
    "pigeonpeas":  {"Water": "Low–Moderate (600 mm)",     "Soil": "Sandy/Red",          "Season": "Kharif",            "Fertilizer": "DAP",              "Tools": "Sprayer"},
    "mothbeans":   {"Water": "Low (200–350 mm)",          "Soil": "Sandy",              "Season": "Kharif",            "Fertilizer": "20-20 NPK",        "Tools": "Sprayer"},
    "mungbean":    {"Water": "Low–Moderate (400 mm)",     "Soil": "Sandy Loam",         "Season": "Summer/Kharif",     "Fertilizer": "DAP",              "Tools": "Sprayer"},
    "blackgram":   {"Water": "Moderate (500 mm)",         "Soil": "Loamy/Clay",         "Season": "Kharif/Rabi",       "Fertilizer": "DAP",              "Tools": "Sprayer"},
    "lentil":      {"Water": "Low (350 mm)",              "Soil": "Sandy Loam",         "Season": "Rabi",              "Fertilizer": "20-20 NPK",        "Tools": "Sprayer"},
    "pomegranate": {"Water": "Low (500 mm)",              "Soil": "Sandy Loam/Red",     "Season": "Whole Year",        "Fertilizer": "17-17-17 NPK",     "Tools": "Drip Irrigation"},
    "banana":      {"Water": "High (1000 mm)",            "Soil": "Loamy/Black",        "Season": "Whole Year",        "Fertilizer": "Urea + MOP",       "Tools": "Drip Irrigation"},
    "mango":       {"Water": "Moderate (750 mm)",         "Soil": "Loamy/Sandy",        "Season": "Summer",            "Fertilizer": "17-17-17 NPK",     "Tools": "Sprayer"},
    "grapes":      {"Water": "Moderate (700 mm)",         "Soil": "Sandy Loam",         "Season": "Rabi",              "Fertilizer": "14-35-14 NPK",     "Tools": "Drip, Pruner"},
    "watermelon":  {"Water": "Moderate (400–600 mm)",     "Soil": "Sandy",              "Season": "Summer",            "Fertilizer": "20-20 NPK",        "Tools": "Drip, Sprayer"},
    "muskmelon":   {"Water": "Moderate (400 mm)",         "Soil": "Sandy/Loamy",        "Season": "Summer",            "Fertilizer": "20-20 NPK",        "Tools": "Sprayer"},
    "apple":       {"Water": "Moderate (900 mm)",         "Soil": "Loamy",              "Season": "Winter/Whole Year", "Fertilizer": "17-17-17 NPK",     "Tools": "Sprayer, Cold Storage"},
    "orange":      {"Water": "Moderate (600–700 mm)",     "Soil": "Sandy Loam",         "Season": "Rabi",              "Fertilizer": "17-17-17 NPK",     "Tools": "Sprayer"},
    "papaya":      {"Water": "Moderate (800 mm)",         "Soil": "Sandy Loam/Loamy",   "Season": "Whole Year",        "Fertilizer": "Urea + MOP",       "Tools": "Drip Irrigation"},
    "coconut":     {"Water": "High (1000–2500 mm)",       "Soil": "Loamy/Sandy",        "Season": "Whole Year",        "Fertilizer": "Urea + DAP",       "Tools": "Sprayer"},
    "cotton":      {"Water": "Moderate (500–700 mm)",     "Soil": "Black/Loamy",        "Season": "Kharif",            "Fertilizer": "14-35-14 NPK",     "Tools": "Sprayer, Picker"},
    "jute":        {"Water": "High (1000–1500 mm)",       "Soil": "Loamy/Sandy",        "Season": "Kharif",            "Fertilizer": "Urea",             "Tools": "Harvester"},
    "coffee":      {"Water": "High (1500 mm)",            "Soil": "Red/Loamy",          "Season": "Whole Year",        "Fertilizer": "17-17-17 NPK",     "Tools": "Sprayer, Pulper"},
}

CROP_EMOJI = {
    "rice": "🌾", "maize": "🌽", "chickpea": "🫘", "kidneybeans": "🫘",
    "pigeonpeas": "🌿", "mothbeans": "🌱", "mungbean": "🌱", "blackgram": "🫘",
    "lentil": "🫘", "pomegranate": "🍎", "banana": "🍌", "mango": "🥭",
    "grapes": "🍇", "watermelon": "🍉", "muskmelon": "🍈", "apple": "🍎",
    "orange": "🍊", "papaya": "🍐", "coconut": "🥥", "cotton": "🌸",
    "jute": "🌿", "coffee": "☕",
}

# ─── Model Loading ────────────────────────────────────────────────────────────
MODEL_DIR = Path("models")

@st.cache_resource(show_spinner=False)
def load_models():
    """Load all saved models and encoders."""
    models = {}
    if not MODEL_DIR.exists():
        return None, None, None, None

    try:
        models["lr_cls"]    = joblib.load(MODEL_DIR / "lr_cls.pkl")
        models["dt_cls"]    = joblib.load(MODEL_DIR / "dt_cls.pkl")
        models["rf_cls"]    = joblib.load(MODEL_DIR / "rf_cls.pkl")
        models["knn_cls"]   = joblib.load(MODEL_DIR / "knn_cls.pkl")
        models["svm_cls"]   = joblib.load(MODEL_DIR / "svm_cls.pkl")
        models["lr_reg"]    = joblib.load(MODEL_DIR / "lr_reg.pkl")
        models["dt_reg"]    = joblib.load(MODEL_DIR / "dt_reg.pkl")
        models["rf_reg"]    = joblib.load(MODEL_DIR / "rf_reg.pkl")
        models["knn_reg"]   = joblib.load(MODEL_DIR / "knn_reg.pkl")
        models["svr"]       = joblib.load(MODEL_DIR / "svr.pkl")
        le_crop             = joblib.load(MODEL_DIR / "le_crop.pkl")
        le_dict_yield       = joblib.load(MODEL_DIR / "le_dict_yield.pkl")
        scaler_cls          = joblib.load(MODEL_DIR / "scaler_cls.pkl")
        scaler_reg          = joblib.load(MODEL_DIR / "scaler_reg.pkl")

        # Try loading Keras models (optional)
        try:
            import tensorflow as tf
            for key, fname in [("ann_cls","ann_cls.h5"),("cnn_cls","cnn_cls.h5"),
                                ("ann_reg","ann_reg.h5"),("cnn_reg","cnn_reg.h5")]:
                p = MODEL_DIR / fname
                if not p.exists():
                    p = MODEL_DIR / fname.replace(".h5", ".keras")
                if p.exists():
                    models[key] = tf.keras.models.load_model(str(p))
        except Exception:
            pass

        return models, le_crop, le_dict_yield, scaler_cls, scaler_reg
    except Exception as e:
        return None, None, None, None, None


def recommend_crops(n, p, k, temperature, humidity, ph, rainfall,
                    land_area_ha, models, le_crop, scaler_cls,
                    cls_model_name="Random Forest", top_n=3):
    """Run crop recommendation using loaded models."""
    feat = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    feat_scaled = scaler_cls.transform(feat)

    cls_map = {
        "Logistic Regression": ("lr_cls",  True),
        "Decision Tree":       ("dt_cls",  False),
        "Random Forest":       ("rf_cls",  False),
        "KNN":                 ("knn_cls", True),
        "SVM":                 ("svm_cls", True),
        "ANN":                 ("ann_cls", True),
        "1D-CNN":              ("cnn_cls", True),
    }

    model_key, needs_scale = cls_map[cls_model_name]
    if model_key not in models:
        return []

    mdl = models[model_key]
    inp = feat_scaled if needs_scale else feat

    if cls_model_name == "ANN":
        proba = mdl.predict(inp, verbose=0)[0]
    elif cls_model_name == "1D-CNN":
        proba = mdl.predict(inp.reshape(-1, 7, 1), verbose=0)[0]
    elif hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(inp)[0]
    else:
        pred = mdl.predict(inp)[0]
        proba = np.zeros(len(le_crop.classes_))
        proba[pred] = 1.0

    top_indices = np.argsort(proba)[::-1][:top_n]
    recs = []
    for idx in top_indices:
        crop = le_crop.classes_[idx].lower()
        conf = round(float(proba[idx]) * 100, 1)
        eco  = CROP_ECONOMICS.get(crop, {"cost": 25000, "profit": 20000, "market": "N/A"})
        req  = CROP_REQUIREMENTS.get(crop, {})
        total_cost   = eco["cost"]   * land_area_ha
        total_profit = eco["profit"] * land_area_ha
        roi = (total_profit / total_cost * 100) if total_cost > 0 else 0

        recs.append({
            "Crop":         crop,
            "Confidence":   conf,
            "Cost/ha":      eco["cost"],
            "Profit/ha":    eco["profit"],
            "Total Cost":   total_cost,
            "Total Profit": total_profit,
            "ROI (%)":      roi,
            "Market Price": eco["market"],
            "Water Needs":  req.get("Water",  "N/A"),
            "Soil Type":    req.get("Soil",   "N/A"),
            "Best Season":  req.get("Season", "N/A"),
            "Fertilizer":   req.get("Fertilizer", "N/A"),
            "Tools Needed": req.get("Tools", "N/A"),
        })
    return recs


# ─── Top Bar ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="topbar">
  <div>
    <div class="app-logo">🌾 AgroSync</div>
    <div class="app-sub">Smart Agriculture Prediction System</div>
  </div>
  <div style="text-align:right">
    <div style="font-size:12px; color:#4ade80; font-weight:600;">● ONLINE</div>
    <div style="font-size:11px; color:#475569;">AI-Powered · Multi-Model</div>
  </div>
</div>
""", unsafe_allow_html=True)

# ─── Load Models ─────────────────────────────────────────────────────────────
with st.spinner("Loading models..."):
    result = load_models()

if result[0] is None:
    # Demo / No-model mode
    models_loaded = False
    models_obj = le_crop = le_dict_yield = scaler_cls = scaler_reg = None
    st.warning("⚠️ **Models not found.** Place your `models/` folder in the same directory as `app.py` and restart. Running in **Demo Mode** — results are simulated.")
else:
    models_loaded = True
    models_obj, le_crop, le_dict_yield, scaler_cls, scaler_reg = result

# ─── Sidebar Inputs ───────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧪 Soil & Field Parameters")
    st.markdown("---")

    n_val = st.number_input("Nitrogen (N) — mg/kg",       min_value=0,    max_value=200,  value=90,    step=1)
    p_val = st.number_input("Phosphorus (P) — mg/kg",     min_value=0,    max_value=200,  value=42,    step=1)
    k_val = st.number_input("Potassium (K) — mg/kg",      min_value=0,    max_value=200,  value=43,    step=1)
    ph_val = st.slider(      "Soil pH",                    min_value=0.0,  max_value=14.0, value=6.5,   step=0.1)

    st.markdown("### 🌦️ Climate Conditions")
    st.markdown("---")
    temp_val     = st.number_input("Temperature — °C",   min_value=-10.0, max_value=55.0,  value=25.0, step=0.5)
    humidity_val = st.number_input("Humidity — %",       min_value=0.0,   max_value=100.0, value=75.0, step=1.0)
    rainfall_val = st.number_input("Rainfall — mm",      min_value=0.0,   max_value=3000.0,value=180.0,step=10.0)

    st.markdown("### 🏞️ Farm Details")
    st.markdown("---")
    land_ha = st.number_input("Land Area — hectares",    min_value=0.1,   max_value=1000.0, value=2.5, step=0.5)
    top_n   = st.selectbox("Top Recommendations",        options=[1, 2, 3, 4, 5], index=2)

    st.markdown("### 🤖 ML Model Selection")
    st.markdown("---")
    cls_models_available = ["Random Forest", "Decision Tree", "KNN",
                             "Logistic Regression", "SVM"]
    if models_loaded and "ann_cls" in models_obj:
        cls_models_available += ["ANN", "1D-CNN"]

    model_choice = st.selectbox("Classifier", cls_models_available, index=0)
    st.markdown("---")
    run_btn = st.button("🌾 Get Crop Recommendations", use_container_width=True)

# ─── Main Tabs ────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["🌱 Crop Recommendations", "📊 Model Accuracy", "📖 How It Works"])

# ════════════════════════════════════════════════════════════
# TAB 1 — Crop Recommendations
# ════════════════════════════════════════════════════════════
with tab1:
    # Always show current input summary
    st.markdown("""<div class="section-header">
      <div class="dot"></div>
      <div class="section-title">Current Field Summary</div>
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4, c5, c6 = st.columns(6)
    def mini_card(col, label, value, sub=""):
        col.markdown(f"""<div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{value}</div>
          <div class="sub">{sub}</div>
        </div>""", unsafe_allow_html=True)

    mini_card(c1, "Nitrogen",    f"{n_val}",        "mg/kg")
    mini_card(c2, "Phosphorus",  f"{p_val}",        "mg/kg")
    mini_card(c3, "Potassium",   f"{k_val}",        "mg/kg")
    mini_card(c4, "pH",          f"{ph_val}",       "soil pH")
    mini_card(c5, "Temperature", f"{temp_val}°C",   "ambient")
    mini_card(c6, "Rainfall",    f"{rainfall_val}", "mm/year")

    st.markdown("---")

    if run_btn or st.session_state.get("last_run"):
        st.session_state["last_run"] = True

        st.markdown("""<div class="section-header">
          <div class="dot"></div>
          <div class="section-title">AI Recommendations</div>
        </div>""", unsafe_allow_html=True)

        with st.spinner("Running AI prediction engine..."):
            if models_loaded:
                recs = recommend_crops(
                    n=n_val, p=p_val, k=k_val,
                    temperature=temp_val, humidity=humidity_val,
                    ph=ph_val, rainfall=rainfall_val,
                    land_area_ha=land_ha,
                    models=models_obj, le_crop=le_crop, scaler_cls=scaler_cls,
                    cls_model_name=model_choice, top_n=top_n
                )
            else:
                # Demo results
                demo_crops = ["rice", "maize", "mungbean", "chickpea", "banana"]
                recs = []
                for i, crop in enumerate(demo_crops[:top_n]):
                    eco = CROP_ECONOMICS.get(crop, {"cost":25000,"profit":20000,"market":"N/A"})
                    req = CROP_REQUIREMENTS.get(crop, {})
                    tc = eco["cost"]   * land_ha
                    tp = eco["profit"] * land_ha
                    recs.append({
                        "Crop": crop,
                        "Confidence": round(95 - i*15, 1),
                        "Cost/ha": eco["cost"], "Profit/ha": eco["profit"],
                        "Total Cost": tc, "Total Profit": tp,
                        "ROI (%)": tp/tc*100,
                        "Market Price": eco["market"],
                        "Water Needs": req.get("Water","N/A"),
                        "Soil Type":   req.get("Soil","N/A"),
                        "Best Season": req.get("Season","N/A"),
                        "Fertilizer":  req.get("Fertilizer","N/A"),
                        "Tools Needed":req.get("Tools","N/A"),
                    })

        rank_classes = ["rank-1", "rank-2", "rank-3", "rank-2", "rank-3"]
        rank_badges  = ["🥇 Top Pick", "🥈 2nd Choice", "🥉 3rd Choice", "4th", "5th"]

        for i, rec in enumerate(recs):
            crop     = rec["Crop"]
            emoji    = CROP_EMOJI.get(crop, "🌿")
            rc       = rank_classes[i] if i < len(rank_classes) else "rank-3"
            badge    = rank_badges[i] if i < len(rank_badges) else f"#{i+1}"
            conf_col = "#4ade80" if rec["Confidence"] >= 70 else "#facc15" if rec["Confidence"] >= 40 else "#f87171"

            st.markdown(f"""
            <div class="crop-card {rc}">
              <div style="display:flex; align-items:center; justify-content:space-between; flex-wrap:wrap; gap:8px;">
                <div>
                  <span style="font-size:28px;">{emoji}</span>
                  <span class="crop-name"> {crop.title()}</span>
                  <span class="confidence-badge" style="background:{conf_col}22; color:{conf_col};">
                    {rec['Confidence']}% confidence
                  </span>
                </div>
                <div style="font-size:12px; color:#64748b; font-weight:600;">{badge}</div>
              </div>

              <div class="stat-row">
                <div class="stat-item">
                  <div class="s-label">💰 Cost / ha</div>
                  <div class="s-val">₹{rec['Cost/ha']:,.0f}</div>
                </div>
                <div class="stat-item">
                  <div class="s-label">📈 Profit / ha</div>
                  <div class="s-val">₹{rec['Profit/ha']:,.0f}</div>
                </div>
                <div class="stat-item">
                  <div class="s-label">🏠 Total Cost</div>
                  <div class="s-val">₹{rec['Total Cost']:,.0f}</div>
                </div>
                <div class="stat-item">
                  <div class="s-label">💵 Total Profit</div>
                  <div class="s-val">₹{rec['Total Profit']:,.0f}</div>
                </div>
                <div class="stat-item">
                  <div class="s-label">📊 ROI</div>
                  <div class="s-val">{rec['ROI (%)']:.1f}%</div>
                </div>
                <div class="stat-item">
                  <div class="s-label">💹 Market</div>
                  <div class="s-val">{rec['Market Price']}</div>
                </div>
                <div class="stat-item">
                  <div class="s-label">💧 Water</div>
                  <div class="s-val" style="font-size:12px;">{rec['Water Needs']}</div>
                </div>
                <div class="stat-item">
                  <div class="s-label">🌱 Soil</div>
                  <div class="s-val" style="font-size:12px;">{rec['Soil Type']}</div>
                </div>
                <div class="stat-item">
                  <div class="s-label">🗓 Season</div>
                  <div class="s-val" style="font-size:12px;">{rec['Best Season']}</div>
                </div>
                <div class="stat-item">
                  <div class="s-label">🧪 Fertilizer</div>
                  <div class="s-val" style="font-size:12px;">{rec['Fertilizer']}</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # Summary table
        if recs:
            st.markdown("""<div class="section-header" style="margin-top:28px;">
              <div class="dot"></div>
              <div class="section-title">Quick Comparison Table</div>
            </div>""", unsafe_allow_html=True)
            df_show = pd.DataFrame(recs)[["Crop","Confidence","Cost/ha","Profit/ha","ROI (%)","Best Season","Fertilizer"]]
            df_show["Crop"] = df_show["Crop"].str.title()
            df_show["Confidence"] = df_show["Confidence"].astype(str) + "%"
            df_show["ROI (%)"] = df_show["ROI (%)"].round(1).astype(str) + "%"
            df_show = df_show.rename(columns={"Cost/ha":"Cost/ha (₹)","Profit/ha":"Profit/ha (₹)"})
            st.dataframe(df_show, use_container_width=True, hide_index=True)

    else:
        st.info("👈 Fill in field parameters in the sidebar and click **Get Crop Recommendations**.")


# ════════════════════════════════════════════════════════════
# TAB 2 — Model Accuracy
# ════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""<div class="section-header">
      <div class="dot"></div>
      <div class="section-title">Trained Model Performance</div>
    </div>""", unsafe_allow_html=True)

    cls_data = {
        "Model":    ["Logistic Regression","Decision Tree","Random Forest","KNN","SVM","ANN","1D-CNN"],
        "Accuracy": [96.82, 98.18, 99.55, 97.27, 98.64, 99.09, 98.64],
        "Type":     ["Classical","Classical","Classical","Classical","Classical","Deep Learning","Deep Learning"],
    }
    reg_data = {
        "Model":  ["Linear Regression","Decision Tree","Random Forest","KNN","SVR","ANN","1D-CNN"],
        "R² Score":[0.71, 0.89, 0.93, 0.78, 0.75, 0.71, 0.73],
        "RMSE":   [5.92, 3.80, 2.95, 4.81, 5.21, 5.92, 5.63],
        "Type":   ["Classical","Classical","Classical","Classical","Classical","Deep Learning","Deep Learning"],
    }

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### 🌾 Crop Classification — Accuracy (%)")
        df_cls = pd.DataFrame(cls_data)
        df_cls = df_cls.sort_values("Accuracy", ascending=False)

        # Bar chart via streamlit
        st.bar_chart(
            df_cls.set_index("Model")["Accuracy"],
            use_container_width=True,
            height=280,
            color="#4ade80"
        )
        st.dataframe(df_cls[["Model","Accuracy","Type"]], use_container_width=True, hide_index=True)

    with col_r:
        st.markdown("#### 🌾 Yield Regression — R² Score")
        df_reg = pd.DataFrame(reg_data)
        df_reg = df_reg.sort_values("R² Score", ascending=False)

        st.bar_chart(
            df_reg.set_index("Model")["R² Score"],
            use_container_width=True,
            height=280,
            color="#3b82f6"
        )
        st.dataframe(df_reg[["Model","R² Score","RMSE","Type"]], use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("""<div class="section-header">
      <div class="dot"></div>
      <div class="section-title">Datasets Used</div>
    </div>""", unsafe_allow_html=True)

    d1, d2, d3 = st.columns(3)
    d1.markdown("""<div class="metric-card">
      <div class="label">📦 Crop Recommendation</div>
      <div class="value">2,200</div>
      <div class="sub">rows · 7 soil+climate features</div>
    </div>""", unsafe_allow_html=True)
    d2.markdown("""<div class="metric-card">
      <div class="label">📦 Crop Yield</div>
      <div class="value">19,689</div>
      <div class="sub">rows · Indian state-wise</div>
    </div>""", unsafe_allow_html=True)
    d3.markdown("""<div class="metric-card">
      <div class="label">📦 Fertilizer</div>
      <div class="value">99</div>
      <div class="sub">rows · soil × crop combos</div>
    </div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════
# TAB 3 — How It Works
# ════════════════════════════════════════════════════════════
with tab3:
    st.markdown("""<div class="section-header">
      <div class="dot"></div>
      <div class="section-title">System Architecture</div>
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    ### 🔄 Prediction Pipeline

    **Inputs →** Soil NPK, pH, Temperature, Humidity, Rainfall, Land Area

    | Step | Component | Output |
    |------|-----------|--------|
    | 1️⃣ | **StandardScaler** | Normalized feature vector |
    | 2️⃣ | **Classifier** (RF/SVM/ANN/CNN) | Crop class probabilities |
    | 3️⃣ | **Top-N Selection** | Ranked crop candidates |
    | 4️⃣ | **Economics Engine** | Cost, Profit, ROI per hectare |
    | 5️⃣ | **Requirements Lookup** | Water, Soil, Season, Fertilizer |

    ---

    ### 🤖 Available Models

    | Model | Type | Best For |
    |-------|------|----------|
    | Random Forest | Classical ML | Best overall accuracy (99.5%) |
    | 1D-CNN | Deep Learning | Pattern learning in sequences |
    | ANN | Deep Learning | Non-linear feature interactions |
    | SVM | Classical ML | High-dimensional spaces |
    | KNN | Classical ML | Local similarity matching |
    | Decision Tree | Classical ML | Interpretable decisions |
    | Logistic Regression | Classical ML | Fast baseline |

    ---

    ### 📁 Required `models/` Folder Contents

    ```
    models/
    ├── rf_cls.pkl       ← Random Forest Classifier
    ├── lr_cls.pkl       ← Logistic Regression Classifier
    ├── dt_cls.pkl       ← Decision Tree Classifier
    ├── knn_cls.pkl      ← KNN Classifier
    ├── svm_cls.pkl      ← SVM Classifier
    ├── ann_cls.h5       ← ANN Classifier (Keras)
    ├── cnn_cls.h5       ← 1D-CNN Classifier (Keras)
    ├── rf_reg.pkl       ← Random Forest Regressor
    ├── lr_reg.pkl       ← Linear Regression
    ├── dt_reg.pkl       ← Decision Tree Regressor
    ├── knn_reg.pkl      ← KNN Regressor
    ├── svr.pkl          ← SVR Regressor
    ├── ann_reg.h5       ← ANN Regressor
    ├── cnn_reg.h5       ← 1D-CNN Regressor
    ├── le_crop.pkl      ← Label Encoder (crops)
    ├── le_dict_yield.pkl← Label Encoders (yield)
    ├── scaler_cls.pkl   ← StandardScaler (classification)
    └── scaler_reg.pkl   ← StandardScaler (regression)
    ```
    """)


# ─── Footer ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  AgroSync · Smart Agriculture Prediction System · Built with Streamlit + Scikit-learn + TensorFlow
</div>
""", unsafe_allow_html=True)
