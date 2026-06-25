"""
Smart Agriculture Predictor — Streamlit Dashboard
Single page: Crop Predictor only
New features: Soil Health Score, Season Filter, Budget Filter,
              Break-even Calculator, Crop Comparison Table,
              Input Warning System, Download Report
"""

import streamlit as st
import numpy as np
import joblib
import csv
import io
from pathlib import Path
import matplotlib.pyplot as plt

# ─────────────────────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Smart Agriculture Predictor",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# GLOBAL STYLING  (same EcoSync palette — unchanged)
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
html, body, [class*="css"] {
    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
}
.stApp { background-color: #0D1117; color: #E6EDF3; }

[data-testid="stSidebar"] {
    background-color: #161B22 !important;
    border-right: 1px solid #30363D;
}
[data-testid="stSidebar"] * { color: #E6EDF3 !important; }

.metric-card {
    background: #161B22;
    border: 1px solid #30363D;
    border-radius: 12px;
    padding: 20px 22px;
    margin-bottom: 12px;
}
.metric-card .label {
    font-size: 11px; text-transform: uppercase;
    letter-spacing: 0.08em; color: #8B949E; margin-bottom: 6px;
}
.metric-card .value { font-size: 28px; font-weight: 700; color: #E6EDF3; line-height: 1.1; }
.metric-card .sub   { font-size: 12px; color: #3FB950; margin-top: 4px; }

.crop-card {
    background: #161B22; border: 1px solid #238636;
    border-radius: 14px; padding: 22px 24px; margin-bottom: 14px;
}
.crop-card.rank1 { border-color: #3FB950; border-width: 2px; }
.crop-card .rank-badge {
    display: inline-block; background: #238636; color: #fff;
    font-size: 11px; font-weight: 700; border-radius: 20px;
    padding: 3px 10px; margin-bottom: 8px; letter-spacing: 0.05em;
}
.crop-card .crop-name { font-size: 22px; font-weight: 700; color: #E6EDF3; }
.crop-card .confidence-bar-wrap {
    background: #21262D; border-radius: 6px; height: 8px; margin: 10px 0 14px; overflow: hidden;
}
.crop-card .confidence-bar {
    height: 8px; border-radius: 6px;
    background: linear-gradient(90deg, #238636, #3FB950);
}
.crop-card .detail-grid {
    display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px;
}
.crop-card .detail-item .di-label {
    font-size: 10px; color: #8B949E;
    text-transform: uppercase; letter-spacing: 0.07em;
}
.crop-card .detail-item .di-val { font-size: 14px; font-weight: 600; color: #E6EDF3; margin-top: 2px; }
.di-val.green { color: #3FB950; }
.di-val.blue  { color: #58A6FF; }

.section-header {
    display: flex; align-items: center; gap: 10px;
    margin: 28px 0 16px; padding-bottom: 10px; border-bottom: 1px solid #30363D;
}
.section-header .sh-title { font-size: 16px; font-weight: 700; color: #E6EDF3; }
.section-header .sh-dot {
    width: 8px; height: 8px; border-radius: 50%; background: #3FB950; flex-shrink: 0;
}

.warn-box {
    background: #1C1C10; border: 1px solid #BB8009;
    border-radius: 10px; padding: 12px 16px;
    color: #D29922; font-size: 13px; margin-bottom: 14px;
}
.good-box {
    background: #0D2818; border: 1px solid #238636;
    border-radius: 10px; padding: 12px 16px;
    color: #3FB950; font-size: 13px; margin-bottom: 14px;
}

.hero {
    background: linear-gradient(135deg, #161B22 0%, #0D2818 100%);
    border: 1px solid #30363D; border-radius: 16px;
    padding: 28px 32px; margin-bottom: 24px;
}
.hero h1 { font-size: 26px; font-weight: 800; color: #E6EDF3; margin: 0 0 6px; }
.hero p  { font-size: 14px; color: #8B949E; margin: 0; }
.hero .pill {
    display: inline-block; background: #238636; color: #fff;
    border-radius: 20px; font-size: 11px; font-weight: 700;
    padding: 3px 12px; margin-right: 6px; margin-top: 12px;
}

/* health score bar */
.health-bar-wrap {
    background: #21262D; border-radius: 8px; height: 14px;
    margin: 10px 0 6px; overflow: hidden;
}
.health-bar { height: 14px; border-radius: 8px; transition: width .4s; }

/* comparison table */
.cmp-table { width:100%; border-collapse:collapse; font-size:13px; }
.cmp-table th {
    background:#21262D; color:#8B949E; font-size:10px;
    text-transform:uppercase; letter-spacing:.06em;
    padding:8px 12px; text-align:left; border-bottom:1px solid #30363D;
}
.cmp-table td { padding:10px 12px; border-bottom:1px solid #21262D; color:#E6EDF3; }
.cmp-table tr:last-child td { border-bottom:none; }
.cmp-table tr.best-row td { background:#0D2818; }
.tag-green {
    display:inline-block; background:#1C2B20; color:#3FB950;
    border:1px solid #238636; border-radius:5px;
    font-size:10px; padding:2px 7px;
}
.tag-blue {
    display:inline-block; background:#0D1F38; color:#58A6FF;
    border:1px solid #1F6FEB; border-radius:5px;
    font-size:10px; padding:2px 7px;
}

.stButton > button {
    background: linear-gradient(135deg, #238636, #1A6B2A) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 15px !important; padding: 14px 0 !important;
    width: 100%; transition: opacity .2s;
}
.stButton > button:hover { opacity: .85; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# CROP DATA
# ─────────────────────────────────────────────────────────────────────────────
CROP_ECONOMICS = {
    "rice":        {"cost": 35000, "profit": 25000, "market": "₹18–22/kg"},
    "maize":       {"cost": 28000, "profit": 20000, "market": "₹15–18/kg"},
    "chickpea":    {"cost": 22000, "profit": 30000, "market": "₹45–60/kg"},
    "kidneybeans": {"cost": 24000, "profit": 28000, "market": "₹60–80/kg"},
    "pigeonpeas":  {"cost": 20000, "profit": 26000, "market": "₹50–65/kg"},
    "mothbeans":   {"cost": 18000, "profit": 22000, "market": "₹40–55/kg"},
    "mungbean":    {"cost": 20000, "profit": 25000, "market": "₹55–70/kg"},
    "blackgram":   {"cost": 18000, "profit": 22000, "market": "₹45–60/kg"},
    "lentil":      {"cost": 22000, "profit": 28000, "market": "₹50–70/kg"},
    "pomegranate": {"cost": 80000, "profit": 120000, "market": "₹60–100/kg"},
    "banana":      {"cost": 50000, "profit": 60000,  "market": "₹15–25/kg"},
    "mango":       {"cost": 60000, "profit": 80000,  "market": "₹30–80/kg"},
    "grapes":      {"cost": 90000, "profit": 130000, "market": "₹40–120/kg"},
    "watermelon":  {"cost": 25000, "profit": 30000,  "market": "₹5–10/kg"},
    "muskmelon":   {"cost": 22000, "profit": 28000,  "market": "₹8–15/kg"},
    "apple":       {"cost": 100000,"profit": 150000, "market": "₹80–200/kg"},
    "orange":      {"cost": 60000, "profit": 80000,  "market": "₹20–40/kg"},
    "papaya":      {"cost": 40000, "profit": 55000,  "market": "₹10–20/kg"},
    "coconut":     {"cost": 45000, "profit": 70000,  "market": "₹15–30/piece"},
    "cotton":      {"cost": 35000, "profit": 30000,  "market": "₹55–65/kg"},
    "jute":        {"cost": 25000, "profit": 18000,  "market": "₹35–45/kg"},
    "coffee":      {"cost": 70000, "profit": 100000, "market": "₹150–300/kg"},
}

CROP_REQUIREMENTS = {
    "rice":        {"Water":"High",     "Soil":"Clayey",           "Season":"Kharif",        "Fertilizer":"Urea, DAP",         "Tools":"Transplanter, Harvester"},
    "maize":       {"Water":"Medium",   "Soil":"Loamy",            "Season":"Kharif/Rabi",   "Fertilizer":"NPK",               "Tools":"Planter, Sheller"},
    "chickpea":    {"Water":"Low",      "Soil":"Sandy-Loam",       "Season":"Rabi",          "Fertilizer":"SSP, Rhizobium",    "Tools":"Seed Drill"},
    "kidneybeans": {"Water":"Medium",   "Soil":"Well-drained",     "Season":"Kharif",        "Fertilizer":"NPK",               "Tools":"Planter"},
    "pigeonpeas":  {"Water":"Low",      "Soil":"Loamy",            "Season":"Kharif",        "Fertilizer":"DAP",               "Tools":"Seed Drill"},
    "mothbeans":   {"Water":"Very Low", "Soil":"Sandy",            "Season":"Kharif",        "Fertilizer":"Phosphorus",        "Tools":"Manual"},
    "mungbean":    {"Water":"Low-Med",  "Soil":"Loamy",            "Season":"Kharif",        "Fertilizer":"Rhizobium",         "Tools":"Seed Drill"},
    "blackgram":   {"Water":"Low",      "Soil":"Loamy",            "Season":"Kharif",        "Fertilizer":"SSP",               "Tools":"Seed Drill"},
    "lentil":      {"Water":"Low",      "Soil":"Loamy",            "Season":"Rabi",          "Fertilizer":"DAP",               "Tools":"Seed Drill"},
    "pomegranate": {"Water":"Low",      "Soil":"Well-drained",     "Season":"Year-round",    "Fertilizer":"FYM, NPK",          "Tools":"Pruning Shears"},
    "banana":      {"Water":"High",     "Soil":"Rich Loam",        "Season":"Year-round",    "Fertilizer":"Potassium-rich",    "Tools":"Irrigation System"},
    "mango":       {"Water":"Medium",   "Soil":"Deep Loam",        "Season":"Summer",        "Fertilizer":"NPK",               "Tools":"Sprayer"},
    "grapes":      {"Water":"Medium",   "Soil":"Well-drained",     "Season":"Rabi",          "Fertilizer":"Potassium",         "Tools":"Pruning Shears, Trellis"},
    "watermelon":  {"Water":"Medium",   "Soil":"Sandy",            "Season":"Summer",        "Fertilizer":"NPK",               "Tools":"Drip Irrigation"},
    "muskmelon":   {"Water":"Medium",   "Soil":"Sandy-Loam",       "Season":"Summer",        "Fertilizer":"NPK",               "Tools":"Drip Irrigation"},
    "apple":       {"Water":"Medium",   "Soil":"Loamy",            "Season":"Rabi",          "Fertilizer":"Calcium, Boron",    "Tools":"Sprayer, Harvester"},
    "orange":      {"Water":"Medium",   "Soil":"Sandy-Loam",       "Season":"Rabi",          "Fertilizer":"NPK",               "Tools":"Sprayer"},
    "papaya":      {"Water":"Medium",   "Soil":"Loamy",            "Season":"Year-round",    "Fertilizer":"Nitrogen-rich",     "Tools":"Irrigation System"},
    "coconut":     {"Water":"High",     "Soil":"Sandy-Loam",       "Season":"Year-round",    "Fertilizer":"NPK + Micronutrients","Tools":"Climbing Equipment"},
    "cotton":      {"Water":"Medium",   "Soil":"Black Cotton Soil","Season":"Kharif",        "Fertilizer":"DAP, Urea",         "Tools":"Cotton Picker"},
    "jute":        {"Water":"High",     "Soil":"Sandy-Loam",       "Season":"Kharif",        "Fertilizer":"NPK",               "Tools":"Retting Tanks"},
    "coffee":      {"Water":"Medium",   "Soil":"Well-drained",     "Season":"Year-round",    "Fertilizer":"NPK + Shade Trees", "Tools":"Pulper, Dryer"},
}

# yield per hectare (kg) — approximate, used for break-even
CROP_YIELD_KG = {
    "rice":900,"maize":2500,"chickpea":1200,"kidneybeans":800,"pigeonpeas":900,
    "mothbeans":600,"mungbean":800,"blackgram":700,"lentil":1000,
    "pomegranate":8000,"banana":20000,"mango":5000,"grapes":8000,
    "watermelon":15000,"muskmelon":12000,"apple":10000,"orange":8000,
    "papaya":25000,"coconut":6000,"cotton":1500,"jute":2000,"coffee":1200,
}

# ─────────────────────────────────────────────────────────────────────────────
# MODEL LOADING
# ─────────────────────────────────────────────────────────────────────────────
MODELS_DIR = Path("models")

@st.cache_resource(show_spinner=False)
def load_models():
    bundle = {}
    files = {
        "rf_cls":"rf_cls.pkl","lr_cls":"lr_cls.pkl","dt_cls":"dt_cls.pkl",
        "knn_cls":"knn_cls.pkl","svm_cls":"svm_cls.pkl",
        "scaler_cls":"scaler_cls.pkl","le_crop":"le_crop.pkl",
    }
    missing = []
    for key, fname in files.items():
        path = MODELS_DIR / fname
        if path.exists():
            bundle[key] = joblib.load(path)
        else:
            bundle[key] = None
            missing.append(fname)
    brain_path = Path("brain.pkl")
    if brain_path.exists():
        b = joblib.load(brain_path)
        if bundle["rf_cls"]     is None: bundle["rf_cls"]     = b.get("model")
        if bundle["scaler_cls"] is None: bundle["scaler_cls"] = b.get("scaler")
        if bundle["le_crop"]    is None: bundle["le_crop"]    = b.get("encoder")
    return bundle, missing

models, missing_files = load_models()

# ─────────────────────────────────────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────────────────────────────────────
def soil_health_score(n, p, k, ph):
    """Return score 0-100 and label."""
    score = 0
    # N optimal 60-80
    score += 25 if 60 <= n <= 80 else (15 if 40 <= n <= 100 else 5)
    # P optimal 30-60
    score += 25 if 30 <= p <= 60 else (15 if 15 <= p <= 90 else 5)
    # K optimal 40-80
    score += 25 if 40 <= k <= 80 else (15 if 20 <= k <= 120 else 5)
    # pH optimal 6.0-7.5
    score += 25 if 6.0 <= ph <= 7.5 else (15 if 5.5 <= ph <= 8.0 else 5)
    if score >= 85:
        return score, "Excellent", "#3FB950"
    elif score >= 65:
        return score, "Good", "#58A6FF"
    elif score >= 40:
        return score, "Average", "#D29922"
    else:
        return score, "Poor", "#F85149"

def input_warnings(n, p, k, temp, hum, ph, rain):
    """Return list of warning strings for out-of-range or unusual values."""
    warns = []
    if n < 10:  warns.append("⚠️ Nitrogen too low — crops may show yellowing. Add Urea or DAP.")
    if n > 120: warns.append("⚠️ Very high Nitrogen — risk of excessive leaf growth, fewer grains.")
    if p < 10:  warns.append("⚠️ Phosphorus too low — weak root system. Apply SSP or DAP.")
    if k < 10:  warns.append("⚠️ Potassium too low — poor disease resistance. Apply MOP.")
    if ph < 5.0: warns.append("⚠️ Soil too acidic (pH < 5) — add lime to correct.")
    if ph > 8.5: warns.append("⚠️ Soil too alkaline (pH > 8.5) — add gypsum or sulfur.")
    if temp > 40: warns.append("⚠️ Very high temperature — consider heat-tolerant varieties.")
    if hum < 20: warns.append("⚠️ Very low humidity — ensure adequate irrigation.")
    if rain < 30: warns.append("⚠️ Low rainfall value — verify this is dataset-scale (20–298 mm).")
    return warns

def predict_crops(n, p, k, temperature, humidity, ph, rainfall,
                  land_area_ha, model_name, top_n, budget, season_filter):
    scaler = models.get("scaler_cls")
    le     = models.get("le_crop")
    model_map = {
        "Random Forest":       ("rf_cls",  False),
        "Logistic Regression": ("lr_cls",  True),
        "Decision Tree":       ("dt_cls",  False),
        "KNN":                 ("knn_cls", True),
        "SVM":                 ("svm_cls", True),
    }
    key, needs_scale = model_map[model_name]
    mdl = models.get(key)
    if mdl is None or scaler is None or le is None:
        return []

    feat        = np.array([[n, p, k, temperature, humidity, ph, rainfall]])
    feat_scaled = scaler.transform(feat)
    inp         = feat_scaled if needs_scale else feat

    if hasattr(mdl, "predict_proba"):
        proba = mdl.predict_proba(inp)[0]
    else:
        pred  = mdl.predict(inp)[0]
        proba = np.zeros(len(le.classes_))
        proba[pred] = 1.0

    # get all crops sorted by confidence
    all_idx = np.argsort(proba)[::-1]
    recs = []
    for idx in all_idx:
        if len(recs) == top_n:
            break
        crop = le.classes_[idx]
        conf = proba[idx] * 100
        eco  = CROP_ECONOMICS.get(crop, {"cost":30000,"profit":25000,"market":"N/A"})
        req  = CROP_REQUIREMENTS.get(crop, {})
        tc   = eco["cost"]   * land_area_ha
        tp   = eco["profit"] * land_area_ha
        roi  = (tp / tc * 100) if tc > 0 else 0

        # ── FEATURE 2: Season filter ──
        if season_filter != "All Seasons":
            crop_season = req.get("Season", "")
            if season_filter not in crop_season and "Year-round" not in crop_season:
                continue

        # ── FEATURE 3: Budget filter ──
        if budget > 0 and tc > budget:
            continue

        # ── FEATURE 4: Break-even ──
        yield_kg    = CROP_YIELD_KG.get(crop, 1000) * land_area_ha
        mkt_str     = eco["market"]
        try:
            mkt_low = float(mkt_str.replace("₹","").split("–")[0].replace("/kg","").replace("/piece","").strip())
        except:
            mkt_low = 10.0
        breakeven_kg = tc / mkt_low if mkt_low > 0 else 0
        breakeven_pct = (breakeven_kg / yield_kg * 100) if yield_kg > 0 else 0

        recs.append({
            "crop":          crop.title(),
            "crop_key":      crop,
            "confidence":    round(conf, 1),
            "cost_ha":       eco["cost"],
            "profit_ha":     eco["profit"],
            "total_cost":    tc,
            "total_profit":  tp,
            "roi":           round(roi, 1),
            "market":        mkt_str,
            "water":         req.get("Water",      "N/A"),
            "soil":          req.get("Soil",       "N/A"),
            "season":        req.get("Season",     "N/A"),
            "fertilizer":    req.get("Fertilizer", "N/A"),
            "tools":         req.get("Tools",      "N/A"),
            "breakeven_kg":  round(breakeven_kg, 1),
            "breakeven_pct": round(breakeven_pct, 1),
            "yield_kg":      round(yield_kg, 1),
        })
    return recs

def make_report(recs, n, p, k, temp, hum, ph, rain, land, model_name, health_score, health_label):
    """Generate a CSV string for download."""
    output = io.StringIO()
    w = csv.writer(output)
    w.writerow(["=== Smart Agriculture Predictor Report ==="])
    w.writerow([])
    w.writerow(["INPUT PARAMETERS"])
    w.writerow(["Nitrogen (N)", n, "kg/ha"])
    w.writerow(["Phosphorus (P)", p, "kg/ha"])
    w.writerow(["Potassium (K)", k, "kg/ha"])
    w.writerow(["Temperature", temp, "°C"])
    w.writerow(["Humidity", hum, "%"])
    w.writerow(["Soil pH", ph])
    w.writerow(["Rainfall", rain, "mm"])
    w.writerow(["Land Area", land, "hectares"])
    w.writerow(["Model Used", model_name])
    w.writerow(["Soil Health Score", f"{health_score}/100", health_label])
    w.writerow([])
    w.writerow(["RECOMMENDATIONS"])
    w.writerow(["Rank","Crop","Confidence %","Total Cost (₹)","Total Profit (₹)",
                "ROI %","Market Price","Water Needs","Soil Type","Best Season",
                "Fertilizer","Break-even (kg)"])
    for i, r in enumerate(recs, 1):
        w.writerow([i, r["crop"], r["confidence"], r["total_cost"], r["total_profit"],
                    r["roi"], r["market"], r["water"], r["soil"], r["season"],
                    r["fertilizer"], r["breakeven_kg"]])
    return output.getvalue()


# ─────────────────────────────────────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style='padding:10px 0 20px;'>
      <div style='font-size:20px;font-weight:800;color:#E6EDF3;'>🌾 Smart Agri</div>
      <div style='font-size:11px;color:#8B949E;margin-top:2px;'>AI-Powered Crop Advisor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("**🔬 Soil Nutrients**")
    n_val   = st.slider("Nitrogen (N) kg/ha",   0,   140, 60,  step=1)
    p_val   = st.slider("Phosphorus (P) kg/ha", 5,   145, 40,  step=1)
    k_val   = st.slider("Potassium (K) kg/ha",  5,   205, 60,  step=1)

    st.markdown("---")
    st.markdown("**🌡️ Climate & Soil Conditions**")
    temp_val = st.slider("Temperature (°C)", 8,   44,  26,  step=1)
    hum_val  = st.slider("Humidity (%)",     14,  100, 65,  step=1)
    ph_val   = st.slider("Soil pH",          3.5, 9.9, 6.5, step=0.1)
    rain_val = st.slider("Rainfall (mm)",    20,  298, 140, step=2,
                         help="Dataset scale — divide annual rainfall by ~4")

    st.markdown("---")
    st.markdown("**⚙️ Prediction Settings**")
    land_val     = st.number_input("Land Area (hectares)", min_value=0.5,
                                   max_value=100.0, value=2.0, step=0.5)
    model_choice = st.selectbox("ML Model",
                                ["Random Forest","Logistic Regression",
                                 "Decision Tree","KNN","SVM"])
    top_n_val    = st.radio("Top Recommendations", [1, 2, 3], index=2, horizontal=True)

    st.markdown("---")
    st.markdown("**🗓️ Season Filter**")
    season_val = st.selectbox("Current / Target Season",
                              ["All Seasons", "Kharif", "Rabi", "Summer", "Year-round"])

    st.markdown("**💰 Budget Filter**")
    budget_val = st.number_input("Max Budget (₹) — 0 = no limit",
                                 min_value=0, max_value=10000000,
                                 value=0, step=5000,
                                 help="Total budget for your land area. Crops above this cost are removed.")

    st.markdown("---")
    predict_btn = st.button("🔍  Analyse & Predict", use_container_width=True)


# ─────────────────────────────────────────────────────────────────────────────
# HERO
# ─────────────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="hero">
  <div style="display:flex;align-items:flex-start;justify-content:space-between;flex-wrap:wrap;gap:16px;">
    <div>
      <h1>🌾 Smart Agriculture Dashboard</h1>
      <p>Enter soil &amp; climate data on the left → get AI-powered crop recommendations with economics.</p>
      <span class="pill">5 ML Models</span>
      <span class="pill">22 Crops</span>
      <span class="pill">ROI Calculator</span>
      <span class="pill">Break-even</span>
      <span class="pill">Season Filter</span>
    </div>
    <div style="text-align:right;">
      <div style="font-size:11px;color:#8B949E;">Active Model</div>
      <div style="font-size:22px;font-weight:700;color:#3FB950;">{model_choice}</div>
      <div style="font-size:11px;color:#8B949E;margin-top:4px;">Area: {land_val} ha</div>
    </div>
  </div>
</div>
""", unsafe_allow_html=True)

# missing models warning
if missing_files:
    needed = [f for f in missing_files if "cls" in f or "le_crop" in f or "scaler_cls" in f]
    if needed:
        st.markdown(f"""
        <div class="warn-box">
          ⚠️ <strong>Models not found:</strong> {', '.join(needed)}<br>
          Run the Jupyter notebook first to generate the <code>models/</code> folder,
          then place it next to <code>app.py</code>.
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 1 — SOIL HEALTH SCORE  (always visible, live)
# ─────────────────────────────────────────────────────────────────────────────
health_score, health_label, health_color = soil_health_score(n_val, p_val, k_val, ph_val)

st.markdown("""
<div class="section-header">
  <div class="sh-dot"></div>
  <div class="sh-title">Soil Health Score</div>
</div>
""", unsafe_allow_html=True)

col_h1, col_h2 = st.columns([1, 3])
with col_h1:
    st.markdown(f"""
    <div class="metric-card" style="text-align:center;">
      <div class="label">Health Score</div>
      <div class="value" style="font-size:42px;color:{health_color};">{health_score}</div>
      <div style="font-size:14px;font-weight:700;color:{health_color};">{health_label}</div>
      <div class="sub">out of 100</div>
    </div>
    """, unsafe_allow_html=True)

with col_h2:
    st.markdown(f"""
    <div class="metric-card">
      <div class="label">Score Breakdown</div>
      <div style="margin-top:10px;">
        <div style="display:flex;justify-content:space-between;font-size:12px;color:#8B949E;margin-bottom:4px;">
          <span>Overall Health</span><span style="color:{health_color};font-weight:700;">{health_score}/100 — {health_label}</span>
        </div>
        <div class="health-bar-wrap">
          <div class="health-bar" style="width:{health_score}%;background:{health_color};"></div>
        </div>
        <div style="display:grid;grid-template-columns:repeat(4,1fr);gap:10px;margin-top:14px;">
          <div>
            <div style="font-size:10px;color:#8B949E;text-transform:uppercase;">Nitrogen</div>
            <div style="font-size:15px;font-weight:700;color:#E6EDF3;">{n_val} <span style="font-size:10px;color:#8B949E;">kg/ha</span></div>
            <div style="font-size:10px;color:{'#3FB950' if 60<=n_val<=80 else '#D29922'};">{'✅ Optimal' if 60<=n_val<=80 else '⚠️ Adjust'}</div>
          </div>
          <div>
            <div style="font-size:10px;color:#8B949E;text-transform:uppercase;">Phosphorus</div>
            <div style="font-size:15px;font-weight:700;color:#E6EDF3;">{p_val} <span style="font-size:10px;color:#8B949E;">kg/ha</span></div>
            <div style="font-size:10px;color:{'#3FB950' if 30<=p_val<=60 else '#D29922'};">{'✅ Optimal' if 30<=p_val<=60 else '⚠️ Adjust'}</div>
          </div>
          <div>
            <div style="font-size:10px;color:#8B949E;text-transform:uppercase;">Potassium</div>
            <div style="font-size:15px;font-weight:700;color:#E6EDF3;">{k_val} <span style="font-size:10px;color:#8B949E;">kg/ha</span></div>
            <div style="font-size:10px;color:{'#3FB950' if 40<=k_val<=80 else '#D29922'};">{'✅ Optimal' if 40<=k_val<=80 else '⚠️ Adjust'}</div>
          </div>
          <div>
            <div style="font-size:10px;color:#8B949E;text-transform:uppercase;">Soil pH</div>
            <div style="font-size:15px;font-weight:700;color:#E6EDF3;">{ph_val}</div>
            <div style="font-size:10px;color:{'#3FB950' if 6.0<=ph_val<=7.5 else '#D29922'};">{'✅ Optimal' if 6.0<=ph_val<=7.5 else '⚠️ Adjust'}</div>
          </div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# FEATURE 6 — INPUT WARNINGS  (always visible, live)
# ─────────────────────────────────────────────────────────────────────────────
warns = input_warnings(n_val, p_val, k_val, temp_val, hum_val, ph_val, rain_val)
if warns:
    for w in warns:
        st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)
else:
    st.markdown('<div class="good-box">✅ All input values are within healthy ranges. Good to predict!</div>',
                unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# INPUT SUMMARY CARDS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-header">
  <div class="sh-dot"></div>
  <div class="sh-title">Current Input Summary</div>
</div>
""", unsafe_allow_html=True)

cols = st.columns(7)
params = [
    ("N",    n_val,    "kg/ha"),
    ("P",    p_val,    "kg/ha"),
    ("K",    k_val,    "kg/ha"),
    ("Temp", temp_val, "°C"),
    ("Hum",  hum_val,  "%"),
    ("pH",   ph_val,   ""),
    ("Rain", rain_val, "mm"),
]
for col, (label, val, unit) in zip(cols, params):
    with col:
        st.markdown(f"""
        <div class="metric-card">
          <div class="label">{label}</div>
          <div class="value">{val}</div>
          <div class="sub">{unit}</div>
        </div>
        """, unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────────────────────
# PREDICTION RESULTS
# ─────────────────────────────────────────────────────────────────────────────
if predict_btn:
    with st.spinner("🌿 Analysing soil & climate data..."):
        recs = predict_crops(
            n_val, p_val, k_val, temp_val, hum_val,
            ph_val, rain_val, land_val, model_choice,
            top_n_val, budget_val, season_val
        )

    if not recs:
        st.markdown("""
        <div class="warn-box">
          ❌ No crops matched your filters (season / budget). Try relaxing your filters or selecting "All Seasons".
        </div>
        """, unsafe_allow_html=True)
    else:
        # ── Recommendation cards ─────────────────────────────────────────────
        st.markdown("""
        <div class="section-header">
          <div class="sh-dot"></div>
          <div class="sh-title">Top Crop Recommendations</div>
        </div>
        """, unsafe_allow_html=True)

        for i, r in enumerate(recs, 1):
            rank_cls = "rank1" if i == 1 else ""
            badge    = "🥇 Best Match" if i == 1 else f"#{i} Recommendation"
            st.markdown(f"""
            <div class="crop-card {rank_cls}">
              <div class="rank-badge">{badge}</div>
              <div class="crop-name">{r['crop']}</div>
              <div style="font-size:12px;color:#8B949E;margin:4px 0;">
                Confidence: <span style="color:#3FB950;font-weight:700;">{r['confidence']}%</span>
                &nbsp;·&nbsp; Season: <span style="color:#58A6FF;">{r['season']}</span>
              </div>
              <div class="confidence-bar-wrap">
                <div class="confidence-bar" style="width:{min(r['confidence'],100)}%;"></div>
              </div>
              <div class="detail-grid">
                <div class="detail-item">
                  <div class="di-label">Total Cost</div>
                  <div class="di-val">₹{r['total_cost']:,.0f}</div>
                </div>
                <div class="detail-item">
                  <div class="di-label">Total Profit</div>
                  <div class="di-val green">₹{r['total_profit']:,.0f}</div>
                </div>
                <div class="detail-item">
                  <div class="di-label">ROI</div>
                  <div class="di-val blue">{r['roi']}%</div>
                </div>
                <div class="detail-item">
                  <div class="di-label">Market Price</div>
                  <div class="di-val">{r['market']}</div>
                </div>
                <div class="detail-item">
                  <div class="di-label">Water Needs</div>
                  <div class="di-val">{r['water']}</div>
                </div>
                <div class="detail-item">
                  <div class="di-label">Soil Type</div>
                  <div class="di-val">{r['soil']}</div>
                </div>
                <div class="detail-item">
                  <div class="di-label">Fertilizer</div>
                  <div class="di-val">{r['fertilizer']}</div>
                </div>
                <div class="detail-item">
                  <div class="di-label">Tools Needed</div>
                  <div class="di-val">{r['tools']}</div>
                </div>
                <div class="detail-item">
                  <div class="di-label">Break-even Qty</div>
                  <div class="di-val">{r['breakeven_kg']:,} kg</div>
                </div>
              </div>
            </div>
            """, unsafe_allow_html=True)

        # ── FEATURE 4: Break-even chart ──────────────────────────────────────
        st.markdown("""
        <div class="section-header">
          <div class="sh-dot"></div>
          <div class="sh-title">Break-even Analysis</div>
        </div>
        """, unsafe_allow_html=True)

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_facecolor("#0D1117")

        names   = [r["crop"] for r in recs]
        rois    = [r["roi"]  for r in recs]
        bk_pcts = [r["breakeven_pct"] for r in recs]

        for ax in axes:
            ax.set_facecolor("#161B22")
            ax.tick_params(colors="#E6EDF3", labelsize=10)
            ax.spines[:].set_color("#30363D")

        # ROI bar
        colors_roi = ["#3FB950" if i == 0 else "#238636" for i in range(len(recs))]
        axes[0].barh(names, rois, color=colors_roi, height=0.5)
        axes[0].set_title("ROI Comparison (%)", color="#E6EDF3", fontsize=12, pad=10)
        axes[0].set_xlabel("ROI (%)", color="#8B949E")
        for i, (val, name) in enumerate(zip(rois, names)):
            axes[0].text(val + 0.5, i, f"{val:.1f}%", va="center",
                         color="#E6EDF3", fontsize=10, fontweight="bold")
        axes[0].set_xlim(0, max(rois) * 1.3 if rois else 100)

        # Break-even % of yield
        colors_bk = ["#58A6FF" if i == 0 else "#1F6FEB" for i in range(len(recs))]
        axes[1].barh(names, bk_pcts, color=colors_bk, height=0.5)
        axes[1].set_title("Break-even (% of Expected Yield)", color="#E6EDF3", fontsize=12, pad=10)
        axes[1].set_xlabel("% of yield needed to break even", color="#8B949E")
        for i, (val, name) in enumerate(zip(bk_pcts, names)):
            axes[1].text(val + 0.5, i, f"{val:.1f}%", va="center",
                         color="#E6EDF3", fontsize=10, fontweight="bold")
        axes[1].set_xlim(0, max(bk_pcts) * 1.3 if bk_pcts else 100)

        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close()

        # ── FEATURE 5: Crop comparison table ────────────────────────────────
        if len(recs) > 1:
            st.markdown("""
            <div class="section-header">
              <div class="sh-dot"></div>
              <div class="sh-title">Side-by-Side Comparison</div>
            </div>
            """, unsafe_allow_html=True)

            header = ["Crop","Confidence","Total Cost","Total Profit",
                      "ROI","Market Price","Water","Season","Break-even (kg)"]
            rows_html = ""
            for i, r in enumerate(recs):
                best_cls = "best-row" if i == 0 else ""
                best_tag = '<span class="tag-green">★ Best</span>' if i == 0 else ""
                rows_html += f"""
                <tr class="{best_cls}">
                  <td>{r['crop']} {best_tag}</td>
                  <td><span class="tag-green">{r['confidence']}%</span></td>
                  <td>₹{r['total_cost']:,.0f}</td>
                  <td style="color:#3FB950;">₹{r['total_profit']:,.0f}</td>
                  <td><span class="tag-blue">{r['roi']}%</span></td>
                  <td>{r['market']}</td>
                  <td>{r['water']}</td>
                  <td>{r['season']}</td>
                  <td>{r['breakeven_kg']:,} kg</td>
                </tr>"""

            th_html = "".join(f"<th>{h}</th>" for h in header)
            st.markdown(f"""
            <div class="metric-card" style="overflow-x:auto;padding:16px;">
              <table class="cmp-table">
                <thead><tr>{th_html}</tr></thead>
                <tbody>{rows_html}</tbody>
              </table>
            </div>
            """, unsafe_allow_html=True)

        # ── FEATURE 7: Download report ───────────────────────────────────────
        st.markdown("""
        <div class="section-header">
          <div class="sh-dot"></div>
          <div class="sh-title">Download Report</div>
        </div>
        """, unsafe_allow_html=True)

        csv_data = make_report(
            recs, n_val, p_val, k_val, temp_val, hum_val,
            ph_val, rain_val, land_val, model_choice, health_score, health_label
        )
        st.download_button(
            label="⬇️  Download Prediction Report (CSV)",
            data=csv_data,
            file_name="agrosync_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

else:
    st.markdown("""
    <div style="text-align:center;padding:60px 20px;color:#8B949E;">
      <div style="font-size:48px;">🌱</div>
      <div style="font-size:16px;margin-top:12px;">
        Adjust sliders on the left, then click
        <strong style="color:#3FB950;">Analyse &amp; Predict</strong>
      </div>
    </div>
    """, unsafe_allow_html=True)
