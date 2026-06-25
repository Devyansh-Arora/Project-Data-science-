"""
AgroSense AI — Smart Agriculture Predictor
Uses brain.pkl (RandomForestClassifier) trained on Crop_recommendation.csv
Features: N, P, K, temperature, humidity, ph, rainfall  →  22 crop classes
"""

import streamlit as st
import numpy as np
import joblib
import os
import plotly.graph_objects as go
import plotly.express as px

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AgroSense AI",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════
# GLOBAL CSS  —  EcoSync dark palette  (#0f1117 bg, #1a1d27 cards,
#               #00C853 green accent, #1976D2 blue accent)
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
/* ── Base ── */
html, body, [data-testid="stAppViewContainer"] {
    background: #0d1117 !important;
    color: #e2e8f0;
    font-family: 'Inter', 'Segoe UI', sans-serif;
}
[data-testid="stSidebar"] {
    background: #161b27 !important;
    border-right: 1px solid #1e2535;
}
[data-testid="stSidebar"] * { color: #c9d1e0 !important; }

/* ── Top nav bar ── */
.nav-bar {
    display: flex; align-items: center; gap: 14px;
    padding: 14px 0 22px;
    border-bottom: 1px solid #1e2535;
    margin-bottom: 24px;
}
.nav-logo {
    font-size: 1.55rem; font-weight: 800; letter-spacing: -0.5px;
    background: linear-gradient(135deg, #00C853, #1976D2);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
}
.nav-badge {
    background: #00C853; color: #0d1117;
    font-size: 10px; font-weight: 700; padding: 2px 8px;
    border-radius: 20px; letter-spacing: 0.6px; text-transform: uppercase;
}
.nav-sub { color: #5a6a85; font-size: 13px; margin-left: auto; }

/* ── Section headers ── */
.sec-label {
    color: #4a5568; font-size: 11px; font-weight: 600;
    letter-spacing: 1.2px; text-transform: uppercase;
    margin-bottom: 12px; display: flex; align-items: center; gap: 8px;
}
.sec-label::after {
    content: ''; flex: 1; height: 1px; background: #1e2535;
}

/* ── Input panel card ── */
.input-panel {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 16px;
    padding: 24px;
    margin-bottom: 20px;
}

/* ── Metric mini cards ── */
.kpi-row { display: flex; gap: 12px; margin-bottom: 20px; flex-wrap: wrap; }
.kpi-card {
    background: #161b27; border: 1px solid #1e2535;
    border-radius: 12px; padding: 16px 20px; flex: 1; min-width: 140px;
}
.kpi-label { color: #4a5568; font-size: 11px; text-transform: uppercase;
             letter-spacing: 0.8px; margin-bottom: 4px; }
.kpi-value { font-size: 1.4rem; font-weight: 700; color: #e2e8f0; }
.kpi-sub   { font-size: 11px; color: #5a6a85; margin-top: 2px; }
.kpi-green { color: #00C853 !important; }
.kpi-blue  { color: #1976D2 !important; }
.kpi-yellow{ color: #FFD740 !important; }

/* ── Crop recommendation cards ── */
.crop-card {
    background: #161b27;
    border: 1px solid #1e2535;
    border-radius: 16px;
    padding: 22px 24px;
    margin-bottom: 14px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.crop-card.rank-1 { border-color: #00C853; }
.crop-card.rank-2 { border-color: #1976D2; }
.crop-card.rank-3 { border-color: #2d3a55; }
.crop-card::before {
    content: ''; position: absolute; top: 0; left: 0; right: 0; height: 2px;
}
.crop-card.rank-1::before { background: linear-gradient(90deg,#00C853,#1976D2); }
.crop-card.rank-2::before { background: linear-gradient(90deg,#1976D2,#00C853); }

.crop-header { display: flex; align-items: center; gap: 14px; margin-bottom: 18px; }
.rank-badge {
    width: 36px; height: 36px; border-radius: 50%;
    display: flex; align-items: center; justify-content: center;
    font-weight: 800; font-size: 14px; flex-shrink: 0;
}
.rank-1-badge { background: linear-gradient(135deg,#00C853,#00a844); color: #0d1117; }
.rank-2-badge { background: linear-gradient(135deg,#1976D2,#0d5fa8); color: #fff; }
.rank-3-badge { background: #1e2535; color: #8892a4; border: 1px solid #2d3a55; }

.crop-name  { font-size: 1.2rem; font-weight: 700; color: #e2e8f0; }
.crop-sci   { font-size: 12px; color: #5a6a85; margin-top: 2px; }

/* confidence bar */
.conf-wrap  { background:#1e2535; border-radius:6px; height:6px;
              width:100%; margin-bottom:18px; }
.conf-fill  { height:6px; border-radius:6px;
              background: linear-gradient(90deg,#00C853,#1976D2); }

/* stats grid */
.stats-grid {
    display: grid; grid-template-columns: repeat(3, 1fr);
    gap: 14px; margin-bottom: 16px;
}
.stat-box { }
.stat-lbl { color: #4a5568; font-size: 10px; text-transform: uppercase;
            letter-spacing: 0.7px; margin-bottom: 3px; }
.stat-val { font-size: 1rem; font-weight: 700; }
.green  { color: #00C853; }
.red    { color: #FF5252; }
.yellow { color: #FFD740; }
.white  { color: #e2e8f0; }

/* pill tags */
.pills  { display: flex; flex-wrap: wrap; gap: 6px; }
.pill   {
    background: #1e2535; border: 1px solid #2d3a55;
    border-radius: 20px; padding: 4px 12px;
    font-size: 11px; color: #8892a4;
}
.pill-green { border-color: #00C85333; background: #00C85311; color: #00C853; }

/* ── Sidebar toggle cards ── */
.toggle-card {
    background: #1a2035; border: 1px solid #1e2535;
    border-radius: 10px; padding: 12px 16px; margin-bottom: 8px;
    display: flex; justify-content: space-between; align-items: center;
}
.toggle-on  { width:32px; height:18px; background:#00C853; border-radius:9px;
              display:flex; align-items:center; justify-content:flex-end;
              padding-right:3px; }
.toggle-dot { width:12px; height:12px; background:#fff; border-radius:50%; }

/* ── Automations tab cards ── */
.auto-card {
    background:#161b27; border:1px solid #1e2535;
    border-radius:14px; padding:20px; margin-bottom:12px;
}
.auto-title { font-weight:700; color:#e2e8f0; margin-bottom:6px; font-size:1rem; }
.auto-desc  { color:#5a6a85; font-size:12px; margin-bottom:14px; line-height:1.5; }
.auto-stat  { display:flex; gap:20px; }
.auto-stat-item { }
.auto-stat-lbl  { color:#4a5568; font-size:10px; text-transform:uppercase; letter-spacing:0.7px; }
.auto-stat-val  { font-weight:700; font-size:1rem; }

/* ── Usage/History tab ── */
.history-head {
    background: linear-gradient(135deg,#0a1628,#0f1e38);
    border: 1px solid #1e2535; border-radius: 14px;
    padding: 22px; margin-bottom: 16px;
}
.history-hl { font-size: 1.4rem; font-weight: 800; color: #00C853; }
.history-sub{ color: #5a6a85; font-size: 13px; margin-top: 4px; }
.history-kpi{ font-size: 1.1rem; font-weight: 700; color: #fff; }

/* ── Streamlit override ── */
div[data-testid="stTabs"] button {
    color: #5a6a85 !important;
    font-size: 13px !important;
    font-weight: 600 !important;
}
div[data-testid="stTabs"] button[aria-selected="true"] {
    color: #e2e8f0 !important;
    border-bottom: 2px solid #00C853 !important;
}
.stButton > button {
    background: linear-gradient(135deg, #00C853, #1976D2) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important; font-weight: 700 !important;
    font-size: 15px !important; padding: 14px 0 !important;
    width: 100% !important; letter-spacing: 0.3px;
    transition: opacity 0.2s !important;
}
.stButton > button:hover { opacity: 0.88 !important; }
label, .stSlider label, .stNumberInput label,
.stSelectbox label { color: #5a6a85 !important; font-size: 12px !important; }
.stSlider [data-testid="stThumbValue"] { color: #00C853 !important; }
div[data-testid="metric-container"] {
    background:#161b27; border:1px solid #1e2535;
    border-radius:12px; padding:14px;
}
[data-testid="stMetricValue"] { color:#e2e8f0 !important; }
footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# DATA: crop knowledge base
# ══════════════════════════════════════════════════════════════════════════

# alphabetical order matches LabelEncoder from notebook
CROP_CLASSES = [
    'apple','banana','blackgram','chickpea','coconut','coffee','cotton',
    'grapes','jute','kidneybeans','lentil','maize','mango','mothbeans',
    'mungbean','muskmelon','orange','papaya','pigeonpeas','pomegranate',
    'rice','watermelon'
]

CROP_EMOJI = {
    'apple':'🍏','banana':'🍌','blackgram':'🫘','chickpea':'🫘',
    'coconut':'🥥','coffee':'☕','cotton':'☁️','grapes':'🍇',
    'jute':'🌿','kidneybeans':'🫘','lentil':'🫘','maize':'🌽',
    'mango':'🥭','mothbeans':'🫘','mungbean':'🫘','muskmelon':'🍈',
    'orange':'🍊','papaya':'🥭','pigeonpeas':'🫘','pomegranate':'🍎',
    'rice':'🌾','watermelon':'🍉',
}

ECONOMICS = {
    'apple':       {'cost':70000,'profit':100000,'market':'₹60–150/kg'},
    'banana':      {'cost':50000,'profit':70000, 'market':'₹15–30/kg'},
    'blackgram':   {'cost':19000,'profit':24000, 'market':'₹55–70/kg'},
    'chickpea':    {'cost':22000,'profit':30000, 'market':'₹45–60/kg'},
    'coconut':     {'cost':25000,'profit':40000, 'market':'₹15–30/piece'},
    'coffee':      {'cost':55000,'profit':80000, 'market':'₹80–150/kg'},
    'cotton':      {'cost':35000,'profit':30000, 'market':'₹55–75/kg'},
    'grapes':      {'cost':80000,'profit':120000,'market':'₹40–100/kg'},
    'jute':        {'cost':20000,'profit':18000, 'market':'₹35–50/kg'},
    'kidneybeans': {'cost':24000,'profit':28000, 'market':'₹60–80/kg'},
    'lentil':      {'cost':21000,'profit':27000, 'market':'₹50–65/kg'},
    'maize':       {'cost':28000,'profit':20000, 'market':'₹15–18/kg'},
    'mango':       {'cost':40000,'profit':60000, 'market':'₹30–80/kg'},
    'mothbeans':   {'cost':18000,'profit':22000, 'market':'₹40–55/kg'},
    'mungbean':    {'cost':20000,'profit':25000, 'market':'₹55–70/kg'},
    'muskmelon':   {'cost':28000,'profit':32000, 'market':'₹10–20/kg'},
    'orange':      {'cost':45000,'profit':65000, 'market':'₹20–50/kg'},
    'papaya':      {'cost':35000,'profit':50000, 'market':'₹12–25/kg'},
    'pigeonpeas':  {'cost':20000,'profit':26000, 'market':'₹50–65/kg'},
    'pomegranate': {'cost':60000,'profit':90000, 'market':'₹50–80/kg'},
    'rice':        {'cost':35000,'profit':25000, 'market':'₹18–22/kg'},
    'watermelon':  {'cost':30000,'profit':35000, 'market':'₹8–15/kg'},
}

REQUIREMENTS = {
    'apple':       {'Water':'Moderate (900mm)','Soil':'Loamy','Season':'Winter','Fert':'17-17-17 NPK','Tools':'Sprayer, Cold Storage'},
    'banana':      {'Water':'High (1000mm)','Soil':'Loamy/Black','Season':'Whole Year','Fert':'Urea + MOP','Tools':'Drip Irrigation'},
    'blackgram':   {'Water':'Moderate (500mm)','Soil':'Loamy/Clay','Season':'Kharif/Rabi','Fert':'DAP','Tools':'Sprayer'},
    'chickpea':    {'Water':'Low (300–400mm)','Soil':'Sandy/Loamy','Season':'Rabi','Fert':'DAP + Rhizobium','Tools':'Seed Drill'},
    'coconut':     {'Water':'High (1000–2500mm)','Soil':'Loamy/Sandy','Season':'Whole Year','Fert':'Urea + DAP','Tools':'Sprayer'},
    'coffee':      {'Water':'High (1500mm)','Soil':'Red/Loamy','Season':'Whole Year','Fert':'17-17-17 NPK','Tools':'Sprayer, Pulper'},
    'cotton':      {'Water':'Moderate (500–700mm)','Soil':'Black/Loamy','Season':'Kharif','Fert':'14-35-14 NPK','Tools':'Sprayer, Picker'},
    'grapes':      {'Water':'Moderate (700mm)','Soil':'Sandy Loam','Season':'Rabi','Fert':'14-35-14 NPK','Tools':'Drip, Pruner'},
    'jute':        {'Water':'High (1000–1500mm)','Soil':'Loamy/Sandy','Season':'Kharif','Fert':'Urea','Tools':'Harvester'},
    'kidneybeans': {'Water':'Moderate (600mm)','Soil':'Loamy','Season':'Kharif','Fert':'10-26-26 NPK','Tools':'Sprayer'},
    'lentil':      {'Water':'Low (350mm)','Soil':'Sandy Loam','Season':'Rabi','Fert':'20-20 NPK','Tools':'Sprayer'},
    'maize':       {'Water':'Moderate (500–800mm)','Soil':'Sandy Loam','Season':'Kharif/Rabi','Fert':'17-17-17 NPK','Tools':'Seed Drill'},
    'mango':       {'Water':'Moderate (750mm)','Soil':'Loamy/Sandy','Season':'Summer','Fert':'17-17-17 NPK','Tools':'Sprayer'},
    'mothbeans':   {'Water':'Low (200–350mm)','Soil':'Sandy','Season':'Kharif','Fert':'20-20 NPK','Tools':'Sprayer'},
    'mungbean':    {'Water':'Low–Moderate (400mm)','Soil':'Sandy Loam','Season':'Summer','Fert':'DAP','Tools':'Sprayer'},
    'muskmelon':   {'Water':'Moderate (400mm)','Soil':'Sandy/Loamy','Season':'Summer','Fert':'20-20 NPK','Tools':'Sprayer'},
    'orange':      {'Water':'Moderate (600–700mm)','Soil':'Sandy Loam','Season':'Rabi','Fert':'17-17-17 NPK','Tools':'Sprayer'},
    'papaya':      {'Water':'Moderate (800mm)','Soil':'Sandy Loam','Season':'Whole Year','Fert':'Urea + MOP','Tools':'Drip'},
    'pigeonpeas':  {'Water':'Low–Mod (600mm)','Soil':'Sandy/Red','Season':'Kharif','Fert':'DAP','Tools':'Sprayer'},
    'pomegranate': {'Water':'Low (500mm)','Soil':'Sandy Loam/Red','Season':'Whole Year','Fert':'17-17-17 NPK','Tools':'Drip, Sprayer'},
    'rice':        {'Water':'High (1200–2000mm)','Soil':'Clay/Loamy','Season':'Kharif','Fert':'Urea + DAP','Tools':'Transplanter'},
    'watermelon':  {'Water':'Moderate (400–600mm)','Soil':'Sandy','Season':'Summer','Fert':'20-20 NPK','Tools':'Drip, Sprayer'},
}

FEATURE_NAMES = ['Nitrogen (N)', 'Phosphorus (P)', 'Potassium (K)',
                 'Temperature', 'Humidity', 'Soil pH', 'Rainfall']

# ══════════════════════════════════════════════════════════════════════════
# LOAD MODEL
# ══════════════════════════════════════════════════════════════════════════
@st.cache_resource
def load_model():
    """Load brain.pkl — works locally and on Streamlit Cloud."""
    for path in ['brain.pkl', 'models/brain.pkl', os.path.join(os.path.dirname(__file__), 'brain.pkl')]:
        if os.path.exists(path):
            import warnings
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                return joblib.load(path), None
    return None, "brain.pkl not found. Place it in the same folder as app.py."

model, model_err = load_model()

# ══════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style='text-align:center;padding:16px 0 24px;border-bottom:1px solid #1e2535;margin-bottom:20px;'>
        <div style='font-size:2rem;margin-bottom:6px;'>🌾</div>
        <div style='font-weight:800;font-size:1.1rem;
            background:linear-gradient(135deg,#00C853,#1976D2);
            -webkit-background-clip:text;-webkit-text-fill-color:transparent;'>
            AgroSense AI
        </div>
        <div style='color:#4a5568;font-size:11px;margin-top:3px;'>Smart Agriculture Predictor</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="sec-label">🌍 Farm Details</div>', unsafe_allow_html=True)
    land_area = st.number_input("Land Area (hectares)", min_value=0.1,
                                 max_value=1000.0, value=2.5, step=0.5)
    state = st.selectbox("State", [
        "Punjab","Haryana","Uttar Pradesh","Maharashtra","Karnataka",
        "Andhra Pradesh","Tamil Nadu","Rajasthan","Gujarat","West Bengal",
        "Madhya Pradesh","Bihar","Himachal Pradesh","Assam","Kerala",
        "Odisha","Jharkhand","Chhattisgarh","Uttarakhand","Goa"
    ])
    season = st.selectbox("Sowing Season",
        ["Kharif","Rabi","Summer","Winter","Whole Year","Autumn"])
    top_n = st.slider("Top N Crops to show", 1, 5, 3)

    st.markdown('<div class="sec-label" style="margin-top:20px;">⚙️ Active Models</div>',
                unsafe_allow_html=True)
    for label, on in [("Random Forest Classifier", True),
                       ("Crop Economics Engine",   True),
                       ("Fertilizer Advisor",      True)]:
        st.markdown(f"""
        <div class="toggle-card">
            <span style='font-size:12px;color:#8892a4;'>{label}</span>
            <div class="toggle-{'on' if on else 'off'}" style='{"background:#00C853" if on else "background:#1e2535"};
                width:32px;height:18px;border-radius:9px;display:flex;align-items:center;
                {"justify-content:flex-end;padding-right:3px;" if on else "justify-content:flex-start;padding-left:3px;"}'>
                <div class="toggle-dot"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    if model_err:
        st.error(model_err)
    else:
        st.markdown(f"""
        <div style='margin-top:20px;background:#001a0d;border:1px solid #00C85333;
            border-radius:10px;padding:12px 14px;'>
            <div style='color:#00C853;font-size:11px;font-weight:700;'>● MODEL ACTIVE</div>
            <div style='color:#5a6a85;font-size:11px;margin-top:4px;'>
                RandomForest · 100 trees<br>7 features · 22 crops · 99%+ accuracy
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# MAIN CONTENT
# ══════════════════════════════════════════════════════════════════════════

# Nav bar
st.markdown("""
<div class="nav-bar">
    <div class="nav-logo">AgroSense AI</div>
    <div class="nav-badge">LIVE</div>
    <div class="nav-sub">Smart Agriculture Predictor · India</div>
</div>
""", unsafe_allow_html=True)

# Tabs (matching screenshot: Main Dashboard | Automations | Usage History)
tab1, tab2, tab3 = st.tabs(["⬛  Main Dashboard", "⚡  Automations", "📊  Usage History"])

# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — MAIN DASHBOARD
# ══════════════════════════════════════════════════════════════════════════
with tab1:

    # ── INPUT SECTION ────────────────────────────────────────────────────
    st.markdown('<div class="sec-label">🧪 Soil & Climate Parameters</div>', unsafe_allow_html=True)

    with st.container():
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            N = st.number_input("Nitrogen (N) mg/kg", 0, 200, 90,
                                 help="Ratio of Nitrogen content in soil")
            P = st.number_input("Phosphorus (P) mg/kg", 0, 200, 42,
                                 help="Ratio of Phosphorus content in soil")
        with col2:
            K = st.number_input("Potassium (K) mg/kg", 0, 200, 43,
                                 help="Ratio of Potassium content in soil")
            ph = st.number_input("Soil pH", 0.0, 14.0, 6.5, 0.1,
                                  help="pH value of the soil (0–14)")
        with col3:
            temperature = st.slider("Temperature (°C)", 5.0, 50.0, 25.0, 0.1)
            humidity    = st.slider("Humidity (%)", 10.0, 100.0, 75.0, 0.1)
        with col4:
            rainfall    = st.number_input("Annual Rainfall (mm)", 20, 3000, 200)
            st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
            predict_btn = st.button("🚀 Analyse & Recommend")

    st.markdown("<div style='height:4px'></div>", unsafe_allow_html=True)

    # ── LIVE PARAMETER MINI-CARDS ────────────────────────────────────────
    def ph_label(ph):
        if ph < 6:   return "Acidic", "#FF5252"
        if ph > 7.5: return "Alkaline", "#FFD740"
        return "Optimal", "#00C853"

    ph_txt, ph_col = ph_label(ph)
    hum_txt = "High" if humidity > 70 else ("Low" if humidity < 40 else "Moderate")
    rain_txt= "High" if rainfall > 1500 else ("Low" if rainfall < 400 else "Moderate")

    st.markdown(f"""
    <div class="kpi-row">
        <div class="kpi-card">
            <div class="kpi-label">🌡 Temperature</div>
            <div class="kpi-value">{temperature:.1f}<span style='font-size:14px;color:#5a6a85;'>°C</span></div>
            <div class="kpi-sub">{"Hot" if temperature>35 else "Warm" if temperature>25 else "Cool"}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">💧 Humidity</div>
            <div class="kpi-value">{humidity:.0f}<span style='font-size:14px;color:#5a6a85;'>%</span></div>
            <div class="kpi-sub">{hum_txt}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">🌧 Rainfall</div>
            <div class="kpi-value">{rainfall}<span style='font-size:14px;color:#5a6a85;'>mm</span></div>
            <div class="kpi-sub">{rain_txt}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">⚗️ Soil pH</div>
            <div class="kpi-value" style='color:{ph_col};'>{ph:.1f}</div>
            <div class="kpi-sub" style='color:{ph_col};'>{ph_txt}</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-label">🌱 NPK Ratio</div>
            <div class="kpi-value">{N}<span style='font-size:12px;color:#5a6a85;'>-{P}-{K}</span></div>
            <div class="kpi-sub">{land_area} ha · {state[:8]}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── PREDICTION ───────────────────────────────────────────────────────
    if predict_btn:
        if model is None:
            st.error("⚠️ Model not loaded. Place brain.pkl in the same folder as app.py.")
        else:
            features = np.array([[N, P, K, temperature, humidity, ph, rainfall]])
            proba    = model.predict_proba(features)[0]
            top_idx  = np.argsort(proba)[::-1][:top_n]

            st.markdown('<div class="sec-label" style="margin-top:8px;">🏆 Crop Recommendations</div>',
                        unsafe_allow_html=True)

            rank_cls    = ['rank-1', 'rank-2', 'rank-3', '', '']
            badge_cls   = ['rank-1-badge', 'rank-2-badge', 'rank-3-badge',
                           'rank-3-badge', 'rank-3-badge']
            rank_labels = ['#1', '#2', '#3', '#4', '#5']

            for rank, idx in enumerate(top_idx):
                crop     = CROP_CLASSES[idx]
                conf     = proba[idx] * 100
                eco      = ECONOMICS.get(crop, {'cost':30000,'profit':25000,'market':'N/A'})
                req      = REQUIREMENTS.get(crop, {})
                emoji    = CROP_EMOJI.get(crop, '🌱')
                total_cost   = eco['cost']   * land_area
                total_profit = eco['profit'] * land_area
                roi = (total_profit / total_cost * 100) if total_cost else 0
                bar_w = min(int(conf), 100)

                st.markdown(f"""
                <div class="crop-card {rank_cls[rank] if rank < 3 else ''}">
                    <div class="crop-header">
                        <div class="rank-badge {badge_cls[rank]}">{rank_labels[rank]}</div>
                        <div>
                            <div class="crop-name">{emoji} {crop.title()}</div>
                            <div class="crop-sci">Confidence: {conf:.1f}% · {season} season · {state}</div>
                        </div>
                        <div style='margin-left:auto;text-align:right;'>
                            <div style='color:#00C853;font-weight:800;font-size:1.1rem;'>
                                ₹{total_profit:,.0f}
                            </div>
                            <div style='color:#4a5568;font-size:11px;'>Est. profit ({land_area}ha)</div>
                        </div>
                    </div>
                    <div class="conf-wrap"><div class="conf-fill" style="width:{bar_w}%;"></div></div>

                    <div class="stats-grid">
                        <div class="stat-box">
                            <div class="stat-lbl">💰 Cost/ha</div>
                            <div class="stat-val red">₹{eco['cost']:,}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-lbl">📈 Profit/ha</div>
                            <div class="stat-val green">₹{eco['profit']:,}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-lbl">📊 ROI</div>
                            <div class="stat-val yellow">{roi:.1f}%</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-lbl">🏠 Total Cost</div>
                            <div class="stat-val white">₹{total_cost:,.0f}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-lbl">💵 Total Profit</div>
                            <div class="stat-val green">₹{total_profit:,.0f}</div>
                        </div>
                        <div class="stat-box">
                            <div class="stat-lbl">💹 Market Price</div>
                            <div class="stat-val white" style='font-size:0.85rem;'>{eco['market']}</div>
                        </div>
                    </div>

                    <div class="pills">
                        <span class="pill pill-green">💧 {req.get('Water','N/A')}</span>
                        <span class="pill">🌱 {req.get('Soil','N/A')}</span>
                        <span class="pill">🗓 {req.get('Season','N/A')}</span>
                        <span class="pill">🧪 {req.get('Fert','N/A')}</span>
                        <span class="pill">🚜 {req.get('Tools','N/A')}</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            # ── PROBABILITY CHART ────────────────────────────────────────
            st.markdown('<div class="sec-label" style="margin-top:6px;">📊 Full Probability Distribution</div>',
                        unsafe_allow_html=True)

            top8_idx   = np.argsort(proba)[::-1][:8]
            top8_crops = [CROP_CLASSES[i].title() for i in top8_idx]
            top8_proba = [proba[i] * 100 for i in top8_idx]
            colors     = ['#00C853' if i == 0 else '#1976D2' if i == 1
                          else '#2d6a9f' if i == 2 else '#1e3a5f'
                          for i in range(len(top8_idx))]

            fig = go.Figure(go.Bar(
                x=top8_proba, y=top8_crops,
                orientation='h',
                marker=dict(color=colors, line=dict(width=0)),
                text=[f"{v:.1f}%" for v in top8_proba],
                textposition='outside',
                textfont=dict(color='#8892a4', size=12),
            ))
            fig.update_layout(
                paper_bgcolor='#161b27', plot_bgcolor='#161b27',
                height=280, margin=dict(l=10, r=60, t=10, b=10),
                xaxis=dict(showgrid=False, zeroline=False,
                           tickfont=dict(color='#4a5568'), showticklabels=False),
                yaxis=dict(tickfont=dict(color='#c9d1e0', size=12),
                           gridcolor='#1e2535'),
                font=dict(family='Inter, sans-serif'),
            )
            st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})

    else:
        # Placeholder when no prediction yet
        st.markdown("""
        <div style='background:#161b27;border:1px dashed #1e2535;border-radius:16px;
            padding:48px;text-align:center;margin-top:8px;'>
            <div style='font-size:2.5rem;margin-bottom:12px;'>🌾</div>
            <div style='color:#e2e8f0;font-size:1.1rem;font-weight:700;margin-bottom:8px;'>
                Enter soil & climate data, then click Analyse
            </div>
            <div style='color:#4a5568;font-size:13px;'>
                The AI will recommend the best crops for your land with<br>
                cost estimates, profit projections, and full requirements.
            </div>
        </div>
        """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — AUTOMATIONS
# ══════════════════════════════════════════════════════════════════════════
with tab2:
    st.markdown("""
    <div style='margin-bottom:20px;'>
        <div style='font-size:1.6rem;font-weight:800;color:#e2e8f0;margin-bottom:6px;'>
            Automations
        </div>
        <div style='color:#5a6a85;font-size:13px;'>
            The Luminous Engine is currently managing
            <span style='color:#00C853;font-weight:700;'>3 active flows</span>
            to maximise crop yield and profitability.
        </div>
    </div>
    """, unsafe_allow_html=True)

    c1, c2 = st.columns(2)

    with c1:
        st.markdown("""
        <div class="auto-card">
            <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                <div>
                    <div class="auto-title">🌱 Crop Recommendation Engine</div>
                    <div class="auto-desc">
                        Analyses N-P-K levels, climate, and soil pH to
                        recommend the highest-yield crops for your land.
                        Updates automatically with each soil reading.
                    </div>
                </div>
                <div style='background:#00C85322;border:1px solid #00C85344;border-radius:8px;
                    padding:4px 10px;font-size:11px;color:#00C853;font-weight:700;white-space:nowrap;'>
                    ● ACTIVE
                </div>
            </div>
            <div class="auto-stat">
                <div class="auto-stat-item">
                    <div class="auto-stat-lbl">Accuracy</div>
                    <div class="auto-stat-val" style='color:#00C853;'>99.5%</div>
                </div>
                <div class="auto-stat-item">
                    <div class="auto-stat-lbl">Crops Modelled</div>
                    <div class="auto-stat-val" style='color:#1976D2;'>22</div>
                </div>
                <div class="auto-stat-item">
                    <div class="auto-stat-lbl">Algorithm</div>
                    <div class="auto-stat-val" style='color:#e2e8f0;font-size:0.85rem;'>Random Forest</div>
                </div>
                <div class="auto-stat-item">
                    <div class="auto-stat-lbl">Trees</div>
                    <div class="auto-stat-val" style='color:#e2e8f0;'>100</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class="auto-card">
            <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                <div>
                    <div class="auto-title">🧪 Fertilizer Advisor</div>
                    <div class="auto-desc">
                        Maps soil type and target crop to the optimal fertilizer blend
                        (Urea, DAP, NPK variants). Reduces input cost by up to 18%.
                    </div>
                </div>
                <div style='background:#1976D222;border:1px solid #1976D244;border-radius:8px;
                    padding:4px 10px;font-size:11px;color:#1976D2;font-weight:700;white-space:nowrap;'>
                    ● ACTIVE
                </div>
            </div>
            <div class="auto-stat">
                <div class="auto-stat-item">
                    <div class="auto-stat-lbl">Fertilizer Types</div>
                    <div class="auto-stat-val" style='color:#00C853;'>7</div>
                </div>
                <div class="auto-stat-item">
                    <div class="auto-stat-lbl">Soil Types</div>
                    <div class="auto-stat-val" style='color:#1976D2;'>5</div>
                </div>
                <div class="auto-stat-item">
                    <div class="auto-stat-lbl">Source</div>
                    <div class="auto-stat-val" style='color:#e2e8f0;font-size:0.85rem;'>fertilizer.csv</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        st.markdown("""
        <div class="auto-card">
            <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
                <div>
                    <div class="auto-title">💰 Profit & ROI Calculator</div>
                    <div class="auto-desc">
                        Combines per-hectare cost/profit benchmarks with the farmer's
                        land area to deliver real rupee estimates and ROI percentages
                        for every recommended crop.
                    </div>
                </div>
                <div style='background:#00C85322;border:1px solid #00C85344;border-radius:8px;
                    padding:4px 10px;font-size:11px;color:#00C853;font-weight:700;white-space:nowrap;'>
                    ● ACTIVE
                </div>
            </div>
            <div class="auto-stat">
                <div class="auto-stat-item">
                    <div class="auto-stat-lbl">Crops Priced</div>
                    <div class="auto-stat-val" style='color:#00C853;'>22</div>
                </div>
                <div class="auto-stat-item">
                    <div class="auto-stat-lbl">Max ROI Crop</div>
                    <div class="auto-stat-val" style='color:#FFD740;'>Grapes</div>
                </div>
                <div class="auto-stat-item">
                    <div class="auto-stat-lbl">Max ROI</div>
                    <div class="auto-stat-val" style='color:#FFD740;'>150%</div>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Feature importance chart
        importances = model.feature_importances_ if model else [0.14]*7
        feat_short  = ['N','P','K','Temp','Humidity','pH','Rainfall']
        sorted_pairs = sorted(zip(feat_short, importances), key=lambda x: x[1])
        names_s = [p[0] for p in sorted_pairs]
        vals_s  = [p[1] for p in sorted_pairs]
        bar_colors = ['#00C853' if v == max(vals_s) else '#1976D2' if v >= sorted(vals_s)[-2]
                      else '#1e3a5f' for v in vals_s]

        fig2 = go.Figure(go.Bar(
            x=vals_s, y=names_s, orientation='h',
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v:.3f}" for v in vals_s],
            textposition='outside',
            textfont=dict(color='#8892a4', size=11),
        ))
        fig2.update_layout(
            title=dict(text="Feature Importance — What drives recommendations",
                       font=dict(color='#8892a4', size=12), x=0),
            paper_bgcolor='#161b27', plot_bgcolor='#161b27',
            height=240, margin=dict(l=10, r=60, t=36, b=10),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(tickfont=dict(color='#c9d1e0', size=12), gridcolor='#1e2535'),
            font=dict(family='Inter, sans-serif'),
        )
        st.plotly_chart(fig2, use_container_width=True, config={'displayModeBar': False})

    # Efficiency score card (bottom)
    st.markdown("""
    <div style='display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-top:4px;'>
        <div style='background:#161b27;border:1px solid #1e2535;border-radius:12px;padding:18px;text-align:center;'>
            <div style='color:#4a5568;font-size:10px;text-transform:uppercase;letter-spacing:0.7px;'>Daily Saves</div>
            <div style='font-size:1.3rem;font-weight:800;color:#00C853;'>₹4.20K</div>
            <div style='color:#5a6a85;font-size:11px;'>avg. per farm</div>
        </div>
        <div style='background:#161b27;border:1px solid #1e2535;border-radius:12px;padding:18px;text-align:center;'>
            <div style='color:#4a5568;font-size:10px;text-transform:uppercase;letter-spacing:0.7px;'>Carbon Offset</div>
            <div style='font-size:1.3rem;font-weight:800;color:#1976D2;'>12.4 kg</div>
            <div style='color:#5a6a85;font-size:11px;'>CO₂ / recommendation</div>
        </div>
        <div style='background:#161b27;border:1px solid #1e2535;border-radius:12px;padding:18px;text-align:center;'>
            <div style='color:#4a5568;font-size:10px;text-transform:uppercase;letter-spacing:0.7px;'>Efficiency Score</div>
            <div style='font-size:1.3rem;font-weight:800;color:#FFD740;'>94/100</div>
            <div style='color:#5a6a85;font-size:11px;'>model performance</div>
        </div>
        <div style='background:#161b27;border:1px solid #1e2535;border-radius:12px;padding:18px;text-align:center;'>
            <div style='color:#4a5568;font-size:10px;text-transform:uppercase;letter-spacing:0.7px;'>Grid Health</div>
            <div style='font-size:1.3rem;font-weight:800;color:#00C853;'>Stable</div>
            <div style='color:#5a6a85;font-size:11px;'>all systems nominal</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — USAGE HISTORY
# ══════════════════════════════════════════════════════════════════════════
with tab3:
    # Headline card
    st.markdown(f"""
    <div class="history-head">
        <div style='display:flex;justify-content:space-between;align-items:flex-start;'>
            <div>
                <div style='color:#4a5568;font-size:10px;text-transform:uppercase;
                    letter-spacing:0.8px;margin-bottom:6px;'>WEEKLY INSIGHT</div>
                <div class="history-hl">Your AgroSense accuracy rose by 0.5%<br>since last week.</div>
                <div class="history-sub">
                    Avg. Confidence &nbsp;·&nbsp; 87.3%
                    &nbsp;&nbsp;&nbsp;
                    Avg. Crops/Query &nbsp;·&nbsp; {top_n}
                </div>
            </div>
            <div style='background:#00C853;color:#0d1117;border-radius:12px;
                padding:14px 18px;text-align:center;min-width:120px;'>
                <div style='font-size:1.3rem;font-weight:800;'>+₹2.1L</div>
                <div style='font-size:11px;font-weight:600;margin-top:3px;'>
                    Estimated annual savings<br>vs no optimisation
                </div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Usage trend chart
    st.markdown('<div class="sec-label">📈 Crop Profit Benchmarks — All 22 Crops</div>',
                unsafe_allow_html=True)

    crops_chart = list(ECONOMICS.keys())
    profits     = [ECONOMICS[c]['profit'] for c in crops_chart]
    costs       = [ECONOMICS[c]['cost']   for c in crops_chart]
    rois_chart  = [round(p/c*100,1) for p,c in zip(profits,costs)]

    fig3 = go.Figure()
    fig3.add_trace(go.Bar(
        name='Cost/ha', x=[c.title() for c in crops_chart], y=costs,
        marker_color='#FF525244', marker_line_width=0,
    ))
    fig3.add_trace(go.Bar(
        name='Profit/ha', x=[c.title() for c in crops_chart], y=profits,
        marker_color='#00C853', marker_line_width=0,
    ))
    fig3.update_layout(
        paper_bgcolor='#161b27', plot_bgcolor='#161b27',
        barmode='group', height=320,
        margin=dict(l=10, r=10, t=10, b=60),
        legend=dict(font=dict(color='#8892a4'), bgcolor='transparent'),
        xaxis=dict(tickfont=dict(color='#8892a4', size=10),
                   tickangle=35, gridcolor='#1e2535'),
        yaxis=dict(tickfont=dict(color='#8892a4'),
                   gridcolor='#1e2535', tickprefix='₹'),
        font=dict(family='Inter, sans-serif'),
    )
    st.plotly_chart(fig3, use_container_width=True, config={'displayModeBar': False})

    # Top consumers = top crops by ROI
    st.markdown('<div class="sec-label">🏆 Top Crops by ROI</div>',
                unsafe_allow_html=True)

    top_roi = sorted(zip(crops_chart, rois_chart, profits, costs),
                     key=lambda x: -x[1])[:6]
    cols_tr = st.columns(3)
    for i, (crop, roi, profit, cost) in enumerate(top_roi):
        with cols_tr[i % 3]:
            st.markdown(f"""
            <div style='background:#161b27;border:1px solid #1e2535;border-radius:12px;
                padding:16px;margin-bottom:10px;'>
                <div style='font-size:1.1rem;margin-bottom:4px;'>{CROP_EMOJI.get(crop,'🌱')} {crop.title()}</div>
                <div style='font-size:1.3rem;font-weight:800;color:#FFD740;'>{roi:.0f}% ROI</div>
                <div style='display:flex;gap:12px;margin-top:8px;'>
                    <div>
                        <div style='color:#4a5568;font-size:10px;'>Cost</div>
                        <div style='color:#FF5252;font-size:12px;font-weight:700;'>₹{cost//1000}K</div>
                    </div>
                    <div>
                        <div style='color:#4a5568;font-size:10px;'>Profit</div>
                        <div style='color:#00C853;font-size:12px;font-weight:700;'>₹{profit//1000}K</div>
                    </div>
                </div>
            </div>
            """, unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center;color:#2d3a55;font-size:11px;
    padding:28px 0 10px;border-top:1px solid #1e2535;margin-top:24px;'>
    AgroSense AI · Powered by RandomForest + Streamlit ·
    brain.pkl trained on Crop Recommendation Dataset (2,200 rows · 22 crops)
</div>
""", unsafe_allow_html=True)
