# AgroSense AI — Smart Agriculture Predictor

**Tech:** Streamlit · RandomForest · Plotly  
**Model:** brain.pkl (trained on Crop Recommendation dataset)  
**Crops:** 22 · **Features:** N, P, K, Temperature, Humidity, pH, Rainfall

---

## Project Structure

```
agrosense/
├── app.py              ← Streamlit GUI
├── brain.pkl           ← Your trained RandomForest model
├── requirements.txt    ← Python dependencies
└── README.md
```

---

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
```
Opens at: http://localhost:8501

---

## Deploy to Streamlit Cloud (FREE, takes 5 min)

### Step 1 — Create GitHub repo
1. Go to https://github.com → click **New repository**
2. Name it `agrosense-ai` → Public → Create
3. Upload these 3 files: `app.py`, `brain.pkl`, `requirements.txt`

### Step 2 — Deploy
1. Go to https://share.streamlit.io
2. Sign in with GitHub
3. Click **New app**
4. Select repo: `agrosense-ai` · Branch: `main` · File: `app.py`
5. Click **Deploy!**

Your live URL: `https://yourname-agrosense-ai-app-xxxx.streamlit.app`

---

## Feature Reference

| Feature       | Range     | Unit   |
|---------------|-----------|--------|
| Nitrogen (N)  | 0 – 200   | mg/kg  |
| Phosphorus (P)| 0 – 200   | mg/kg  |
| Potassium (K) | 0 – 200   | mg/kg  |
| Temperature   | 5 – 50    | °C     |
| Humidity      | 10 – 100  | %      |
| pH            | 0 – 14    | —      |
| Rainfall      | 20 – 3000 | mm/yr  |
