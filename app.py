
import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime
import re

# Optional sentiment with NLTK (downloads on first run)
try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    import nltk
    _ = nltk.data.find('sentiment/vader_lexicon.zip')
except Exception:
    import nltk
    nltk.download('vader_lexicon')
    from nltk.sentiment import SentimentIntensityAnalyzer

st.set_page_config(page_title="Deal Risk Radar Demo - Ajitesh", page_icon="📈", layout="wide")

st.title("📈 Deal Risk Radar by Ajitesh Das")
st.write("Upload sales call transcripts or notes to score sentiment, flag risks, and spot deals that need attention.")
st.link_button("📬 Contact me", "mailto:ajiteshdas@gmail.com")

@st.cache_data
def load_sample():
    return pd.read_csv("sample_data.csv")

def show_sample_download():
    with open("sample_data.csv", "rb") as f:
        st.download_button(
            "📥 Download sample CSV",
            data=f,
            file_name="sample_data.csv",
            mime="text/csv",
            help="Grab a copy of the sample and tweak it if you like."
        )

uploaded = st.file_uploader("Upload a CSV (columns: deal_id, stage, date, duration_min, transcript)", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
else:
    st.caption("No file uploaded — using sample data.")
    df = load_sample()

# Basic hygiene
required_cols = {"deal_id","stage","date","duration_min","transcript"}
missing = required_cols - set([c.lower() for c in df.columns])
if missing:
    st.error(f"Missing required columns: {missing}. Expected columns: {sorted(list(required_cols))}")
    st.stop()

# Normalize columns
df.columns = [c.lower() for c in df.columns]
df['date'] = pd.to_datetime(df['date'], errors='coerce')

# Sentiment + simple risk rules
sia = SentimentIntensityAnalyzer()

RISK_KEYWORDS = {
    "budget": 0.25,
    "delay": 0.2,
    "postpone": 0.2,
    "competitor": 0.2,
    "stall": 0.2,
    "not moving": 0.3,
    "cancel": 0.35,
    "no decision": 0.3,
    "push back": 0.2,
    "concern": 0.15,
    "risk": 0.15,
    "legal": 0.1,
    "security": 0.1,
}
pattern = re.compile("|".join(re.escape(k) for k in RISK_KEYWORDS.keys()), re.IGNORECASE)

def score_row(text):
    text = str(text or "")
    s = sia.polarity_scores(text)["compound"]  # -1..1
    matches = pattern.findall(text)
    kw_score = sum(RISK_KEYWORDS.get(m.lower(), 0.0) for m in matches)
    # Convert sentiment to risk (negative sentiment => higher risk)
    sentiment_risk = np.interp(-s, [-1, 1], [0, 0.6])  # 0..0.6
    total_risk = min(1.0, sentiment_risk + kw_score)
    return s, len(matches), kw_score, total_risk, ", ".join(sorted(set(m.lower() for m in matches)))

df[['sentiment', 'risk_hits', 'kw_risk', 'risk_score', 'risk_terms']] = df['transcript'].apply(
    lambda t: pd.Series(score_row(t))
)

# Sidebar filters
st.sidebar.header("Filters")
st.sidebar.caption("Use these to focus your view.")
stages = sorted(df['stage'].dropna().astype(str).unique().tolist())
stage_sel = st.sidebar.multiselect("Stage", stages, default=stages)
date_min = st.sidebar.date_input("Start date", value=df['date'].min().date() if df['date'].notna().any() else None)
date_max = st.sidebar.date_input("End date", value=df['date'].max().date() if df['date'].notna().any() else None)
risk_threshold = st.sidebar.slider("Risk threshold", 0.0, 1.0, 0.5, 0.05)

mask = df['stage'].astype(str).isin(stage_sel)
if date_min:
    mask &= df['date'] >= pd.to_datetime(date_min)
if date_max:
    mask &= df['date'] <= pd.to_datetime(date_max)
f = df[mask].copy()

# KPI row
col1, col2, col3, col4 = st.columns(4)
col1.metric("Deals analyzed", f['deal_id'].nunique())
col2.metric("High-risk deals", f.loc[f['risk_score'] >= risk_threshold, 'deal_id'].nunique())
col3.metric("Avg. sentiment", f['sentiment'].mean().round(3) if len(f) else 0.0)
col4.metric("Avg. call length (min)", f['duration_min'].mean().round(1) if len(f) else 0.0)

st.subheader("Risk Scatter")
st.caption("Each point is a call/interaction. Higher = riskier. Click a row below to review context.")
import plotly.express as px
fig = px.scatter(
    f,
    x="date",
    y="risk_score",
    color="stage",
    hover_data=["deal_id", "sentiment", "risk_terms", "duration_min"],
    size="duration_min",
)
st.plotly_chart(fig, use_container_width=True)

st.subheader("High-Risk Calls")
hr = f.sort_values("risk_score", ascending=False)
hr = hr[hr['risk_score'] >= risk_threshold]
st.dataframe(
    hr[['deal_id','stage','date','risk_score','sentiment','risk_hits','risk_terms','duration_min','transcript']]
        .reset_index(drop=True),
    use_container_width=True, height=350
)

st.subheader("How scoring works")
with st.expander("Details"):
    st.markdown("""
**Risk score** combines:
- VADER sentiment: negative tone increases risk (0–0.6 of score).
- Keyword flags: words like *budget*, *delay*, *cancel*, *competitor* add risk (up to 0.4).
- Final score is capped at 1.0.

> This is a demo. In production, connect to your CRM (e.g., Pipedrive) and call recordings, and replace keywords with a learned model.
""")

st.sidebar.markdown("---")
st.sidebar.caption("👋 Built for quick HR/PM demos. Upload your own CSV to try it with your data.")
