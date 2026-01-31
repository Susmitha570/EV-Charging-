import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

CLEAN_DATA_PATH = "C:\ev project\dataset\ev_dataset_clean_with_charging_duration.csv"
REPORT_PATH = "reports"

st.set_page_config(page_title="EV Dashboard", layout="wide")

st.title("🚗⚡ EV Charging Infrastructure Dashboard")

# Load clean data
@st.cache_data
def load_data():
    if not os.path.exists(CLEAN_DATA_PATH):
        st.error("Clean dataset not found. Run main.py first to create clean_ev_dataset.csv")
        st.stop()
    return pd.read_csv(CLEAN_DATA_PATH)

df = load_data()

# Sidebar filters
st.sidebar.header("Filters")
if "source_city" in df.columns:
    sources = ["All"] + sorted(df["source_city"].unique().tolist())
    src = st.sidebar.selectbox("Source City", sources)
else:
    src = "All"

if "destination_city" in df.columns:
    dests = ["All"] + sorted(df["destination_city"].unique().tolist())
    dst = st.sidebar.selectbox("Destination City", dests)
else:
    dst = "All"

filtered = df.copy()
if src != "All" and "source_city" in df.columns:
    filtered = filtered[filtered["source_city"] == src]
if dst != "All" and "destination_city" in df.columns:
    filtered = filtered[filtered["destination_city"] == dst]

# KPIs
col1, col2, col3, col4 = st.columns(4)
if "distance_km" in filtered.columns:
    col1.metric("Avg Distance (km)", round(filtered["distance_km"].mean(), 2))
if "num_ev_stations" in filtered.columns:
    col2.metric("Avg Stations", round(filtered["num_ev_stations"].mean(), 2))
if "charging_capacity_kw" in filtered.columns:
    col3.metric("Avg Capacity (kW)", round(filtered["charging_capacity_kw"].mean(), 2))
col4.metric("Rows", len(filtered))

st.subheader("📄 Data Preview")
st.dataframe(filtered.head(50), use_container_width=True)

# Charts
st.subheader("📊 Charts")
c1, c2 = st.columns(2)

with c1:
    if "distance_km" in filtered.columns:
        fig = plt.figure()
        sns.histplot(filtered["distance_km"], kde=True)
        plt.title("Distance Distribution")
        st.pyplot(fig)

with c2:
    numeric_df = filtered.select_dtypes(include=["number"])
    if numeric_df.shape[1] >= 2:
        fig = plt.figure(figsize=(8,5))
        sns.heatmap(numeric_df.corr(), cmap="coolwarm")
        plt.title("Correlation Heatmap")
        st.pyplot(fig)

# If you created engineered features
st.subheader("🧠 Feature Insights")
features = [c for c in ["energy_required_kwh", "charging_time_hr", "stations_per_100km"] if c in filtered.columns]
if features:
    st.write("Engineered Features Preview:")
    st.dataframe(filtered[features].head(20), use_container_width=True)
else:
    st.info("Engineered features not found. (Optional)")

st.success("✅ Dashboard is ready for reviewer!")
