import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pycountry

# ===================== PAGE CONFIG =====================
st.set_page_config(page_title="ClimateScope Dashboard", layout="wide")

st.title("🌍 ClimateScope Analytics Dashboard")

# ===================== LOAD DATA =====================
BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "processed" / "cleaned_weather_data.csv"

df = pd.read_csv(DATA_FILE)

df["date"] = pd.to_datetime(df["last_updated"])
df = df.sort_values("date")

df["month"] = df["date"].dt.month


def get_season(month):
    if month in [12,1,2]:
        return "Winter"
    elif month in [3,4,5]:
        return "Summer"
    elif month in [6,7,8]:
        return "Monsoon"
    else:
        return "Autumn"


df["season"] = df["month"].apply(get_season)


# ISO3 conversion
def country_to_iso3(name):
    try:
        return pycountry.countries.lookup(name).alpha_3
    except:
        return None


df["iso3"] = df["country"].apply(country_to_iso3)

# ===================== FEATURE ENGINEERING =====================

df["7day_temp_avg"] = df.groupby("country")["temperature_celsius"].rolling(7).mean().reset_index(0, drop=True)

df["temp_volatility_7day"] = (
    df.groupby("country")["temperature_celsius"]
    .rolling(7)
    .std()
    .reset_index(0, drop=True)
)

df["temp_zscore"] = (
    (df["temperature_celsius"] - df["temperature_celsius"].mean())
    / df["temperature_celsius"].std()
)

df["temp_anomaly"] = abs(df["temp_zscore"]) > 2

heat_threshold = df["temperature_celsius"].quantile(0.95)
df["heatwave"] = df["temperature_celsius"] > heat_threshold

df["temperature_risk"] = (
    df["temperature_celsius"] * 0.6 +
    df["humidity"] * 0.2 +
    df["precip_mm"] * 0.1 -
    df["wind_kph"] * 0.1
)

# ===================== SIDEBAR FILTER =====================

st.sidebar.header("Filters")

countries = df["country"].unique()

selected_country = st.sidebar.multiselect(
    "Select Country",
    countries,
    default=[countries[0]]
)

start_date = st.sidebar.date_input("Start Date", df["date"].min())
end_date = st.sidebar.date_input("End Date", df["date"].max())

# ===================== FILTER DATA =====================

dff = df[
    df["country"].isin(selected_country) &
    (df["date"] >= pd.to_datetime(start_date)) &
    (df["date"] <= pd.to_datetime(end_date))
]

# ===================== KPI CARDS =====================

col1, col2, col3, col4 = st.columns(4)

mean_temp = round(dff["temperature_celsius"].mean(),2)
skewness = round(dff["temperature_celsius"].skew(),2)
IQR = round(
    dff["temperature_celsius"].quantile(0.75) -
    dff["temperature_celsius"].quantile(0.25),2
)
health_score = round(100 - dff["temperature_celsius"].std(),2)

col1.metric("🌡 Mean Temp", f"{mean_temp} °C")
col2.metric("📊 Skewness", skewness)
col3.metric("📏 IQR", IQR)
col4.metric("💚 Health Score", f"{health_score}%")

# ===================== CHARTS =====================

trend_fig = px.line(
    dff,
    x="date",
    y="temperature_celsius",
    color="country",
    template="plotly_dark",
    title="Temperature Trend Over Time"
)

volatility_chart = px.line(
    dff,
    x="date",
    y="temp_volatility_7day",
    color="country",
    template="plotly_dark",
    title="7-Day Temperature Volatility"
)

latitude_gradient_chart = px.scatter(
    dff,
    x="latitude",
    y="temperature_celsius",
    color="country",
    template="plotly_dark",
    title="Latitudinal Temperature Gradient"
)

seasonal_data = dff.groupby(["country","season"])["temperature_celsius"].mean().reset_index()
seasonal_pivot = seasonal_data.pivot(index="season",columns="country",values="temperature_celsius").fillna(0)

seasonal_heatmap = px.imshow(
    seasonal_pivot,
    text_auto=True,
    template="plotly_dark",
    title="Seasonal Temperature Heatmap"
)

corr = dff[["temperature_celsius","humidity","wind_kph","pressure_mb","precip_mm"]].corr()

correlation_heatmap = px.imshow(
    corr,
    text_auto=True,
    template="plotly_dark",
    title="Correlation Heatmap"
)

distribution_chart = px.histogram(
    dff,
    x="temperature_celsius",
    color="country",
    marginal="box",
    template="plotly_dark",
    title="Temperature Distribution"
)

temp_humidity_chart = px.scatter(
    dff,
    x="temperature_celsius",
    y="humidity",
    color="country",
    template="plotly_dark",
    title="Temperature vs Humidity"
)

risk_data = dff.groupby("country")["temperature_risk"].mean().reset_index()

choropleth_chart = px.choropleth(
    risk_data,
    locations="country",
    locationmode="country names",
    color="temperature_risk",
    color_continuous_scale="inferno",
    template="plotly_dark",
    title="🌡 Temperature Risk Index by Country"
)

anomaly_chart = px.scatter(
    dff,
    x="date",
    y="temperature_celsius",
    color=dff["temp_anomaly"],
    template="plotly_dark",
    title="Temperature Anomaly Detection"
)

violin_chart = px.violin(
    dff,
    x="country",
    y="temperature_celsius",
    box=True,
    template="plotly_dark",
    title="Temperature Distribution by Country"
)

season_counts = dff["season"].value_counts().reset_index()
season_counts.columns=["season","count"]

season_pie_chart = px.pie(
    season_counts,
    names="season",
    values="count",
    template="plotly_dark",
    title="Season Distribution"
)

bar_chart = px.bar(
    dff.groupby("country")["temperature_celsius"].mean().reset_index(),
    x="country",
    y="temperature_celsius",
    color="country",
    template="plotly_dark",
    title="Average Temperature by Country"
)

cluster_chart = px.bar(
    dff.groupby("country")[["temperature_celsius","humidity"]].mean().reset_index(),
    x="country",
    y=["temperature_celsius","humidity"],
    barmode="group",
    template="plotly_dark",
    title="Temperature vs Humidity Comparison"
)

stacked_chart = px.bar(
    dff.groupby(["country","season"])["temperature_celsius"].mean().reset_index(),
    x="country",
    y="temperature_celsius",
    color="season",
    barmode="stack",
    template="plotly_dark",
    title="Seasonal Temperature Contribution"
)

box_chart = px.box(
    dff,
    x="country",
    y="temperature_celsius",
    color="country",
    template="plotly_dark",
    title="Temperature Spread"
)

area_chart = px.area(
    dff,
    x="date",
    y="temperature_celsius",
    color="country",
    template="plotly_dark",
    title="Temperature Trend Area Chart"
)

# ===================== DISPLAY CHARTS =====================

st.plotly_chart(trend_fig, use_container_width=True)
st.plotly_chart(volatility_chart, use_container_width=True)
st.plotly_chart(latitude_gradient_chart, use_container_width=True)

col1,col2 = st.columns(2)
col1.plotly_chart(seasonal_heatmap, use_container_width=True)
col2.plotly_chart(correlation_heatmap, use_container_width=True)

col1,col2 = st.columns(2)
col1.plotly_chart(distribution_chart, use_container_width=True)
col2.plotly_chart(temp_humidity_chart, use_container_width=True)

st.plotly_chart(choropleth_chart, use_container_width=True)
st.plotly_chart(anomaly_chart, use_container_width=True)

col1,col2 = st.columns(2)
col1.plotly_chart(violin_chart, use_container_width=True)
col2.plotly_chart(season_pie_chart, use_container_width=True)

col1,col2 = st.columns(2)
col1.plotly_chart(bar_chart, use_container_width=True)
col2.plotly_chart(cluster_chart, use_container_width=True)

col1,col2 = st.columns(2)
col1.plotly_chart(stacked_chart, use_container_width=True)
col2.plotly_chart(box_chart, use_container_width=True)

st.plotly_chart(area_chart, use_container_width=True)