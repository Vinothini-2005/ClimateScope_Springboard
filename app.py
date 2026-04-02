import pandas as pd
import numpy as np
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import pycountry
import sqlite3
import hashlib

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

menu = ["Login", "Register"]
choice = st.sidebar.selectbox("Menu", menu)

def hash_password(password):
    return hashlib.sha256(password.encode()).hexdigest()
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

def init_auth_db():
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT,
            password TEXT
        )
    """)
    conn.commit()
    conn.close()

init_auth_db()

def register_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute("INSERT INTO users VALUES (?, ?)", (username, password))
    conn.commit()
    conn.close()

def login_user(username, password):
    conn = sqlite3.connect("users.db")
    cursor = conn.cursor()
    cursor.execute(
        "SELECT * FROM users WHERE username=? AND password=?",
        (username, password)
    )
    data = cursor.fetchone()
    conn.close()
    return data
if not st.session_state.logged_in:
    
    if choice == "Login":
        st.subheader("🔐 Login")

        username = st.text_input("Username", key="login_user")
        password = st.text_input("Password", type="password", key="login_pass")

        if st.button("Login"):
            if username and password:
                if login_user(username, hash_password(password)):
                    st.session_state.logged_in = True
                    st.session_state.user = username
                    st.rerun()
                else:
                    st.error("Invalid credentials")
            else:
                st.warning("Please enter username and password")

    elif choice == "Register":
        st.subheader("📝 Register")

        new_user = st.text_input("Username", key="reg_user")
        new_pass = st.text_input("Password", type="password", key="reg_pass")

        if st.button("Register"):
            if new_user and new_pass:
                register_user(new_user, hash_password(new_pass))
                st.success("User registered!")
            else:
                st.warning("Please fill all fields")

    st.stop()   

st.sidebar.write(f"👋 Welcome {st.session_state.user}")

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

tab1, tab2, tab3 = st.tabs([
    "📘 Overview",
    "📊 Analytics",
    "🌾 Crop Recommendation"
])


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
mean_temp = round(dff["temperature_celsius"].mean(),2)
skewness = round(dff["temperature_celsius"].skew(),2)
IQR = round(
    dff["temperature_celsius"].quantile(0.75) -
    dff["temperature_celsius"].quantile(0.25),2
)
health_score = round(100 - dff["temperature_celsius"].std(),2)


with tab1:
    st.header("📘 Project Overview")

    st.markdown("""
    ClimateScope analyzes global weather data to provide:
    - 🌡 Temperature insights
    - 📊 Statistical analysis
    - 🌍 Country comparisons
    - 🌾 Smart farming support
    """)

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("🌡 Mean Temp", f"{mean_temp} °C")
    col2.metric("📊 Skewness", skewness)
    col3.metric("📏 IQR", IQR)
    col4.metric("💚 Health Score", f"{health_score}%")

# ===================== CHARTS =====================
with tab2:
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

    dff["climate_index"] = (
        dff["temperature_celsius"] * 0.5 +
        dff["humidity"] * 0.3 +
        dff["precip_mm"] * 0.2
    )
    map_data = dff.groupby("country")["climate_index"].mean().reset_index()

    choropleth_chart = px.choropleth(
        map_data,
        locations="country",
        locationmode="country names",
        color="climate_index",

        color_continuous_scale="Turbo",  # 🔥 more colors
        
        template="plotly_dark",
        title="🌍 Climate Index (Temp + Humidity + Rainfall)",

        hover_data={
            "climate_index": True
        }
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



with tab3:
    st.header("🌾 Crop Recommendation System")

    for country in selected_country:
        country_df = dff[dff["country"] == country]

        if not country_df.empty:
            avg_temp = country_df["temperature_celsius"].mean()
            avg_rain = country_df["precip_mm"].mean()
            avg_humidity = country_df["humidity"].mean()

            if avg_rain > 200 and avg_humidity > 70:
                crop = "Rice"
                confidence = 90
            elif 20 < avg_temp < 30:
                crop = "Wheat"
                confidence = 75
            else:
                crop = "Millets"
                confidence = 70

            st.subheader(f"🌍 {country}")
            st.metric("🌱 Recommended Crop", crop)
            st.metric("📊 Confidence", f"{confidence}%")

            # Get country data
            country_df = dff[dff["country"] == country]

            # Calculate averages
            avg_temp = country_df["temperature_celsius"].mean()
            avg_rain = country_df["precip_mm"].mean()
            avg_humidity = country_df["humidity"].mean()

            # Normalize
            def normalize(val, min_val, max_val):
                return (val - min_val) / (max_val - min_val)

            temp_score = normalize(avg_temp, 0, 40)
            rain_score = normalize(avg_rain, 0, 300)
            humidity_score = normalize(avg_humidity, 0, 100)

            # Final score
            suitability_score = round(
                (temp_score * 0.4 + rain_score * 0.3 + humidity_score * 0.3) * 100,
                2
            )

            # Display
            st.metric("🌱 Suitability Score", f"{suitability_score}%")
            st.progress(suitability_score / 100)
            
            chart_df = pd.DataFrame({
                "Factor": ["Temperature", "Rainfall", "Humidity"],
                "Value": [avg_temp, avg_rain, avg_humidity]
            })

            fig = px.bar(chart_df, x="Factor", y="Value", title="Climate Factors")
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")