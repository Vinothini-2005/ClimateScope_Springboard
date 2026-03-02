import pandas as pd
import numpy as np
from pathlib import Path
from dash import Dash, dcc, html, Input, Output
import plotly.express as px
import plotly.graph_objects as go
import pycountry

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
df["30day_temp_avg"] = df.groupby("country")["temperature_celsius"].rolling(30).mean().reset_index(0, drop=True)

df["rain_7day_avg"] = df.groupby("country")["precip_mm"].rolling(7).mean().reset_index(0, drop=True)

df["temp_volatility_7day"] = (
    df.groupby("country")["temperature_celsius"]
    .rolling(7).std()
    .reset_index(0, drop=True)
)

df["comfort_index"] = 100 - (
    abs(df["temperature_celsius"] - 22) * 2 +
    abs(df["humidity"] - 50) * 0.5
)

df["flood_risk_score"] = (
    df["precip_mm"] * 0.5 +
    df["humidity"] * 0.3 -
    df["pressure_mb"] * 0.2
)

df["flood_risk_level"] = pd.cut(
    df["flood_risk_score"],
    bins=3,
    labels=["Low", "Moderate", "High"]
)

heat_threshold = df["temperature_celsius"].quantile(0.95)
df["heatwave"] = df["temperature_celsius"] > heat_threshold

extreme_threshold = df["temperature_celsius"].quantile(0.99)
df["extreme_heat"] = df["temperature_celsius"] > extreme_threshold

df["temp_zscore"] = (
    (df["temperature_celsius"] - df["temperature_celsius"].mean())
    / df["temperature_celsius"].std()
)

df["temp_anomaly"] = abs(df["temp_zscore"]) > 2


# ===================== DASH APP =====================
app = Dash(__name__)
app.title = "ClimateScope Dashboard"

countries = df["country"].unique()

# ===================== LAYOUT =====================
app.layout = html.Div(style={
    "backgroundColor": "#0f0f1a",
    "color": "white",
    "fontFamily": "Arial",
    "display": "flex"
}, children=[

    # Sidebar
    html.Div(style={
        "width": "20%",
        "padding": "20px",
        "backgroundColor": "#1e1e2f"
    }, children=[
        html.H2("Filters"),

        html.Label("Select Country"),
        dcc.Dropdown(
            id="country_filter",
            options=[{"label": c, "value": c} for c in countries],
            value=[countries[0]],
            multi=True
        ),

        html.Br(),

        html.Label("Select Date Range"),
        dcc.DatePickerRange(
            id="date_filter",
            min_date_allowed=df["date"].min(),
            max_date_allowed=df["date"].max(),
            start_date=df["date"].min(),
            end_date=df["date"].max(),
            display_format="YYYY-MM-DD"
        )
    ]),

    # Main Content
    html.Div(style={
        "width": "80%",
        "padding": "20px"
    }, children=[

        html.H1("🌍 ClimateScope Analytics"),

        html.Div(id="kpi_cards", style={
            "display": "flex",
            "justifyContent": "space-between",
            "marginBottom": "30px"
        }),

        dcc.Graph(id="trend_chart"),
        dcc.Graph(id="volatility_chart"),
        dcc.Graph(id="latitude_gradient_chart"),

        html.Div([
            html.Div(dcc.Graph(id="seasonal_heatmap"), style={"width": "49%"}),
            html.Div(dcc.Graph(id="correlation_heatmap"), style={"width": "49%"})
        ], style={"display": "flex", "justifyContent": "space-between"}),

        html.Div([
            html.Div(dcc.Graph(id="distribution_chart"), style={"width": "49%"}),
            html.Div(dcc.Graph(id="temp_humidity_chart"), style={"width": "49%"})
        ], style={"display": "flex", "justifyContent": "space-between"}),

        dcc.Graph(id="choropleth_chart"),
        dcc.Graph(id="anomaly_chart"),

        html.Div([
            html.Div(dcc.Graph(id="violin_chart"), style={"width": "49%"}),
            html.Div(dcc.Graph(id="season_pie_chart"), style={"width": "49%"})
        ], style={"display": "flex", "justifyContent": "space-between"}),
    ])
])


# ===================== CALLBACK =====================
@app.callback(
    Output("kpi_cards", "children"),
    Output("trend_chart", "figure"),
    Output("volatility_chart", "figure"),
    Output("latitude_gradient_chart", "figure"),
    Output("seasonal_heatmap", "figure"),
    Output("correlation_heatmap", "figure"),
    Output("distribution_chart", "figure"),
    Output("temp_humidity_chart", "figure"),
    Output("choropleth_chart", "figure"),
    Output("anomaly_chart", "figure"),
    Output("violin_chart", "figure"),
    Output("season_pie_chart", "figure"),
    Input("country_filter", "value"),
    Input("date_filter", "start_date"),
    Input("date_filter", "end_date"),
)
def update_dashboard(selected_country, start_date, end_date):

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    dff = df[
        df["country"].isin(selected_country) &
        (df["date"] >= start_date) &
        (df["date"] <= end_date)
    ].sort_values("date")

    # 🔹 SAFE EMPTY CHECK
    if dff.empty:
        empty_fig = go.Figure()
        empty_fig.update_layout(template="plotly_dark")
        return ([],) + (empty_fig,) * 11

    # ================= KPI =================
    mean_temp = round(dff["temperature_celsius"].mean(), 2)
    skewness = round(dff["temperature_celsius"].skew(), 2)
    Q1 = dff["temperature_celsius"].quantile(0.25)
    Q3 = dff["temperature_celsius"].quantile(0.75)
    IQR = round(Q3 - Q1, 2)
    std_dev = dff["temperature_celsius"].std()
    health_score = round(100 - std_dev, 2)

    card_style = {
        "backgroundColor": "#1e1e2f",
        "padding": "20px",
        "borderRadius": "12px",
        "width": "23%",
        "textAlign": "center"
    }

    kpis = [
        html.Div([html.H4("🌡 Mean Temp"), html.H2(f"{mean_temp} °C")], style=card_style),
        html.Div([html.H4("📊 Skewness"), html.H2(f"{skewness}")], style=card_style),
        html.Div([html.H4("📏 IQR"), html.H2(f"{IQR}")], style=card_style),
        html.Div([html.H4("💚 Health Score"), html.H2(f"{health_score}%")], style=card_style),
    ]

    # ================= FIGURES =================
    trend_fig = px.line(dff, x="date", y="temperature_celsius",
                        color="country", template="plotly_dark", title="📈 Temperature Trend Over Time (Line chart)")

    volatility_chart = px.line(dff, x="date", y="temp_volatility_7day",
                               color="country", template="plotly_dark",title="📊 7-Day Rolling Temperature Volatility (Line chart)")

    latitude_gradient_chart = px.scatter(
        dff, x="latitude", y="temperature_celsius",
        color="country", 
        template="plotly_dark", title="🌍 Latitudinal Temperature Gradient Analysis (Scatter Plot)"
    )

    seasonal_data = dff.groupby(["country","season"])["temperature_celsius"].mean().reset_index()
    seasonal_pivot = seasonal_data.pivot(
        index="season", columns="country",
        values="temperature_celsius"
    ).fillna(0)

    seasonal_heatmap = px.imshow(seasonal_pivot,
                                 text_auto=True,
                                 template="plotly_dark",
                                 title="🔥 Seasonal Average Temperature Comparison (Heatmap)")

    corr = dff[["temperature_celsius","humidity",
                "wind_kph","pressure_mb","precip_mm"]].corr()

    correlation_heatmap = px.imshow(corr,
                                    text_auto=True,
                                    template="plotly_dark", title="🔗 Correlation Between Climate Variables")

    distribution_chart = px.histogram(
        dff, x="temperature_celsius",
        marginal="box", color="country",
        template="plotly_dark", title="📦 Temperature Distribution with Box Summary "
    )

    temp_humidity_chart = px.scatter(
        dff, x="temperature_celsius",
        y="humidity", color="country",
        template="plotly_dark",
        title="💧 Temperature vs Humidity Relationship (Scatter Plot)"
    )

    choropleth_chart = px.choropleth(
        dff[dff["heatwave"]],
        locations="country",
        locationmode="country names",
        color="temperature_celsius",
        template="plotly_dark", title="🌡 Heatwave Intensity by Country (Choropleath Map)"
    )

    anomaly_chart = px.scatter(
        dff, x="date", y="temperature_celsius",
        color=dff["temp_anomaly"],
        template="plotly_dark", title="🚨 Temperature Anomaly Detection (Z-Score > 2)"
    )

    violin_chart = px.violin(
        dff, x="country", y="temperature_celsius",
        box=True, points="outliers",
        template="plotly_dark",
        title="🎻 Temperature Distribution by Country (Violin Plot)"
    )

    season_counts = dff["season"].value_counts().reset_index()
    season_counts.columns = ["season", "count"]

    season_pie_chart = px.pie(
        season_counts,
        names="season",
        values="count",
        template="plotly_dark",
        title="🥧 Seasonal Distribution Overview (Pie Chart)"
    )

    return (
        kpis,
        trend_fig,
        volatility_chart,
        latitude_gradient_chart,
        seasonal_heatmap,
        correlation_heatmap,
        distribution_chart,
        temp_humidity_chart,
        choropleth_chart,
        anomaly_chart,
        violin_chart,
        season_pie_chart
    )


if __name__ == "__main__":
    app.run(debug=True)