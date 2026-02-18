import pandas as pd

df = pd.read_csv("data/processed/cleaned_weather_data.csv")

date_col = [c for c in df.columns if "date" in c.lower()][0]
df[date_col] = pd.to_datetime(df[date_col])

df["year_month"] = df[date_col].dt.to_period("M")

numeric_cols = df.select_dtypes(include="number").columns
monthly_df = df.groupby("year_month")[numeric_cols].mean().reset_index()

monthly_df.to_csv(
    "data/processed/monthly_weather_data.csv", index=False
)

print("Monthly aggregated data created")
