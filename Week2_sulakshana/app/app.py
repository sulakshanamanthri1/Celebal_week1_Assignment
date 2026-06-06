
import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA

st.set_page_config(page_title="Tesla Price Predictor", page_icon="🚗", layout="wide")


# Title & Intro
st.markdown("<h1 style='text-align:center;color:#2c3e50;'>🚗 Tesla Price Prediction Dashboard</h1>", unsafe_allow_html=True)
st.markdown("<p style='text-align:center;'>Fill in the details below to get predictions and explore insights.</p>", unsafe_allow_html=True)


# Input Form (Main Page)
st.header("🔧 Vehicle & Market Details")

col1, col2 = st.columns(2)

with col1:
    region = st.selectbox("🌍 Region", ["North America", "Europe", "Asia"])
    model = st.selectbox("🚘 Model", ["Model 3", "Model Y", "Model S", "Model X"])
    source_type = st.selectbox("📊 Source Type", ["Official", "Survey", "Estimate"])
    year = st.number_input("📅 Year", min_value=2015, max_value=2026, value=2025)
    month = st.number_input("📅 Month", min_value=1, max_value=12, value=6)

with col2:
    estimated_deliveries = st.slider("📦 Estimated Deliveries", 1000, 100000, 50000)
    production_units = st.slider("🏭 Production Units", 1000, 100000, 52000)
    battery_capacity = st.slider("🔋 Battery Capacity (kWh)", 50, 150, 75)
    range_km = st.slider("🛣️ Range (km)", 200, 800, 450)
    co2_saved = st.slider("🌱 CO₂ Saved (tons)", 100, 10000, 2000)
    charging_stations = st.slider("⚡ Charging Stations", 100, 5000, 1200)


# Build dataframe from inputs
user_data = pd.DataFrame([{
    "Region": region,
    "Model": model,
    "Source_Type": source_type,
    "Year": year,
    "Month": month,
    "Estimated_Deliveries": estimated_deliveries,
    "Production_Units": production_units,
    "Battery_Capacity_kWh": battery_capacity,
    "Range_km": range_km,
    "CO2_Saved_tons": co2_saved,
    "Charging_Stations": charging_stations,
    "YearMonth": f"{year}-{month}"
}])


# Preprocessor
categorical = ["Region","Model","Source_Type","YearMonth"]
numeric = [
    "Estimated_Deliveries","Production_Units","Battery_Capacity_kWh",
    "Range_km","CO2_Saved_tons","Charging_Stations"
]

preprocessor = ColumnTransformer([
    ("num", Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ]), numeric),
    ("cat", OneHotEncoder(handle_unknown="ignore"), categorical)
])


# Models (Ridge + XGBoost ensemble)
ridge = Pipeline([("preprocessor", preprocessor), ("ridge", Ridge(alpha=1.0))])
xgb = Pipeline([("preprocessor", preprocessor), ("xgb", XGBRegressor(n_estimators=200, learning_rate=0.1, max_depth=6, random_state=42, n_jobs=-1))])

# Synthetic training dataset (replace with your real one)
df = pd.DataFrame({
    "Region": ["North America"]*50,
    "Model": ["Model 3"]*50,
    "Source_Type": ["Official"]*50,
    "YearMonth": pd.date_range("2021-01", periods=50, freq="M").strftime("%Y-%m"),
    "Estimated_Deliveries": np.random.randint(20000,60000,50),
    "Production_Units": np.random.randint(20000,60000,50),
    "Battery_Capacity_kWh": np.random.choice([60,75,100],50),
    "Range_km": np.random.randint(300,600,50),
    "CO2_Saved_tons": np.random.randint(1000,5000,50),
    "Charging_Stations": np.random.randint(500,2000,50),
    "Avg_Price_USD": np.random.randint(35000,60000,50)
})

X = df.drop(columns=["Avg_Price_USD"])
y = df["Avg_Price_USD"]

ridge.fit(X, y)
xgb.fit(X, y)


# Prediction + Visualizations
if st.button("🔮 Predict Price"):
    ridge_pred = ridge.predict(user_data)[0]
    xgb_pred = xgb.predict(user_data)[0]
    avg_pred = (ridge_pred + xgb_pred) / 2

    st.markdown(f"<h2 style='color:#27ae60;text-align:center;'>Predicted Average Price: ${avg_pred:,.2f}</h2>", unsafe_allow_html=True)

    st.header("📊 Insights & Visualizations")

    # Distribution of prices
    fig1, ax1 = plt.subplots()
    sns.histplot(df["Avg_Price_USD"], bins=20, kde=True, ax=ax1, color="skyblue")
    ax1.set_title("Distribution of Tesla Prices")
    st.pyplot(fig1)

    # Deliveries vs Price
    fig2, ax2 = plt.subplots()
    sns.scatterplot(x=df["Estimated_Deliveries"], y=df["Avg_Price_USD"], ax=ax2, color="orange")
    ax2.set_title("Deliveries vs Price")
    st.pyplot(fig2)

    # Feature importance (XGBoost)
    importance = xgb.named_steps["xgb"].feature_importances_
    feat_names = xgb.named_steps["preprocessor"].get_feature_names_out()
    imp_df = pd.DataFrame({"Feature": feat_names, "Importance": importance}).sort_values("Importance", ascending=False).head(10)

    fig3, ax3 = plt.subplots()
    sns.barplot(x="Importance", y="Feature", data=imp_df, ax=ax3, palette="viridis")
    ax3.set_title("Top 10 Feature Importances (XGBoost)")
    st.pyplot(fig3)

    
    # Time-Series Forecasting
    st.write("### 📈 Price Forecasting (ARIMA)")

    # Convert YearMonth to datetime
    ts_df = df.copy()
    ts_df["YearMonth"] = pd.to_datetime(ts_df["YearMonth"])
    ts_df = ts_df.sort_values("YearMonth")

    # Fit ARIMA model
    model_arima = ARIMA(ts_df["Avg_Price_USD"], order=(2,1,2))
    model_fit = model_arima.fit()

    forecast = model_fit.forecast(steps=12)
    forecast_index = pd.date_range(ts_df["YearMonth"].iloc[-1] + pd.offsets.MonthBegin(1), periods=12, freq="M")

    fig4, ax4 = plt.subplots(figsize=(10,5))
    ax4.plot(ts_df["YearMonth"], ts_df["Avg_Price_USD"], label="Historical", color="blue")
    ax4.plot(forecast_index, forecast, label="Forecast", color="red", linestyle="--", marker="o")
    ax4.set_title("Tesla Average Price Forecast (Next 12 Months)")
    ax4.set_xlabel("Date")
    ax4.set_ylabel("Avg Price (USD)")
    ax4.legend()
    st.pyplot(fig4)
