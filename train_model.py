# train_model.py
import os
import joblib
import numpy as np
import pandas as pd
import inspect

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Use uploaded CSV path (developer instruction)
CSV_PATH = "BMW sales data.csv"
PIPELINE_OUT = "bmw_pipeline.pkl"
BASE_YEAR = 2010

if not os.path.exists(CSV_PATH):
    raise FileNotFoundError(f"{CSV_PATH} not found. Put the CSV there or change CSV_PATH.")

print("Loading CSV:", CSV_PATH)
df = pd.read_csv(CSV_PATH)
df.columns = df.columns.str.strip()

# Normalize string columns
for c in df.select_dtypes(include=["object"]).columns:
    df[c] = df[c].astype(str).str.strip()

# Ensure Year and target exist
df["Year"] = pd.to_numeric(df.get("Year"), errors="coerce")
if "Sales_Volume" not in df.columns:
    raise RuntimeError("CSV must contain 'Sales_Volume' as the target column.")
df = df.dropna(subset=["Year", "Sales_Volume"])
df["Year"] = df["Year"].astype(int)

# Feature engineering
df["Years_Since_Base"] = df["Year"] - BASE_YEAR

# Compute per-model linear trend (slope)
model_trend = {}
for model_name, grp in df.groupby("Model"):
    if grp["Year"].nunique() > 1:
        model_trend[model_name] = float(np.polyfit(grp["Year"].astype(float), grp["Sales_Volume"].astype(float), 1)[0])
    else:
        model_trend[model_name] = 0.0

# Last year sales (shift)
df = df.sort_values(["Model", "Year"]).reset_index(drop=True)
df["Last_Year_Sales"] = df.groupby("Model")["Sales_Volume"].shift(1)
df["Last_Year_Sales"] = df.groupby("Model")["Last_Year_Sales"].transform(lambda s: s.fillna(s.mean()))
df["Last_Year_Sales"] = df["Last_Year_Sales"].fillna(0.0)

max_known_year = int(df["Year"].max())

# Compute historical averages per model for plotting
history_by_model = {}
for model_name, grp in df.groupby("Model"):
    g = grp.groupby("Year")["Sales_Volume"].mean().reset_index().sort_values("Year")
    history = [{"year": int(row.Year), "avg": float(row.Sales_Volume)} for row in g.itertuples(index=False)]
    history_by_model[model_name] = history

# Features used by pipeline
features = ["Model", "Fuel_Type", "Transmission", "Years_Since_Base", "Last_Year_Sales"]
target = "Sales_Volume"

missing_cols = [c for c in ["Model", "Fuel_Type", "Transmission"] if c not in df.columns]
if missing_cols:
    raise RuntimeError(f"Missing required columns in CSV: {missing_cols}")

X = df[features]
y = df[target]

categorical = ["Model", "Fuel_Type", "Transmission"]
numeric = ["Years_Since_Base", "Last_Year_Sales"]

# Build OneHotEncoder robustly
ohe_init_sig = inspect.signature(OneHotEncoder.__init__)
if "sparse_output" in ohe_init_sig.parameters:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
elif "sparse" in ohe_init_sig.parameters:
    ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)
else:
    ohe = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), numeric),
    ("cat", ohe, categorical)
], remainder="drop")

estimator = RandomForestRegressor(n_estimators=300, max_depth=12, min_samples_leaf=2, random_state=42, n_jobs=-1)

pipeline = Pipeline([("preprocessor", preprocessor), ("model", estimator)])

# Train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Training model (may take a minute)...")
pipeline.fit(X_train, y_train)

y_pred = pipeline.predict(X_test)
mse = float(mean_squared_error(y_test, y_pred))
rmse = float(np.sqrt(mse))
r2 = float(r2_score(y_test, y_pred))
print(f"Training finished. RMSE={rmse:.2f}, R2={r2:.4f}")

# Unique lists for dropdowns
unique_vals = {}
for c in ["Model", "Fuel_Type", "Transmission"]:
    unique_vals[c] = sorted(df[c].dropna().astype(str).unique().tolist())

# Save bundle including history_by_model
joblib.dump({
    "pipeline": pipeline,
    "features": features,
    "base_year": BASE_YEAR,
    "model_trend": model_trend,
    "max_known_year": max_known_year,
    "mse": mse, "rmse": rmse, "r2_score": r2,
    "unique_vals": unique_vals,
    "history_by_model": history_by_model
}, PIPELINE_OUT)

print("Saved pipeline and metadata to", PIPELINE_OUT)
