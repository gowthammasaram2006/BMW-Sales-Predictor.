# app.py
import os
import io
import joblib
import pandas as pd
import matplotlib
matplotlib.use('Agg')   # use non-interactive backend
import matplotlib.pyplot as plt
from flask import Flask, request, jsonify, render_template, send_file, redirect, url_for

BUNDLE_PATH = "bmw_pipeline.pkl"
if not os.path.exists(BUNDLE_PATH):
    raise FileNotFoundError(f"{BUNDLE_PATH} not found. Run train_model.py first.")

bundle = joblib.load(BUNDLE_PATH)
pipeline = bundle["pipeline"]
features = bundle["features"]
base_year = bundle["base_year"]
model_trend = bundle["model_trend"]
max_known_year = bundle["max_known_year"]
metrics = {"mse": bundle.get("mse"), "rmse": bundle.get("rmse"), "r2_score": bundle.get("r2_score")}
unique_vals = bundle.get("unique_vals", {})
history_by_model = bundle.get("history_by_model", {})

app = Flask(__name__, template_folder="templates", static_folder="static")

@app.route("/")
def index():
    # render index with dropdown options
    return render_template("index.html", unique_vals=unique_vals)

def iterative_forecast(model_name, fuel, trans, start_year, years, start_last_year_sales):
    series = []
    current_last = float(start_last_year_sales)
    for i in range(years):
        y = int(start_year + i)
        yrs_since = float(y - base_year)
        row = {
            "Model": model_name,
            "Fuel_Type": fuel,
            "Transmission": trans,
            "Years_Since_Base": yrs_since,
            "Last_Year_Sales": current_last
        }
        df_row = pd.DataFrame([row], columns=features)
        pred = pipeline.predict(df_row)[0]
        series.append({"year": int(y), "prediction": float(pred)})
        current_last = float(pred)
    return series

@app.route("/result", methods=["POST"])
def result():
    form = request.form
    model_name = form.get("Model") or ""
    fuel = form.get("Fuel_Type") or ""
    trans = form.get("Transmission") or ""
    start_year = int(form.get("Year"))
    horizon = int(form.get("Horizon", 10))

    # initial last year sales fallback
    last_year_sales = form.get("Last_Year_Sales", None)
    try:
        last_year_sales = float(last_year_sales) if last_year_sales not in (None, "") else None
    except Exception:
        last_year_sales = None
    if last_year_sales is None:
        last_year_sales = float(max(0.0, model_trend.get(model_name, 0.0)))

    # compute forecast series
    series = iterative_forecast(model_name, fuel, trans, start_year, horizon, last_year_sales)

    # determine predicted value for chosen year (first of series)
    pred_for_start = series[0]["prediction"] if series else None

    # prepare historical series if available
    history = history_by_model.get(model_name, [])
    # build plot (matplotlib) combining history + predicted series
    fig, ax = plt.subplots(figsize=(10,5))
    # historical line
    if history:
        years_hist = [h["year"] for h in history]
        vals_hist = [h["avg"] for h in history]
        ax.plot(years_hist, vals_hist, '-o', color='#1f77b4', label='Historical (avg per year)', linewidth=2, markersize=6)
    # predicted dashed
    years_pred = [s["year"] for s in series]
    vals_pred = [s["prediction"] for s in series]
    ax.plot(years_pred, vals_pred, '--o', color='#ff7f0e', label='Predicted', linewidth=1.6, markersize=5)
    # highlight selected prediction year (first)
    if series:
        ax.scatter([series[0]["year"]], [series[0]["prediction"]], s=100, color='#1f77b4', zorder=5, label=f'Prediction {series[0]["year"]}')
    # labels, grid, legend, title
    ax.set_xlabel("Year", fontsize=12)
    ax.set_ylabel("Sales Volume", fontsize=12)
    ax.set_title(f"{model_name} — Sales Trend (historical + predicted)", fontsize=14)
    ax.grid(axis='y', linestyle='--', alpha=0.4)
    ax.legend()
    # adjust limits a bit
    ymin = min((min(vals_hist) if history else float(min(vals_pred))) * 0.95, min(vals_pred)*0.95)
    ymax = max((max(vals_hist) if history else max(vals_pred)) * 1.05, max(vals_pred)*1.05)
    ax.set_ylim(ymin, ymax)

    # save plot to bytes buffer (PNG)
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)

    # store image in-memory and return template with embedded image (data URI) OR save to static file
    img_data = buf.getvalue()
    import base64
    img_b64 = base64.b64encode(img_data).decode('utf8')
    img_url = "data:image/png;base64," + img_b64

    # render result template
    return render_template("result.html",
                           model=model_name, fuel=fuel, transmission=trans,
                           pred_year=start_year, predicted=int(round(pred_for_start)) if pred_for_start is not None else None,
                           mse=metrics.get("mse"), rmse=metrics.get("rmse"), r2=metrics.get("r2_score"),
                           img_url=img_url)

if __name__ == "__main__":
    app.run(debug=True, port=5000)
