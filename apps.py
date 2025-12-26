"""
Bandwidth Allocation Dashboard for 5G QoS Monitoring
Complete Streamlit Application for FYP
"""

import io
from math import sqrt
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from matplotlib.dates import AutoDateLocator, DateFormatter

# Statistical models
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler

# For Excel export
from io import BytesIO

# For PDF export
from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Table, TableStyle, Paragraph, Spacer
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle


def load_data(uploaded_file) -> pd.DataFrame:
    """
    Securely load and validate CSV input.
    - Enforces size limits
    - Validates required columns
    - Sanitizes strings
    - Enforces numeric types
    """

    # ---- 1. File size check (100 MB limit) ----
    uploaded_file.seek(0, io.SEEK_END)
    file_size = uploaded_file.tell()
    uploaded_file.seek(0)

    if file_size > 100 * 1024 * 1024:
        raise ValueError("CSV file is too large (max 100 MB).")

    # ---- 2. Read CSV with strict options ----
    try:
        df = pd.read_csv(
            uploaded_file,
            encoding="utf-8",
            sep=",",
            engine="python",
            on_bad_lines="skip"
        )
    except Exception as e:
        raise ValueError(f"Invalid CSV format: {e}")

    # ---- 3. Required column validation ----
    required_columns = {"Timestamp", "Required_Bandwidth", "Allocated_Bandwidth"}
    missing = required_columns - set(df.columns)

    if missing:
        raise ValueError(f"Missing required columns: {', '.join(missing)}")

    # ---- 4. Drop unnamed / junk columns ----
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed", regex=True)]

    # ---- 5. Timestamp parsing ----
    df["Timestamp"] = pd.to_datetime(
        df["Timestamp"],
        errors="coerce",
        infer_datetime_format=True
    )
    df = df.dropna(subset=["Timestamp"])

    # ---- 6. Enforce numeric columns ----
    numeric_columns = [
        "Required_Bandwidth",
        "Allocated_Bandwidth",
        "Signal_Strength",
        "Latency"
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # ---- 7. CSV formula injection protection ----
    # Prevent Excel formula execution (=, +, -, @)
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = df[col].astype(str).str.replace(
            r"^[=+\-@]", "'", regex=True
        )

    # ---- 8. Sort and index ----
    df = df.sort_values("Timestamp").set_index("Timestamp")

    if df.empty:
        raise ValueError("CSV contains no valid data after validation.")

    return df


def prepare_time_series(df: pd.DataFrame, target_col: str = "Required_Bandwidth") -> pd.Series:
    """Return hourly time series of target column."""
    if target_col not in df.columns:
        raise ValueError(f"Column '{target_col}' not found in data.")
    
    df = df.loc[:, ~df.columns.str.contains(r"^Unnamed")]
    ts = df[target_col].copy()
    ts = ts.interpolate()
    ts_hourly = ts.resample("h").mean().interpolate()
    
    return ts_hourly


def train_test_split_series(ts: pd.Series, test_days: int = 7):
    """Split hourly series into train and test."""
    points_per_day = 24
    horizon = points_per_day * test_days
    
    if len(ts) <= horizon:
        raise ValueError("Time series is too short relative to test period.")
    
    train = ts.iloc[:-horizon]
    test = ts.iloc[-horizon:]
    
    return train, test


def eval_metrics(true, pred):
    """Calculate evaluation metrics."""
    true = np.array(true)
    pred = np.array(pred)
    
    rmse = sqrt(mean_squared_error(true, pred))
    mae = mean_absolute_error(true, pred)
    r2 = r2_score(true, pred)
    mape = np.mean(np.abs((true - pred) / (true + 1e-6))) * 100
    
    return rmse, mae, mape, r2


def format_date(dt):
    """Safely format datetime-like objects as DD/MM/YYYY HH:MM."""
    if dt is None or pd.isna(dt):
        return "N/A"

    # Convert numpy / pandas datetime safely
    try:
        dt = pd.to_datetime(dt)
    except Exception:
        return str(dt)

    return dt.strftime("%d/%m/%Y %H:%M")


def run_arima(train, test):
    """Run ARIMA model."""
    try:
        model = ARIMA(train, order=(2, 1, 2))
        res = model.fit()
        forecast = res.forecast(steps=len(test))
        rmse, mae, mape, r2 = eval_metrics(test.values, forecast.values)
        return pd.Series(forecast.values, index=test.index), {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}
    except Exception as e:
        st.warning(f"ARIMA failed: {e}")
        forecast = np.full(len(test), train.mean())
        rmse, mae, mape, r2 = eval_metrics(test.values, forecast)
        return pd.Series(forecast, index=test.index), {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def run_sarima(train, test, max_train_points=2000):
    """Run SARIMA model with seasonal component."""
    train_sub = train if len(train) <= max_train_points else train.iloc[-max_train_points:]
    
    try:
        seasonal_period = 24
        model = SARIMAX(
            train_sub,
            order=(1, 1, 1),
            seasonal_order=(1, 0, 1, seasonal_period),
            enforce_stationarity=False,
            enforce_invertibility=False,
        )
        res = model.fit(disp=False)
        forecast_vals = res.forecast(steps=len(test))
        forecast_series = pd.Series(forecast_vals.values, index=test.index)
        rmse, mae, mape, r2 = eval_metrics(test.values, forecast_series.values)
        return forecast_series, {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}
    except Exception as e:
        st.warning(f"SARIMA failed ({e}). Falling back to ARIMA.")
        return run_arima(train, test)


def run_prophet(train, test):
    """Run Prophet model."""
    try:
        from prophet import Prophet
        
        df_train = train.reset_index()
        df_train.columns = ["ds", "y"]
        
        m = Prophet(daily_seasonality=True, weekly_seasonality=True, yearly_seasonality=False)
        m.fit(df_train)
        
        future = m.make_future_dataframe(periods=len(test), freq="h")
        forecast_full = m.predict(future)
        forecast = forecast_full.iloc[-len(test):]["yhat"].values
        
        rmse, mae, mape, r2 = eval_metrics(test.values, forecast)
        forecast_series = pd.Series(forecast, index=test.index)
        return forecast_series, {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}
    except ImportError:
        st.warning("Prophet not installed. Using exponential smoothing instead.")
        return run_exponential_smoothing(train, test, "Prophet")
    except Exception as e:
        st.warning(f"Prophet failed: {e}. Using fallback.")
        return run_exponential_smoothing(train, test, "Prophet")


def run_lstm(train, test, lookback=24):
    """Run LSTM model."""
    try:
        from tensorflow.keras.models import Sequential
        from tensorflow.keras.layers import LSTM, Dense
        from tensorflow.keras.callbacks import EarlyStopping
        import warnings
        warnings.filterwarnings('ignore')
        
        scaler = MinMaxScaler()
        train_scaled = scaler.fit_transform(train.values.reshape(-1, 1))
        test_scaled = scaler.transform(test.values.reshape(-1, 1))
        
        def create_sequences(data, lookback_):
            X, y = [], []
            for i in range(len(data) - lookback_):
                X.append(data[i:i + lookback_])
                y.append(data[i + lookback_])
            return np.array(X), np.array(y)
        
        X_train, y_train = create_sequences(train_scaled, lookback)
        
        model = Sequential()
        model.add(LSTM(64, input_shape=(lookback, 1)))
        model.add(Dense(1))
        model.compile(optimizer="adam", loss="mse")
        
        es = EarlyStopping(patience=5, restore_best_weights=True)
        
        model.fit(
            X_train, y_train,
            epochs=30,
            batch_size=32,
            validation_split=0.1,
            callbacks=[es],
            verbose=0,
        )
        
        combined = np.concatenate([train_scaled[-lookback:], test_scaled], axis=0)
        X_test, y_test = create_sequences(combined, lookback)
        
        y_pred_scaled = model.predict(X_test, verbose=0)
        y_pred = scaler.inverse_transform(y_pred_scaled).flatten()
        
        forecast_series = pd.Series(y_pred[-len(test):], index=test.index)
        rmse, mae, mape, r2 = eval_metrics(test.values[-len(y_pred):], y_pred[-len(test):])
        
        return forecast_series, {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}
    except ImportError:
        st.warning("TensorFlow not installed. Using Holt's method instead.")
        return run_holts_method(train, test, "LSTM")
    except Exception as e:
        st.warning(f"LSTM failed: {e}. Using fallback.")
        return run_holts_method(train, test, "LSTM")


def run_exponential_smoothing(train, test, model_name):
    """Fallback exponential smoothing method."""
    alpha = 0.3
    forecast_values = []
    last_value = train.iloc[-1]
    
    for i in range(len(test)):
        if i == 0:
            forecast_values.append(last_value)
        else:
            forecast_values.append(alpha * test.iloc[i-1] + (1 - alpha) * forecast_values[-1])
    
    forecast = np.array(forecast_values)
    rmse, mae, mape, r2 = eval_metrics(test.values, forecast)
    
    return pd.Series(forecast, index=test.index), {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def run_holts_method(train, test, model_name):
    """Holt's linear trend method as LSTM fallback."""
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    
    try:
        model = ExponentialSmoothing(train, trend='add', seasonal=None)
        res = model.fit()
        forecast = res.forecast(len(test))
        rmse, mae, mape, r2 = eval_metrics(test.values, forecast.values)
        return pd.Series(forecast.values, index=test.index), {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}
    except:
        forecast = np.full(len(test), train.mean())
        rmse, mae, mape, r2 = eval_metrics(test.values, forecast)
        return pd.Series(forecast, index=test.index), {"RMSE": rmse, "MAE": mae, "MAPE": mape, "R2": r2}


def build_summary_excel(df, best_model_name, all_metrics, best_forecast, test, risk_df, anomalies_df, stats):
    """Create comprehensive Excel report."""
    output = BytesIO()
    
    with pd.ExcelWriter(output, engine="openpyxl") as writer:
        summary_data = {
            "Metric": [
                "Total Records", "Date Range Start", "Date Range End", "Total Days",
                "Average Signal Strength (dBm)", "Average Latency (ms)",
                "Average Required Bandwidth (Mbps)", "Average Allocated Bandwidth (Mbps)",
                "Congestion Events", "Peak Bandwidth (Mbps)",
            ],
            "Value": [
                stats.get("total_readings", 0),
                format_date(df.index.min()) if len(df) > 0 else "N/A",
                format_date(df.index.max()) if len(df) > 0 else "N/A",
                (df.index.max() - df.index.min()).days if len(df) > 0 else 0,
                f"{stats.get('avg_signal', 0):.2f}",
                f"{stats.get('avg_latency', 0):.2f}",
                f"{stats.get('avg_bandwidth', 0):.2f}",
                f"{stats.get('avg_allocated', 0):.2f}",
                stats.get("congestion_events", 0),
                f"{stats.get('peak_bandwidth', 0):.2f}",
            ]
        }
        pd.DataFrame(summary_data).to_excel(writer, sheet_name="Dataset_Summary", index=False)
        
        metrics_df = pd.DataFrame(all_metrics).T
        metrics_df.index.name = "Model"
        metrics_df.to_excel(writer, sheet_name="Model_Comparison")
        
        forecast_df = pd.DataFrame({
            "Timestamp": [format_date(t) for t in test.index],
            "Actual_Required_Mbps": test.values,
            "Forecast_Required_Mbps": best_forecast.reindex(test.index).values,
            "Difference": test.values - best_forecast.reindex(test.index).values,
        })
        forecast_df.to_excel(writer, sheet_name="Forecast_vs_Actual", index=False)
        
        if risk_df is not None and not risk_df.empty:
            risk_export = risk_df.copy()
            if "Time" in risk_export.columns:
                risk_export["Time"] = risk_export["Time"].apply(lambda x: format_date(x) if hasattr(x, 'strftime') else str(x))
            risk_export.to_excel(writer, sheet_name="Congestion_Risk", index=False)
        
        if anomalies_df is not None and not anomalies_df.empty:
            anomalies_df.to_excel(writer, sheet_name="Anomalies", index=False)
        
        peak_bw = float(best_forecast.max())
        recommended = peak_bw * 1.2
        recommendations = {
            "Category": [
                "Best Forecasting Model", "Model RMSE", "Model MAE", "Model MAPE", "Model R²",
                "Peak Forecast Bandwidth (Mbps)", "Recommended Capacity (Mbps)", "Capacity Buffer",
                "High Risk Hours", "Medium Risk Hours", "Immediate Action Required",
            ],
            "Value": [
                best_model_name,
                f"{all_metrics[best_model_name]['RMSE']:.4f}",
                f"{all_metrics[best_model_name]['MAE']:.4f}",
                f"{all_metrics[best_model_name]['MAPE']:.2f}%",
                f"{all_metrics[best_model_name]['R2']:.4f}",
                f"{peak_bw:.2f}",
                f"{recommended:.2f}",
                "20% above peak",
                len(risk_df[risk_df["Risk_Level"] == "High"]) if risk_df is not None else 0,
                len(risk_df[risk_df["Risk_Level"] == "Medium"]) if risk_df is not None else 0,
                "Yes" if (risk_df is not None and len(risk_df[risk_df["Risk_Level"] == "High"]) > 0) else "No",
            ]
        }
        pd.DataFrame(recommendations).to_excel(writer, sheet_name="Recommendations", index=False)
    
    return output.getvalue()


def build_summary_pdf(df, location_name, best_model_name, all_metrics, peak_bw, peak_time, recommended_capacity, risk_df, stats):
    """Create comprehensive PDF report."""
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    
    y = height - 50
    
    c.setFont("Helvetica-Bold", 18)
    c.drawString(40, y, "5G QoS Bandwidth Forecast Report")
    y -= 30
    
    c.setFont("Helvetica", 10)
    c.drawString(40, y, f"Generated: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
    y -= 25
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "1. Dataset Summary")
    y -= 20
    
    c.setFont("Helvetica", 10)
    if location_name:
        c.drawString(50, y, f"Location: {location_name}")
        y -= 15
    
    c.drawString(50, y, f"Total Records: {stats.get('total_readings', 0):,}")
    y -= 15
    c.drawString(50, y, f"Date Range: {format_date(df.index.min())} to {format_date(df.index.max())}")
    y -= 15
    c.drawString(50, y, f"Average Signal Strength: {stats.get('avg_signal', 0):.2f} dBm")
    y -= 15
    c.drawString(50, y, f"Average Latency: {stats.get('avg_latency', 0):.2f} ms")
    y -= 15
    c.drawString(50, y, f"Average Required Bandwidth: {stats.get('avg_bandwidth', 0):.2f} Mbps")
    y -= 15
    c.drawString(50, y, f"Congestion Events: {stats.get('congestion_events', 0)}")
    y -= 30
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "2. Model Performance Comparison")
    y -= 20
    
    c.setFont("Helvetica-Bold", 10)
    c.drawString(50, y, f"Best Model Selected: {best_model_name}")
    y -= 20
    
    c.setFont("Helvetica", 10)
    for model_name, metrics in all_metrics.items():
        marker = "→ " if model_name == best_model_name else "  "
        c.drawString(50, y, f"{marker}{model_name}: RMSE={metrics['RMSE']:.4f}, MAE={metrics['MAE']:.4f}, MAPE={metrics['MAPE']:.2f}%, R²={metrics['R2']:.4f}")
        y -= 15
    
    y -= 15
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "3. Capacity Recommendations")
    y -= 20
    
    c.setFont("Helvetica", 10)
    c.drawString(50, y, f"Peak Forecast Bandwidth: {peak_bw:.2f} Mbps")
    y -= 15
    c.drawString(50, y, f"Time of Peak: {format_date(peak_time)}")
    y -= 15
    c.drawString(50, y, f"Recommended Capacity (Peak + 20%): {recommended_capacity:.2f} Mbps")
    y -= 30
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "4. Congestion Risk Summary")
    y -= 20
    
    c.setFont("Helvetica", 10)
    if risk_df is not None and not risk_df.empty:
        high_risk = len(risk_df[risk_df["Risk_Level"] == "High"])
        medium_risk = len(risk_df[risk_df["Risk_Level"] == "Medium"])
        low_risk = len(risk_df[risk_df["Risk_Level"] == "Low"])
        
        c.drawString(50, y, f"High Risk Hours: {high_risk}")
        y -= 15
        c.drawString(50, y, f"Medium Risk Hours: {medium_risk}")
        y -= 15
        c.drawString(50, y, f"Low Risk Hours: {low_risk}")
        y -= 25
        
        if high_risk > 0:
            c.setFont("Helvetica-Bold", 10)
            c.drawString(50, y, "Critical Hours Requiring Immediate Attention:")
            y -= 15
            c.setFont("Helvetica", 9)
            
            high_risk_rows = risk_df[risk_df["Risk_Level"] == "High"].head(10)
            for _, row in high_risk_rows.iterrows():
                time_str = format_date(row["Time"]) if hasattr(row["Time"], 'strftime') else str(row["Time"])
                c.drawString(60, y, f"• {time_str}: Gap = {row['Gap_Mbps']:.2f} Mbps")
                y -= 12
                if y < 100:
                    c.showPage()
                    y = height - 50
    else:
        c.drawString(50, y, "No congestion risk detected in the forecast period.")
    
    y -= 25
    
    if y < 200:
        c.showPage()
        y = height - 50
    
    c.setFont("Helvetica-Bold", 14)
    c.drawString(40, y, "5. Interpretation Guide")
    y -= 20
    
    c.setFont("Helvetica", 9)
    interpretations = [
        "• RMSE (Root Mean Square Error): Lower values indicate better prediction accuracy.",
        "• MAE (Mean Absolute Error): Average prediction error in Mbps.",
        "• MAPE (Mean Absolute Percentage Error): Prediction error as a percentage.",
        "• R² (R-Squared): Values closer to 1 indicate better model fit.",
        "• High Risk: Forecast demand exceeds allocated bandwidth by >15 Mbps.",
        "• Medium Risk: Gap between 5-15 Mbps.",
        "• Low Risk: Gap less than 5 Mbps.",
        "• Recommended Capacity includes 20% buffer above peak demand."
    ]
    
    for text in interpretations:
        c.drawString(50, y, text)
        y -= 12
    
    c.save()
    return buffer.getvalue()


st.set_page_config(page_title="5G Bandwidth Forecast Dashboard", page_icon="📡", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #0f172a; }
    [data-testid="stSidebar"] { background-color: #1e293b !important; }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    h1, h2, h3 { color: #f8fafc !important; }
    [data-testid="metric-container"] { background-color: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 15px; }
    [data-testid="stMetricValue"] { color: #3b82f6 !important; }
    [data-testid="stMetricLabel"] { color: #94a3b8 !important; }
    .interpretation-box { background-color: #1e293b; border-left: 4px solid #3b82f6; padding: 15px; border-radius: 0 8px 8px 0; margin: 10px 0; color: #e2e8f0; }
    .section-title { font-size: 1.3rem; font-weight: 600; color: #f8fafc; margin-top: 30px; margin-bottom: 15px; padding-bottom: 10px; border-bottom: 2px solid #3b82f6; }
    .stSuccess { background-color: #065f46 !important; color: #d1fae5 !important; }
    .stWarning { background-color: #78350f !important; color: #fef3c7 !important; }
    .stError { background-color: #7f1d1d !important; color: #fecaca !important; }
    .stDataFrame { background-color: #1e293b; }
    [data-testid="stFileUploader"] { background-color: #1e293b; border: 2px dashed #475569; border-radius: 8px; padding: 20px; }
    .high-risk { color: #ef4444; font-weight: bold; }
    .medium-risk { color: #f59e0b; font-weight: bold; }
    .low-risk { color: #22c55e; font-weight: bold; }
</style>
""", unsafe_allow_html=True)


def main():
    st.title("📡 5G QoS Bandwidth Forecast Dashboard")
    st.markdown("*ISP Network Capacity Planning & Congestion Risk Analysis*")
    
    st.sidebar.header("⚙️ Settings")
    st.sidebar.markdown("### 📁 Upload Your Dataset")
    st.sidebar.markdown("""
    **Required CSV Format:**
    - `Timestamp` (M/D/YYYY H:MM format)
    - `Required_Bandwidth` (Mbps)
    - `Allocated_Bandwidth` (Mbps)
    - `Signal_Strength` (dBm) - optional
    - `Latency` (ms) - optional
    - `Location` - optional
    """)
    
    uploaded_file = st.sidebar.file_uploader("Upload QoS CSV file", type=["csv"], help="Upload your ISP's QoS monitoring data in CSV format")
    test_days = st.sidebar.slider("Forecast Horizon (days)", min_value=3, max_value=14, value=7, help="Number of days to use for testing/forecasting")
    
    if uploaded_file is None:
        st.info("👈 Please upload a CSV file using the sidebar to begin analysis.")
        st.markdown("### 📋 Expected CSV Format Example")
        example_df = pd.DataFrame({
            "Timestamp": ["1/1/2025 0:00", "1/1/2025 0:01", "1/1/2025 0:02"],
            "Required_Bandwidth": [45.2, 47.8, 43.1],
            "Allocated_Bandwidth": [50.0, 50.0, 50.0],
            "Signal_Strength": [-65.3, -64.8, -66.1],
            "Latency": [12.5, 11.8, 13.2],
            "Location": ["Area_A", "Area_A", "Area_A"]
        })
        st.dataframe(example_df)
        return
    
    try:
        df = load_data(uploaded_file)
        st.sidebar.success(f"✅ Loaded {len(df):,} records")
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return
    
    selected_location = None
    if "Location" in df.columns:
        locations = sorted(df["Location"].dropna().unique().tolist())
        if locations:
            selected_location = st.sidebar.selectbox("Select Location", locations)
            df = df[df["Location"] == selected_location]
            st.sidebar.info(f"📍 Filtering: {selected_location}")
    
    stats = {
        "total_readings": len(df),
        "avg_signal": df["Signal_Strength"].mean() if "Signal_Strength" in df.columns else 0,
        "avg_latency": df["Latency"].mean() if "Latency" in df.columns else 0,
        "avg_bandwidth": df["Required_Bandwidth"].mean() if "Required_Bandwidth" in df.columns else 0,
        "avg_allocated": df["Allocated_Bandwidth"].mean() if "Allocated_Bandwidth" in df.columns else 0,
        "peak_bandwidth": df["Required_Bandwidth"].max() if "Required_Bandwidth" in df.columns else 0,
        "congestion_events": 0,
    }
    
    if "Required_Bandwidth" in df.columns and "Allocated_Bandwidth" in df.columns:
        stats["congestion_events"] = int((df["Required_Bandwidth"] > df["Allocated_Bandwidth"]).sum())
    
    st.markdown('<div class="section-title">📊 A. Dataset Summary Statistics</div>', unsafe_allow_html=True)
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Readings", f"{stats['total_readings']:,}")
    col2.metric("Avg Signal Strength", f"{stats['avg_signal']:.1f} dBm")
    col3.metric("Avg Latency", f"{stats['avg_latency']:.1f} ms")
    col4.metric("Congestion Events", f"{stats['congestion_events']:,}")
    
    st.markdown("""
    <div class="interpretation-box">
    <strong>📖 Interpretation:</strong><br>
    • <strong>Total Readings:</strong> Number of data points in your dataset<br>
    • <strong>Signal Strength:</strong> Higher values (closer to 0) indicate better signal quality<br>
    • <strong>Latency:</strong> Lower values indicate faster response times<br>
    • <strong>Congestion Events:</strong> Times when required bandwidth exceeded allocated bandwidth
    </div>
    """, unsafe_allow_html=True)
    
    try:
        ts_hourly = prepare_time_series(df, target_col="Required_Bandwidth")
        train, test = train_test_split_series(ts_hourly, test_days=test_days)
    except Exception as e:
        st.error(f"Error preparing time series: {e}")
        return
    
    st.write(f"**Data Range:** {format_date(df.index.min())} → {format_date(df.index.max())}")
    st.write(f"**Train Period:** {format_date(train.index[0])} → {format_date(train.index[-1])}")
    st.write(f"**Test Period:** {format_date(test.index[0])} → {format_date(test.index[-1])}")
    
    st.markdown('<div class="section-title">🤖 B. Model Performance Comparison</div>', unsafe_allow_html=True)
    
    with st.spinner("Training forecasting models... This may take a few minutes."):
        arima_forecast, arima_metrics = run_arima(train, test)
        sarima_forecast, sarima_metrics = run_sarima(train, test)
        prophet_forecast, prophet_metrics = run_prophet(train, test)
        lstm_forecast, lstm_metrics = run_lstm(train, test)
    
    all_metrics = {"ARIMA": arima_metrics, "SARIMA": sarima_metrics, "Prophet": prophet_metrics, "LSTM": lstm_metrics}
    forecasts = {"ARIMA": arima_forecast, "SARIMA": sarima_forecast, "Prophet": prophet_forecast, "LSTM": lstm_forecast}
    
    best_model_name = min(all_metrics.keys(), key=lambda m: all_metrics[m]["RMSE"])
    best_metrics = all_metrics[best_model_name]
    best_forecast = forecasts[best_model_name]
    
    metrics_df = pd.DataFrame(all_metrics).T.round(4)
    st.dataframe(metrics_df.style.highlight_min(axis=0, subset=["RMSE", "MAE", "MAPE"]).highlight_max(axis=0, subset=["R2"]))
    
    st.success(f"✅ Best Model Selected: **{best_model_name}** (Lowest RMSE: {best_metrics['RMSE']:.4f})")
    
    st.markdown("""
    <div class="interpretation-box">
    <strong>📖 Model Metrics Interpretation:</strong><br>
    • <strong>RMSE (Root Mean Square Error):</strong> Measures prediction accuracy. Lower = better. Penalizes large errors.<br>
    • <strong>MAE (Mean Absolute Error):</strong> Average prediction error in Mbps. Lower = better.<br>
    • <strong>MAPE (Mean Absolute Percentage Error):</strong> Error as percentage. Lower = better. <10% is excellent.<br>
    • <strong>R² (R-Squared):</strong> Model fit quality. Closer to 1 = better. >0.8 is good.
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">📈 C. Bandwidth Allocation vs. Demand</div>', unsafe_allow_html=True)
    
    display_hours = min(168, len(ts_hourly))
    display_data = ts_hourly.iloc[-display_hours:]
    
    alloc_ts_hourly = None
    if "Allocated_Bandwidth" in df.columns:
        alloc_ts = df["Allocated_Bandwidth"].resample("h").mean()
        alloc_ts_hourly = alloc_ts.reindex(ts_hourly.index).interpolate()
        alloc_display = alloc_ts_hourly.iloc[-display_hours:]
    
    fig1, ax1 = plt.subplots(figsize=(14, 5))
    ax1.fill_between(display_data.index, 0, display_data.values, alpha=0.5, label="Required Bandwidth", color="#3b82f6")
    if alloc_ts_hourly is not None:
        ax1.plot(alloc_display.index, alloc_display.values, label="Allocated Bandwidth", color="#22c55e", linewidth=2)
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Bandwidth (Mbps)")
    ax1.set_title("Bandwidth Allocation vs. Demand (Last 7 Days)")
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    locator = AutoDateLocator()
    formatter = DateFormatter("%d/%m\n%H:%M")
    ax1.xaxis.set_major_locator(locator)
    ax1.xaxis.set_major_formatter(formatter)
    fig1.autofmt_xdate()
    st.pyplot(fig1)
    
    st.markdown("""
    <div class="interpretation-box">
    <strong>📖 Interpretation:</strong><br>
    • <strong>Blue area:</strong> Required bandwidth demand from users<br>
    • <strong>Green line:</strong> Allocated bandwidth capacity<br>
    • <strong>Gap Analysis:</strong> When blue exceeds green, congestion occurs<br>
    • <strong>Action:</strong> Identify peak demand periods and ensure allocated bandwidth covers them
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">🎯 D. Forecast vs Actual (Best Model)</div>', unsafe_allow_html=True)
    
    history_days = min(3, test_days)
    history_points = history_days * 24
    train_tail = train.iloc[-history_points:]
    
    fig2, ax2 = plt.subplots(figsize=(14, 5))
    ax2.plot(train_tail.index, train_tail.values, label=f"Historical (last {history_days} days)", color="#64748b", linewidth=1.5)
    ax2.plot(test.index, test.values, label="Actual", color="#3b82f6", linewidth=2)
    ax2.plot(best_forecast.index, best_forecast.values, label=f"Forecast ({best_model_name})", color="#ef4444", linewidth=2, linestyle="--")
    
    if alloc_ts_hourly is not None:
        alloc_test = alloc_ts_hourly.reindex(test.index)
        ax2.plot(alloc_test.index, alloc_test.values, label="Allocated", color="#22c55e", linewidth=1.5, linestyle=":")
    
    ax2.axvline(x=test.index[0], color="#f59e0b", linestyle="--", linewidth=1.5, alpha=0.8)
    ax2.set_xlabel("Time")
    ax2.set_ylabel("Required Bandwidth (Mbps)")
    ax2.set_title(f"Forecast vs Actual - {best_model_name}")
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_locator(locator)
    ax2.xaxis.set_major_formatter(formatter)
    fig2.autofmt_xdate()
    st.pyplot(fig2)
    
    col_m1, col_m2, col_m3, col_m4 = st.columns(4)
    col_m1.metric("RMSE", f"{best_metrics['RMSE']:.4f}")
    col_m2.metric("MAE", f"{best_metrics['MAE']:.4f}")
    col_m3.metric("MAPE", f"{best_metrics['MAPE']:.2f}%")
    col_m4.metric("R²", f"{best_metrics['R2']:.4f}")
    
    st.markdown("""
    <div class="interpretation-box">
    <strong>📖 Interpretation:</strong><br>
    • <strong>Blue line (Actual):</strong> Real bandwidth demand during test period<br>
    • <strong>Red dashed line (Forecast):</strong> Model's prediction<br>
    • <strong>Close alignment:</strong> Model is accurately predicting demand patterns<br>
    • <strong>Large gaps:</strong> May indicate unusual events or model limitations
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">⚠️ E. Capacity Recommendation & Congestion Risk Hours</div>', unsafe_allow_html=True)
    
    peak_bw = float(best_forecast.max())
    peak_time = best_forecast.idxmax()
    recommended_capacity = 1.20 * peak_bw
    
    col_c1, col_c2, col_c3 = st.columns(3)
    col_c1.metric("Peak Forecast BW", f"{peak_bw:.2f} Mbps")
    col_c2.metric("Time of Peak", format_date(peak_time))
    col_c3.metric("Recommended Capacity", f"{recommended_capacity:.2f} Mbps")
    
    st.markdown(f"""
    <div class="interpretation-box">
    <strong>📖 Capacity Recommendation:</strong><br>
    • Forecast peak demand: <strong>{peak_bw:.2f} Mbps</strong><br>
    • Recommended minimum capacity (peak + 20% buffer): <strong>{recommended_capacity:.2f} Mbps</strong><br>
    • This buffer helps accommodate unexpected demand spikes and ensures service quality.
    </div>
    """, unsafe_allow_html=True)
    
    risk_df = None
    if alloc_ts_hourly is not None:
        alloc_forecast = alloc_ts_hourly.reindex(best_forecast.index)
        
        risk_df = pd.DataFrame({
            "Time": best_forecast.index,
            "Forecast_Required_Mbps": best_forecast.values,
            "Allocated_Mbps": alloc_forecast.values,
        })
        risk_df["Gap_Mbps"] = risk_df["Forecast_Required_Mbps"] - risk_df["Allocated_Mbps"]
        
        def label_risk(gap):
            if pd.isna(gap) or gap <= 0:
                return "No Risk"
            if gap < 5:
                return "Low"
            if gap < 15:
                return "Medium"
            return "High"
        
        risk_df["Risk_Level"] = risk_df["Gap_Mbps"].apply(label_risk)
        
        congested = risk_df[risk_df["Gap_Mbps"] > 0].copy()
        
        if not congested.empty:
            fig3, ax3 = plt.subplots(figsize=(14, 4))
            
            colors_map = {"High": "#ef4444", "Medium": "#f59e0b", "Low": "#22c55e"}
            bar_colors = [colors_map.get(r, "#64748b") for r in congested["Risk_Level"]]
            
            x_positions = range(len(congested))
            ax3.bar(x_positions, congested["Gap_Mbps"].values, color=bar_colors)
            
            n = max(1, len(congested) // 20)
            ax3.set_xticks(x_positions[::n])
            ax3.set_xticklabels([format_date(t)[-11:] for t in congested["Time"].values[::n]], rotation=45, ha="right")
            
            ax3.set_ylabel("Bandwidth Gap (Mbps)")
            ax3.set_title("Congestion Risk Hours (Forecast Demand > Allocated)")
            ax3.grid(True, alpha=0.3, axis="y")
            
            from matplotlib.patches import Patch
            legend_elements = [
                Patch(facecolor="#ef4444", label="High Risk (>15 Mbps)"),
                Patch(facecolor="#f59e0b", label="Medium Risk (5-15 Mbps)"),
                Patch(facecolor="#22c55e", label="Low Risk (<5 Mbps)")
            ]
            ax3.legend(handles=legend_elements, loc="upper right")
            
            plt.tight_layout()
            st.pyplot(fig3)
        else:
            st.success("✅ No congestion predicted! Allocated bandwidth is sufficient.")
    
    st.markdown("""
    <div class="interpretation-box">
    <strong>📖 Risk Level Interpretation:</strong><br>
    • <span class="high-risk">High Risk (Red):</span> Gap >15 Mbps - Immediate capacity upgrade strongly recommended<br>
    • <span class="medium-risk">Medium Risk (Orange):</span> Gap 5-15 Mbps - Monitor closely and consider targeted increase<br>
    • <span class="low-risk">Low Risk (Green):</span> Gap <5 Mbps - Manageable but worth monitoring
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">📋 F. All Congested Hours Requiring More Bandwidth</div>', unsafe_allow_html=True)
    
    if risk_df is not None:
        congested_hours = risk_df[risk_df["Gap_Mbps"] > 0].copy()
        
        if congested_hours.empty:
            st.success("✅ No congested hours detected in the forecast period.")
        else:
            st.warning(f"⚠️ {len(congested_hours)} hours with potential congestion detected.")
            
            display_df = congested_hours.copy()
            display_df["Time"] = display_df["Time"].apply(lambda x: format_date(x))
            display_df = display_df.round(2)
            
            def highlight_risk(row):
                if row["Risk_Level"] == "High":
                    return ["background-color: #7f1d1d; color: white"] * len(row)
                elif row["Risk_Level"] == "Medium":
                    return ["background-color: #78350f; color: white"] * len(row)
                elif row["Risk_Level"] == "Low":
                    return ["background-color: #14532d; color: white"] * len(row)
                return [""] * len(row)
            
            st.dataframe(display_df.style.apply(highlight_risk, axis=1), use_container_width=True, height=400)
    
    st.markdown("""
    <div class="interpretation-box">
    <strong>📖 Interpretation:</strong><br>
    • This table shows all hours where forecast demand exceeds allocated bandwidth<br>
    • <strong>Gap_Mbps:</strong> Additional bandwidth needed to meet demand<br>
    • Use this data to plan capacity upgrades during specific time periods
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">🔮 G. What-If Capacity Scenario</div>', unsafe_allow_html=True)
    
    if alloc_ts_hourly is not None:
        capacity_factor = st.slider("Simulate capacity multiplier (× current allocation)", min_value=0.5, max_value=2.0, value=1.0, step=0.1, help="1.0 = current capacity, 1.2 = +20% more bandwidth")
        
        alloc_forecast = alloc_ts_hourly.reindex(best_forecast.index)
        sim_alloc = alloc_forecast * capacity_factor
        
        sim_gap = best_forecast.values - sim_alloc.values
        sim_congested = np.sum(sim_gap > 0)
        
        col_s1, col_s2, col_s3 = st.columns(3)
        col_s1.metric("Capacity Factor", f"×{capacity_factor:.1f}")
        col_s2.metric("Simulated Capacity", f"{(alloc_forecast.mean() * capacity_factor):.2f} Mbps (avg)")
        col_s3.metric("Congested Hours", f"{sim_congested}")
        
        if sim_congested == 0:
            st.success(f"✅ With ×{capacity_factor:.1f} capacity, no congestion expected!")
        else:
            st.warning(f"⚠️ With ×{capacity_factor:.1f} capacity, {sim_congested} hours still congested.")
        
        st.markdown("""
        <div class="interpretation-box">
        <strong>📖 Interpretation:</strong><br>
        • Adjust the slider to simulate different capacity scenarios<br>
        • Find the minimum capacity multiplier that eliminates all congestion<br>
        • Use this to plan infrastructure investments and cost-benefit analysis
        </div>
        """, unsafe_allow_html=True)
    else:
        st.info("Allocated_Bandwidth column not found. Cannot simulate capacity scenarios.")
    
    st.markdown('<div class="section-title">🔍 H. Anomaly Detection (Unexpected Demand Spikes)</div>', unsafe_allow_html=True)
    
    ts_df = ts_hourly.to_frame("Bandwidth")
    ts_df["RollingMean"] = ts_df["Bandwidth"].rolling(window=24, min_periods=6).mean()
    ts_df["RollingStd"] = ts_df["Bandwidth"].rolling(window=24, min_periods=6).std()
    ts_df["Zscore"] = (ts_df["Bandwidth"] - ts_df["RollingMean"]) / (ts_df["RollingStd"] + 1e-6)
    
    anomalies = ts_df[ts_df["Zscore"].abs() > 3].dropna()
    
    if anomalies.empty:
        st.success("✅ No anomalous demand spikes detected.")
        anomalies_df = pd.DataFrame()
    else:
        st.error(f"⚠️ {len(anomalies)} anomalous demand spikes detected!")
        
        anomalies_df = anomalies[["Bandwidth", "Zscore"]].copy()
        anomalies_df["Timestamp"] = [format_date(t) for t in anomalies_df.index]
        anomalies_df = anomalies_df[["Timestamp", "Bandwidth", "Zscore"]]
        anomalies_df.columns = ["Timestamp", "Bandwidth (Mbps)", "Z-Score"]
        
        st.dataframe(anomalies_df.style.format({"Bandwidth (Mbps)": "{:.2f}", "Z-Score": "{:.2f}"}), use_container_width=True)
    
    st.markdown("""
    <div class="interpretation-box">
    <strong>📖 Interpretation:</strong><br>
    • <strong>Z-Score:</strong> Number of standard deviations from the rolling mean<br>
    • <strong>|Z-Score| > 3:</strong> Indicates unusual demand (anomaly)<br>
    • Investigate these timestamps for special events, outages, or data quality issues<br>
    • High positive Z-scores indicate unexpected demand spikes
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-title">📥 I. Download Reports</div>', unsafe_allow_html=True)
    
    col_r1, col_r2 = st.columns(2)
    
    with col_r1:
        excel_bytes = build_summary_excel(
            df=df, best_model_name=best_model_name, all_metrics=all_metrics,
            best_forecast=best_forecast, test=test, risk_df=risk_df,
            anomalies_df=anomalies_df if not anomalies.empty else pd.DataFrame(), stats=stats,
        )
        st.download_button(
            label="📊 Download Excel Report", data=excel_bytes,
            file_name=f"bandwidth_forecast_report_{datetime.now().strftime('%Y%m%d')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        )
        st.caption("Includes: Dataset Summary, Model Comparison, Forecast Data, Congestion Risk, Anomalies, Recommendations")
    
    with col_r2:
        pdf_bytes = build_summary_pdf(
            df=df, location_name=selected_location, best_model_name=best_model_name,
            all_metrics=all_metrics, peak_bw=peak_bw, peak_time=peak_time,
            recommended_capacity=recommended_capacity, risk_df=risk_df, stats=stats,
        )
        st.download_button(
            label="📄 Download PDF Summary", data=pdf_bytes,
            file_name=f"bandwidth_forecast_summary_{datetime.now().strftime('%Y%m%d')}.pdf",
            mime="application/pdf",
        )
        st.caption("Executive summary with key findings and recommendations")
    
    st.info("💡 **Tip:** The Excel report contains detailed hour-by-hour forecast data in the 'Forecast_vs_Actual' sheet.")


if __name__ == "__main__":
    main()
