import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from typing import List
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from solar_forecasting.point_forecast_prediction import (
    prepare_training_data,
    train_regression_tree,
    forecast_point_generation
)


def plot_seasonal_forecasts(
    df: pd.DataFrame,
    selected_days: dict,
    quantile_prefix: str = "Ppp_",
    measured_col: str = "pv_output_kw"
):
    """
    Plots stacked quantile forecasts and measured PV output for user-specified seasonal days.

    Parameters:
    - df (pd.DataFrame): DataFrame with timestamped forecast and measured power in kW.
    - selected_days (dict): Dict of {season: 'YYYY-MM-DD'} date values to visualize.
    - quantile_prefix (str): Prefix for quantile forecast columns.
    - measured_col (str): Column name for measured PV output (in kW).
    """
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour

    fig, axs = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True)
    axs = axs.flatten()

    quantiles = list(range(10, 100, 10))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(quantiles)))[::-1]

    for i, (season, date_str) in enumerate(selected_days.items()):
        ax = axs[i]
        date = pd.to_datetime(date_str).date()
        daily_df = df[df["date"] == date].copy()

        if daily_df.empty:
            ax.set_title(f"{season} – {date_str} (No data)")
            ax.axis("off")
            continue

        hourly = daily_df.groupby("hour").mean(numeric_only=True)
        bottoms = np.zeros_like(hourly.index, dtype=float)

        for j, q in enumerate(quantiles):
            col = f"{quantile_prefix}{q}"
            if col in hourly:
                values = hourly[col].values
                ax.bar(hourly.index, values, bottom=bottoms, width=0.8, color=colors[j], edgecolor='none', label=f'q={q}%')
                bottoms += values

        ax.plot(hourly.index, hourly[measured_col], color="black", linewidth=2, label="PV Measurement")
        ax.set_title(f"{season} – {date}")
        ax.set_xlabel("Hour of Day")
        ax.set_ylabel("Power [kW]")
        ax.grid(True)

    # Move the legend to the bottom
    handles, labels = axs[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Stacked Quantile Forecasts with Measured PV Output (in kW)", fontsize=16)
    plt.tight_layout(rect=[0, 0.05, 1, 0.95])  # Make room for legend at bottom
    plt.show()


def plot_probabilistic_forecast(
    df: pd.DataFrame,
    date_str: str,
    season: str,
    quantile_prefix: str = "Ppp_",
    measured_col: str = "pv_output_kw"
):
    """
    Plot a stacked quantile bar chart with measured PV output for a specific date and season.

    Parameters:
    - df (pd.DataFrame): Data with forecast quantiles and measured power (in kW).
    - date_str (str): The date to visualize (format: 'YYYY-MM-DD').
    - season (str): Label for the season (e.g., 'Summer').
    - quantile_prefix (str): Prefix for quantile columns (e.g., 'Ppp_').
    - measured_col (str): Column for measured PV power (in kW).
    """
    df["date"] = df["timestamp"].dt.date
    df["hour"] = df["timestamp"].dt.hour
    date = pd.to_datetime(date_str).date()
    daily_df = df[df["date"] == date].copy()

    if daily_df.empty:
        print(f"[WARNING] No data available for {season} – {date_str}")
        return

    hourly = daily_df.groupby("hour").mean(numeric_only=True)
    quantiles = list(range(10, 100, 10))
    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(quantiles)))[::-1]

    # Plot
    fig, ax = plt.subplots(figsize=(10, 5))
    bottoms = np.zeros_like(hourly.index, dtype=float)

    for j, q in enumerate(quantiles):
        col = f"{quantile_prefix}{q}"
        if col in hourly:
            values = hourly[col].values
            ax.bar(hourly.index, values, bottom=bottoms, width=0.8, color=colors[j], edgecolor='none', label=f'q={q}%')
            bottoms += values

    ax.plot(hourly.index, hourly[measured_col], color="black", linewidth=2, label="PV Measurement")
    ax.set_title(f"{season} – {date}")
    ax.set_xlabel("Hour of Day")
    ax.set_ylabel("Power [kW]")
    ax.grid(True)

    # Add legend below plot
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=6, bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout(rect=[0, 0.05, 1, 1])
    plt.show()


def simulate_realtime_forecast(
    df: pd.DataFrame,
    feature_cols: List[str],
    start_date: str,
    end_date: str,
    output_col: str = "Ppf",
    target_col: str = "Pm",
    reference_col: str = "Ppccs"
) -> pd.DataFrame:
    """
    Simulates real-time day-ahead training and forecasting using the user's own point forecasting pipeline.
    
    Parameters:
        df: Full DataFrame with hourly resolution and necessary columns.
        feature_cols: List of weather columns used as features (e.g., ["temperature_wx", "cloudiness", "humidity"])
        start_date: 'YYYY-MM-DD'
        end_date: 'YYYY-MM-DD'
        output_col: column name for predicted power (default: "Ppf")
        target_col: measured PV output (default: "Pm")
        reference_col: clear-sky corrected power (default: "Ppccs")

    Returns:
        DataFrame with datetime index and forecast results.
    """
    df = df.copy()
    # Ensure the index is timezone-naive
    df.index = pd.to_datetime(df.index).tz_localize(None)
    all_predictions = []

    for day in tqdm(pd.date_range(start=start_date, end=end_date)):
        forecast_hours = pd.date_range(day, periods=24, freq="h")

        # Training data: all data before forecast day
        train_df = df[df.index < day].copy()
        test_df = df.loc[forecast_hours].copy()

        # Skip if not enough data
        if len(train_df) < 200 or test_df.empty:
            continue

        # Prepare training data
        X_train, y_train = prepare_training_data(
            train_df,
            feature_cols=feature_cols,
            target_col=target_col,
            reference_col=reference_col
        )

        if len(X_train) < 100:
            continue  # too little training data

        # Train model (you can modify hyperparameters here)
        model = train_regression_tree(
            X_train, y_train,
            n_estimators=10,
            max_samples=0.9,
            bootstrap=True,
            n_jobs=None,
            random_state=42,
            base_estimator=DecisionTreeRegressor(
                max_depth=None,
                min_samples_leaf=1,
                random_state=42
            )
        )

        # Forecast next 24 hours
        test_df = forecast_point_generation(
            model,
            test_df,
            feature_cols=feature_cols,
            reference_col=reference_col,
            output_col=output_col
        )
        test_df["lambda_hat"] = np.abs((test_df[reference_col] - test_df[output_col]) / test_df[reference_col])
        # Clip forecast to physical limits
        test_df[output_col] = np.clip(test_df[output_col], 0, test_df[reference_col])
    
        all_predictions.append(test_df[[target_col, reference_col, output_col, "lambda_hat"]])

    return pd.concat(all_predictions)


def load_prepare_data(file_path: str, df_w: pd.DataFrame) -> pd.DataFrame:
    # Load and parse PV data
    df_pv = pd.read_csv(file_path, skiprows=3, sep=',', parse_dates=["time"])
    df_pv.rename(columns={"electricity": "pv_output_kw", "time": "timestamp"}, inplace=True)
    df_pv.drop(columns=["local_time"], inplace=True)

    if df_pv["timestamp"].dt.tz is None:
        df_pv["timestamp"] = df_pv["timestamp"].dt.tz_localize("UTC")
    else:
        df_pv["timestamp"] = df_pv["timestamp"].dt.tz_convert("UTC")

    df_pv.set_index("timestamp", inplace=True)

    # Weather data preparation
    df_w = df_w.rename(columns={"datetime": "timestamp"})
    df_w["timestamp"] = pd.to_datetime(df_w["timestamp"])

    if df_w["timestamp"].dt.tz is None:
        df_w["timestamp"] = df_w["timestamp"].dt.tz_localize("UTC")
    else:
        df_w["timestamp"] = df_w["timestamp"].dt.tz_convert("UTC")

   # Merge on timestamp column
    df_merged = pd.merge(df_pv, df_w, on="timestamp", how="inner", suffixes=("_pv", "_wx"))
    
    return df_merged

