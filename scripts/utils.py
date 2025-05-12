import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

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
