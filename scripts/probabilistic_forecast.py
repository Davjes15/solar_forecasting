import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
from scipy import stats

# ------------------------------
# 1. Probabilistic Forecast
# ------------------------------

def compute_relative_error(
    df: pd.DataFrame,
    forecast_col: str = "Ppf",
    actual_col: str = "Pm",
    cloud_col: str = "cloudiness",
    timestamp_col: str = "timestamp"
) -> pd.DataFrame:
    """
    Resamples the data to 1-minute resolution and computes the relative error:
    lambda(t) = (Ppf(t) - Pm(t)) / Ppf(t)

    Parameters:
    - df (pd.DataFrame): Input DataFrame with forecast, actual, and cloudiness columns.
    - forecast_col (str): Name of the forecast column (default = "Ppf").
    - actual_col (str): Name of the actual power output column (default = "Pm").
    - cloud_col (str): Name of the cloudiness column (default = "cloudiness").
    - timestamp_col (str): Name of the timestamp column (default = "timestamp").

    Returns:
    - DataFrame with 1-min resolution and a 'lambda' column.
    """
    df = df.copy()
    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df.set_index(timestamp_col, inplace=True)
    # Resample to 1-minute resolution and interpolate missing values
    df_1min = df.resample("1min").interpolate(method="linear")
    df_1min[cloud_col] = df_1min[cloud_col].ffill()  # fills NaNs by carrying forward the last known value of cloudiness

    mask = df_1min[forecast_col] > 0  # Avoid division by zero
    df_1min["lambda"] = ((df_1min[forecast_col] - df_1min[actual_col]) / df_1min[forecast_col]).where(mask)
    return df_1min


def categorize_by_cloudiness(
    df: pd.DataFrame,
    lambda_col: str = "lambda",
    cloud_col: str = "cloudiness",
    bins: int = 8
) -> Tuple[Dict[int, pd.Series], pd.DataFrame]:
    """
    Categorizes relative errors (lambda) into bins by forecasted cloudiness level.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with 'lambda' and 'cloudiness' columns.
    - lambda_col (str): Name of the relative error column (default = "lambda").
    - cloud_col (str): Name of the cloudiness column (default = "cloudiness").
    - bins (int): Number of cloudiness bins (default = 8).

    Returns:
    - Tuple of a dictionary with categorized relative errors and the updated DataFrame.
    """
    df = df.copy()
    cloud_bins = np.linspace(0, 1, bins + 1)  # Equally spaced bins
    df['lambda_class'] = pd.cut(df[cloud_col], bins=cloud_bins, labels=False, include_lowest=True)  # Assign bin number

    categorized = {}
    for i in range(bins):
        subset = df[df['lambda_class'] == i][lambda_col].dropna()
        if not subset.empty:
            categorized[i] = subset
    return categorized, df


def fit_best_distribution(
    data: pd.Series
) -> Tuple[stats.rv_continuous, Tuple]:
    """
    Fits multiple distributions to data and selects the best using BIC.
    Implements Equation (13).

    Parameters:
    - data (pd.Series): Input data to fit a distribution.

    Returns:
    - Tuple of the best distribution and its parameters.
    """
    # List of distributions to fit based on the paper and literature for solar forecasting
    distributions = [
        stats.norm, stats.logistic, stats.genextreme, stats.t, stats.gumbel_r,
        stats.rayleigh, stats.genpareto, stats.skewnorm, stats.beta,
        stats.gamma, stats.weibull_min, stats.laplace
    ]

    best_fit = None
    lowest_bic = np.inf
    n = len(data)

    for dist in distributions:
        try:
            params = dist.fit(data)
            log_likelihood = np.sum(dist.logpdf(data, *params))
            k = len(params)
            bic = -2 * log_likelihood + k * np.log(n)
            if bic < lowest_bic:
                lowest_bic = bic
                best_fit = (dist, params)
        except Exception:
            continue

    return best_fit


def generate_probabilistic_forecast(
    df: pd.DataFrame,
    fitted_cdfs: Dict[int, Tuple[stats.rv_continuous, Tuple]],
    quantiles: List[float],
    forecast_col: str = "Ppf",
    class_col: str = "lambda_class"
) -> pd.DataFrame:
    """
    Applies Equations (14) and (15) to generate probabilistic forecasts
    for each time step and quantile.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with 'lambda_class', 'Ppf', and 'timestamp' columns.
    - fitted_cdfs (Dict): Dictionary of fitted distributions per cloudiness bin.
    - quantiles (List): List of quantiles to compute (default = [0.1, 0.2, ..., 0.9]).
    - forecast_col (str): Name of the forecast column (default = "Ppf").
    - class_col (str): Name of the cloudiness class column (default = "lambda_class").

    Returns:
    - DataFrame with added columns for quantile forecasts.
    """
    df = df.copy()
    for q in quantiles:
        forecasts = []
        for _, row in df.iterrows():
            cloud_class = row[class_col]
            if cloud_class in fitted_cdfs:
                dist, params = fitted_cdfs[cloud_class]
                lambda_q = dist.ppf(q, *params)
                pf_q = row[forecast_col] * (1 - lambda_q)
                forecasts.append(pf_q)
            else:
                forecasts.append(np.nan)
        df[f"Ppp_{int(q * 100)}"] = forecasts
    return df


def run_probabilistic_forecast_pipeline(
    df: pd.DataFrame,
    quantiles: List[float] = [i / 100 for i in range(10, 100, 10)]  
) -> pd.DataFrame:
    """
    Main pipeline to prepare data, fit error distributions per cloudiness bin,
    and produce quantile forecasts.

    Parameters:
    - df (pd.DataFrame): Input DataFrame with 'Ppf', 'Pm', 'cloudiness', and 'timestamp' columns.
    - quantiles (List): List of quantiles to compute (default = [0.1, 0.2, ..., 0.9]).

    Returns:
    - DataFrame with added columns for quantile forecasts.
    """
    df_processed = compute_relative_error(df)
    lambda_bins, df_processed = categorize_by_cloudiness(df_processed)
    fitted_cdfs = {i: fit_best_distribution(data) for i, data in lambda_bins.items()}
    df_result = generate_probabilistic_forecast(df_processed, fitted_cdfs, quantiles)
    return df_result
