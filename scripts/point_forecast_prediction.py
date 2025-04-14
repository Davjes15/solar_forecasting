import pandas as pd
import numpy as np
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from typing import List, Tuple

# --------------------------------------------------------------
# 1. Regression Tree for Point Forecasting
# --------------------------------------------------------------

def prepare_training_data(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = "Pm",
    reference_col: str = "Ppccs"
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare feature matrix X and target vector y for training.

    Parameters:
        df (pd.DataFrame): Input data.
        feature_cols (List[str]): List of column names to use as features.
        target_col (str): Column name for actual power measurements.
        reference_col (str): Column name for clear-sky corrected power.

    Returns:
        Tuple containing feature matrix X and target vector y (absolute relative error).
    """
    df = df.copy(deep=True)
    mask = df[reference_col] > 0
    df = df.loc[mask].copy()
    df["Lambda"] = ((df[target_col] - df[reference_col]) / df[reference_col]).abs()
    X = df[feature_cols]
    y = df["Lambda"]
    return X, y

def train_regression_tree(
    X: pd.DataFrame,
    y: pd.Series,
    base_estimator=None,
    n_estimators: int = 10,
    random_state: int = 42
) -> BaggingRegressor:
    """
    Train a BaggingRegressor using the training data.

    Parameters:
        X (pd.DataFrame): Feature matrix.
        y (pd.Series): Target vector.
        base_estimator: Regressor to use in the ensemble.
        n_estimators (int): Number of estimators in the ensemble.
        random_state (int): Seed for reproducibility.

    Returns:
        Trained BaggingRegressor model.
    """
    if base_estimator is None:
        base_estimator = DecisionTreeRegressor()
    model = BaggingRegressor(
        estimator=base_estimator,
        n_estimators=n_estimators,
        random_state=random_state
    )
    model.fit(X, y)
    return model

def forecast_point_generation(
    model: BaggingRegressor,
    df: pd.DataFrame,
    feature_cols: List[str],
    reference_col: str = "Ppccs",
    output_col: str = "Ppf"
) -> pd.DataFrame:
    """
    Generate point forecasts using Equation (12).

    Parameters:
        model (BaggingRegressor): Trained regression model.
        df (pd.DataFrame): Input data.
        feature_cols (List[str]): Feature columns used for prediction.
        reference_col (str): Column with Ppccs (clear-sky corrected power).
        output_col (str): Output column for point forecast power.

    Returns:
        pd.DataFrame with a new column containing predicted PV power.
    """
    df = df.copy()
    X_pred = df[feature_cols]
    lambda_pred = model.predict(X_pred)
    df[output_col] = df[reference_col] * (1 - np.abs(lambda_pred))
    return df

def create_synthetic_forecast_data(seed: int = 0, hours: int = 24) -> pd.DataFrame:
    """
    Create synthetic hourly forecast data for one day.

    Parameters:
        seed (int): Random seed for reproducibility.
        hours (int): Number of hourly records to generate.

    Returns:
        pd.DataFrame with synthetic weather features, Ppccs, and simulated Pm.
    """
    np.random.seed(seed)
    timestamps = pd.date_range("2022-06-01", periods=hours, freq="H", tz="UTC")
    df = pd.DataFrame({"timestamp": timestamps})
    df["temperature"] = 25 + 5 * np.sin(np.pi * df.index / hours)
    df["cloudiness"] = np.clip(np.random.normal(0.5, 0.3, size=hours), 0, 1)
    df["humidity"] = 60 + 10 * np.random.rand(hours)
    df["Ppccs"] = 5000 * np.clip(np.sin(np.pi * df.index / hours), 0, 1)
    df["Pm"] = df["Ppccs"] * (1 - df["cloudiness"] * 0.6)
    return df

if __name__ == "__main__":
    df = create_synthetic_forecast_data()
    features = ["temperature", "cloudiness", "humidity"]
    X_train, y_train = prepare_training_data(df, features)
    model = train_regression_tree(X_train, y_train)
    df_forecast = forecast_point_generation(model, df, features)
    print(df_forecast[["timestamp", "Ppccs", "Pm", "Ppf"]].round(2))

