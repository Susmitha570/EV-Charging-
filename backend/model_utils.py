import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

def load_and_clean_dataset(data_path: str) -> pd.DataFrame:
    df = pd.read_csv(data_path)

    df.drop_duplicates(inplace=True)

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include="object").columns
    for c in num_cols:
        df[c] = df[c].fillna(df[c].median())
    for c in cat_cols:
        df[c] = df[c].fillna("Unknown")

    # Feature Engineering (same as main.py)
    df["distance_per_station"] = df["distance_km"] / (df["num_ev_stations_route"] + 1)
    df["charging_efficiency"] = df["charging_capacity_kWh"] / (df["charger_power_kW"] + 1)
    df["energy_per_km"] = df["charging_capacity_kWh"] / (df["distance_km"] + 1)

    return df

def train_or_load_model(df: pd.DataFrame, model_path: str, scaler_path: str, features_path: str):
    if os.path.exists(model_path) and os.path.exists(scaler_path) and os.path.exists(features_path):
        model = joblib.load(model_path)
        scaler = joblib.load(scaler_path)
        feature_cols = joblib.load(features_path)
        return model, scaler, feature_cols, None

    TARGET = "charging_duration_hours"
    TEXT_COLS = ["source_state", "source_city", "destination_state", "destination_city"]

    X = df.drop(columns=[TARGET] + [c for c in TEXT_COLS if c in df.columns], errors="ignore")
    y = df[TARGET].astype(float)

    if "vehicle_type" in X.columns:
        X = pd.get_dummies(X, columns=["vehicle_type"], drop_first=False)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
    scaler = StandardScaler()
    X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features].astype(float))
    X_test[numeric_features] = scaler.transform(X_test[numeric_features].astype(float))

    rf = RandomForestRegressor(random_state=42)
    param_grid = {"n_estimators":[100,200], "max_depth":[10,20,None], "min_samples_split":[2,5]}

    grid = GridSearchCV(rf, param_grid=param_grid, cv=3, scoring="r2", n_jobs=-1, verbose=0)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    y_pred = best_model.predict(X_test)

    metrics = {
        "best_params": grid.best_params_,
        "r2": round(float(r2_score(y_test, y_pred)), 3),
        "mae": round(float(mean_absolute_error(y_test, y_pred)), 3),
        "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 3),
    }

    feature_cols = X.columns.tolist()

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)
    joblib.dump(feature_cols, features_path)

    return best_model, scaler, feature_cols, metrics
