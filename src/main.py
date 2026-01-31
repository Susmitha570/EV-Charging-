# ==========================================================
# EV PROJECT – SINGLE main.py (FIXED DISTANCE USING OSRM)
# ETL + EDA + FEATURE ENGINEERING
# + SAVE processed_dataset.csv
# + ML (Linear Regression + Random Forest) + SAVE MODELS + SAVE predictions.csv
# + ROUTE RECOMMENDATION + MAP (NO POPUPS)
# ==========================================================

import os
import time
import warnings
warnings.filterwarnings("ignore")
import math


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import folium
from folium.features import DivIcon

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib


print("\n🚗 EV CHARGING PROJECT STARTED 🚗\n")

# ==========================================================
# STEP 0: PATH SETUP
# ==========================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_PATH = r"C:\ev project\dataset\ev_dataset_10000_with_charging_duration.csv"

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
EDA_DIR = os.path.join(OUTPUT_DIR, "eda")
MAPS_DIR = os.path.join(OUTPUT_DIR, "maps")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(EDA_DIR, exist_ok=True)
os.makedirs(MAPS_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

clean_path = os.path.join(OUTPUT_DIR, "clean_ev_dataset.csv")
processed_file = os.path.join(OUTPUT_DIR, "processed_dataset.csv")
predictions_file = os.path.join(OUTPUT_DIR, "predictions.csv")

print("✅ BASE_DIR:", BASE_DIR)
print("✅ DATA_PATH:", DATA_PATH)
print("✅ OUTPUT_DIR:", OUTPUT_DIR)


# ==========================================================
# HELPERS: GEO + OSRM ROUTES + DISTANCE + POINTS
# ==========================================================
def geocode_city(city, state=None, country="India"):
    """
    Returns (lat, lon) or (None, None)
    """
    try:
        time.sleep(1)  # avoid nominatim block
        q = f"{city}, {state}, {country}" if state and state != "Unknown" else f"{city}, {country}"
        url = "https://nominatim.openstreetmap.org/search"
        params = {"q": q, "format": "json", "limit": 1}
        headers = {"User-Agent": "EV-Project/1.0 (education)"}
        r = requests.get(url, params=params, headers=headers, timeout=20)
        if r.status_code == 200 and r.json():
            return float(r.json()[0]["lat"]), float(r.json()[0]["lon"])
    except:
        pass
    return None, None


def osrm_routes_full(src_lat, src_lon, dst_lat, dst_lon, top_n=3):
    """
    Returns list of routes:
      [{distance_km, duration_min, coords[(lat,lon),...]}, ...]
    Uses OSRM alternatives = true, so distances are REAL map distances.
    """
    try:
        url = f"https://router.project-osrm.org/route/v1/driving/{src_lon},{src_lat};{dst_lon},{dst_lat}"
        params = {
            "overview": "full",
            "geometries": "geojson",
            "alternatives": "true"
        }
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
        if data.get("code") != "Ok":
            return []

        routes = []
        for rt in data.get("routes", [])[:top_n]:
            dist_km = float(rt["distance"] / 1000.0)
            dur_min = float(rt["duration"] / 60.0)
            line = rt["geometry"]["coordinates"]  # [lon,lat]
            coords = [(pt[1], pt[0]) for pt in line]  # -> (lat,lon)
            routes.append({
                "distance_km": dist_km,
                "duration_min": dur_min,
                "coords": coords
            })
        return routes
    except:
        return []


def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat / 2) ** 2) + np.cos(lat1) * np.cos(lat2) * (np.sin(dlon / 2) ** 2)
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    return float(R * c)


def route_cumdist(coords):
    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + haversine_km(coords[i - 1][0], coords[i - 1][1], coords[i][0], coords[i][1]))
    return cum


def points_every_km(coords, interval_km=50):
    """
    Returns points along route every interval_km.
    These points are where you will "show stations" on map.
    """
    if coords is None or len(coords) < 2:
        return []
    cum = route_cumdist(coords)
    total = cum[-1]
    if total <= 0:
        return []

    targets = np.arange(interval_km, total, interval_km)
    pts = []
    j = 1
    for t in targets:
        while j < len(cum) and cum[j] < t:
            j += 1
        if j >= len(cum):
            break
        d0, d1 = cum[j - 1], cum[j]
        if d1 == d0:
            pts.append(coords[j])
            continue
        ratio = (t - d0) / (d1 - d0)
        lat = coords[j - 1][0] + (coords[j][0] - coords[j - 1][0]) * ratio
        lon = coords[j - 1][1] + (coords[j][1] - coords[j - 1][1]) * ratio
        pts.append((lat, lon))
    return pts


def add_label(map_obj, lat, lon, text, color="black", bg="white", border="1px solid black", size=12):
    folium.Marker(
        [lat, lon],
        icon=DivIcon(
            icon_size=(260, 36),
            icon_anchor=(0, 0),
            html=f"""
            <div style="
                font-size:{size}px;
                color:{color};
                background:{bg};
                padding:2px 4px;
                border:{border};
                border-radius:4px;
                font-weight:bold;">
                {text}
            </div>
            """
        )
    ).add_to(map_obj)


# ==========================================================
# STEP 1: LOAD DATA
# ==========================================================
print("\n🔹 STEP 1: LOADING RAW DATASET...\n")
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(f"❌ Dataset not found: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
print("✅ Dataset Size:", df.shape)
print(df.head())


# ==========================================================
# STEP 2: CHECK MISSING VALUES
# ==========================================================
print("\n🔹 STEP 2: CHECKING MISSING VALUES (BEFORE CLEANING)\n")
print(df.isnull().sum())


# ==========================================================
# STEP 3: CLEANING
# ==========================================================
print("\n🔹 STEP 3: DATA CLEANING STARTED\n")
before = df.shape[0]
df.drop_duplicates(inplace=True)
after = df.shape[0]
print(f"Duplicates Removed: {before - after}")

num_cols = df.select_dtypes(include=np.number).columns
cat_cols = df.select_dtypes(include="object").columns

for col in num_cols:
    df[col].fillna(df[col].median(), inplace=True)

for col in cat_cols:
    df[col].fillna("Unknown", inplace=True)

print("Missing values filled successfully")


# ==========================================================
# STEP 4: AFTER CLEANING CHECK
# ==========================================================
print("\n🔹 STEP 4: AFTER CLEANING VERIFICATION\n")
print(df.isnull().sum())


# ==========================================================
# STEP 5: FEATURE ENGINEERING
# ==========================================================
print("\n🔹 STEP 5: FEATURE ENGINEERING\n")
needed_cols = ["distance_km", "num_ev_stations_route", "charging_capacity_kWh", "charger_power_kW"]
for col in needed_cols:
    if col not in df.columns:
        raise ValueError(f"❌ Required column missing: {col}")

df["distance_per_station"] = df["distance_km"] / (df["num_ev_stations_route"] + 1)
df["charging_efficiency"] = df["charging_capacity_kWh"] / (df["charger_power_kW"] + 1)
df["energy_per_km"] = df["charging_capacity_kWh"] / (df["distance_km"] + 1)

print("✅ New features created")


# ==========================================================
# STEP 6: EDA PLOTS
# ==========================================================
print("\n🔹 STEP 6: EDA PLOTS\n")
try:
    plt.figure()
    sns.histplot(df["charging_duration_hours"], kde=True)
    plt.title("Charging Duration Distribution")
    plt.savefig(os.path.join(EDA_DIR, "charging_duration.png"), dpi=200, bbox_inches="tight")
    plt.close()

    if "vehicle_type" in df.columns:
        plt.figure(figsize=(10, 5))
        sns.boxplot(x="vehicle_type", y="charging_duration_hours", data=df)
        plt.title("Vehicle Type vs Charging Duration")
        plt.xticks(rotation=30)
        plt.savefig(os.path.join(EDA_DIR, "vehicle_vs_duration.png"), dpi=200, bbox_inches="tight")
        plt.close()

    plt.figure(figsize=(10, 6))
    sns.heatmap(df.select_dtypes(include=np.number).corr(), cmap="coolwarm")
    plt.title("Correlation Heatmap (Numeric Only)")
    plt.savefig(os.path.join(EDA_DIR, "correlation_heatmap.png"), dpi=200, bbox_inches="tight")
    plt.close()

    print("✅ EDA saved -> outputs/eda")
except Exception as e:
    print("⚠ EDA skipped:", e)


# ==========================================================
# STEP 7: SAVE CLEAN + PROCESSED
# ==========================================================
df.to_csv(clean_path, index=False)
df.to_csv(processed_file, index=False)
print("\n✅ Clean dataset saved ->", clean_path)
print("✅ Processed dataset saved ->", processed_file)


# ==========================================================
# STEP 8: ML PREP
# ==========================================================
print("\n🔹 STEP 8: ML PREPARATION\n")
target = "charging_duration_hours"
if target not in df.columns:
    raise ValueError(f"❌ Target column missing: {target}")

drop_cols = [target, "source_state", "source_city", "destination_state", "destination_city"]
X = df.drop(columns=[c for c in drop_cols if c in df.columns], errors="ignore")
y = df[target].astype(float)

if "vehicle_type" in X.columns:
    X = pd.get_dummies(X, columns=["vehicle_type"], drop_first=False)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
numeric_features = X_train.select_dtypes(include=np.number).columns.tolist()
scaler = StandardScaler()
X_train[numeric_features] = scaler.fit_transform(X_train[numeric_features])
X_test[numeric_features] = scaler.transform(X_test[numeric_features])

# LR
print("\n🔹 TRAINING LINEAR REGRESSION\n")
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr = lr_model.predict(X_test)

print("LR MAE :", mean_absolute_error(y_test, y_pred_lr))
print("LR RMSE:", float(np.sqrt(mean_squared_error(y_test, y_pred_lr))))
print("LR R2  :", r2_score(y_test, y_pred_lr))

joblib.dump(lr_model, os.path.join(MODEL_DIR, "linear_regression_model.pkl"))

# RF
print("\n🔹 TRAINING RANDOM FOREST\n")
rf_model = RandomForestRegressor(n_estimators=200, random_state=42, n_jobs=-1)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_test)

print("RF MAE :", mean_absolute_error(y_test, y_pred_rf))
print("RF RMSE:", float(np.sqrt(mean_squared_error(y_test, y_pred_rf))))
print("RF R2  :", r2_score(y_test, y_pred_rf))

joblib.dump(rf_model, os.path.join(MODEL_DIR, "random_forest_model.pkl"))

# Save predictions
predictions = pd.DataFrame({
    "Actual": y_test.reset_index(drop=True),
    "Predicted_LR": pd.Series(y_pred_lr).reset_index(drop=True),
    "Predicted_RF": pd.Series(y_pred_rf).reset_index(drop=True),
})
predictions.to_csv(predictions_file, index=False)
print(f"\n✅ Predictions saved at: {predictions_file}")


# ==========================================================
# STEP 9: ROUTE RECOMMENDATION (OSRM EXACT DISTANCE) ✅✅
# ==========================================================
print("\n🔹 ROUTE RECOMMENDATION + MAP (OSRM EXACT)\n")

source_city_in = input("Enter Source City: ").strip()
destination_city_in = input("Enter Destination City: ").strip()
vehicle_type_in = input("Enter Vehicle Type (2W / 3W / 4W / Bus): ").strip().lower()

# interval
interval_map = {"2w": 40, "3w": 50, "4w": 60, "bus": 70}
interval_km = interval_map.get(vehicle_type_in, 50)

# find state from dataset (optional help for geocoding)
df["src_city"] = df["source_city"].astype(str).str.lower().str.strip()
df["dst_city"] = df["destination_city"].astype(str).str.lower().str.strip()

src_state = None
dst_state = None
try:
    src_state = df[df["src_city"] == source_city_in.lower()]["source_state"].iloc[0]
except:
    pass
try:
    dst_state = df[df["dst_city"] == destination_city_in.lower()]["destination_state"].iloc[0]
except:
    pass

# geocode using state if found
src_lat, src_lon = geocode_city(source_city_in, src_state)
dst_lat, dst_lon = geocode_city(destination_city_in, dst_state)

if src_lat is None or dst_lat is None:
    print("❌ Geocoding failed. Try: City, State, India (example: Nellore, Andhra Pradesh)")
    raise SystemExit

# OSRM: get Top 3 alternatives (REAL map distances)
routes_osrm = osrm_routes_full(src_lat, src_lon, dst_lat, dst_lon, top_n=3)
if not routes_osrm:
    print("❌ OSRM route fetch failed. Try again.")
    raise SystemExit

best = routes_osrm[0]
distance_km = best["distance_km"]
route_coords = best["coords"]

# stations from interval points
station_points = points_every_km(route_coords, interval_km=interval_km)

# dataset available stations (only for showing "available")
# if dataset doesn't have exact match, just take median (fallback)
available_dataset = 0
try:
    df["veh_type"] = df["vehicle_type"].astype(str).str.lower().str.strip()
    match = df[
        (df["src_city"] == source_city_in.lower().strip()) &
        (df["dst_city"] == destination_city_in.lower().strip()) &
        (df["veh_type"] == vehicle_type_in)
    ]
    if not match.empty:
        available_dataset = int(float(match.iloc[0]["num_ev_stations_route"]))
    else:
        available_dataset = int(float(df["num_ev_stations_route"].median()))
except:
    available_dataset = 0

required_stations = int(math.ceil(distance_km / interval_km))
shown_on_map = min(len(station_points), available_dataset) if available_dataset > 0 else len(station_points)

extra_needed = max(0, required_stations - shown_on_map)

print("\n✅ BEST ROUTE (OSRM MAP EXACT)\n")
print(f"Source      : {source_city_in}")
print(f"Destination : {destination_city_in}")
print(f"Vehicle     : {vehicle_type_in.upper()}")
print(f"📏 Distance : {distance_km:.2f} km  (OSRM)")
print(f"⏱ Duration : {best['duration_min']:.1f} min (OSRM)")
print(f"Interval rule: every {interval_km} km")
print(f"Stations Required: {required_stations}")
print(f"Stations Points on Route: {len(station_points)}")
print(f"Stations Available (dataset): {available_dataset}")
print(f"Stations Shown on Map: {shown_on_map}")
print(f"Extra Needed: {extra_needed}")

print("\n🔁 OTHER ROUTES (Top 3 – OSRM alternatives)\n")
for i, rt in enumerate(routes_osrm, start=1):
    print(f"- Route {i} | Distance: {rt['distance_km']:.2f} km | Duration: {rt['duration_min']:.1f} min")

# ==========================================================
# MAP (NO POPUPS) ✅
# ==========================================================
print("\n🗺 Generating map (NO POPUPS)...\n")
m = folium.Map(location=[(src_lat + dst_lat) / 2, (src_lon + dst_lon) / 2], zoom_start=7)

# route line
folium.PolyLine(route_coords, weight=5, opacity=0.9).add_to(m)

add_label(m, src_lat, src_lon, f"Source: {source_city_in}", color="green", border="2px solid green", size=13)
add_label(m, dst_lat, dst_lon, f"Destination: {destination_city_in}", color="red", border="2px solid red", size=13)

mid = route_coords[len(route_coords)//2]
add_label(m, mid[0], mid[1], f"Distance(OSRM): {distance_km:.2f} km", color="black", border="2px solid black", size=12)

# label stations on route points
for idx, (lat, lon) in enumerate(station_points, start=1):
    if idx <= shown_on_map:
        add_label(m, lat, lon, f"EV Station {idx}", color="blue", border="2px dashed blue", size=11)
    else:
        add_label(m, lat, lon, f"Suggested {idx}", color="orange", border="2px dashed orange", size=11)

safe_src = source_city_in.replace(" ", "_").lower()
safe_dst = destination_city_in.replace(" ", "_").lower()
map_path = os.path.join(MAPS_DIR, f"route_map_{safe_src}_to_{safe_dst}.html")
m.save(map_path)

print("✅ Map saved:", map_path)
print("👉 Open the .html file in Chrome to see route + EXACT distance + stations labels.")
print("\n🚀 ALL DONE 🚀")
