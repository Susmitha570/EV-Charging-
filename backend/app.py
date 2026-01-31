import os
import time
import math
import warnings
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import requests
import pandas as pd
import joblib

# -----------------------------
# CONFIG
# -----------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
MAPS_DIR = os.path.join(OUTPUTS_DIR, "maps")
MODELS_DIR = os.path.join(OUTPUTS_DIR, "models")

os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(MAPS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Optional dataset paths
CLEAN_DATASET_PATH = os.path.join(OUTPUTS_DIR, "clean_ev_dataset.csv")  # if you saved from main.py
RAW_DATASET_PATH = os.environ.get("EV_DATASET_PATH", "")  # optionally set env var to raw csv path

# Optional ML model paths (if you already trained & saved)
LR_MODEL_PATH = os.path.join(MODELS_DIR, "linear_regression_model.pkl")
RF_MODEL_PATH = os.path.join(MODELS_DIR, "random_forest_model.pkl")
SCALER_PATH   = os.path.join(MODELS_DIR, "scaler.pkl")  # optional if you saved scaler
FEATURES_PATH = os.path.join(MODELS_DIR, "feature_columns.pkl")  # optional list of columns

# -----------------------------
# APP
# -----------------------------
app = Flask(__name__, static_folder="../frontend", static_url_path="/")
CORS(app)

# -----------------------------
# HELPERS: GEO + OSRM + DIST + POINTS
# -----------------------------
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
    Uses OSRM alternatives=true so distances are REAL map distances.
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
    import numpy as np
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
    """
    if coords is None or len(coords) < 2:
        return []
    cum = route_cumdist(coords)
    total = cum[-1]
    if total <= 0:
        return []

    import numpy as np
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


def safe_name(s):
    return str(s).strip().lower().replace(" ", "_")


# -----------------------------
# DATASET LOADER (optional)
# -----------------------------
def load_dataset_if_any():
    """
    Tries to load clean_ev_dataset.csv first, else RAW path from env,
    else returns None.
    """
    if os.path.exists(CLEAN_DATASET_PATH):
        try:
            return pd.read_csv(CLEAN_DATASET_PATH)
        except:
            return None

    if RAW_DATASET_PATH and os.path.exists(RAW_DATASET_PATH):
        try:
            return pd.read_csv(RAW_DATASET_PATH)
        except:
            return None

    return None


def dataset_state_for_city(df, city_col, state_col, city_name):
    """
    city_col: 'source_city' or 'destination_city'
    """
    try:
        tmp = df.copy()
        tmp["_city_"] = tmp[city_col].astype(str).str.lower().str.strip()
        pick = tmp[tmp["_city_"] == city_name.lower().strip()]
        if not pick.empty and state_col in pick.columns:
            return str(pick.iloc[0][state_col])
    except:
        pass
    return None


def available_stations_from_dataset(df, source_city, destination_city, vehicle_type):
    """
    Returns num_ev_stations_route for matching row, else median fallback.
    """
    try:
        tmp = df.copy()
        tmp["src_city"] = tmp["source_city"].astype(str).str.lower().str.strip()
        tmp["dst_city"] = tmp["destination_city"].astype(str).str.lower().str.strip()
        if "vehicle_type" in tmp.columns:
            tmp["veh"] = tmp["vehicle_type"].astype(str).str.lower().str.strip()
        else:
            tmp["veh"] = ""

        match = tmp[
            (tmp["src_city"] == source_city.lower().strip()) &
            (tmp["dst_city"] == destination_city.lower().strip()) &
            ((tmp["veh"] == vehicle_type.lower().strip()) if "vehicle_type" in tmp.columns else True)
        ]
        if not match.empty and "num_ev_stations_route" in match.columns:
            return int(float(match.iloc[0]["num_ev_stations_route"]))

        if "num_ev_stations_route" in tmp.columns:
            return int(float(tmp["num_ev_stations_route"].median()))
    except:
        pass
    return 0


# -----------------------------
# MAP GENERATION (NO POPUPS)
# -----------------------------
def generate_map_html(route_coords, src_lat, src_lon, dst_lat, dst_lon,
                      source_city, destination_city, distance_km,
                      station_points, shown_on_map):
    import folium
    from folium.features import DivIcon

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

    m = folium.Map(location=[(src_lat + dst_lat) / 2, (src_lon + dst_lon) / 2], zoom_start=7)

    # route line
    folium.PolyLine(route_coords, weight=5, opacity=0.9).add_to(m)

    add_label(m, src_lat, src_lon, f"Source: {source_city}", color="green", border="2px solid green", size=13)
    add_label(m, dst_lat, dst_lon, f"Destination: {destination_city}", color="red", border="2px solid red", size=13)

    mid = route_coords[len(route_coords) // 2]
    add_label(m, mid[0], mid[1], f"Distance(OSRM): {distance_km:.2f} km", color="black", border="2px solid black", size=12)

    # station labels
    for idx, (lat, lon) in enumerate(station_points, start=1):
        if idx <= shown_on_map:
            add_label(m, lat, lon, f"EV Station {idx}", color="blue", border="2px dashed blue", size=11)
        else:
            add_label(m, lat, lon, f"Suggested {idx}", color="orange", border="2px dashed orange", size=11)

    file_name = f"route_map_{safe_name(source_city)}_to_{safe_name(destination_city)}.html"
    map_path = os.path.join(MAPS_DIR, file_name)
    m.save(map_path)
    return file_name


# -----------------------------
# OPTIONAL ML PREDICTION (safe fallback)
# -----------------------------
def try_predict_duration(distance_km, interval_km, available_dataset):
    """
    If models exist, returns predicted charging duration hours (LR & RF)
    Else returns None.
    """
    if not (os.path.exists(LR_MODEL_PATH) and os.path.exists(RF_MODEL_PATH)):
        return None

    try:
        lr = joblib.load(LR_MODEL_PATH)
        rf = joblib.load(RF_MODEL_PATH)

        scaler = None
        feature_cols = None
        if os.path.exists(SCALER_PATH):
            scaler = joblib.load(SCALER_PATH)
        if os.path.exists(FEATURES_PATH):
            feature_cols = joblib.load(FEATURES_PATH)

        # Minimal feature vector (generic)
        # If your training used many features, you MUST save feature_columns.pkl to align.
        x = pd.DataFrame([{
            "distance_km": float(distance_km),
            "num_ev_stations_route": float(available_dataset),
            "charger_power_kW": 50.0,
            "charging_capacity_kWh": 60.0,
            "distance_per_station": float(distance_km) / (float(available_dataset) + 1.0),
            "charging_efficiency": 60.0 / (50.0 + 1.0),
            "energy_per_km": 60.0 / (float(distance_km) + 1.0),
            "interval_km": float(interval_km)
        }])

        if feature_cols is not None:
            # add missing columns as 0
            for c in feature_cols:
                if c not in x.columns:
                    x[c] = 0.0
            x = x[feature_cols]

        if scaler is not None:
            num_cols = x.select_dtypes(include="number").columns
            x[num_cols] = scaler.transform(x[num_cols])

        pred_lr = float(lr.predict(x)[0])
        pred_rf = float(rf.predict(x)[0])
        return {"predicted_lr_hours": pred_lr, "predicted_rf_hours": pred_rf}
    except:
        return None


# -----------------------------
# ROUTES
# -----------------------------
@app.get("/api/health")
def health():
    return jsonify({"ok": True, "message": "Backend running ✅"})


@app.post("/api/recommend")
def recommend():
    data = request.get_json(force=True)

    source_city = str(data.get("source_city", "")).strip()
    destination_city = str(data.get("destination_city", "")).strip()
    vehicle_type = str(data.get("vehicle_type", "")).strip().lower()

    if not source_city or not destination_city or not vehicle_type:
        return jsonify({"ok": False, "error": "source_city, destination_city, vehicle_type are required"}), 400

    interval_map = {"2w": 40, "3w": 50, "4w": 60, "bus": 70}
    interval_km = interval_map.get(vehicle_type, 50)

    df = load_dataset_if_any()

    # find state (optional help for geocode)
    src_state = None
    dst_state = None
    if df is not None:
        src_state = dataset_state_for_city(df, "source_city", "source_state", source_city)
        dst_state = dataset_state_for_city(df, "destination_city", "destination_state", destination_city)

    # geocode
    src_lat, src_lon = geocode_city(source_city, src_state)
    dst_lat, dst_lon = geocode_city(destination_city, dst_state)

    if src_lat is None or dst_lat is None:
        return jsonify({
            "ok": False,
            "error": "Geocoding failed. Try: City + State (example: Nellore, Andhra Pradesh)"
        }), 400

    # osrm routes
    routes_osrm = osrm_routes_full(src_lat, src_lon, dst_lat, dst_lon, top_n=3)
    if not routes_osrm:
        return jsonify({"ok": False, "error": "OSRM route fetch failed. Try again."}), 400

    best = routes_osrm[0]
    distance_km = best["distance_km"]
    duration_min = best["duration_min"]
    route_coords = best["coords"]

    # station points along route
    station_points = points_every_km(route_coords, interval_km=interval_km)

    available_dataset = 0
    if df is not None:
        available_dataset = available_stations_from_dataset(df, source_city, destination_city, vehicle_type)

    required_stations = int(math.ceil(distance_km / interval_km))
    shown_on_map = min(len(station_points), available_dataset) if available_dataset > 0 else len(station_points)
    extra_needed = max(0, required_stations - shown_on_map)

    # map html
    map_file = generate_map_html(
        route_coords, src_lat, src_lon, dst_lat, dst_lon,
        source_city, destination_city, distance_km,
        station_points, shown_on_map
    )

    # optional prediction
    preds = try_predict_duration(distance_km, interval_km, available_dataset)

    other_routes = [
        {"route_no": i + 1, "distance_km": rt["distance_km"], "duration_min": rt["duration_min"]}
        for i, rt in enumerate(routes_osrm)
    ]

    return jsonify({
        "ok": True,
        "input": {
            "source_city": source_city,
            "destination_city": destination_city,
            "vehicle_type": vehicle_type.upper()
        },
        "best_route": {
            "distance_km": distance_km,
            "duration_min": duration_min
        },
        "kpis": {
            "interval_km": interval_km,
            "stations_required": required_stations,
            "stations_points_on_route": len(station_points),
            "stations_available_dataset": int(available_dataset),
            "stations_shown_on_map": int(shown_on_map),
            "extra_needed": int(extra_needed)
        },
        "other_routes": other_routes,
        "map_url": f"/maps/{map_file}",
        "predictions": preds
    })


@app.get("/maps/<path:filename>")
def serve_map(filename):
    return send_from_directory(MAPS_DIR, filename)


# Serve frontend
@app.get("/")
def home():
    return send_from_directory(app.static_folder, "index.html")


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
