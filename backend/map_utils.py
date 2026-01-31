import numpy as np
import requests
import folium
from folium.features import DivIcon

def geocode_city(city, state=None, country="India"):
    try:
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

def osrm_route_full(src_lat, src_lon, dst_lat, dst_lon):
    try:
        url = f"http://router.project-osrm.org/route/v1/driving/{src_lon},{src_lat};{dst_lon},{dst_lat}"
        params = {"overview": "full", "geometries": "geojson"}
        r = requests.get(url, params=params, timeout=25)
        r.raise_for_status()
        data = r.json()
        route = data["routes"][0]
        distance_km = float(route["distance"] / 1000.0)
        line = route["geometry"]["coordinates"]  # [lon,lat]
        coords = [(pt[1], pt[0]) for pt in line] # (lat,lon)
        return distance_km, coords
    except:
        return None, None

def haversine_km(lat1, lon1, lat2, lon2):
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = (np.sin(dlat/2)**2) + np.cos(lat1)*np.cos(lat2)*(np.sin(dlon/2)**2)
    c = 2*np.arctan2(np.sqrt(a), np.sqrt(1-a))
    return float(R*c)

def route_cumdist(coords):
    cum = [0.0]
    for i in range(1, len(coords)):
        cum.append(cum[-1] + haversine_km(coords[i-1][0], coords[i-1][1], coords[i][0], coords[i][1]))
    return cum

def points_every_km(coords, interval_km=50):
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
        d0, d1 = cum[j-1], cum[j]
        if d1 == d0:
            pts.append(coords[j])
            continue
        ratio = (t - d0) / (d1 - d0)
        lat = coords[j-1][0] + (coords[j][0] - coords[j-1][0]) * ratio
        lon = coords[j-1][1] + (coords[j][1] - coords[j-1][1]) * ratio
        pts.append((lat, lon))
    return pts

def add_label(map_obj, lat, lon, text, color="black", border="1px solid black", size=12):
    folium.Marker(
        [lat, lon],
        icon=DivIcon(
            icon_size=(300, 36),
            icon_anchor=(0, 0),
            html=f"""
            <div style="
                font-size:{size}px;
                color:{color};
                background:white;
                padding:2px 4px;
                border:{border};
                border-radius:4px;
                font-weight:bold;">
                {text}
            </div>
            """
        )
    ).add_to(map_obj)
