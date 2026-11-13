import html
import streamlit as st
import sqlite3
from datetime import datetime
import pandas as pd
import numpy as np
import requests
import folium
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(page_title="EcoRoute EV", page_icon="🔋", layout="wide")


@st.cache_data(ttl=1800, show_spinner=False)
def get_charging_stations(south, west, north, east):
    """Fetch mapped EV charging stations from public Overpass mirrors."""
    query = f"""
    [out:json][timeout:35];
    nwr["amenity"="charging_station"]({south},{west},{north},{east});
    out center tags;
    """
    endpoints = [
        "https://overpass-api.de/api/interpreter",
        "https://overpass.kumi.systems/api/interpreter",
    ]
    last_error = None
    for endpoint in endpoints:
        try:
            response = requests.get(
                endpoint,
                params={"data": query},
                headers={"User-Agent": "EcoRouteEV/1.1 (educational project)"},
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            return payload.get("elements", [])
        except (requests.RequestException, ValueError) as error:
            last_error = error
    raise RuntimeError(f"Charging-station map service unavailable: {last_error}")



def haversine_km(lat1, lon1, lat2, lon2):
    """Approximate great-circle distance between two latitude/longitude points."""
    radius_km = 6371.0
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp = np.radians(lat2 - lat1)
    dl = np.radians(lon2 - lon1)
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * radius_km * np.arcsin(np.sqrt(a))


def prepare_route_stations(route_coords, station_records, corridor_km=35.0):
    """Project OSM charging-station records approximately onto the route."""
    if len(route_coords) < 2:
        return []

    route_lons = np.asarray([point[0] for point in route_coords], dtype=float)
    route_lats = np.asarray([point[1] for point in route_coords], dtype=float)
    segment_lengths = np.asarray([
        haversine_km(route_lats[i], route_lons[i], route_lats[i + 1], route_lons[i + 1])
        for i in range(len(route_coords) - 1)
    ])
    cumulative_km = np.concatenate(([0.0], np.cumsum(segment_lengths)))

    candidates = []
    seen = set()
    for station in station_records:
        lat = station.get("lat")
        lon = station.get("lon")
        if lat is None:
            lat = station.get("center", {}).get("lat")
        if lon is None:
            lon = station.get("center", {}).get("lon")
        if lat is None or lon is None:
            continue
        lat, lon = float(lat), float(lon)
        identity = (round(lat, 5), round(lon, 5))
        if identity in seen:
            continue
        seen.add(identity)

        distances = np.asarray([
            haversine_km(lat, lon, route_lats[i], route_lons[i])
            for i in range(len(route_coords))
        ])
        nearest_index = int(np.argmin(distances))
        detour_km = float(distances[nearest_index])
        if detour_km > corridor_km:
            continue

        tags = station.get("tags", {})
        candidates.append({
            "name": tags.get("name") or tags.get("operator") or "EV Charging Station",
            "lat": lat,
            "lon": lon,
            "route_km": float(cumulative_km[nearest_index]),
            "detour_km": detour_km,
            "operator": tags.get("operator", "Not listed"),
            "socket_output": tags.get("socket:output", "Not listed"),
            "google_maps_url": f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}&travelmode=driving",
        })

    return sorted(candidates, key=lambda item: item["route_km"])


def calculate_smart_charging_stops(
    route_stations, distance_km, energy_kwh, battery_capacity_kwh,
    starting_soc_pct, reserve_pct, target_soc_pct, charger_power_kw,
):
    """Estimate reachable charging stops; all energy figures are prototype estimates."""
    if distance_km <= 0 or energy_kwh <= 0 or battery_capacity_kwh <= 0:
        return [], "The route or energy estimate is not valid for charging-stop planning."

    consumption_kwh_per_km = energy_kwh / distance_km
    current_energy = battery_capacity_kwh * starting_soc_pct / 100.0
    reserve_energy = battery_capacity_kwh * reserve_pct / 100.0
    target_energy = battery_capacity_kwh * target_soc_pct / 100.0
    current_progress = 0.0
    stops = []
    available_stations = [s for s in route_stations if s["route_km"] > 0.5]

    for _ in range(20):
        remaining_distance = max(0.0, distance_km - current_progress)
        energy_needed = remaining_distance * consumption_kwh_per_km
        if energy_needed <= max(0.0, current_energy - reserve_energy):
            return stops, "The destination is reachable without another charging stop."

        max_reachable_distance = max(0.0, (current_energy - reserve_energy) / consumption_kwh_per_km)
        furthest_reachable_km = current_progress + max_reachable_distance
        reachable = [
            station for station in available_stations
            if current_progress + 1.0 < station["route_km"] <= furthest_reachable_km
        ]
        if not reachable:
            return stops, (
                "No listed charging station was found within the estimated safe driving range. "
                "Check another charging source or start with a higher battery level."
            )

        # Choose the furthest reachable station to reduce unnecessary stops.
        chosen = max(reachable, key=lambda item: item["route_km"])
        travel_from_last_stop = chosen["route_km"] - current_progress
        arrival_energy = current_energy - travel_from_last_stop * consumption_kwh_per_km
        charge_to_energy = max(0.0, target_energy - arrival_energy)
        charging_minutes = (charge_to_energy / charger_power_kw) * 60.0 if charger_power_kw > 0 else 0.0
        chosen = dict(chosen)
        chosen.update({
            "distance_from_previous_stop_km": travel_from_last_stop,
            "arrival_energy_kwh": max(0.0, arrival_energy),
            "arrival_soc_pct": max(0.0, arrival_energy / battery_capacity_kwh * 100.0),
            "charge_added_kwh": charge_to_energy,
            "target_soc_pct": target_soc_pct,
            "charging_minutes": charging_minutes,
        })
        stops.append(chosen)
        current_energy = target_energy
        current_progress = chosen["route_km"]
        available_stations = [s for s in available_stations if s["route_km"] > current_progress + 0.5]

    return stops, "The planner reached its safety limit while estimating multiple stops. Review the journey manually."


# ---------------------------
# Free weather lookup (Open-Meteo)
# ---------------------------
@st.cache_data(ttl=1800, show_spinner=False)
def get_weather_for_coordinates(latitude, longitude):
    """Fetch current weather from Open-Meteo; no API key required."""
    response = requests.get(
        "https://api.open-meteo.com/v1/forecast",
        params={
            "latitude": latitude,
            "longitude": longitude,
            "current": "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,weather_code",
            "timezone": "auto",
        },
        timeout=15,
    )
    response.raise_for_status()
    return response.json().get("current", {})


def local_ev_answer(question):
    """Small offline keyword-based EV help assistant; not a generative LLM."""
    q = question.lower().strip()
    answers = [
        (("hello", "hi", "hey", "vanakkam"), "Hello! 👋 I'm EcoRoute EV Assistant. Ask me about batteries, charging, range, energy, cost or this app."),
        (("what is ev", "electric vehicle"), "An electric vehicle uses an electric motor powered by a rechargeable battery. It has no tailpipe emissions while driving."),
        (("range", "how far", "battery distance"), "EV range depends on usable battery capacity and energy consumption. Approximate range = usable battery energy (kWh) ÷ consumption (kWh/km). Speed, traffic, temperature and driving style affect it."),
        (("charging time", "charge time", "how long.*charge"), "Approximate charging time = energy to add (kWh) ÷ charger power (kW). Real charging can take longer due to charging losses and battery charge-rate limits."),
        (("charging station", "charger near", "find charger"), "EcoRoute EV searches OpenStreetMap for mapped charging stations near your route. Map listings do not confirm live availability, working status or connector compatibility."),
        (("regenerative braking", "regen braking"), "Regenerative braking converts some kinetic energy into electrical energy during deceleration and returns it to the battery."),
        (("cost", "electricity bill", "price per km"), "Estimated trip cost = energy consumed (kWh) × electricity tariff (₹/kWh). Example: 15 kWh × ₹8 = ₹120."),
        (("carbon", "co2", "emission"), "An EV produces no tailpipe emissions while driving. Total lifecycle emissions also depend on electricity generation and vehicle and battery manufacturing."),
        (("battery health", "degradation"), "Battery health compares the battery's current usable capacity with its original capacity. Age, heat and usage can affect it."),
        (("save energy", "efficiency", "eco driving"), "Drive smoothly, maintain steady speeds, avoid rapid acceleration, check tyre pressure and use regenerative braking where available."),
        (("machine learning", "model", "prediction"), "This prototype uses a Random Forest model trained on synthetic demonstration data. Its estimates are educational and are not validated real-world vehicle predictions."),
        (("how to use", "ecoroute", "this application"), "Use Route Planner to plan a trip, Energy Lab to explore consumption, Trip History for saved journeys, and the other dashboard tabs for weather, analytics, charging and EV settings."),
        (("thank", "thanks"), "You're welcome! ⚡ Have a safe and energy-efficient journey."),
    ]
    for keys, answer in answers:
        if any(key in q for key in keys):
            return answer
    return "I can help with EV range, battery health, charging time, charging stations, electricity cost, carbon savings, regenerative braking and EcoRoute EV. Try asking one of these topics. ⚡"


# ---------------------------
# Local trip-history storage
# ---------------------------
HISTORY_DB = "ecoroute_history.db"


def initialize_history_db():
    """Create the local trip-history table if it does not already exist."""
    with sqlite3.connect(HISTORY_DB) as connection:
        connection.execute(
            """CREATE TABLE IF NOT EXISTS trip_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                trip_time TEXT NOT NULL,
                origin TEXT NOT NULL,
                destination TEXT NOT NULL,
                distance_km REAL NOT NULL,
                duration_min REAL NOT NULL,
                energy_kwh REAL NOT NULL,
                route_name TEXT NOT NULL
            )"""
        )


def save_trip_to_history(trip):
    """Save one successfully calculated trip to the local SQLite database."""
    with sqlite3.connect(HISTORY_DB) as connection:
        connection.execute(
            """INSERT INTO trip_history
               (trip_time, origin, destination, distance_km, duration_min, energy_kwh, route_name)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                trip["origin"], trip["destination"], trip["distance_km"],
                trip["duration_min"], trip["predicted_energy"],
                trip["recommended_route_name"],
            ),
        )


def load_trip_history():
    """Return saved trips, newest first."""
    with sqlite3.connect(HISTORY_DB) as connection:
        return pd.read_sql_query(
            "SELECT * FROM trip_history ORDER BY id DESC", connection
        )


initialize_history_db()



# ---------------------------


# ---------------------------
# EcoRoute EV — Premium Light UI
# ---------------------------
st.markdown(r"""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');

:root {
    --eco-primary: #13865c;
    --eco-primary-dark: #096744;
    --eco-accent: #35c98a;
    --eco-bg: #f5f9f6;
    --eco-surface: #ffffff;
    --eco-text: #18382b;
    --eco-muted: #6b8175;
    --eco-border: #dfebe3;
}

/* Main application */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

.stApp {
    background:
        radial-gradient(ellipse at 5% 0%, #e1f8e9 0%, transparent 32%),
        radial-gradient(ellipse at 100% 25%, #eef9f2 0%, transparent 28%),
        var(--eco-bg);
    color: var(--eco-text);
}

.block-container {
    max-width: 1440px;
    padding-top: 1.5rem;
    padding-bottom: 3rem;
}

/* Headings */
h1, h2, h3 {
    font-family: 'Manrope', sans-serif !important;
    color: var(--eco-text) !important;
    letter-spacing: -0.035em;
}

h1 {
    font-weight: 800 !important;
}

h2, h3 {
    font-weight: 700 !important;
}

/* Readable body text */
.stApp p,
.stApp li,
.stApp label,
.stApp [data-testid="stCaptionContainer"] {
    color: var(--eco-text);
}

.stApp [data-testid="stCaptionContainer"] {
    color: var(--eco-muted);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #edfaf1, #ffffff);
    border-right: 1px solid var(--eco-border);
}

[data-testid="stSidebar"] * {
    color: var(--eco-text);
}


/* Premium glass-style navigation tabs */
.stTabs [data-baseweb="tab-list"] {
    display: flex;
    gap: 10px;
    padding: 8px;
    border: 1px solid rgba(19, 134, 92, 0.13);
    border-radius: 18px;
    background: rgba(255, 255, 255, 0.38);
    backdrop-filter: blur(16px);
    -webkit-backdrop-filter: blur(16px);
    box-shadow: 0 8px 24px rgba(24, 56, 43, 0.04);
}

/* Individual tabs */
.stTabs [data-baseweb="tab"] {
    position: relative;
    background: transparent !important;
    border: none !important;
    border-radius: 12px;
    padding: 12px 18px;
    transition: background 0.25s ease, transform 0.25s ease;
}

/* Tab labels */
.stTabs [data-baseweb="tab"] p {
    color: #496657 !important;
    font-weight: 600;
    transition: color 0.25s ease;
}

/* Hover glass effect */
.stTabs [data-baseweb="tab"]:hover {
    background: rgba(255, 255, 255, 0.55) !important;
    transform: translateY(-1px);
}

/* Active tab — no white rectangular box */
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: rgba(32, 184, 121, 0.10) !important;
    border: 1px solid rgba(19, 134, 92, 0.18) !important;
    border-radius: 12px;
}

/* Active tab text */
.stTabs [data-baseweb="tab"][aria-selected="true"] p {
    color: #087653 !important;
    font-weight: 700 !important;
}


/* Polished interactive navigation and tab-content entrance */
.stTabs [data-baseweb="tab"] {
    min-height: 44px;
    white-space: nowrap;
}
.stTabs [data-baseweb="tab-list"] {
    overflow-x: auto;
    scrollbar-width: thin;
    scrollbar-color: #a8d8ba transparent;
}
@keyframes ecoTabEnter {
    0% { opacity: 0; transform: translateY(12px) scale(.995); filter: blur(2px); }
    100% { opacity: 1; transform: translateY(0) scale(1); filter: blur(0); }
}
.stTabs [data-baseweb="tab-panel"] {
    animation: ecoTabEnter .55s cubic-bezier(.2,.75,.25,1) both;
}
.stTabs [data-baseweb="tab-panel"] [data-testid="stVerticalBlock"] > div {
    transition: transform .22s ease, box-shadow .22s ease;
}
.stTabs [data-baseweb="tab-panel"] [data-testid="stMetric"],
.stTabs [data-baseweb="tab-panel"] [data-testid="stForm"],
.stTabs [data-baseweb="tab-panel"] [data-testid="stExpander"] {
    transition: transform .22s ease, box-shadow .22s ease, border-color .22s ease;
}
.stTabs [data-baseweb="tab-panel"] [data-testid="stForm"]:hover,
.stTabs [data-baseweb="tab-panel"] [data-testid="stExpander"]:hover {
    border-color: #b6dec4;
    box-shadow: 0 12px 30px rgba(18, 91, 58, .07);
}
.stTabs [data-baseweb="tab"]:focus-visible {
    outline: 2px solid #159b68 !important;
    outline-offset: 2px;
}
@media (prefers-reduced-motion: reduce) {
    .stTabs [data-baseweb="tab-panel"] { animation: none !important; }
}

/* Remove the extra default tab border */
.stTabs [data-baseweb="tab-border"] {
    background: transparent !important;
}

/* One clean emerald indicator line */
.stTabs [data-baseweb="tab-highlight"] {
    background-color: #20b879 !important;
    height: 3px !important;
    border-radius: 999px !important;
}




/* Soft green text and number inputs */
.stApp [data-testid="stTextInput"] input,
.stApp [data-testid="stNumberInput"] input,
.stApp textarea {
    background: #eaf6ee !important;
    color: #18382b !important;
    -webkit-text-fill-color: #18382b !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* One clean border around each input */
.stApp [data-testid="stTextInput"] div[data-baseweb="base-input"],
.stApp [data-testid="stNumberInput"] div[data-baseweb="input"] {
    background: #eaf6ee !important;
    border: 1px solid #b8dcc5 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}


/* Clean focus effect for text and number inputs */
.stApp [data-testid="stTextInput"] [data-baseweb="base-input"]:focus-within,
.stApp [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within {
    border: 1px solid #20b879 !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Remove extra focus decorations */
.stApp [data-testid="stTextInput"] [data-baseweb="base-input"]::after,
.stApp [data-testid="stNumberInput"] [data-baseweb="input"]::after {
    border: none !important;
    box-shadow: none !important;
    outline: none !important;
}


/* Matte green number-input controls */
.stApp [data-testid="stNumberInput"] > div {
    background: #eaf6ee !important;
    border: 1px solid #c4e2cf !important;
    border-radius: 12px !important;
    overflow: hidden;
}

.stApp [data-testid="stNumberInput"] button {
    background: #eaf6ee !important;
    color: #087653 !important;
    border: none !important;
    box-shadow: none !important;
}

.stApp [data-testid="stNumberInput"] button:hover {
    background: #d7efdf !important;
    color: #065c40 !important;
}

/* Emerald green buttons — EcoRoute EV */
.stApp button[kind="primary"],
.stApp button[data-testid="stBaseButton-primary"],
.stApp [data-testid="stFormSubmitButton"] button,
.stApp [data-testid="stDownloadButton"] button,
.stApp .stButton button {
    background: linear-gradient(135deg, #20b879 0%, #087653 100%) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: 1px solid #087653 !important;
    border-radius: 12px !important;
    font-weight: 700 !important;
    min-height: 44px;
    box-shadow: 0 5px 14px rgba(8, 118, 83, 0.18);
    transition: all 0.2s ease;
}

/* Keep button text white */
.stApp button[kind="primary"] *,
.stApp button[data-testid="stBaseButton-primary"] *,
.stApp [data-testid="stFormSubmitButton"] button *,
.stApp [data-testid="stDownloadButton"] button *,
.stApp .stButton button * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
}

/* Hover effect */
.stApp button[kind="primary"]:hover,
.stApp button[data-testid="stBaseButton-primary"]:hover,
.stApp [data-testid="stFormSubmitButton"] button:hover,
.stApp [data-testid="stDownloadButton"] button:hover,
.stApp .stButton button:hover {
    background: linear-gradient(135deg, #159b68 0%, #065c40 100%) !important;
    color: #ffffff !important;
    border-color: #065c40 !important;
    transform: translateY(-2px);
    box-shadow: 0 7px 18px rgba(8, 118, 83, 0.22);
}

/* Primary buttons stay emerald on hover */
.stApp .stButton > button[kind="primary"]:hover,
.stApp .stButton > button[data-testid="stBaseButton-primary"]:hover,
.stApp [data-testid="stFormSubmitButton"] button[kind="primary"]:hover,
.stApp [data-testid="stFormSubmitButton"] button[data-testid="stBaseButton-primary"]:hover {
    background: linear-gradient(135deg, #159b68 0%, #065c40 100%) !important;
    color: #ffffff !important;
}

/* Metric cards */
[data-testid="stMetric"] {
    background: rgba(255, 255, 255, 0.92);
    border: 1px solid var(--eco-border);
    border-radius: 18px;
    padding: 18px;
    box-shadow: 0 8px 24px rgba(24, 56, 43, 0.045);
    transition: transform 0.2s ease, box-shadow 0.2s ease;
}

[data-testid="stMetric"]:hover {
    transform: translateY(-2px);
    box-shadow: 0 12px 28px rgba(24, 56, 43, 0.08);
}

[data-testid="stMetricLabel"],
[data-testid="stMetricLabel"] * {
    color: var(--eco-muted) !important;
    font-weight: 600;
}

[data-testid="stMetricValue"],
[data-testid="stMetricValue"] * {
    color: var(--eco-primary-dark) !important;
    font-family: 'Manrope', sans-serif;
    font-weight: 800;
}

/* Forms and containers */
[data-testid="stForm"],
[data-testid="stExpander"] {
    background: rgba(255, 255, 255, 0.88);
    border: 1px solid var(--eco-border);
    border-radius: 18px;
    padding: 1rem;
}

/* Data tables */
[data-testid="stDataFrame"],
[data-testid="stTable"] {
    border: 1px solid var(--eco-border);
    border-radius: 14px;
    overflow: hidden;
}

/* Alerts and progress */
div[data-testid="stAlert"] {
    border-radius: 14px;
}

.stProgress > div > div > div {
    background: linear-gradient(90deg, #13865c, #55d99b);
}

/* Dividers */
hr {
    border-color: var(--eco-border);
}

/* Sliders */
.stSlider {
    color: var(--eco-primary);
}

/* Subtle page entrance */
@keyframes ecoFadeIn {
    from {
        opacity: 0;
        transform: translateY(6px);
    }
    to {
        opacity: 1;
        transform: translateY(0);
    }
}

[data-testid="stVerticalBlock"] {
    animation: ecoFadeIn 0.35s ease-out;
}

/* Respect reduced-motion settings */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation: none !important;
        transition: none !important;
    }
}

/* Mobile-friendly layout */
@media (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem 2rem;
    }

    [data-testid="stMetric"] {
        padding: 12px;
        border-radius: 14px;
    }

    .stTabs [data-baseweb="tab"] {
        padding: 9px 10px;
    }
}

/* EcoRoute EV hero banner */
.eco-hero {
    position: relative;
    display: flex;
    align-items: center;
    justify-content: space-between;
    gap: 2rem;
    overflow: hidden;
    padding: clamp(26px, 5vw, 54px);
    margin-bottom: 28px;
    border: 1px solid #d8ecdf;
    border-radius: 26px;
    background:
        radial-gradient(circle at 85% 20%, #dff8e9 0%, transparent 34%),
        linear-gradient(120deg, #ffffff 0%, #f4fbf6 58%, #eaf8ef 100%);
    box-shadow: 0 16px 42px rgba(24, 72, 48, 0.07);
}

.eco-hero-content {
    position: relative;
    z-index: 2;
    max-width: 650px;
}

.eco-eyebrow {
    display: inline-flex;
    align-items: center;
    gap: 9px;
    padding: 8px 12px;
    margin-bottom: 20px;
    border: 1px solid #d6ebdd;
    border-radius: 999px;
    background: #effaf2;
    color: #176d4a;
    font-size: 11px;
    font-weight: 800;
    letter-spacing: 1.2px;
}

.eco-status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #20b879;
    box-shadow: 0 0 0 4px #d8f5e3;
}

.eco-hero h1 {
    margin: 0 0 10px 0;
    color: #18382b !important;
    font-size: clamp(38px, 5vw, 62px);
    font-weight: 800 !important;
    letter-spacing: -2.5px;
    line-height: 1.08;
}

.eco-hero h1 span {
    color: #13865c;
}

.eco-hero-description {
    margin: 0 0 8px 0;
    color: #244b38 !important;
    font-size: clamp(17px, 2vw, 22px);
    font-weight: 700;
}

.eco-hero-subtitle {
    max-width: 560px;
    margin: 0;
    color: #6b8175 !important;
    font-size: 15px;
    line-height: 1.8;
}

.eco-hero-tags {
    display: flex;
    flex-wrap: wrap;
    gap: 9px;
    margin-top: 24px;
}

.eco-hero-tags span {
    padding: 9px 12px;
    border: 1px solid #dcece1;
    border-radius: 10px;
    background: rgba(255, 255, 255, 0.85);
    color: #315d45;
    font-size: 12px;
    font-weight: 700;
}

.eco-hero-visual {
    position: relative;
    display: flex;
    flex: 0 0 230px;
    align-items: center;
    justify-content: center;
    width: 230px;
    height: 230px;
}

.eco-orbit {
    position: absolute;
    width: 185px;
    height: 185px;
    border: 1px solid rgba(19, 134, 92, 0.2);
    border-radius: 50%;
}

.eco-orbit-two {
    width: 225px;
    height: 225px;
    border-style: dashed;
    border-color: rgba(19, 134, 92, 0.16);
}

.eco-hero-icon {
    display: flex;
    align-items: center;
    justify-content: center;
    width: 112px;
    height: 112px;
    border: 1px solid #c9ead5;
    border-radius: 32px;
    background: linear-gradient(145deg, #ffffff, #e0f6e8);
    box-shadow: 0 18px 36px rgba(19, 134, 92, 0.13);
    font-size: 54px;
    animation: ecoFloat 4s ease-in-out infinite;
}

.eco-floating-card {
    position: absolute;
    right: -8px;
    bottom: 8px;
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px 14px;
    border: 1px solid #dcece1;
    border-radius: 14px;
    background: rgba(255, 255, 255, 0.96);
    box-shadow: 0 10px 28px rgba(24, 72, 48, 0.09);
}

.eco-floating-icon {
    font-size: 24px;
}

.eco-floating-card strong,
.eco-floating-card small {
    display: block;
    white-space: nowrap;
}

.eco-floating-card strong {
    color: #18382b;
    font-size: 12px;
}

.eco-floating-card small {
    margin-top: 3px;
    color: #718579;
    font-size: 10px;
}

@keyframes ecoFloat {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-7px); }
}

/* Subtle EV motion accents - keeps the existing light-green visual style */
.eco-ev-motion {
    position: relative; height: 48px; overflow: hidden; margin-top: 12px;
    border: 1px solid #c9ead5; border-radius: 14px;
    background: linear-gradient(180deg, #eef9f1, #dff2e5);
}
.eco-ev-motion .road-line {
    position: absolute; left: 0; right: 0; top: 31px;
    border-top: 2px dashed #9bc8aa;
}
.eco-ev-motion .moving-car {
    position: absolute; top: 2px; left: -45px; font-size: 27px;
    animation: ecoDriveAcross 8s linear infinite;
}
.eco-ev-motion .charging-icon {
    position: absolute; right: 12px; top: 8px; font-size: 24px;
    animation: ecoChargePulse 1.6s ease-in-out infinite;
}
@keyframes ecoDriveAcross { from { left: -45px; } to { left: calc(100% - 42px); } }
@keyframes ecoChargePulse { 0%,100% { transform: scale(1); opacity: .7; } 50% { transform: scale(1.2); opacity: 1; } }
.stApp [data-testid="stMetric"] { transition: transform .2s ease, box-shadow .2s ease; }
.stApp [data-testid="stMetric"]:hover { transform: translateY(-2px); }
@media (prefers-reduced-motion: reduce) {
    .eco-ev-motion .moving-car, .eco-ev-motion .charging-icon { animation: none; }
}

@media (max-width: 760px) {
    .eco-hero {
        padding: 26px 20px;
    }

    .eco-hero-visual {
        display: none;
    }

    .eco-hero h1 {
        font-size: 42px;
    }

    .eco-hero-tags {
        gap: 7px;
    }
}

@media (prefers-reduced-motion: reduce) {
    .eco-hero-icon {
        animation: none;
    }
}


/* Input box background and border */
.stApp [data-testid="stTextInput"] [data-baseweb="base-input"],
.stApp [data-testid="stNumberInput"] [data-baseweb="input"],
.stApp [data-testid="stNumberInput"] [data-baseweb="base-input"] {
    background-color: #eaf6ee !important;
    border: 1px solid #b8dcc5 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

/* Input text */
.stApp [data-testid="stTextInput"] input,
.stApp [data-testid="stNumberInput"] input {
    background-color: transparent !important;
    color: #18382b !important;
    -webkit-text-fill-color: #18382b !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Focus border */
.stApp [data-testid="stTextInput"] [data-baseweb="base-input"]:focus-within,
.stApp [data-testid="stNumberInput"] [data-baseweb="input"]:focus-within,
.stApp [data-testid="stNumberInput"] [data-baseweb="base-input"]:focus-within {
    border-color: #20b879 !important;
    box-shadow: 0 0 0 2px rgba(32, 184, 121, 0.12) !important;
}
.stTextInput input {
    background-color: #eaf6ee !important;
    color: #18382b !important;
    -webkit-text-fill-color: #18382b !important;
    border: 1px solid #b8dcc5 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}
/* Fix Starting location and Destination */
.stApp div[data-testid="stTextInput"] input {
    background-color: #eaf6ee !important;
    color: #18382b !important;
    -webkit-text-fill-color: #18382b !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

.stApp div[data-testid="stTextInput"] div[data-baseweb="base-input"] {
    background-color: #eaf6ee !important;
    border: 1px solid #b8dcc5 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}
/* Fix Battery capacity input */
.stApp [data-testid="stNumberInput"] input {
    background-color: #eaf6ee !important;
    color: #18382b !important;
    -webkit-text-fill-color: #18382b !important;
    border: none !important;
    box-shadow: none !important;
}

.stApp [data-testid="stNumberInput"] [data-baseweb="input"] {
    background-color: #eaf6ee !important;
    border: 1px solid #b8dcc5 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}
/* Remove black borders from all three inputs */
.stApp [data-testid="stTextInput"] input,
.stApp [data-testid="stNumberInput"] input,
.stApp [data-testid="stTextInput"] [data-baseweb="base-input"],
.stApp [data-testid="stNumberInput"] [data-baseweb="input"],
.stApp [data-testid="stNumberInput"] [data-baseweb="base-input"] {
    border-color: #b8dcc5 !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Keep the number input wrapper green */
.stApp [data-testid="stNumberInput"] [data-baseweb="input"] {
    background-color: #eaf6ee !important;
    border: 1px solid #b8dcc5 !important;
    border-radius: 12px !important;
}
/* Remove double lines when input is clicked */
.stApp [data-testid="stTextInput"] div[data-baseweb="base-input"]:focus-within,
.stApp [data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
.stApp [data-testid="stNumberInput"] div[data-baseweb="base-input"]:focus-within {
    border: 1px solid #b8dcc5 !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Fix double border on location inputs */
.stApp [data-testid="stTextInputRootElement"] {
    border: 1px solid #b8dcc5 !important;
    border-radius: 12px !important;
    outline: none !important;
    box-shadow: none !important;
    background: #eaf6ee !important;
}

.stApp [data-testid="stTextInputRootElement"]:focus-within {
    border: 1px solid #20b879 !important;
    outline: none !important;
    box-shadow: none !important;
}

.stApp [data-testid="stTextInputRootElement"] input {
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
    background: transparent !important;
}


/* Battery capacity - target actual Streamlit container */
.stApp [data-testid="stNumberInputContainer"] {
    background: #eaf6ee !important;
    border: 1px solid #b8dcc5 !important;
    border-radius: 12px !important;
    outline: none !important;
    box-shadow: none !important;
    overflow: hidden !important;
}

/* Battery number field */
.stApp [data-testid="stNumberInputField"] {
    background: #eaf6ee !important;
    color: #18382b !important;
    -webkit-text-fill-color: #18382b !important;
    border: none !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Focus state */
.stApp [data-testid="stNumberInputContainer"]:focus-within {
    border: 1px solid #20b879 !important;
    outline: none !important;
    box-shadow: none !important;
}

/* Battery increment/decrement buttons */
.stApp [data-testid="stNumberInputStepDown"],
.stApp [data-testid="stNumberInputStepUp"] {
    background: #eaf6ee !important;
    color: #087653 !important;
    border: none !important;
    box-shadow: none !important;
}
/* EcoRoute EV selectbox: force a light control in Streamlit/BaseWeb */
.stApp [data-testid="stSelectbox"] {
    color-scheme: light !important;
}

.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div,
.stApp [data-testid="stSelectbox"] [role="combobox"] {
    background: #eaf6ee !important;
    background-color: #eaf6ee !important;
    color: #18382b !important;
    border: 1px solid #b8dcc5 !important;
    border-radius: 12px !important;
    box-shadow: none !important;
}

.stApp [data-testid="stSelectbox"] [data-baseweb="select"] input,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] span,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] div {
    color: #18382b !important;
    -webkit-text-fill-color: #18382b !important;
}

.stApp [data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill: #18382b !important;
    color: #18382b !important;
}

.stApp [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within > div {
    border-color: #20b879 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(32, 184, 121, 0.15) !important;
}

/* Dropdown option list */
div[data-baseweb="popover"],
div[data-baseweb="popover"] [data-baseweb="menu"],
div[data-baseweb="menu"] li {
    background: #ffffff !important;
    color: #18382b !important;
}

/* Fix complete dropdown background and black border */
.stApp [data-testid="stSelectbox"] [data-baseweb="select"],
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div > div,
.stApp [data-testid="stSelectbox"] [data-baseweb="select"] > div > div > div {
    background-color: #eaf6ee !important;
    border-color: #b8dcc5 !important;
    color: #18382b !important;
    box-shadow: none !important;
}

.stApp [data-testid="stSelectbox"] [data-baseweb="select"] * {
    color: #18382b !important;
    -webkit-text-fill-color: #18382b !important;
}

.stApp [data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill: #18382b !important;
    color: #18382b !important;
}

.stApp [data-testid="stSelectbox"] [data-baseweb="select"]:focus-within {
    outline: none !important;
    box-shadow: none !important;
}
/* ===== SELECTBOX: one green box, nothing dark inside ===== */
[data-testid="stSelectbox"] [data-baseweb="select"] {
    background: #eaf6ee !important;
    border: 1px solid #b8dcc5 !important;
    border-radius: 12px !important;
    overflow: hidden !important;
    box-shadow: none !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] *,
[data-testid="stSelectbox"] [data-baseweb="select"] > div,
[data-testid="stSelectbox"] [role="combobox"],
[data-testid="stSelectbox"] [role="combobox"] > div {
    background: transparent !important;
    border: none !important;
    border-radius: 0 !important;
    outline: none !important;
    box-shadow: none !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"]:focus-within {
    border: 1px solid #20b879 !important;
    box-shadow: 0 0 0 2px rgba(32, 184, 121, 0.15) !important;
}
[data-testid="stSelectbox"] [data-baseweb="select"] svg {
    fill: #18382b !important;
    color: #18382b !important;
}

/* ===== DROPDOWN LIST (opens below the box) ===== */
div[data-baseweb="popover"] {
    background: transparent !important;
}
div[data-baseweb="popover"] > div,
div[data-baseweb="popover"] [data-baseweb="menu"],
div[data-baseweb="popover"] ul,
div[data-baseweb="popover"] [role="listbox"] {
    background: #ffffff !important;
    border: 1px solid #b8dcc5 !important;
    border-radius: 12px !important;
    box-shadow: 0 12px 30px rgba(24, 56, 43, 0.12) !important;
}
div[data-baseweb="popover"] li,
div[data-baseweb="popover"] [role="option"] {
    background: #ffffff !important;
    color: #18382b !important;
    -webkit-text-fill-color: #18382b !important;
}
div[data-baseweb="popover"] li:hover,
div[data-baseweb="popover"] [role="option"]:hover,
div[data-baseweb="popover"] li[aria-selected="true"] {
    background: #eaf6ee !important;
    color: #087653 !important;
    -webkit-text-fill-color: #087653 !important;
}

/* ===== BLACK LINES (dividers, containers, tab bar) ===== */
.stApp hr,
.stApp [data-testid="stDivider"] hr {
    border: none !important;
    border-top: 1px solid #dfebe3 !important;
    background: transparent !important;
    height: 0 !important;
}
.stApp [data-testid="stVerticalBlockBorderWrapper"] {
    border-color: #dfebe3 !important;
}
.stTabs [data-baseweb="tab-border"] {
    background: transparent !important;
    border: none !important;
}


/* ===== Stitch-inspired EcoRoute EV desktop polish ===== */
:root {
    --stitch-forest: #064e3b;
    --stitch-emerald: #10b981;
    --stitch-pine: #134e4a;
    --stitch-canvas: #f8fafc;
    --stitch-mint: #ecfdf5;
    --stitch-ink: #1e293b;
    --stitch-line: rgba(6, 78, 59, .10);
    --stitch-shadow: 0 4px 20px -2px rgba(6, 78, 59, .06), 0 2px 6px -1px rgba(0,0,0,.025);
}
.stApp {
    background:
      radial-gradient(ellipse at 0% 0%, rgba(16,185,129,.09), transparent 34%),
      linear-gradient(180deg, #f8fafc 0%, #f4faf6 100%) !important;
    color: var(--stitch-ink) !important;
}
.block-container { max-width: 1500px !important; padding-top: 1.15rem !important; }
h1, h2, h3, h4 { font-family: 'Manrope', 'DM Sans', sans-serif !important; }
.stApp [data-testid="stMetric"],
.stApp [data-testid="stForm"],
.stApp [data-testid="stExpander"],
.stApp [data-testid="stDataFrame"],
.stApp [data-testid="stPlotlyChart"],
.stApp [data-testid="stImage"] {
    border: 1px solid var(--stitch-line) !important;
    border-radius: 16px !important;
    box-shadow: var(--stitch-shadow) !important;
    background: rgba(255,255,255,.92) !important;
}
.stApp [data-testid="stMetric"] { padding: 15px 16px !important; }
.stApp [data-testid="stMetricLabel"] { color: #64748b !important; font-weight: 600 !important; }
.stApp [data-testid="stMetricValue"] { color: var(--stitch-forest) !important; font-weight: 800 !important; }
.stApp [data-testid="stForm"] { padding: 18px !important; }
.stApp [data-testid="stDataFrame"] { overflow: hidden; }
.stApp [data-testid="stAlert"] { border-radius: 14px !important; }
.stApp button[kind="primary"], .stApp button[data-testid="stBaseButton-primary"],
.stApp [data-testid="stFormSubmitButton"] button {
    background: linear-gradient(120deg, #0b6b4d, #10b981) !important;
    border: 1px solid #087653 !important;
    box-shadow: 0 5px 16px rgba(6,78,59,.16) !important;
}
.stApp button[kind="primary"]:hover, .stApp button[data-testid="stBaseButton-primary"]:hover,
.stApp [data-testid="stFormSubmitButton"] button:hover {
    transform: translateY(-1px); filter: brightness(1.04);
    box-shadow: 0 9px 22px rgba(6,78,59,.20) !important;
}
.stTabs [data-baseweb="tab-list"] {
    background: rgba(255,255,255,.82) !important;
    border: 1px solid var(--stitch-line) !important;
    box-shadow: 0 5px 18px rgba(6,78,59,.045) !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] {
    background: var(--stitch-mint) !important;
    border-color: rgba(16,185,129,.25) !important;
}
.stTabs [data-baseweb="tab"][aria-selected="true"] p { color: var(--stitch-forest) !important; }
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #f0fdf4 0%, #ffffff 76%) !important;
    border-right: 1px solid var(--stitch-line) !important;
}
.stApp input, .stApp textarea { border-radius: 10px !important; }
@keyframes stitchCardIn {
    from { opacity: 0; transform: translateY(8px); }
    to { opacity: 1; transform: translateY(0); }
}
.stTabs [data-baseweb="tab-panel"] { animation: stitchCardIn .35s ease-out both; }
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after { animation-duration: .01ms !important; transition-duration: .01ms !important; }
}

/* Stitch-inspired EcoRoute EV visual system */
:root {
  --stitch-primary: #064e3b;
  --stitch-accent: #10b981;
  --stitch-canvas: #f8fafc;
  --stitch-mint: #ecfdf5;
  --stitch-ink: #1e293b;
  --stitch-muted: #64748b;
  --stitch-line: rgba(6,78,59,.10);
}
.stApp { background: radial-gradient(ellipse at 12% 0%, #e8f8ef 0%, transparent 34%), #f8fafc !important; color:var(--stitch-ink)!important; }
.block-container { max-width: 1500px !important; padding-top: 1.2rem !important; }
h1,h2,h3,[data-testid="stMetricValue"] { font-family:'Plus Jakarta Sans','DM Sans',sans-serif!important; letter-spacing:-.025em!important; }
h1,h2,h3 { color:var(--stitch-primary)!important; }
.stTabs [data-baseweb="tab-list"] { gap:7px!important; padding:9px!important; border:1px solid var(--stitch-line)!important; border-radius:16px!important; background:rgba(255,255,255,.82)!important; box-shadow:0 5px 22px rgba(6,78,59,.045)!important; }
.stTabs [data-baseweb="tab"] { border-radius:11px!important; padding:10px 14px!important; transition:all .22s ease!important; }
.stTabs [data-baseweb="tab"]:hover { background:#ecfdf5!important; transform:translateY(-2px)!important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] { background:#064e3b!important; border:1px solid #064e3b!important; }
.stTabs [data-baseweb="tab"][aria-selected="true"] p { color:#fff!important; }
.stTabs [data-baseweb="tab-panel"] { animation: stitchFade .42s ease both; }
@keyframes stitchFade { from {opacity:0;transform:translateY(9px)} to {opacity:1;transform:translateY(0)} }
[data-testid="stMetric"], [data-testid="stForm"], [data-testid="stExpander"], [data-testid="stDataFrame"] { background:#fff!important; border:1px solid var(--stitch-line)!important; border-radius:16px!important; padding:12px!important; box-shadow:0 5px 20px rgba(6,78,59,.035)!important; }
.stApp input,.stApp textarea,[data-baseweb="select"] > div { border-radius:11px!important; }
.stApp button[kind="primary"],.stApp [data-testid="stFormSubmitButton"] button,.stApp [data-testid="stDownloadButton"] button,.stApp .stButton button { background:linear-gradient(135deg,#087653,#064e3b)!important; border:0!important; border-radius:11px!important; box-shadow:0 5px 14px rgba(6,78,59,.16)!important; }
[data-testid="stSidebar"] { background:linear-gradient(180deg,#f0fdf4,#fff)!important; border-right:1px solid var(--stitch-line)!important; }
@media (prefers-reduced-motion:reduce) { .stTabs [data-baseweb="tab-panel"] {animation:none!important;} }
</style>
""", unsafe_allow_html=True)



# ---------------------------
# EcoRoute EV — Light Hero Banner
# ---------------------------
st.html("""
<div class="eco-hero">
    <div class="eco-hero-content">
        <div class="eco-eyebrow">
            <span class="eco-status-dot"></span>
            ELECTRIC MOBILITY · MACHINE LEARNING
        </div>

        <h1>EcoRoute <span>EV</span></h1>

        <p class="eco-hero-description">
            Smarter routes. Better energy decisions.
        </p>

        <p class="eco-hero-subtitle">
            Plan your journey, estimate battery consumption, and explore
            a more sustainable way to drive.
        </p>

        <div class="eco-hero-tags">
            <span>✦ Smart Routing</span>
            <span>⚡ Energy Prediction</span>
            <span>🌿 Eco Driving</span>
        </div>
    </div>

    <div class="eco-hero-visual" aria-hidden="true">
        <div class="eco-orbit eco-orbit-one"></div>
        <div class="eco-orbit eco-orbit-two"></div>
        <div class="eco-hero-icon">⚡</div>
        <div class="eco-floating-card">
            <span class="eco-floating-icon">🌱</span>
            <span>
                <strong>Drive smarter</strong>
                <small>Energy-aware journeys</small>
            </span>
        </div>
        <div class="eco-ev-motion">
            <div class="road-line"></div>
            <div class="moving-car">🚙</div>
            <div class="charging-icon">⚡</div>
        </div>
    </div>
</div>
""")


# ---------------------------
# Demonstration ML model
# ---------------------------



# ---------------------------
# Demonstration ML model
# ---------------------------
@st.cache_resource
def train_model():
    rng = np.random.default_rng(42)
    n = 1500
    speed = rng.uniform(10, 120, n)
    temperature = rng.uniform(5, 40, n)
    traffic = rng.uniform(0, 1, n)
    driving_style = rng.uniform(0, 1, n)
    distance = rng.uniform(1, 200, n)
    energy = distance * (
        0.14 + speed * 0.0015 + traffic * 0.06
        + driving_style * 0.08 + np.abs(22 - temperature) * 0.001
    )
    X = pd.DataFrame({
        "speed": speed, "temperature": temperature, "traffic": traffic,
        "driving_style": driving_style, "distance": distance,
    })
    X_train, X_test, y_train, y_test = train_test_split(
        X, energy, test_size=0.2, random_state=42
    )
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, mae

model, mae = train_model()

# Keep results in session state so Streamlit reruns don't erase them.
if "route_result" not in st.session_state:
    st.session_state.route_result = None
if "energy_result" not in st.session_state:
    st.session_state.energy_result = None

st.markdown("""
<div class="eco-stitch-hero">
  <div class="eco-stitch-hero-copy">
    <div class="eco-stitch-eyebrow"><span class="eco-live-dot"></span> ELECTRIC MOBILITY • SMARTER JOURNEYS</div>
    <div class="eco-stitch-title">EcoRoute <span>EV</span></div>
    <div class="eco-stitch-subtitle">Plan smarter routes. Find charging stops. Understand every kilometre.</div>
  </div>
  <div class="eco-stitch-road"><div class="eco-road-line"></div><div class="eco-car">🚙</div><div class="eco-charge">⚡</div></div>
</div>
<style>
.eco-stitch-hero{position:relative;display:flex;align-items:center;justify-content:space-between;gap:24px;overflow:hidden;margin:4px 0 22px;padding:25px 30px;border-radius:22px;background:linear-gradient(112deg,#064e3b 0%,#0b6b4d 58%,#134e4a 100%);color:white;box-shadow:0 12px 32px rgba(6,78,59,.13)}
.eco-stitch-eyebrow{font:700 10px 'Inter',sans-serif;letter-spacing:2px;color:#a7f3d0;display:flex;align-items:center;gap:9px}.eco-live-dot{width:8px;height:8px;border-radius:50%;background:#6ffbbe;box-shadow:0 0 0 5px rgba(111,251,190,.14);animation:ecoPulse 1.8s infinite}
.eco-stitch-title{font:800 clamp(30px,4vw,43px) 'Plus Jakarta Sans',sans-serif;letter-spacing:-2px;margin-top:8px;color:#fff}.eco-stitch-title span{color:#6ffbbe}.eco-stitch-subtitle{font:400 14px 'Inter',sans-serif;color:#d1fae5;margin-top:5px}
.eco-stitch-road{position:relative;width:34%;min-width:210px;height:75px;border-bottom:3px dashed rgba(255,255,255,.32);transform:skewY(-5deg)}.eco-car{position:absolute;left:12%;top:13px;font-size:35px;filter:drop-shadow(0 5px 5px rgba(0,0,0,.2));animation:ecoDrive 6s ease-in-out infinite alternate}.eco-charge{position:absolute;right:5%;top:8px;font-size:27px;color:#a7f3d0;animation:ecoPulse 1.6s infinite}.eco-road-line{position:absolute;left:0;right:0;bottom:13px;border-bottom:1px solid rgba(167,243,208,.35)}
@keyframes ecoDrive{from{transform:translateX(0)}to{transform:translateX(145px)}}@keyframes ecoPulse{50%{opacity:.55;transform:scale(1.12)}}
@media(max-width:700px){.eco-stitch-hero{padding:20px}.eco-stitch-road{display:none}}
@media(prefers-reduced-motion:reduce){.eco-car,.eco-live-dot,.eco-charge{animation:none}}
/* Functional dashboard and route-comparison cards inspired by Stitch exports */
.eco-dashboard-welcome{display:flex;align-items:center;justify-content:space-between;gap:20px;padding:23px 25px;margin:6px 0 18px;border-radius:20px;background:linear-gradient(115deg,#064e3b,#0b6b4d 62%,#13865c);color:#fff;box-shadow:0 12px 28px rgba(6,78,59,.12)}
.eco-dashboard-kicker{font-size:10px;font-weight:800;letter-spacing:1.8px;color:#a7f3d0}.eco-dashboard-heading{font:800 clamp(23px,3vw,32px) 'Plus Jakarta Sans',sans-serif;color:#fff;margin-top:5px;letter-spacing:-1px}.eco-dashboard-copy{font-size:13px;color:#d1fae5;margin-top:5px}
.eco-dashboard-battery{display:flex;align-items:center;gap:11px;padding:12px 15px;border:1px solid rgba(255,255,255,.2);border-radius:14px;background:rgba(255,255,255,.10);min-width:180px}.eco-dashboard-battery>span{font-size:27px}.eco-dashboard-battery b,.eco-dashboard-battery small{display:block;color:#fff}.eco-dashboard-battery small{font-size:11px;color:#d1fae5;margin-top:2px}
.eco-route-summary{padding:16px 18px;border:1px solid #dcebe2;background:#fff;border-radius:15px;margin:10px 0 12px}.eco-route-point{display:flex;align-items:center;gap:12px}.eco-route-point small,.eco-route-point b{display:block}.eco-route-point small{font-size:9px;letter-spacing:1px;color:#789083;font-weight:800}.eco-route-point b{font-size:14px;color:#064e3b}.eco-route-dot{width:10px;height:10px;border-radius:50%;background:#10b981;box-shadow:0 0 0 4px #d1fae5}.eco-route-pin{color:#0b6b4d;font-size:15px}.eco-route-connector{height:18px;border-left:2px dashed #b8d9c5;margin:3px 0 3px 4px}
.eco-feature-stack{display:grid;gap:9px;margin-top:10px}.eco-feature-row{display:flex;align-items:center;gap:10px;padding:11px 12px;background:#fff;border:1px solid #e1eee6;border-radius:13px}.eco-feature-row>span{font-size:20px}.eco-feature-row b,.eco-feature-row small{display:block}.eco-feature-row b{font-size:12px;color:#164b36}.eco-feature-row small{font-size:10px;color:#718579;margin-top:2px}.eco-feature-row em{margin-left:auto;font-style:normal;font-size:9px;font-weight:800;color:#087653;background:#ecfdf5;padding:5px 7px;border-radius:20px;white-space:nowrap}
.eco-option-card{height:100%;padding:15px 16px;border:1px solid #dcebe2;border-radius:15px;background:#fff;box-shadow:0 4px 15px rgba(6,78,59,.04);margin:4px 0 12px;transition:transform .2s ease,box-shadow .2s ease}.eco-option-card:hover{transform:translateY(-3px);box-shadow:0 10px 24px rgba(6,78,59,.09)}.eco-option-best{border:1.5px solid #10b981;background:linear-gradient(145deg,#f0fdf4,#fff)}.eco-option-top{display:flex;align-items:center;justify-content:space-between;gap:7px}.eco-option-top b{font-size:13px;color:#164b36}.eco-option-top span{font-size:8px;font-weight:800;letter-spacing:.5px;color:#087653;background:#d1fae5;padding:5px 7px;border-radius:20px;white-space:nowrap}.eco-option-energy{font:800 29px 'Plus Jakarta Sans',sans-serif;color:#064e3b;margin-top:13px}.eco-option-energy small{font:600 12px 'DM Sans',sans-serif;color:#789083}.eco-option-meta{font-size:11px;color:#718579;margin-top:2px}
@media(max-width:700px){.eco-dashboard-welcome{align-items:flex-start;flex-direction:column}.eco-dashboard-battery{width:100%}.eco-feature-row{flex-wrap:wrap}}
</style>
""", unsafe_allow_html=True)

dashboard_tab, route_tab, energy_tab, history_tab, weather_tab, analytics_tab, charging_tab, compare_tab, carbon_tab, profile_tab, about_tab = st.tabs([
    "🏡 Overview", "🗺️ Route Planner", "⚡ Energy Lab", "📊 Trip History",
    "🌦️ Weather Lab", "📈 Analytics", "🔌 Charging Hubs", "🔋 Energy Compare",
    "🌍 Carbon Savings", "🚗 My EV", "ℹ️ About"
])

# ---------------------------
# Dashboard overview — functional Stitch dashboard screen
# ---------------------------
with dashboard_tab:
    st.markdown("## Your mobility overview")
    st.caption("A live summary built from your saved journeys and the latest route you planned.")
    dashboard_history = load_trip_history()
    dashboard_route = st.session_state.route_result
    dashboard_energy = st.session_state.energy_result
    total_saved_trips = len(dashboard_history)
    total_saved_distance = float(dashboard_history["distance_km"].sum()) if total_saved_trips else 0.0
    total_saved_energy = float(dashboard_history["energy_kwh"].sum()) if total_saved_trips else 0.0
    current_vehicle = st.session_state.get("ev_vehicle_name", "My EV")
    current_capacity = float(st.session_state.get("ev_battery_capacity", 60.0))
    current_tariff = float(st.session_state.get("ev_electricity_tariff", 8.0))

    st.markdown(
        f"<div class='eco-dashboard-welcome'><div><div class='eco-dashboard-kicker'>PERSONAL EV CONTROL CENTRE</div>"
        f"<div class='eco-dashboard-heading'>Welcome back, EV driver</div>"
        f"<div class='eco-dashboard-copy'>Your journeys, energy insights and charging plan — all in one place.</div></div>"
        f"<div class='eco-dashboard-battery'><span>🔋</span><div><b>{html.escape(str(current_vehicle))}</b><small>{current_capacity:.0f} kWh battery profile</small></div></div></div>",
        unsafe_allow_html=True,
    )

    d1, d2, d3, d4 = st.columns(4)
    d1.metric("Saved journeys", f"{total_saved_trips}", help="Trips stored in the local SQLite history.")
    d2.metric("Distance covered", f"{total_saved_distance:,.1f} km")
    d3.metric("Estimated energy", f"{total_saved_energy:,.1f} kWh")
    d4.metric("Electricity tariff", f"₹{current_tariff:.2f}/kWh")

    left_dash, right_dash = st.columns([1.35, 1], gap="large")
    with left_dash:
        st.markdown("### Latest journey")
        if dashboard_route:
            dr1, dr2, dr3 = st.columns(3)
            dr1.metric("Distance", f"{dashboard_route['distance_km']:.1f} km")
            dr2.metric("Travel time", f"{dashboard_route['duration_min']:.0f} min")
            dr3.metric("Energy estimate", f"{dashboard_route['predicted_energy']:.1f} kWh")
            st.markdown(
                f"<div class='eco-route-summary'><div class='eco-route-point'><span class='eco-route-dot'></span>"
                f"<div><small>FROM</small><b>{html.escape(str(dashboard_route['origin']))}</b></div></div>"
                f"<div class='eco-route-connector'></div><div class='eco-route-point'><span class='eco-route-pin'>●</span>"
                f"<div><small>TO</small><b>{html.escape(str(dashboard_route['destination']))}</b></div></div></div>",
                unsafe_allow_html=True,
            )
            st.caption(f"Recommended option: {dashboard_route['recommended_route_name']}. Estimates are experimental and use the project's synthetic-data model.")
        else:
            st.info("No active journey yet. Open Route Planner to choose a start point and destination.")
        st.markdown("### Recent journey activity")
        if total_saved_trips:
            recent = dashboard_history.head(6).copy()
            recent["Trip"] = recent["origin"].astype(str) + " → " + recent["destination"].astype(str)
            st.bar_chart(recent.set_index("Trip")[["distance_km"]], height=220)
        else:
            st.caption("Your distance chart will appear here after you save your first route.")

    with right_dash:
        st.markdown("### Energy snapshot")
        if dashboard_energy:
            st.metric("Last lab prediction", f"{dashboard_energy['predicted']:.2f} kWh")
            st.caption(f"Based on {dashboard_energy['distance']} km, {dashboard_energy['speed']} km/h and {dashboard_energy['traffic'].lower()} traffic.")
        elif dashboard_route:
            st.metric("Current route energy", f"{dashboard_route['predicted_energy']:.2f} kWh")
            st.caption("Calculated for your latest planned journey.")
        else:
            st.metric("Energy prediction", "Ready")
            st.caption("Run an estimate in Energy Lab to see your personalised snapshot.")
        st.markdown(
            "<div class='eco-feature-stack'>"
            "<div class='eco-feature-row'><span>🌦️</span><div><b>Weather impact</b><small>Temperature, rain and wind adjustment</small></div><em>Weather Lab</em></div>"
            "<div class='eco-feature-row'><span>⚡</span><div><b>Smart charging</b><small>Mapped hubs and estimated stop planning</small></div><em>Charging</em></div>"
            "<div class='eco-feature-row'><span>🌿</span><div><b>Carbon savings</b><small>EV electricity vs petrol tailpipe estimate</small></div><em>Carbon</em></div>"
            "<div class='eco-feature-row'><span>🤖</span><div><b>Local EV assistant</b><small>Free keyword-based help, no API key</small></div><em>Assistant</em></div>"
            "</div>",
            unsafe_allow_html=True,
        )

    if total_saved_trips:
        st.markdown("### Your recent trips")
        recent_table = dashboard_history.head(5).rename(columns={
            "trip_time": "Date", "origin": "From", "destination": "To",
            "distance_km": "Distance (km)", "duration_min": "Time (min)",
            "energy_kwh": "Energy (kWh)", "route_name": "Recommended route",
        }).drop(columns=["id"])
        st.dataframe(recent_table, use_container_width=True, hide_index=True)

# ---------------------------
# Route planner
# ---------------------------
with route_tab:
    st.header("Plan your next journey")
    st.caption(
        "Choose your locations and battery level to estimate your EV journey."
    )

    with st.form("route_form"):
        col1, col2 = st.columns(2)

        with col1:
            origin = st.text_input(
                "Starting location",
                value="Chennai",
                placeholder="e.g. Chennai"
            )

        with col2:
            destination = st.text_input(
                "Destination",
                value="Mahabalipuram",
                placeholder="e.g. Mahabalipuram"
            )

        col3, col4 = st.columns(2)

        with col3:
            battery_capacity = st.number_input(
                "Battery capacity (kWh)",
                min_value=10.0,
                max_value=200.0,
                value=60.0,
                step=5.0
            )

        with col4:
            battery_percent = st.slider(
                "Current battery (%)", 0, 100, 80
            )

        col5, col6, col7 = st.columns(3)
        with col5:
            reserve_percent = st.slider(
                "Minimum arrival reserve (%)", 5, 30, 15,
                help="The planner tries to keep at least this battery level when reaching a charger or destination."
            )
        with col6:
            target_charge_percent = st.slider(
                "Charge target at stop (%)", 50, 100, 80
            )
        with col7:
            assumed_charger_power = st.number_input(
                "Assumed charger power (kW)", min_value=3.0, max_value=350.0,
                value=50.0, step=5.0,
                help="Used only to estimate charging time when station power is not available in map data."
            )

        submitted_route = st.form_submit_button(
            "Find route and estimate energy",
            type="primary",
            use_container_width=True
        )

    if submitted_route:
        if not origin.strip() or not destination.strip():
            st.session_state.route_result = None
            st.error("Please enter both a starting location and a destination.")
        elif origin.strip().casefold() == destination.strip().casefold():
            st.session_state.route_result = None
            st.warning("Choose two different locations for your journey.")
        else:
            with st.spinner("Finding locations and comparing routes…"):
                try:
                    def geocode(place):
                        response = requests.get(
                            "https://nominatim.openstreetmap.org/search",
                            params={"q": place, "format": "json", "limit": 1},
                            headers={"User-Agent": "EcoRouteEV-Student-Project/1.0 (educational demo)"},
                            timeout=20,
                        )
                        response.raise_for_status()
                        results = response.json()
                        if not results:
                            raise ValueError(
                                f"Location not found: {place}. Try adding the city or state."
                            )
                        return float(results[0]["lat"]), float(results[0]["lon"])

                    start = geocode(origin.strip())
                    end = geocode(destination.strip())
                    route_response = requests.get(
                        f"https://router.project-osrm.org/route/v1/driving/{start[1]},{start[0]};{end[1]},{end[0]}",
                        params={
                            "overview": "full",
                            "geometries": "geojson",
                            "alternatives": "true",
                            "steps": "false",
                        },
                        timeout=30,
                    )
                    route_response.raise_for_status()
                    route_data = route_response.json()
                    if route_data.get("code") != "Ok" or not route_data.get("routes"):
                        raise ValueError("No driving route was returned for these locations.")

                    # Compare every route returned by OSRM using the demonstration ML model.
                    route_options = []
                    for index, option in enumerate(route_data["routes"], start=1):
                        option_distance_km = option["distance"] / 1000
                        option_duration_min = option["duration"] / 60
                        option_features = pd.DataFrame([{
                            "speed": 45,
                            "temperature": 28,
                            "traffic": 0.4,
                            "driving_style": 0.3,
                            "distance": option_distance_km,
                        }])
                        option_energy = max(
                            0.0, float(model.predict(option_features)[0])
                        )
                        route_options.append({
                            "name": f"Route {index}",
                            "route": option,
                            "distance_km": option_distance_km,
                            "duration_min": option_duration_min,
                            "predicted_energy": option_energy,
                        })

                    best_option = min(
                        route_options,
                        key=lambda item: item["predicted_energy"],
                    )
                    route = best_option["route"]
                    available_energy = battery_capacity * battery_percent / 100
                    remaining_energy = available_energy - best_option["predicted_energy"]

                    st.session_state.route_result = {
                        "origin": origin.strip(),
                        "destination": destination.strip(),
                        "start": start,
                        "end": end,
                        "geometry": route["geometry"],
                        "distance_km": best_option["distance_km"],
                        "duration_min": best_option["duration_min"],
                        "predicted_energy": best_option["predicted_energy"],
                        "battery_capacity": battery_capacity,
                        "battery_percent": battery_percent,
                        "reserve_percent": reserve_percent,
                        "target_charge_percent": target_charge_percent,
                        "assumed_charger_power": assumed_charger_power,
                        "available_energy": available_energy,
                        "remaining_energy": remaining_energy,
                        "alternatives_count": len(route_options),
                        "route_options": [
                            {
                                "name": item["name"],
                                "distance_km": item["distance_km"],
                                "duration_min": item["duration_min"],
                                "predicted_energy": item["predicted_energy"],
                                "is_recommended": item["name"] == best_option["name"],
                            }
                            for item in route_options
                        ],
                        "recommended_route_name": best_option["name"],
                    }
                    save_trip_to_history(st.session_state.route_result)

                except requests.Timeout:
                    st.error("The map service took too long to respond. Please wait a moment and try again.")
                except requests.RequestException as error:
                    st.error(f"Could not connect to the map service: {error}")
                except Exception as error:
                    st.error(f"Could not plan this journey: {error}")
                    st.caption("OpenStreetMap and OSRM are free public services; they can occasionally be busy or unavailable.")

    # Render stored route results outside the submit block so they persist after reruns.
    result = st.session_state.route_result
    if result:
        st.divider()
        st.subheader(f"{result['origin']} → {result['destination']}")
        st.success(f"Recommended: {result['recommended_route_name']} — lowest estimated energy use")

        st.subheader("Compare available routes")
        route_card_cols = st.columns(min(3, max(1, len(result["route_options"]))))
        for route_index, route_option in enumerate(result["route_options"]):
            with route_card_cols[route_index % len(route_card_cols)]:
                is_best = route_option["is_recommended"]
                st.markdown(
                    f"<div class='eco-option-card {'eco-option-best' if is_best else ''}'>"
                    f"<div class='eco-option-top'><b>{route_option['name']}</b>"
                    f"<span>{'✦ RECOMMENDED' if is_best else 'ALTERNATIVE'}</span></div>"
                    f"<div class='eco-option-energy'>{route_option['predicted_energy']:.1f}<small> kWh</small></div>"
                    f"<div class='eco-option-meta'>{route_option['distance_km']:.1f} km · {route_option['duration_min']:.0f} min</div>"
                    f"</div>", unsafe_allow_html=True,
                )
        comparison_rows = []
        for option in result["route_options"]:
            comparison_rows.append({
                "Route": option["name"] + (" ⭐ Recommended" if option["is_recommended"] else ""),
                "Distance (km)": round(option["distance_km"], 1),
                "Travel time (min)": round(option["duration_min"], 0),
                "Predicted energy (kWh)": round(option["predicted_energy"], 2),
            })
        st.dataframe(
            pd.DataFrame(comparison_rows),
            use_container_width=True,
            hide_index=True,
        )
        st.caption(
            "Route recommendation is based on the lowest predicted energy from the demonstration ML model. "
            "The model uses synthetic training data and fixed example driving conditions."
        )

        m1, m2, m3 = st.columns(3)
        m1.metric("Recommended route distance", f"{result['distance_km']:.1f} km")
        m2.metric("Estimated travel time", f"{result['duration_min']:.0f} min")
        m3.metric("Predicted energy", f"{result['predicted_energy']:.1f} kWh")
        m4, m5 = st.columns(2)
        m4.metric("Starting available energy", f"{result['available_energy']:.1f} kWh")
        m5.metric("Estimated battery after journey", f"{result['remaining_energy']:.1f} kWh")
        if result["remaining_energy"] < 0:
            st.warning("The estimated energy is greater than your available battery energy. Consider charging before travelling. This is only a demonstration estimate.")
        else:
            remaining_pct = result["remaining_energy"] / result["battery_capacity"] * 100
            st.info(f"Approximate battery remaining: {remaining_pct:.1f}% of full capacity. This is an experimental estimate, not a vehicle guarantee.")

        # Trip Cost Calculator
        st.divider()
        st.subheader("💰 Trip Cost Calculator")
        tariff_default = float(st.session_state.get("ev_electricity_tariff", 8.0))
        trip_tariff = st.number_input(
            "Electricity tariff (₹ per kWh)", min_value=0.0, max_value=100.0,
            value=tariff_default, step=0.5, key="route_trip_tariff"
        )
        trip_cost = result["predicted_energy"] * trip_tariff
        cost_per_km = trip_cost / result["distance_km"] if result["distance_km"] > 0 else 0.0
        tc1, tc2, tc3 = st.columns(3)
        tc1.metric("Estimated trip cost", f"₹{trip_cost:.2f}")
        tc2.metric("Cost per km", f"₹{cost_per_km:.2f}")
        tc3.metric("Energy used", f"{result['predicted_energy']:.2f} kWh")

        # Driving-style energy comparison using the existing demonstration model
        st.subheader("🚘 Energy Consumption Comparison")
        style_rows = []
        for style_name, style_value in [("Eco", 0.1), ("Normal", 0.5), ("Aggressive", 0.9)]:
            style_input = pd.DataFrame([{
                "speed": 45, "temperature": 28, "traffic": 0.4,
                "driving_style": style_value, "distance": result["distance_km"],
            }])
            style_energy = max(0.0, float(model.predict(style_input)[0]))
            style_rows.append({"Driving style": style_name, "Estimated energy (kWh)": round(style_energy, 2)})
        style_df = pd.DataFrame(style_rows)
        st.bar_chart(style_df.set_index("Driving style"))
        st.caption("Illustrative model comparison using fixed example speed, traffic and temperature. Synthetic-model estimates are not real-world guarantees.")

        route_map = folium.Map(location=result["start"], zoom_start=8, control_scale=True)
        folium.Marker(
            result["start"], tooltip="Start", popup=result["origin"],
            icon=folium.Icon(color="green", icon="play"),
        ).add_to(route_map)
        folium.Marker(
            result["end"], tooltip="Destination", popup=result["destination"],
            icon=folium.Icon(color="red", icon="flag"),
        ).add_to(route_map)
        folium.GeoJson(
            result["geometry"],
            name="Recommended energy-efficient route",
            style_function=lambda feature: {"color": "#10a875", "weight": 6, "opacity": 0.9},
        ).add_to(route_map)

        # Smart Charging Stop Planner: discover nearby OSM stations and estimate reachable stops.
        smart_stops = []
        station_records = []
        planner_message = "Charging station data has not been loaded yet."
        try:
            route_coords = result["geometry"]["coordinates"]
            lons = [point[0] for point in route_coords]
            lats = [point[1] for point in route_coords]
            south, north = min(lats) - 0.12, max(lats) + 0.12
            west, east = min(lons) - 0.12, max(lons) + 0.12
            station_records = get_charging_stations(south, west, north, east)
            route_stations = prepare_route_stations(route_coords, station_records, corridor_km=35.0)
            st.session_state["eco_route_stations"] = route_stations
            st.session_state["eco_station_lookup_message"] = f"Found {len(route_stations)} mapped station(s) near this route."
            smart_stops, planner_message = calculate_smart_charging_stops(
                route_stations=route_stations,
                distance_km=result["distance_km"],
                energy_kwh=result["predicted_energy"],
                battery_capacity_kwh=result["battery_capacity"],
                starting_soc_pct=result["battery_percent"],
                reserve_pct=result["reserve_percent"],
                target_soc_pct=result["target_charge_percent"],
                charger_power_kw=result["assumed_charger_power"],
            )

            for stop_number, stop in enumerate(smart_stops, start=1):
                stop_popup = f"""
                <div style="width:240px;font-family:Arial;">
                    <h4 style="color:#087653;">⚡ Recommended charging stop {stop_number}</h4>
                    <p><b>{stop['name']}</b></p>
                    <p>Approx. {stop['route_km']:.1f} km along route</p>
                    <p>Estimated arrival battery: {stop['arrival_soc_pct']:.1f}%</p>
                    <p>Estimated charge time: {stop['charging_minutes']:.0f} min</p>
                    <a href="{stop['google_maps_url']}" target="_blank">Get directions</a>
                </div>
                """
                folium.Marker(
                    location=[stop["lat"], stop["lon"]],
                    tooltip=f"⚡ Charging stop {stop_number}: {stop['name']}",
                    popup=folium.Popup(stop_popup, max_width=300),
                    icon=folium.Icon(color="purple", icon="flash", prefix="fa"),
                ).add_to(route_map)

            # Show other nearby charging stations with a separate marker style.
            recommended_coords = {(round(s["lat"], 5), round(s["lon"], 5)) for s in smart_stops}
            for station in route_stations:
                if (round(station["lat"], 5), round(station["lon"], 5)) in recommended_coords:
                    continue
                folium.CircleMarker(
                    location=[station["lat"], station["lon"]],
                    radius=4, color="#64748b", fill=True, fill_opacity=0.75,
                    tooltip=f"Nearby charger: {station['name']} (not selected)",
                ).add_to(route_map)
        except Exception as error:
            route_stations = []
            planner_message = f"Charging station lookup is temporarily unavailable: {error}"

        st.divider()
        st.subheader("⚡ Smart Charging Stop Planner")
        st.caption(
            "Finds OpenStreetMap charging stations near your route and checks whether each suggested stop is reachable "
            "while keeping your selected battery reserve."
        )
        if smart_stops:
            st.success(f"Recommended {len(smart_stops)} charging stop(s) for this journey.")
            charging_rows = []
            for number, stop in enumerate(smart_stops, start=1):
                charging_rows.append({
                    "Stop": number,
                    "Station": stop["name"],
                    "Distance along route (km)": round(stop["route_km"], 1),
                    "Detour from route (km)": round(stop["detour_km"], 1),
                    "Battery on arrival (%)": round(stop["arrival_soc_pct"], 1),
                    "Energy to add (kWh)": round(stop["charge_added_kwh"], 1),
                    "Estimated charging time (min)": round(stop["charging_minutes"]),
                })
            st.dataframe(pd.DataFrame(charging_rows), use_container_width=True, hide_index=True)
            for number, stop in enumerate(smart_stops, start=1):
                st.markdown(
                    f"**Stop {number}: {stop['name']}** · "
                    f"[Get directions]({stop['google_maps_url']}) · "
                    f"Estimated charge time: **{stop['charging_minutes']:.0f} minutes**"
                )
        else:
            st.info(planner_message)
        st.caption(
            f"Planner settings: {result['reserve_percent']}% minimum reserve, "
            f"{result['target_charge_percent']}% target charge, "
            f"{result['assumed_charger_power']:.0f} kW assumed charger power. "
            "Charging time is an estimate; actual charging speed, station operation and live availability are not verified. "
            "Battery consumption is based on this project's demonstration model."
        )

        folium.LayerControl().add_to(route_map)
        st.subheader("Your recommended route map")
        st.components.v1.html(route_map._repr_html_(), height=460, scrolling=False)

        report = pd.DataFrame([{
            "Origin": result["origin"],
            "Destination": result["destination"],
            "Recommended_route": result["recommended_route_name"],
            "Distance_km": round(result["distance_km"], 2),
            "Travel_time_min": round(result["duration_min"], 1),
            "Predicted_energy_kWh": round(result["predicted_energy"], 2),
            "Starting_available_energy_kWh": round(result["available_energy"], 2),
            "Battery_remaining_kWh": round(result["remaining_energy"], 2),
        }])
        st.download_button(
            "Download journey report (CSV)",
            report.to_csv(index=False),
            file_name="ecoroute_journey.csv",
            mime="text/csv",
        )

# ---------------------------
# Energy prediction lab
# ---------------------------
with energy_tab:
    st.header("Energy prediction lab")
    st.caption(
    "Explore how distance, speed, temperature, traffic, and driving style "
    "affect estimated EV energy consumption."
)
    with st.container(border=True):
        col1, col2 = st.columns(2)
        with col1:
            distance = st.slider("Trip distance (km)", 1, 300, 50)
            speed = st.slider("Average speed (km/h)", 10, 120, 50)
            temperature = st.slider("Outside temperature (°C)", 0, 45, 28)
        with col2:
            traffic_label = st.selectbox("Traffic conditions", ["Low", "Medium", "High"])
            style_label = st.selectbox("Driving style", ["Eco", "Normal", "Aggressive"])
            st.write(" ")
            predict_clicked = st.button("Predict energy consumption", type="primary", use_container_width=True)

    if predict_clicked:
        traffic_value = {"Low": 0.1, "Medium": 0.5, "High": 0.9}[traffic_label]
        style_value = {"Eco": 0.1, "Normal": 0.5, "Aggressive": 0.9}[style_label]
        features = pd.DataFrame([{
            "speed": speed, "temperature": temperature, "traffic": traffic_value,
            "driving_style": style_value, "distance": distance,
        }])
        st.session_state.energy_result = {
            "predicted": max(0.0, float(model.predict(features)[0])),
            "style": style_label, "mae": mae,
            "distance": distance, "speed": speed, "temperature": temperature,
            "traffic": traffic_label,
        }

    energy_result = st.session_state.energy_result
    if energy_result:
        st.divider()
        st.metric("Estimated energy needed", f"{energy_result['predicted']:.2f} kWh")
        st.caption(f"Demonstration model test MAE: {energy_result['mae']:.2f} kWh. Trained on synthetic data; this is not a real-world accuracy guarantee.")
        st.subheader("Eco-driving feedback")
        if energy_result["style"] == "Eco":
            st.success("Good choice! Maintain a steady speed and accelerate gently.")
        elif energy_result["style"] == "Normal":
            st.info("Try smoother acceleration, steady cruising, and regenerative braking where available.")
        else:
            st.warning("Aggressive driving can increase energy use. Reduce rapid acceleration and braking.")
        tips = pd.DataFrame({"Tip": [
            "Maintain steady speeds", "Accelerate smoothly", "Use regenerative braking",
            "Check tyre pressure", "Precondition the cabin while plugged in when practical",
        ]})
        st.dataframe(tips, use_container_width=True, hide_index=True)

# ---------------------------
# Trip history dashboard
# ---------------------------
with history_tab:
    st.header("Trip History Dashboard")
    st.caption("Your successfully planned journeys are saved locally on this computer.")
    history_df = load_trip_history()

    if history_df.empty:
        st.info("No trips saved yet. Plan a journey in the Route Planner tab to build your history.")
    else:
        total_trips = len(history_df)
        total_distance = float(history_df["distance_km"].sum())
        total_energy = float(history_df["energy_kwh"].sum())
        h1, h2, h3 = st.columns(3)
        h1.metric("Total trips", f"{total_trips}")
        h2.metric("Total distance", f"{total_distance:.1f} km")
        h3.metric("Total predicted energy", f"{total_energy:.1f} kWh")

        display_history = history_df.rename(columns={
            "trip_time": "Date and time",
            "origin": "Starting point",
            "destination": "Destination",
            "distance_km": "Distance (km)",
            "duration_min": "Travel time (min)",
            "energy_kwh": "Energy (kWh)",
            "route_name": "Recommended route",
        })
        display_history = display_history.drop(columns=["id"])
        st.dataframe(display_history, use_container_width=True, hide_index=True)
        st.download_button(
            "Download trip history (CSV)",
            history_df.drop(columns=["id"]).to_csv(index=False),
            file_name="ecoroute_trip_history.csv",
            mime="text/csv",
        )
        if st.button("Clear trip history", type="secondary"):
            with sqlite3.connect(HISTORY_DB) as connection:
                connection.execute("DELETE FROM trip_history")
            st.success("Trip history cleared.")
            st.rerun()


# ---------------------------
# Weather-based energy prediction
# ---------------------------
with weather_tab:
    st.header("🌦️ Weather-Based Energy Prediction")
    st.caption("Uses free Open-Meteo current weather data for the starting point of your latest planned route.")
    latest_route = st.session_state.route_result
    if not latest_route:
        st.info("Plan a route first. Weather-based estimates will use its starting coordinates.")
    else:
        try:
            weather = get_weather_for_coordinates(*latest_route["start"])
            if not weather:
                st.warning("Weather data was not returned. Please try again later.")
            else:
                temp_c = float(weather.get("temperature_2m", 22))
                rain_mm = float(weather.get("precipitation", 0))
                wind_kmh = float(weather.get("wind_speed_10m", 0))
                humidity = weather.get("relative_humidity_2m", "—")
                w1, w2, w3, w4 = st.columns(4)
                w1.metric("Temperature", f"{temp_c:.1f} °C")
                w2.metric("Precipitation", f"{rain_mm:.1f} mm")
                w3.metric("Wind speed", f"{wind_kmh:.1f} km/h")
                w4.metric("Humidity", f"{humidity}%" if humidity != "—" else "—")
                # Transparent illustrative adjustment, not a validated vehicle model.
                weather_factor = 1.0 + max(0, abs(temp_c - 22) - 5) * 0.003 + min(rain_mm, 10) * 0.004 + min(wind_kmh, 60) * 0.0008
                adjusted_energy = latest_route["predicted_energy"] * weather_factor
                st.metric("Weather-adjusted illustrative energy", f"{adjusted_energy:.2f} kWh", delta=f"{adjusted_energy - latest_route['predicted_energy']:+.2f} kWh")
                st.caption("Weather adjustment is a simple educational heuristic. It is not calibrated to a specific EV and should not be used as a safety-critical range estimate.")
        except requests.RequestException as error:
            st.warning(f"Could not retrieve weather right now: {error}")

# ---------------------------
# Advanced analytics dashboard
# ---------------------------
with analytics_tab:
    st.header("📈 Advanced Analytics Dashboard")
    st.caption("Analytics are calculated from journeys saved on this computer.")
    analytics_df = load_trip_history()
    if analytics_df.empty:
        st.info("No saved journeys yet. Plan trips in Route Planner to populate your analytics.")
    else:
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Total trips", f"{len(analytics_df)}")
        a2.metric("Total distance", f"{analytics_df['distance_km'].sum():.1f} km")
        a3.metric("Total energy", f"{analytics_df['energy_kwh'].sum():.1f} kWh")
        tariff = float(st.session_state.get("ev_electricity_tariff", 8.0))
        a4.metric("Estimated energy cost", f"₹{analytics_df['energy_kwh'].sum() * tariff:,.2f}")
        chart_df = analytics_df.copy()
        chart_df["trip_label"] = chart_df["origin"].astype(str) + " → " + chart_df["destination"].astype(str)
        st.subheader("Energy consumed by trip")
        st.bar_chart(chart_df.set_index("trip_label")[["energy_kwh"]])
        st.subheader("Distance travelled over saved trips")
        st.line_chart(chart_df.reset_index(drop=True)[["distance_km"]])
        st.caption("Costs are estimated using your saved electricity tariff. History is stored locally in SQLite.")

# ---------------------------
# Charging station availability checker
# ---------------------------
with charging_tab:
    st.header("🔌 Charging Station Availability Checker")
    st.caption("Explore OpenStreetMap charging locations along your planned route or around a city.")
    station_list = st.session_state.get("eco_route_stations", [])
    charging_route = st.session_state.route_result

    if charging_route:
        st.markdown(f"**Route:** {charging_route['origin']} → {charging_route['destination']}")
        if st.button("↻ Refresh charging station data", key="refresh_route_chargers"):
            try:
                route_coords = charging_route["geometry"]["coordinates"]
                route_lons = [point[0] for point in route_coords]
                route_lats = [point[1] for point in route_coords]
                records = get_charging_stations(
                    min(route_lats) - 0.12, min(route_lons) - 0.12,
                    max(route_lats) + 0.12, max(route_lons) + 0.12,
                )
                station_list = prepare_route_stations(route_coords, records, corridor_km=35.0)
                st.session_state["eco_route_stations"] = station_list
                st.session_state["eco_station_lookup_message"] = f"Found {len(station_list)} mapped station(s) near this route."
                st.rerun()
            except Exception as error:
                st.warning(f"Charging station lookup is temporarily unavailable: {error}")

        hub_map = folium.Map(location=charging_route["start"], zoom_start=7, control_scale=True, tiles="CartoDB positron")
        folium.PolyLine(
            locations=[[lat, lon] for lon, lat in charging_route["geometry"]["coordinates"]],
            color="#159b68", weight=5, opacity=0.85, tooltip="Planned EV route",
        ).add_to(hub_map)
        folium.Marker(charging_route["start"], tooltip="Journey start", popup=charging_route["origin"],
                      icon=folium.Icon(color="green", icon="play")).add_to(hub_map)
        folium.Marker(charging_route["end"], tooltip="Destination", popup=charging_route["destination"],
                      icon=folium.Icon(color="red", icon="flag")).add_to(hub_map)

        if station_list:
            for station in station_list:
                popup_html = f"""<div style='min-width:210px;font-family:Arial,sans-serif'>
                <b style='color:#087653'>⚡ {html.escape(str(station['name']))}</b><br>
                <span>Operator: {html.escape(str(station['operator']))}</span><br>
                <span>Route position: {station['route_km']:.1f} km</span><br>
                <span>Detour: {station['detour_km']:.1f} km</span><br>
                <span>Connector/output: {html.escape(str(station['socket_output']))}</span><br>
                <a href='{station['google_maps_url']}' target='_blank'>Open directions</a></div>"""
                folium.Marker([station["lat"], station["lon"]], tooltip=f"⚡ {station['name']}",
                              popup=folium.Popup(popup_html, max_width=320),
                              icon=folium.Icon(color="purple", icon="flash", prefix="fa")).add_to(hub_map)
            lats = [charging_route["start"][0], charging_route["end"][0]] + [x["lat"] for x in station_list]
            lons = [charging_route["start"][1], charging_route["end"][1]] + [x["lon"] for x in station_list]
            hub_map.fit_bounds([[min(lats), min(lons)], [max(lats), max(lons)]], padding=(24, 24))
            st.success(f"{len(station_list)} mapped charging hub(s) found within 35 km of the route corridor.")
        else:
            hub_map.fit_bounds([charging_route["start"], charging_route["end"]], padding=(24, 24))
            st.warning("No mapped charging hubs were returned for this route. Try refreshing or search by city below.")
        st.components.v1.html(hub_map.get_root().render(), height=540, scrolling=False)

        if station_list:
            st.subheader("Charging hubs along your journey")
            station_df = pd.DataFrame([{
                "Station": item["name"], "Operator": item["operator"],
                "Approx. route position (km)": round(item["route_km"], 1),
                "Detour (km)": round(item["detour_km"], 1),
                "Connector / output info": item["socket_output"],
                "Directions": item["google_maps_url"],
            } for item in station_list])
            st.dataframe(station_df, use_container_width=True, hide_index=True,
                         column_config={"Directions": st.column_config.LinkColumn("Directions")})

    else:
        st.info("No route is planned yet. Search a city or area to view nearby mapped charging hubs.")
        with st.form("charging_area_search_form"):
            search_area = st.text_input("City / area", value=st.session_state.get("eco_charging_search_area", "Chennai, India"),
                                        placeholder="Example: Chennai, India")
            search_area_submit = st.form_submit_button("Find charging stations ⚡", type="primary")
        area_result = st.session_state.get("eco_area_charging_result")
        if search_area_submit:
            if not search_area.strip():
                st.warning("Enter a city or area to search.")
                area_result = None
            else:
                try:
                    with st.spinner("Finding the area and loading mapped charging stations..."):
                        geo_response = requests.get(
                            "https://nominatim.openstreetmap.org/search",
                            params={"q": search_area.strip(), "format": "json", "limit": 1},
                            headers={"User-Agent": "EcoRouteEV/1.1 (educational project)"}, timeout=20,
                        )
                        geo_response.raise_for_status()
                        geo_items = geo_response.json()
                        if not geo_items:
                            st.warning("That area could not be found. Try a nearby city or a more specific place name.")
                            area_result = None
                        else:
                            center_lat = float(geo_items[0]["lat"])
                            center_lon = float(geo_items[0]["lon"])
                            radius_deg = 0.18
                            raw_stations = get_charging_stations(
                                center_lat-radius_deg, center_lon-radius_deg,
                                center_lat+radius_deg, center_lon+radius_deg,
                            )
                            mapped = []
                            seen = set()
                            for record in raw_stations:
                                tags = record.get("tags", {})
                                lat = record.get("lat", record.get("center", {}).get("lat"))
                                lon = record.get("lon", record.get("center", {}).get("lon"))
                                if lat is None or lon is None:
                                    continue
                                lat, lon = float(lat), float(lon)
                                key = (round(lat, 5), round(lon, 5))
                                if key in seen:
                                    continue
                                seen.add(key)
                                mapped.append({
                                    "name": tags.get("name") or tags.get("operator") or "EV Charging Station",
                                    "operator": tags.get("operator", "Not listed"),
                                    "socket_output": tags.get("socket:output", "Not listed"),
                                    "lat": lat, "lon": lon,
                                    "google_maps_url": f"https://www.google.com/maps/dir/?api=1&destination={lat},{lon}&travelmode=driving",
                                })
                            area_result = {"area": search_area.strip(), "center": [center_lat, center_lon], "stations": mapped}
                            st.session_state["eco_area_charging_result"] = area_result
                            st.session_state["eco_charging_search_area"] = search_area.strip()
                except Exception as error:
                    st.error(f"Could not load charging stations right now: {error}")
                    area_result = None
        if area_result:
            st.success(f"Found {len(area_result['stations'])} mapped charging station(s) around {area_result['area']}.")
            hub_map = folium.Map(location=area_result["center"], zoom_start=11, control_scale=True, tiles="CartoDB positron")
            folium.Marker(area_result["center"], tooltip=f"Search area: {area_result['area']}",
                          icon=folium.Icon(color="green", icon="search")).add_to(hub_map)
            for station in area_result["stations"]:
                popup_html = f"<b>{html.escape(str(station['name']))}</b><br>Operator: {html.escape(str(station['operator']))}<br><a href='{station['google_maps_url']}' target='_blank'>Directions</a>"
                folium.Marker([station["lat"], station["lon"]], tooltip=f"⚡ {station['name']}",
                              popup=folium.Popup(popup_html, max_width=280),
                              icon=folium.Icon(color="purple", icon="flash", prefix="fa")).add_to(hub_map)
            if area_result["stations"]:
                hub_map.fit_bounds([[x["lat"], x["lon"]] for x in area_result["stations"]] + [area_result["center"]], padding=(20, 20))
            st.components.v1.html(hub_map.get_root().render(), height=540, scrolling=False)
            if area_result["stations"]:
                area_df = pd.DataFrame([{
                    "Station": x["name"], "Operator": x["operator"],
                    "Connector / output info": x["socket_output"], "Directions": x["google_maps_url"],
                } for x in area_result["stations"]])
                st.dataframe(area_df, use_container_width=True, hide_index=True,
                             column_config={"Directions": st.column_config.LinkColumn("Directions")})
            else:
                st.warning("OpenStreetMap returned no mapped charging hubs for this area. The map is displayed, but station coverage may be incomplete.")
        else:
            # Always show a working base map instead of a blank section before a search.
            base_map = folium.Map(location=[13.0827, 80.2707], zoom_start=10, control_scale=True, tiles="CartoDB positron")
            folium.Marker([13.0827, 80.2707], tooltip="Example search area — Chennai", icon=folium.Icon(color="green")).add_to(base_map)
            st.components.v1.html(base_map.get_root().render(), height=500, scrolling=False)

    st.info("OpenStreetMap shows mapped charger locations only. It does not confirm live availability, working status, pricing or connector compatibility. Verify details with the station operator before travelling.")

# ---------------------------
# Energy consumption comparison
# ---------------------------
with compare_tab:
    st.header("🔋 Energy Consumption Comparison")
    st.caption("Compare energy estimates across driving styles and route alternatives.")
    compare_route = st.session_state.route_result
    if not compare_route:
        st.info("Plan a route first to compare its route alternatives and driving styles.")
    else:
        alt_df = pd.DataFrame([{
            "Route": item["name"], "Distance (km)": item["distance_km"],
            "Energy (kWh)": item["predicted_energy"],
        } for item in compare_route["route_options"]])
        st.subheader("Route alternative comparison")
        st.bar_chart(alt_df.set_index("Route")[["Energy (kWh)"]])
        style_values = []
        for style_name, style_value in [("Eco", 0.1), ("Normal", 0.5), ("Aggressive", 0.9)]:
            features = pd.DataFrame([{
                "speed": 45, "temperature": 28, "traffic": 0.4,
                "driving_style": style_value, "distance": compare_route["distance_km"],
            }])
            estimate = max(0.0, float(model.predict(features)[0]))
            style_values.append({"Driving style": style_name, "Energy (kWh)": estimate})
        style_compare_df = pd.DataFrame(style_values)
        st.subheader("Driving style comparison")
        st.bar_chart(style_compare_df.set_index("Driving style"))
        st.dataframe(style_compare_df.round(2), use_container_width=True, hide_index=True)
        st.caption("All values are estimates from the synthetic-data demonstration model.")

# ---------------------------
# Carbon emission savings calculator
# ---------------------------
with carbon_tab:
    st.header("🌍 Carbon Emission Savings Calculator")
    st.caption("Illustrative comparison of EV electricity emissions and petrol vehicle tailpipe emissions.")
    carbon_distance = st.number_input("Distance travelled (km)", min_value=1.0, max_value=10000.0, value=100.0, step=10.0, key="carbon_distance")
    ev_efficiency = st.number_input("EV energy consumption (kWh per 100 km)", min_value=1.0, max_value=100.0, value=16.0, step=0.5, key="carbon_efficiency")
    grid_factor = st.number_input("Electricity carbon factor (kg CO₂ per kWh)", min_value=0.0, max_value=2.0, value=0.70, step=0.05, key="grid_carbon_factor", help="Editable assumption. Use a current local grid factor if you have a trusted source.")
    petrol_factor = st.number_input("Petrol car tailpipe factor (kg CO₂ per km)", min_value=0.0, max_value=1.0, value=0.192, step=0.005, key="petrol_carbon_factor")
    ev_emissions = (carbon_distance * ev_efficiency / 100.0) * grid_factor
    petrol_emissions = carbon_distance * petrol_factor
    avoided_emissions = petrol_emissions - ev_emissions
    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated EV electricity emissions", f"{ev_emissions:.2f} kg CO₂")
    c2.metric("Petrol tailpipe comparison", f"{petrol_emissions:.2f} kg CO₂")
    c3.metric("Estimated difference", f"{avoided_emissions:.2f} kg CO₂", delta="Lower than petrol" if avoided_emissions >= 0 else "Higher under these assumptions")
    st.caption("This is a simplified estimate, not a full lifecycle assessment. It excludes vehicle and battery manufacturing and depends strongly on the electricity mix and chosen factors.")

# ---------------------------
# Personalized EV dashboard
# ---------------------------
with profile_tab:
    st.header("🚗 My EV Dashboard")
    st.caption("Set your vehicle preferences. These settings are kept in this browser session.")
    with st.form("personal_ev_profile"):
        profile_name = st.text_input("Vehicle name", value=st.session_state.get("ev_vehicle_name", "My EV"))
        profile_capacity = st.number_input("Battery capacity (kWh)", min_value=10.0, max_value=250.0, value=float(st.session_state.get("ev_battery_capacity", 60.0)), step=1.0)
        profile_tariff = st.number_input("Electricity tariff (₹/kWh)", min_value=0.0, max_value=100.0, value=float(st.session_state.get("ev_electricity_tariff", 8.0)), step=0.5)
        profile_reserve = st.slider("Preferred minimum battery reserve (%)", 5, 40, int(st.session_state.get("ev_reserve_percent", 15)))
        profile_submitted = st.form_submit_button("Save my EV settings", type="primary", use_container_width=True)
    if profile_submitted:
        st.session_state.ev_vehicle_name = profile_name.strip() or "My EV"
        st.session_state.ev_battery_capacity = profile_capacity
        st.session_state.ev_electricity_tariff = profile_tariff
        st.session_state.ev_reserve_percent = profile_reserve
        st.success("Your EV preferences have been saved for this session.")
    p1, p2, p3 = st.columns(3)
    p1.metric("Vehicle", st.session_state.get("ev_vehicle_name", "My EV"))
    p2.metric("Battery capacity", f"{st.session_state.get('ev_battery_capacity', 60.0):.1f} kWh")
    p3.metric("Preferred reserve", f"{st.session_state.get('ev_reserve_percent', 15)}%")
    st.info("These preferences currently last for the active Streamlit session. Persistent profiles across app restarts would require adding local database storage.")

# ---------------------------
# About
# ---------------------------
with about_tab:
    st.header("About EcoRoute EV")
    st.write("""
    EcoRoute EV is a student demonstration project combining route planning,
    machine-learning-based energy estimates, and eco-driving feedback.

    **Technologies:** Python, Streamlit, Pandas, NumPy, Scikit-learn,
    OpenStreetMap, OSRM, Folium, and SQLite.

    **Included features:** Multiple route comparison, experimental ML energy estimates,
    weather-based estimates, mapped charging-station search, smart charging stops,
    trip-cost and carbon calculators, EV profile settings, analytics, and a local EV assistant.

    **Important limitations:** The model uses synthetic demonstration data, not
    verified vehicle telemetry. Energy estimates are experimental and should not
    be used as the sole basis for real driving or charging decisions. Route and
    charging-station lookup require internet access. Trip history is stored locally
    in an SQLite database on the computer running this app.
    """)
    st.info("This is an educational prototype. Please verify routes and charging needs with reliable real-world sources before travelling.")


# ---------------------------
# Floating local EV assistant (available from every tab)
# ---------------------------
st.markdown("""
<style>
.st-key-floating_ev_chat{position:fixed!important;right:24px;bottom:22px;z-index:999999!important;width:auto!important}
.st-key-floating_ev_chat [data-testid="stPopover"]>button{border-radius:999px!important;background:linear-gradient(135deg,#087653,#064e3b)!important;color:#fff!important;border:1px solid #b7e4cb!important;box-shadow:0 8px 24px rgba(6,78,59,.28)!important;padding:12px 18px!important;font-weight:800!important}
.st-key-floating_ev_chat [data-testid="stPopover"]>button:hover{transform:translateY(-2px);box-shadow:0 12px 28px rgba(6,78,59,.34)!important}
</style>
""", unsafe_allow_html=True)
with st.container(key="floating_ev_chat"):
    with st.popover("💬 EV Assistant", help="Open your free EcoRoute EV assistant"):
        st.markdown("### 🤖 EcoRoute EV Assistant")
        st.caption("Free local EV help — no API key required.")
        if "ev_chat_history" not in st.session_state:
            st.session_state.ev_chat_history = []
        chat_panel = st.container(height=300)
        with chat_panel:
            for chat_message in st.session_state.ev_chat_history:
                with st.chat_message(chat_message["role"]):
                    st.write(chat_message["content"])
        with st.form("floating_local_ev_chat_form", clear_on_submit=True):
            question = st.text_input("Ask your EV question", placeholder="How can I improve EV range?")
            send_question = st.form_submit_button("Send ⚡", use_container_width=True)
        if send_question:
            if question.strip():
                reply = local_ev_answer(question)
                st.session_state.ev_chat_history.append({"role": "user", "content": question.strip()})
                st.session_state.ev_chat_history.append({"role": "assistant", "content": reply})
                st.rerun()
            else:
                st.warning("Please type a question first.")
        if st.session_state.ev_chat_history and st.button("Clear conversation", key="clear_floating_ev_chat"):
            st.session_state.ev_chat_history = []
            st.rerun()
        st.caption("Keyword-based offline assistant; it may not understand every phrasing.")
