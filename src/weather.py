import requests
from typing import Optional, Dict, Any

GEO_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
VARS = "temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,uv_index,weather_code"
# Local-hour windows for hourly requests (end exclusive).
PERIODS = {
    "morning": range(6, 12), "afternoon": range(12, 17),
    "evening": range(17, 21), "night": range(21, 24), "day": range(6, 22),
}
THUNDER_CODES = {95, 96, 99}


def _vals(xs):
    return [x for x in xs if x is not None]


def _max(xs):
    v = _vals(xs)
    return max(v) if v else None


def _mean(xs):
    v = _vals(xs)
    return round(sum(v) / len(v), 1) if v else None


def _build(temp, hum, precip, wind, uv, codes, location, when) -> Dict[str, Any]:
    return {
        "temperature": temp,
        "humidity": hum,
        "precipitation": precip,
        "wind_speed": wind,
        "uv_index": uv,
        "thunderstorm_active": any(c in THUNDER_CODES for c in _vals(codes)),
        "regional_rain_system": (precip or 0) > 10.0,  # proxy: >10 mm precipitation
        "location": location,
        "when": when,
    }


def get_weather_for_location(location_name: str, day: str = "today",
                             period: str = "now") -> Optional[Dict[str, Any]]:
    """
    Geocode a city, then fetch Open-Meteo weather.
      period="now"  -> current conditions (original behaviour)
      period in morning/afternoon/evening/night/day -> hourly forecast for `day` (today|tomorrow).
        temperature/humidity = window mean; precipitation/wind/UV = window max;
        thunderstorm = any hour in the window.
    Returns None on ANY failure (honest failure) - never guessed values.
    """
    if day not in ("today", "tomorrow"):
        return None
    try:
        geo = requests.get(GEO_URL, params={"name": location_name, "count": 1}, timeout=5).json()
        place = (geo.get("results") or [None])[0]
        if not place:
            return None
        lat, lon = place["latitude"], place["longitude"]
        location = ", ".join(p for p in (place.get("name"), place.get("admin1"), place.get("country")) if p)
    except Exception:
        return None

    if period == "now" and day == "tomorrow":
        period = "day"
    base = {"latitude": lat, "longitude": lon, "timezone": "auto"}
    try:
        if period == "now":
            data = requests.get(FORECAST_URL, params={**base, "current": VARS}, timeout=5).json()
            c = data.get("current")
            if not c:
                return None
            return _build(c.get("temperature_2m"), c.get("relative_humidity_2m"), c.get("precipitation"),
                          c.get("wind_speed_10m"), c.get("uv_index"), [c.get("weather_code")],
                          location, "current conditions")

        if period not in PERIODS:
            return None
        data = requests.get(FORECAST_URL, params={**base, "hourly": VARS, "forecast_days": 2}, timeout=5).json()
        h = data.get("hourly")
        if not h or not h.get("time"):
            return None
        times = h["time"]
        dates = sorted({t[:10] for t in times})  # local dates (timezone=auto)
        target = dates[0] if day == "today" else (dates[1] if len(dates) > 1 else None)
        if target is None:
            return None
        idx = [i for i, t in enumerate(times) if t[:10] == target and int(t[11:13]) in PERIODS[period]]
        if not idx:
            return None

        def pick(key):
            return [h[key][i] for i in idx]

        return _build(_mean(pick("temperature_2m")), _mean(pick("relative_humidity_2m")),
                      _max(pick("precipitation")), _max(pick("wind_speed_10m")), _max(pick("uv_index")),
                      pick("weather_code"), location, f"{day} {period} ({target})")
    except Exception:
        return None