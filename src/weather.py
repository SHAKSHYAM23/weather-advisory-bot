import requests
from typing import Optional, Dict, Any

def get_weather_for_location(location_name: str) -> Optional[Dict[str, Any]]:
    """
    Geocodes a city name and fetches current weather from Open-Meteo.
    Returns None if geocoding or weather fetching fails (Honest Failure).
    """
   
    geo_url = f"https://geocoding-api.open-meteo.com/v1/search?name={location_name}&count=1"
    try:
        geo_resp = requests.get(geo_url, timeout=5).json()
        if not geo_resp.get("results"):
            return None
        lat = geo_resp["results"][0]["latitude"]
        lon = geo_resp["results"][0]["longitude"]
    except Exception:
        return None  


    weather_url = (
        f"https://api.open-meteo.com/v1/forecast"
        f"?latitude={lat}&longitude={lon}"
        f"&current=temperature_2m,relative_humidity_2m,precipitation,wind_speed_10m,uv_index,weather_code"
    )
    
    try:
        weather_resp = requests.get(weather_url, timeout=5).json()
        if "current" not in weather_resp:
            return None
            
        current = weather_resp["current"]
        
       
        wmo_code = current.get("weather_code", 0)
        thunderstorm_active = True if wmo_code in [95, 96, 99] else False
        
       
        regional_rain_system = True if current.get("precipitation", 0) > 10.0 else False


        return {
            "temperature": current.get("temperature_2m"),
            "humidity": current.get("relative_humidity_2m"),
            "precipitation": current.get("precipitation"),
            "wind_speed": current.get("wind_speed_10m"),
            "uv_index": current.get("uv_index"),
            "thunderstorm_active": thunderstorm_active,
            "regional_rain_system": regional_rain_system
        }
    except Exception:

        return None