"""
Weather Service – OpenWeatherMap API
Fetches real weather and generates agricultural risk alerts per your paper:
  • Humidity > 80% AND Temperature > 25°C → 'High Fungal Risk' alert
  • Additional storm, frost, heat alerts
Falls back to mock data when API key is absent.
"""

import os, json, requests
from datetime import datetime, timedelta


class WeatherService:
    def __init__(self):
        self.api_key  = os.getenv("OPENWEATHER_API_KEY", "")
        self.base_url = "https://api.openweathermap.org/data/2.5"

    def get_weather_forecast(self, location: str) -> str:
        if not self.api_key:
            return self._mock_weather(location)
        try:
            r = requests.get(
                f"{self.base_url}/weather",
                params={"q": location, "appid": self.api_key, "units": "metric"},
                timeout=10,
            )
            if r.status_code == 200:
                return self._format(r.json())
            return self._mock_weather(location)
        except Exception:
            return self._mock_weather(location)

    def get_5_day_forecast(self, location: str) -> str:
        if not self.api_key:
            return self._mock_forecast(location)
        try:
            r = requests.get(
                f"{self.base_url}/forecast",
                params={"q": location, "appid": self.api_key, "units": "metric"},
                timeout=10,
            )
            if r.status_code == 200:
                data = r.json()
                forecasts = []
                for item in data["list"][:5]:
                    forecasts.append({
                        "datetime":   item["dt_txt"],
                        "temperature": round(item["main"]["temp"]),
                        "humidity":   item["main"]["humidity"],
                        "weather":    item["weather"][0]["description"].title(),
                        "wind_speed": round(item["wind"]["speed"] * 3.6, 1),
                        "rainfall":   item.get("rain", {}).get("3h", 0),
                    })
                return json.dumps({"location": data["city"]["name"], "forecasts": forecasts})
            return self._mock_forecast(location)
        except Exception:
            return self._mock_forecast(location)

    def _format(self, d: dict) -> str:
        temp   = round(d["main"]["temp"])
        humid  = d["main"]["humidity"]
        wind   = round(d["wind"]["speed"] * 3.6, 1)
        rain   = d.get("rain", {}).get("1h", 0)
        cond   = d["weather"][0]["main"]
        desc   = d["weather"][0]["description"].title()

        alerts = self._build_alerts(temp, humid, rain, wind, cond)

        info = {
            "location":            d["name"],
            "country":             d["sys"]["country"],
            "temperature":         temp,
            "feels_like":          round(d["main"]["feels_like"]),
            "humidity":            humid,
            "pressure":            d["main"]["pressure"],
            "wind_speed":          wind,
            "wind_direction":      d["wind"].get("deg", 0),
            "weather_main":        cond,
            "weather_description": desc,
            "visibility":          round(d.get("visibility", 10000) / 1000, 1),
            "rainfall":            rain,
            "timestamp":           datetime.now().isoformat(),
            "farming_alerts":      alerts,
            "farming_advice":      self._advice(temp, humid, cond),
        }
        return json.dumps(info)

    def _build_alerts(self, temp, humid, rain, wind, cond) -> list:
        """
        Core alert logic from the paper:
        humidity > 80% AND temp > 25°C → High Fungal Risk
        """
        alerts = []

        # ── Paper's primary alert ──────────────────────────────────────────
        if humid > 80 and temp > 25:
            alerts.append({
                "level":   "danger",
                "icon":    "🍄",
                "en":      (f"HIGH FUNGAL RISK – Humidity {humid}% + Temperature {temp}°C. "
                            "Conditions are ideal for late blight, powdery mildew and leaf spot. "
                            "Apply preventive fungicide TODAY."),
                "ta":      (f"அதிக பூஞ்சை அபாயம் – ஈரப்பதம் {humid}%, வெப்பநிலை {temp}°C. "
                            "Late blight, powdery mildew பரவும் சாத்தியம் அதிகம். "
                            "இன்றே தடுப்பு பூஞ்சைக்கொல்லி தெளிக்கவும்."),
            })

        # ── Supplementary alerts ──────────────────────────────────────────
        if temp > 40:
            alerts.append({
                "level": "warning", "icon": "🔥",
                "en": f"Extreme heat {temp}°C – increase irrigation, provide shade for sensitive crops.",
                "ta": f"அதிக வெப்பம் {temp}°C – நீர்ப்பாசனம் அதிகரிக்கவும்.",
            })
        elif temp < 10:
            alerts.append({
                "level": "warning", "icon": "🥶",
                "en": f"Cold weather {temp}°C – protect crops from frost damage.",
                "ta": f"குளிர் வானிலை {temp}°C – பயிர்களை பனியிலிருந்து பாதுகாக்கவும்.",
            })
        if rain > 15:
            alerts.append({
                "level": "warning", "icon": "🌧️",
                "en": "Heavy rain – check drainage, postpone spraying.",
                "ta": "கனமழை – வடிகால் சரிபார்க்கவும், மருந்து தெளிப்பை தாமதிக்கவும்.",
            })
        if wind > 30:
            alerts.append({
                "level": "warning", "icon": "💨",
                "en": f"High wind {wind} km/h – secure structures, avoid spraying.",
                "ta": f"அதிக காற்று {wind} km/h – கட்டமைப்புகள் பாதுகாக்கவும்.",
            })
        if humid < 30:
            alerts.append({
                "level": "info", "icon": "🏜️",
                "en": "Low humidity – increase irrigation frequency, mulch soil.",
                "ta": "குறைந்த ஈரப்பதம் – நீர்ப்பாசனம் அதிகரிக்கவும்.",
            })

        return alerts

    def _advice(self, temp, humid, cond) -> list:
        advice = []
        if temp > 35:
            advice.append("High temperature: irrigate early morning or evening.")
        if humid > 80:
            advice.append("High humidity: monitor for fungal diseases daily.")
        if "rain" in cond.lower():
            advice.append("Rainy: check drainage, hold pesticide application.")
        elif "clear" in cond.lower():
            advice.append("Clear sky: ideal day for field operations.")
        if not advice:
            advice.append("Moderate weather: good conditions for most farm activities.")
        return advice

    def get_weather_summary_text(self, location: str) -> str:
        """Returns a short plain-text summary for injection into LLM prompts."""
        try:
            raw = self.get_weather_forecast(location)
            d   = json.loads(raw)
            alerts_text = "; ".join(
                a["en"] for a in d.get("farming_alerts", [])
            ) or "No major alerts."
            return (
                f"Location: {d.get('location','?')}, "
                f"Temp: {d.get('temperature','?')}°C, "
                f"Humidity: {d.get('humidity','?')}%, "
                f"Rainfall: {d.get('rainfall','?')} mm, "
                f"Condition: {d.get('weather_description','?')}. "
                f"Alerts: {alerts_text}"
            )
        except Exception:
            return ""

    def _mock_weather(self, location: str) -> str:
        import random
        temp  = random.randint(27, 36)
        humid = random.randint(65, 92)
        rain  = round(random.uniform(0, 8), 1) if random.random() > 0.6 else 0
        wind  = round(random.uniform(5, 18), 1)
        cond  = random.choice(["Clouds", "Clear", "Rain"])
        alerts = self._build_alerts(temp, humid, rain, wind, cond)
        return json.dumps({
            "location": location, "country": "IN",
            "temperature": temp, "feels_like": temp + 2,
            "humidity": humid, "pressure": 1008,
            "wind_speed": wind, "wind_direction": 180,
            "weather_main": cond, "weather_description": f"{cond} sky (mock)",
            "visibility": 10.0, "rainfall": rain,
            "timestamp": datetime.now().isoformat(),
            "farming_alerts": alerts,
            "farming_advice": self._advice(temp, humid, cond),
            "mock": True,
        })

    def _mock_forecast(self, location: str) -> str:
        import random
        forecasts = []
        for i in range(5):
            dt = datetime.now() + timedelta(days=i)
            forecasts.append({
                "datetime":    dt.strftime("%Y-%m-%d %H:%M:%S"),
                "temperature": random.randint(25, 36),
                "humidity":    random.randint(55, 90),
                "weather":     random.choice(["Clear Sky", "Partly Cloudy", "Light Rain"]),
                "wind_speed":  round(random.uniform(5, 20), 1),
                "rainfall":    round(random.uniform(0, 6), 1) if random.random() > 0.7 else 0,
            })
        return json.dumps({"location": location, "forecasts": forecasts, "mock": True})
