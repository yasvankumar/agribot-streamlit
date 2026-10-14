"""
Market Service – price data for detected crops.
Gets price for a SPECIFIC crop (by name from YOLO result) to show in disease report.
Falls back to simulated data.
"""

import os, json, random
from typing import Optional
import pandas as pd
from datetime import datetime, timedelta
from data.crop_market_data import MARKET_DATA


# Map YOLO crop names → market commodity names
YOLO_TO_MARKET = {
    "Apple":       "Mango",       # closest proxy
    "Blueberry":   "Banana",
    "Cherry":      "Banana",
    "Maize":       "Maize",
    "Corn":        "Maize",
    "Grape":       "Mango",
    "Orange":      "Mango",
    "Peach":       "Mango",
    "Bell Pepper": "Chili",
    "Potato":      "Potato",
    "Raspberry":   "Banana",
    "Soybean":     "Soybean",
    "Squash":      "Potato",
    "Strawberry":  "Banana",
    "Tomato":      "Tomato",
    "Rice":        "Rice",
    "Wheat":       "Wheat",
    "Cotton":      "Cotton",
    "Sugarcane":   "Sugarcane",
    "Groundnut":   "Groundnut",
    "Chili":       "Chili",
    "Onion":       "Onion",
    "Turmeric":    "Turmeric",
    "Coconut":     "Coconut",
}


class MarketService:
    def __init__(self):
        self.data        = MARKET_DATA
        self.commodities = list(self.data.keys())
        self.api_key     = os.getenv("AGMARKNET_API_KEY", "")

    def get_market_prices(self) -> str:
        prices = []
        for commodity in self.commodities:
            base = self.data[commodity]["base_price"]
            curr = base * random.uniform(0.92, 1.08)
            prices.append({
                "commodity":        commodity,
                "tamil_name":       self.data[commodity]["tamil_name"],
                "current_price":    round(curr, 2),
                "unit":             self.data[commodity]["unit"],
                "previous_price":   round(base, 2),
                "change":           round(curr - base, 2),
                "change_pct":       round(((curr - base) / base) * 100, 2),
                "trend":            random.choice(["up", "down", "stable"]),
                "market_location":  self.data[commodity]["market"],
                "last_updated":     datetime.now().strftime("%Y-%m-%d %H:%M"),
            })
        return pd.DataFrame(prices).to_json(orient="records")

    def get_crop_price(self, crop_name: str) -> Optional[dict]:
        """
        Return price info for a specific crop (by YOLO crop name).
        Used to display market info alongside disease detection result.
        """
        commodity = YOLO_TO_MARKET.get(crop_name, crop_name)
        if commodity not in self.data:
            # try partial match
            for key in self.data:
                if crop_name.lower() in key.lower() or key.lower() in crop_name.lower():
                    commodity = key
                    break
            else:
                return None

        base = self.data[commodity]["base_price"]
        curr = base * random.uniform(0.92, 1.08)
        d    = self.data[commodity]
        seasonal = d.get("seasonal_pattern", {})
        peak_months = seasonal.get("peak_price_months", [])
        harvest_months = seasonal.get("harvest_months", [])

        now_month = datetime.now().strftime("%b")
        is_peak   = now_month in peak_months
        is_harvest = now_month in harvest_months

        selling_tip_en = (
            f"Prices are currently {'HIGH – good time to sell! 🟢' if is_peak else 'moderate'}."
            if is_peak else
            (f"Harvest season – prices may dip slightly. Consider storage if possible."
             if is_harvest else "Normal price season. Monitor weekly for optimal selling window.")
        )
        selling_tip_ta = (
            f"இப்போது விலை {'அதிகமாக' if is_peak else 'சாதாரணமாக'} உள்ளது."
        )

        return {
            "commodity":        commodity,
            "tamil_name":       d["tamil_name"],
            "current_price":    round(curr, 2),
            "unit":             d["unit"],
            "market_location":  d["market"],
            "change_pct":       round(((curr - base) / base) * 100, 2),
            "trend":            "up" if curr > base else "down",
            "is_peak_season":   is_peak,
            "peak_months":      peak_months,
            "harvest_months":   harvest_months,
            "selling_tip_en":   selling_tip_en,
            "selling_tip_ta":   selling_tip_ta,
        }

    def get_crop_price_text(self, crop_name: str) -> str:
        """Plain text for LLM prompt injection."""
        info = self.get_crop_price(crop_name)
        if not info:
            return ""
        return (
            f"{info['commodity']} ({info['tamil_name']}): "
            f"₹{info['current_price']} {info['unit']} at {info['market_location']}. "
            f"Trend: {info['trend']}. {info['selling_tip_en']}"
        )

    def get_price_history(self, commodity: str, days: int = 30) -> str:
        if commodity not in self.data:
            return json.dumps({"error": f"{commodity} not found"})
        base = self.data[commodity]["base_price"]
        history = []
        for i in range(days):
            date = datetime.now() - timedelta(days=days - i)
            price = base * random.uniform(0.88, 1.12)
            history.append({"date": date.strftime("%Y-%m-%d"), "price": round(price, 2)})
        return json.dumps(history)

    def get_market_alerts(self) -> list:
        alerts = []
        for commodity in random.sample(self.commodities, min(3, len(self.commodities))):
            t = random.choice(["price_spike", "price_drop", "demand_high", "supply_low"])
            msgs = {
                "price_spike": (f"{commodity} prices rose 8–12% – high demand.",
                                f"{self.data[commodity]['tamil_name']} விலை 8-12% உயர்ந்தது."),
                "price_drop":  (f"{commodity} prices dropped 5–8% – increased supply.",
                                f"{self.data[commodity]['tamil_name']} விலை 5-8% குறைந்தது."),
                "demand_high": (f"High demand for {commodity} in major markets.",
                                f"முக்கிய சந்தைகளில் {self.data[commodity]['tamil_name']} தேவை அதிகம்."),
                "supply_low":  (f"Supply shortage for {commodity} expected next week.",
                                f"அடுத்த வாரம் {self.data[commodity]['tamil_name']} பற்றாக்குறை."),
            }
            en, ta = msgs[t]
            alerts.append({"commodity": commodity, "type": t, "en": en, "ta": ta,
                           "severity": random.choice(["low","medium","high"]),
                           "timestamp": datetime.now().isoformat()})
        return alerts
