"""
Weather tool for TripPilot agents.

Provides weather information and forecasts for destinations.
"""

import asyncio
from dataclasses import dataclass

import httpx
import structlog

from trippilot.schemas.travel import WeatherInfo

logger = structlog.get_logger()


@dataclass
class WeatherData:
    """Weather data for a location."""

    location: str
    temperature_celsius: float
    feels_like_celsius: float
    humidity: int
    description: str
    wind_speed_kmh: float
    condition: str


class WeatherTool:
    """Weather information tool using Open-Meteo API (free, no key required)."""

    GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

    def __init__(self):
        self.client = httpx.AsyncClient(timeout=30.0)

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()

    async def get_coordinates(self, location: str) -> tuple[float, float] | None:
        """
        Get coordinates for a location.

        Args:
            location: City or place name

        Returns:
            Tuple of (latitude, longitude) or None if not found
        """
        try:
            response = await self.client.get(
                self.GEOCODING_URL,
                params={"name": location, "count": 1, "language": "en"},
            )
            response.raise_for_status()
            data = response.json()

            if "results" in data and data["results"]:
                result = data["results"][0]
                return result["latitude"], result["longitude"]

            logger.warning("Location not found", location=location)
            return None
        except Exception as e:
            logger.error("Geocoding failed", error=str(e), location=location)
            return None

    async def get_current_weather(self, location: str) -> WeatherData | None:
        """
        Get current weather for a location.

        Args:
            location: City or place name

        Returns:
            WeatherData object or None if failed
        """
        coords = await self.get_coordinates(location)
        if not coords:
            return None

        lat, lon = coords

        try:
            response = await self.client.get(
                self.WEATHER_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "current": [
                        "temperature_2m",
                        "relative_humidity_2m",
                        "apparent_temperature",
                        "weather_code",
                        "wind_speed_10m",
                    ],
                },
            )
            response.raise_for_status()
            data = response.json()

            current = data.get("current", {})
            weather_code = current.get("weather_code", 0)

            return WeatherData(
                location=location,
                temperature_celsius=current.get("temperature_2m", 0),
                feels_like_celsius=current.get("apparent_temperature", 0),
                humidity=current.get("relative_humidity_2m", 0),
                description=self._get_weather_description(weather_code),
                wind_speed_kmh=current.get("wind_speed_10m", 0),
                condition=self._get_weather_condition(weather_code),
            )
        except Exception as e:
            logger.error("Weather fetch failed", error=str(e), location=location)
            return None

    async def get_forecast(
        self, location: str, days: int = 7
    ) -> list[dict] | None:
        """
        Get weather forecast for a location.

        Args:
            location: City or place name
            days: Number of forecast days (1-16)

        Returns:
            List of daily forecasts or None if failed
        """
        coords = await self.get_coordinates(location)
        if not coords:
            return None

        lat, lon = coords

        try:
            response = await self.client.get(
                self.WEATHER_URL,
                params={
                    "latitude": lat,
                    "longitude": lon,
                    "daily": [
                        "temperature_2m_max",
                        "temperature_2m_min",
                        "precipitation_probability_max",
                        "weather_code",
                    ],
                    "forecast_days": min(days, 16),
                },
            )
            response.raise_for_status()
            data = response.json()

            daily = data.get("daily", {})
            dates = daily.get("time", [])
            max_temps = daily.get("temperature_2m_max", [])
            min_temps = daily.get("temperature_2m_min", [])
            precip_probs = daily.get("precipitation_probability_max", [])
            weather_codes = daily.get("weather_code", [])

            forecasts = []
            for i, date in enumerate(dates):
                forecasts.append(
                    {
                        "date": date,
                        "high_celsius": max_temps[i] if i < len(max_temps) else None,
                        "low_celsius": min_temps[i] if i < len(min_temps) else None,
                        "precipitation_probability": (
                            precip_probs[i] if i < len(precip_probs) else None
                        ),
                        "condition": self._get_weather_condition(
                            weather_codes[i] if i < len(weather_codes) else 0
                        ),
                        "description": self._get_weather_description(
                            weather_codes[i] if i < len(weather_codes) else 0
                        ),
                    }
                )

            return forecasts
        except Exception as e:
            logger.error("Forecast fetch failed", error=str(e), location=location)
            return None

    async def get_weather_info(
        self, destination: str, month: str | None = None
    ) -> WeatherInfo:
        """
        Get comprehensive weather information for travel planning.

        Args:
            destination: Travel destination
            month: Target month for travel (e.g., "December")

        Returns:
            WeatherInfo object with travel-relevant weather data
        """
        current = await self.get_current_weather(destination)
        forecast = await self.get_forecast(destination, days=14)

        # Calculate averages from forecast
        avg_high = None
        avg_low = None
        precip_days = 0

        if forecast:
            highs = [f["high_celsius"] for f in forecast if f["high_celsius"] is not None]
            lows = [f["low_celsius"] for f in forecast if f["low_celsius"] is not None]
            precips = [
                f["precipitation_probability"]
                for f in forecast
                if f["precipitation_probability"] is not None
            ]

            if highs:
                avg_high = sum(highs) / len(highs)
            if lows:
                avg_low = sum(lows) / len(lows)
            if precips:
                precip_days = sum(1 for p in precips if p > 50)

        # Generate packing suggestions
        packing = self._generate_packing_suggestions(avg_high, avg_low, precip_days)

        # Create weather summary
        summary = self._generate_weather_summary(
            destination, avg_high, avg_low, precip_days, current
        )

        period = month or "Next 2 weeks"

        return WeatherInfo(
            destination=destination,
            period=period,
            average_high_celsius=round(avg_high, 1) if avg_high else None,
            average_low_celsius=round(avg_low, 1) if avg_low else None,
            precipitation_days=precip_days,
            humidity_percent=current.humidity if current else None,
            summary=summary,
            packing_suggestions=packing,
        )

    def _get_weather_condition(self, code: int) -> str:
        """Convert WMO weather code to condition string."""
        conditions = {
            0: "clear",
            1: "mostly_clear",
            2: "partly_cloudy",
            3: "overcast",
            45: "fog",
            48: "fog",
            51: "drizzle",
            53: "drizzle",
            55: "drizzle",
            61: "rain",
            63: "rain",
            65: "heavy_rain",
            71: "snow",
            73: "snow",
            75: "heavy_snow",
            80: "showers",
            81: "showers",
            82: "heavy_showers",
            95: "thunderstorm",
            96: "thunderstorm",
            99: "thunderstorm",
        }
        return conditions.get(code, "unknown")

    def _get_weather_description(self, code: int) -> str:
        """Convert WMO weather code to human description."""
        descriptions = {
            0: "Clear sky",
            1: "Mainly clear",
            2: "Partly cloudy",
            3: "Overcast",
            45: "Foggy",
            48: "Depositing rime fog",
            51: "Light drizzle",
            53: "Moderate drizzle",
            55: "Dense drizzle",
            61: "Slight rain",
            63: "Moderate rain",
            65: "Heavy rain",
            71: "Slight snow",
            73: "Moderate snow",
            75: "Heavy snow",
            80: "Slight rain showers",
            81: "Moderate rain showers",
            82: "Violent rain showers",
            95: "Thunderstorm",
            96: "Thunderstorm with hail",
            99: "Thunderstorm with heavy hail",
        }
        return descriptions.get(code, "Unknown conditions")

    def _generate_packing_suggestions(
        self,
        avg_high: float | None,
        avg_low: float | None,
        precip_days: int,
    ) -> list[str]:
        """Generate packing suggestions based on weather."""
        suggestions = []

        if avg_high is not None:
            if avg_high > 30:
                suggestions.extend(
                    [
                        "Light, breathable clothing",
                        "Sunscreen (SPF 30+)",
                        "Sunglasses",
                        "Hat or cap",
                        "Refillable water bottle",
                    ]
                )
            elif avg_high > 20:
                suggestions.extend(
                    [
                        "Light layers",
                        "Comfortable walking shoes",
                        "Sunglasses",
                    ]
                )
            elif avg_high > 10:
                suggestions.extend(
                    [
                        "Layered clothing",
                        "Light jacket or sweater",
                        "Comfortable walking shoes",
                    ]
                )
            else:
                suggestions.extend(
                    [
                        "Warm jacket or coat",
                        "Layers for varying temperatures",
                        "Warm accessories (scarf, gloves)",
                        "Waterproof boots",
                    ]
                )

        if precip_days > 3:
            suggestions.extend(
                [
                    "Rain jacket or umbrella",
                    "Waterproof bag for electronics",
                ]
            )

        if avg_low is not None and avg_low < 10:
            if "Warm jacket" not in str(suggestions):
                suggestions.append("Warm jacket for evenings")

        return list(set(suggestions))  # Remove duplicates

    def _generate_weather_summary(
        self,
        destination: str,
        avg_high: float | None,
        avg_low: float | None,
        precip_days: int,
        current: WeatherData | None,
    ) -> str:
        """Generate a weather summary for the destination."""
        parts = [f"Weather forecast for {destination}:"]

        if avg_high is not None and avg_low is not None:
            parts.append(
                f"Expect temperatures between {avg_low:.0f}C and {avg_high:.0f}C."
            )

        if precip_days > 5:
            parts.append(f"Rain is likely on {precip_days} days - pack accordingly.")
        elif precip_days > 2:
            parts.append("Some rainy days possible - bring a light rain jacket.")
        else:
            parts.append("Generally dry conditions expected.")

        if current:
            parts.append(f"Current conditions: {current.description}, {current.temperature_celsius:.0f}C.")

        return " ".join(parts)
