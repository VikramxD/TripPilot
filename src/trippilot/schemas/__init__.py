"""Pydantic schemas for TripPilot."""

from trippilot.schemas.travel import (
    Activity,
    Attraction,
    Budget,
    BudgetBreakdown,
    DayPlan,
    Destination,
    Flight,
    Hotel,
    Itinerary,
    LocalTip,
    Restaurant,
    TravelPreferences,
    TravelQuery,
    TripRecommendation,
    WeatherInfo,
)

__all__ = [
    "TravelQuery",
    "TravelPreferences",
    "Destination",
    "Hotel",
    "Restaurant",
    "Attraction",
    "Activity",
    "Flight",
    "DayPlan",
    "Itinerary",
    "Budget",
    "BudgetBreakdown",
    "LocalTip",
    "WeatherInfo",
    "TripRecommendation",
]
