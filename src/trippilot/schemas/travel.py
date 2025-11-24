"""
Travel domain schemas for TripPilot.

Defines structured outputs for all travel-related data using Pydantic v2.
"""

from datetime import date, datetime
from enum import Enum
from typing import Annotated

from pydantic import BaseModel, Field, HttpUrl


class TravelStyle(str, Enum):
    """Travel style preferences."""

    ADVENTURE = "adventure"
    RELAXATION = "relaxation"
    CULTURAL = "cultural"
    FOODIE = "foodie"
    BUDGET = "budget"
    LUXURY = "luxury"
    FAMILY = "family"
    ROMANTIC = "romantic"
    SOLO = "solo"
    BUSINESS = "business"


class BudgetLevel(str, Enum):
    """Budget level categories."""

    BUDGET = "budget"
    MODERATE = "moderate"
    PREMIUM = "premium"
    LUXURY = "luxury"


class TravelPreferences(BaseModel):
    """User's travel preferences and constraints."""

    styles: list[TravelStyle] = Field(
        default_factory=lambda: [TravelStyle.CULTURAL],
        description="Preferred travel styles",
    )
    budget_level: BudgetLevel = Field(
        default=BudgetLevel.MODERATE, description="Budget category"
    )
    daily_budget_usd: float | None = Field(
        default=None, ge=0, description="Daily budget in USD"
    )
    interests: list[str] = Field(
        default_factory=list, description="Specific interests (e.g., museums, hiking)"
    )
    dietary_restrictions: list[str] = Field(
        default_factory=list, description="Dietary needs (e.g., vegetarian, halal)"
    )
    accessibility_needs: list[str] = Field(
        default_factory=list, description="Accessibility requirements"
    )
    pace: Annotated[str, Field(pattern="^(relaxed|moderate|fast)$")] = "moderate"
    avoid: list[str] = Field(
        default_factory=list, description="Things to avoid (e.g., crowds, heights)"
    )


class TravelQuery(BaseModel):
    """A user's travel planning query."""

    destination: str = Field(..., min_length=1, description="Destination city/region")
    origin: str | None = Field(default=None, description="Starting location")
    start_date: date | None = Field(default=None, description="Trip start date")
    end_date: date | None = Field(default=None, description="Trip end date")
    duration_days: int | None = Field(
        default=None, ge=1, le=60, description="Trip duration in days"
    )
    travelers: int = Field(default=1, ge=1, le=20, description="Number of travelers")
    preferences: TravelPreferences = Field(default_factory=TravelPreferences)
    special_requests: str | None = Field(
        default=None, description="Any special requests or notes"
    )
    query_text: str | None = Field(
        default=None, description="Natural language query from user"
    )


class Destination(BaseModel):
    """Information about a travel destination."""

    name: str = Field(..., description="Destination name")
    country: str = Field(..., description="Country name")
    region: str | None = Field(default=None, description="Region or state")
    description: str = Field(..., description="Brief description of the destination")
    highlights: list[str] = Field(
        default_factory=list, description="Top highlights and attractions"
    )
    best_time_to_visit: str | None = Field(
        default=None, description="Recommended visiting season"
    )
    average_daily_cost_usd: float | None = Field(
        default=None, ge=0, description="Average daily cost for tourists"
    )
    languages: list[str] = Field(default_factory=list, description="Languages spoken")
    currency: str | None = Field(default=None, description="Local currency")
    timezone: str | None = Field(default=None, description="Timezone")
    visa_info: str | None = Field(default=None, description="Visa requirements summary")
    safety_rating: Annotated[int, Field(ge=1, le=5)] | None = Field(
        default=None, description="Safety rating 1-5"
    )
    image_url: HttpUrl | None = Field(default=None, description="Destination image")


class Hotel(BaseModel):
    """Hotel or accommodation information."""

    name: str = Field(..., description="Hotel name")
    address: str | None = Field(default=None, description="Full address")
    star_rating: Annotated[float, Field(ge=1, le=5)] | None = Field(
        default=None, description="Star rating"
    )
    price_per_night_usd: float | None = Field(
        default=None, ge=0, description="Price per night in USD"
    )
    amenities: list[str] = Field(default_factory=list, description="Available amenities")
    review_score: Annotated[float, Field(ge=0, le=10)] | None = Field(
        default=None, description="Average review score"
    )
    review_summary: str | None = Field(
        default=None, description="Summary of guest reviews"
    )
    booking_url: HttpUrl | None = Field(default=None, description="Booking link")
    distance_to_center: str | None = Field(
        default=None, description="Distance to city center"
    )
    highlights: list[str] = Field(
        default_factory=list, description="Key selling points"
    )


class Restaurant(BaseModel):
    """Restaurant information."""

    name: str = Field(..., description="Restaurant name")
    cuisine: list[str] = Field(default_factory=list, description="Cuisine types")
    address: str | None = Field(default=None, description="Full address")
    price_range: str | None = Field(
        default=None, description="Price range (e.g., $$, $$$)"
    )
    average_cost_usd: float | None = Field(
        default=None, ge=0, description="Average meal cost per person"
    )
    rating: Annotated[float, Field(ge=0, le=5)] | None = Field(
        default=None, description="Average rating"
    )
    specialties: list[str] = Field(
        default_factory=list, description="Signature dishes"
    )
    dietary_options: list[str] = Field(
        default_factory=list, description="Dietary accommodations"
    )
    reservation_required: bool = False
    opening_hours: str | None = Field(default=None, description="Operating hours")


class Attraction(BaseModel):
    """Tourist attraction information."""

    name: str = Field(..., description="Attraction name")
    category: str = Field(..., description="Category (museum, park, landmark, etc.)")
    description: str = Field(..., description="Description of the attraction")
    address: str | None = Field(default=None, description="Location address")
    admission_fee_usd: float | None = Field(
        default=None, ge=0, description="Admission fee in USD"
    )
    duration_hours: float | None = Field(
        default=None, ge=0, description="Suggested visit duration in hours"
    )
    best_time: str | None = Field(
        default=None, description="Best time to visit"
    )
    tips: list[str] = Field(default_factory=list, description="Visitor tips")
    rating: Annotated[float, Field(ge=0, le=5)] | None = Field(
        default=None, description="Average visitor rating"
    )
    accessibility: str | None = Field(
        default=None, description="Accessibility information"
    )


class Activity(BaseModel):
    """Activity or experience."""

    name: str = Field(..., description="Activity name")
    category: str = Field(..., description="Activity category")
    description: str = Field(..., description="Activity description")
    duration_hours: float = Field(..., ge=0, description="Duration in hours")
    price_usd: float | None = Field(default=None, ge=0, description="Price in USD")
    difficulty_level: str | None = Field(
        default=None, description="Difficulty (easy, moderate, challenging)"
    )
    included: list[str] = Field(
        default_factory=list, description="What's included"
    )
    requirements: list[str] = Field(
        default_factory=list, description="Requirements or prerequisites"
    )
    booking_required: bool = False
    best_season: str | None = Field(
        default=None, description="Best season for this activity"
    )


class Flight(BaseModel):
    """Flight information."""

    airline: str = Field(..., description="Airline name")
    flight_number: str | None = Field(default=None, description="Flight number")
    departure_airport: str = Field(..., description="Departure airport code")
    arrival_airport: str = Field(..., description="Arrival airport code")
    departure_time: datetime | None = Field(default=None, description="Departure time")
    arrival_time: datetime | None = Field(default=None, description="Arrival time")
    duration_hours: float | None = Field(
        default=None, ge=0, description="Flight duration"
    )
    price_usd: float | None = Field(default=None, ge=0, description="Price in USD")
    class_type: str = Field(default="economy", description="Cabin class")
    stops: int = Field(default=0, ge=0, description="Number of stops")


class DayPlan(BaseModel):
    """A single day's itinerary."""

    day_number: int = Field(..., ge=1, description="Day number in the trip")
    date: date | None = Field(default=None, description="Actual date")
    title: str = Field(..., description="Day theme or title")
    description: str = Field(..., description="Day overview")
    morning: list[Activity | Attraction] = Field(
        default_factory=list, description="Morning activities"
    )
    afternoon: list[Activity | Attraction] = Field(
        default_factory=list, description="Afternoon activities"
    )
    evening: list[Activity | Attraction] = Field(
        default_factory=list, description="Evening activities"
    )
    meals: list[Restaurant] = Field(
        default_factory=list, description="Restaurant recommendations"
    )
    accommodation: Hotel | None = Field(
        default=None, description="Where to stay"
    )
    estimated_cost_usd: float | None = Field(
        default=None, ge=0, description="Estimated daily cost"
    )
    transportation_notes: str | None = Field(
        default=None, description="How to get around"
    )
    tips: list[str] = Field(default_factory=list, description="Tips for the day")


class BudgetBreakdown(BaseModel):
    """Detailed budget breakdown."""

    accommodation: float = Field(default=0, ge=0, description="Accommodation costs")
    food: float = Field(default=0, ge=0, description="Food and dining costs")
    transportation: float = Field(default=0, ge=0, description="Transportation costs")
    activities: float = Field(default=0, ge=0, description="Activities and attractions")
    flights: float = Field(default=0, ge=0, description="Flight costs")
    miscellaneous: float = Field(default=0, ge=0, description="Other expenses")

    @property
    def total(self) -> float:
        """Calculate total budget."""
        return (
            self.accommodation
            + self.food
            + self.transportation
            + self.activities
            + self.flights
            + self.miscellaneous
        )


class Budget(BaseModel):
    """Trip budget estimate."""

    currency: str = Field(default="USD", description="Budget currency")
    total_estimated: float = Field(..., ge=0, description="Total estimated cost")
    breakdown: BudgetBreakdown = Field(
        default_factory=BudgetBreakdown, description="Cost breakdown by category"
    )
    daily_average: float | None = Field(
        default=None, ge=0, description="Average daily cost"
    )
    money_saving_tips: list[str] = Field(
        default_factory=list, description="Tips to save money"
    )
    splurge_suggestions: list[str] = Field(
        default_factory=list, description="Worth-it splurges"
    )


class LocalTip(BaseModel):
    """Local insider tip."""

    category: str = Field(..., description="Tip category")
    tip: str = Field(..., description="The tip content")
    source: str | None = Field(default=None, description="Source of the tip")
    location: str | None = Field(default=None, description="Relevant location")


class WeatherInfo(BaseModel):
    """Weather information for a destination."""

    destination: str = Field(..., description="Destination name")
    period: str = Field(..., description="Time period (e.g., 'December 2024')")
    average_high_celsius: float | None = Field(
        default=None, description="Average high temperature"
    )
    average_low_celsius: float | None = Field(
        default=None, description="Average low temperature"
    )
    precipitation_days: int | None = Field(
        default=None, ge=0, description="Expected rainy days"
    )
    humidity_percent: int | None = Field(
        default=None, ge=0, le=100, description="Average humidity"
    )
    summary: str = Field(..., description="Weather summary")
    packing_suggestions: list[str] = Field(
        default_factory=list, description="What to pack"
    )


class Itinerary(BaseModel):
    """Complete travel itinerary."""

    id: str | None = Field(default=None, description="Unique itinerary ID")
    title: str = Field(..., description="Itinerary title")
    destination: Destination = Field(..., description="Main destination info")
    start_date: date | None = Field(default=None, description="Trip start date")
    end_date: date | None = Field(default=None, description="Trip end date")
    duration_days: int = Field(..., ge=1, description="Total days")
    travelers: int = Field(default=1, ge=1, description="Number of travelers")
    overview: str = Field(..., description="Trip overview")
    highlights: list[str] = Field(
        default_factory=list, description="Trip highlights"
    )
    daily_plans: list[DayPlan] = Field(
        default_factory=list, description="Day-by-day plans"
    )
    budget: Budget | None = Field(default=None, description="Budget estimate")
    weather: WeatherInfo | None = Field(default=None, description="Weather info")
    local_tips: list[LocalTip] = Field(
        default_factory=list, description="Local tips and insights"
    )
    packing_list: list[str] = Field(
        default_factory=list, description="Suggested packing list"
    )
    important_info: list[str] = Field(
        default_factory=list, description="Important travel information"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow, description="Creation timestamp"
    )


class TripRecommendation(BaseModel):
    """A travel recommendation response."""

    query: TravelQuery = Field(..., description="Original query")
    itinerary: Itinerary | None = Field(
        default=None, description="Generated itinerary"
    )
    alternative_destinations: list[Destination] = Field(
        default_factory=list, description="Alternative destination suggestions"
    )
    hotels: list[Hotel] = Field(
        default_factory=list, description="Hotel recommendations"
    )
    flights: list[Flight] = Field(
        default_factory=list, description="Flight options"
    )
    confidence_score: Annotated[float, Field(ge=0, le=1)] = Field(
        default=0.8, description="Recommendation confidence"
    )
    sources: list[str] = Field(
        default_factory=list, description="Information sources used"
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Generation timestamp"
    )
