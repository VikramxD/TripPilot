"""Tests for TripPilot schemas."""

import pytest
from datetime import date

from trippilot.schemas.travel import (
    TravelQuery,
    TravelPreferences,
    TravelStyle,
    BudgetLevel,
    Destination,
    Hotel,
    Restaurant,
    Attraction,
    Activity,
    DayPlan,
    Itinerary,
    Budget,
    BudgetBreakdown,
)


class TestTravelPreferences:
    """Tests for TravelPreferences schema."""

    def test_default_preferences(self):
        """Test default preferences are set correctly."""
        prefs = TravelPreferences()
        assert prefs.budget_level == BudgetLevel.MODERATE
        assert prefs.pace == "moderate"
        assert TravelStyle.CULTURAL in prefs.styles

    def test_custom_preferences(self):
        """Test custom preferences."""
        prefs = TravelPreferences(
            styles=[TravelStyle.ADVENTURE, TravelStyle.FOODIE],
            budget_level=BudgetLevel.LUXURY,
            daily_budget_usd=500,
            interests=["hiking", "wine tasting"],
            pace="fast",
        )
        assert TravelStyle.ADVENTURE in prefs.styles
        assert prefs.budget_level == BudgetLevel.LUXURY
        assert prefs.daily_budget_usd == 500
        assert "hiking" in prefs.interests


class TestTravelQuery:
    """Tests for TravelQuery schema."""

    def test_minimal_query(self):
        """Test query with minimal required fields."""
        query = TravelQuery(destination="Tokyo, Japan")
        assert query.destination == "Tokyo, Japan"
        assert query.travelers == 1
        assert query.preferences is not None

    def test_full_query(self):
        """Test query with all fields."""
        query = TravelQuery(
            destination="Paris, France",
            origin="New York, USA",
            start_date=date(2025, 6, 1),
            end_date=date(2025, 6, 10),
            duration_days=10,
            travelers=2,
            preferences=TravelPreferences(
                styles=[TravelStyle.ROMANTIC],
                budget_level=BudgetLevel.PREMIUM,
            ),
            special_requests="Anniversary trip",
        )
        assert query.duration_days == 10
        assert query.travelers == 2
        assert query.special_requests == "Anniversary trip"

    def test_query_validation(self):
        """Test query validation."""
        with pytest.raises(ValueError):
            TravelQuery(destination="")  # Empty destination

        with pytest.raises(ValueError):
            TravelQuery(destination="Tokyo", duration_days=100)  # Too long


class TestDestination:
    """Tests for Destination schema."""

    def test_destination_creation(self):
        """Test creating a destination."""
        dest = Destination(
            name="Kyoto",
            country="Japan",
            description="Ancient capital with beautiful temples",
            highlights=["Fushimi Inari", "Kinkaku-ji", "Arashiyama"],
        )
        assert dest.name == "Kyoto"
        assert len(dest.highlights) == 3


class TestBudget:
    """Tests for Budget schema."""

    def test_budget_breakdown_total(self):
        """Test budget breakdown total calculation."""
        breakdown = BudgetBreakdown(
            accommodation=500,
            food=300,
            transportation=200,
            activities=150,
            flights=800,
            miscellaneous=50,
        )
        assert breakdown.total == 2000

    def test_budget_creation(self):
        """Test creating a budget."""
        budget = Budget(
            currency="USD",
            total_estimated=2000,
            breakdown=BudgetBreakdown(
                accommodation=500,
                food=300,
                transportation=200,
                activities=150,
                flights=800,
                miscellaneous=50,
            ),
            daily_average=200,
            money_saving_tips=["Book in advance", "Use public transport"],
        )
        assert budget.total_estimated == 2000
        assert len(budget.money_saving_tips) == 2


class TestItinerary:
    """Tests for Itinerary schema."""

    def test_itinerary_creation(self):
        """Test creating an itinerary."""
        itinerary = Itinerary(
            title="Tokyo Adventure",
            destination=Destination(
                name="Tokyo",
                country="Japan",
                description="Modern metropolis",
            ),
            duration_days=5,
            overview="Explore the best of Tokyo",
            highlights=["Shibuya Crossing", "Senso-ji Temple"],
            daily_plans=[
                DayPlan(
                    day_number=1,
                    title="Arrival & Shibuya",
                    description="Explore Shibuya area",
                    morning=[
                        Activity(
                            name="Hotel Check-in",
                            category="logistics",
                            description="Check into hotel",
                            duration_hours=1,
                        )
                    ],
                    afternoon=[
                        Attraction(
                            name="Shibuya Crossing",
                            category="landmark",
                            description="Famous pedestrian crossing",
                        )
                    ],
                    evening=[],
                    tips=["Get a Suica card"],
                )
            ],
        )
        assert itinerary.title == "Tokyo Adventure"
        assert len(itinerary.daily_plans) == 1
        assert itinerary.daily_plans[0].day_number == 1
