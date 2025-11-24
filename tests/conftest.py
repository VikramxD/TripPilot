"""Pytest configuration and fixtures."""

import pytest
from datetime import date

from trippilot.schemas.travel import (
    TravelQuery,
    TravelPreferences,
    TravelStyle,
    BudgetLevel,
)


@pytest.fixture
def sample_travel_query():
    """Create a sample travel query for testing."""
    return TravelQuery(
        destination="Tokyo, Japan",
        duration_days=7,
        travelers=2,
        preferences=TravelPreferences(
            styles=[TravelStyle.CULTURAL, TravelStyle.FOODIE],
            budget_level=BudgetLevel.MODERATE,
            interests=["temples", "ramen", "anime"],
            pace="moderate",
        ),
    )


@pytest.fixture
def sample_budget_query():
    """Create a sample query for budget testing."""
    return TravelQuery(
        destination="Paris, France",
        duration_days=5,
        travelers=1,
        preferences=TravelPreferences(
            budget_level=BudgetLevel.BUDGET,
            daily_budget_usd=100,
        ),
    )


@pytest.fixture
def sample_luxury_query():
    """Create a sample luxury trip query."""
    return TravelQuery(
        destination="Maldives",
        start_date=date(2025, 12, 20),
        end_date=date(2025, 12, 27),
        duration_days=7,
        travelers=2,
        preferences=TravelPreferences(
            styles=[TravelStyle.RELAXATION, TravelStyle.ROMANTIC],
            budget_level=BudgetLevel.LUXURY,
        ),
        special_requests="Honeymoon trip, ocean view room preferred",
    )
