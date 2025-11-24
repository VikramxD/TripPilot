"""Tests for TripPilot tools."""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from trippilot.tools.search import WebSearchTool, SearchResult
from trippilot.tools.weather import WeatherTool


class TestWebSearchTool:
    """Tests for WebSearchTool."""

    def test_search_result_creation(self):
        """Test creating a search result."""
        result = SearchResult(
            title="Test Title",
            url="https://example.com",
            snippet="Test snippet",
            source="example.com",
        )
        assert result.title == "Test Title"
        assert result.url == "https://example.com"

    def test_format_results_empty(self):
        """Test formatting empty results."""
        tool = WebSearchTool()
        formatted = tool.format_results([])
        assert formatted == "No search results found."

    def test_format_results(self):
        """Test formatting search results."""
        tool = WebSearchTool()
        results = [
            SearchResult(
                title="Result 1",
                url="https://example1.com",
                snippet="Snippet 1",
            ),
            SearchResult(
                title="Result 2",
                url="https://example2.com",
                snippet="Snippet 2",
            ),
        ]
        formatted = tool.format_results(results)
        assert "Result 1" in formatted
        assert "Result 2" in formatted
        assert "https://example1.com" in formatted


class TestWeatherTool:
    """Tests for WeatherTool."""

    def test_weather_description(self):
        """Test weather code to description conversion."""
        tool = WeatherTool()
        assert tool._get_weather_description(0) == "Clear sky"
        assert tool._get_weather_description(61) == "Slight rain"
        assert tool._get_weather_description(95) == "Thunderstorm"

    def test_weather_condition(self):
        """Test weather code to condition conversion."""
        tool = WeatherTool()
        assert tool._get_weather_condition(0) == "clear"
        assert tool._get_weather_condition(61) == "rain"
        assert tool._get_weather_condition(71) == "snow"

    def test_packing_suggestions_hot(self):
        """Test packing suggestions for hot weather."""
        tool = WeatherTool()
        suggestions = tool._generate_packing_suggestions(35, 25, 1)
        assert any("sunscreen" in s.lower() for s in suggestions)
        assert any("light" in s.lower() for s in suggestions)

    def test_packing_suggestions_cold(self):
        """Test packing suggestions for cold weather."""
        tool = WeatherTool()
        suggestions = tool._generate_packing_suggestions(5, -5, 2)
        assert any("warm" in s.lower() or "jacket" in s.lower() for s in suggestions)

    def test_packing_suggestions_rainy(self):
        """Test packing suggestions for rainy weather."""
        tool = WeatherTool()
        suggestions = tool._generate_packing_suggestions(20, 15, 10)
        assert any("rain" in s.lower() or "umbrella" in s.lower() for s in suggestions)
