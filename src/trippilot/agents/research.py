"""
Research Agent for TripPilot.

Specializes in gathering comprehensive information about travel destinations,
including attractions, culture, practical tips, and current conditions.
"""

import asyncio
import json
import time
from typing import Any

import structlog

from trippilot.agents.base import AgentResult, BaseAgent
from trippilot.schemas.travel import Attraction, Destination, TravelQuery
from trippilot.tools.weather import WeatherTool

logger = structlog.get_logger()


class ResearchAgent(BaseAgent):
    """Agent for researching travel destinations."""

    name = "research_agent"
    description = "Researches travel destinations, attractions, and practical information"

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.weather_tool = WeatherTool()

    @property
    def system_prompt(self) -> str:
        return """You are an expert travel researcher with deep knowledge of destinations worldwide.
Your role is to gather comprehensive, accurate, and up-to-date information about travel destinations.

When researching a destination, provide:
1. Destination Overview: Brief but informative description, location context, and why it's worth visiting
2. Top Attractions: Key sights, landmarks, and must-see places with descriptions
3. Practical Information: Best time to visit, visa requirements, language, currency, safety
4. Cultural Insights: Local customs, etiquette, and cultural highlights
5. Getting There & Around: Transportation options and local mobility

Always be factual and specific. Include concrete details like:
- Names of specific attractions, neighborhoods, and restaurants
- Price ranges when relevant
- Time needed to visit attractions
- Practical tips that help travelers

Format your response as valid JSON matching this structure:
{
  "destination": {
    "name": "string",
    "country": "string",
    "region": "string or null",
    "description": "string",
    "highlights": ["string"],
    "best_time_to_visit": "string",
    "average_daily_cost_usd": number or null,
    "languages": ["string"],
    "currency": "string",
    "timezone": "string",
    "visa_info": "string",
    "safety_rating": 1-5 or null
  },
  "attractions": [
    {
      "name": "string",
      "category": "string",
      "description": "string",
      "admission_fee_usd": number or null,
      "duration_hours": number or null,
      "best_time": "string or null",
      "tips": ["string"],
      "rating": 0-5 or null
    }
  ],
  "practical_tips": ["string"],
  "cultural_notes": ["string"],
  "transportation": "string"
}"""

    async def execute(
        self,
        query: TravelQuery | None = None,
        destination: str | None = None,
    ) -> AgentResult[dict[str, Any]]:
        """
        Research a destination and gather comprehensive information.

        Args:
            query: TravelQuery object with destination and preferences
            destination: Simple destination string (alternative to query)

        Returns:
            AgentResult containing destination info, attractions, and tips
        """
        start_time = time.time()

        dest_name = destination or (query.destination if query else None)
        if not dest_name:
            return AgentResult(
                success=False,
                error="No destination provided",
            )

        logger.info("Researching destination", destination=dest_name, agent=self.name)

        try:
            # Gather information in parallel
            search_results, weather_info = await asyncio.gather(
                self._gather_search_results(dest_name),
                self.weather_tool.get_weather_info(dest_name),
                return_exceptions=True,
            )

            # Handle search results
            if isinstance(search_results, Exception):
                logger.warning("Search failed", error=str(search_results))
                search_results = ""

            # Handle weather info
            if isinstance(weather_info, Exception):
                logger.warning("Weather fetch failed", error=str(weather_info))
                weather_info = None

            # Build context from search results
            context = f"""Search results about {dest_name}:
{search_results}

Weather Information:
{weather_info.model_dump_json(indent=2) if weather_info else 'Not available'}"""

            # Build user prompt
            user_prompt = f"""Research the travel destination: {dest_name}

Please provide comprehensive information for travelers planning to visit.
Focus on practical, actionable information that helps with trip planning.

Return your research as valid JSON following the specified structure."""

            if query:
                user_prompt += f"""

Additional context from traveler:
- Travel style: {', '.join(s.value for s in query.preferences.styles)}
- Budget level: {query.preferences.budget_level.value}
- Interests: {', '.join(query.preferences.interests) or 'General exploration'}
- Duration: {query.duration_days or 'Flexible'} days
- Travelers: {query.travelers}"""

            # Call LLM
            messages = self._build_messages(user_prompt, context)
            response, tokens = await self._call_llm(messages, json_mode=True)

            # Parse response
            try:
                data = json.loads(response)
            except json.JSONDecodeError:
                # Try to extract JSON from response
                logger.warning("Failed to parse JSON response, extracting...")
                data = self._extract_json(response)

            # Add weather info to result
            if weather_info:
                data["weather"] = weather_info.model_dump()

            execution_time = (time.time() - start_time) * 1000

            return AgentResult(
                success=True,
                data=data,
                sources=self._extract_sources(search_results),
                reasoning=f"Researched {dest_name} using web search and weather data",
                execution_time_ms=execution_time,
                tokens_used=tokens,
            )

        except Exception as e:
            logger.error(
                "Research failed",
                destination=dest_name,
                error=str(e),
                agent=self.name,
            )
            return AgentResult(
                success=False,
                error=self._format_error(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _gather_search_results(self, destination: str) -> str:
        """Gather search results from multiple queries."""
        queries = [
            f"{destination} travel guide overview",
            f"{destination} top attractions things to do",
            f"{destination} travel tips practical information",
            f"{destination} local culture customs",
        ]

        results = []
        for query in queries:
            result = await self._search(query)
            results.append(f"=== {query} ===\n{result}")

        return "\n\n".join(results)

    def _extract_json(self, text: str) -> dict:
        """Try to extract JSON from text that might have extra content."""
        # Try to find JSON block
        import re

        # Look for JSON between code blocks
        json_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Look for JSON object directly
        json_match = re.search(r"(\{.*\})", text, re.DOTALL)
        if json_match:
            return json.loads(json_match.group(1))

        # Return empty structure as fallback
        return {
            "destination": {"name": "", "country": "", "description": text},
            "attractions": [],
            "practical_tips": [],
        }

    def _extract_sources(self, search_results: str) -> list[str]:
        """Extract source URLs from search results."""
        import re

        urls = re.findall(r"https?://[^\s<>\"{}|\\^`\[\]]+", search_results)
        return list(set(urls))[:10]  # Dedupe and limit


class AttractionResearchAgent(BaseAgent):
    """Specialized agent for detailed attraction research."""

    name = "attraction_research_agent"
    description = "Researches specific attractions in detail"

    @property
    def system_prompt(self) -> str:
        return """You are an expert on tourist attractions worldwide.
Your role is to provide detailed, practical information about specific attractions.

For each attraction, provide:
1. Full description and historical/cultural significance
2. Practical visiting information (hours, prices, best times)
3. Tips for visitors
4. Nearby attractions or combinations
5. Accessibility information

Always format your response as valid JSON."""

    async def execute(
        self,
        attraction_name: str,
        destination: str,
    ) -> AgentResult[Attraction]:
        """Research a specific attraction in detail."""
        start_time = time.time()

        logger.info(
            "Researching attraction",
            attraction=attraction_name,
            destination=destination,
            agent=self.name,
        )

        try:
            # Search for attraction details
            search_result = await self._search(
                f"{attraction_name} {destination} visitor guide tips"
            )

            user_prompt = f"""Provide detailed information about: {attraction_name} in {destination}

Include practical visitor information, tips, and recommendations.
Return your response as JSON with these fields:
- name
- category
- description (detailed)
- address
- admission_fee_usd
- duration_hours (recommended visit time)
- best_time
- tips (list of visitor tips)
- rating (estimated out of 5)
- accessibility"""

            messages = self._build_messages(user_prompt, search_result)
            response, tokens = await self._call_llm(messages, json_mode=True)

            data = json.loads(response)
            attraction = Attraction(**data)

            return AgentResult(
                success=True,
                data=attraction,
                execution_time_ms=(time.time() - start_time) * 1000,
                tokens_used=tokens,
            )

        except Exception as e:
            logger.error(
                "Attraction research failed",
                attraction=attraction_name,
                error=str(e),
            )
            return AgentResult(
                success=False,
                error=self._format_error(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
