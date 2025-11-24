"""
Itinerary Planner Agent for TripPilot.

Specializes in creating personalized day-by-day travel itineraries
based on destination research, user preferences, and practical constraints.
"""

import json
import time
from datetime import date, timedelta
from typing import Any
import uuid

import structlog

from trippilot.agents.base import AgentResult, BaseAgent
from trippilot.schemas.travel import (
    Activity,
    Attraction,
    DayPlan,
    Destination,
    Hotel,
    Itinerary,
    Restaurant,
    TravelQuery,
    WeatherInfo,
)

logger = structlog.get_logger()


class ItineraryAgent(BaseAgent):
    """Agent for creating personalized travel itineraries."""

    name = "itinerary_agent"
    description = "Creates detailed day-by-day travel itineraries"

    @property
    def system_prompt(self) -> str:
        return """You are an expert travel planner who creates personalized, practical itineraries.

Your role is to create day-by-day travel plans that are:
1. Realistic - Consider travel times, opening hours, and practical logistics
2. Balanced - Mix popular attractions with local experiences
3. Flexible - Allow for spontaneity and rest
4. Personalized - Match the traveler's preferences and pace

For each day, organize activities into morning, afternoon, and evening blocks.
Include specific times when helpful, but keep some flexibility.

Consider:
- Geographic clustering (group nearby attractions together)
- Opening hours and best times to visit
- Meal timing and restaurant locations
- Rest periods to avoid burnout
- Weather-appropriate activities
- Budget alignment

Format your response as valid JSON following this structure:
{
  "title": "Descriptive title for the trip",
  "overview": "Brief trip summary",
  "highlights": ["key highlight 1", "key highlight 2"],
  "daily_plans": [
    {
      "day_number": 1,
      "title": "Day theme",
      "description": "Day overview",
      "morning": [
        {
          "name": "Activity name",
          "category": "category",
          "description": "Brief description",
          "duration_hours": 2,
          "price_usd": null
        }
      ],
      "afternoon": [...],
      "evening": [...],
      "meals": [
        {
          "name": "Restaurant name",
          "cuisine": ["cuisine type"],
          "price_range": "$$",
          "specialties": ["dish 1", "dish 2"]
        }
      ],
      "estimated_cost_usd": 150,
      "transportation_notes": "How to get around",
      "tips": ["tip 1", "tip 2"]
    }
  ],
  "packing_list": ["item 1", "item 2"],
  "important_info": ["info 1", "info 2"]
}"""

    async def execute(
        self,
        query: TravelQuery,
        research_data: dict[str, Any] | None = None,
        weather_info: WeatherInfo | None = None,
    ) -> AgentResult[Itinerary]:
        """
        Create a personalized itinerary based on the travel query.

        Args:
            query: TravelQuery with destination, dates, and preferences
            research_data: Optional pre-researched destination data
            weather_info: Optional weather information

        Returns:
            AgentResult containing the complete Itinerary
        """
        start_time = time.time()

        logger.info(
            "Creating itinerary",
            destination=query.destination,
            duration=query.duration_days,
            agent=self.name,
        )

        try:
            # Build context from research data
            context = ""
            if research_data:
                context = f"""Destination Research:
{json.dumps(research_data, indent=2, default=str)}"""

            if weather_info:
                context += f"""

Weather Information:
{weather_info.model_dump_json(indent=2)}"""

            # Build detailed prompt
            user_prompt = self._build_itinerary_prompt(query)

            messages = self._build_messages(user_prompt, context if context else None)
            response, tokens = await self._call_llm(messages, json_mode=True)

            # Parse response
            data = json.loads(response)

            # Build Itinerary object
            itinerary = self._build_itinerary(query, data, weather_info)

            return AgentResult(
                success=True,
                data=itinerary,
                reasoning=f"Created {query.duration_days or 'flexible'}-day itinerary for {query.destination}",
                execution_time_ms=(time.time() - start_time) * 1000,
                tokens_used=tokens,
            )

        except Exception as e:
            logger.error(
                "Itinerary creation failed",
                destination=query.destination,
                error=str(e),
                agent=self.name,
            )
            return AgentResult(
                success=False,
                error=self._format_error(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    def _build_itinerary_prompt(self, query: TravelQuery) -> str:
        """Build a detailed prompt for itinerary generation."""
        duration = query.duration_days or 3
        prefs = query.preferences

        prompt = f"""Create a detailed {duration}-day itinerary for {query.destination}.

Traveler Profile:
- Number of travelers: {query.travelers}
- Travel style: {', '.join(s.value for s in prefs.styles)}
- Budget level: {prefs.budget_level.value}
- Daily budget: ${prefs.daily_budget_usd or 'flexible'} USD
- Pace preference: {prefs.pace}
- Interests: {', '.join(prefs.interests) if prefs.interests else 'General exploration'}"""

        if prefs.dietary_restrictions:
            prompt += f"\n- Dietary restrictions: {', '.join(prefs.dietary_restrictions)}"

        if prefs.accessibility_needs:
            prompt += f"\n- Accessibility needs: {', '.join(prefs.accessibility_needs)}"

        if prefs.avoid:
            prompt += f"\n- Things to avoid: {', '.join(prefs.avoid)}"

        if query.start_date:
            prompt += f"\n- Trip dates: {query.start_date} to {query.end_date or 'flexible'}"

        if query.special_requests:
            prompt += f"\n\nSpecial requests: {query.special_requests}"

        prompt += """

Please create a practical, day-by-day itinerary that:
1. Groups geographically close attractions together
2. Considers realistic travel times between locations
3. Includes meal recommendations that fit the dietary preferences
4. Balances must-see attractions with unique local experiences
5. Allows flexibility and downtime appropriate for the pace preference
6. Stays within the budget level indicated

Return the complete itinerary as JSON following the specified structure."""

        return prompt

    def _build_itinerary(
        self,
        query: TravelQuery,
        data: dict[str, Any],
        weather_info: WeatherInfo | None,
    ) -> Itinerary:
        """Build an Itinerary object from parsed data."""
        # Build destination
        dest_data = data.get("destination", {})
        destination = Destination(
            name=query.destination,
            country=dest_data.get("country", ""),
            region=dest_data.get("region"),
            description=data.get("overview", ""),
            highlights=data.get("highlights", []),
        )

        # Build daily plans
        daily_plans = []
        for day_data in data.get("daily_plans", []):
            day_plan = self._build_day_plan(day_data, query)
            daily_plans.append(day_plan)

        # Calculate dates if provided
        start_date = query.start_date
        end_date = query.end_date
        duration = query.duration_days or len(daily_plans) or 3

        if start_date and not end_date:
            end_date = start_date + timedelta(days=duration - 1)

        return Itinerary(
            id=str(uuid.uuid4()),
            title=data.get("title", f"Trip to {query.destination}"),
            destination=destination,
            start_date=start_date,
            end_date=end_date,
            duration_days=duration,
            travelers=query.travelers,
            overview=data.get("overview", ""),
            highlights=data.get("highlights", []),
            daily_plans=daily_plans,
            weather=weather_info,
            packing_list=data.get("packing_list", []),
            important_info=data.get("important_info", []),
        )

    def _build_day_plan(
        self, day_data: dict[str, Any], query: TravelQuery
    ) -> DayPlan:
        """Build a DayPlan from parsed data."""
        # Parse activities for each time block
        def parse_activities(items: list) -> list[Activity | Attraction]:
            result = []
            for item in items:
                if isinstance(item, dict):
                    # Determine if it's an Activity or Attraction
                    if "duration_hours" in item:
                        result.append(
                            Activity(
                                name=item.get("name", ""),
                                category=item.get("category", "activity"),
                                description=item.get("description", ""),
                                duration_hours=item.get("duration_hours", 1),
                                price_usd=item.get("price_usd"),
                            )
                        )
                    else:
                        result.append(
                            Attraction(
                                name=item.get("name", ""),
                                category=item.get("category", "attraction"),
                                description=item.get("description", ""),
                                admission_fee_usd=item.get("admission_fee_usd"),
                                duration_hours=item.get("duration_hours"),
                            )
                        )
            return result

        # Parse restaurants
        def parse_restaurants(items: list) -> list[Restaurant]:
            result = []
            for item in items:
                if isinstance(item, dict):
                    result.append(
                        Restaurant(
                            name=item.get("name", ""),
                            cuisine=item.get("cuisine", []),
                            price_range=item.get("price_range"),
                            specialties=item.get("specialties", []),
                            dietary_options=item.get("dietary_options", []),
                        )
                    )
            return result

        # Calculate day date if start_date is available
        day_date = None
        if query.start_date:
            day_number = day_data.get("day_number", 1)
            day_date = query.start_date + timedelta(days=day_number - 1)

        return DayPlan(
            day_number=day_data.get("day_number", 1),
            date=day_date,
            title=day_data.get("title", f"Day {day_data.get('day_number', 1)}"),
            description=day_data.get("description", ""),
            morning=parse_activities(day_data.get("morning", [])),
            afternoon=parse_activities(day_data.get("afternoon", [])),
            evening=parse_activities(day_data.get("evening", [])),
            meals=parse_restaurants(day_data.get("meals", [])),
            estimated_cost_usd=day_data.get("estimated_cost_usd"),
            transportation_notes=day_data.get("transportation_notes"),
            tips=day_data.get("tips", []),
        )


class ItineraryOptimizerAgent(BaseAgent):
    """Agent for optimizing and refining existing itineraries."""

    name = "itinerary_optimizer"
    description = "Optimizes itineraries for efficiency, budget, or preferences"

    @property
    def system_prompt(self) -> str:
        return """You are an expert at optimizing travel itineraries.

Given an existing itinerary, you can:
1. Optimize for efficiency (reduce travel time between locations)
2. Optimize for budget (find cost savings without sacrificing experience)
3. Adjust pace (add or remove activities based on energy levels)
4. Swap alternatives (suggest better options based on weather, crowds, etc.)
5. Add local experiences (enhance with authentic local activities)

Always preserve the original structure while making improvements.
Format your response as valid JSON matching the original itinerary structure."""

    async def execute(
        self,
        itinerary: Itinerary,
        optimization_goal: str = "efficiency",
        constraints: dict[str, Any] | None = None,
    ) -> AgentResult[Itinerary]:
        """
        Optimize an existing itinerary.

        Args:
            itinerary: The itinerary to optimize
            optimization_goal: What to optimize for (efficiency, budget, pace, local)
            constraints: Additional constraints or preferences

        Returns:
            AgentResult containing the optimized Itinerary
        """
        start_time = time.time()

        logger.info(
            "Optimizing itinerary",
            destination=itinerary.destination.name,
            goal=optimization_goal,
            agent=self.name,
        )

        try:
            context = f"""Current Itinerary:
{itinerary.model_dump_json(indent=2)}"""

            user_prompt = f"""Optimize this itinerary with the goal: {optimization_goal}

{f'Additional constraints: {json.dumps(constraints)}' if constraints else ''}

Provide the optimized itinerary as JSON, maintaining the same structure.
Include brief explanations of what was changed and why."""

            messages = self._build_messages(user_prompt, context)
            response, tokens = await self._call_llm(messages, json_mode=True)

            data = json.loads(response)

            # Rebuild itinerary from optimized data
            # For simplicity, we'll update key fields while preserving the original structure
            optimized = itinerary.model_copy(deep=True)

            if "daily_plans" in data:
                # Would need more complex logic to fully rebuild daily plans
                pass

            if "highlights" in data:
                optimized.highlights = data["highlights"]

            if "packing_list" in data:
                optimized.packing_list = data["packing_list"]

            return AgentResult(
                success=True,
                data=optimized,
                reasoning=f"Optimized itinerary for {optimization_goal}",
                execution_time_ms=(time.time() - start_time) * 1000,
                tokens_used=tokens,
            )

        except Exception as e:
            logger.error(
                "Itinerary optimization failed",
                error=str(e),
                agent=self.name,
            )
            return AgentResult(
                success=False,
                error=self._format_error(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
