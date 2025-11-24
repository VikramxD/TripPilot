"""
Budget Agent for TripPilot.

Specializes in cost estimation, budget planning, and finding deals
for travel destinations and itineraries.
"""

import json
import time
from typing import Any

import structlog

from trippilot.agents.base import AgentResult, BaseAgent
from trippilot.schemas.travel import (
    Budget,
    BudgetBreakdown,
    BudgetLevel,
    Itinerary,
    TravelQuery,
)

logger = structlog.get_logger()


class BudgetAgent(BaseAgent):
    """Agent for budget estimation and optimization."""

    name = "budget_agent"
    description = "Estimates costs, creates budgets, and finds money-saving tips"

    @property
    def system_prompt(self) -> str:
        return """You are a travel budget expert who helps travelers plan financially.

Your expertise includes:
1. Accurate cost estimation for destinations worldwide
2. Budget breakdown by category (accommodation, food, transport, activities)
3. Money-saving tips specific to each destination
4. Identifying value options that don't sacrifice quality
5. Suggesting worthwhile splurges for each budget level

Provide realistic estimates based on:
- Current prices (consider inflation, seasonality)
- The specific destination's cost of living
- The traveler's budget level and preferences
- Group size and duration

Be specific with numbers and provide ranges when appropriate.
Always explain the assumptions behind your estimates.

Format your response as valid JSON:
{
  "currency": "USD",
  "total_estimated": 0,
  "breakdown": {
    "accommodation": 0,
    "food": 0,
    "transportation": 0,
    "activities": 0,
    "flights": 0,
    "miscellaneous": 0
  },
  "daily_average": 0,
  "money_saving_tips": ["tip1", "tip2"],
  "splurge_suggestions": ["suggestion1", "suggestion2"],
  "assumptions": ["assumption1", "assumption2"],
  "price_ranges": {
    "budget": {"min": 0, "max": 0},
    "moderate": {"min": 0, "max": 0},
    "luxury": {"min": 0, "max": 0}
  }
}"""

    async def execute(
        self,
        query: TravelQuery | None = None,
        destination: str | None = None,
        duration_days: int | None = None,
        travelers: int = 1,
        budget_level: BudgetLevel = BudgetLevel.MODERATE,
        itinerary: Itinerary | None = None,
    ) -> AgentResult[Budget]:
        """
        Estimate budget for a trip.

        Args:
            query: TravelQuery with full trip details
            destination: Simple destination string
            duration_days: Trip duration
            travelers: Number of travelers
            budget_level: Budget category
            itinerary: Optional itinerary for more accurate estimation

        Returns:
            AgentResult containing Budget estimate
        """
        start_time = time.time()

        # Extract parameters from query if provided
        if query:
            destination = query.destination
            duration_days = query.duration_days
            travelers = query.travelers
            budget_level = query.preferences.budget_level

        if not destination:
            return AgentResult(
                success=False,
                error="No destination provided",
            )

        duration_days = duration_days or 5

        logger.info(
            "Estimating budget",
            destination=destination,
            duration=duration_days,
            travelers=travelers,
            budget_level=budget_level.value,
            agent=self.name,
        )

        try:
            # Search for cost information
            search_results = await self._gather_cost_info(destination)

            context = f"""Cost Research for {destination}:
{search_results}"""

            if itinerary:
                context += f"""

Planned Itinerary:
{itinerary.model_dump_json(indent=2)}"""

            user_prompt = f"""Create a detailed budget estimate for:
- Destination: {destination}
- Duration: {duration_days} days
- Travelers: {travelers}
- Budget level: {budget_level.value}

Provide:
1. Total estimated cost in USD
2. Breakdown by category
3. Daily average per person
4. Money-saving tips specific to this destination
5. Worth-it splurges for this budget level
6. Price ranges for different budget levels

Be realistic and specific. Include assumptions behind your estimates."""

            messages = self._build_messages(user_prompt, context)
            response, tokens = await self._call_llm(messages, json_mode=True)

            data = json.loads(response)

            # Build Budget object
            budget = Budget(
                currency=data.get("currency", "USD"),
                total_estimated=data.get("total_estimated", 0),
                breakdown=BudgetBreakdown(
                    accommodation=data.get("breakdown", {}).get("accommodation", 0),
                    food=data.get("breakdown", {}).get("food", 0),
                    transportation=data.get("breakdown", {}).get("transportation", 0),
                    activities=data.get("breakdown", {}).get("activities", 0),
                    flights=data.get("breakdown", {}).get("flights", 0),
                    miscellaneous=data.get("breakdown", {}).get("miscellaneous", 0),
                ),
                daily_average=data.get("daily_average"),
                money_saving_tips=data.get("money_saving_tips", []),
                splurge_suggestions=data.get("splurge_suggestions", []),
            )

            return AgentResult(
                success=True,
                data=budget,
                reasoning=f"Estimated {budget_level.value} budget for {duration_days} days in {destination}",
                execution_time_ms=(time.time() - start_time) * 1000,
                tokens_used=tokens,
            )

        except Exception as e:
            logger.error(
                "Budget estimation failed",
                destination=destination,
                error=str(e),
                agent=self.name,
            )
            return AgentResult(
                success=False,
                error=self._format_error(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _gather_cost_info(self, destination: str) -> str:
        """Gather cost information from web searches."""
        queries = [
            f"{destination} travel cost budget daily expenses 2024",
            f"{destination} hotel prices accommodation cost",
            f"{destination} food prices restaurant costs",
            f"{destination} transportation costs getting around",
        ]

        results = []
        for query in queries:
            result = await self._search(query)
            results.append(f"=== {query} ===\n{result}")

        return "\n\n".join(results)

    async def analyze_itinerary_costs(
        self, itinerary: Itinerary
    ) -> AgentResult[dict[str, Any]]:
        """
        Analyze costs for an existing itinerary.

        Args:
            itinerary: The itinerary to analyze

        Returns:
            AgentResult with detailed cost analysis
        """
        start_time = time.time()

        logger.info(
            "Analyzing itinerary costs",
            destination=itinerary.destination.name,
            duration=itinerary.duration_days,
            agent=self.name,
        )

        try:
            user_prompt = f"""Analyze the costs for this itinerary and provide:
1. Estimated cost for each day
2. Total trip cost
3. Cost breakdown by category
4. Areas where money could be saved
5. Potential hidden costs to watch for

Return as JSON with this structure:
{{
  "daily_costs": [
    {{"day": 1, "estimated": 150, "breakdown": {{...}}}}
  ],
  "total": 0,
  "per_person": 0,
  "savings_opportunities": ["tip1"],
  "hidden_costs": ["cost1"],
  "value_ratings": {{"accommodation": "good", "activities": "excellent"}}
}}"""

            context = f"Itinerary to analyze:\n{itinerary.model_dump_json(indent=2)}"

            messages = self._build_messages(user_prompt, context)
            response, tokens = await self._call_llm(messages, json_mode=True)

            data = json.loads(response)

            return AgentResult(
                success=True,
                data=data,
                reasoning=f"Analyzed costs for {itinerary.duration_days}-day itinerary",
                execution_time_ms=(time.time() - start_time) * 1000,
                tokens_used=tokens,
            )

        except Exception as e:
            logger.error("Cost analysis failed", error=str(e), agent=self.name)
            return AgentResult(
                success=False,
                error=self._format_error(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )


class DealFinderAgent(BaseAgent):
    """Agent specialized in finding travel deals and discounts."""

    name = "deal_finder_agent"
    description = "Finds travel deals, discounts, and money-saving opportunities"

    @property
    def system_prompt(self) -> str:
        return """You are an expert at finding travel deals and discounts.

Your expertise includes:
1. Finding current deals on flights, hotels, and activities
2. Identifying discount programs (city passes, combo tickets)
3. Timing strategies (best times to book, off-peak travel)
4. Loyalty programs and credit card benefits
5. Local deals and coupons

Always provide actionable, specific recommendations.
Include booking tips and timing advice.

Format your response as valid JSON:
{
  "deals": [
    {
      "type": "accommodation/flight/activity/pass",
      "name": "Deal name",
      "description": "What you get",
      "savings": "Amount or percentage saved",
      "booking_tip": "How to get this deal",
      "valid_until": "Expiry if known"
    }
  ],
  "discount_programs": ["program1", "program2"],
  "timing_tips": ["tip1", "tip2"],
  "booking_strategies": ["strategy1", "strategy2"]
}"""

    async def execute(
        self,
        destination: str,
        travel_dates: tuple[str, str] | None = None,
        interests: list[str] | None = None,
    ) -> AgentResult[dict[str, Any]]:
        """
        Find deals and discounts for a destination.

        Args:
            destination: Travel destination
            travel_dates: Optional (start_date, end_date) tuple
            interests: Optional list of interests to focus on

        Returns:
            AgentResult with deals and savings opportunities
        """
        start_time = time.time()

        logger.info(
            "Finding deals",
            destination=destination,
            agent=self.name,
        )

        try:
            # Search for deals
            search_queries = [
                f"{destination} travel deals discounts 2024",
                f"{destination} city pass tourist card",
                f"{destination} cheap flights hotel deals",
            ]

            search_results = []
            for query in search_queries:
                result = await self._search(query)
                search_results.append(result)

            context = "\n\n".join(search_results)

            user_prompt = f"""Find the best current deals and money-saving opportunities for traveling to {destination}.

{f'Travel dates: {travel_dates[0]} to {travel_dates[1]}' if travel_dates else ''}
{f'Interests: {", ".join(interests)}' if interests else ''}

Include:
1. Current deals on accommodation and flights
2. Tourist passes and combo tickets
3. Best booking timing strategies
4. Local discount opportunities"""

            messages = self._build_messages(user_prompt, context)
            response, tokens = await self._call_llm(messages, json_mode=True)

            data = json.loads(response)

            return AgentResult(
                success=True,
                data=data,
                reasoning=f"Found deals and discounts for {destination}",
                execution_time_ms=(time.time() - start_time) * 1000,
                tokens_used=tokens,
            )

        except Exception as e:
            logger.error("Deal finding failed", error=str(e), agent=self.name)
            return AgentResult(
                success=False,
                error=self._format_error(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
