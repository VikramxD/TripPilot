"""
Local Expert Agent for TripPilot.

Specializes in providing insider knowledge, hidden gems, local customs,
and authentic experiences that tourists typically miss.
"""

import json
import time
from typing import Any

import structlog

from trippilot.agents.base import AgentResult, BaseAgent
from trippilot.schemas.travel import LocalTip, Restaurant, TravelQuery

logger = structlog.get_logger()


class LocalExpertAgent(BaseAgent):
    """Agent providing local insider knowledge and recommendations."""

    name = "local_expert_agent"
    description = "Provides local tips, hidden gems, and authentic experiences"

    @property
    def system_prompt(self) -> str:
        return """You are a local expert who has lived in or extensively explored many destinations.

Your role is to share insider knowledge that helps travelers experience destinations authentically:

1. Hidden Gems: Lesser-known attractions, neighborhoods, and viewpoints
2. Local Favorites: Where locals actually eat, drink, and hang out
3. Cultural Insights: Customs, etiquette, and social norms
4. Practical Tips: Scams to avoid, local hacks, best times to visit places
5. Authentic Experiences: Activities that give a real taste of local life
6. Food & Drink: Local specialties, street food, and authentic restaurants
7. Neighborhoods: Which areas to explore for different experiences

Avoid generic tourist advice. Focus on unique, specific recommendations
that travelers won't find in standard guidebooks.

Format your response as valid JSON:
{
  "hidden_gems": [
    {
      "name": "Place name",
      "category": "viewpoint/restaurant/neighborhood/etc",
      "description": "Why it's special",
      "location": "Where to find it",
      "tip": "How to best experience it"
    }
  ],
  "local_favorites": {
    "restaurants": [...],
    "cafes": [...],
    "bars": [...],
    "markets": [...]
  },
  "cultural_tips": [
    {
      "category": "etiquette/custom/warning",
      "tip": "The actual tip",
      "context": "Why this matters"
    }
  ],
  "neighborhoods": [
    {
      "name": "Neighborhood name",
      "vibe": "Description of atmosphere",
      "best_for": "What it's best for",
      "highlights": ["thing1", "thing2"]
    }
  ],
  "food_guide": {
    "must_try": ["dish1", "dish2"],
    "where_to_try": ["place1", "place2"],
    "food_tips": ["tip1", "tip2"]
  },
  "local_hacks": ["hack1", "hack2"],
  "avoid": ["tourist_trap1", "scam1"]
}"""

    async def execute(
        self,
        query: TravelQuery | None = None,
        destination: str | None = None,
        interests: list[str] | None = None,
    ) -> AgentResult[dict[str, Any]]:
        """
        Get local expert knowledge for a destination.

        Args:
            query: TravelQuery with destination and preferences
            destination: Simple destination string
            interests: Specific interests to focus on

        Returns:
            AgentResult with local tips and recommendations
        """
        start_time = time.time()

        dest_name = destination or (query.destination if query else None)
        if not dest_name:
            return AgentResult(
                success=False,
                error="No destination provided",
            )

        # Get interests from query if not provided directly
        if not interests and query:
            interests = query.preferences.interests

        logger.info(
            "Getting local expertise",
            destination=dest_name,
            interests=interests,
            agent=self.name,
        )

        try:
            # Search for local insights
            search_results = await self._gather_local_insights(dest_name)

            context = f"""Research about local life in {dest_name}:
{search_results}"""

            user_prompt = f"""Share your local expertise about {dest_name}.

Focus on authentic, insider knowledge that helps travelers experience the real {dest_name}:
- Hidden gems that most tourists miss
- Where locals actually go (not tourist traps)
- Cultural tips and etiquette
- Best neighborhoods to explore
- Must-try local food and where to find it
- Practical tips and local hacks
- Things to avoid (scams, tourist traps, etc.)"""

            if interests:
                user_prompt += f"""

Pay special attention to these interests: {', '.join(interests)}"""

            if query:
                user_prompt += f"""

Traveler context:
- Travel style: {', '.join(s.value for s in query.preferences.styles)}
- Budget: {query.preferences.budget_level.value}
- Pace: {query.preferences.pace}"""

            messages = self._build_messages(user_prompt, context)
            response, tokens = await self._call_llm(messages, json_mode=True)

            data = json.loads(response)

            return AgentResult(
                success=True,
                data=data,
                reasoning=f"Gathered local insights for {dest_name}",
                execution_time_ms=(time.time() - start_time) * 1000,
                tokens_used=tokens,
            )

        except Exception as e:
            logger.error(
                "Local expertise gathering failed",
                destination=dest_name,
                error=str(e),
                agent=self.name,
            )
            return AgentResult(
                success=False,
                error=self._format_error(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )

    async def _gather_local_insights(self, destination: str) -> str:
        """Gather local insights from web searches."""
        queries = [
            f"{destination} local tips insider guide hidden gems",
            f"{destination} where locals eat best local food",
            f"{destination} neighborhoods guide local favorites",
            f"{destination} travel tips avoid tourist traps",
            f"{destination} cultural tips etiquette customs",
        ]

        results = []
        for query in queries:
            result = await self._search(query)
            results.append(f"=== {query} ===\n{result}")

        return "\n\n".join(results)

    async def get_restaurant_recommendations(
        self,
        destination: str,
        cuisine_type: str | None = None,
        budget: str = "moderate",
        dietary_restrictions: list[str] | None = None,
    ) -> AgentResult[list[Restaurant]]:
        """
        Get local restaurant recommendations.

        Args:
            destination: Where to find restaurants
            cuisine_type: Optional specific cuisine
            budget: Budget level (budget, moderate, upscale)
            dietary_restrictions: Any dietary needs

        Returns:
            AgentResult with restaurant recommendations
        """
        start_time = time.time()

        logger.info(
            "Getting restaurant recommendations",
            destination=destination,
            cuisine=cuisine_type,
            agent=self.name,
        )

        try:
            search_query = f"{destination} best local restaurants"
            if cuisine_type:
                search_query += f" {cuisine_type}"
            search_query += f" {budget} 2024"

            search_result = await self._search(search_query)

            user_prompt = f"""Recommend authentic local restaurants in {destination}.

Requirements:
- Budget level: {budget}
- Cuisine preference: {cuisine_type or 'local specialties'}
{f'- Dietary needs: {", ".join(dietary_restrictions)}' if dietary_restrictions else ''}

For each restaurant, provide:
- Name
- Type of cuisine
- Price range
- Signature dishes
- What makes it special
- Any tips for visiting

Return as JSON array:
[
  {{
    "name": "Restaurant name",
    "cuisine": ["type1", "type2"],
    "price_range": "$$",
    "average_cost_usd": 25,
    "specialties": ["dish1", "dish2"],
    "dietary_options": ["option1"],
    "why_special": "What makes it stand out",
    "tip": "Reservation needed? Best time to go?"
  }}
]"""

            messages = self._build_messages(user_prompt, search_result)
            response, tokens = await self._call_llm(messages, json_mode=True)

            data = json.loads(response)

            # Handle both array and object responses
            if isinstance(data, dict) and "restaurants" in data:
                restaurants_data = data["restaurants"]
            elif isinstance(data, list):
                restaurants_data = data
            else:
                restaurants_data = []

            restaurants = [
                Restaurant(
                    name=r.get("name", ""),
                    cuisine=r.get("cuisine", []),
                    price_range=r.get("price_range"),
                    average_cost_usd=r.get("average_cost_usd"),
                    specialties=r.get("specialties", []),
                    dietary_options=r.get("dietary_options", []),
                )
                for r in restaurants_data
            ]

            return AgentResult(
                success=True,
                data=restaurants,
                reasoning=f"Found {len(restaurants)} restaurant recommendations",
                execution_time_ms=(time.time() - start_time) * 1000,
                tokens_used=tokens,
            )

        except Exception as e:
            logger.error(
                "Restaurant recommendations failed",
                error=str(e),
                agent=self.name,
            )
            return AgentResult(
                success=False,
                error=self._format_error(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )


class CulturalAdvisorAgent(BaseAgent):
    """Agent specialized in cultural guidance and etiquette."""

    name = "cultural_advisor_agent"
    description = "Provides cultural context, etiquette, and customs information"

    @property
    def system_prompt(self) -> str:
        return """You are a cultural advisor with deep knowledge of customs and etiquette worldwide.

Your role is to help travelers:
1. Understand local customs and traditions
2. Navigate social situations appropriately
3. Show respect for local culture
4. Avoid cultural faux pas
5. Connect more meaningfully with locals

Provide specific, actionable guidance covering:
- Greetings and social interactions
- Dress codes and appearance
- Dining etiquette
- Religious and sacred sites
- Business customs
- Gift-giving traditions
- Taboos and sensitive topics
- Language basics and useful phrases

Format your response as valid JSON:
{
  "overview": "Cultural summary",
  "greetings": {
    "formal": "How to greet formally",
    "informal": "Casual greetings",
    "tips": ["tip1", "tip2"]
  },
  "dress_code": {
    "general": "General advice",
    "religious_sites": "For temples/churches/mosques",
    "business": "Professional settings"
  },
  "dining": {
    "customs": ["custom1", "custom2"],
    "tipping": "Tipping norms",
    "taboos": ["taboo1"]
  },
  "social_norms": ["norm1", "norm2"],
  "taboos": ["taboo1", "taboo2"],
  "useful_phrases": [
    {"phrase": "local phrase", "meaning": "English meaning", "pronunciation": "how to say it"}
  ],
  "dos_and_donts": {
    "dos": ["do1", "do2"],
    "donts": ["dont1", "dont2"]
  }
}"""

    async def execute(
        self,
        destination: str,
        context: str | None = None,
    ) -> AgentResult[dict[str, Any]]:
        """
        Get cultural guidance for a destination.

        Args:
            destination: The destination country/city
            context: Optional specific context (e.g., "business trip", "wedding")

        Returns:
            AgentResult with cultural guidance
        """
        start_time = time.time()

        logger.info(
            "Getting cultural guidance",
            destination=destination,
            context=context,
            agent=self.name,
        )

        try:
            search_result = await self._search(
                f"{destination} culture customs etiquette tips for tourists"
            )

            user_prompt = f"""Provide comprehensive cultural guidance for travelers visiting {destination}.

{f'Specific context: {context}' if context else ''}

Cover:
- How to greet people appropriately
- Dress codes for different settings
- Dining customs and etiquette
- Social norms and expectations
- Things that are taboo or offensive
- Useful local phrases
- Dos and don'ts for respectful travel"""

            messages = self._build_messages(user_prompt, search_result)
            response, tokens = await self._call_llm(messages, json_mode=True)

            data = json.loads(response)

            return AgentResult(
                success=True,
                data=data,
                reasoning=f"Compiled cultural guidance for {destination}",
                execution_time_ms=(time.time() - start_time) * 1000,
                tokens_used=tokens,
            )

        except Exception as e:
            logger.error(
                "Cultural guidance failed",
                error=str(e),
                agent=self.name,
            )
            return AgentResult(
                success=False,
                error=self._format_error(e),
                execution_time_ms=(time.time() - start_time) * 1000,
            )
