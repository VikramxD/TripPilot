"""
TripPilot Orchestrator - Coordinates multiple AI agents for travel planning.

The orchestrator manages the flow of information between specialized agents
to produce comprehensive travel recommendations and itineraries.
"""

import asyncio
import time
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog

from trippilot.agents.base import AgentResult
from trippilot.agents.budget import BudgetAgent, DealFinderAgent
from trippilot.agents.itinerary import ItineraryAgent
from trippilot.agents.local_expert import CulturalAdvisorAgent, LocalExpertAgent
from trippilot.agents.research import ResearchAgent
from trippilot.core.config import settings
from trippilot.rag.retriever import TravelKnowledgeRetriever
from trippilot.schemas.travel import (
    Budget,
    Itinerary,
    TravelQuery,
    TripRecommendation,
    WeatherInfo,
)

logger = structlog.get_logger()


@dataclass
class OrchestratorResult:
    """Result from the orchestrator."""

    success: bool
    recommendation: TripRecommendation | None = None
    error: str | None = None
    agent_results: dict[str, AgentResult] = field(default_factory=dict)
    total_execution_time_ms: float = 0
    total_tokens_used: int = 0


class TripPilotOrchestrator:
    """
    Main orchestrator that coordinates all TripPilot agents.

    The orchestrator follows this flow:
    1. Parse and validate the travel query
    2. Research the destination (parallel: destination info + weather)
    3. Get local expertise (parallel with budget)
    4. Generate itinerary based on research
    5. Estimate budget and find deals
    6. Compile final recommendation
    """

    def __init__(
        self,
        use_rag: bool = True,
        parallel_execution: bool = True,
    ):
        """
        Initialize the orchestrator.

        Args:
            use_rag: Whether to use RAG for knowledge retrieval
            parallel_execution: Whether to run independent agents in parallel
        """
        self.use_rag = use_rag
        self.parallel_execution = parallel_execution

        # Initialize agents
        self.research_agent = ResearchAgent()
        self.itinerary_agent = ItineraryAgent()
        self.budget_agent = BudgetAgent()
        self.local_expert_agent = LocalExpertAgent()
        self.deal_finder_agent = DealFinderAgent()
        self.cultural_advisor_agent = CulturalAdvisorAgent()

        # Initialize RAG retriever
        if use_rag:
            self.retriever = TravelKnowledgeRetriever()
        else:
            self.retriever = None

        logger.info(
            "Orchestrator initialized",
            use_rag=use_rag,
            parallel_execution=parallel_execution,
        )

    async def plan_trip(self, query: TravelQuery) -> OrchestratorResult:
        """
        Plan a complete trip based on the travel query.

        This is the main entry point that coordinates all agents.

        Args:
            query: TravelQuery with destination, dates, and preferences

        Returns:
            OrchestratorResult with complete trip recommendation
        """
        start_time = time.time()
        agent_results = {}
        total_tokens = 0

        logger.info(
            "Starting trip planning",
            destination=query.destination,
            duration=query.duration_days,
            travelers=query.travelers,
        )

        try:
            # Step 1: Research destination (required for everything else)
            logger.info("Phase 1: Researching destination")
            research_result = await self.research_agent.execute(query=query)
            agent_results["research"] = research_result
            total_tokens += research_result.tokens_used

            if not research_result.success:
                return OrchestratorResult(
                    success=False,
                    error=f"Research failed: {research_result.error}",
                    agent_results=agent_results,
                    total_execution_time_ms=(time.time() - start_time) * 1000,
                )

            research_data = research_result.data
            weather_info = None

            # Extract weather info if available
            if research_data and "weather" in research_data:
                weather_info = WeatherInfo(**research_data["weather"])

            # Step 2: Parallel execution of local expertise, budget, and deals
            logger.info("Phase 2: Gathering local expertise and budget info")

            if self.parallel_execution:
                local_result, budget_result, deals_result, cultural_result = await asyncio.gather(
                    self.local_expert_agent.execute(query=query),
                    self.budget_agent.execute(query=query),
                    self.deal_finder_agent.execute(
                        destination=query.destination,
                        travel_dates=(
                            (str(query.start_date), str(query.end_date))
                            if query.start_date and query.end_date
                            else None
                        ),
                        interests=query.preferences.interests,
                    ),
                    self.cultural_advisor_agent.execute(destination=query.destination),
                    return_exceptions=True,
                )

                # Handle exceptions
                if isinstance(local_result, Exception):
                    local_result = AgentResult(success=False, error=str(local_result))
                if isinstance(budget_result, Exception):
                    budget_result = AgentResult(success=False, error=str(budget_result))
                if isinstance(deals_result, Exception):
                    deals_result = AgentResult(success=False, error=str(deals_result))
                if isinstance(cultural_result, Exception):
                    cultural_result = AgentResult(success=False, error=str(cultural_result))
            else:
                local_result = await self.local_expert_agent.execute(query=query)
                budget_result = await self.budget_agent.execute(query=query)
                deals_result = await self.deal_finder_agent.execute(
                    destination=query.destination
                )
                cultural_result = await self.cultural_advisor_agent.execute(
                    destination=query.destination
                )

            agent_results["local_expert"] = local_result
            agent_results["budget"] = budget_result
            agent_results["deals"] = deals_result
            agent_results["cultural"] = cultural_result

            total_tokens += sum(
                r.tokens_used for r in [local_result, budget_result, deals_result, cultural_result]
                if isinstance(r, AgentResult)
            )

            # Step 3: Generate itinerary
            logger.info("Phase 3: Generating itinerary")
            itinerary_result = await self.itinerary_agent.execute(
                query=query,
                research_data=research_data,
                weather_info=weather_info,
            )
            agent_results["itinerary"] = itinerary_result
            total_tokens += itinerary_result.tokens_used

            if not itinerary_result.success:
                return OrchestratorResult(
                    success=False,
                    error=f"Itinerary generation failed: {itinerary_result.error}",
                    agent_results=agent_results,
                    total_execution_time_ms=(time.time() - start_time) * 1000,
                    total_tokens_used=total_tokens,
                )

            # Step 4: Compile final recommendation
            logger.info("Phase 4: Compiling recommendation")
            itinerary = itinerary_result.data

            # Enhance itinerary with local tips
            if local_result.success and local_result.data:
                local_data = local_result.data
                if "hidden_gems" in local_data:
                    # Add hidden gems to important info
                    for gem in local_data.get("hidden_gems", [])[:3]:
                        itinerary.important_info.append(
                            f"Hidden gem: {gem.get('name', '')} - {gem.get('description', '')}"
                        )

            # Add cultural tips
            if cultural_result.success and cultural_result.data:
                cultural_data = cultural_result.data
                dos_donts = cultural_data.get("dos_and_donts", {})
                for dont in dos_donts.get("donts", [])[:3]:
                    itinerary.important_info.append(f"Cultural tip: {dont}")

            # Add budget
            budget = None
            if budget_result.success and isinstance(budget_result.data, Budget):
                budget = budget_result.data
                itinerary.budget = budget

            # Compile sources
            sources = []
            for result in agent_results.values():
                if isinstance(result, AgentResult) and result.sources:
                    sources.extend(result.sources)
            sources = list(set(sources))[:20]  # Dedupe and limit

            # Create recommendation
            recommendation = TripRecommendation(
                query=query,
                itinerary=itinerary,
                alternative_destinations=[],
                hotels=[],
                flights=[],
                confidence_score=self._calculate_confidence(agent_results),
                sources=sources,
            )

            execution_time = (time.time() - start_time) * 1000

            logger.info(
                "Trip planning completed",
                destination=query.destination,
                execution_time_ms=execution_time,
                total_tokens=total_tokens,
            )

            return OrchestratorResult(
                success=True,
                recommendation=recommendation,
                agent_results=agent_results,
                total_execution_time_ms=execution_time,
                total_tokens_used=total_tokens,
            )

        except Exception as e:
            logger.error(
                "Trip planning failed",
                destination=query.destination,
                error=str(e),
            )
            return OrchestratorResult(
                success=False,
                error=str(e),
                agent_results=agent_results,
                total_execution_time_ms=(time.time() - start_time) * 1000,
                total_tokens_used=total_tokens,
            )

    async def quick_research(self, destination: str) -> AgentResult:
        """
        Quick destination research without full trip planning.

        Args:
            destination: Destination to research

        Returns:
            AgentResult with research data
        """
        return await self.research_agent.execute(destination=destination)

    async def get_local_tips(
        self, destination: str, interests: list[str] | None = None
    ) -> AgentResult:
        """
        Get local tips for a destination.

        Args:
            destination: Destination to get tips for
            interests: Optional interests to focus on

        Returns:
            AgentResult with local tips
        """
        return await self.local_expert_agent.execute(
            destination=destination,
            interests=interests,
        )

    async def estimate_budget(
        self,
        query: TravelQuery | None = None,
        destination: str | None = None,
        duration_days: int = 5,
    ) -> AgentResult:
        """
        Estimate budget for a trip.

        Args:
            query: Full TravelQuery or simple parameters
            destination: Destination if not using query
            duration_days: Trip duration if not using query

        Returns:
            AgentResult with budget estimate
        """
        if query:
            return await self.budget_agent.execute(query=query)
        return await self.budget_agent.execute(
            destination=destination,
            duration_days=duration_days,
        )

    def _calculate_confidence(self, agent_results: dict[str, AgentResult]) -> float:
        """Calculate confidence score based on agent results."""
        total = 0
        success_count = 0

        for result in agent_results.values():
            if isinstance(result, AgentResult):
                total += 1
                if result.success:
                    success_count += 1

        if total == 0:
            return 0.5

        # Base confidence on success rate, with minimum of 0.3
        return max(0.3, success_count / total)

    async def add_knowledge(
        self,
        content: str,
        destination: str | None = None,
        category: str | None = None,
        source: str | None = None,
    ) -> str | None:
        """
        Add knowledge to the RAG system.

        Args:
            content: Knowledge content
            destination: Related destination
            category: Knowledge category
            source: Source of information

        Returns:
            Document ID if successful, None otherwise
        """
        if not self.retriever:
            logger.warning("RAG not enabled, cannot add knowledge")
            return None

        return self.retriever.add_travel_knowledge(
            content=content,
            destination=destination,
            category=category,
            source=source,
        )


# Convenience function for quick trip planning
async def plan_trip(
    destination: str,
    duration_days: int = 5,
    travelers: int = 1,
    **preferences,
) -> TripRecommendation | None:
    """
    Convenience function for quick trip planning.

    Args:
        destination: Where to go
        duration_days: How long
        travelers: How many people
        **preferences: Additional preferences

    Returns:
        TripRecommendation or None if failed
    """
    from trippilot.schemas.travel import TravelPreferences

    query = TravelQuery(
        destination=destination,
        duration_days=duration_days,
        travelers=travelers,
        preferences=TravelPreferences(**preferences) if preferences else TravelPreferences(),
    )

    orchestrator = TripPilotOrchestrator()
    result = await orchestrator.plan_trip(query)

    return result.recommendation if result.success else None
