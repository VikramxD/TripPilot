"""
FastAPI application for TripPilot.

Provides REST API endpoints for travel planning and recommendations.
"""

from contextlib import asynccontextmanager
from datetime import date
from typing import Any

import structlog
from fastapi import FastAPI, HTTPException, Query, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from trippilot import __version__
from trippilot.core.config import settings
from trippilot.core.orchestrator import TripPilotOrchestrator
from trippilot.schemas.travel import (
    BudgetLevel,
    TravelPreferences,
    TravelQuery,
    TravelStyle,
    TripRecommendation,
)

logger = structlog.get_logger()

# Global orchestrator instance
orchestrator: TripPilotOrchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage application lifespan."""
    global orchestrator

    # Startup
    logger.info("Starting TripPilot API", version=__version__)
    orchestrator = TripPilotOrchestrator(
        use_rag=True,
        parallel_execution=True,
    )

    yield

    # Shutdown
    logger.info("Shutting down TripPilot API")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    app = FastAPI(
        title="TripPilot API",
        description="AI-powered travel companion with multi-agent architecture",
        version=__version__,
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(health_router)
    app.include_router(travel_router)

    return app


# ============== Request/Response Models ==============

class TripPlanRequest(BaseModel):
    """Request model for trip planning."""

    destination: str = Field(..., min_length=1, description="Destination city/region")
    origin: str | None = Field(default=None, description="Starting location")
    start_date: date | None = Field(default=None, description="Trip start date")
    end_date: date | None = Field(default=None, description="Trip end date")
    duration_days: int | None = Field(default=None, ge=1, le=60, description="Trip duration")
    travelers: int = Field(default=1, ge=1, le=20, description="Number of travelers")
    budget_level: BudgetLevel = Field(default=BudgetLevel.MODERATE, description="Budget category")
    daily_budget_usd: float | None = Field(default=None, ge=0, description="Daily budget")
    travel_styles: list[TravelStyle] = Field(
        default_factory=lambda: [TravelStyle.CULTURAL],
        description="Preferred travel styles",
    )
    interests: list[str] = Field(default_factory=list, description="Specific interests")
    dietary_restrictions: list[str] = Field(default_factory=list, description="Dietary needs")
    accessibility_needs: list[str] = Field(default_factory=list, description="Accessibility needs")
    pace: str = Field(default="moderate", pattern="^(relaxed|moderate|fast)$")
    special_requests: str | None = Field(default=None, description="Special requests")

    def to_travel_query(self) -> TravelQuery:
        """Convert to TravelQuery object."""
        return TravelQuery(
            destination=self.destination,
            origin=self.origin,
            start_date=self.start_date,
            end_date=self.end_date,
            duration_days=self.duration_days,
            travelers=self.travelers,
            preferences=TravelPreferences(
                styles=self.travel_styles,
                budget_level=self.budget_level,
                daily_budget_usd=self.daily_budget_usd,
                interests=self.interests,
                dietary_restrictions=self.dietary_restrictions,
                accessibility_needs=self.accessibility_needs,
                pace=self.pace,
            ),
            special_requests=self.special_requests,
        )


class QuickResearchRequest(BaseModel):
    """Request model for quick research."""

    destination: str = Field(..., min_length=1, description="Destination to research")


class LocalTipsRequest(BaseModel):
    """Request model for local tips."""

    destination: str = Field(..., min_length=1, description="Destination")
    interests: list[str] = Field(default_factory=list, description="Specific interests")


class BudgetEstimateRequest(BaseModel):
    """Request model for budget estimation."""

    destination: str = Field(..., min_length=1, description="Destination")
    duration_days: int = Field(default=5, ge=1, le=60, description="Trip duration")
    travelers: int = Field(default=1, ge=1, description="Number of travelers")
    budget_level: BudgetLevel = Field(default=BudgetLevel.MODERATE, description="Budget category")


class ApiResponse(BaseModel):
    """Standard API response wrapper."""

    success: bool
    data: Any | None = None
    error: str | None = None
    meta: dict[str, Any] = Field(default_factory=dict)


# ============== Routers ==============

from fastapi import APIRouter

health_router = APIRouter(tags=["Health"])
travel_router = APIRouter(prefix="/api/v1", tags=["Travel"])


@health_router.get("/")
async def root():
    """Root endpoint."""
    return {
        "name": "TripPilot API",
        "version": __version__,
        "status": "running",
    }


@health_router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "version": __version__,
        "environment": settings.environment,
    }


@travel_router.post("/plan", response_model=ApiResponse)
async def plan_trip(request: TripPlanRequest) -> ApiResponse:
    """
    Plan a complete trip with AI agents.

    This endpoint coordinates multiple specialized agents to:
    - Research the destination
    - Create a personalized itinerary
    - Estimate budget and find deals
    - Provide local tips and cultural advice
    """
    global orchestrator

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    logger.info(
        "Trip planning request",
        destination=request.destination,
        duration=request.duration_days,
    )

    try:
        query = request.to_travel_query()
        result = await orchestrator.plan_trip(query)

        if not result.success:
            return ApiResponse(
                success=False,
                error=result.error,
                meta={
                    "execution_time_ms": result.total_execution_time_ms,
                    "tokens_used": result.total_tokens_used,
                },
            )

        return ApiResponse(
            success=True,
            data=result.recommendation.model_dump() if result.recommendation else None,
            meta={
                "execution_time_ms": result.total_execution_time_ms,
                "tokens_used": result.total_tokens_used,
                "agents_used": list(result.agent_results.keys()),
            },
        )

    except Exception as e:
        logger.error("Trip planning failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@travel_router.post("/research", response_model=ApiResponse)
async def research_destination(request: QuickResearchRequest) -> ApiResponse:
    """
    Quick research on a destination.

    Returns destination overview, attractions, and practical information
    without full trip planning.
    """
    global orchestrator

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await orchestrator.quick_research(request.destination)

        return ApiResponse(
            success=result.success,
            data=result.data,
            error=result.error,
            meta={
                "execution_time_ms": result.execution_time_ms,
                "tokens_used": result.tokens_used,
                "sources": result.sources,
            },
        )

    except Exception as e:
        logger.error("Research failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@travel_router.post("/local-tips", response_model=ApiResponse)
async def get_local_tips(request: LocalTipsRequest) -> ApiResponse:
    """
    Get local tips and hidden gems for a destination.

    Returns insider knowledge, local favorites, and cultural tips.
    """
    global orchestrator

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await orchestrator.get_local_tips(
            destination=request.destination,
            interests=request.interests if request.interests else None,
        )

        return ApiResponse(
            success=result.success,
            data=result.data,
            error=result.error,
            meta={
                "execution_time_ms": result.execution_time_ms,
                "tokens_used": result.tokens_used,
            },
        )

    except Exception as e:
        logger.error("Local tips failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@travel_router.post("/budget", response_model=ApiResponse)
async def estimate_budget(request: BudgetEstimateRequest) -> ApiResponse:
    """
    Estimate budget for a trip.

    Returns detailed cost breakdown, money-saving tips, and splurge suggestions.
    """
    global orchestrator

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        query = TravelQuery(
            destination=request.destination,
            duration_days=request.duration_days,
            travelers=request.travelers,
            preferences=TravelPreferences(budget_level=request.budget_level),
        )

        result = await orchestrator.estimate_budget(query=query)

        return ApiResponse(
            success=result.success,
            data=result.data.model_dump() if result.data else None,
            error=result.error,
            meta={
                "execution_time_ms": result.execution_time_ms,
                "tokens_used": result.tokens_used,
            },
        )

    except Exception as e:
        logger.error("Budget estimation failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@travel_router.get("/destinations/{destination}")
async def get_destination_info(destination: str) -> ApiResponse:
    """
    Get quick information about a destination.

    Simpler endpoint for destination lookup without full research.
    """
    global orchestrator

    if not orchestrator:
        raise HTTPException(status_code=503, detail="Service not initialized")

    try:
        result = await orchestrator.quick_research(destination)

        return ApiResponse(
            success=result.success,
            data=result.data,
            error=result.error,
        )

    except Exception as e:
        logger.error("Destination info failed", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# Create the app instance
app = create_app()
