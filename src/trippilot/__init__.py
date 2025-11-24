"""
TripPilot - AI-powered travel companion with multi-agent architecture.

A 2025-ready intelligent travel system that combines:
- Multi-agent orchestration for specialized travel tasks
- RAG pipeline for contextual travel knowledge
- Real-time search and recommendations
- Personalized itinerary generation
"""

__version__ = "2.0.0"
__author__ = "VikramxD"

from trippilot.core.config import settings
from trippilot.core.orchestrator import TripPilotOrchestrator

__all__ = ["settings", "TripPilotOrchestrator", "__version__"]
