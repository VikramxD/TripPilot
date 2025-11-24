"""
Web search tool for TripPilot agents.

Provides real-time web search capabilities using DuckDuckGo.
"""

import asyncio
from dataclasses import dataclass
from typing import Any

import structlog
from duckduckgo_search import DDGS
from tenacity import retry, stop_after_attempt, wait_exponential

from trippilot.core.config import settings

logger = structlog.get_logger()


@dataclass
class SearchResult:
    """A single search result."""

    title: str
    url: str
    snippet: str
    source: str | None = None


class WebSearchTool:
    """Web search tool using DuckDuckGo."""

    def __init__(self, max_results: int | None = None, timeout: int | None = None):
        self.max_results = max_results or settings.max_search_results
        self.timeout = timeout or settings.search_timeout

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def search(
        self,
        query: str,
        region: str = "wt-wt",
        time_filter: str | None = None,
    ) -> list[SearchResult]:
        """
        Perform a web search.

        Args:
            query: Search query string
            region: Region code (default: worldwide)
            time_filter: Time filter (d=day, w=week, m=month, y=year)

        Returns:
            List of search results
        """
        logger.info("Performing web search", query=query, region=region)

        try:
            # Run synchronous DDGS in thread pool
            loop = asyncio.get_event_loop()
            results = await loop.run_in_executor(
                None,
                lambda: self._sync_search(query, region, time_filter),
            )
            return results
        except Exception as e:
            logger.error("Search failed", error=str(e), query=query)
            return []

    def _sync_search(
        self,
        query: str,
        region: str,
        time_filter: str | None,
    ) -> list[SearchResult]:
        """Synchronous search implementation."""
        with DDGS() as ddgs:
            raw_results = list(
                ddgs.text(
                    query,
                    region=region,
                    timelimit=time_filter,
                    max_results=self.max_results,
                )
            )

        results = []
        for r in raw_results:
            results.append(
                SearchResult(
                    title=r.get("title", ""),
                    url=r.get("href", ""),
                    snippet=r.get("body", ""),
                    source=r.get("source"),
                )
            )

        logger.info("Search completed", query=query, num_results=len(results))
        return results

    async def search_travel(self, destination: str, topic: str) -> list[SearchResult]:
        """
        Search for travel-specific information.

        Args:
            destination: Travel destination
            topic: Topic to search (e.g., "hotels", "restaurants", "attractions")

        Returns:
            List of search results
        """
        query = f"{destination} {topic} travel guide 2024 2025"
        return await self.search(query)

    async def search_hotels(self, destination: str) -> list[SearchResult]:
        """Search for hotels in a destination."""
        return await self.search_travel(destination, "best hotels where to stay")

    async def search_restaurants(self, destination: str) -> list[SearchResult]:
        """Search for restaurants in a destination."""
        return await self.search_travel(destination, "best restaurants local food")

    async def search_attractions(self, destination: str) -> list[SearchResult]:
        """Search for attractions in a destination."""
        return await self.search_travel(destination, "top attractions things to do")

    async def search_local_tips(self, destination: str) -> list[SearchResult]:
        """Search for local tips and insider information."""
        return await self.search_travel(destination, "local tips hidden gems insider guide")

    async def search_budget(self, destination: str) -> list[SearchResult]:
        """Search for budget information."""
        return await self.search_travel(destination, "travel cost budget daily expenses")

    def format_results(self, results: list[SearchResult]) -> str:
        """Format search results as a string for LLM context."""
        if not results:
            return "No search results found."

        formatted = []
        for i, r in enumerate(results, 1):
            formatted.append(f"{i}. **{r.title}**\n   {r.snippet}\n   Source: {r.url}")

        return "\n\n".join(formatted)

    async def search_and_format(self, query: str) -> str:
        """Search and return formatted results."""
        results = await self.search(query)
        return self.format_results(results)
