"""
Base agent class for TripPilot.

Provides common functionality for all specialized agents.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Generic, TypeVar

import structlog
from openai import AsyncOpenAI
from anthropic import AsyncAnthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from trippilot.core.config import settings
from trippilot.tools.search import WebSearchTool

logger = structlog.get_logger()

T = TypeVar("T")


@dataclass
class AgentResult(Generic[T]):
    """Result from an agent execution."""

    success: bool
    data: T | None = None
    error: str | None = None
    sources: list[str] = field(default_factory=list)
    reasoning: str | None = None
    execution_time_ms: float = 0
    tokens_used: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)


class BaseAgent(ABC):
    """Base class for all TripPilot agents."""

    name: str = "base_agent"
    description: str = "Base agent class"
    model: str = settings.default_model

    def __init__(self, model: str | None = None):
        """Initialize the agent."""
        self.model = model or settings.default_model
        self.search_tool = WebSearchTool()
        self._setup_llm_client()

    def _setup_llm_client(self):
        """Set up the LLM client based on configuration."""
        if settings.llm_provider == "openai" and settings.openai_api_key:
            self.client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value()
            )
            self.provider = "openai"
        elif settings.llm_provider == "anthropic" and settings.anthropic_api_key:
            self.client = AsyncAnthropic(
                api_key=settings.anthropic_api_key.get_secret_value()
            )
            self.provider = "anthropic"
        else:
            # Default to OpenAI client (will fail if no key, but allows testing)
            self.client = AsyncOpenAI(api_key="placeholder")
            self.provider = "openai"
            logger.warning(
                "No API key configured",
                provider=settings.llm_provider,
                agent=self.name,
            )

    @property
    @abstractmethod
    def system_prompt(self) -> str:
        """Get the system prompt for this agent."""
        pass

    @abstractmethod
    async def execute(self, **kwargs) -> AgentResult:
        """Execute the agent's main task."""
        pass

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
    )
    async def _call_llm(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> tuple[str, int]:
        """
        Call the LLM with the given messages.

        Args:
            messages: List of message dicts with role and content
            temperature: Override default temperature
            max_tokens: Override default max tokens
            json_mode: Whether to request JSON output

        Returns:
            Tuple of (response content, tokens used)
        """
        temperature = temperature or settings.temperature
        max_tokens = max_tokens or settings.max_tokens

        logger.debug(
            "Calling LLM",
            agent=self.name,
            provider=self.provider,
            model=self.model,
            num_messages=len(messages),
        )

        try:
            if self.provider == "openai":
                response = await self._call_openai(
                    messages, temperature, max_tokens, json_mode
                )
            else:
                response = await self._call_anthropic(
                    messages, temperature, max_tokens
                )

            content, tokens = response
            logger.info(
                "LLM call successful",
                agent=self.name,
                tokens=tokens,
            )
            return content, tokens

        except Exception as e:
            logger.error("LLM call failed", agent=self.name, error=str(e))
            raise

    async def _call_openai(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
        json_mode: bool,
    ) -> tuple[str, int]:
        """Call OpenAI API."""
        kwargs = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}

        response = await self.client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""
        tokens = response.usage.total_tokens if response.usage else 0

        return content, tokens

    async def _call_anthropic(
        self,
        messages: list[dict[str, str]],
        temperature: float,
        max_tokens: int,
    ) -> tuple[str, int]:
        """Call Anthropic API."""
        # Extract system message if present
        system = None
        filtered_messages = []
        for msg in messages:
            if msg["role"] == "system":
                system = msg["content"]
            else:
                filtered_messages.append(msg)

        kwargs = {
            "model": self.model,
            "messages": filtered_messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }

        if system:
            kwargs["system"] = system

        response = await self.client.messages.create(**kwargs)

        content = response.content[0].text if response.content else ""
        tokens = response.usage.input_tokens + response.usage.output_tokens

        return content, tokens

    async def _search(self, query: str) -> str:
        """Perform a web search and return formatted results."""
        return await self.search_tool.search_and_format(query)

    def _build_messages(
        self,
        user_content: str,
        context: str | None = None,
    ) -> list[dict[str, str]]:
        """Build message list for LLM call."""
        messages = [
            {"role": "system", "content": self.system_prompt},
        ]

        if context:
            messages.append(
                {
                    "role": "user",
                    "content": f"Context information:\n{context}",
                }
            )
            messages.append(
                {
                    "role": "assistant",
                    "content": "I'll use this context to help with your request.",
                }
            )

        messages.append({"role": "user", "content": user_content})

        return messages

    def _format_error(self, error: Exception) -> str:
        """Format an error for the result."""
        return f"{type(error).__name__}: {str(error)}"
