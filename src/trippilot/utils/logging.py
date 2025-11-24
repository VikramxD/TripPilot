"""
Logging configuration for TripPilot.

Provides structured logging with optional Weights & Biases integration.
"""

import logging
import sys
from typing import Any

import structlog
from rich.console import Console
from rich.logging import RichHandler

from trippilot.core.config import settings


def setup_logging(
    level: str | None = None,
    json_format: bool = False,
    enable_wandb: bool | None = None,
) -> None:
    """
    Configure structured logging for the application.

    Args:
        level: Log level (DEBUG, INFO, WARNING, ERROR)
        json_format: Whether to output JSON format (for production)
        enable_wandb: Whether to enable W&B logging
    """
    level = level or settings.log_level
    enable_wandb = enable_wandb if enable_wandb is not None else settings.wandb_enabled

    # Configure standard logging
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(message)s",
        datefmt="[%X]",
        handlers=[
            RichHandler(
                console=Console(stderr=True),
                rich_tracebacks=True,
                show_path=False,
            )
        ],
    )

    # Silence noisy loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("anthropic").setLevel(logging.WARNING)
    logging.getLogger("chromadb").setLevel(logging.WARNING)

    # Configure structlog
    processors = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
    ]

    if json_format:
        processors.append(structlog.processors.JSONRenderer())
    else:
        processors.append(
            structlog.dev.ConsoleRenderer(
                colors=True,
                exception_formatter=structlog.dev.RichTracebackFormatter(),
            )
        )

    structlog.configure(
        processors=processors,
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, level.upper())
        ),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Initialize W&B if enabled
    if enable_wandb:
        try:
            import wandb

            wandb.init(
                project=settings.wandb_project,
                config={
                    "environment": settings.environment,
                    "llm_provider": settings.llm_provider,
                    "default_model": settings.default_model,
                },
            )
            structlog.get_logger().info("W&B logging enabled", project=settings.wandb_project)
        except Exception as e:
            structlog.get_logger().warning("Failed to initialize W&B", error=str(e))


def get_logger(name: str | None = None) -> structlog.BoundLogger:
    """
    Get a structured logger instance.

    Args:
        name: Optional logger name

    Returns:
        Configured structlog logger
    """
    logger = structlog.get_logger()
    if name:
        logger = logger.bind(logger_name=name)
    return logger


class AgentLogger:
    """Logger wrapper for AI agents with metrics tracking."""

    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger = get_logger(agent_name)
        self._metrics: dict[str, Any] = {
            "calls": 0,
            "successes": 0,
            "failures": 0,
            "total_tokens": 0,
            "total_time_ms": 0,
        }

    def log_call_start(self, **context):
        """Log the start of an agent call."""
        self._metrics["calls"] += 1
        self.logger.info(
            "Agent call started",
            agent=self.agent_name,
            call_number=self._metrics["calls"],
            **context,
        )

    def log_call_success(self, tokens_used: int = 0, execution_time_ms: float = 0, **context):
        """Log a successful agent call."""
        self._metrics["successes"] += 1
        self._metrics["total_tokens"] += tokens_used
        self._metrics["total_time_ms"] += execution_time_ms

        self.logger.info(
            "Agent call succeeded",
            agent=self.agent_name,
            tokens_used=tokens_used,
            execution_time_ms=execution_time_ms,
            **context,
        )

    def log_call_failure(self, error: str, **context):
        """Log a failed agent call."""
        self._metrics["failures"] += 1
        self.logger.error(
            "Agent call failed",
            agent=self.agent_name,
            error=error,
            **context,
        )

    def get_metrics(self) -> dict[str, Any]:
        """Get accumulated metrics."""
        return {
            **self._metrics,
            "success_rate": (
                self._metrics["successes"] / self._metrics["calls"]
                if self._metrics["calls"] > 0
                else 0
            ),
            "avg_tokens_per_call": (
                self._metrics["total_tokens"] / self._metrics["calls"]
                if self._metrics["calls"] > 0
                else 0
            ),
            "avg_time_per_call_ms": (
                self._metrics["total_time_ms"] / self._metrics["calls"]
                if self._metrics["calls"] > 0
                else 0
            ),
        }

    def log_metrics(self):
        """Log current metrics."""
        metrics = self.get_metrics()
        self.logger.info(
            "Agent metrics",
            agent=self.agent_name,
            **metrics,
        )

        # Log to W&B if enabled
        if settings.wandb_enabled:
            try:
                import wandb

                wandb.log({f"{self.agent_name}/{k}": v for k, v in metrics.items()})
            except Exception:
                pass
