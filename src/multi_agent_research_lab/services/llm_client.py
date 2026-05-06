"""LLM client abstraction.

Production note: agents should depend on this interface instead of importing an SDK directly.
"""

import os
from dataclasses import dataclass
from typing import Any

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from tenacity import retry, stop_after_attempt, wait_exponential


@dataclass(frozen=True)
class LLMResponse:
    content: str
    input_tokens: int | None = None
    output_tokens: int | None = None
    cost_usd: float | None = None


# Pricing per 1K tokens (as of 2025) - gpt-4o-mini
_MODEL_PRICING: dict[str, dict[str, float]] = {
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "o1-mini": {"input": 0.0011, "output": 0.0044},
    "o1": {"input": 0.015, "output": 0.060},
    "o3-mini": {"input": 0.0011, "output": 0.0044},
}


def _calculate_cost(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """Calculate the cost in USD for a given model and token counts."""
    pricing = _MODEL_PRICING.get(model)
    if not pricing:
        return None  # Unknown model pricing
    input_cost = (input_tokens / 1000) * pricing["input"]
    output_cost = (output_tokens / 1000) * pricing["output"]
    return round(input_cost + output_cost, 6)


class LLMClient:
    """Provider-agnostic LLM client skeleton."""

    def __init__(self) -> None:
        """Initialize the LLM client with OpenAI configuration."""
        self.settings = get_settings()
        self._client: Any = None
        self._model: str = self.settings.openai_model

    @property
    def client(self) -> Any:
        """Lazy-initialize OpenAI client."""
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError as e:
                raise StudentTodoError(
                    "OpenAI package not installed. Install with: uv pip install openai"
                ) from e

            api_key = self.settings.openai_api_key or os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise StudentTodoError(
                    "OpenAI API key not configured. Set OPENAI_API_KEY in environment or .env file."
                )

            self._client = OpenAI(
                api_key=api_key,
                timeout=self.settings.timeout_seconds,
            )
        return self._client

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def complete(self, system_prompt: str, user_prompt: str) -> LLMResponse:
        """Return a model completion.

        Features:
        - Retry with exponential backoff (3 attempts max)
        - Timeout configured via settings
        - Token logging and cost calculation
        - Provider-agnostic interface
        """
        if not system_prompt or not user_prompt:
            raise ValueError("Both system_prompt and user_prompt are required")

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            response = self.client.chat.completions.create(
                model=self._model,
                messages=messages,
                timeout=self.settings.timeout_seconds,
            )

            content = response.choices[0].message.content or ""
            input_tokens = response.usage.prompt_tokens if response.usage else None
            output_tokens = response.usage.completion_tokens if response.usage else None
            cost_usd = _calculate_cost(self._model, input_tokens or 0, output_tokens or 0)

            return LLMResponse(
                content=content,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                cost_usd=cost_usd,
            )

        except Exception as e:
            # Re-raise for tenacity to handle retryable errors
            raise
