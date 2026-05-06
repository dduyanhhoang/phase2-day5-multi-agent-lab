"""Tracing hooks with optional LangSmith integration."""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

from multi_agent_research_lab.core.config import get_settings

_settings = get_settings()
_USE_LANGSMITH = _settings.langsmith_api_key is not None

if _USE_LANGSMITH:
    try:
        from langsmith import Client as LangSmithClient
        _langsmith_client = LangSmithClient(api_key=_settings.langsmith_api_key)
    except ImportError:
        _USE_LANGSMITH = False


@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Create a tracing span with optional LangSmith integration.

    When LANGSMITH_API_KEY is configured, creates spans in LangSmith.
    Otherwise, uses simple in-memory timing.
    """
    span: dict[str, Any] = {
        "name": name,
        "attributes": attributes or {},
        "duration_seconds": None
    }

    start_time = perf_counter()
    run = None

    if _USE_LANGSMITH:
        try:
            run = _langsmith_client.create_run(  # type: ignore[func-returns-value]
                name=name,
                run_type="chain",
                inputs=attributes or {},
                project_name=_settings.langsmith_project
            )
            if run is not None:
                span["langsmith_run_id"] = run.id
        except Exception:
            # Silently fall back to no-op if LangSmith fails
            start_time = perf_counter()
    else:
        start_time = perf_counter()

    try:
        yield span
    finally:
        duration = perf_counter() - start_time
        span["duration_seconds"] = round(duration, 6)

        if _USE_LANGSMITH and run is not None:
            try:
                _langsmith_client.update_run(
                    run.id,
                    end_time=True,
                    outputs={"duration_seconds": duration}
                )
            except Exception:
                pass  # Don't fail if tracing update errors
