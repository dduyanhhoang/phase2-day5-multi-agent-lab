"""Benchmark module for single-agent vs multi-agent comparison."""

from time import perf_counter
from typing import Callable

from multi_agent_research_lab.core.schemas import BenchmarkMetrics
from multi_agent_research_lab.core.state import ResearchState

Runner = Callable[[str], ResearchState]


def run_benchmark(run_name: str, query: str, runner: Runner) -> tuple[ResearchState, BenchmarkMetrics]:
    """Run benchmark with cost aggregation and quality scoring.

    Metrics collected:
    - latency_seconds: Total execution time
    - estimated_cost_usd: Sum of costs from all agent_results
    - quality_score: Based on answer length (max 8) + source count (max 2) = max 10
    - notes: Summary of sources, iterations, route path
    """
    started = perf_counter()
    state = runner(query)
    latency = perf_counter() - started

    # Aggregate costs from agent_results metadata
    total_cost = 0.0
    for result in state.agent_results:
        cost = result.metadata.get("cost_usd")
        if cost:
            total_cost += cost

    # Quality scoring:
    # - Answer length component (max 8 points, up to 800 chars)
    # - Source count component (max 2 points, up to 10 sources)
    quality = 0.0
    if state.final_answer:
        quality += min(len(state.final_answer) / 100, 8)
    if state.sources:
        quality += min(len(state.sources) / 5, 2)

    metrics = BenchmarkMetrics(
        run_name=run_name,
        latency_seconds=round(latency, 4),
        estimated_cost_usd=round(total_cost, 6),
        quality_score=round(quality, 1),
        notes=f"Sources: {len(state.sources)}, Iterations: {state.iteration}, Routes: {'->'.join(state.route_history)}"
    )
    return state, metrics
