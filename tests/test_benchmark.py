"""Tests for benchmark module."""

from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.core.schemas import BenchmarkMetrics, AgentName, SourceDocument
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import ResearchQuery, AgentResult


def test_benchmark_aggregates_metrics():
    """Test that benchmark aggregates costs and calculates quality score."""
    def mock_runner(query: str) -> ResearchState:
        return ResearchState(
            request=ResearchQuery(query=query),
            final_answer="Test answer with sufficient length to score quality points.",
            sources=[SourceDocument(title="Source", snippet="Test snippet")],
            agent_results=[
                AgentResult(
                    agent=AgentName.WRITER,
                    content="Test",
                    metadata={
                        "input_tokens": 100,
                        "output_tokens": 50,
                        "cost_usd": 0.0005
                    }
                )
            ]
        )

    state, metrics = run_benchmark("test_run", "test query", mock_runner)

    assert isinstance(metrics, BenchmarkMetrics)
    assert metrics.run_name == "test_run"
    assert metrics.latency_seconds >= 0
    assert metrics.estimated_cost_usd == 0.0005
    assert metrics.quality_score > 0
    assert "Sources: 1" in metrics.notes


def test_benchmark_quality_scoring():
    """Test quality scoring logic."""
    def runner_with_params(answer_len: int, num_sources: int):
        def runner(query: str) -> ResearchState:
            return ResearchState(
                request=ResearchQuery(query=query),
                final_answer="x" * answer_len,
                sources=[SourceDocument(title=f"S{i}", snippet="test") for i in range(num_sources)],
                agent_results=[]
            )
        return runner

    # Short answer, few sources -> lower quality
    state, metrics = run_benchmark("short", "testing", runner_with_params(50, 1))
    assert metrics.quality_score < 5

    # Long answer, many sources -> higher quality (capped at 10)
    state, metrics = run_benchmark("long", "testing", runner_with_params(1000, 10))
    assert metrics.quality_score >= 8


def test_benchmark_handles_missing_costs():
    """Test that benchmark handles agent_results without cost gracefully."""
    def mock_runner(query: str) -> ResearchState:
        return ResearchState(
            request=ResearchQuery(query=query),
            final_answer="Answer",
            sources=[],
            agent_results=[
                AgentResult(
                    agent=AgentName.RESEARCHER,
                    content="Test",
                    metadata={}  # No cost info
                )
            ]
        )

    state, metrics = run_benchmark("no_cost", "testing", mock_runner)
    assert metrics.estimated_cost_usd == 0.0


def test_benchmark_latency_measurement():
    """Test that latency is measured accurately."""
    def slow_runner(query: str) -> ResearchState:
        import time
        time.sleep(0.01)  # Small delay to ensure measurable latency
        return ResearchState(
            request=ResearchQuery(query=query),
            final_answer="Done"
        )

    state, metrics = run_benchmark("timed", "testing", slow_runner)
    assert metrics.latency_seconds >= 0.01
