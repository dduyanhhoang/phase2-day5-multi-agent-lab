"""Tests for WriterAgent."""

from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import ResearchQuery, SourceDocument


def test_writer_agent_produces_answer():
    """Test that WriterAgent produces a final answer."""
    state = ResearchState(
        request=ResearchQuery(query="What is LangGraph?", audience="technical learners"),
        analysis_notes="LangGraph is a library for building stateful, multi-actor LLM applications with cyclic workflows.",
        sources=[
            SourceDocument(
                title="LangGraph Documentation",
                url="https://langchain-ai.github.io/langgraph/",
                snippet="LangGraph enables cyclic workflows for agent orchestration"
            )
        ]
    )
    agent = WriterAgent()
    result = agent.run(state)

    assert result.final_answer, "Should produce final answer"
    assert len(result.final_answer) >= 50
    assert result.route_history[-1] == "writer"


def test_writer_handles_no_sources():
    """Test that WriterAgent handles missing sources gracefully."""
    state = ResearchState(
        request=ResearchQuery(query="Testing long query"),
        analysis_notes="Analysis notes content."
    )
    agent = WriterAgent()
    result = agent.run(state)
    assert result.final_answer
    assert len(result.final_answer) > 0


def test_writer_adds_trace_events():
    """Test that WriterAgent records trace events."""
    state = ResearchState(
        request=ResearchQuery(query="LangGraph"),
        analysis_notes="Analysis content.",
        sources=[SourceDocument(title="Test", snippet="Test snippet")]
    )
    agent = WriterAgent()
    result = agent.run(state)

    trace_names = [t["name"] for t in result.trace]
    assert "llm_complete" in trace_names


def test_writer_agent_results_have_metadata():
    """Test that agent_results include token and cost metadata."""
    state = ResearchState(
        request=ResearchQuery(query="LangGraph"),
        analysis_notes="Analysis content.",
        sources=[SourceDocument(title="Source", snippet="Snippet")]
    )
    agent = WriterAgent()
    result = agent.run(state)

    assert len(result.agent_results) > 0
    ar = result.agent_results[-1]
    assert ar.agent == "writer"
    assert ar.metadata["input_tokens"] is not None
    assert ar.metadata["output_tokens"] is not None
    assert ar.metadata["cost_usd"] is not None
