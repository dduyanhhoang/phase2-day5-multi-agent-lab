"""Tests for AnalystAgent."""

from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import ResearchQuery


def test_analyst_agent_transforms_notes():
    """Test that AnalystAgent produces analysis from research notes."""
    state = ResearchState(
        request=ResearchQuery(query="Test query"),
        research_notes="Finding: LangGraph enables cyclic workflows for agent orchestration."
    )
    agent = AnalystAgent()
    result = agent.run(state)

    assert result.analysis_notes, "Should produce analysis notes"
    assert len(result.analysis_notes) > 20
    assert result.route_history[-1] == "analyst"


def test_analyst_handles_missing_notes():
    """Test that AnalystAgent handles missing research_notes gracefully."""
    state = ResearchState(request=ResearchQuery(query="Testing"))
    agent = AnalystAgent()
    result = agent.run(state)
    assert result.analysis_notes == "No research notes to analyze."


def test_analyst_adds_trace_events():
    """Test that AnalystAgent records trace events."""
    state = ResearchState(
        request=ResearchQuery(query="LangGraph"),
        research_notes="Some research notes to analyze."
    )
    agent = AnalystAgent()
    result = agent.run(state)

    trace_names = [t["name"] for t in result.trace]
    assert "llm_complete" in trace_names


def test_analyst_agent_results_have_metadata():
    """Test that agent_results include token and cost metadata."""
    state = ResearchState(
        request=ResearchQuery(query="LangGraph analysis"),
        research_notes="Research notes content here."
    )
    agent = AnalystAgent()
    result = agent.run(state)

    assert len(result.agent_results) > 0
    ar = result.agent_results[-1]
    assert ar.agent == "analyst"
    assert ar.metadata["input_tokens"] is not None
    assert ar.metadata["output_tokens"] is not None
    assert ar.metadata["cost_usd"] is not None
