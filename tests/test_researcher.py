"""Tests for ResearcherAgent."""

from unittest.mock import patch

from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import ResearchQuery


def test_researcher_agent_populates_state():
    """Test that ResearcherAgent populates sources and research_notes."""
    state = ResearchState(request=ResearchQuery(query="What is LangGraph?", max_sources=3))
    agent = ResearcherAgent()
    result = agent.run(state)

    assert result.sources, "Should have collected sources"
    assert len(result.sources) <= 3, "Should respect max_sources"
    assert result.research_notes, "Should have research_notes"
    assert result.agent_results, "Should have agent_results"
    assert result.route_history[-1] == "researcher", "Should record route"


def test_researcher_handles_no_sources():
    """Test that ResearcherAgent handles empty search results gracefully."""
    from multi_agent_research_lab.services.search_client import SearchClient

    state = ResearchState(request=ResearchQuery(query="Testing"))
    with patch.object(SearchClient, 'search', return_value=[]):
        agent = ResearcherAgent()
        result = agent.run(state)
        assert result.research_notes == "No sources found."
        assert result.sources == []


def test_researcher_adds_trace_events():
    """Test that ResearcherAgent records trace events."""
    state = ResearchState(request=ResearchQuery(query="Test query"))
    agent = ResearcherAgent()
    result = agent.run(state)

    trace_names = [t["name"] for t in result.trace]
    assert "search" in trace_names
    assert "llm_complete" in trace_names


def test_researcher_agent_results_have_metadata():
    """Test that agent_results include token and cost metadata."""
    state = ResearchState(request=ResearchQuery(query="LangGraph"))
    agent = ResearcherAgent()
    result = agent.run(state)

    assert len(result.agent_results) > 0
    ar = result.agent_results[-1]
    assert ar.agent == "researcher"
    assert ar.metadata["input_tokens"] is not None
    assert ar.metadata["output_tokens"] is not None
    assert ar.metadata["cost_usd"] is not None
