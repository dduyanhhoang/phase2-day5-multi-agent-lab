"""Tests for SupervisorAgent."""

from unittest.mock import patch
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import ResearchQuery


class MockSource:
    """Simple mock source for testing."""
    def __init__(self):
        self.title = "Test Source"
        self.snippet = "Test snippet"


def test_supervisor_routing_decisions():
    """Test supervisor routing decisions recorded in route_history."""
    state = ResearchState(request=ResearchQuery(query="Testing long enough"))
    agent = SupervisorAgent()

    # Initially should route to researcher (no research_notes)
    result = agent.run(state)
    assert state.route_history[-1] == "researcher"

    # After research_notes set, should route to analyst
    state.research_notes = "Research complete"
    result = agent.run(state)
    assert state.route_history[-1] == "analyst"

    # After analysis, should route to writer
    state.analysis_notes = "Analysis complete"
    result = agent.run(state)
    assert state.route_history[-1] == "writer"

    # After final answer, should route to done
    state.final_answer = "Final answer"
    result = agent.run(state)
    assert state.route_history[-1] == "done"


def test_supervisor_max_iterations():
    """Test that supervisor stops when max iterations reached."""
    from multi_agent_research_lab.core.config import Settings

    # Create a state with high iteration count
    state = ResearchState(request=ResearchQuery(query="Testing"))
    state.iteration = 10  # Force high iteration

    # Create settings with low max_iterations
    settings = Settings(max_iterations=5)

    with patch('multi_agent_research_lab.agents.supervisor.get_settings', return_value=settings):
        agent = SupervisorAgent()
        result = agent.run(state)
        assert state.route_history[-1] == "done"


def test_supervisor_records_trace():
    """Test that supervisor adds trace events."""
    state = ResearchState(request=ResearchQuery(query="LangGraph"))
    agent = SupervisorAgent()
    agent.run(state)

    trace_names = [t["name"] for t in state.trace]
    assert "supervisor" in trace_names

    # Check trace payload structure
    supervisor_traces = [t for t in state.trace if t["name"] == "supervisor"]
    assert len(supervisor_traces) > 0
    payload = supervisor_traces[-1]["payload"]
    assert "decision" in payload
    assert "iteration" in payload
