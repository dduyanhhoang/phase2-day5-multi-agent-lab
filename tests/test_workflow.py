"""Tests for MultiAgentWorkflow."""

from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import ResearchQuery


def test_workflow_builds_graph():
    """Test that MultiAgentWorkflow.build() creates a valid graph."""
    wf = MultiAgentWorkflow()
    graph = wf.build()

    assert graph is not None
    # Compiled graph should have invoke method
    assert hasattr(graph, 'invoke')


def test_workflow_runs_to_completion():
    """Test that workflow executes through all agents to final answer."""
    wf = MultiAgentWorkflow()
    state = ResearchState(request=ResearchQuery(query="What is Python?", max_sources=2))
    result = wf.run(state)

    assert result.final_answer, "Should have final answer"
    assert len(result.route_history) >= 3, f"Expected >=3 routes, got {len(result.route_history)}"
    assert "researcher" in result.route_history
    assert "analyst" in result.route_history
    assert "writer" in result.route_history
    assert result.iteration >= 3


def test_workflow_traces_all_steps():
    """Test that workflow records trace events for all steps."""
    wf = MultiAgentWorkflow()
    state = ResearchState(request=ResearchQuery(query="LangGraph"))
    result = wf.run(state)

    trace_names = [t["name"] for t in result.trace]
    assert "search" in trace_names
    assert "llm_complete" in trace_names
    assert "supervisor" in trace_names


def test_workflow_aggregates_costs():
    """Test that workflow result includes cost information in agent_results."""
    wf = MultiAgentWorkflow()
    state = ResearchState(request=ResearchQuery(query="AI agents"))
    result = wf.run(state)

    # Check that agent_results have cost metadata
    total_cost = 0.0
    for ar in result.agent_results:
        if "cost_usd" in ar.metadata:
            total_cost += ar.metadata["cost_usd"]

    assert total_cost > 0, "Expected some cost accumulation from LLM calls"
