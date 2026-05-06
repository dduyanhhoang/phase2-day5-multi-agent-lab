"""Full integration test for multi-agent workflow."""

from unittest.mock import patch, MagicMock
import sys


def test_full_workflow_integration():
    """Test complete multi-agent workflow with mocked LLMClient."""
    mock_llm_response = MagicMock(
        content="Mock response from agent",
        input_tokens=50,
        output_tokens=100,
        cost_usd=0.001
    )

    with patch('multi_agent_research_lab.agents.researcher.LLMClient') as mock_llm_researcher, \
         patch('multi_agent_research_lab.agents.analyst.LLMClient') as mock_llm_analyst, \
         patch('multi_agent_research_lab.agents.writer.LLMClient') as mock_llm_writer:

        # All agents use the same mock response
        for mock_llm in [mock_llm_researcher, mock_llm_analyst, mock_llm_writer]:
            mock_llm.return_value.complete.return_value = mock_llm_response

        from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
        from multi_agent_research_lab.core.state import ResearchState
        from multi_agent_research_lab.core.schemas import ResearchQuery

        wf = MultiAgentWorkflow()
        state = ResearchState(request=ResearchQuery(query="What is LangGraph?", max_sources=2))
        result = wf.run(state)

        # Verify final answer is produced
        assert result.final_answer, "Should have final answer"
        assert "Mock response" in result.final_answer

        # Verify all agents were called
        route_names = [r for r in result.route_history if r != "done"]
        assert "researcher" in route_names
        assert "analyst" in route_names
        assert "writer" in route_names

        # Verify sources were collected
        assert len(result.sources) > 0

        # Verify trace events were recorded
        trace_names = [t["name"] for t in result.trace]
        assert "search" in trace_names
        assert "supervisor" in trace_names


def test_workflow_with_short_circuit():
    """Test workflow handles gracefully when sources fail."""
    mock_llm_response = MagicMock(
        content="Mock analysis and answer",
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.0001
    )

    with patch('multi_agent_research_lab.services.search_client.SearchClient.search', return_value=[]), \
         patch('multi_agent_research_lab.agents.researcher.LLMClient') as mock_llm_r, \
         patch('multi_agent_research_lab.agents.analyst.LLMClient') as mock_llm_a, \
         patch('multi_agent_research_lab.agents.writer.LLMClient') as mock_llm_w:

        for mock_llm in [mock_llm_r, mock_llm_a, mock_llm_w]:
            mock_llm.return_value.complete.return_value = mock_llm_response

        from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
        from multi_agent_research_lab.core.state import ResearchState
        from multi_agent_research_lab.core.schemas import ResearchQuery

        wf = MultiAgentWorkflow()
        state = ResearchState(request=ResearchQuery(query="Testing long", max_sources=2))
        result = wf.run(state)

        # Should still complete with analysis and writer even with empty sources
        assert result.research_notes is not None, "Should have research notes"
        assert result.analysis_notes is not None, "Should have analysis notes"
        assert result.final_answer is not None, "Should have final answer"


def test_trace_events_throughout_workflow():
    """Test that trace events are recorded for all steps."""
    mock_llm_response = MagicMock(
        content="Response",
        input_tokens=10,
        output_tokens=20,
        cost_usd=0.0001
    )

    with patch('multi_agent_research_lab.agents.researcher.LLMClient') as mock_llm_r, \
         patch('multi_agent_research_lab.agents.analyst.LLMClient') as mock_llm_a, \
         patch('multi_agent_research_lab.agents.writer.LLMClient') as mock_llm_w:

        for mock_llm in [mock_llm_r, mock_llm_a, mock_llm_w]:
            mock_llm.return_value.complete.return_value = mock_llm_response

        from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
        from multi_agent_research_lab.core.state import ResearchState
        from multi_agent_research_lab.core.schemas import ResearchQuery

        wf = MultiAgentWorkflow()
        state = ResearchState(request=ResearchQuery(query="LangGraph"))
        result = wf.run(state)

        # Check trace contains expected event types
        trace_names = [t["name"] for t in result.trace]
        assert "search" in trace_names
        assert "llm_complete" in trace_names
        assert "supervisor" in trace_names

        # Multiple LLM calls should produce multiple llm_complete events
        assert trace_names.count("llm_complete") >= 3  # researcher, analyst, writer
