"""Supervisor / router implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.state import ResearchState


class SupervisorAgent(BaseAgent):
    """Decides which worker should run next and when to stop."""

    name = "supervisor"

    def __init__(self) -> None:
        self.settings = get_settings()

    def run(self, state: ResearchState) -> ResearchState:
        """Update state with routing decision, return modified state."""
        # Check iteration limit
        if state.iteration >= self.settings.max_iterations:
            state.add_trace_event("supervisor", {
                "decision": "stop",
                "reason": "max_iterations_reached",
                "iteration": state.iteration
            })
            state.record_route("done")
            return state

        # Determine next step based on state progression
        # Use output fields (research_notes, analysis_notes, final_answer) as progress indicators
        if state.research_notes is None:
            next_agent = "researcher"
        elif state.analysis_notes is None:
            next_agent = "analyst"
        elif state.final_answer is None:
            next_agent = "writer"
        else:
            next_agent = "done"  # All outputs produced

        state.add_trace_event("supervisor", {
            "decision": next_agent,
            "iteration": state.iteration,
            "sources_collected": len(state.sources),
            "has_research": state.research_notes is not None,
            "has_analysis": state.analysis_notes is not None,
            "has_final_answer": state.final_answer is not None
        })
        state.record_route(next_agent)
        return state
