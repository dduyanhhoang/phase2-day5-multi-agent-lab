"""LangGraph workflow implementation."""

from typing import Any
from langgraph.graph import StateGraph, END
from multi_agent_research_lab.agents.supervisor import SupervisorAgent
from multi_agent_research_lab.agents.researcher import ResearcherAgent
from multi_agent_research_lab.agents.analyst import AnalystAgent
from multi_agent_research_lab.agents.writer import WriterAgent
from multi_agent_research_lab.core.state import ResearchState


class MultiAgentWorkflow:
    """Builds and runs the multi-agent graph.

    Keep orchestration here; keep agent internals in `agents/`.
    """

    def build(self) -> Any:
        """Create a LangGraph StateGraph."""
        workflow = StateGraph(ResearchState)

        # Add nodes - use agent run methods
        workflow.add_node("supervisor", SupervisorAgent().run)
        workflow.add_node("researcher", ResearcherAgent().run)
        workflow.add_node("analyst", AnalystAgent().run)
        workflow.add_node("writer", WriterAgent().run)

        # Set entry point
        workflow.set_entry_point("supervisor")

        # Routing function: looks at route_history to decide next node
        def decide_next_node(state: ResearchState) -> str:
            # Get the most recent route decision from supervisor
            if state.route_history:
                decision = state.route_history[-1]
                if decision == "done" or decision is None:
                    return END
                return decision
            # Default fallback - should not happen
            return "researcher"

        # Conditional edges from supervisor
        workflow.add_conditional_edges(
            "supervisor",
            decide_next_node,
            {
                "researcher": "researcher",
                "analyst": "analyst",
                "writer": "writer",
                END: END
            }
        )

        # All worker nodes go back to supervisor after executing
        for agent_name in ["researcher", "analyst", "writer"]:
            workflow.add_edge(agent_name, "supervisor")

        return workflow.compile()

    def run(self, state: ResearchState) -> ResearchState:
        """Execute the graph and return final state."""
        graph = self.build()
        # LangGraph's invoke returns a dict; convert back to ResearchState
        result_dict = graph.invoke(state)
        # If result is already ResearchState, return it; otherwise reconstruct
        if isinstance(result_dict, ResearchState):
            return result_dict
        return ResearchState(**result_dict)
