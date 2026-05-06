"""Analyst agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.schemas import AgentResult
from multi_agent_research_lab.services.llm_client import LLMClient


class AnalystAgent(BaseAgent):
    """Turns research notes into structured insights."""

    name = "analyst"

    def __init__(self) -> None:
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.analysis_notes` with structured analysis."""
        if not state.research_notes:
            state.analysis_notes = "No research notes to analyze."
            state.record_route(self.name)
            return state

        prompt = f"""Analyze these research notes and produce a structured analysis.

Query: {state.request.query}

Research Notes:
{state.research_notes}

Provide your analysis in this format:
1. Key Claims (2-3 main findings)
2. Evidence Strength (strong/moderate/weak for each claim)
3. Identified Gaps or Uncertainties
4. Overall Assessment

Format with clear section headers:"""

        response = self.llm.complete(
            "You are an analyst identifying patterns, evaluating evidence, and surfacing insights.",
            prompt
        )
        state.analysis_notes = response.content
        state.agent_results.append(AgentResult(
            agent=AgentName.ANALYST,
            content=response.content,
            metadata={
                "input_tokens": response.input_tokens,
                "output_tokens": response.output_tokens,
                "cost_usd": response.cost_usd
            }
        ))
        state.add_trace_event("llm_complete", {
            "agent": self.name,
            "input_tokens": response.input_tokens,
            "output_tokens": response.output_tokens,
            "cost_usd": response.cost_usd
        })
        state.record_route(self.name)
        return state
