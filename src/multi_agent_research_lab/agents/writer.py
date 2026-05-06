"""Writer agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.schemas import AgentResult
from multi_agent_research_lab.services.llm_client import LLMClient


class WriterAgent(BaseAgent):
    """Produces final answer from research and analysis notes."""

    name = "writer"

    def __init__(self) -> None:
        self.llm = LLMClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.final_answer` with a comprehensive response."""
        # Build sources citation list
        sources_list = ""
        if state.sources:
            for i, source in enumerate(state.sources):
                citation = f"[{i+1}]"
                sources_list += f"{citation} {source.title}"
                if source.url:
                    sources_list += f" ({source.url})"
                sources_list += "\n"
        else:
            sources_list = "No sources available."

        prompt = f"""Write a comprehensive answer to the research question.

Question: {state.request.query}

Analysis:
{state.analysis_notes or 'No analysis available.'}

Available Sources for Citation:
{sources_list}

Write a clear, well-structured answer (minimum 200 words) that:
- Directly answers the question
- Cites sources using [1], [2] format when referencing specific information
- Incorporates insights from the analysis
- Is suitable for {state.request.audience}"""

        response = self.llm.complete(
            "You are a technical writer producing clear, well-cited responses.",
            prompt
        )
        state.final_answer = response.content
        state.agent_results.append(AgentResult(
            agent=AgentName.WRITER,
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
