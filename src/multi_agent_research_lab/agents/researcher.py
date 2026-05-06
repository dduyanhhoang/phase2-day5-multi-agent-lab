"""Researcher agent implementation."""

from multi_agent_research_lab.agents.base import BaseAgent
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.core.schemas import AgentName
from multi_agent_research_lab.core.schemas import AgentResult
from multi_agent_research_lab.services.llm_client import LLMClient
from multi_agent_research_lab.services.search_client import SearchClient


class ResearcherAgent(BaseAgent):
    """Collects sources and creates concise research notes."""

    name = "researcher"

    def __init__(self) -> None:
        self.llm = LLMClient()
        self.search = SearchClient()

    def run(self, state: ResearchState) -> ResearchState:
        """Populate `state.sources` and `state.research_notes`."""
        query = state.request.query

        # Search for sources
        sources = self.search.search(query, max_results=state.request.max_sources)
        state.sources = sources
        state.add_trace_event("search", {"query": query, "num_sources": len(sources)})

        # Generate research notes with LLM if sources found
        if sources:
            source_text = "\n".join(f"- {s.title}: {s.snippet}" for s in sources)
            prompt = f"""Based on these sources, write concise research notes about: {query}

Sources:
{source_text}

Write 2-3 paragraphs summarizing the key information:"""

            response = self.llm.complete(
                "You are a research assistant summarizing information from multiple sources.",
                prompt
            )
            state.research_notes = response.content
            state.agent_results.append(AgentResult(
                agent=AgentName.RESEARCHER,
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
        else:
            state.research_notes = "No sources found."

        state.record_route(self.name)
        return state
