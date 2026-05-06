"""Search client abstraction for ResearcherAgent."""

from multi_agent_research_lab.core.schemas import SourceDocument


class SearchClient:
    """Provider-agnostic search client with mock data for testing."""

    def search(self, query: str, max_results: int = 5) -> list[SourceDocument]:
        """Search for documents relevant to a query.

        Returns mock data for testing. Production implementation can use Tavily,
        Bing, SerpAPI, or other search providers.
        """
        # Generate mock results based on query
        mock_results = [
            SourceDocument(
                title=f"About {query}",
                url=f"https://example.com/{query.replace(' ', '-')}",
                snippet=f"Comprehensive resource about {query}: Covers core concepts, examples, and best practices for understanding and applying {query} in real-world scenarios.",
                metadata={"source": "mock", "relevance": 0.95, "index": 0}
            ),
            SourceDocument(
                title=f"Introduction to {query}",
                url=None,
                snippet=f"Beginner's guide to {query}: Learn the fundamentals with step-by-step tutorials and practical examples to get started quickly.",
                metadata={"source": "mock", "relevance": 0.85, "index": 1}
            ),
            SourceDocument(
                title=f"Advanced {query} Techniques",
                url=f"https://tech.example.com/{query.replace(' ', '-')}/advanced",
                snippet=f"Deep dive into advanced {query} patterns, performance optimization, and production-ready implementations for experienced developers.",
                metadata={"source": "mock", "relevance": 0.75, "index": 2}
            ),
            SourceDocument(
                title=f"{query.title()} Best Practices",
                url=f"https://docs.example.com/{query.replace(' ', '-')}",
                snippet=f"Industry-standard best practices for {query}: Code organization, error handling, testing strategies, and scalability considerations.",
                metadata={"source": "mock", "relevance": 0.70, "index": 3}
            ),
            SourceDocument(
                title=f"Common Pitfalls in {query}",
                url=None,
                snippet=f"Avoid these common mistakes when working with {query}: Debugging tips, known issues, and solutions to frequently encountered problems.",
                metadata={"source": "mock", "relevance": 0.65, "index": 4}
            ),
        ]

        return mock_results[:max_results]
