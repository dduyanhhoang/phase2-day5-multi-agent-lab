"""Tests for SearchClient."""

from multi_agent_research_lab.services.search_client import SearchClient
from multi_agent_research_lab.core.schemas import SourceDocument


def test_search_returns_sources():
    """Test that search returns SourceDocument objects."""
    client = SearchClient()
    results = client.search("machine learning", max_results=3)

    assert len(results) > 0
    assert all(isinstance(r, SourceDocument) for r in results)
    assert all(r.title and r.snippet for r in results)


def test_search_respects_max_results():
    """Test that max_results parameter is honored."""
    client = SearchClient()
    results = client.search("test", max_results=1)
    assert len(results) == 1

    results = client.search("test", max_results=2)
    assert len(results) == 2

    # Should not exceed available mock data
    results = client.search("test", max_results=10)
    assert len(results) == 5  # Only 5 mock items available


def test_search_includes_query_in_titles():
    """Test that search results include the query in titles."""
    client = SearchClient()
    results = client.search("langgraph")
    assert any("langgraph" in r.title.lower() for r in results)


def test_search_metadata_structure():
    """Test that returned sources have expected metadata."""
    client = SearchClient()
    results = client.search("python")
    for r in results:
        assert "source" in r.metadata
        assert r.metadata["source"] == "mock"
        assert "relevance" in r.metadata
