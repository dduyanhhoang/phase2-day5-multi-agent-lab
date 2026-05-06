"""Tests for tracing module."""

from multi_agent_research_lab.observability.tracing import trace_span


def test_trace_span_records_duration():
    """Test that trace_span records duration."""
    with trace_span("test_op") as span:
        pass

    assert "duration_seconds" in span
    assert span["duration_seconds"] > 0
    assert span["name"] == "test_op"


def test_trace_span_with_attributes():
    """Test that trace_span includes attributes."""
    with trace_span("test", {"key": "value"}) as span:
        assert span["attributes"]["key"] == "value"


def test_trace_span_multiple_operations():
    """Test that multiple trace spans work correctly."""
    outer = {}
    inner = {}

    with trace_span("outer") as o:
        outer["span"] = o
        with trace_span("inner") as i:
            inner["span"] = i

    # After both contexts exit, durations should be set
    assert outer["span"]["duration_seconds"] is not None
    assert inner["span"]["duration_seconds"] is not None
    assert outer["span"]["duration_seconds"] >= inner["span"]["duration_seconds"]
