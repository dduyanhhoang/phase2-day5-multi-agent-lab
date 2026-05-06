"""Benchmark report rendering."""

from multi_agent_research_lab.core.schemas import BenchmarkMetrics


def render_markdown_report(metrics: list[BenchmarkMetrics]) -> str:
    """Render detailed benchmark comparison report."""
    lines = [
        "# Multi-Agent Research Lab - Benchmark Report\n",
        "## Summary",
        "| Run | Latency (s) | Cost (USD) | Quality | Notes |",
        "|--:|--:|--:|--:|--|"
    ]

    for item in metrics:
        cost = f"{item.estimated_cost_usd:.6f}" if item.estimated_cost_usd is not None else "-"
        quality = f"{item.quality_score:.1f}" if item.quality_score is not None else "-"
        lines.append(f"| {item.run_name} | {item.latency_seconds:.2f} | {cost} | {quality} | {item.notes} |")

    lines.extend([
        "\n## Analysis",
        "Compare baseline (single-agent) vs multi-agent workflows above.",
        "- Lower latency with acceptable quality indicates efficient routing",
        "- Higher quality at similar cost indicates better synthesis",
        "- Trace details available in each run's trace events",
        ""
    ])

    return "\n".join(lines)
