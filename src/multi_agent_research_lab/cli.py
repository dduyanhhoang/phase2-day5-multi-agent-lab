"""Command-line entrypoint for the lab starter."""

from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.panel import Panel

from multi_agent_research_lab.core.config import get_settings
from multi_agent_research_lab.core.errors import StudentTodoError
from multi_agent_research_lab.core.schemas import AgentName, AgentResult, ResearchQuery
from multi_agent_research_lab.core.state import ResearchState
from multi_agent_research_lab.graph.workflow import MultiAgentWorkflow
from multi_agent_research_lab.observability.logging import configure_logging
from multi_agent_research_lab.evaluation.benchmark import run_benchmark
from multi_agent_research_lab.evaluation.report import render_markdown_report
from multi_agent_research_lab.services.llm_client import LLMClient

app = typer.Typer(help="Multi-Agent Research Lab starter CLI")
console = Console()


def _init() -> None:
    settings = get_settings()
    configure_logging(settings.log_level)


@app.command()
def baseline(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run a minimal single-agent baseline."""

    _init()
    client = LLMClient()

    # Define runner for benchmark
    def runner(q: str) -> ResearchState:
        resp = client.complete(
            system_prompt="You are a research assistant. Provide clear, concise, well-cited answers.",
            user_prompt=q
        )
        return ResearchState(
            request=ResearchQuery(query=q),
            final_answer=resp.content,
            agent_results=[
                AgentResult(
                    agent=AgentName.WRITER,
                    content=resp.content,
                    metadata={
                        "input_tokens": resp.input_tokens,
                        "output_tokens": resp.output_tokens,
                        "cost_usd": resp.cost_usd
                    }
                )
            ]
        )

    # Run benchmark
    state_result, metrics = run_benchmark("baseline", query, runner)

    # Display result
    console.print(Panel.fit(state_result.final_answer or "No answer", title="Single-Agent Baseline"))

    # Save report
    Path("reports").mkdir(exist_ok=True)
    report_path = Path("reports/benchmark.md")
    if report_path.exists():
        existing = report_path.read_text()
        if "| baseline |" in existing:
            existing = existing.replace(
                existing.split("| baseline |")[1].split("\n")[0],
                f" {metrics.latency_seconds:.2f} | {metrics.estimated_cost_usd:.6f} | {metrics.quality_score:.1f} | {metrics.notes} |"
            )
            report = existing
        else:
            report = existing + "\n" + render_markdown_report([metrics])
    else:
        report = render_markdown_report([metrics])
    report_path.write_text(report)

    # Save trace
    trace_path = Path("reports/baseline_trace.json")
    import json
    trace_path.write_text(json.dumps(state_result.trace, indent=2))


@app.command("multi-agent")
def multi_agent(
    query: Annotated[str, typer.Option("--query", "-q", help="Research query")],
) -> None:
    """Run the multi-agent workflow."""

    _init()
    workflow = MultiAgentWorkflow()
    state = ResearchState(request=ResearchQuery(query=query))

    try:
        result = workflow.run(state)
    except StudentTodoError as exc:
        console.print(Panel.fit(str(exc), title="Expected TODO", style="yellow"))
        raise typer.Exit(code=2) from exc

    console.print(Panel.fit(result.final_answer or "No answer produced", title="Multi-Agent Result"))

    # Save benchmark report
    Path("reports").mkdir(exist_ok=True)
    _, metrics = run_benchmark("multi-agent", query, lambda q: workflow.run(ResearchState(request=ResearchQuery(query=q))))

    # Append or create report
    report_path = Path("reports/benchmark.md")
    if report_path.exists():
        existing = report_path.read_text()
        if "| multi-agent |" in existing:
            existing = existing.replace(
                existing.split("| multi-agent |")[1].split("\n")[0],
                f" {metrics.latency_seconds:.2f} | {metrics.estimated_cost_usd:.6f} | {metrics.quality_score:.1f} | {metrics.notes} |"
            )
            report = existing
        else:
            report = existing + "\n" + render_markdown_report([metrics])
    else:
        report = render_markdown_report([metrics])

    report_path.write_text(report)

    # Save trace for this run
    trace_path = Path("reports/multi-agent_trace.json")
    import json
    trace_path.write_text(json.dumps(result.trace, indent=2))


if __name__ == "__main__":
    app()