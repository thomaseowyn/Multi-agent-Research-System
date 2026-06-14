from dotenv import load_dotenv
import os
from graph.builder import build_agent
from graph.state import ResearchState

load_dotenv()


def run_research(company: str):
    """Run the graph pipeline and return the final report"""

    # compiled graph
    app = build_agent()

    initial_state = ResearchState(
        company=company,
        research_tasks=[],
        findings=[],
        validated_findings=[],
        final_report="",
        max_retries=2,
    )

    print(f"\n{"="*60}")
    print(f"Startin research: {company}")
    print(f"\n{"="*60}")

    # invoke the graph
    final_state = app.invoke(initial_state, config={"recursion_limit": 50})

    return final_state["final_report"]


if __name__ == "__main__":
    report = run_research("OpenAI")
    print(f"\n{"="*60}")
    print("FINAL REPORT")
    print(f"\n{"="*60}")
    print(report)

    with open("report.md", "w") as f:
        f.write(report)
    print("\nReport saved to report.md")
