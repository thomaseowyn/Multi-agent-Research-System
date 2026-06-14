from graph.state import ResearchTask, ResearchFinding
from agent.researcher import _run_single_research_task
from agent.validator import validate_finding
from agent.retry_planner import revise_query


def research_task_node(state: dict) -> dict:
    """
    Runs one research task to completion, including retries.
    This is the target of Send() fan out, so only receive Send() payload, not full state
    """
    task: ResearchTask = state['current_task']
    max_retries: int = state['max_retries']

    finding: ResearchFinding | None = None

    while True:
        print(f"\n[ResearchTask] '{task['topic']}' "
              f"(attempt {task['retry_count'] + 1}/{max_retries + 1})")
        print(f"  Query: {task['search_query']}")

        finding = _run_single_research_task(task)

        verdict, reason = validate_finding(task, finding)
        finding['is_valid'] = (verdict == "VALID")
        finding['validation_reason'] = reason

        print(f"  Verdict: {verdict} ({reason})")

        if verdict == "VALID":
            break

        if task['retry_count'] >= max_retries:
            print(f"  Retries exhausted for '{task['topic']}' — giving up.")
            break

        # Revise query and try again
        task = revise_query(task, finding, reason)

    update: dict = {"findings": [finding]}
    if finding['is_valid']:
        update['validated_findings'] = [finding]

    return update
