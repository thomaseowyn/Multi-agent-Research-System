from graph.state import ResearchState
from langgraph.types import Send


def fan_out_to_researchers(state: ResearchState) -> list[Send]:
    """Create parallel research_task per task produced by planner"""
    return [
        Send(
            "research_task",
            {
                "current_task": task,
                "max_retries": state['max_retries']
            }
        )
        for task in state['research_tasks']
    ]
