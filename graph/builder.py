from langgraph.graph import StateGraph, START, END

from graph.state import ResearchState
from agent.planner import planner_agent
from agent.synthesizer import synthesizer_agent
from agent.research_task import research_task_node
from graph.edges import fan_out_to_researchers


def build_agent():
    """
    Construct and compile graph
    START -> planner -> researcher -> synthesizer -> END (phase 1)
    """
    graph = StateGraph(ResearchState)

    graph.add_node("planner", planner_agent)
    graph.add_node("research_task", research_task_node)
    graph.add_node("synthesizer", synthesizer_agent)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges("planner", fan_out_to_researchers)

    graph.add_edge("research_task", "synthesizer")
    graph.add_edge("synthesizer", END)

    app = graph.compile()
    return app
