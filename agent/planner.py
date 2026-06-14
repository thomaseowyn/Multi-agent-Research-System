import uuid
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import ResearchState, ResearchTask
from models.schemas import ResearchPlan
from config import get_settings

settings = get_settings()


def planner_agent(state: ResearchState) -> dict:
    """Decomposes a research goal into specific subtasks"""
    llm = ChatOpenAI(
        model=settings.planner_model,
        temperature=0,      # make the model deterministic
        api_key=settings.openai_api_key,
    )

    # with_structured_output() parse and validate the response back into a pydantic instance
    structured_llm = llm.with_structured_output(ResearchPlan)

    system_prompt = """You are a research planning expert. Your job is to break down 
a company research request into specific, actionable research tasks.

For each task:
- Write a clear topic description
- Write an optimized search query that will find high-quality, specific information

Good search queries are specific and include the company name.
Bad: "competitors"
Good: "OpenAI main competitors market share 2024"

Generate exactly {task_count} research tasks that together provide comprehensive 
coverage of the company.""".format(task_count=settings.research_tasks_count)

    human_prompt = f"""Create a comprehensive research plan for: {state["company"]}

The research should cover:
- Company overview and history
- Core products and services  
- Main competitors and market position
- Recent news and developments
- Strengths, weaknesses, and strategic opportunities

Generate {settings.research_tasks_count} specific research tasks."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt),
    ]

    # returns a ResearchPlan pydantic object that is validated
    plan: ResearchPlan = structured_llm.invoke(messages)

    # convert the pydantic object back to the TypedDict for keeping state
    tasks: list[ResearchTask] = [
        ResearchTask(
            task_id=str(uuid.uuid4()),
            topic=task.topic,
            search_query=task.search_query,
            retry_count=0,
        )
        for task in plan.tasks]

    print(f"\n[Planner] Generated {len(tasks)} research tasks:")
    for task in tasks:
        print(f"    - {task['topic']}")
        print(f"    Query: {task['search_query']}")

    return {
        "research_tasks": tasks,
    }
