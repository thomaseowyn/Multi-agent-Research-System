from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import ResearchTask, ResearchFinding
from models.schemas import RevisedQuery
from config import get_settings

settings = get_settings()


def revise_query(task: ResearchTask, finding: ResearchFinding, reason: str) -> ResearchTask:
    """
    Revise the search query for task that fails validation
    Need: reason, content, and previous search query
    """

    llm = ChatOpenAI(
        model=settings.planner_model,
        temperature=0,
        api_key=settings.openai_api_key
    )

    llm = llm.with_structured_output(RevisedQuery)

    system_prompt = """You are a search query optimization expert.

A previous research attempt failed validation. Your job is to write a
revised search query that is more likely to return useful, relevant results.

Common reasons searches fail and how to fix them:
- Too vague -> add specific terms (company name, year, "official", "report")
- Too narrow -> broaden terms or remove overly specific constraints
- Wrong angle -> approach the topic from a different framing
- Returned generic/marketing content -> add terms like "analysis", "comparison", "news"

Do not change the underlying topic — only improve HOW we search for it.
"""
    human_prompt = f"""Research topic: {task['topic']}
Previous search query: {task['search_query']}

This is the reason for failing the validation:
{reason}

These are the contents from the research finding:
{finding['content'][:500]}

Write a revised search query that addresses this specific failure.
"""
    messages = [
        HumanMessage(content=human_prompt),
        SystemMessage(content=system_prompt)
    ]

    result: RevisedQuery = llm.invoke(messages)

    # create new fixed research task
    revised_task = ResearchTask(
        task_id=task['task_id'],
        topic=task['topic'],
        search_query=result.revised_query,
        retry_count=task['retry_count'] + 1
    )

    return revised_task
