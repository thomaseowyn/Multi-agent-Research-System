from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
from langchain.agents import create_agent

from graph.state import ResearchTask, ResearchFinding
from tools.search import get_search_tool, fetch_url_content
from config import get_settings

settings = get_settings()


def _run_single_research_task(task: ResearchTask) -> ResearchFinding:
    """
    Execute a single research task using langgraph react agent
    """
    llm = ChatOpenAI(
        model=settings.researcher_model,
        temperature=0,
        api_key=settings.openai_api_key
    )

    tools = [get_search_tool(
        max_results=settings.max_search_results), fetch_url_content]

    agent = create_agent(
        model=llm,
        tools=tools,
    )

    research_prompt = f"""
You are a professional business researcher.

Research task: {task['topic']}
Search query to start with: {task['search_query']}

Instructions:
1. Use search tool with the provided query
2. If information is highly relevant, use fetch_url_content
3. Collect factual and specific information
4. Keep track of source URLs

After researching, provide:
- A detailed summary of findings (at least 200 words)
- All source URLs used

Be specific. Include facts, figures, dates, and names where available.
"""
    try:
        result = agent.invoke(
            {
                "messages": [
                    HumanMessage(content=research_prompt)
                ]
            }
        )

        messages = result["messages"]

        # last AI response
        final_response = messages[-1].content

        # grab URLs from the ToolMessages
        sources = []
        for message in messages:
            if getattr(message, "type", None) == "tool":
                content = str(message.content)

                import re

                urls = re.findall(
                    r"https?://[^\s]+",
                    content
                )

                sources.extend(urls)

        return ResearchFinding(
            task_id=task["task_id"],
            topic=task["topic"],
            content=final_response,
            sources=list(set(sources)),
            is_valid=None,
            validation_reason=""
        )
    except Exception as e:
        return ResearchFinding(
            task_id=task["task_id"],
            topic=task["topic"],
            content=f"Research failed: {e}",
            sources=[],
            is_valid=None,
            validation_reason=""
        )
