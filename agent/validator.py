from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import ResearchTask, ResearchFinding
from models.schemas import ValidationOutput
from config import get_settings

settings = get_settings()


def validate_finding(task: ResearchTask, finding: ResearchFinding) -> tuple[str, str]:
    """Judge whether the most recent finding is valid or invalid"""

    content = finding['content'].strip()

    # Fast check error, empty content, or too short
    if not content or content.startswith("Research failed:"):
        return "INVALID", "Content is empty or research execution failed."

    if len(content) < 50:
        return "INVALID", f"Content is too short ({len(content)} chars) to be substantive."

    llm = ChatOpenAI(
        model=settings.validator_model,
        temperature=0,
        api_key=settings.openai_api_key
    )

    # Make sure output based on the schema
    llm = llm.with_structured_output(ValidationOutput)

    system_prompt = """You are a quality control reviewer for a research pipeline.

You will be given a research TASK (what the researcher was asked to find)
and the CONTENT they produced.

Judge the content:
- VALID if it is relevant to the task, contains specific factual information,
  and would be useful in a competitive research report.
- INVALID if it is off-topic, vague, generic filler, or doesn't actually
  address the task.

Be reasonably lenient — partial information on-topic is still VALID.
Only mark INVALID if the content fails to address the task at all."""

    human_prompt = f"""Task topic: {task['topic']}
Search query: {task['search_query']}

Content:
{content}

Judge this content."""

    messages = [
        SystemMessage(content=system_prompt),
        HumanMessage(content=human_prompt)
    ]

    result: ValidationOutput = llm.invoke(messages)

    return result.verdict, result.reason
