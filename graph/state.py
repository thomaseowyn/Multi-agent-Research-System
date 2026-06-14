from typing import Annotated, TypedDict
from operator import add


class ResearchTask(TypedDict):
    """A single unit of research work. This is what the planner produces"""
    task_id: str
    topic: str      # general description of what to research
    search_query: str   # actual string sent to the search tool
    retry_count: int


class ResearchFinding(TypedDict):
    """The result from the researcher agent"""
    task_id: str
    topic: str
    content: str  # actual text extracted
    sources: list[str]
    is_valid: bool | None   # it needs to enter the validator agent first to decide
    validation_reason: str


class ValidationResult(TypedDict):
    """Output of Validator for the most recent finding"""
    is_valid: bool
    reason: str


class ResearchState(TypedDict):
    """The state for all nodes"""
    company: str
    research_tasks: list[ResearchTask]
    findings: Annotated[list[ResearchFinding], add]
    validated_findings: Annotated[list[ResearchFinding], add]
    final_report: str
    max_retries: int
