from pydantic import BaseModel, Field
from typing import Literal


class ResearchTaskOutput(BaseModel):
    """
    A single research task as produced by the Planner.
    Use pydantic to validate output 
    """
    topic: str = Field(
        description="Human-readable topic to research, e.g. 'Main competitors of OpenAI'"
    )
    search_query: str = Field(
        description="Optimized search query to find information on this topic"
    )


class ResearchPlan(BaseModel):
    """Complete research plan from Planner"""
    company: str = Field(description="The company being researched")
    tasks: list[ResearchTaskOutput] = Field(
        description="List of specific research tasks to execute")
    reasoning: str = Field(
        description="Brief explanation of why these tasks cover the research goal")


class ValidationOutput(BaseModel):
    """Structured output from Validator"""
    verdict: Literal["VALID", "INVALID"] = Field(
        description="VALID if the finding is relevant, non-empty, and useful. "
                    "INVALID if it's empty, off topic and error message"
    )
    reason: str = Field(
        description="One or two sentences explaining the verdict"
    )


class RevisedQuery(BaseModel):
    """Structured output from Retry Planner"""
    revised_query: str = Field(
        description="A new, more specific research query likely to return better result than previous attempt"
    )
    strategy_note: str = Field(
        description="Brief note on what changed and why (for logging and debugging)"
    )
