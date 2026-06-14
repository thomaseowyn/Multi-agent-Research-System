from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    openai_api_key: str
    tavily_api_key: str

    planner_model: str = "gpt-4o-mini"
    researcher_model: str = "gpt-4o-mini"
    validator_model: str = "gpt-4o-mini"
    synthesizer_model: str = "gpt-4o"

    # configuration for behavior
    max_retries: int = 2
    max_search_results: int = 3         # how many search per task
    research_tasks_count: int = 3       # how many subtasks the planner generates

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
