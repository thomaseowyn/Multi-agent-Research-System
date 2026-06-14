from langchain_tavily import TavilySearch
from langchain_core.tools import tool
from bs4 import BeautifulSoup
import httpx


def get_search_tool(max_results: int = 5) -> TavilySearch:
    """Return a tavily search tool that accepts a search query string"""
    return TavilySearch(
        max_results=max_results,
        include_answer=False,  # dont include Tavily auto summariser
        include_raw_content=False,
        include_images=False,
    )


@tool
def fetch_url_content(url: str) -> str:
    """Fetch and clean webpage content"""

    try:
        # Make http request and return the a Response object
        response = httpx.get(
            url,
            timeout=10,
            follow_redirects=True,
            headers={
                "User-Agent": "Mozilla/5.0 (research-agent/1.0)"
            }
        )

        response.raise_for_status()

        # Create html parser
        parser = BeautifulSoup(
            response.text,      # return the html of response
            "html.parser"
        )

        # clean up the html by removing unnecessary stuff
        for tag in parser.find_all(["script", "style", "noscript"]):
            tag.dispose()

        # extract the text only
        text = parser.get_text(
            separator=" ",
            strip=True
        )

        return text[:4000]

    except httpx.TimeoutException:
        return "Request timed out"

    except httpx.HTTPStatusError as e:
        return f"HTTP error: {e.response.status_code}"

    except Exception as e:
        return f"Unexpected error: {e}"
