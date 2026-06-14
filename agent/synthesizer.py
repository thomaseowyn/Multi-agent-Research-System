from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

from graph.state import ResearchState, ResearchFinding
from config import get_settings

settings = get_settings()


def synthesizer_agent(state: ResearchState) -> ResearchState:
    """Combine all findings into research report"""

    llm = ChatOpenAI(
        model=settings.synthesizer_model,
        temperature=0.2,
        api_key=settings.openai_api_key
    )

    # filter validated findings
    findings: list[ResearchFinding] = state['validated_findings']

    # convert typeddict findings into readable text for llm
    findings_text = ""
    all_sources = []

    for i, finding in enumerate(findings):
        findings_text += f"\n### Finding {i}: {finding['topic']}\n"
        findings_text += f"{finding['content']}\n"
        if finding['sources']:
            findings_text += f"Sources: {", ".join(finding['sources'])}\n"
            all_sources.extend(finding['sources'])

    unique_sources = list(set(all_sources))

    system_prompt = """You are a senior business analyst writing a professional research report.

Your job is to synthesize multiple research findings into a single, coherent, 
well-structured report. 

Report Requirements:
1. Use clear markdown headers for each section
2. Remove redundancy — if multiple findings cover the same fact, mention it once
3. Be specific — include names, numbers, dates from the findings
4. Preserve all important information — don't over-summarize
5. End with a complete Sources section listing all URLs

Report Structure:
# [Company Name] Research Report

## Executive Summary
## Company Overview
## Products and Services
## Competitive Landscape
## Market Position
## Recent Developments
## Strengths and Weaknesses
## Strategic Opportunities
## Sources"""

    human_prompt = f"""Create a comprehensive research report for: {state["company"]}

Based on the following research findings:

{findings_text}

All source URLs to include in the Sources section:
{chr(10).join(unique_sources)}

Write the complete research report now."""

    messages = [
        HumanMessage(content=human_prompt),
        SystemMessage(content=system_prompt)
    ]

    response = llm.invoke(messages)
    report = response.content

    print(f"\n[Synthesizer] Report generated ({len(report)}) characters")
    print(f"[Synthesizer] Sources cited: {len(unique_sources)}")

    return {"final_report": report}
