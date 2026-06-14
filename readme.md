# Multi-Agent Competitive Research System

A production-style multi-agent system built with **LangGraph** and **LangChain** that automates competitive business research. Given a company name (e.g., `"OpenAI"`), the system autonomously plans research subtasks, gathers information from the web in parallel, validates the quality of what it finds, retries low-quality results with improved search queries, and synthesizes everything into a structured markdown report with citations.

```
Input:  "Research OpenAI"

Output: A markdown report covering:
  - Company Overview
  - Products & Services
  - Competitive Landscape
  - Market Position
  - Recent Developments
  - Strengths & Weaknesses
  - Strategic Opportunities
  - Sources
```

---

## Why This Project

A single LLM call asked to "research a company" tends to hallucinate, miss key facts, and produce a flat, unverifiable wall of text. This project explores a different approach: decompose the problem across **specialized agents**, each with a narrow responsibility, coordinated through a **shared state graph**. Quality control (validation + retry) is built into the pipeline itself, so low-quality information is structurally prevented from reaching the final report — not just discouraged via prompting.

The project was built incrementally, phase by phase, with each phase introducing one new architectural concept. The goal was not just to produce a working tool, but to deeply understand *why* each piece of the architecture exists.

---

## Architecture Overview (Current: Phase 4)

```
                    ┌──> Send(research_task, {task: T1}) ──┐
                    │                                        │
START -> planner ───┼──> Send(research_task, {task: T2}) ──┼──> synthesizer -> END
                    │                                        │
                    └──> Send(research_task, {task: TN}) ──┘

planner        : decomposes "Research <Company>" into N specific subtasks
research_task  : runs N times IN PARALLEL — each instance:
                    1. searches the web (Tavily) for its assigned subtask
                    2. validates its own findings (relevant? substantive?)
                    3. if invalid, rewrites the search query and retries
                       (up to max_retries times)
synthesizer     : waits for ALL parallel branches to finish, then combines
                  every validated finding into one cohesive report
```

### Agents

| Agent | Role | Model |
|---|---|---|
| **Planner** | Breaks "Research X" into N specific subtasks + optimized search queries | GPT-4o |
| **Researcher** | ReAct agent using Tavily search + URL fetching to gather facts and sources | GPT-4o-mini |
| **Validator** | Judges whether a finding is relevant, substantive, and useful (VALID/INVALID) | GPT-4o-mini |
| **Retry Planner** | Rewrites a failed search query based on *why* it failed | GPT-4o |
| **Synthesizer** | Combines all validated findings into a structured, cited markdown report | GPT-4o |

### Tech Stack

- **Python 3.12**
- **LangGraph** — graph orchestration (`StateGraph`, conditional edges, `Send()`)
- **LangChain** — agent framework, tool integration, structured output parsing
- **OpenAI API** — GPT-4o / GPT-4o-mini
- **Tavily Search API** — LLM-oriented web search
- **Pydantic** — structured/validated LLM outputs
- **httpx** — direct URL content fetching

---

## How It Works, Step by Step

1. **Planning**: The user provides a company name. The Planner agent uses a structured output schema (`ResearchPlan`, a Pydantic model) to produce a fixed number of `ResearchTask` objects — each with a topic and an optimized search query.

2. **Parallel Fan-Out**: Instead of researching tasks one at a time, LangGraph's `Send()` primitive dispatches **all tasks simultaneously** to parallel instances of a single `research_task` node. The number of parallel branches is determined *at runtime* (it depends on how many tasks the Planner generated).

3. **Research + Self-Validation + Retry (per branch)**: Each parallel branch:
   - Runs a ReAct agent (search → observe → reason → search again if needed → final answer) to gather information and source URLs.
   - Validates its own output: cheap deterministic checks first (empty content? error message? too short?), then an LLM judgment call (relevant? substantive? useful?).
   - If invalid and retries remain, an LLM rewrites the search query **based on the validator's stated reason for failure**, increments a retry counter, and tries again.
   - This loop terminates because `retry_count` strictly increases every attempt — guaranteeing the branch eventually exits (success or "give up").

4. **Fan-In + Synthesis**: LangGraph automatically waits for *all* parallel branches to complete (no explicit "wait" logic needed — it's inherent to how LangGraph's superstep execution model resolves dependent nodes). The Synthesizer then receives only the **validated** findings and writes the final report, organizing content into sections and compiling a deduplicated source list.

---

## Key Design Decisions & Tradeoffs

This section is the part I think matters most for understanding the project — not just *what* it does, but *why* it's built this way.

### 1. State accumulates via reducers (`operator.add`), not overwrites

`findings` and `validated_findings` are typed as `Annotated[list[...], operator.add]`. Every node that contributes a finding returns `{"findings": [one_finding]}` — a list with exactly one item. LangGraph's reducer *appends* this to whatever's already in state, rather than replacing it.

This single design choice is what makes the system trivially extensible from "process tasks sequentially in a loop" (early phases) to "process tasks in parallel via `Send()`" (current phase) **without changing any node's return statement**. Sequential loop iterations and concurrent parallel branches both just "append one item" — the reducer doesn't know or care which is happening.

### 2. Cheap checks before expensive LLM calls

The Validator runs deterministic Python checks (is the content empty? does it start with an error message? is it under 50 characters?) *before* making any LLM call. Only content that passes these free, instant checks gets sent to an LLM for a relevance/quality judgment. This is a real production cost-optimization pattern: never pay for an LLM call to answer a question plain code can answer.

### 3. Retry logic moved from graph edges to a Python loop

Earlier versions of this project implemented retries as **graph-level conditional edges** — a 3-way branch (`retry | continue | done`) with a dedicated `retry_planner` node that the graph would loop back to. This worked well for *sequential* processing, where there's a single "current task" the whole graph shares.

Once research became parallel (`Send()`-based), there's no single "current task" — there are N of them simultaneously. Retrying-as-graph-edges would require either (a) a separate retry sub-loop *per parallel branch*, which LangGraph supports via subgraphs but adds significant complexity, or (b) what was actually implemented: each parallel branch is a single node that runs a bounded `while` loop internally (research → validate → maybe revise query → repeat), using the *same underlying logic* (`validate_finding()`, `revise_query()`) as plain, state-independent functions.

**Tradeoff accepted**: individual retry attempts are no longer visible as separate steps in LangGraph's execution trace — from the graph's perspective, a branch that retried twice looks identical to one that succeeded immediately. The upgrade path (subgraphs per branch) is a refactor, not a rewrite, since the core decision logic is already extracted as standalone functions.

### 4. The retry loop's termination guarantee

Every retry produces a *new* `ResearchTask` with the same `task_id` and `topic` (it's the same logical task, just a better query) but `retry_count + 1`. The loop's exit condition is `retry_count >= max_retries`. Because this increment happens unconditionally on every retry, the loop is **guaranteed to terminate** — there's no code path where a task can retry forever. This single line (`retry_count: task["retry_count"] + 1`) is, in a real sense, the load-bearing component of the entire retry mechanism.

### 5. Different models for different agents

| Task | Model | Reasoning |
|---|---|---|
| Planning, query revision, synthesis | GPT-4o | Requires deeper reasoning/strategy |
| Research execution, validation | GPT-4o-mini | High-volume, narrower judgment calls — 10x cheaper |

Matching model capability to task difficulty (rather than using one model everywhere) is a basic but important production cost lever.

---

## Project Structure

```
research_agent/
├── agents/
│   ├── planner.py        # Decomposes goal into research tasks (structured output)
│   ├── researcher.py      # ReAct agent: search + fetch + summarize
│   ├── validator.py        # validate_finding(): VALID/INVALID + reason
│   ├── retry_planner.py     # revise_query(): rewrites failed search queries
│   ├── research_task.py     # Combined node: research + validate + retry loop
│   └── synthesizer.py        # Combines validated findings into final report
│
├── graph/
│   ├── state.py            # ResearchState, ResearchTask, ResearchFinding (TypedDicts)
│   ├── edges.py             # fan_out_to_researchers() -> list[Send]
│   └── builder.py           # StateGraph construction and compilation
│
├── tools/
│   └── search.py            # Tavily search tool + URL content fetcher
│
├── models/
│   └── schemas.py            # Pydantic schemas for structured LLM outputs
│
├── config.py                  # Centralized settings (models, API keys, retry limits)
├── main.py                     # Entry point
└── requirements.txt
```

---

## Running It

```bash
pip install -r requirements.txt
```

Create a `.env` file:
```
OPENAI_API_KEY=sk-...
TAVILY_API_KEY=tvly-...
```

Run:
```bash
python main.py
```

This researches a hardcoded company (edit `main.py` to change it) and writes the result to `report.md`.

---

## What I Learned

- **State design determines what's possible.** Deciding early that `findings` would be a reducer-based accumulating list (rather than a single overwritten field) is what made the later jump to parallelism nearly free — the node code didn't need to change at all.
- **Conditional edges vs. plain control flow is a real architectural choice, not just syntax.** Graph-level branching gives you visibility/tracing per decision; inlined Python loops are simpler but opaque to the orchestration layer. Neither is "more correct" — it depends on whether you need per-step observability.
- **Quality gates work best when they're structural, not just prompted.** The Synthesizer never *sees* invalid findings — they're filtered out before it runs. This is more robust than instructing the Synthesizer to "ignore bad information," which relies on the LLM following that instruction perfectly every time.
- **Off-by-one errors in retry/loop bounds are easy to introduce and important to document.** `max_retries=2` meaning "3 total attempts" is a small detail that, if inconsistent between the loop condition and the documentation, causes confusing bugs.

---

## Possible Future Extensions

- **Source-level citation tracking**: tie specific claims in the final report to specific source URLs with inline numbered references, rather than a flat "Sources" list at the end.
- **Persistent memory via a vector database**: store findings from past research runs so future queries about the same company (or related companies) can reuse prior research rather than starting from scratch.
- **Subgraph-based retries**: refactor each parallel branch into a compiled subgraph (with its own validator/retry-planner nodes and conditional edges) for full per-retry observability in LangGraph traces — a natural next step since the core validation/revision logic is already extracted as standalone, reusable functions.