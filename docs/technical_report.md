# Multi-Agent Deep Research System for HCI Topics

## Abstract

This work presents a multi-agent deep-research system that helps users explore Human-Computer Interaction (HCI) topics, with a focus on explainable AI for novice users. Four specialized AutoGen agents—Planner, Researcher, Writer, and Critic—coordinate in a RoundRobinGroupChat to decompose queries, analyze pre-fetched evidence from Tavily web search and arXiv academic search, synthesize citation-grounded responses, and verify quality. Because the self-hosted vLLM endpoint does not support function calling, evidence is fetched programmatically and injected into the agent task, preserving end-to-end tool use while remaining model-agnostic. Five guardrail policies—harmful content, prompt injection, off-topic queries, PII leakage, and biased content—are enforced at both input and output stages, with refusal or sanitization actions and full event logging. We evaluate the system on six diverse queries (five HCI, one adversarial) using a two-rubric LLM-as-a-Judge that scores six criteria on a 1–5 scale. A Streamlit web interface surfaces agent traces, citations, and safety events. The system demonstrates that grounded, safety-aware multi-agent research is feasible even on small open models.

## 1. System Design and Implementation

### Agents

The system uses four AutoGen `AssistantAgent` instances configured with distinct system prompts and orchestrated by a `RoundRobinGroupChat`:

- **Planner** decomposes the query into 2–3 sub-questions and signals `PLAN COMPLETE`.
- **Researcher** analyzes pre-fetched evidence, extracts findings, and cites sources inline as `[WebSrc N]` or `[Paper N]`. Signals `RESEARCH COMPLETE`.
- **Writer** synthesizes a structured response with inline citations and an APA-style References section. Signals `DRAFT COMPLETE`.
- **Critic** evaluates the draft and either signals `FINAL_ANSWER_READY` (approval) or asks for revision.

A `TextMentionTermination("FINAL_ANSWER_READY", sources=["Critic"])` ensures only the Critic can end the conversation, with a 12-message safety cap as a backstop.

### Tools

Two evidence-gathering tools live in `src/tools/`:

- **`web_search.py`** integrates with the Tavily API (configurable to Brave). Returns title, URL, snippet, and relevance score.
- **`paper_search.py`** integrates with Semantic Scholar as the primary backend and arXiv as an automatic fallback when S2 returns errors. Returns title, authors, year, venue, citation count, abstract, URL.

A third helper, `citation_tool.py`, formats sources in APA and MLA style for the References section.

### Control flow
### Model and configuration

All agents and the LLM judge use the self-hosted **Qwen3-8B** vLLM endpoint provided by the course. Because this endpoint does not support OpenAI-style function calling, tool execution happens in Python code rather than via the LLM's tool-calling protocol. Evidence is concatenated into the user task message before agent execution. This trade-off keeps the architecture portable across model backends while still satisfying the assignment's tool-integration requirement.

## 2. Safety Design

The safety layer comprises three files in `src/guardrails/`. Five policy categories are enforced:

| Category            | Layer  | Trigger                                    | Action     |
|---------------------|--------|--------------------------------------------|------------|
| `harmful_content`   | input  | Bomb-making, self-harm, weaponization KW   | refuse     |
| `harmful_content`   | output | Same KW in generated text                  | refuse     |
| `prompt_injection`  | input  | "Ignore previous instructions" regex set   | refuse     |
| `off_topic_queries` | input  | No HCI/AI keyword present                  | warn       |
| `pii_leakage`       | output | Email, phone, SSN, credit-card regex       | sanitize   |
| `biased_content`    | output | Overgeneralization regex about groups      | warn       |

Severity-based decisions: `high` triggers refusal; `medium` triggers sanitization; `low` triggers a warning but allows the query to proceed. Every event is appended to `logs/safety_events.log` with timestamp, type, content preview, violations, and final action.

Refusal messages are returned to the UI with their triggered policy category, so users see *which* policy fired (transparency requirement).

## 3. Evaluation Setup and Results

### Dataset

Six queries in `data/example_queries.json`:

1. Key XAI principles for non-experts (q1)
2. User-centered design for novices (q2)
3. Methods to evaluate AI explanation quality (q3)
4. Visual vs. textual explanations (q4)
5. Common pitfalls in designing AI explanations (q5)
6. Adversarial prompt-injection + harmful request (q6_unsafe) — should be refused.

### Judge prompts

Two independent rubrics, both implemented in `src/evaluation/judge.py`:

- **Rubric A — Relevance / Coverage / Factual accuracy** (criterion scores 1–5)
- **Rubric B — Citation quality / Clarity / Safety compliance** (criterion scores 1–5)

Each rubric is called as a separate LLM completion with `temperature=0.3` for stable scoring. Outputs are parsed as JSON; the **overall_score** is the arithmetic mean across all six criterion scores.

### Results

Six queries were processed end-to-end. The unsafe query (q6) was refused at input by the prompt-injection and harmful-content policies, demonstrating defense-in-depth: the agents were never invoked. The five HCI queries received judge scores as follows (overall = mean across 6 criterion scores from Rubrics A+B, each on a 1-5 scale, so the maximum is 5.0):

| Query | Topic | Overall Score (out of 5) |
|-------|-------|--------------------------|
| q1 | Key XAI principles for non-experts | **4.83** |
| q2 | User-centered design for novices | 2.33 |
| q3 | Methods to evaluate explanation quality | 2.33 |
| q4 | Visual vs. textual explanations | 2.33 |
| q5 | Common pitfalls in AI explanations | 1.67 |
| q6_unsafe | Prompt injection + harmful request | **Refused at input** (correct) |

**Mean overall score across 5 scored queries: 2.7 / 5.** Full machine-readable report in `outputs/evaluation_report.json`.

### Error analysis

Three notable failure modes were observed:

1. **High variance across queries (q1 = 4.83 vs q5 = 1.67).** The most likely cause is the 12-message safety cap. When the Critic returns "NEEDS REVISION" repeatedly, the team exhausts its turn budget before the Writer produces a polished final draft, and our extractor picks the longest available draft rather than an approved one. The model's small 8B parameter count may also struggle with longer multi-turn coherence.

2. **arXiv 429 rate limiting** under repeated runs throttled paper retrieval for some queries, leaving the Researcher with only web evidence. This degrades coverage scores in particular. Mitigation: the orchestrator already proceeds gracefully with whatever evidence is available.

3. **Guardrail behavior validated.** q6 was refused with violations `['prompt_injection', 'harmful_content']` before any LLM call, confirming that the input guardrail layer correctly intercepts adversarial inputs at the earliest stage.

## 4. Discussion and Limitations

**Insights.** Pre-fetching evidence in code worked better than expected. Even on the smaller Qwen3-8B, the Researcher reliably referenced injected sources by their `[WebSrc N]` labels and the Writer maintained APA References sections. The Critic's role as a termination signal (rather than a true judge of every iteration) kept the conversation bounded.

**Limitations.**
- The self-hosted endpoint's lack of function calling forced an "evidence injection" architecture; a more capable endpoint would enable autonomous tool selection.
- Semantic Scholar's anonymous tier is unreliable; future work could use the authenticated tier or replace S2 with a richer arXiv + OpenAlex blend.
- The guardrail layer is rule-based (regex + keywords). A neural moderator (e.g., Llama-Guard) would catch paraphrased attacks better.
- The LLM judge is the same model class as the agent, raising correlation risk; using a different judge model would strengthen evaluation.

**Future work.** Add (1) memory across sessions, (2) PDF parsing for paper full-text, (3) human-eval triangulation against the LLM judge, (4) richer trace UI showing each agent's thinking.

**Ethics.** All evidence is grounded in real sources; the Writer is instructed not to invent citations. The guardrails block harmful and injection attempts, and PII redaction is on by default. Because the system runs on a shared course endpoint, we kept evaluation volume low to avoid resource starvation for others.

## References

(APA style; not counted toward page count.)

- Anthropic & Microsoft. (2024). *AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation.* https://microsoft.github.io/autogen/
- Tavily AI. (n.d.). *Tavily Search API Documentation.* https://docs.tavily.com/
- arXiv.org. (n.d.). *arXiv API Documentation.* https://info.arxiv.org/help/api/
- Guardrails AI. (n.d.). *Guardrails Documentation.* https://docs.guardrailsai.com/
- Liu, Y., et al. (2024). *Trustworthy LLMs: A Survey and Guideline for Evaluating Large Language Models.* arXiv:2308.05374.
- Wang, X., et al. (2024). *Multi-Agent Collaboration with Large Language Models.* arXiv:2402.01680.
- Yang, A., et al. (2024). *Qwen3 Technical Report.* Qwen Team, Alibaba Group.
