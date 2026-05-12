"""
AutoGen-Based Orchestrator.

Pre-fetches evidence using the async tool APIs (avoiding asyncio.run() nesting),
then injects the evidence into the RoundRobinGroupChat task message.
"""

import logging
import asyncio
import os
import re
from typing import Dict, Any, List

from dotenv import load_dotenv
load_dotenv()

from src.agents.autogen_agents import create_research_team
from src.tools.web_search import WebSearchTool
from src.tools.paper_search import PaperSearchTool


class AutoGenOrchestrator:
    """Coordinates the multi-agent research workflow."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.logger = logging.getLogger("autogen_orchestrator")
        self.logger.info("Creating research team...")
        self.team = create_research_team(config)
        self.logger.info("Research team created successfully")

    def process_query(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        self.logger.info(f"Processing query: {query}")
        try:
            try:
                asyncio.get_running_loop()
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
                    future = pool.submit(asyncio.run, self._process_query_async(query, max_rounds))
                    result = future.result()
            except RuntimeError:
                result = asyncio.run(self._process_query_async(query, max_rounds))
            self.logger.info("Query processing complete")
            return result
        except Exception as e:
            self.logger.error(f"Error processing query: {e}", exc_info=True)
            return {
                "query": query,
                "error": str(e),
                "response": f"An error occurred: {str(e)}",
                "conversation_history": [],
                "metadata": {"error": True},
            }

    async def _process_query_async(self, query: str, max_rounds: int = 20) -> Dict[str, Any]:
        self.logger.info(f"TAVILY_API_KEY present: {bool(os.getenv('TAVILY_API_KEY'))}")

        # ---- Pre-fetch evidence using ASYNC tool APIs (no asyncio.run nesting) ----
        web_tool = WebSearchTool(provider="tavily", max_results=5)
        paper_tool = PaperSearchTool(max_results=5)

        self.logger.info("Pre-fetching web search results...")
        try:
            web_results_list = await web_tool.search(query)
            web_results = self._format_web_results(query, web_results_list)
        except Exception as e:
            self.logger.error(f"Web search failed: {e}", exc_info=True)
            web_results = "No web results available."
            web_results_list = []

        self.logger.info("Pre-fetching academic papers...")
        try:
            paper_results_list = await paper_tool.search(query)
            paper_results = self._format_paper_results(query, paper_results_list)
        except Exception as e:
            self.logger.error(f"Paper search failed: {e}", exc_info=True)
            paper_results = "No academic papers available."
            paper_results_list = []

        self.logger.info(f"Got {len(web_results_list)} web results, {len(paper_results_list)} papers")

        # Truncate to keep prompt manageable for 8B model
        web_trunc = web_results[:3500]
        paper_trunc = paper_results[:3500]

        # ---- Task message - DO NOT include the termination string ----
        # The termination condition is now Critic-only-sourced, so even if "FINAL_ANSWER_READY"
        # were in the task it wouldn't trip - but we keep it out anyway as defense in depth.
        task_message = (
            f"Research Query: {query}\n\n"
            "Evidence has been gathered. Use ONLY this evidence; do not invent sources.\n\n"
            "=== WEB SEARCH EVIDENCE ===\n"
            f"{web_trunc}\n\n"
            "=== ACADEMIC PAPER EVIDENCE ===\n"
            f"{paper_trunc}\n\n"
            "=== WORKFLOW INSTRUCTIONS ===\n"
            "- Planner: produce a short plan (2-3 sub-questions). End with: PLAN COMPLETE.\n"
            "- Researcher: analyze the evidence, extract key findings, cite as [WebSrc N] or [Paper N]. End with: RESEARCH COMPLETE.\n"
            "- Writer: synthesize a clear response with inline citations and a References section. End with: DRAFT COMPLETE.\n"
            "- Critic: evaluate the response. If acceptable, end your message with the approval phrase the system expects. Otherwise list specific improvements and say NEEDS REVISION."
        )

        # ---- Run the team ----
        result = await self.team.run(task=task_message)

        # ---- Extract conversation history ----
        messages = []
        for message in result.messages:
            messages.append({
                "source": getattr(message, "source", "unknown"),
                "content": message.content if hasattr(message, "content") else str(message),
            })

        # ---- Pick final response: best Writer draft ----
        # Prefer the longest Writer message (real synthesis, not a rubber-stamp)
        writer_msgs = [m.get("content", "") for m in messages if m.get("source") == "Writer"]
        writer_msgs = [c for c in writer_msgs if c and len(c.strip()) > 100]
        if writer_msgs:
            # Longest writer message tends to be the actual draft
            final_response = max(writer_msgs, key=len)
        else:
            # Fallback: any reasonably long non-user message
            non_user = [m.get("content", "") for m in messages if m.get("source") not in ("user", "unknown")]
            non_user = [c for c in non_user if c and len(c.strip()) > 100]
            final_response = max(non_user, key=len) if non_user else (messages[-1].get("content", "") if messages else "")

        # Build sources list from what we actually fetched
        sources = []
        for i, r in enumerate(web_results_list, 1):
            sources.append({
                "ref": f"WebSrc {i}",
                "title": r.get("title", ""),
                "url": r.get("url", ""),
                "type": "webpage",
            })
        for i, p in enumerate(paper_results_list, 1):
            sources.append({
                "ref": f"Paper {i}",
                "title": p.get("title", ""),
                "url": p.get("url", ""),
                "authors": p.get("authors", []),
                "year": p.get("year"),
                "venue": p.get("venue", ""),
                "type": "paper",
            })

        return self._extract_results(query, messages, final_response, sources)

    def _format_web_results(self, query, results):
        if not results:
            return "No search results found."
        out = f"Found {len(results)} web search results for '{query}':\n\n"
        for i, r in enumerate(results, 1):
            out += f"[WebSrc {i}] {r.get('title', '')}\n"
            out += f"   URL: {r.get('url', '')}\n"
            snippet = (r.get('snippet') or '')[:400]
            out += f"   {snippet}\n\n"
        return out

    def _format_paper_results(self, query, results):
        if not results:
            return "No academic papers found."
        src = results[0].get("source", "unknown")
        out = f"Found {len(results)} academic papers for '{query}' (source: {src}):\n\n"
        for i, p in enumerate(results, 1):
            authors = ", ".join([a.get("name", "") for a in p.get("authors", [])[:3]])
            if len(p.get("authors", [])) > 3:
                authors += " et al."
            out += f"[Paper {i}] {p.get('title', '')}\n"
            out += f"   Authors: {authors}\n"
            out += f"   Year: {p.get('year')} | Venue: {p.get('venue', 'n/a')}\n"
            abstract = (p.get('abstract') or '')[:300]
            if abstract:
                out += f"   Abstract: {abstract}\n"
            out += f"   URL: {p.get('url', '')}\n\n"
        return out

    def _extract_results(self, query, messages, final_response, sources):
        plan = ""
        research_findings = []
        critique = ""
        for msg in messages:
            source = msg.get("source", "")
            content = msg.get("content", "")
            if source == "Planner" and not plan:
                plan = content
            elif source == "Researcher":
                research_findings.append(content)
            elif source == "Critic":
                critique = content

        # Strip control tokens and Qwen's <think> blocks
        if final_response:
            for tok in ("FINAL_ANSWER_READY", "DRAFT COMPLETE", "TERMINATE"):
                final_response = final_response.replace(tok, "").strip()
            final_response = re.sub(r"<think>.*?</think>", "", final_response, flags=re.DOTALL).strip()

        return {
            "query": query,
            "response": final_response,
            "conversation_history": messages,
            "sources": sources,
            "metadata": {
                "num_messages": len(messages),
                "num_sources": len(sources),
                "plan": plan,
                "research_findings": research_findings,
                "critique": critique,
                "agents_involved": list({m.get("source", "") for m in messages}),
            },
        }

    def get_agent_descriptions(self):
        return {
            "Planner": "Breaks down research queries into actionable steps",
            "Researcher": "Analyzes pre-fetched evidence from web + academic sources",
            "Writer": "Synthesizes findings into coherent responses",
            "Critic": "Evaluates quality and signals when the answer is ready",
        }

    def visualize_workflow(self):
        return """
1. Query
2. Orchestrator pre-fetches evidence (Tavily web + arXiv papers)
3. Planner -> 4. Researcher -> 5. Writer -> 6. Critic
7. Termination: Critic emits FINAL_ANSWER_READY (or 12-message safety cap)
"""


if __name__ == "__main__":
    import yaml
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    o = AutoGenOrchestrator(cfg)
    print(o.visualize_workflow())
