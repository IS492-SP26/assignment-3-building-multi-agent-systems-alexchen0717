"""
Paper Search Tool
Integrates with Semantic Scholar API for academic paper search,
with arXiv as a fallback when Semantic Scholar is unavailable.
"""

from typing import List, Dict, Any, Optional
import os
import logging
import asyncio
import urllib.parse
import xml.etree.ElementTree as ET


class PaperSearchTool:
    """
    Tool for searching academic papers.

    Primary backend: Semantic Scholar (requires optional API key for reliability).
    Fallback backend: arXiv (free, no key, public API).

    The tool tries Semantic Scholar first; on any failure (403, timeout,
    network error, missing key) it transparently falls back to arXiv so
    the rest of the system always gets paper results.
    """

    def __init__(self, max_results: int = 10):
        self.max_results = max_results
        self.logger = logging.getLogger("tools.paper_search")
        self.api_key = os.getenv("SEMANTIC_SCHOLAR_API_KEY")

        if not self.api_key:
            self.logger.info(
                "No Semantic Scholar API key found. Will try anonymous S2, "
                "fall back to arXiv on failure."
            )

    async def search(
        self,
        query: str,
        year_from: Optional[int] = None,
        year_to: Optional[int] = None,
        min_citations: int = 0,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        """
        Search for academic papers. Tries Semantic Scholar first,
        falls back to arXiv if S2 returns nothing or errors out.
        """
        self.logger.info(f"Searching papers: {query}")

        # --- Try Semantic Scholar first ---
        papers = await self._search_semantic_scholar(
            query, year_from, year_to, min_citations, **kwargs
        )

        if papers:
            self.logger.info(f"Semantic Scholar returned {len(papers)} papers")
            return papers

        # --- Fall back to arXiv ---
        self.logger.warning(
            "Semantic Scholar returned no results / errored. Falling back to arXiv."
        )
        papers = await self._search_arxiv(query, year_from, year_to)
        self.logger.info(f"arXiv returned {len(papers)} papers")
        return papers

    # ---------- Semantic Scholar backend ----------

    async def _search_semantic_scholar(
        self,
        query: str,
        year_from: Optional[int],
        year_to: Optional[int],
        min_citations: int,
        **kwargs,
    ) -> List[Dict[str, Any]]:
        try:
            from semanticscholar import SemanticScholar

            sch = SemanticScholar(api_key=self.api_key)
            fields = kwargs.get(
                "fields",
                [
                    "paperId", "title", "authors", "year", "abstract",
                    "citationCount", "url", "venue", "openAccessPdf",
                ],
            )
            results = sch.search_paper(query, limit=self.max_results, fields=fields)
            return self._parse_s2_results(results, year_from, year_to, min_citations)
        except Exception as e:
            self.logger.warning(f"Semantic Scholar unavailable: {e}")
            return []

    def _parse_s2_results(
        self,
        results: Any,
        year_from: Optional[int],
        year_to: Optional[int],
        min_citations: int,
    ) -> List[Dict[str, Any]]:
        papers = []
        try:
            for paper in results:
                if not paper or not hasattr(paper, "title"):
                    continue
                papers.append({
                    "paper_id": getattr(paper, "paperId", None),
                    "title": getattr(paper, "title", "Unknown"),
                    "authors": [{"name": a.name} for a in (getattr(paper, "authors", []) or [])],
                    "year": getattr(paper, "year", None),
                    "abstract": getattr(paper, "abstract", "") or "",
                    "citation_count": getattr(paper, "citationCount", 0) or 0,
                    "url": getattr(paper, "url", "") or "",
                    "venue": getattr(paper, "venue", "") or "",
                    "pdf_url": (paper.openAccessPdf.get("url")
                                if getattr(paper, "openAccessPdf", None) else None),
                    "source": "semantic_scholar",
                })
        except Exception as e:
            self.logger.warning(f"Error parsing S2 results: {e}")
            return []

        if year_from:
            papers = [p for p in papers if p.get("year") and p["year"] >= year_from]
        if year_to:
            papers = [p for p in papers if p.get("year") and p["year"] <= year_to]
        papers = [p for p in papers if p.get("citation_count", 0) >= min_citations]
        return papers

    # ---------- arXiv backend ----------

    async def _search_arxiv(
        self,
        query: str,
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> List[Dict[str, Any]]:
        """Public arXiv API — no key required."""
        try:
            import aiohttp

            base_url = "http://export.arxiv.org/api/query"
            params = {
                "search_query": f"all:{query}",
                "start": 0,
                "max_results": self.max_results,
                "sortBy": "relevance",
                "sortOrder": "descending",
            }
            url = f"{base_url}?{urllib.parse.urlencode(params)}"

            async with aiohttp.ClientSession() as session:
                async with session.get(url, timeout=20) as response:
                    if response.status != 200:
                        self.logger.error(f"arXiv API error: HTTP {response.status}")
                        return []
                    text = await response.text()

            return self._parse_arxiv_results(text, year_from, year_to)
        except Exception as e:
            self.logger.error(f"arXiv search error: {e}")
            return []

    def _parse_arxiv_results(
        self,
        xml_text: str,
        year_from: Optional[int],
        year_to: Optional[int],
    ) -> List[Dict[str, Any]]:
        ns = {"atom": "http://www.w3.org/2005/Atom"}
        papers = []
        try:
            root = ET.fromstring(xml_text)
            for entry in root.findall("atom:entry", ns):
                title_el = entry.find("atom:title", ns)
                summary_el = entry.find("atom:summary", ns)
                published_el = entry.find("atom:published", ns)
                id_el = entry.find("atom:id", ns)

                title = (title_el.text or "").strip() if title_el is not None else "Unknown"
                abstract = (summary_el.text or "").strip() if summary_el is not None else ""
                published = (published_el.text or "")[:4] if published_el is not None else None
                year = int(published) if published and published.isdigit() else None
                url = (id_el.text or "").strip() if id_el is not None else ""
                paper_id = url.split("/")[-1] if url else None

                authors = []
                for a in entry.findall("atom:author", ns):
                    name_el = a.find("atom:name", ns)
                    if name_el is not None and name_el.text:
                        authors.append({"name": name_el.text.strip()})

                papers.append({
                    "paper_id": paper_id,
                    "title": title.replace("\n", " "),
                    "authors": authors,
                    "year": year,
                    "abstract": abstract.replace("\n", " "),
                    "citation_count": 0,  # arXiv doesn't expose citation counts
                    "url": url,
                    "venue": "arXiv (preprint)",
                    "pdf_url": url.replace("/abs/", "/pdf/") if url else None,
                    "source": "arxiv",
                })
        except Exception as e:
            self.logger.error(f"Error parsing arXiv XML: {e}")
            return []

        if year_from:
            papers = [p for p in papers if p.get("year") and p["year"] >= year_from]
        if year_to:
            papers = [p for p in papers if p.get("year") and p["year"] <= year_to]
        return papers

    # ---------- Optional methods kept for compatibility ----------

    async def get_paper_details(self, paper_id: str) -> Dict[str, Any]:
        try:
            from semanticscholar import SemanticScholar
            sch = SemanticScholar(api_key=self.api_key)
            paper = sch.get_paper(paper_id)
            return {
                "paper_id": paper.paperId,
                "title": paper.title,
                "authors": [{"name": a.name} for a in paper.authors] if paper.authors else [],
                "year": paper.year,
                "abstract": paper.abstract,
                "citation_count": paper.citationCount,
                "url": paper.url,
                "venue": paper.venue,
                "pdf_url": paper.openAccessPdf.get("url") if paper.openAccessPdf else None,
            }
        except Exception as e:
            self.logger.error(f"Error getting paper details: {e}")
            return {}


# Synchronous wrapper for use with AutoGen tools
def paper_search(query: str, max_results: int = 10, year_from: Optional[int] = None) -> str:
    """Synchronous wrapper for paper search (AutoGen tool integration)."""
    tool = PaperSearchTool(max_results=max_results)
    results = asyncio.run(tool.search(query, year_from=year_from))

    if not results:
        return "No academic papers found."

    source_used = results[0].get("source", "unknown")
    output = f"Found {len(results)} academic papers for '{query}' (source: {source_used}):\n\n"

    for i, paper in enumerate(results, 1):
        authors = ", ".join([a["name"] for a in paper["authors"][:3]])
        if len(paper["authors"]) > 3:
            authors += " et al."
        output += f"{i}. {paper['title']}\n"
        output += f"   Authors: {authors}\n"
        output += f"   Year: {paper['year']} | Citations: {paper['citation_count']}"
        if paper.get("venue"):
            output += f" | Venue: {paper['venue']}"
        output += "\n"
        if paper.get("abstract"):
            abstract = paper["abstract"][:200] + "..." if len(paper["abstract"]) > 200 else paper["abstract"]
            output += f"   Abstract: {abstract}\n"
        output += f"   URL: {paper['url']}\n\n"

    return output