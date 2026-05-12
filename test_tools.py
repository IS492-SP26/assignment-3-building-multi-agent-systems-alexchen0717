"""Quick smoke test for web_search and paper_search tools."""
from dotenv import load_dotenv
load_dotenv()

from src.tools.web_search import web_search
from src.tools.paper_search import paper_search
from src.tools.citation_tool import CitationTool

print("=" * 60)
print("TEST 1: Web search via Tavily")
print("=" * 60)
result = web_search("explainable AI for novices HCI", max_results=3)
print(result)

print("=" * 60)
print("TEST 2: Academic paper search via Semantic Scholar")
print("=" * 60)
result = paper_search("explainable AI usability", max_results=3, year_from=2020)
print(result)

print("=" * 60)
print("TEST 3: Citation formatter (APA)")
print("=" * 60)
tool = CitationTool(style="apa")
sample = {
    "type": "paper",
    "title": "Designing Explainable AI for Non-Experts",
    "authors": [{"name": "Jane Smith"}, {"name": "John Doe"}],
    "year": 2023,
    "venue": "CHI 2023",
    "url": "https://example.com/paper",
}
tool.add_citation(sample)
print(tool.format_citation(sample))
print("Bibliography:", tool.generate_bibliography())

print("\n✅ All tool tests finished.")
