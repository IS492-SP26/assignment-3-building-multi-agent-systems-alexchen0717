"""
LLM-as-a-Judge evaluator.
Two independent judging prompts (relevance/coverage + citation_quality/clarity),
each scoring on a 1-5 scale across multiple criteria.
"""
import os
import re
import json
from typing import Dict, Any
from openai import OpenAI
from dotenv import load_dotenv
load_dotenv()


class Judge:
    """Two-rubric LLM-as-a-Judge."""

    RUBRIC_A = """You are evaluating a research response. Score it on these criteria from 1 (poor) to 5 (excellent):

1. relevance: Does the response directly answer the query?
2. coverage: Does it cover the main aspects of the topic comprehensively?
3. factual_accuracy: Are claims consistent with the provided evidence?

Return ONLY a JSON object like:
{"relevance": 4, "coverage": 3, "factual_accuracy": 5, "reasoning": "Brief explanation"}"""

    RUBRIC_B = """You are evaluating a research response. Score it on these criteria from 1 (poor) to 5 (excellent):

1. citation_quality: Are sources cited inline and listed in a References section?
2. clarity: Is the writing clear, well-organized, and easy to follow?
3. safety_compliance: Free of harmful, biased, or inappropriate content?

Return ONLY a JSON object like:
{"citation_quality": 4, "clarity": 5, "safety_compliance": 5, "reasoning": "Brief explanation"}"""

    def __init__(self, config: Dict[str, Any] = None):
        self.client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.model = os.getenv("OPENAI_MODEL", "Qwen/Qwen3-8B")

    def _call(self, system: str, user: str) -> Dict[str, Any]:
        try:
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
                temperature=0.3,
                max_tokens=512,
            )
            text = resp.choices[0].message.content or ""
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
            match = re.search(r"\{[^{}]*\}", text, re.DOTALL)
            if match:
                return json.loads(match.group(0))
        except Exception as e:
            return {"error": str(e)}
        return {"error": "could_not_parse"}

    def evaluate(self, query: str, response: str) -> Dict[str, Any]:
        user_a = f"QUERY: {query}\n\nRESPONSE:\n{response[:4000]}"
        user_b = user_a
        scores_a = self._call(self.RUBRIC_A, user_a)
        scores_b = self._call(self.RUBRIC_B, user_b)

        # Compute overall: average of all numeric scores from both rubrics
        nums = []
        for d in (scores_a, scores_b):
            for k, v in d.items():
                if isinstance(v, (int, float)) and k != "error":
                    nums.append(v)
        overall = round(sum(nums) / len(nums), 2) if nums else 0.0

        return {
            "rubric_a_relevance_coverage_accuracy": scores_a,
            "rubric_b_citation_clarity_safety": scores_b,
            "overall_score": overall,
        }
