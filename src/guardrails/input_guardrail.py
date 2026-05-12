"""
Input Guardrail
Detects unsafe user inputs across three policy categories:
1. Harmful content (violence, illegal activity, self-harm)
2. Prompt injection attempts (role override, system prompt leak attempts)
3. Off-topic queries (outside the configured HCI research scope)
"""

from typing import Dict, Any, List
import re


class InputGuardrail:
    """Input safety validator with keyword + regex policy checks."""

    # --- Policy 1: Harmful content keywords ---
    HARMFUL_KEYWORDS = [
        "make a bomb", "build a bomb", "how to kill", "how to murder",
        "self harm", "self-harm", "kill myself", "suicide method",
        "child sexual", "csam", "child porn",
        "synthesize methamphetamine", "cook meth",
        "buy fentanyl", "obtain ricin", "weaponize anthrax",
    ]

    # --- Policy 2: Prompt injection patterns ---
    INJECTION_PATTERNS = [
        r"ignore (?:all |the )?(?:previous|above|prior) (?:instructions|prompts?|rules?)",
        r"disregard (?:all |the )?(?:previous|above|prior) (?:instructions|prompts?|rules?)",
        r"forget (?:everything|all instructions|your instructions)",
        r"you are now (?:a |an )?(?:different|new|jailbroken)",
        r"system\s*:",
        r"reveal (?:your |the )?(?:system )?prompt",
        r"print (?:your |the )?(?:system )?prompt",
        r"<\|im_start\|>",
        r"###\s*instruction",
        r"act as (?:a |an )?(?:dan|developer mode|unrestricted)",
    ]

    # --- Policy 3: On-topic HCI keywords (require at least one to be on-topic) ---
    HCI_KEYWORDS = [
        "hci", "human-computer", "human computer", "ui", "ux", "user interface",
        "user experience", "usability", "accessibility", "interaction design",
        "explainable", "xai", "interpretab", "explanation", "interpret",
        "ai", "artificial intelligence", "machine learning", "ml", "llm",
        "design", "user", "interface", "novice", "expert", "cognitive",
        "research", "study", "evaluation", "prototype", "agent", "chatbot",
        "ar", "vr", "mixed reality", "augmented", "virtual", "wearable",
        "ethics", "trust", "transparency", "fairness", "bias", "education",
    ]

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        safety_cfg = self.config.get("safety", self.config)
        self.min_length = safety_cfg.get("min_query_length", 5)
        self.max_length = safety_cfg.get("max_query_length", 2000)
        self.check_relevance = safety_cfg.get("check_relevance", True)

    def validate(self, query: str) -> Dict[str, Any]:
        """Run all input checks and return a structured result."""
        violations: List[Dict[str, Any]] = []
        text = (query or "").strip()
        lower = text.lower()

        # Length checks
        if len(text) < self.min_length:
            violations.append({
                "validator": "length", "category": "format",
                "reason": "Query too short (min %d chars)" % self.min_length,
                "severity": "low",
            })
        if len(text) > self.max_length:
            violations.append({
                "validator": "length", "category": "format",
                "reason": "Query too long (max %d chars)" % self.max_length,
                "severity": "medium",
            })

        # Policy checks
        violations.extend(self._check_harmful_content(lower))
        violations.extend(self._check_prompt_injection(lower))
        if self.check_relevance:
            violations.extend(self._check_relevance(lower))

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "sanitized_input": text,
        }

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for kw in self.HARMFUL_KEYWORDS:
            if kw in text:
                violations.append({
                    "validator": "harmful_content",
                    "category": "harmful_content",
                    "reason": f"Matched harmful keyword: '{kw}'",
                    "severity": "high",
                })
        return violations

    def _check_prompt_injection(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for pattern in self.INJECTION_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append({
                    "validator": "prompt_injection",
                    "category": "prompt_injection",
                    "reason": f"Matched injection pattern: '{pattern}'",
                    "severity": "high",
                })
        return violations

    def _check_relevance(self, text: str) -> List[Dict[str, Any]]:
        """Off-topic if NO HCI-related keyword is present."""
        if any(kw in text for kw in self.HCI_KEYWORDS):
            return []
        if len(text.split()) < 3:
            return []  # too short to judge, length check already handles
        return [{
            "validator": "relevance",
            "category": "off_topic_queries",
            "reason": "Query appears off-topic (no HCI/AI terms detected)",
            "severity": "low",
        }]