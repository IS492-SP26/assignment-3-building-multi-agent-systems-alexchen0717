"""
Output Guardrail
Checks system outputs for safety violations:
1. PII leakage (emails, phone numbers, SSNs, credit cards)
2. Harmful content in generated text
3. Bias indicators (overgeneralizations about groups)
"""

from typing import Dict, Any, List
import re


class OutputGuardrail:
    """Output safety validator. Redacts PII and flags unsafe content."""

    PII_PATTERNS = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b',
        "phone_us": r'\b\(?\d{3}\)?[-.\s]?\d{3}[-.\s]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b(?:\d[ -]*?){13,16}\b',
    }

    HARMFUL_OUTPUT_KEYWORDS = [
        "step-by-step bomb", "how to make a bomb",
        "synthesize methamphetamine", "synthesize ricin",
        "kill yourself", "commit suicide by",
    ]

    BIAS_PATTERNS = [
        r"\ball (?:women|men|asians|blacks|whites|jews|muslims|christians) are\b",
        r"\b(?:women|men) (?:can't|cannot|shouldn't) (?:do|be|have)\b",
    ]

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}

    def validate(self, response: str, sources: List[Dict[str, Any]] = None) -> Dict[str, Any]:
        violations: List[Dict[str, Any]] = []
        text = response or ""

        violations.extend(self._check_pii(text))
        violations.extend(self._check_harmful_content(text))
        violations.extend(self._check_bias(text))

        sanitized = self._sanitize(text, violations) if violations else text

        return {
            "valid": len(violations) == 0,
            "violations": violations,
            "sanitized_output": sanitized,
        }

    def _check_pii(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for pii_type, pattern in self.PII_PATTERNS.items():
            matches = re.findall(pattern, text)
            # Filter common false positives
            real_matches = [
                m for m in matches
                if not (pii_type == "email" and ("example.com" in m or "noreply" in m or "doi.org" in m))
            ]
            if real_matches:
                violations.append({
                    "validator": "pii",
                    "category": "pii_leakage",
                    "pii_type": pii_type,
                    "reason": f"Output contains {pii_type}",
                    "severity": "high",
                    "matches": real_matches,
                })
        return violations

    def _check_harmful_content(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        lower = text.lower()
        for kw in self.HARMFUL_OUTPUT_KEYWORDS:
            if kw in lower:
                violations.append({
                    "validator": "harmful_content",
                    "category": "harmful_content",
                    "reason": f"Output may contain harmful instruction: '{kw}'",
                    "severity": "high",
                })
        return violations

    def _check_bias(self, text: str) -> List[Dict[str, Any]]:
        violations = []
        for pattern in self.BIAS_PATTERNS:
            if re.search(pattern, text, re.IGNORECASE):
                violations.append({
                    "validator": "bias",
                    "category": "biased_content",
                    "reason": "Output contains overgeneralized statement about a group",
                    "severity": "medium",
                })
        return violations

    def _sanitize(self, text: str, violations: List[Dict[str, Any]]) -> str:
        sanitized = text
        for v in violations:
            if v.get("validator") == "pii":
                pii_type = v.get("pii_type", "info")
                for match in v.get("matches", []):
                    sanitized = sanitized.replace(match, f"[REDACTED-{pii_type.upper()}]")
        return sanitized