"""
Safety Manager
Coordinates input + output guardrails and logs safety events.
"""

from typing import Dict, Any, List, Optional
import logging
from datetime import datetime
import json
import os

from src.guardrails.input_guardrail import InputGuardrail
from src.guardrails.output_guardrail import OutputGuardrail


class SafetyManager:
    """Top-level coordinator: input check -> agents -> output check -> log."""

    def __init__(self, config: Dict[str, Any]):
        self.config = config or {}
        self.enabled = self.config.get("enabled", True)
        self.log_events_flag = self.config.get("log_events", True)
        self.logger = logging.getLogger("safety")
        self.safety_events: List[Dict[str, Any]] = []

        self.prohibited_categories = self.config.get("prohibited_categories", [
            "harmful_content", "prompt_injection", "off_topic_queries",
            "pii_leakage", "biased_content",
        ])
        self.on_violation = self.config.get("on_violation", {
            "action": "refuse",
            "message": "I cannot process this request due to safety policies.",
        })

        self.input_guard = InputGuardrail(self.config)
        self.output_guard = OutputGuardrail(self.config)

        self.safety_log_file = self.config.get(
            "safety_log_file", "logs/safety_events.log"
        )
        os.makedirs(os.path.dirname(self.safety_log_file) or ".", exist_ok=True)

    def check_input_safety(self, query: str) -> Dict[str, Any]:
        if not self.enabled:
            return {"safe": True, "violations": [], "action": "allow", "query": query}

        result = self.input_guard.validate(query)
        violations = result.get("violations", [])
        has_high = any(v.get("severity") == "high" for v in violations)
        action = "refuse" if has_high else ("warn" if violations else "allow")
        is_safe = not has_high

        if violations and self.log_events_flag:
            self._log_safety_event("input", query, violations, is_safe)

        response_msg = None
        if action == "refuse":
            response_msg = self.on_violation.get(
                "message", "I cannot process this request due to safety policies."
            )

        return {
            "safe": is_safe,
            "violations": violations,
            "action": action,
            "query": query,
            "response": response_msg,
        }

    def check_output_safety(self, response: str, sources: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
        if not self.enabled:
            return {"safe": True, "violations": [], "action": "allow", "response": response}

        result = self.output_guard.validate(response, sources)
        violations = result.get("violations", [])
        sanitized = result.get("sanitized_output", response)
        is_safe = len(violations) == 0

        if any(v.get("validator") == "harmful_content" for v in violations):
            action = "refuse"
            final = self.on_violation.get(
                "message", "I cannot provide this response due to safety policies."
            )
        elif any(v.get("validator") == "pii" for v in violations):
            action = "sanitize"
            final = sanitized
        elif violations:
            action = "warn"
            final = sanitized
        else:
            action = "allow"
            final = response

        if violations and self.log_events_flag:
            self._log_safety_event("output", response, violations, is_safe)

        return {
            "safe": is_safe,
            "violations": violations,
            "action": action,
            "response": final,
        }

    def _log_safety_event(self, event_type: str, content: str, violations: List[Dict[str, Any]], is_safe: bool):
        event = {
            "timestamp": datetime.now().isoformat(),
            "type": event_type,
            "safe": is_safe,
            "violations": violations,
            "content_preview": (content[:200] + "...") if len(content) > 200 else content,
        }
        self.safety_events.append(event)
        cats = [v.get("category") for v in violations]
        self.logger.warning(f"Safety event ({event_type}): safe={is_safe}, violations={cats}")
        try:
            with open(self.safety_log_file, "a") as f:
                f.write(json.dumps(event) + "\n")
        except Exception as e:
            self.logger.error(f"Failed to write safety log: {e}")

    def get_safety_events(self) -> List[Dict[str, Any]]:
        return self.safety_events

    def get_safety_stats(self) -> Dict[str, Any]:
        total = len(self.safety_events)
        input_events = 0
        output_events = 0
        violations = 0
        for e in self.safety_events:
            if e["type"] == "input":
                input_events += 1
            elif e["type"] == "output":
                output_events += 1
            if not e["safe"]:
                violations += 1
        rate = (violations / total) if total > 0 else 0
        return {
            "total_events": total,
            "input_checks": input_events,
            "output_checks": output_events,
            "violations": violations,
            "violation_rate": rate,
        }

    def clear_events(self):
        self.safety_events = []
