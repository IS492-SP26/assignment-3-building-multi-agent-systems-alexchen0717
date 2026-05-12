"""Batch evaluator: runs N queries through the orchestrator + judge."""
import json
import os
from typing import Dict, Any, List
from datetime import datetime

from src.autogen_orchestrator import AutoGenOrchestrator
from src.guardrails.safety_manager import SafetyManager
from src.evaluation.judge import Judge


class SystemEvaluator:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.orchestrator = AutoGenOrchestrator(config)
        self.safety = SafetyManager(config.get("safety", {}))
        self.judge = Judge(config)

    def run_full_evaluation(self, queries_path: str = "data/example_queries.json") -> Dict[str, Any]:
        with open(queries_path) as f:
            data = json.load(f)

        results = []
        for q in data["queries"]:
            qid = q["id"]
            query = q["query"]
            should_refuse = q.get("should_refuse", False)
            print(f"\n=== {qid}: {query[:80]} ===")

            # Input safety check
            sin = self.safety.check_input_safety(query)
            if not sin.get("safe"):
                print(f"  REFUSED at input. Categories: {[v.get('category') for v in sin['violations']]}")
                results.append({
                    "id": qid, "query": query, "response": sin.get("response", ""),
                    "refused_at": "input", "violations": sin["violations"],
                    "scores": {"overall_score": "N/A (refused)"},
                    "guardrail_correct": should_refuse,
                })
                continue

            # Run orchestrator
            try:
                result = self.orchestrator.process_query(query)
                response = result.get("response", "")
            except Exception as e:
                print(f"  ERROR: {e}")
                results.append({"id": qid, "query": query, "error": str(e)})
                continue

            # Output safety
            sout = self.safety.check_output_safety(response)
            final_resp = sout["response"]

            # Judge
            scores = self.judge.evaluate(query, final_resp)
            print(f"  Overall: {scores.get('overall_score')}")

            results.append({
                "id": qid,
                "query": query,
                "response": final_resp[:2000],
                "scores": scores,
                "num_sources": result.get("metadata", {}).get("num_sources", 0),
                "agents_involved": result.get("metadata", {}).get("agents_involved", []),
                "output_action": sout.get("action"),
                "guardrail_correct": should_refuse == False,
            })

        report = {
            "timestamp": datetime.now().isoformat(),
            "num_queries": len(results),
            "results": results,
            "aggregate": self._aggregate(results),
        }

        os.makedirs("outputs", exist_ok=True)
        out_path = "outputs/evaluation_report.json"
        with open(out_path, "w") as f:
            json.dump(report, f, indent=2)
        print(f"\nReport saved to {out_path}")
        return report

    def _aggregate(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        scores = [r["scores"]["overall_score"] for r in results
                  if isinstance(r.get("scores", {}).get("overall_score"), (int, float))]
        return {
            "mean_overall_score": round(sum(scores) / len(scores), 2) if scores else 0,
            "num_scored": len(scores),
            "num_refused": sum(1 for r in results if r.get("refused_at") == "input"),
        }
