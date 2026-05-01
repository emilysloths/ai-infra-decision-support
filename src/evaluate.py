"""Evaluation helpers for answer correctness, retrieval quality, and report export."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import csv
import json
from pathlib import Path
from typing import Any

from .agent import InfrastructureAssistant


@dataclass
class EvalCase:
    """One benchmark scenario used for regression checks."""

    question: str
    expected_keyword: str
    expected_site: str | None = None
    expected_sources: list[str] = field(default_factory=list)


def load_eval_cases(path: Path) -> list[EvalCase]:
    """Load benchmark cases from JSON so evaluation can scale without code edits."""

    raw_cases = json.loads(path.read_text(encoding="utf-8"))
    return [EvalCase(**item) for item in raw_cases]


def run_smoke_eval(
    agent: InfrastructureAssistant, cases: list[EvalCase]
) -> list[dict[str, object]]:
    """Run benchmark cases and collect answer and retrieval quality metrics."""

    results: list[dict[str, object]] = []
    for case in cases:
        answer = agent.answer(case.question)
        combined_text = " ".join(item["text"] for item in answer["evidence"]).lower()
        retrieved_sources = [item["source"] for item in answer["evidence"]]
        keyword_passed = _keyword_expectation_met(case.expected_keyword, combined_text)
        site_match = (
            True
            if case.expected_site is None
            else case.expected_site.lower() in answer["recommendation"].lower()
        )
        retrieval_hit = (
            True
            if not case.expected_sources
            else any(source in retrieved_sources for source in case.expected_sources)
        )

        precision_at_k = _precision_at_k(retrieved_sources, case.expected_sources)
        recall_at_k = _recall_at_k(retrieved_sources, case.expected_sources)
        results.append(
            {
                "question": case.question,
                "expected_keyword": case.expected_keyword,
                "expected_site": case.expected_site or "",
                "expected_sources": ", ".join(case.expected_sources),
                "passed": keyword_passed and site_match and retrieval_hit,
                "keyword_passed": keyword_passed,
                "site_passed": site_match,
                "retrieval_hit": retrieval_hit,
                "precision_at_k": round(precision_at_k, 4),
                "recall_at_k": round(recall_at_k, 4),
                "retrieval_backend": answer["retrieval_backend"],
                "confidence_label": answer["confidence"]["label"],
                "retrieved_sources": ", ".join(retrieved_sources),
                "top_recommendation": answer["recommendation"],
            }
        )
    return results


def summarize_results(results: list[dict[str, object]]) -> dict[str, Any]:
    """Aggregate case-level outcomes into a compact summary block."""

    if not results:
        return {
            "case_count": 0,
            "passed": 0,
            "pass_rate": 0.0,
            "average_precision_at_k": 0.0,
            "average_recall_at_k": 0.0,
        }

    passed = sum(1 for item in results if item["passed"])
    average_precision = sum(float(item["precision_at_k"]) for item in results) / len(results)
    average_recall = sum(float(item["recall_at_k"]) for item in results) / len(results)
    return {
        "case_count": len(results),
        "passed": passed,
        "pass_rate": round(passed / len(results), 4),
        "average_precision_at_k": round(average_precision, 4),
        "average_recall_at_k": round(average_recall, 4),
    }


def export_results(
    results: list[dict[str, object]],
    summary: dict[str, Any],
    output_path: Path,
) -> None:
    """Export evaluation results to CSV or JSON depending on file extension."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = output_path.suffix.lower()
    if suffix == ".json":
        output_path.write_text(
            json.dumps({"summary": summary, "results": results}, indent=2),
            encoding="utf-8",
        )
        return

    if suffix == ".csv":
        with output_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(results[0].keys()))
            writer.writeheader()
            writer.writerows(results)
        return

    raise ValueError("Unsupported export format. Use .csv or .json")


def _keyword_expectation_met(expected_keyword: str, combined_text: str) -> bool:
    """Treat keyword expectations as phrase-or-token coverage instead of exact-string only."""

    expected = expected_keyword.lower()
    if expected in combined_text:
        return True

    tokens = [token for token in expected.split() if token]
    return all(token in combined_text for token in tokens)


def _precision_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Compute precision over the retrieved source list."""

    if not retrieved_sources:
        return 0.0
    if not expected_sources:
        return 1.0
    hits = sum(1 for source in retrieved_sources if source in expected_sources)
    return hits / len(retrieved_sources)


def _recall_at_k(retrieved_sources: list[str], expected_sources: list[str]) -> float:
    """Compute recall over the expected source set."""

    if not expected_sources:
        return 1.0
    hits = sum(1 for source in expected_sources if source in retrieved_sources)
    return hits / len(expected_sources)
