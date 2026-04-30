from __future__ import annotations

from dataclasses import dataclass

from .agent import InfrastructureAssistant


@dataclass
class EvalCase:
    question: str
    expected_keyword: str
    expected_site: str | None = None


def run_smoke_eval(agent: InfrastructureAssistant, cases: list[EvalCase]) -> list[dict[str, object]]:
    results: list[dict[str, object]] = []
    for case in cases:
        answer = agent.answer(case.question)
        combined_text = " ".join(item["text"] for item in answer["evidence"]).lower()
        passed = case.expected_keyword.lower() in combined_text
        site_match = (
            True
            if case.expected_site is None
            else case.expected_site.lower() in answer["recommendation"].lower()
        )
        results.append(
            {
                "question": case.question,
                "expected_keyword": case.expected_keyword,
                "passed": passed and site_match,
            }
        )
    return results
