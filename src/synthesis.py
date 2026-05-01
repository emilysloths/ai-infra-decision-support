"""Structured answer synthesis over retrieved evidence and scorecards."""

from __future__ import annotations

from typing import Any


def synthesize_answer(
    question: str,
    recommendation: str,
    reasoning: str,
    scorecard: list[dict[str, Any]],
    evidence: list[dict[str, Any]],
) -> dict[str, Any]:
    """Create a user-facing answer structure with citations, tradeoffs, and confidence."""

    citations = _build_citations(evidence)
    executive_summary = recommendation
    if citations:
        executive_summary = f"{recommendation} Supported by {', '.join(citations[:2])}."

    tradeoffs = _extract_tradeoffs(scorecard)
    confidence = _confidence_summary(evidence)

    return {
        "executive_summary": executive_summary,
        "recommendation_body": _recommendation_body(recommendation, reasoning, citations),
        "tradeoffs": tradeoffs,
        "citations": citations,
        "confidence": confidence,
    }


def _build_citations(evidence: list[dict[str, Any]]) -> list[str]:
    """Citations point back to the ranked evidence shown in the UI and CLI."""

    citations: list[str] = []
    for idx, item in enumerate(evidence, start=1):
        citations.append(f"[{idx}] {item['source']}")
    return citations


def _recommendation_body(
    recommendation: str, reasoning: str, citations: list[str]
) -> str:
    """Combine the core recommendation with rationale and explicit evidence references."""

    citation_text = ", ".join(citations[:3]) if citations else "no evidence references available"
    return (
        f"{recommendation}\n\n"
        f"Rationale: {reasoning}\n\n"
        f"Primary evidence: {citation_text}"
    )


def _extract_tradeoffs(scorecard: list[dict[str, Any]]) -> list[str]:
    """Summarize how the top options differ so users can compare candidates quickly."""

    if len(scorecard) < 2:
        return ["Only one ranked option is currently available for comparison."]

    top = scorecard[0]
    second = scorecard[1]
    return [
        f"{top['site']} ranks first with overall score {top['score']:.2f}.",
        f"{second['site']} is the next strongest option with score {second['score']:.2f}.",
        f"Comparison focus: {top['summary']} versus {second['summary']}.",
    ]


def _confidence_summary(evidence: list[dict[str, Any]]) -> dict[str, Any]:
    """Estimate evidence quality from retrieval coverage and score concentration."""

    if not evidence:
        return {
            "label": "low",
            "average_score": 0.0,
            "evidence_count": 0,
        }

    average_score = sum(item["score"] for item in evidence) / len(evidence)
    label = "high" if average_score >= 0.20 else "medium" if average_score >= 0.10 else "low"
    return {
        "label": label,
        "average_score": round(average_score, 4),
        "evidence_count": len(evidence),
    }
