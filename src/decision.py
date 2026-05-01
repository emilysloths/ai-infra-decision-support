"""Decision support logic for ranking candidate options from retrieved evidence."""

from __future__ import annotations

from collections import defaultdict
import re
from typing import Any

from .config import DecisionConfig
from .retrieval import RetrievedChunk


def build_recommendation(
    question: str,
    chunks: list[RetrievedChunk],
    decision_config: DecisionConfig | None = None,
) -> dict[str, Any]:
    """Convert retrieved evidence into a ranked recommendation and score breakdown."""

    decision_config = decision_config or DecisionConfig()
    if not chunks:
        return {
            "recommendation": "Insufficient evidence available.",
            "reasoning": "No relevant planning documents were retrieved for this question.",
            "scorecard": [],
            "comparison_table": [],
            "confidence": {"label": "low", "average_score": 0.0, "evidence_count": 0},
        }

    site_scores = _score_sites(question, chunks, decision_config)
    if not site_scores:
        return {
            "recommendation": (
                "Use the retrieved evidence to compare candidate options before selecting a final site."
            ),
            "reasoning": (
                "The documents are relevant to the question, but they do not contain a clearly repeated "
                "site or option label that can be ranked automatically."
            ),
            "scorecard": [],
            "comparison_table": [],
            "confidence": _confidence_from_chunks(chunks),
        }

    ranked = sorted(site_scores.items(), key=lambda item: item[1]["score"], reverse=True)
    top_site, top_data = ranked[0]
    reasoning = (
        f"{top_site} ranks highest when balancing resilience, implementation readiness, "
        f"cyber maturity, and cost considerations derived from the retrieved evidence. "
        f"The effective question-sensitive weight profile is: {_describe_weights(question, decision_config)}."
    )

    return {
        "recommendation": (
            f"The strongest current candidate is {top_site}, based on retrieved planning context "
            f"and the weighted decision criteria."
        ),
        "reasoning": reasoning,
        "scorecard": [
            {
                "site": site,
                "score": round(data["score"], 2),
                "resilience": data["resilience"],
                "cyber_maturity": data["cyber_maturity"],
                "cost_efficiency": data["cost_efficiency"],
                "implementation_readiness": data["implementation_readiness"],
                "relevance_bonus": data["relevance_bonus"],
                "summary": (
                    f"resilience={data['resilience']}, cyber={data['cyber_maturity']}, "
                    f"cost={data['cost_efficiency']}, readiness={data['implementation_readiness']}"
                ),
            }
            for site, data in ranked
        ],
        "comparison_table": _comparison_table(ranked),
        "confidence": _confidence_from_chunks(chunks),
    }


def _score_sites(
    question: str,
    chunks: list[RetrievedChunk],
    decision_config: DecisionConfig,
) -> dict[str, dict[str, float]]:
    """Aggregate chunk evidence into site-level weighted scores."""

    weights = _weights_for_question(question, decision_config)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    source_profiles = _source_profiles(chunks)
    for chunk in chunks:
        source_profile = source_profiles.get(chunk.source, {})
        site = str(chunk.metadata.get("site", source_profile.get("site", ""))) or None
        metrics: dict[str, float] = {}
        chunk_text_lower = chunk.text.lower()
        for key in (
            "resilience_score",
            "cyber_maturity_score",
            "cost_efficiency_score",
            "implementation_readiness_score",
        ):
            if key in chunk.metadata:
                try:
                    metrics[key] = float(chunk.metadata[key])
                except (TypeError, ValueError):
                    pass
        for line in chunk.text.splitlines():
            line = line.strip()
            if line.lower().startswith("site:"):
                site = line.split(":", 1)[1].strip()
            elif ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                try:
                    metrics[key] = float(value)
                except ValueError:
                    continue

        if not site:
            continue

        resilience = metrics.get(
            "resilience_score", float(source_profile.get("resilience_score", 3.0))
        )
        cyber_maturity = metrics.get(
            "cyber_maturity_score", float(source_profile.get("cyber_maturity_score", 3.0))
        )
        cost_efficiency = metrics.get(
            "cost_efficiency_score", float(source_profile.get("cost_efficiency_score", 3.0))
        )
        implementation_readiness = metrics.get(
            "implementation_readiness_score",
            float(source_profile.get("implementation_readiness_score", 3.0)),
        )
        criteria_score = (
            resilience * weights["resilience"]
            + cyber_maturity * weights["cyber_maturity"]
            + cost_efficiency * weights["cost_efficiency"]
            + implementation_readiness * weights["implementation_readiness"]
        )

        # Blend structured scores with query/evidence alignment so direct matches win appropriately.
        relevance_bonus = chunk.score * 6.0
        overlap_bonus = _keyword_overlap_bonus(question, chunk_text_lower)
        phrase_bonus = _phrase_bonus(question, chunk_text_lower)
        domain_bonus = _domain_bonus(question, chunk_text_lower)
        total = criteria_score + relevance_bonus + overlap_bonus + phrase_bonus + domain_bonus
        scores[site] = {
            "score": total,
            "resilience": resilience,
            "cyber_maturity": cyber_maturity,
            "cost_efficiency": cost_efficiency,
            "implementation_readiness": implementation_readiness,
            "relevance_bonus": round(
                relevance_bonus + overlap_bonus + phrase_bonus + domain_bonus, 2
            ),
        }
    return scores


def _source_profiles(chunks: list[RetrievedChunk]) -> dict[str, dict[str, float | str]]:
    """Map each source file to its declared site and structured metrics."""

    mapping: dict[str, dict[str, float | str]] = {}
    for chunk in chunks:
        profile = mapping.setdefault(chunk.source, {})
        for line in chunk.text.splitlines():
            line = line.strip()
            if line.lower().startswith("site:"):
                profile["site"] = line.split(":", 1)[1].strip()
            elif ":" in line:
                key, value = line.split(":", 1)
                key = key.strip().lower().replace(" ", "_")
                value = value.strip()
                try:
                    profile[key] = float(value)
                except ValueError:
                    continue
    return mapping


def _weights_for_question(question: str, decision_config: DecisionConfig) -> dict[str, float]:
    """Start from configured base weights and adjust them based on query intent."""

    question_lower = question.lower()
    weights = decision_config.normalized()
    if "cost" in question_lower or "budget" in question_lower:
        weights["cost_efficiency"] += 0.10
        weights["resilience"] = max(weights["resilience"] - 0.05, 0.0)
    if "cyber" in question_lower or "maturity" in question_lower or "c2m2" in question_lower:
        weights["cyber_maturity"] += 0.20
        weights["cost_efficiency"] = max(weights["cost_efficiency"] - 0.10, 0.0)
    if "resilien" in question_lower or "backup" in question_lower or "continuity" in question_lower:
        weights["resilience"] += 0.10
        weights["cost_efficiency"] = max(weights["cost_efficiency"] - 0.05, 0.0)

    total = sum(weights.values())
    return {key: value / total for key, value in weights.items()}


def _describe_weights(question: str, decision_config: DecisionConfig) -> str:
    """Human-readable rendering of the final active weight profile."""

    weights = _weights_for_question(question, decision_config)
    return ", ".join(f"{key}={value:.2f}" for key, value in weights.items())


def _keyword_overlap_bonus(question: str, chunk_text_lower: str) -> float:
    """Reward chunks that cover individual task keywords."""

    keywords = {
        token
        for token in re.findall(r"[a-zA-Z]+", question.lower())
        if token not in {"which", "site", "best", "for", "the", "is", "a", "an", "has"}
    }
    bonus = 0.0
    for keyword in keywords:
        if keyword in chunk_text_lower:
            bonus += 0.25
    return bonus


def _phrase_bonus(question: str, chunk_text_lower: str) -> float:
    """Favor evidence chunks that match multi-word phrases from the query."""

    terms = [
        token
        for token in re.findall(r"[a-zA-Z]+", question.lower())
        if token not in {"which", "site", "best", "for", "the", "is", "a", "an", "has"}
    ]
    phrases = []
    for size in (3, 2):
        for idx in range(len(terms) - size + 1):
            phrases.append(" ".join(terms[idx : idx + size]))

    bonus = 0.0
    for phrase in phrases:
        if phrase in chunk_text_lower:
            bonus += 1.0 if len(phrase.split()) == 3 else 0.6
    return bonus


def _domain_bonus(question: str, chunk_text_lower: str) -> float:
    """Boost domain phrases that indicate especially relevant planning evidence."""

    question_lower = question.lower()
    bonus = 0.0
    if ("backup" in question_lower or "continuity" in question_lower) and (
        "backup infrastructure" in chunk_text_lower
        or "continuity of operations" in chunk_text_lower
    ):
        bonus += 1.5
    if ("cyber" in question_lower or "maturity" in question_lower) and (
        "cybersecurity maturity" in chunk_text_lower or "cyber maturity" in chunk_text_lower
    ):
        bonus += 1.5
    if ("critical" in question_lower and "operations" in question_lower) and (
        "critical operations" in chunk_text_lower
        or "critical infrastructure modernization" in chunk_text_lower
    ):
        bonus += 1.2
    if "flood" in question_lower and "flood exposure" in chunk_text_lower:
        bonus += 1.5
    return bonus


def _comparison_table(
    ranked: list[tuple[str, dict[str, float]]]
) -> list[dict[str, float | str]]:
    """Return a tabular comparison that can be displayed directly in the UI."""

    return [
        {
            "site": site,
            "overall_score": round(data["score"], 2),
            "resilience": data["resilience"],
            "cyber_maturity": data["cyber_maturity"],
            "cost_efficiency": data["cost_efficiency"],
            "implementation_readiness": data["implementation_readiness"],
            "relevance_bonus": data["relevance_bonus"],
        }
        for site, data in ranked
    ]


def _confidence_from_chunks(chunks: list[RetrievedChunk]) -> dict[str, Any]:
    """Estimate answer confidence from retrieval score concentration."""

    if not chunks:
        return {"label": "low", "average_score": 0.0, "evidence_count": 0}

    average_score = sum(chunk.score for chunk in chunks) / len(chunks)
    label = "high" if average_score >= 0.20 else "medium" if average_score >= 0.10 else "low"
    return {
        "label": label,
        "average_score": round(average_score, 4),
        "evidence_count": len(chunks),
    }
