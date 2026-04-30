from __future__ import annotations

from collections import defaultdict
import re

from .retrieval import RetrievedChunk


def build_recommendation(question: str, chunks: list[RetrievedChunk]) -> dict[str, str]:
    if not chunks:
        return {
            "recommendation": "Insufficient evidence available.",
            "reasoning": "No relevant planning documents were retrieved for this question.",
            "scorecard": [],
        }

    site_scores = _score_sites(question, chunks)
    if site_scores:
        ranked = sorted(site_scores.items(), key=lambda item: item[1]["score"], reverse=True)
        top_site, top_data = ranked[0]
        recommendation = (
            f"The strongest current candidate is {top_site}, based on retrieved planning context "
            f"and the weighted decision criteria."
        )
        reasoning = (
            f"{top_site} ranks highest when balancing resilience, implementation readiness, "
            f"cyber maturity, and cost considerations derived from the retrieved evidence. "
            f"The current question emphasis is: {_describe_weights(question)}."
        )
        scorecard = [
            {
                "site": site,
                "score": round(data["score"], 2),
                "summary": (
                    f"resilience={data['resilience']}, cyber={data['cyber_maturity']}, "
                    f"cost={data['cost_efficiency']}, readiness={data['implementation_readiness']}"
                ),
            }
            for site, data in ranked
        ]
    else:
        recommendation = (
            "Use the retrieved evidence to compare candidate options before selecting a final site."
        )
        reasoning = (
            "The documents are relevant to the question, but they do not contain a clearly repeated "
            "site or option label that can be ranked automatically."
        )
        scorecard = []

    return {
        "recommendation": recommendation,
        "reasoning": reasoning,
        "scorecard": scorecard,
    }


def _score_sites(question: str, chunks: list[RetrievedChunk]) -> dict[str, dict[str, float]]:
    weights = _weights_for_question(question)
    scores: dict[str, dict[str, float]] = defaultdict(dict)
    for chunk in chunks:
        site = None
        metrics: dict[str, float] = {}
        chunk_text_lower = chunk.text.lower()
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

        resilience = metrics.get("resilience_score", 3.0)
        cyber_maturity = metrics.get("cyber_maturity_score", 3.0)
        cost_efficiency = metrics.get("cost_efficiency_score", 3.0)
        implementation_readiness = metrics.get("implementation_readiness_score", 3.0)
        criteria_score = (
            resilience * weights["resilience"]
            + cyber_maturity * weights["cyber_maturity"]
            + cost_efficiency * weights["cost_efficiency"]
            + implementation_readiness * weights["implementation_readiness"]
        )
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


def _weights_for_question(question: str) -> dict[str, float]:
    question_lower = question.lower()
    weights = {
        "resilience": 0.3,
        "cyber_maturity": 0.3,
        "cost_efficiency": 0.2,
        "implementation_readiness": 0.2,
    }
    if "cost" in question_lower or "budget" in question_lower:
        weights["cost_efficiency"] = 0.35
        weights["resilience"] = 0.25
    if "cyber" in question_lower or "maturity" in question_lower or "c2m2" in question_lower:
        weights["cyber_maturity"] = 0.4
        weights["implementation_readiness"] = 0.15
    if "resilien" in question_lower or "backup" in question_lower or "continuity" in question_lower:
        weights["resilience"] = 0.4
        weights["cost_efficiency"] = 0.15
    return weights


def _describe_weights(question: str) -> str:
    weights = _weights_for_question(question)
    return ", ".join(f"{key}={value:.2f}" for key, value in weights.items())


def _keyword_overlap_bonus(question: str, chunk_text_lower: str) -> float:
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
    terms = [
        token
        for token in re.findall(r"[a-zA-Z]+", question.lower())
        if token not in {"which", "site", "best", "for", "the", "is", "a", "an", "has"}
    ]
    phrases = []
    for size in (3, 2):
        for idx in range(len(terms) - size + 1):
            phrase = " ".join(terms[idx : idx + size])
            phrases.append(phrase)

    bonus = 0.0
    for phrase in phrases:
        if phrase in chunk_text_lower:
            bonus += 1.0 if len(phrase.split()) == 3 else 0.6
    return bonus


def _domain_bonus(question: str, chunk_text_lower: str) -> float:
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
    if "flood" in question_lower and "flood exposure" in chunk_text_lower:
        bonus += 1.5
    return bonus
