"""Configuration objects shared across ingestion, retrieval, and decision support."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass
class RetrievalConfig:
    """Controls how documents are chunked, indexed, and retrieved."""

    backend: str = "auto"
    top_k: int = 4
    chunk_size: int = 450
    chunk_overlap: int = 80
    vector_store_dir: str = "chroma_db"
    reuse_index: bool = True


@dataclass
class DecisionConfig:
    """Controls weighted ranking across the planning criteria."""

    resilience: float = 0.30
    cyber_maturity: float = 0.30
    cost_efficiency: float = 0.20
    implementation_readiness: float = 0.20

    def normalized(self) -> dict[str, float]:
        """Normalize weights so UI sliders and config files can use any positive scale."""

        values = asdict(self)
        total = sum(values.values())
        if total <= 0:
            return {
                "resilience": 0.25,
                "cyber_maturity": 0.25,
                "cost_efficiency": 0.25,
                "implementation_readiness": 0.25,
            }
        return {key: value / total for key, value in values.items()}


@dataclass
class AssistantConfig:
    """Top-level configuration for the assistant runtime."""

    data_dir: str = "data"
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    decision: DecisionConfig = field(default_factory=DecisionConfig)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly representation for manifests and debugging."""

        return {
            "data_dir": self.data_dir,
            "retrieval": asdict(self.retrieval),
            "decision": asdict(self.decision),
        }
