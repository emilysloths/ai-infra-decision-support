"""Top-level orchestration for ingestion, retrieval, ranking, and synthesis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import AssistantConfig, DecisionConfig, RetrievalConfig
from .decision import build_recommendation
from .retrieval import RetrievedChunk, Retriever
from .synthesis import synthesize_answer


@dataclass
class InfrastructureAssistant:
    """Facade object used by the CLI, UI, and evaluation workflows."""

    data_dir: str = "data"
    top_k: int = 4
    vector_store_dir: str = "chroma_db"
    retrieval_backend: str = "auto"
    chunk_size: int = 450
    chunk_overlap: int = 80
    reuse_index: bool = True
    decision_config: DecisionConfig | None = None

    def __post_init__(self) -> None:
        """Build a retriever and preserve the ingestion report for diagnostics."""

        self.config = AssistantConfig(
            data_dir=self.data_dir,
            retrieval=RetrievalConfig(
                backend=self.retrieval_backend,
                top_k=self.top_k,
                chunk_size=self.chunk_size,
                chunk_overlap=self.chunk_overlap,
                vector_store_dir=self.vector_store_dir,
                reuse_index=self.reuse_index,
            ),
            decision=self.decision_config or DecisionConfig(),
        )
        self.retriever, self.ingestion_report = Retriever.from_directory(
            Path(self.data_dir),
            config=self.config.retrieval,
        )

    def answer(self, question: str) -> dict[str, Any]:
        """Run the full question-answering pipeline and return a structured response."""

        matches = self.retriever.search(question, top_k=self.config.retrieval.top_k)
        recommendation = build_recommendation(
            question,
            matches,
            decision_config=self.config.decision,
        )
        evidence = [self._serialize_chunk(chunk) for chunk in matches]
        synthesis = synthesize_answer(
            question=question,
            recommendation=recommendation["recommendation"],
            reasoning=recommendation["reasoning"],
            scorecard=recommendation["scorecard"],
            evidence=evidence,
        )

        return {
            "question": question,
            "retrieval_backend": self.retriever.backend,
            "corpus_stats": self.retriever.corpus_stats(),
            "ingestion_warnings": [issue.__dict__ for issue in self.ingestion_report.warnings],
            "ingestion_errors": [issue.__dict__ for issue in self.ingestion_report.errors],
            "recommendation": recommendation["recommendation"],
            "reasoning": recommendation["reasoning"],
            "scorecard": recommendation["scorecard"],
            "comparison_table": recommendation["comparison_table"],
            "confidence": recommendation["confidence"],
            "evidence": evidence,
            "executive_summary": synthesis["executive_summary"],
            "recommendation_body": synthesis["recommendation_body"],
            "tradeoffs": synthesis["tradeoffs"],
            "citations": synthesis["citations"],
        }

    @staticmethod
    def _serialize_chunk(chunk: RetrievedChunk) -> dict[str, Any]:
        """Return a JSON-like evidence object that UIs and exporters can consume directly."""

        return {
            "chunk_id": chunk.chunk_id,
            "source": chunk.source,
            "path": chunk.path,
            "doc_type": chunk.doc_type,
            "text": chunk.text,
            "score": chunk.score,
            "metadata": chunk.metadata,
        }
