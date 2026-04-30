from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .decision import build_recommendation
from .retrieval import RetrievedChunk, Retriever


@dataclass
class InfrastructureAssistant:
    data_dir: str = "data"
    top_k: int = 4

    def __post_init__(self) -> None:
        self.retriever = Retriever.from_directory(Path(self.data_dir))

    def answer(self, question: str) -> dict[str, Any]:
        matches = self.retriever.search(question, top_k=self.top_k)
        recommendation = build_recommendation(question, matches)

        return {
            "question": question,
            "recommendation": recommendation["recommendation"],
            "reasoning": recommendation["reasoning"],
            "scorecard": recommendation["scorecard"],
            "evidence": [self._serialize_chunk(chunk) for chunk in matches],
        }

    @staticmethod
    def _serialize_chunk(chunk: RetrievedChunk) -> dict[str, Any]:
        return {
            "source": chunk.source,
            "text": chunk.text,
            "score": chunk.score,
        }
