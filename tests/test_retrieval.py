"""Tests for ingestion and retrieval behavior."""

from pathlib import Path
import unittest

from src.agent import InfrastructureAssistant
from src.config import DecisionConfig
from src.ingest import load_documents


class IngestionTests(unittest.TestCase):
    """Validate chunk metadata and ingestion diagnostics."""

    def test_load_documents_preserves_metadata(self) -> None:
        report = load_documents(Path("data"))
        self.assertTrue(report.chunks)
        first_chunk = report.chunks[0]
        self.assertTrue(first_chunk.chunk_id)
        self.assertTrue(first_chunk.path.endswith(first_chunk.source))
        self.assertIn("title", first_chunk.metadata)
        self.assertTrue(report.fingerprint)


class RetrievalTests(unittest.TestCase):
    """Validate the main answer path and returned retrieval metadata."""

    def test_assistant_returns_evidence(self) -> None:
        assistant = InfrastructureAssistant(data_dir="data")
        result = assistant.answer("Which site is best for resilient backup infrastructure?")
        self.assertTrue(result["evidence"])
        self.assertIn("recommendation", result)
        self.assertTrue(result["scorecard"])
        self.assertIn("retrieval_backend", result)
        self.assertIn("doc_type", result["evidence"][0])
        self.assertIn("metadata", result["evidence"][0])
        self.assertIn("executive_summary", result)
        self.assertIn("citations", result)

    def test_assistant_accepts_custom_decision_weights(self) -> None:
        assistant = InfrastructureAssistant(
            data_dir="data",
            decision_config=DecisionConfig(
                resilience=0.10,
                cyber_maturity=0.60,
                cost_efficiency=0.10,
                implementation_readiness=0.20,
            ),
        )
        result = assistant.answer("Which site is best for critical cyber maturity planning?")
        self.assertIn("confidence", result)
        self.assertTrue(result["comparison_table"])


if __name__ == "__main__":
    unittest.main()
