import unittest

from src.agent import InfrastructureAssistant


class RetrievalTests(unittest.TestCase):
    def test_assistant_returns_evidence(self) -> None:
        assistant = InfrastructureAssistant(data_dir="data")
        result = assistant.answer("Which site is best for resilient backup infrastructure?")
        self.assertTrue(result["evidence"])
        self.assertIn("recommendation", result)
        self.assertTrue(result["scorecard"])


if __name__ == "__main__":
    unittest.main()
