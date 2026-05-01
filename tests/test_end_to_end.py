"""End-to-end validation for the CLI-oriented answer flow."""

import unittest

from src.agent import InfrastructureAssistant


class EndToEndTests(unittest.TestCase):
    """Ensure the top-level pipeline returns the full Milestone 2 shape."""

    def test_full_answer_contains_milestone_two_fields(self) -> None:
        assistant = InfrastructureAssistant(data_dir="data")
        result = assistant.answer("Which site has elevated flood exposure?")
        self.assertIn("corpus_stats", result)
        self.assertIn("comparison_table", result)
        self.assertIn("recommendation_body", result)
        self.assertIn("tradeoffs", result)
        self.assertIn(result["confidence"]["label"], {"medium", "high"})


if __name__ == "__main__":
    unittest.main()
