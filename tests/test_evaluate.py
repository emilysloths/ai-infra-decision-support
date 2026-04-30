import unittest

from src.agent import InfrastructureAssistant
from src.evaluate import EvalCase, run_smoke_eval


class EvaluateTests(unittest.TestCase):
    def test_smoke_eval_runs(self) -> None:
        agent = InfrastructureAssistant(data_dir="data")
        cases = [
            EvalCase(
                question="Which site supports resilient backup infrastructure?",
                expected_keyword="backup infrastructure",
                expected_site="High Desert",
            )
        ]
        results = run_smoke_eval(agent, cases)
        self.assertTrue(results[0]["passed"])


if __name__ == "__main__":
    unittest.main()
