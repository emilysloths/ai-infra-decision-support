"""Tests for benchmark evaluation and export."""

from pathlib import Path
import unittest

from src.agent import InfrastructureAssistant
from src.evaluate import export_results, load_eval_cases, run_smoke_eval, summarize_results


class EvaluateTests(unittest.TestCase):
    """Validate benchmark execution, metrics, and export helpers."""

    def test_smoke_eval_runs(self) -> None:
        agent = InfrastructureAssistant(data_dir="data")
        cases = load_eval_cases(Path("benchmarks/benchmark_cases.json"))
        results = run_smoke_eval(agent, cases)
        summary = summarize_results(results)
        self.assertTrue(results[0]["passed"])
        self.assertGreaterEqual(summary["pass_rate"], 0.75)

    def test_export_results_writes_json(self) -> None:
        agent = InfrastructureAssistant(data_dir="data")
        cases = load_eval_cases(Path("benchmarks/benchmark_cases.json"))
        results = run_smoke_eval(agent, cases)
        summary = summarize_results(results)
        output_dir = Path("artifacts")
        output_dir.mkdir(exist_ok=True)
        output_path = output_dir / "test_results.json"
        export_results(results, summary, output_path)
        self.assertTrue(output_path.exists())


if __name__ == "__main__":
    unittest.main()
