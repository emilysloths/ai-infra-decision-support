"""CLI runner for benchmark evaluation and report export."""

from pathlib import Path

from src.agent import InfrastructureAssistant
from src.evaluate import export_results, load_eval_cases, run_smoke_eval, summarize_results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run Milestone 2 evaluation benchmarks.")
    parser.add_argument(
        "--benchmark-file",
        default="benchmarks/benchmark_cases.json",
        help="Path to the JSON benchmark case file.",
    )
    parser.add_argument(
        "--output",
        default="artifacts/eval_results.json",
        help="Optional export path (.json or .csv).",
    )
    args = parser.parse_args()

    agent = InfrastructureAssistant(data_dir="data")
    cases = load_eval_cases(Path(args.benchmark_file))
    results = run_smoke_eval(agent, cases)
    summary = summarize_results(results)
    export_results(results, summary, Path(args.output))

    print("Evaluation results:")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(
            f"- {status}: {item['question']} | "
            f"precision@k={item['precision_at_k']:.2f} "
            f"recall@k={item['recall_at_k']:.2f}"
        )

    print("\nSummary:")
    for key, value in summary.items():
        print(f"- {key}: {value}")

    print(f"\nExported report: {args.output}")


if __name__ == "__main__":
    main()
