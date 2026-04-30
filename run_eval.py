from src.agent import InfrastructureAssistant
from src.evaluate import EvalCase, run_smoke_eval


def main() -> None:
    agent = InfrastructureAssistant(data_dir="data")
    cases = [
        EvalCase(
            question="Which site is the best fit for resilient backup infrastructure?",
            expected_keyword="resilient backup infrastructure",
            expected_site="High Desert",
        ),
        EvalCase(
            question="Which site has elevated flood exposure?",
            expected_keyword="flood exposure",
            expected_site="River North",
        ),
        EvalCase(
            question="Which site has the strongest cyber maturity for critical operations?",
            expected_keyword="cyber maturity",
            expected_site="Cascade Junction",
        ),
    ]

    results = run_smoke_eval(agent, cases)
    passed = sum(1 for item in results if item["passed"])

    print("Evaluation results:")
    for item in results:
        status = "PASS" if item["passed"] else "FAIL"
        print(f"- {status}: {item['question']}")

    print(f"\nScore: {passed}/{len(results)} passed")


if __name__ == "__main__":
    main()
