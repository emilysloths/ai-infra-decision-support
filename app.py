from src.agent import InfrastructureAssistant


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="AI Infrastructure Planning Assistant")
    parser.add_argument(
        "--question",
        required=True,
        help="Infrastructure planning question to answer.",
    )
    parser.add_argument(
        "--data-dir",
        default="data",
        help="Directory containing planning and policy documents.",
    )
    args = parser.parse_args()

    assistant = InfrastructureAssistant(data_dir=args.data_dir)
    result = assistant.answer(args.question)

    print("\nQuestion:")
    print(result["question"])

    print("\nRecommendation:")
    print(result["recommendation"])

    print("\nDecision Scorecard:")
    for item in result["scorecard"]:
        print(f"- {item['site']}: {item['score']:.2f} ({item['summary']})")

    print("\nReasoning:")
    print(result["reasoning"])

    print("\nEvidence:")
    for idx, item in enumerate(result["evidence"], start=1):
        print(f"{idx}. [{item['source']}] score={item['score']:.4f}")
        print(f"   {item['text']}")


if __name__ == "__main__":
    main()
