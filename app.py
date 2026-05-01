"""Command-line entry point for the planning assistant."""

from src.agent import InfrastructureAssistant
from src.config import DecisionConfig


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
    parser.add_argument(
        "--backend",
        default="auto",
        choices=["auto", "chroma", "tfidf"],
        help="Retrieval backend preference.",
    )
    parser.add_argument("--top-k", type=int, default=4, help="Number of evidence chunks to return.")
    parser.add_argument("--chunk-size", type=int, default=450, help="Chunk size for indexing.")
    parser.add_argument(
        "--chunk-overlap",
        type=int,
        default=80,
        help="Overlap between adjacent chunks.",
    )
    parser.add_argument(
        "--resilience-weight",
        type=float,
        default=0.30,
        help="Base weight for resilience in decision scoring.",
    )
    parser.add_argument(
        "--cyber-weight",
        type=float,
        default=0.30,
        help="Base weight for cyber maturity in decision scoring.",
    )
    parser.add_argument(
        "--cost-weight",
        type=float,
        default=0.20,
        help="Base weight for cost efficiency in decision scoring.",
    )
    parser.add_argument(
        "--readiness-weight",
        type=float,
        default=0.20,
        help="Base weight for implementation readiness in decision scoring.",
    )
    args = parser.parse_args()

    assistant = InfrastructureAssistant(
        data_dir=args.data_dir,
        top_k=args.top_k,
        retrieval_backend=args.backend,
        chunk_size=args.chunk_size,
        chunk_overlap=args.chunk_overlap,
        decision_config=DecisionConfig(
            resilience=args.resilience_weight,
            cyber_maturity=args.cyber_weight,
            cost_efficiency=args.cost_weight,
            implementation_readiness=args.readiness_weight,
        ),
    )
    result = assistant.answer(args.question)

    print("\nQuestion:")
    print(result["question"])

    print("\nExecutive Summary:")
    print(result["executive_summary"])

    print("\nRecommendation:")
    print(result["recommendation_body"])

    print("\nConfidence:")
    print(result["confidence"])

    print("\nDecision Scorecard:")
    for item in result["scorecard"]:
        print(f"- {item['site']}: {item['score']:.2f} ({item['summary']})")

    print("\nTradeoffs:")
    for item in result["tradeoffs"]:
        print(f"- {item}")

    print("\nReasoning:")
    print(result["reasoning"])

    print("\nCorpus Stats:")
    print(result["corpus_stats"])

    if result["ingestion_warnings"]:
        print("\nIngestion Warnings:")
        for issue in result["ingestion_warnings"]:
            print(f"- {issue['path']}: {issue['message']}")

    print("\nEvidence:")
    for idx, item in enumerate(result["evidence"], start=1):
        print(f"{idx}. [{item['source']}] score={item['score']:.4f}")
        print(f"   {item['text']}")


if __name__ == "__main__":
    main()
