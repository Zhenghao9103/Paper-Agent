import argparse
import json
from pathlib import Path

from .rag_quality import evaluate_rag_results, load_eval_payload, metrics_to_dict


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate precomputed RAG QA results.")
    parser.add_argument("input", type=Path, help="JSON file with cases and results arrays.")
    parser.add_argument("--k", type=int, default=5, help="Top-K cutoff for retrieval metrics.")
    args = parser.parse_args()

    payload = json.loads(args.input.read_text(encoding="utf-8"))
    cases, results = load_eval_payload(payload)
    metrics = evaluate_rag_results(cases, results, k=args.k)
    print(json.dumps(metrics_to_dict(metrics), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
