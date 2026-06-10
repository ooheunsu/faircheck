from __future__ import annotations

import argparse
import json

from statute_retrieval import StatuteRetrievalService, StatuteSearchConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="하도급 대금 지급 지연")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-size", type=int, default=30)
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    args = parser.parse_args()

    service = StatuteRetrievalService()
    print(json.dumps(service.health(), ensure_ascii=False, indent=2))

    result = service.search(
        args.query,
        top_k=args.top_k,
        config=StatuteSearchConfig(
            candidate_size=args.candidate_size,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
        ),
    )

    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
