from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from faircheck.hybrid_retrieval import HybridRetrievalService, HybridSearchConfig


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("query", nargs="?", default="건설기계 임대단가 담합")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--candidate-size", type=int, default=30)
    parser.add_argument("--reranker", choices=["none", "bge", "qwen"], default="none")
    parser.add_argument("--dense-weight", type=float, default=1.0)
    parser.add_argument("--bm25-weight", type=float, default=1.0)
    args = parser.parse_args()

    service = HybridRetrievalService()
    print(json.dumps(service.health(), ensure_ascii=False, indent=2))

    result = service.search(
        args.query,
        top_k=args.top_k,
        config=HybridSearchConfig(
            candidate_size=args.candidate_size,
            reranker_backend=args.reranker,
            dense_weight=args.dense_weight,
            bm25_weight=args.bm25_weight,
        ),
    )

    print(json.dumps(
        {
            "query": result["query"],
            "reranker_backend": result["reranker_backend"],
            "timings": result["timings"],
            "doc_results": result["doc_results"],
        },
        ensure_ascii=False,
        indent=2,
    ))


if __name__ == "__main__":
    main()
