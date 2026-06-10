from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from typing import Any

from statute_retrieval import StatuteRetrievalService, StatuteSearchConfig


DEFAULT_QUERIES = [
    "하도급 대금 지급 지연",
    "원사업자가 하도급대금을 지급하지 않아 발주자에게 직접 지급을 요청",
    "건설하도급 계약에서 원사업자가 공사대금 지급보증을 하지 않음",
    "가맹본부가 가맹점주에게 특정 식자재 구매를 강제",
    "사업자들이 입찰 가격을 미리 합의하고 담합",
]


@dataclass(frozen=True)
class RunResult:
    mode: str
    wall_sec: float
    result: dict[str, Any]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("queries", nargs="*", help="비교할 사용자 쿼리입니다. 생략하면 기본 쿼리 세트를 사용합니다.")
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--candidate-size", type=int, default=20)
    parser.add_argument("--rerank-batch-size", type=int, default=8)
    parser.add_argument("--max-rerank-chars", type=int, default=3000)
    args = parser.parse_args()

    queries = args.queries or DEFAULT_QUERIES
    service = StatuteRetrievalService()

    print_header("법령 검색 reranker 비교")
    print_health(service.health())
    print(
        f"\n설정: top_k={args.top_k}, candidate_size={args.candidate_size}, "
        f"rerank_batch_size={args.rerank_batch_size}, max_rerank_chars={args.max_rerank_chars}"
    )

    for index, query in enumerate(queries, start=1):
        print_header(f"Query {index}: {query}")

        none_run = run_search(
            service,
            query,
            mode="none",
            top_k=args.top_k,
            candidate_size=args.candidate_size,
            rerank_batch_size=args.rerank_batch_size,
            max_rerank_chars=args.max_rerank_chars,
        )
        bge_run = run_search(
            service,
            query,
            mode="bge",
            top_k=args.top_k,
            candidate_size=args.candidate_size,
            rerank_batch_size=args.rerank_batch_size,
            max_rerank_chars=args.max_rerank_chars,
        )

        print_speed_table(none_run, bge_run)
        print_side_by_side(none_run, bge_run)
        print_rank_changes(none_run, bge_run)


def run_search(
    service: StatuteRetrievalService,
    query: str,
    mode: str,
    top_k: int,
    candidate_size: int,
    rerank_batch_size: int,
    max_rerank_chars: int,
) -> RunResult:
    started = time.perf_counter()
    result = service.search(
        query,
        top_k=top_k,
        config=StatuteSearchConfig(
            candidate_size=candidate_size,
            reranker_backend=mode,  # type: ignore[arg-type]
            rerank_batch_size=rerank_batch_size,
            max_rerank_chars=max_rerank_chars,
        ),
    )
    return RunResult(
        mode=mode,
        wall_sec=round(time.perf_counter() - started, 3),
        result=result,
    )


def print_header(title: str) -> None:
    print("\n" + "=" * 100)
    print(title)
    print("=" * 100)


def print_health(health: dict[str, Any]) -> None:
    print(
        "인덱스: "
        f"collection={health['collection_name']} "
        f"collection_count={health['collection_count']} "
        f"bm25_documents={health['bm25_documents']}"
    )
    print(f"ChromaDB: {health['chroma_dir']}")
    print(f"BM25: {health['bm25_path']}")


def print_speed_table(none_run: RunResult, bge_run: RunResult) -> None:
    print("\n[속도 비교]")
    print(f"{'mode':<10} {'wall':>8} {'dense':>8} {'bm25':>8} {'rrf':>8} {'rerank':>8}")
    print("-" * 58)
    for run in [none_run, bge_run]:
        timings = run.result["timings"]
        print(
            f"{run.mode:<10} "
            f"{run.wall_sec:>8.3f} "
            f"{timings.get('dense_sec', 0):>8.3f} "
            f"{timings.get('bm25_sec', 0):>8.3f} "
            f"{timings.get('rrf_and_fetch_sec', 0):>8.3f} "
            f"{timings.get('rerank_sec', 0):>8.3f}"
        )


def print_side_by_side(none_run: RunResult, bge_run: RunResult) -> None:
    none_results = none_run.result["results"]
    bge_results = bge_run.result["results"]
    max_rows = max(len(none_results), len(bge_results))

    print("\n[Top 결과 비교]")
    print(
        f"{'rank':<5} "
        f"{'none':<46} "
        f"{'bge':<46}"
    )
    print("-" * 100)
    for idx in range(max_rows):
        none_item = none_results[idx] if idx < len(none_results) else None
        bge_item = bge_results[idx] if idx < len(bge_results) else None
        print(
            f"{idx + 1:<5} "
            f"{format_short_result(none_item):<46} "
            f"{format_short_result(bge_item):<46}"
        )


def print_rank_changes(none_run: RunResult, bge_run: RunResult) -> None:
    none_rank = {
        item["statute_id"]: rank
        for rank, item in enumerate(none_run.result["results"], start=1)
    }
    bge_rank = {
        item["statute_id"]: rank
        for rank, item in enumerate(bge_run.result["results"], start=1)
    }
    statute_ids = list(dict.fromkeys([*none_rank.keys(), *bge_rank.keys()]))

    print("\n[순위 변화]")
    print(f"{'statute_id':<24} {'none':>6} {'bge':>6} {'change':>8}")
    print("-" * 48)
    for statute_id in statute_ids:
        old_rank = none_rank.get(statute_id)
        new_rank = bge_rank.get(statute_id)
        change = format_rank_change(old_rank, new_rank)
        print(
            f"{statute_id:<24} "
            f"{format_rank(old_rank):>6} "
            f"{format_rank(new_rank):>6} "
            f"{change:>8}"
        )


def format_short_result(item: dict[str, Any] | None) -> str:
    if item is None:
        return "-"
    statute = item["statute_id"]
    title = item["jo_title"]
    score = item["rerank_score"]
    dense = item["dense_rank"] or "-"
    bm25 = item["bm25_rank"] or "-"
    return f"{statute} {title[:12]} score={score:.3f} d={dense} b={bm25}"


def format_rank(rank: int | None) -> str:
    return str(rank) if rank is not None else "-"


def format_rank_change(old_rank: int | None, new_rank: int | None) -> str:
    if old_rank is None and new_rank is None:
        return "-"
    if old_rank is None:
        return "new"
    if new_rank is None:
        return "drop"
    diff = old_rank - new_rank
    if diff > 0:
        return f"up {diff}"
    if diff < 0:
        return f"down {-diff}"
    return "same"


if __name__ == "__main__":
    main()
