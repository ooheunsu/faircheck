import argparse
import csv
import json
import time
from pathlib import Path

from rag_answer import analyze_query


def normalize_title(title):
    return " ".join((title or "").split())


def reciprocal_rank(ranked_titles, gold_titles):
    gold_set = {normalize_title(title) for title in gold_titles}
    for index, title in enumerate(ranked_titles, start=1):
        if normalize_title(title) in gold_set:
            return 1.0 / index, index
    return 0.0, None


def hit_at_k(ranked_titles, gold_titles, k=5):
    gold_set = {normalize_title(title) for title in gold_titles}
    return any(normalize_title(title) in gold_set for title in ranked_titles[:k])


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_analysis_args(args):
    class AnalysisArgs:
        provider = "gemini"
        query_analysis = args.query_analysis
        analyzer_model = args.analyzer_model
        thinking_budget = args.thinking_budget

    return AnalysisArgs()


def evaluate(args):
    dataset = load_dataset(args.dataset)
    if args.limit:
        dataset = dataset[:args.limit]

    results_path = Path(args.output)
    results_path.parent.mkdir(parents=True, exist_ok=True)

    analysis_args = make_analysis_args(args)
    retriever = None
    rows = []

    total_start = time.perf_counter()

    for item in dataset:
        query_id = item["id"]
        query = item["query"]
        expected_scope = item["expected_scope"]
        gold_titles = item.get("gold_doc_titles", [])

        print(f"\n[{query_id}] {item['category']}")
        print(query)

        analysis_start = time.perf_counter()
        analysis = analyze_query(query, analysis_args)
        analysis_elapsed = time.perf_counter() - analysis_start

        actual_scope = analysis["scope"]
        scope_correct = actual_scope == expected_scope
        search_query = analysis.get("search_query") or ""

        row = {
            "id": query_id,
            "category": item["category"],
            "query": query,
            "expected_scope": expected_scope,
            "actual_scope": actual_scope,
            "scope_correct": scope_correct,
            "interpreted_issue": analysis.get("interpreted_issue", ""),
            "search_query": search_query,
            "analysis_sec": round(analysis_elapsed, 3),
            "search_sec": "",
            "hit_at_5": "",
            "mrr": "",
            "first_gold_rank": "",
            "top1_title": "",
            "top5_titles": "",
            "gold_titles": " | ".join(gold_titles),
        }

        print(f"scope: {actual_scope} (expected: {expected_scope})")

        if actual_scope == "in_scope" and gold_titles:
            if retriever is None:
                from retriever import FaircheckRetriever, RetrieverConfig

                config = RetrieverConfig(
                    candidate_size=args.candidate_size,
                    doc_top_k=args.doc_top_k,
                    chunks_per_doc=args.chunks_per_doc,
                )
                retriever = FaircheckRetriever(config)

            search_start = time.perf_counter()
            search_result = retriever.search(
                search_query or query,
                candidate_size=args.candidate_size,
            )
            search_elapsed = time.perf_counter() - search_start

            ranked_titles = [
                normalize_title(result[2].get("의결서제목", ""))
                for result in search_result["doc_results"][:args.doc_top_k]
            ]
            rr, first_rank = reciprocal_rank(ranked_titles, gold_titles)
            hit = hit_at_k(ranked_titles, gold_titles, k=5)

            row.update(
                {
                    "search_sec": round(search_elapsed, 3),
                    "hit_at_5": int(hit),
                    "mrr": round(rr, 4),
                    "first_gold_rank": first_rank or "",
                    "top1_title": ranked_titles[0] if ranked_titles else "",
                    "top5_titles": " | ".join(ranked_titles[:5]),
                }
            )

            print(f"hit@5: {int(hit)} | rr: {rr:.4f} | first_gold_rank: {first_rank}")
            print(f"top1: {row['top1_title']}")

        rows.append(row)

    write_csv(results_path, rows)
    print_summary(rows, time.perf_counter() - total_start)
    print(f"\nSaved: {results_path}")


def write_csv(path, rows):
    fieldnames = [
        "id",
        "category",
        "query",
        "expected_scope",
        "actual_scope",
        "scope_correct",
        "interpreted_issue",
        "search_query",
        "analysis_sec",
        "search_sec",
        "hit_at_5",
        "mrr",
        "first_gold_rank",
        "top1_title",
        "top5_titles",
        "gold_titles",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, total_elapsed):
    scope_total = len(rows)
    scope_correct = sum(1 for row in rows if row["scope_correct"])

    retrieval_rows = [
        row
        for row in rows
        if row["expected_scope"] == "in_scope" and row["hit_at_5"] != ""
    ]
    hit_values = [int(row["hit_at_5"]) for row in retrieval_rows]
    mrr_values = [float(row["mrr"]) for row in retrieval_rows]

    avg_hit = sum(hit_values) / len(hit_values) if hit_values else 0.0
    avg_mrr = sum(mrr_values) / len(mrr_values) if mrr_values else 0.0

    analysis_times = [float(row["analysis_sec"]) for row in rows if row["analysis_sec"] != ""]
    search_times = [float(row["search_sec"]) for row in retrieval_rows if row["search_sec"] != ""]

    avg_analysis = sum(analysis_times) / len(analysis_times) if analysis_times else 0.0
    avg_search = sum(search_times) / len(search_times) if search_times else 0.0

    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"scope accuracy: {scope_correct}/{scope_total} = {scope_correct / scope_total:.3f}")
    print(f"doc Hit@5: {avg_hit:.3f} ({sum(hit_values)}/{len(hit_values) if hit_values else 0})")
    print(f"doc MRR: {avg_mrr:.3f}")
    print(f"avg analysis sec: {avg_analysis:.2f}")
    print(f"avg search/rerank sec: {avg_search:.2f}")
    print(f"total sec: {total_elapsed:.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation_queries.json")
    parser.add_argument("--output", default="eval_results/retrieval_eval.csv")
    parser.add_argument("--candidate-size", type=int, default=20)
    parser.add_argument("--doc-top-k", type=int, default=5)
    parser.add_argument("--chunks-per-doc", type=int, default=3)
    parser.add_argument(
        "--query-analysis",
        default="auto",
        choices=["auto", "rule-only", "off"],
    )
    parser.add_argument("--analyzer-model", default="gemini-2.5-flash")
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="앞에서부터 N개 질의만 평가합니다. 0이면 전체를 평가합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
