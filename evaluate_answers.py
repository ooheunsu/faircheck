import argparse
import csv
import json
import time
from pathlib import Path

from rag_answer import (
    analyze_query,
    build_prompt,
    call_gemini,
    extract_json_object,
    format_context,
    format_context_preview,
    format_reference_ids,
)


def load_dataset(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def make_analysis_args(args):
    class AnalysisArgs:
        provider = args.provider
        query_analysis = args.query_analysis
        analyzer_model = args.analyzer_model
        thinking_budget = args.thinking_budget

    return AnalysisArgs()


def build_judge_prompt(question, answer, context_preview):
    return f"""\
당신은 공정거래위원회 의결서 기반 RAG 시스템의 답변 품질 평가자입니다.
아래 사용자 질문, 검색 근거 요약, RAG 답변을 보고 4개 항목을 1~5점으로 평가하세요.

평가 기준:
- faithfulness: 답변이 검색 근거 안에서만 말하는가. 근거에 없는 사실, 사건명, 판단 이유를 만들면 낮게 평가.
- answer_relevance: 답변이 사용자 질문에 직접 답하는가.
- practical_usefulness: 실무자가 확인할 사실, 리스크 포인트, 다음 행동을 이해하는 데 도움이 되는가.
- context_precision: 검색 근거로 제시된 문서/근거가 질문과 실제로 관련 있는가.

점수 기준:
- 5: 매우 좋음
- 4: 대체로 좋음
- 3: 보통, 일부 아쉬움
- 2: 관련성이나 근거 충실성에 뚜렷한 문제
- 1: 거의 실패

주의:
- 최종 법률 판단이 아니라 RAG 답변 품질만 평가하세요.
- 답변이 신중하게 "가능성", "판단 보류", "추가 확인 필요"라고 표현한 것은 감점 사유가 아닙니다.
- 반드시 JSON만 출력하세요. 마크다운 코드블록은 쓰지 마세요.

JSON 형식:
{{
  "faithfulness": 1,
  "answer_relevance": 1,
  "practical_usefulness": 1,
  "context_precision": 1,
  "overall_comment": "한 문장 총평",
  "weaknesses": ["개선점"]
}}

[사용자 질문]
{question}

[검색 근거 요약]
{context_preview}

[RAG 답변]
{answer}
"""


def normalize_score(value):
    try:
        score = int(value)
    except (TypeError, ValueError):
        return 0
    return max(1, min(5, score))


def judge_answer(args, question, answer, context_preview):
    if args.provider == "dry-run":
        return {
            "faithfulness": "",
            "answer_relevance": "",
            "practical_usefulness": "",
            "context_precision": "",
            "overall_comment": "dry-run: judge call skipped",
            "weaknesses": [],
        }

    prompt = build_judge_prompt(question, answer, context_preview)
    text = call_gemini(
        prompt,
        model=args.judge_model,
        temperature=0.0,
        max_output_tokens=args.judge_max_output_tokens,
        thinking_budget=args.thinking_budget,
        show_metadata=False,
    )
    raw = extract_json_object(text)

    return {
        "faithfulness": normalize_score(raw.get("faithfulness")),
        "answer_relevance": normalize_score(raw.get("answer_relevance")),
        "practical_usefulness": normalize_score(raw.get("practical_usefulness")),
        "context_precision": normalize_score(raw.get("context_precision")),
        "overall_comment": raw.get("overall_comment", ""),
        "weaknesses": raw.get("weaknesses", []),
    }


def evaluate(args):
    dataset = load_dataset(args.dataset)
    if args.only_in_scope:
        dataset = [item for item in dataset if item["expected_scope"] == "in_scope"]
    if args.limit:
        dataset = dataset[:args.limit]

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    analysis_args = make_analysis_args(args)
    retriever = None
    rows = []
    total_start = time.perf_counter()

    for item in dataset:
        query_id = item["id"]
        question = item["query"]

        print(f"\n[{query_id}] {item['category']}")
        print(question)

        analysis_start = time.perf_counter()
        analysis = analyze_query(question, analysis_args)
        analysis_sec = time.perf_counter() - analysis_start

        row = {
            "id": query_id,
            "category": item["category"],
            "question": question,
            "expected_scope": item["expected_scope"],
            "actual_scope": analysis["scope"],
            "search_query": analysis.get("search_query") or "",
            "answer": "",
            "faithfulness": "",
            "answer_relevance": "",
            "practical_usefulness": "",
            "context_precision": "",
            "overall_comment": "",
            "weaknesses": "",
            "reference_ids": "",
            "analysis_sec": round(analysis_sec, 3),
            "search_sec": "",
            "answer_sec": "",
            "judge_sec": "",
            "total_sec": "",
        }

        if analysis["scope"] != "in_scope":
            row["overall_comment"] = "Skipped: scope is not in_scope."
            rows.append(row)
            print(f"skip: scope={analysis['scope']}")
            continue

        if retriever is None:
            from retriever import FaircheckRetriever, RetrieverConfig

            config = RetrieverConfig(
                candidate_size=args.candidate_size,
                doc_top_k=args.doc_top_k,
                chunks_per_doc=args.chunks_per_doc,
            )
            retriever = FaircheckRetriever(config)

        item_start = time.perf_counter()

        search_start = time.perf_counter()
        search_result = retriever.search(
            analysis.get("search_query") or question,
            candidate_size=args.candidate_size,
        )
        search_sec = time.perf_counter() - search_start

        context_text = format_context(
            search_result["doc_contexts"],
            max_chars_per_chunk=args.max_chars_per_chunk,
        )
        context_preview = format_context_preview(
            search_result["doc_contexts"],
            max_chars_per_chunk=args.preview_chars_per_chunk,
        )

        prompt = build_prompt(
            question=question,
            search_query=analysis.get("search_query") or question,
            context_text=context_text,
            analysis=analysis,
        )

        answer_start = time.perf_counter()
        if args.provider == "dry-run":
            answer = "dry-run: answer generation skipped"
        else:
            answer = call_gemini(
                prompt,
                model=args.model,
                temperature=args.temperature,
                max_output_tokens=args.max_output_tokens,
                thinking_budget=args.thinking_budget,
                show_metadata=False,
            )
        answer_sec = time.perf_counter() - answer_start

        judge_start = time.perf_counter()
        judge = judge_answer(args, question, answer, context_preview)
        judge_sec = time.perf_counter() - judge_start

        row.update(
            {
                "answer": answer,
                "faithfulness": judge["faithfulness"],
                "answer_relevance": judge["answer_relevance"],
                "practical_usefulness": judge["practical_usefulness"],
                "context_precision": judge["context_precision"],
                "overall_comment": judge["overall_comment"],
                "weaknesses": " | ".join(str(x) for x in judge.get("weaknesses", [])),
                "reference_ids": format_reference_ids(search_result["doc_contexts"]),
                "search_sec": round(search_sec, 3),
                "answer_sec": round(answer_sec, 3),
                "judge_sec": round(judge_sec, 3),
                "total_sec": round(time.perf_counter() - item_start + analysis_sec, 3),
            }
        )

        rows.append(row)

        print(
            "scores: "
            f"faithfulness={row['faithfulness']}, "
            f"relevance={row['answer_relevance']}, "
            f"usefulness={row['practical_usefulness']}, "
            f"context_precision={row['context_precision']}"
        )

    write_csv(output_path, rows)
    print_summary(rows, time.perf_counter() - total_start)
    print(f"\nSaved: {output_path}")


def write_csv(path, rows):
    fieldnames = [
        "id",
        "category",
        "question",
        "expected_scope",
        "actual_scope",
        "search_query",
        "answer",
        "faithfulness",
        "answer_relevance",
        "practical_usefulness",
        "context_precision",
        "overall_comment",
        "weaknesses",
        "reference_ids",
        "analysis_sec",
        "search_sec",
        "answer_sec",
        "judge_sec",
        "total_sec",
    ]

    with open(path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_summary(rows, total_sec):
    scored_rows = [
        row
        for row in rows
        if row["faithfulness"] != ""
        and row["answer_relevance"] != ""
        and row["practical_usefulness"] != ""
        and row["context_precision"] != ""
    ]

    def avg(field):
        values = [float(row[field]) for row in scored_rows if row[field] != ""]
        return sum(values) / len(values) if values else 0.0

    print("\n" + "=" * 60)
    print("Answer Evaluation Summary")
    print("=" * 60)
    print(f"scored answers: {len(scored_rows)}")
    print(f"faithfulness avg: {avg('faithfulness'):.2f}")
    print(f"answer relevance avg: {avg('answer_relevance'):.2f}")
    print(f"practical usefulness avg: {avg('practical_usefulness'):.2f}")
    print(f"context precision avg: {avg('context_precision'):.2f}")
    print(f"total sec: {total_sec:.2f}")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="evaluation_queries.json")
    parser.add_argument("--output", default="eval_results/answer_eval.csv")
    parser.add_argument("--provider", default="gemini", choices=["gemini", "dry-run"])
    parser.add_argument("--model", default="gemini-2.5-flash")
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
    parser.add_argument("--analyzer-model", default="gemini-2.5-flash")
    parser.add_argument("--candidate-size", type=int, default=20)
    parser.add_argument("--doc-top-k", type=int, default=3)
    parser.add_argument("--chunks-per-doc", type=int, default=3)
    parser.add_argument("--max-chars-per-chunk", type=int, default=1200)
    parser.add_argument("--preview-chars-per-chunk", type=int, default=320)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument("--judge-max-output-tokens", type=int, default=700)
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument(
        "--query-analysis",
        default="auto",
        choices=["auto", "rule-only", "off"],
    )
    parser.add_argument(
        "--only-in-scope",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="기본값은 expected_scope가 in_scope인 질의만 답변 품질 평가합니다.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="앞에서부터 N개만 평가합니다. 0이면 전체를 평가합니다.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    evaluate(parse_args())
