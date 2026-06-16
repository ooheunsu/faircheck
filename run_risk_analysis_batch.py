from __future__ import annotations

"""여러 사용자 쿼리를 한 번에 실행하고 결과를 txt 파일로 저장하는 스크립트.

단건 실행은 risk_analysis_pipeline.py를 사용하고, 여러 질문을 비교 평가할 때는
이 파일을 사용합니다. RiskAnalysisPipeline 객체를 한 번만 만들어 재사용하므로
쿼리마다 의결서/법령 검색 모델을 새로 로딩하지 않습니다.
"""

import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from rag_answer import (
    DEFAULT_ANALYZER_MODEL,
    DEFAULT_MODEL,
    analyze_query,
    build_clarification_answer,
    build_out_of_scope_answer,
    format_analysis_preview,
)
from risk_analysis_pipeline import (
    RiskAnalysisConfig,
    RiskAnalysisPipeline,
    elapsed,
)


DEFAULT_QUERIES = [
    "알바 인수인계 기간에 최저보다 적게 주는 거 돼?",
    "내가 이미 마라탕 장사를 하고 있는데 바로 옆옆에 새로운 마라탕 가게가 들어와. 이거 가서 따지고 싶은데 그래도되나",
    "내가 돼지고기집 체인점 사장인데 본사에서 무쌈을 납품받아서 써. 근데 납품안받고 그냥 우리집이 무쌈 장사를 해서 그거를 받아서 쓰고 싶은데 문제 되나?",
    "간판이랑 인테리어 한 지 얼마 안 됐는데 본사에서 또 바꾸라고 강요한다면 어떡해?",
    "일하고 나서 돈을 배 째고 계속 안 주는데 이자까지 다 받아낼 수 있는 방법이 있어?",
    "소프트웨어 개발업을 하는 스타트업인데, 원청 건설 대기업이 정식 계약서 발급도 없이 우리 특허 기술이 담긴 설계 도면과 소스코드를 먼저 전송하라고 요구한다면 어떻게 해야해?",
    "커피 전문 가맹점 점주로 2년째 운영 중인데, 가맹본사가 계약서상 보장된 제 영업지역을 무시하고 불과 100m 거리에 동종 직영점을 새로 오픈하겠다고 통보해 왔습니다.",
    "요식업을 운영하는 사장인데, 재료를 납품받는 업체 사장님이 친해서 우리 가게만 재료비를 할인해주겠다고 합니다. 나중에 문제가 될 수 있을까요?",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        default="risk_analysis_batch_results.txt",
        help="결과를 저장할 txt 파일 경로입니다.",
    )
    parser.add_argument(
        "--provider",
        choices=["gemini", "dry-run"],
        default="gemini",
        help="gemini는 실제 답변 생성, dry-run은 프롬프트까지만 생성합니다.",
    )
    parser.add_argument(
        "--query-analysis",
        default="auto",
        choices=["auto", "rule-only", "off"],
        help="auto는 질의 분석기를 사용하고, off는 원문 질문을 그대로 검색합니다.",
    )
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--analyzer-model", default=DEFAULT_ANALYZER_MODEL)
    parser.add_argument("--decision-candidate-size", type=int, default=20)
    parser.add_argument("--decision-doc-top-k", type=int, default=3)
    parser.add_argument("--decision-chunks-per-doc", type=int, default=3)
    parser.add_argument("--max-chars-per-decision-chunk", type=int, default=1200)
    parser.add_argument("--statute-top-k", type=int, default=5)
    parser.add_argument("--statute-candidate-size", type=int, default=30)
    parser.add_argument("--max-statutes-for-prompt", type=int, default=6)
    parser.add_argument("--max-chars-per-statute", type=int, default=900)
    parser.add_argument("--max-output-tokens", type=int, default=1400)
    parser.add_argument("--thinking-budget", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def build_pipeline(args: argparse.Namespace) -> RiskAnalysisPipeline:
    # 이 함수가 호출될 때 의결서 검색 모델, 의결서 reranker, 법령 검색 DB가 로딩됩니다.
    # batch 실행에서는 이 객체를 한 번만 만들고 모든 쿼리에 재사용합니다.
    config = RiskAnalysisConfig(
        decision_candidate_size=args.decision_candidate_size,
        decision_doc_top_k=args.decision_doc_top_k,
        decision_chunks_per_doc=args.decision_chunks_per_doc,
        max_chars_per_decision_chunk=args.max_chars_per_decision_chunk,
        statute_top_k=args.statute_top_k,
        statute_candidate_size=args.statute_candidate_size,
        max_statutes_for_prompt=args.max_statutes_for_prompt,
        max_chars_per_statute=args.max_chars_per_statute,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        thinking_budget=args.thinking_budget,
    )
    return RiskAnalysisPipeline(config)


def analyzer_args(args: argparse.Namespace) -> SimpleNamespace:
    # rag_answer.analyze_query()는 CLI args 형태의 객체를 기대합니다.
    # batch 스크립트에서는 필요한 값만 SimpleNamespace로 만들어 넘깁니다.
    return SimpleNamespace(
        query_analysis=args.query_analysis,
        provider=args.provider,
        analyzer_model=args.analyzer_model,
        thinking_budget=args.thinking_budget,
    )


def format_batch_item(
    index: int,
    question: str,
    analysis: dict[str, Any],
    result: dict[str, Any] | None,
    answer: str | None,
    analysis_sec: float,
) -> str:
    lines = [
        "=" * 100,
        f"Query {index}",
        "=" * 100,
        "",
        "[사용자 질문]",
        question,
        "",
        "[질의 분석]",
        format_analysis_preview(analysis),
        f"질의 분석 시간: {analysis_sec:.2f}초",
        "",
    ]

    if answer is not None:
        lines.extend(["[RAG 답변]", answer, ""])

    if result is None:
        return "\n".join(lines)

    lines.extend(
        [
            "[검색 질의]",
            result["search_query"],
            "",
            "[참조 의결서]",
        ]
    )
    for reference in result["decision_references"]:
        lines.append(
            f"- {reference['reference_id']} | {reference.get('title', '')} | "
            f"{reference.get('violation_type', '')} | best_doc_id={reference.get('best_doc_id')}"
        )

    lines.extend(["", "[참조 법령]"])
    for reference in result["statute_references"]:
        lines.append(
            f"- {reference['reference_id']} | {reference.get('statute_id')} | "
            f"{reference.get('law_title')} {reference.get('jo_number')} "
            f"{reference.get('jo_title')} | source={reference.get('evidence_source')}"
        )

    lines.extend(["", "[실행 시간]"])
    for key, value in result["timings"].items():
        lines.append(f"- {key}: {value:.2f}초")

    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_path = Path(args.output).expanduser()
    total_started = time.perf_counter()
    pipeline: RiskAnalysisPipeline | None = None
    sections = []

    for index, question in enumerate(DEFAULT_QUERIES, start=1):
        print(f"\n[{index}/{len(DEFAULT_QUERIES)}] 실행 중: {question}")

        analysis_started = time.perf_counter()
        analysis = analyze_query(question, analyzer_args(args))
        analysis_sec = elapsed(analysis_started)

        if analysis["scope"] == "out_of_scope":
            answer = build_out_of_scope_answer(question, analysis)
            sections.append(
                format_batch_item(index, question, analysis, None, answer, analysis_sec)
            )
            continue

        if analysis["scope"] == "needs_clarification":
            answer = build_clarification_answer(question, analysis)
            sections.append(
                format_batch_item(index, question, analysis, None, answer, analysis_sec)
            )
            continue

        if pipeline is None:
            print("검색 모델을 처음 한 번만 로딩합니다...")
            pipeline = build_pipeline(args)

        result = pipeline.run(question, analysis=analysis, provider=args.provider)
        result["timings"]["query_analysis_sec"] = analysis_sec
        answer = result["prompt"] if args.provider == "dry-run" else result["answer"]
        sections.append(
            format_batch_item(index, question, analysis, result, answer, analysis_sec)
        )

    header = "\n".join(
        [
            "FairCheck batch risk analysis results",
            f"provider: {args.provider}",
            f"query_analysis: {args.query_analysis}",
            f"query_count: {len(DEFAULT_QUERIES)}",
            f"total_elapsed_sec: {elapsed(total_started):.2f}",
            "",
        ]
    )
    output_path.write_text(header + "\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"\n저장 완료: {output_path}")
    print(f"전체 실행 시간: {elapsed(total_started):.2f}초")


if __name__ == "__main__":
    main()
