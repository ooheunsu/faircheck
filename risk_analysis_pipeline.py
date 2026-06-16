from __future__ import annotations

"""의결서 검색 + 법령 검색 + Gemini 답변 생성을 하나로 묶는 실행 파일.

터미널에서 `python risk_analysis_pipeline.py ...`를 실행하면 이 파일의
`main()` 함수가 시작점이 됩니다. 나중에 FastAPI를 만들 때도
`RiskAnalysisPipeline` 클래스를 import해서 같은 흐름을 API 안에서 재사용할 수 있습니다.
"""

import argparse
import json
import re
import time
from dataclasses import dataclass
from typing import Any

# rag_answer.py는 "질문 분석", "프롬프트 만들기", "Gemini 호출"을 담당합니다.
# 여기서는 이미 만들어진 함수들을 가져와 통합 파이프라인 안에서 재사용합니다.
from rag_answer import (
    DEFAULT_ANALYZER_MODEL,
    DEFAULT_MODEL,
    SAMPLE_QUESTIONS,
    analyze_query,
    build_clarification_answer,
    build_out_of_scope_answer,
    build_prompt,
    call_gemini,
    format_analysis_preview,
    format_context,
    format_context_preview,
    format_reference_ids,
)
# decision_retriever.py는 친구가 만든 의결서 검색기입니다.
# BGE-M3 dense 검색 + BM25 + RRF + reranker로 유사 의결서 chunk를 찾습니다.
from decision_retriever import FaircheckRetriever, RetrieverConfig
# statute_retriever.py는 우리가 만든 법령 검색기입니다.
# BGE-M3 dense 검색 + BM25 + RRF로 관련 법령 조문을 찾습니다.
from statute_retriever import (
    StatuteRetrievalService,
    StatuteSearchConfig,
    load_env_file,
)


LAW_META_KEYS = (
    # 의결서 metadata에서 "관련 법령"이 들어 있을 수 있는 key 후보들입니다.
    # 파일마다 key 이름이 조금 다를 수 있어서 여러 이름을 함께 확인합니다.
    "related_laws",
    "관련법령",
    "관련 법령",
    "관련법률",
    "적용법조",
    "법령",
)

PDF_META_KEYS = (
    # 나중에 "의결서 카드 클릭 -> 원문 PDF 열기" 기능을 붙일 때 사용할 수 있는
    # PDF 파일명/path 관련 metadata key 후보들입니다.
    "original_filename",
    "원본파일명",
    "파일명",
    "pdf_filename",
    "pdf_path",
)


@dataclass(frozen=True)
class RiskAnalysisConfig:
    # dataclass는 설정값을 담는 작은 상자입니다.
    # frozen=True는 실행 중 실수로 설정값을 바꾸지 못하게 막는 옵션입니다.
    decision_candidate_size: int = 20
    decision_doc_top_k: int = 3
    decision_chunks_per_doc: int = 3
    max_chars_per_decision_chunk: int = 1200

    statute_top_k: int = 5
    statute_candidate_size: int = 30
    max_statutes_for_prompt: int = 6
    max_chars_per_statute: int = 900

    model: str = DEFAULT_MODEL
    temperature: float = 0.2
    max_output_tokens: int = 1400
    thinking_budget: int = 0


class RiskAnalysisPipeline:
    """의결서 검색, 법령 검색, Gemini 답변 생성을 순서대로 실행하는 통합 클래스."""

    def __init__(self, config: RiskAnalysisConfig | None = None) -> None:
        # .env에 적어둔 DB 경로와 API 키를 os.environ에 올립니다.
        # os.environ은 Python 코드가 읽을 수 있는 실행 환경 설정이라고 보면 됩니다.
        load_env_file()
        self.config = config or RiskAnalysisConfig()

        # 친구 의결서 검색기 설정입니다. 통합 파이프라인의 설정값을
        # decision_retriever.py의 RetrieverConfig 형식으로 바꿔 넘깁니다.
        decision_config = RetrieverConfig(
            candidate_size=self.config.decision_candidate_size,
            doc_top_k=self.config.decision_doc_top_k,
            chunks_per_doc=self.config.decision_chunks_per_doc,
        )
        # 실제 의결서 검색기와 법령 검색기를 한 번 로딩해 둡니다.
        # 모델/DB 로딩이 무거우므로, 나중에 API 서버에서는 요청마다 새로 만들지 않고
        # 서버 시작 시 한 번 만들어 재사용하는 쪽이 좋습니다.
        self.decision_retriever = FaircheckRetriever(decision_config)
        self.statute_retriever = StatuteRetrievalService()

    def run(
        self,
        question: str,
        analysis: dict[str, Any] | None = None,
        provider: str = "gemini",
    ) -> dict[str, Any]:
        # run()은 통합 파이프라인의 핵심 함수입니다.
        # 입력: 사용자 질문(question)
        # 출력: 답변, 근거 의결서, 근거 법령, 실행 시간 등을 담은 dict
        started = time.perf_counter()
        search_query = (analysis or {}).get("search_query") or question

        # 1단계: 사용자 질문과 비슷한 공정위 의결서를 찾습니다.
        # decision_result 안에는 chunk 결과, 문서 단위 결과, LLM에 넣을 doc_contexts가 들어 있습니다.
        decision_started = time.perf_counter()
        decision_result = self.decision_retriever.search(
            search_query,
            candidate_size=self.config.decision_candidate_size,
        )
        decision_result["query"] = question
        decision_result["search_query"] = search_query
        decision_elapsed = elapsed(decision_started)

        # 2단계: 의결서 검색 결과 metadata에서 관련 법령 문자열을 모읍니다.
        # 예: "하도급거래 공정화에 관한 법률 제13조 제1항"
        citation_started = time.perf_counter()
        related_law_citations = collect_related_law_citations(
            decision_result["doc_contexts"]
        )
        # 3단계: 의결서에 직접 적힌 법령명을 기준으로 법령 DB에서 조문을 조회합니다.
        # 이 방식은 "의결서가 실제로 사용한 법"을 따라가는 경로입니다.
        citation_statutes = self.statute_retriever.get_by_citations(
            related_law_citations,
            top_k=self.config.max_statutes_for_prompt,
        )
        citation_elapsed = elapsed(citation_started)

        # 4단계: 사용자 질문 자체로도 법령 검색을 합니다.
        # 이 방식은 "의결서에 나온 법"과 별개로 사용자 상황에 직접 가까운 법을 찾는 경로입니다.
        statute_started = time.perf_counter()
        query_statute_result = self.statute_retriever.search(
            search_query,
            top_k=self.config.statute_top_k,
            config=StatuteSearchConfig(
                candidate_size=self.config.statute_candidate_size,
            ),
        )
        query_statutes = query_statute_result["results"]
        statute_elapsed = elapsed(statute_started)

        # 5단계: 두 법령 경로를 합칩니다.
        # 같은 statute_id가 중복되면 한 번만 남깁니다.
        merged_statutes = merge_statute_results(
            citation_statutes,
            query_statutes,
            max_items=self.config.max_statutes_for_prompt,
        )

        # 6단계: 의결서 근거와 법령 근거를 LLM 프롬프트에 넣기 좋은 긴 문자열로 바꿉니다.
        decision_context_text = format_context(
            decision_result["doc_contexts"],
            max_chars_per_chunk=self.config.max_chars_per_decision_chunk,
        )
        statute_context_text = format_statute_context(
            merged_statutes,
            max_chars_per_statute=self.config.max_chars_per_statute,
        )
        # 7단계: rag_answer.py의 build_prompt()를 사용해 최종 Gemini 입력문을 만듭니다.
        prompt = build_prompt(
            question=question,
            search_query=search_query,
            context_text=decision_context_text,
            analysis=analysis,
            statute_context_text=statute_context_text,
        )

        answer = None
        llm_elapsed = 0.0
        if provider != "dry-run":
            # dry-run이 아니면 실제 Gemini API를 호출해 답변을 생성합니다.
            # dry-run은 API 비용 없이 "프롬프트가 어떻게 만들어졌는지"만 확인하는 모드입니다.
            llm_started = time.perf_counter()
            answer = call_gemini(
                prompt,
                model=self.config.model,
                temperature=self.config.temperature,
                max_output_tokens=self.config.max_output_tokens,
                thinking_budget=self.config.thinking_budget,
            )
            llm_elapsed = elapsed(llm_started)

        return {
            # 나중에 FastAPI 응답 모델을 만들 때 이 dict 구조를 기반으로 삼을 수 있습니다.
            "question": question,
            "search_query": search_query,
            "analysis": analysis,
            "answer": answer,
            "prompt": prompt if provider == "dry-run" else None,
            "decision_result": decision_result,
            "related_law_citations": related_law_citations,
            "citation_statutes": citation_statutes,
            "query_statutes": query_statutes,
            "statute_evidence": merged_statutes,
            "decision_references": build_decision_references(
                decision_result["doc_contexts"]
            ),
            "statute_references": build_statute_references(merged_statutes),
            "timings": {
                "decision_search_sec": decision_elapsed,
                "decision_related_statute_lookup_sec": citation_elapsed,
                "query_statute_search_sec": statute_elapsed,
                "llm_answer_sec": llm_elapsed,
                "total_sec": elapsed(started),
            },
        }


def collect_related_law_citations(doc_contexts: list[dict[str, Any]]) -> list[str]:
    """의결서 context에서 관련 법령 문자열만 모읍니다.

    doc_contexts는 decision_retriever.py가 만든 RAG용 의결서 묶음입니다.
    각 문서 meta와 chunk meta를 훑으면서 LAW_META_KEYS에 해당하는 값을 찾습니다.
    """
    citations: list[str] = []
    seen: set[str] = set()

    for context in doc_contexts:
        metas = [context.get("meta", {})]
        metas.extend(chunk.get("meta", {}) for chunk in context.get("chunks", []))

        for meta in metas:
            for key in LAW_META_KEYS:
                for citation in split_law_value(meta.get(key)):
                    normalized = normalize_space(citation)
                    if normalized and normalized not in seen:
                        seen.add(normalized)
                        citations.append(normalized)

    return citations


def split_law_value(value: Any) -> list[str]:
    """metadata에 들어 있는 법령 값을 list[str] 형태로 정리합니다.

    metadata 값은 문자열, list, dict, JSON 문자열 등으로 섞여 있을 수 있습니다.
    그래서 재귀적으로 풀어낸 뒤 줄바꿈/세미콜론/쉼표 기준으로 나눕니다.
    """
    if value is None:
        return []

    if isinstance(value, list):
        items: list[str] = []
        for item in value:
            items.extend(split_law_value(item))
        return items

    if isinstance(value, dict):
        items = []
        for item in value.values():
            items.extend(split_law_value(item))
        return items

    text = str(value).strip()
    if not text:
        return []

    try:
        parsed = json.loads(text)
    except Exception:
        parsed = None
    if parsed is not None and parsed is not value:
        parsed_items = split_law_value(parsed)
        if parsed_items:
            return parsed_items

    parts = re.split(r"[\n;,]+", text)
    return [part.strip(" -ㆍ\t") for part in parts if part.strip(" -ㆍ\t")]


def merge_statute_results(
    citation_statutes: list[dict[str, Any]],
    query_statutes: list[dict[str, Any]],
    max_items: int,
) -> list[dict[str, Any]]:
    """의결서 기반 법령과 쿼리 기반 법령을 합치고 중복 조문을 제거합니다."""
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()

    for source_name, statutes in (
        ("decision_related_laws", citation_statutes),
        ("query_search", query_statutes),
    ):
        for statute in statutes:
            statute_id = statute.get("statute_id") or statute.get("id")
            if not statute_id or statute_id in seen:
                continue
            seen.add(statute_id)
            merged.append({**statute, "evidence_source": source_name})
            if len(merged) >= max_items:
                return merged

    return merged


def format_statute_context(
    statutes: list[dict[str, Any]],
    max_chars_per_statute: int = 900,
) -> str:
    """법령 검색 결과를 Gemini 프롬프트에 넣을 수 있는 텍스트로 바꿉니다."""
    if not statutes:
        return "직접 조회되거나 검색된 관련 법령 근거가 없습니다."

    sections = []
    for index, statute in enumerate(statutes, start=1):
        document = (statute.get("document") or statute.get("snippet") or "")[
            :max_chars_per_statute
        ]
        source_label = {
            "decision_related_laws": "의결서 관련 법령 직접 조회",
            "query_search": "사용자 질의 기반 법령 검색",
        }.get(statute.get("evidence_source"), statute.get("evidence_source", ""))

        lines = [
            f"[법령 {index}]",
            f"법령명: {statute.get('law_title', '')}",
            f"조문번호: {statute.get('jo_number', '')}",
            f"조문제목: {statute.get('jo_title', '')}",
            f"근거 출처: {source_label}",
        ]
        if statute.get("matched_citation"):
            lines.append(f"의결서 기재 법령: {statute['matched_citation']}")
        if statute.get("score") is not None:
            lines.append(f"검색 점수: {statute.get('score')}")
        lines.append(f"내용: {document}")
        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)


def build_decision_references(doc_contexts: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """프론트엔드가 의결서 근거 카드를 만들 때 쓸 참조 정보를 정리합니다.

    지금은 터미널 출력용으로만 보이지만, 나중에 API 응답에 넣으면
    사용자가 의결서 카드를 클릭해서 PDF 원문으로 이동하는 기능에 활용할 수 있습니다.
    """
    references = []
    for index, context in enumerate(doc_contexts, start=1):
        meta = context.get("meta", {})
        references.append(
            {
                "reference_id": f"문서 {index}",
                "title": meta.get("의결서제목", ""),
                "violation_type": meta.get("위반유형", ""),
                "industry": meta.get("업종", ""),
                "best_doc_id": context.get("best_doc_id"),
                "pdf_source": first_meta_value(meta, PDF_META_KEYS),
                "chunks": [
                    {
                        "reference_id": f"문서 {index}-근거 {chunk_index}",
                        "chunk_id": chunk.get("doc_id"),
                        "score": chunk.get("score"),
                    }
                    for chunk_index, chunk in enumerate(context.get("chunks", []), start=1)
                ],
            }
        )
    return references


def build_statute_references(statutes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """프론트엔드가 법령 근거 카드를 만들 때 쓸 참조 정보를 정리합니다."""
    return [
        {
            "reference_id": f"법령 {index}",
            "statute_id": statute.get("statute_id"),
            "law_title": statute.get("law_title"),
            "jo_number": statute.get("jo_number"),
            "jo_title": statute.get("jo_title"),
            "evidence_source": statute.get("evidence_source"),
        }
        for index, statute in enumerate(statutes, start=1)
    ]


def first_meta_value(meta: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = meta.get(key)
        if value:
            return value
    return None


def normalize_space(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 3)


def print_pipeline_result(result: dict[str, Any], provider: str) -> None:
    print("\n" + "=" * 70)
    print("검색 질의")
    print("=" * 70)
    print(result["search_query"])

    print("\n" + "=" * 70)
    print("의결서 근거 요약")
    print("=" * 70)
    print(format_context_preview(result["decision_result"]["doc_contexts"]))

    print("\n" + "=" * 70)
    print("의결서에서 수집한 관련 법령")
    print("=" * 70)
    if result["related_law_citations"]:
        for citation in result["related_law_citations"]:
            print(f"- {citation}")
    else:
        print("수집된 관련 법령 없음")

    print("\n" + "=" * 70)
    print("법령 근거")
    print("=" * 70)
    print(format_statute_context(result["statute_evidence"], max_chars_per_statute=260))

    if provider == "dry-run":
        print("\n" + "=" * 70)
        print("생성된 Prompt")
        print("=" * 70)
        print(result["prompt"])
    else:
        print("\n" + "=" * 70)
        print("RAG 답변")
        print("=" * 70)
        print(result["answer"])

    print("\n" + "=" * 70)
    print("참조 의결서 ID")
    print("=" * 70)
    print(format_reference_ids(result["decision_result"]["doc_contexts"]))

    print("\n" + "=" * 70)
    print("참조 법령 ID")
    print("=" * 70)
    for reference in result["statute_references"]:
        print(
            f"[{reference['reference_id']}] "
            f"{reference.get('statute_id')} "
            f"{reference.get('law_title')} {reference.get('jo_number')} "
            f"{reference.get('jo_title')}"
        )

    print("\n" + "=" * 70)
    print("실행 시간")
    print("=" * 70)
    for key, value in result["timings"].items():
        print(f"{key}: {value:.2f}초")


def parse_args() -> argparse.Namespace:
    # argparse는 터미널 옵션을 Python 객체로 바꿔주는 표준 라이브러리입니다.
    # 예: --provider gemini -> args.provider == "gemini"
    parser = argparse.ArgumentParser()
    parser.add_argument("--question", default=SAMPLE_QUESTIONS["franchise_delivery_fee"])
    parser.add_argument("--sample", choices=sorted(SAMPLE_QUESTIONS.keys()))
    parser.add_argument("--provider", choices=["gemini", "dry-run"], default="gemini")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--analyzer-model", default=DEFAULT_ANALYZER_MODEL)
    parser.add_argument(
        "--query-analysis",
        default="auto",
        choices=["auto", "rule-only", "off"],
    )
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


def main() -> None:
    # 이 파일을 터미널에서 직접 실행할 때 시작되는 함수입니다.
    # import해서 쓰는 경우에는 아래쪽 if __name__ == "__main__" 조건이 false라서 자동 실행되지 않습니다.
    total_started = time.perf_counter()
    args = parse_args()
    question = SAMPLE_QUESTIONS[args.sample] if args.sample else args.question

    analysis_started = time.perf_counter()
    analysis = analyze_query(question, args)
    analysis_elapsed = elapsed(analysis_started)

    print("\n" + "=" * 70)
    print("사용자 질문")
    print("=" * 70)
    print(question)

    print("\n" + "=" * 70)
    print("질의 분석")
    print("=" * 70)
    print(format_analysis_preview(analysis))

    if analysis["scope"] == "out_of_scope":
        print("\n" + "=" * 70)
        print("RAG 답변 (범위 밖)")
        print("=" * 70)
        print(build_out_of_scope_answer(question, analysis))
        print(f"\n질의 분석: {analysis_elapsed:.2f}초")
        print(f"전체: {elapsed(total_started):.2f}초")
        return

    if analysis["scope"] == "needs_clarification":
        print("\n" + "=" * 70)
        print("RAG 답변 (추가 확인 필요)")
        print("=" * 70)
        print(build_clarification_answer(question, analysis))
        print(f"\n질의 분석: {analysis_elapsed:.2f}초")
        print(f"전체: {elapsed(total_started):.2f}초")
        return

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
    pipeline = RiskAnalysisPipeline(config)
    result = pipeline.run(question, analysis=analysis, provider=args.provider)
    result["timings"]["query_analysis_sec"] = analysis_elapsed
    result["timings"]["total_with_analysis_sec"] = elapsed(total_started)
    print_pipeline_result(result, args.provider)


if __name__ == "__main__":
    main()
