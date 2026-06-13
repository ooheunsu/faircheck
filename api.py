from __future__ import annotations

"""FairCheck FastAPI application.

프론트엔드는 이 파일의 `/api/risk-analysis` endpoint를 호출하면 됩니다.
응답은 Pydantic 모델로 고정해 두었기 때문에, 질문이 in_scope / needs_clarification /
out_of_scope 중 어디에 해당하더라도 같은 JSON 구조로 받을 수 있습니다.
"""

import os
import re
import time
import unicodedata
from pathlib import Path
from threading import Lock
from types import SimpleNamespace
from typing import Any, Literal

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, field_validator

from rag_answer import (
    DEFAULT_ANALYZER_MODEL,
    DEFAULT_MODEL,
    analyze_query,
    build_clarification_answer,
    build_out_of_scope_answer,
)
from risk_analysis_pipeline import (
    RiskAnalysisConfig,
    RiskAnalysisPipeline,
    elapsed,
)


Scope = Literal["in_scope", "needs_clarification", "out_of_scope"]
ResultType = Literal["risk_analysis", "needs_clarification", "out_of_scope"]
Provider = Literal["gemini", "dry-run"]
QueryAnalysisMode = Literal["auto", "rule-only", "off"]


class RiskAnalysisRequest(BaseModel):
    """프론트엔드가 백엔드로 보내는 요청 형식입니다."""

    question: str = Field(
        ...,
        min_length=1,
        max_length=3000,
        description="사용자가 입력한 공정거래 리스크 질문입니다.",
        examples=[
            "커피 가맹점주인데 계약서상 보장된 영업지역 100m 안에 본사가 직영점을 열겠다고 합니다."
        ],
    )
    provider: Provider = Field(
        default="gemini",
        description="gemini는 실제 답변 생성, dry-run은 프롬프트까지만 생성합니다.",
    )
    query_analysis: QueryAnalysisMode = Field(
        default="auto",
        description="질의 분석 방식입니다. 보통 프론트에서는 auto를 사용하면 됩니다.",
    )
    include_prompt: bool = Field(
        default=False,
        description="디버깅용입니다. true이면 생성된 LLM prompt를 응답에 포함합니다.",
    )

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("질문을 입력해주세요.")
        if cleaned.lower() in {"string", "null", "none"}:
            raise ValueError("Swagger 예시값이 아니라 실제 질문을 입력해주세요.")
        return cleaned


class AnalysisInfo(BaseModel):
    """질문이 서비스 범위 안인지 분석한 결과입니다."""

    scope: Scope
    source: str = ""
    interpreted_issue: str = ""
    search_query: str | None = None
    needs_clarification: bool = False
    suggested_question: str | None = None
    missing_facts: list[str] = Field(default_factory=list)
    reason: str = ""


class DecisionChunkReference(BaseModel):
    """의결서 카드 안에서 근거 chunk를 표시할 때 쓰는 정보입니다."""

    reference_id: str
    chunk_id: str | None = None
    score: float | None = None


class DecisionReference(BaseModel):
    """프론트엔드의 의결서 근거 카드에 들어갈 정보입니다."""

    reference_id: str
    title: str = ""
    violation_type: str = ""
    industry: str = ""
    best_doc_id: str | None = None
    pdf_source: str | None = None
    chunks: list[DecisionChunkReference] = Field(default_factory=list)


class StatuteReference(BaseModel):
    """프론트엔드의 법령 근거 카드에 들어갈 정보입니다."""

    reference_id: str
    statute_id: str | None = None
    law_title: str | None = None
    jo_number: str | None = None
    jo_title: str | None = None
    evidence_source: str | None = None
    content: str | None = None


class RiskAnalysisTimings(BaseModel):
    """각 단계별 실행 시간입니다. UI에서는 개발/디버깅용으로만 보여줘도 됩니다."""

    query_analysis_sec: float = 0.0
    decision_search_sec: float = 0.0
    decision_related_statute_lookup_sec: float = 0.0
    query_statute_search_sec: float = 0.0
    llm_answer_sec: float = 0.0
    total_sec: float = 0.0
    total_with_analysis_sec: float = 0.0


class RiskAnalysisResponse(BaseModel):
    """모든 케이스에서 공통으로 반환하는 응답 형식입니다.

    result_type이 risk_analysis이면 의결서/법령 근거가 함께 들어갑니다.
    needs_clarification이면 추가 확인이 필요한 사실과 예시 질문이 중심입니다.
    out_of_scope이면 공정거래 서비스 범위 밖이라는 안내가 중심입니다.
    """

    result_type: ResultType
    question: str
    answer: str | None = None
    analysis: AnalysisInfo
    search_query: str | None = None
    decision_references: list[DecisionReference] = Field(default_factory=list)
    statute_references: list[StatuteReference] = Field(default_factory=list)
    used_decision_references: list[DecisionReference] = Field(default_factory=list)
    used_statute_references: list[StatuteReference] = Field(default_factory=list)
    additional_decision_references: list[DecisionReference] = Field(default_factory=list)
    additional_statute_references: list[StatuteReference] = Field(default_factory=list)
    related_law_citations: list[str] = Field(default_factory=list)
    prompt: str | None = None
    timings: RiskAnalysisTimings = Field(default_factory=RiskAnalysisTimings)


class HealthResponse(BaseModel):
    status: Literal["ok"]
    pipeline_loaded: bool


app = FastAPI(
    title="FairCheck API",
    description="공정거래 리스크 진단을 위한 의결서/법령 RAG API",
    version="0.1.0",
)

# 로컬 프론트 개발 서버에서 API를 호출할 수 있도록 CORS를 열어 둡니다.
# 배포 단계에서는 허용 origin을 실제 프론트 주소로 좁히면 됩니다.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

_pipeline: RiskAnalysisPipeline | None = None
_pipeline_lock = Lock()


def decision_pdf_dir() -> Path:
    """의결서 원문 PDF 폴더를 .env에서 읽어옵니다."""

    configured = os.getenv("FAIRCHECK_DECISION_PDF_DIR", "").strip()
    if not configured:
        raise HTTPException(
            status_code=500,
            detail="FAIRCHECK_DECISION_PDF_DIR가 .env에 설정되어 있지 않습니다.",
        )

    pdf_dir = Path(configured).expanduser()
    if not pdf_dir.is_dir():
        raise HTTPException(
            status_code=500,
            detail=f"Decision PDF directory not found: {pdf_dir}",
        )
    return pdf_dir


def pdf_name_candidates(raw_filename: str) -> list[str]:
    """DB metadata에 들어온 원본 파일명을 실제 PDF 파일명 후보로 바꿉니다.

    의결서 metadata에는 PDF 파일명이 바로 들어올 수도 있고, metadata/hybrid JSON명이
    들어올 수도 있습니다. 브라우저에서 받은 값은 파일명만 사용해 경로 조작을 막습니다.
    """

    filename = Path(raw_filename).name.strip()
    if not filename:
        return []

    candidates = [filename]
    if filename.endswith("_metadata.json"):
        candidates.append(filename.replace("_metadata.json", ".pdf"))
    if filename.endswith("_hybrid.json"):
        candidates.append(filename.replace("_hybrid.json", ".pdf"))
    if not filename.lower().endswith(".pdf"):
        candidates.append(f"{filename}.pdf")

    unique_candidates = []
    for candidate in candidates:
        if candidate not in unique_candidates:
            unique_candidates.append(candidate)
    return unique_candidates


def find_decision_pdf(raw_filename: str) -> Path:
    """원문 PDF 폴더에서 요청한 의결서 PDF를 찾습니다."""

    pdf_dir = decision_pdf_dir()
    candidates = pdf_name_candidates(raw_filename)

    for candidate in candidates:
        direct_path = pdf_dir / candidate
        if direct_path.is_file() and direct_path.suffix.lower() == ".pdf":
            return direct_path

    normalized_candidates = {
        unicodedata.normalize("NFC", candidate)
        for candidate in candidates
        if candidate.lower().endswith(".pdf")
    }
    normalized_candidates.update(
        unicodedata.normalize("NFD", candidate)
        for candidate in candidates
        if candidate.lower().endswith(".pdf")
    )

    for pdf_path in pdf_dir.glob("*.pdf"):
        normalized_name = {
            unicodedata.normalize("NFC", pdf_path.name),
            unicodedata.normalize("NFD", pdf_path.name),
        }
        if normalized_name & normalized_candidates:
            return pdf_path

    raise HTTPException(status_code=404, detail=f"PDF not found: {Path(raw_filename).name}")


def get_pipeline() -> RiskAnalysisPipeline:
    """무거운 검색 모델을 최초 1회만 로딩하고 이후 요청에서는 재사용합니다."""

    global _pipeline
    with _pipeline_lock:
        if _pipeline is None:
            _pipeline = RiskAnalysisPipeline(RiskAnalysisConfig())
        return _pipeline


def make_analyzer_args(request: RiskAnalysisRequest) -> SimpleNamespace:
    """rag_answer.analyze_query()가 기대하는 CLI args 형태로 변환합니다."""

    return SimpleNamespace(
        query_analysis=request.query_analysis,
        # CLI에서는 provider=dry-run이면 질의 분석도 꺼지지만,
        # API에서는 dry-run이어도 out_of_scope / needs_clarification 판정이 필요합니다.
        # 그래서 analyzer에는 gemini provider로 전달하고, 실제 답변 생성 provider만 별도로 사용합니다.
        provider="gemini",
        analyzer_model=DEFAULT_ANALYZER_MODEL,
        thinking_budget=0,
    )


def make_analysis_info(analysis: dict[str, Any]) -> AnalysisInfo:
    return AnalysisInfo(
        scope=analysis.get("scope", "in_scope"),
        source=analysis.get("source", ""),
        interpreted_issue=analysis.get("interpreted_issue") or "",
        search_query=analysis.get("search_query"),
        needs_clarification=bool(analysis.get("needs_clarification")),
        suggested_question=analysis.get("suggested_question"),
        missing_facts=list(analysis.get("missing_facts") or []),
        reason=analysis.get("reason") or "",
    )


def make_timings(
    query_analysis_sec: float,
    total_started: float,
    pipeline_timings: dict[str, float] | None = None,
) -> RiskAnalysisTimings:
    pipeline_timings = pipeline_timings or {}
    return RiskAnalysisTimings(
        query_analysis_sec=query_analysis_sec,
        decision_search_sec=pipeline_timings.get("decision_search_sec", 0.0),
        decision_related_statute_lookup_sec=pipeline_timings.get(
            "decision_related_statute_lookup_sec",
            0.0,
        ),
        query_statute_search_sec=pipeline_timings.get("query_statute_search_sec", 0.0),
        llm_answer_sec=pipeline_timings.get("llm_answer_sec", 0.0),
        total_sec=pipeline_timings.get("total_sec", 0.0),
        total_with_analysis_sec=elapsed(total_started),
    )


def decision_references_from(result: dict[str, Any]) -> list[DecisionReference]:
    return [
        DecisionReference(
            reference_id=item.get("reference_id", ""),
            title=item.get("title", ""),
            violation_type=item.get("violation_type", ""),
            industry=item.get("industry", ""),
            best_doc_id=item.get("best_doc_id"),
            pdf_source=item.get("pdf_source"),
            chunks=[
                DecisionChunkReference(
                    reference_id=chunk.get("reference_id", ""),
                    chunk_id=chunk.get("chunk_id"),
                    score=chunk.get("score"),
                )
                for chunk in item.get("chunks", [])
            ],
        )
        for item in result.get("decision_references", [])
    ]


def statute_references_from(result: dict[str, Any]) -> list[StatuteReference]:
    return [
        StatuteReference(
            reference_id=item.get("reference_id", ""),
            statute_id=item.get("statute_id"),
            law_title=item.get("law_title"),
            jo_number=item.get("jo_number"),
            jo_title=item.get("jo_title"),
            evidence_source=item.get("evidence_source"),
            content=item.get("content"),
        )
        for item in result.get("statute_references", [])
    ]


def extract_used_reference_ids(answer: str | None) -> tuple[set[str], set[str]]:
    """LLM 답변 본문에서 실제 인용된 의결서/법령 번호를 찾습니다.

    답변에는 `[문서 1-근거 1]`, `[문서 1]`, `[법령 4]` 같은 번호가 들어갑니다.
    프론트에서는 이 번호가 실제 답변에 사용된 핵심 근거인지 판단해야 하므로,
    여기서 문자열을 정규표현식으로 뽑아냅니다.
    """

    if not answer:
        return set(), set()

    decision_ids = {
        f"문서 {match.group(1)}"
        for match in re.finditer(r"\[문서\s*(\d+)(?:\s*-\s*근거\s*\d+)?\]", answer)
    }
    statute_ids = {
        f"법령 {match.group(1)}"
        for match in re.finditer(r"\[법령\s*(\d+)\]", answer)
    }
    return decision_ids, statute_ids


def split_references_by_usage(
    answer: str | None,
    decision_references: list[DecisionReference],
    statute_references: list[StatuteReference],
) -> tuple[
    list[DecisionReference],
    list[StatuteReference],
    list[DecisionReference],
    list[StatuteReference],
]:
    """전체 검색 후보를 답변에 실제 인용된 근거와 추가 참고 근거로 나눕니다."""

    used_decision_ids, used_statute_ids = extract_used_reference_ids(answer)

    used_decisions = [
        reference
        for reference in decision_references
        if reference.reference_id in used_decision_ids
    ]
    additional_decisions = [
        reference
        for reference in decision_references
        if reference.reference_id not in used_decision_ids
    ]
    used_statutes = [
        reference
        for reference in statute_references
        if reference.reference_id in used_statute_ids
    ]
    additional_statutes = [
        reference
        for reference in statute_references
        if reference.reference_id not in used_statute_ids
    ]

    return used_decisions, used_statutes, additional_decisions, additional_statutes


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(status="ok", pipeline_loaded=_pipeline is not None)


@app.get("/api/decision-pdfs/{filename:path}")
def get_decision_pdf(filename: str) -> FileResponse:
    """의결서 근거 카드에서 PDF 원문을 열 때 사용하는 endpoint입니다."""

    pdf_path = find_decision_pdf(filename)
    return FileResponse(
        path=pdf_path,
        media_type="application/pdf",
        filename=pdf_path.name,
        content_disposition_type="inline",
    )


@app.post("/api/risk-analysis", response_model=RiskAnalysisResponse)
def analyze_risk(request: RiskAnalysisRequest) -> RiskAnalysisResponse:
    total_started = time.perf_counter()
    question = request.question.strip()
    if not question:
        raise HTTPException(status_code=400, detail="question은 비어 있을 수 없습니다.")

    analysis_started = time.perf_counter()
    analysis = analyze_query(question, make_analyzer_args(request))
    analysis_sec = elapsed(analysis_started)
    analysis_info = make_analysis_info(analysis)

    if analysis_info.scope == "out_of_scope":
        return RiskAnalysisResponse(
            result_type="out_of_scope",
            question=question,
            answer=build_out_of_scope_answer(question, analysis),
            analysis=analysis_info,
            search_query=analysis_info.search_query,
            timings=make_timings(analysis_sec, total_started),
        )

    if analysis_info.scope == "needs_clarification":
        return RiskAnalysisResponse(
            result_type="needs_clarification",
            question=question,
            answer=build_clarification_answer(question, analysis),
            analysis=analysis_info,
            search_query=analysis_info.search_query,
            timings=make_timings(analysis_sec, total_started),
        )

    try:
        pipeline = get_pipeline()
        result = pipeline.run(question, analysis=analysis, provider=request.provider)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    decision_references = decision_references_from(result)
    statute_references = statute_references_from(result)
    (
        used_decision_references,
        used_statute_references,
        additional_decision_references,
        additional_statute_references,
    ) = split_references_by_usage(
        result.get("answer"),
        decision_references,
        statute_references,
    )

    return RiskAnalysisResponse(
        result_type="risk_analysis",
        question=question,
        answer=result.get("answer"),
        analysis=analysis_info,
        search_query=result.get("search_query"),
        decision_references=decision_references,
        statute_references=statute_references,
        used_decision_references=used_decision_references,
        used_statute_references=used_statute_references,
        additional_decision_references=additional_decision_references,
        additional_statute_references=additional_statute_references,
        related_law_citations=list(result.get("related_law_citations") or []),
        prompt=result.get("prompt") if request.include_prompt or request.provider == "dry-run" else None,
        timings=make_timings(analysis_sec, total_started, result.get("timings")),
    )
