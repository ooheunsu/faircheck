from __future__ import annotations

from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel, Field

from faircheck.search import DEFAULT_DB_PATH, FairCheckSearchService


app = FastAPI(
    title="FairCheck API",
    description="공정거래위원회 의결서와 법령을 검색해 위반 가능성 경고 근거를 제공하는 API입니다.",
    version="0.1.0",
)

service = FairCheckSearchService(DEFAULT_DB_PATH)


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=2, examples=["건설기계 임대단가 담합"])
    top_k: int = Field(5, ge=1, le=20)


class RiskCheckRequest(BaseModel):
    situation: str = Field(
        ...,
        min_length=2,
        examples=["가맹점주에게 특정 식자재를 본사 지정 업체에서만 구매하도록 요구하고 있습니다."],
    )
    top_k: int = Field(5, ge=1, le=20)


@app.get("/")
def root() -> dict[str, str]:
    return {
        "service": "FairCheck API",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/health")
def health() -> dict[str, Any]:
    return service.health()


@app.post("/search/decisions")
def search_decisions(request: SearchRequest) -> dict[str, Any]:
    return {
        "query": request.query,
        "results": service.search_decisions(request.query, top_k=request.top_k),
    }


@app.get("/decisions/{decision_id}")
def get_decision(decision_id: str) -> dict[str, Any]:
    decision = service.get_decision(decision_id)
    if decision is None:
        raise HTTPException(status_code=404, detail="Decision not found")
    return decision


@app.get("/search/statutes")
def search_statutes(
    query: str = Query(..., min_length=2),
    top_k: int = Query(5, ge=1, le=20),
) -> dict[str, Any]:
    return {
        "query": query,
        "results": service.search_statutes(query, top_k=top_k),
    }


@app.post("/risk-check")
def risk_check(request: RiskCheckRequest) -> dict[str, Any]:
    return service.risk_check(request.situation, top_k=request.top_k)


def create_app(db_path: str | Path = DEFAULT_DB_PATH) -> FastAPI:
    """Factory for tests or future deployment settings."""
    global service
    service = FairCheckSearchService(db_path)
    return app
