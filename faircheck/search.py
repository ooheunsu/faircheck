from __future__ import annotations

import json
import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "faircheck.db"

TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
JO_RE = re.compile(r"제\s*([0-9]+(?:조의[0-9]+|조)?)")


@dataclass(frozen=True)
class SearchConfig:
    chunk_candidate_limit: int = 300
    max_chunks_per_decision: int = 3


class FairCheckSearchService:
    """Small SQLite baseline for FairCheck.

    This is intentionally dependency-light. It gives the FastAPI layer a stable
    interface while the heavier BGE-M3/Chroma/BM25 pipeline is being prepared.
    """

    def __init__(self, db_path: str | Path = DEFAULT_DB_PATH) -> None:
        self.db_path = Path(db_path)
        if not self.db_path.exists():
            raise FileNotFoundError(f"SQLite DB not found: {self.db_path}")

    def health(self) -> dict[str, Any]:
        with self._connect() as conn:
            decision_chunks = conn.execute("select count(*) from decisions").fetchone()[0]
            decisions = conn.execute(
                "select count(distinct decision_id) from decisions"
            ).fetchone()[0]
            statutes = conn.execute("select count(*) from statutes").fetchone()[0]

        return {
            "status": "ok",
            "db_path": str(self.db_path),
            "decision_chunks": decision_chunks,
            "decisions": decisions,
            "statutes": statutes,
        }

    def search_decisions(
        self,
        query: str,
        top_k: int = 5,
        config: SearchConfig | None = None,
    ) -> list[dict[str, Any]]:
        tokens = tokenize(query)
        if not tokens:
            return []

        config = config or SearchConfig()
        rows = self._find_candidate_chunks(tokens, config.chunk_candidate_limit)
        scored_chunks = [
            {**row, "score": score_chunk(row, query, tokens), "snippet": make_snippet(row["page_content"], tokens)}
            for row in rows
        ]
        scored_chunks = [row for row in scored_chunks if row["score"] > 0]
        scored_chunks.sort(key=lambda row: row["score"], reverse=True)

        grouped: dict[str, dict[str, Any]] = {}
        score_sums: defaultdict[str, float] = defaultdict(float)

        for chunk in scored_chunks:
            decision_id = chunk["decision_id"] or chunk["decision_title"]
            score_sums[decision_id] += chunk["score"]

            if decision_id not in grouped:
                grouped[decision_id] = {
                    "decision_id": decision_id,
                    "decision_title": chunk["decision_title"],
                    "release_date": chunk["release_date"],
                    "original_filename": chunk["original_filename"],
                    "industry": chunk["industry"],
                    "violation_actions": parse_json_list(chunk["violation_actions"]),
                    "related_laws": parse_json_list(chunk["related_laws"]),
                    "score": chunk["score"],
                    "matched_chunk_count": 0,
                    "top_chunks": [],
                }

            item = grouped[decision_id]
            item["matched_chunk_count"] += 1
            item["score"] = max(item["score"], chunk["score"])
            if len(item["top_chunks"]) < config.max_chunks_per_decision:
                item["top_chunks"].append(
                    {
                        "chunk_id": chunk["chunk_id"],
                        "chunk_header": chunk["chunk_header"],
                        "chunk_section": chunk["chunk_section"],
                        "chunk_index": chunk["chunk_index"],
                        "score": round(chunk["score"], 3),
                        "snippet": chunk["snippet"],
                    }
                )

        results = list(grouped.values())
        for item in results:
            item["score"] = round(item["score"] + (score_sums[item["decision_id"]] * 0.03), 3)

        results.sort(key=lambda item: item["score"], reverse=True)
        return results[:top_k]

    def get_decision(self, decision_id: str) -> dict[str, Any] | None:
        with self._connect() as conn:
            rows = conn.execute(
                """
                select chunk_id, page_content, decision_title, decision_id, release_date,
                       original_filename, industry, violation_actions, related_laws,
                       chunk_header, chunk_section, chunk_index, total_chunks
                from decisions
                where decision_id = ?
                order by chunk_index
                """,
                (decision_id,),
            ).fetchall()

        if not rows:
            return None

        first = dict(rows[0])
        return {
            "decision_id": first["decision_id"],
            "decision_title": first["decision_title"],
            "release_date": first["release_date"],
            "original_filename": first["original_filename"],
            "industry": first["industry"],
            "violation_actions": parse_json_list(first["violation_actions"]),
            "related_laws": parse_json_list(first["related_laws"]),
            "chunks": [
                {
                    "chunk_id": row["chunk_id"],
                    "chunk_header": row["chunk_header"],
                    "chunk_section": row["chunk_section"],
                    "chunk_index": row["chunk_index"],
                    "total_chunks": row["total_chunks"],
                    "page_content": row["page_content"],
                }
                for row in rows
            ],
        }

    def search_statutes(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        tokens = tokenize(query)
        if not tokens:
            return []

        where = " or ".join(["search_text like ?" for _ in tokens])
        params = [f"%{token}%" for token in tokens]
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select statute_id, law_title, jo_number, jo_title, page_content,
                       lower(coalesce(law_title, '') || ' ' || coalesce(jo_number, '') || ' ' ||
                             coalesce(jo_title, '') || ' ' || coalesce(page_content, '')) as search_text
                from statutes
                where {where}
                limit 200
                """,
                params,
            ).fetchall()

        results = []
        for row in rows:
            item = dict(row)
            item["score"] = score_statute(item, query, tokens)
            item["snippet"] = make_snippet(item["page_content"], tokens)
            item.pop("search_text", None)
            if item["score"] > 0:
                results.append(item)

        results.sort(key=lambda item: item["score"], reverse=True)
        for item in results:
            item["score"] = round(item["score"], 3)
        return results[:top_k]

    def statutes_for_laws(self, related_laws: list[str], top_k: int = 8) -> list[dict[str, Any]]:
        seen: set[str] = set()
        results: list[dict[str, Any]] = []

        with self._connect() as conn:
            for citation in related_laws:
                law_title = infer_law_title(citation)
                jo_number = infer_jo_number(citation)
                if not law_title or not jo_number:
                    continue

                rows = conn.execute(
                    """
                    select statute_id, law_title, jo_number, jo_title, page_content
                    from statutes
                    where law_title = ? and jo_number = ?
                    """,
                    (law_title, jo_number),
                ).fetchall()

                for row in rows:
                    if row["statute_id"] in seen:
                        continue
                    seen.add(row["statute_id"])
                    results.append({**dict(row), "matched_citation": citation})
                    if len(results) >= top_k:
                        return results

        return results

    def risk_check(self, situation: str, top_k: int = 5) -> dict[str, Any]:
        decisions = self.search_decisions(situation, top_k=top_k)
        laws: list[str] = []
        for decision in decisions:
            for law in decision["related_laws"]:
                if law not in laws:
                    laws.append(law)

        statutes = self.statutes_for_laws(laws)
        risk_level = estimate_risk_level(decisions)

        return {
            "query": situation,
            "risk_level": risk_level,
            "message": build_risk_message(risk_level),
            "matched_decisions": decisions,
            "related_statutes": statutes,
            "disclaimer": "이 결과는 유사 의결서와 법령 근거를 찾는 참고용 경고이며, 법률 위반 확정 판단이 아닙니다.",
        }

    def _find_candidate_chunks(self, tokens: list[str], limit: int) -> list[dict[str, Any]]:
        where = " or ".join(["search_text like ?" for _ in tokens])
        params = [f"%{token}%" for token in tokens]
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                select chunk_id, page_content, decision_title, decision_id, release_date,
                       original_filename, industry, violation_actions, related_laws,
                       chunk_header, chunk_section, chunk_index, total_chunks,
                       lower(coalesce(decision_title, '') || ' ' || coalesce(industry, '') || ' ' ||
                             coalesce(violation_actions, '') || ' ' || coalesce(related_laws, '') || ' ' ||
                             coalesce(chunk_header, '') || ' ' || coalesce(chunk_section, '') || ' ' ||
                             coalesce(page_content, '')) as search_text
                from decisions
                where {where}
                limit ?
                """,
                [*params, limit],
            ).fetchall()
        return [dict(row) for row in rows]

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(f"file:{self.db_path}?mode=ro", uri=True)
        conn.row_factory = sqlite3.Row
        return conn


def tokenize(text: str) -> list[str]:
    tokens = []
    for token in TOKEN_RE.findall(text.lower()):
        if len(token) >= 2 and token not in {"관련", "위반", "행위", "대한", "경우"}:
            tokens.append(token)
    return list(dict.fromkeys(tokens))


def parse_json_list(value: str | None) -> list[str]:
    if not value:
        return []
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return [value]
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    return [str(parsed)]


def score_chunk(row: dict[str, Any], query: str, tokens: list[str]) -> float:
    fields = {
        "decision_title": 8.0,
        "industry": 6.0,
        "violation_actions": 6.0,
        "related_laws": 4.0,
        "chunk_header": 2.0,
        "chunk_section": 1.5,
        "page_content": 1.0,
    }
    score = 0.0
    lowered_query = query.lower()
    for field, weight in fields.items():
        text = str(row.get(field) or "").lower()
        for token in tokens:
            score += min(text.count(token), 3) * weight
        if lowered_query and lowered_query in text:
            score += weight * 4

    search_text = str(row.get("search_text") or "").lower()
    matched_tokens = sum(1 for token in tokens if token in search_text)
    if matched_tokens == len(tokens):
        score += 25
    coverage = matched_tokens / max(len(tokens), 1)
    return score * (0.35 + 0.65 * coverage)


def score_statute(row: dict[str, Any], query: str, tokens: list[str]) -> float:
    fields = {
        "law_title": 6.0,
        "jo_number": 5.0,
        "jo_title": 4.0,
        "page_content": 1.0,
    }
    score = 0.0
    lowered_query = query.lower()
    for field, weight in fields.items():
        text = str(row.get(field) or "").lower()
        for token in tokens:
            score += min(text.count(token), 3) * weight
        if lowered_query and lowered_query in text:
            score += weight * 4
    return score


def make_snippet(text: str | None, tokens: list[str], window: int = 180) -> str:
    if not text:
        return ""
    lowered = text.lower()
    positions = [lowered.find(token) for token in tokens if lowered.find(token) >= 0]
    start = max(min(positions) - 40, 0) if positions else 0
    snippet = text[start : start + window].replace("\n", " ").strip()
    if start > 0:
        snippet = "..." + snippet
    if start + window < len(text):
        snippet += "..."
    return snippet


def infer_law_title(citation: str) -> str | None:
    known_titles = [
        "독점규제 및 공정거래에 관한 법률",
        "가맹사업거래의 공정화에 관한 법률",
        "하도급거래 공정화에 관한 법률",
    ]
    for title in known_titles:
        if title in citation:
            return title
    return None


def infer_jo_number(citation: str) -> str | None:
    match = JO_RE.search(citation)
    if not match:
        return None
    jo = match.group(1)
    return jo if jo.endswith("조") or "조의" in jo else f"{jo}조"


def estimate_risk_level(decisions: list[dict[str, Any]]) -> str:
    if not decisions:
        return "low"
    top_score = decisions[0]["score"]
    if top_score >= 80 or len(decisions) >= 4:
        return "high"
    if top_score >= 25 or len(decisions) >= 2:
        return "medium"
    return "low"


def build_risk_message(risk_level: str) -> str:
    messages = {
        "high": "유사 의결서가 여러 건 또는 강하게 검색되어 공정거래 리스크 검토가 필요합니다.",
        "medium": "일부 유사 의결서가 검색되어 계약/거래 조건을 점검하는 것이 좋습니다.",
        "low": "강하게 일치하는 의결서는 적지만, 입력이 짧으면 더 구체적으로 다시 검색해 보세요.",
    }
    return messages[risk_level]
