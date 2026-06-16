from __future__ import annotations

import os
import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import chromadb
from FlagEmbedding import BGEM3FlagModel


PROJECT_ROOT = Path(__file__).resolve().parent

DEFAULT_CHROMA_DIR = PROJECT_ROOT / "db" / "statute_chroma_bge"
DEFAULT_BM25_PATH = PROJECT_ROOT / "db" / "statute_bm25_index.pkl"
DEFAULT_COLLECTION_NAME = "statutes_bge"

TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
JO_RE = re.compile(r"제\s*([0-9]+(?:조의[0-9]+|조)?)")

_kiwi = None
_kiwi_checked = False


@dataclass(frozen=True)
class StatuteSearchConfig:
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    rrf_k: int = 60
    candidate_size: int = 50


class StatuteRetrievalService:
    """BGE-M3 dense + statute BM25 + RRF statute retrieval."""

    def __init__(
        self,
        chroma_dir: str | Path | None = None,
        bm25_path: str | Path | None = None,
        collection_name: str | None = None,
    ) -> None:
        load_env_file()

        self.chroma_dir = Path(
            chroma_dir
            or os.getenv("FAIRCHECK_STATUTE_CHROMA_DIR")
            or DEFAULT_CHROMA_DIR
        ).expanduser()
        self.bm25_path = Path(
            bm25_path
            or os.getenv("FAIRCHECK_STATUTE_BM25_PATH")
            or DEFAULT_BM25_PATH
        ).expanduser()
        self.collection_name = (
            collection_name
            or os.getenv("FAIRCHECK_STATUTE_COLLECTION")
            or DEFAULT_COLLECTION_NAME
        )

        if not self.chroma_dir.exists():
            raise FileNotFoundError(f"Statute ChromaDB directory not found: {self.chroma_dir}")
        if not self.bm25_path.exists():
            raise FileNotFoundError(f"Statute BM25 index not found: {self.bm25_path}")

        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_collection(self.collection_name)

        with self.bm25_path.open("rb") as f:
            bm25_data = pickle.load(f)

        self.bm25 = bm25_data["bm25"]
        self.bm25_ids = bm25_data["ids"]
        self.bm25_documents = bm25_data["documents"]
        self.bm25_metadatas = bm25_data["metadatas"]

        self.embedding_model = None

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "chroma_dir": str(self.chroma_dir),
            "bm25_path": str(self.bm25_path),
            "collection_name": self.collection_name,
            "collection_count": self.collection.count(),
            "bm25_ids": len(self.bm25_ids),
            "bm25_documents": len(self.bm25_documents),
            "bm25_metadatas": len(self.bm25_metadatas),
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        config: StatuteSearchConfig | None = None,
    ) -> dict[str, Any]:
        config = config or StatuteSearchConfig()
        timings: dict[str, float] = {}

        started = time.perf_counter()
        dense_ids = self._dense_search(query, config.candidate_size)
        timings["dense_sec"] = elapsed(started)

        started = time.perf_counter()
        bm25_ids = self._bm25_search(query, config.candidate_size)
        timings["bm25_sec"] = elapsed(started)

        started = time.perf_counter()
        rrf_scores, dense_rank_map, bm25_rank_map = self._rrf_fuse(
            dense_ids,
            bm25_ids,
            config,
        )
        rrf_sorted = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        rrf_top_ids = [doc_id for doc_id, _ in rrf_sorted[: config.candidate_size]]
        candidates = self._get_candidates(rrf_top_ids)
        timings["rrf_and_fetch_sec"] = elapsed(started)

        return {
            "query": query,
            "mode": "statute_bge_m3_chroma_bm25_rrf",
            "results": [
                format_statute_result(item, rrf_scores, dense_rank_map, bm25_rank_map)
                for item in candidates[:top_k]
            ],
            "timings": timings,
        }

    def get_by_citations(self, citations: list[str], top_k: int = 8) -> list[dict[str, Any]]:
        seen: set[str] = set()
        results: list[dict[str, Any]] = []

        for citation in citations:
            law_title = infer_law_title(citation)
            jo_number = infer_jo_number(citation)
            if not law_title or not jo_number:
                continue

            fetched = self.collection.get(
                where={"$and": [{"law_title": law_title}, {"jo_number": jo_number}]},
                include=["documents", "metadatas"],
            )
            for candidate in collection_items(fetched):
                if candidate["id"] in seen:
                    continue
                seen.add(candidate["id"])
                results.append(
                    {
                        **format_statute_result(candidate, {}, {}, {}),
                        "matched_citation": citation,
                    }
                )
                if len(results) >= top_k:
                    return results

        return results

    def _dense_search(self, query: str, candidate_size: int) -> list[str]:
        embedding_model = self._get_embedding_model()
        output = embedding_model.encode(
            [query],
            batch_size=1,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        query_embedding = output["dense_vecs"][0].tolist()

        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_size,
            include=["documents", "metadatas", "distances"],
        )
        return dense_results["ids"][0]

    def _bm25_search(self, query: str, candidate_size: int) -> list[str]:
        tokenized_query = tokenize_korean(query)
        bm25_scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(
            range(len(bm25_scores)),
            key=lambda idx: bm25_scores[idx],
            reverse=True,
        )[:candidate_size]
        return [self.bm25_ids[idx] for idx in top_indices if bm25_scores[idx] > 0]

    def _rrf_fuse(
        self,
        dense_ids: list[str],
        bm25_ids: list[str],
        config: StatuteSearchConfig,
    ) -> tuple[dict[str, float], dict[str, int], dict[str, int]]:
        rrf_scores: dict[str, float] = {}
        dense_rank_map: dict[str, int] = {}
        bm25_rank_map: dict[str, int] = {}

        for rank, doc_id in enumerate(dense_ids, start=1):
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0.0)
                + config.dense_weight * rrf_score(rank, config.rrf_k)
            )
            dense_rank_map[doc_id] = rank

        for rank, doc_id in enumerate(bm25_ids, start=1):
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0.0)
                + config.bm25_weight * rrf_score(rank, config.rrf_k)
            )
            bm25_rank_map[doc_id] = rank

        return rrf_scores, dense_rank_map, bm25_rank_map

    def _get_candidates(self, ordered_ids: list[str]) -> list[dict[str, Any]]:
        if not ordered_ids:
            return []

        candidate_results = self.collection.get(
            ids=ordered_ids,
            include=["documents", "metadatas"],
        )
        result_map = {item["id"]: item for item in collection_items(candidate_results)}
        return [result_map[doc_id] for doc_id in ordered_ids if doc_id in result_map]

    def _get_embedding_model(self):
        if self.embedding_model is None:
            self.embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        return self.embedding_model


def load_env_file(env_path: str | Path | None = None) -> None:
    path = Path(env_path) if env_path is not None else PROJECT_ROOT / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def tokenize_korean(text: str) -> list[str]:
    global _kiwi, _kiwi_checked

    try:
        if not _kiwi_checked:
            from kiwipiepy import Kiwi

            _kiwi = Kiwi()
            _kiwi_checked = True
        if _kiwi is not None:
            tokens = []
            for analyzed in _kiwi.analyze(text):
                for token in analyzed[0]:
                    if token.tag.startswith(("N", "V", "VA")):
                        tokens.append(token.form)
            return tokens
    except Exception:
        _kiwi_checked = True
        _kiwi = None

    clean = "".join(char if char.isalnum() or char.isspace() else " " for char in text)
    tokens: list[str] = []
    for word in TOKEN_RE.findall(clean.lower()):
        if len(word) < 2:
            tokens.append(word)
        else:
            tokens.extend(word[idx : idx + 2] for idx in range(len(word) - 1))
    return tokens


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def format_statute_result(
    candidate: dict[str, Any],
    rrf_scores: dict[str, float],
    dense_rank_map: dict[str, int],
    bm25_rank_map: dict[str, int],
) -> dict[str, Any]:
    doc_id = candidate["id"]
    meta = candidate["metadata"]
    document = candidate["document"]
    score = float(rrf_scores.get(doc_id, 0.0))
    return {
        "statute_id": meta.get("statute_id", doc_id),
        "law_title": meta.get("law_title", ""),
        "jo_number": meta.get("jo_number", ""),
        "jo_title": meta.get("jo_title", ""),
        "doc_type": meta.get("doc_type", "statute"),
        "score": round(score, 6),
        "rrf_score": round(score, 6),
        "dense_rank": dense_rank_map.get(doc_id),
        "bm25_rank": bm25_rank_map.get(doc_id),
        "snippet": document[:300].replace("\n", " "),
        "document": document,
    }


def collection_items(fetched: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {"id": doc_id, "document": document, "metadata": metadata}
        for doc_id, document, metadata in zip(
            fetched.get("ids", []),
            fetched.get("documents", []),
            fetched.get("metadatas", []),
        )
    ]


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


def elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 3)
