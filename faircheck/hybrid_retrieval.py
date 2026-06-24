from __future__ import annotations

import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import chromadb


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CHROMA_DIR = PROJECT_ROOT / "data" / "indexes" / "chroma_bge"
DEFAULT_BM25_PATH = PROJECT_ROOT / "data" / "indexes" / "bm25_index.pkl"
DEFAULT_COLLECTION_NAME = "decisions_bge"

RerankerBackend = Literal["none", "bge", "qwen"]


@dataclass(frozen=True)
class HybridSearchConfig:
    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    rrf_k: int = 60
    candidate_size: int = 50
    use_metadata_for_rerank: bool = True
    doc_level: bool = True
    reranker_backend: RerankerBackend = "none"
    reranker_model: str | None = None
    rerank_batch_size: int = 8
    max_rerank_chars: int = 1800


class HybridRetrievalService:
    """BGE-M3 dense + BM25 sparse + RRF + optional reranker retrieval."""

    def __init__(
        self,
        chroma_dir: str | Path = DEFAULT_CHROMA_DIR,
        bm25_path: str | Path = DEFAULT_BM25_PATH,
        collection_name: str = DEFAULT_COLLECTION_NAME,
    ) -> None:
        self.chroma_dir = Path(chroma_dir)
        self.bm25_path = Path(bm25_path)
        self.collection_name = collection_name

        if not self.chroma_dir.exists():
            raise FileNotFoundError(f"ChromaDB directory not found: {self.chroma_dir}")
        if not self.bm25_path.exists():
            raise FileNotFoundError(f"BM25 index not found: {self.bm25_path}")

        self.client = chromadb.PersistentClient(path=str(self.chroma_dir))
        self.collection = self.client.get_collection(collection_name)

        with self.bm25_path.open("rb") as f:
            bm25_data = pickle.load(f)
        self.bm25 = bm25_data["bm25"]
        self.bm25_ids = bm25_data["ids"]
        self.bm25_texts = bm25_data.get("texts", [])

        self.embedding_model = None
        self.reranker_backend: RerankerBackend | None = None
        self.reranker_model_name: str | None = None
        self.reranker = None

    def health(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "chroma_dir": str(self.chroma_dir),
            "bm25_path": str(self.bm25_path),
            "collection_name": self.collection_name,
            "collection_count": self.collection.count(),
            "bm25_ids": len(self.bm25_ids),
            "bm25_texts": len(self.bm25_texts),
        }

    def search(
        self,
        query: str,
        top_k: int = 5,
        config: HybridSearchConfig | None = None,
    ) -> dict[str, Any]:
        config = config or HybridSearchConfig()
        timings: dict[str, float] = {}

        started = time.perf_counter()
        dense_ids = self._dense_search(query, config.candidate_size)
        timings["dense_sec"] = elapsed(started)

        started = time.perf_counter()
        bm25_top_ids = self._bm25_search(query, config.candidate_size)
        timings["bm25_sec"] = elapsed(started)

        started = time.perf_counter()
        rrf_scores, dense_rank_map, bm25_rank_map = self._rrf_fuse(
            dense_ids,
            bm25_top_ids,
            config,
        )
        rrf_sorted = sorted(rrf_scores.items(), key=lambda item: item[1], reverse=True)
        rrf_top_ids = [doc_id for doc_id, _ in rrf_sorted[: config.candidate_size]]
        candidates = self._get_candidates(rrf_top_ids)
        timings["rrf_and_fetch_sec"] = elapsed(started)

        started = time.perf_counter()
        ranked_candidates = self._rerank_or_keep(query, candidates, rrf_scores, config)
        timings["rerank_sec"] = elapsed(started)

        chunk_results = [
            format_chunk_result(item, rrf_scores, dense_rank_map, bm25_rank_map)
            for item in ranked_candidates[:top_k]
        ]
        doc_results = (
            [
                format_chunk_result(item, rrf_scores, dense_rank_map, bm25_rank_map)
                for item in aggregate_by_document(ranked_candidates, top_k)
            ]
            if config.doc_level
            else []
        )

        return {
            "query": query,
            "mode": "bge_m3_chroma_bm25_rrf",
            "reranker_backend": config.reranker_backend,
            "chunk_results": chunk_results,
            "doc_results": doc_results,
            "timings": timings,
        }

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
        tokenized_query = query.split()
        bm25_scores_all = self.bm25.get_scores(tokenized_query)
        top_bm25_idx = sorted(
            range(len(bm25_scores_all)),
            key=lambda idx: bm25_scores_all[idx],
            reverse=True,
        )[:candidate_size]
        return [self.bm25_ids[idx] for idx in top_bm25_idx]

    def _rrf_fuse(
        self,
        dense_ids: list[str],
        bm25_top_ids: list[str],
        config: HybridSearchConfig,
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

        for rank, doc_id in enumerate(bm25_top_ids, start=1):
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0.0)
                + config.bm25_weight * rrf_score(rank, config.rrf_k)
            )
            bm25_rank_map[doc_id] = rank

        return rrf_scores, dense_rank_map, bm25_rank_map

    def _get_candidates(self, ordered_ids: list[str]) -> list[dict[str, Any]]:
        candidate_results = self.collection.get(
            ids=ordered_ids,
            include=["documents", "metadatas"],
        )
        result_map = {
            doc_id: {"id": doc_id, "document": doc, "metadata": meta}
            for doc_id, doc, meta in zip(
                candidate_results["ids"],
                candidate_results["documents"],
                candidate_results["metadatas"],
            )
        }
        return [result_map[doc_id] for doc_id in ordered_ids if doc_id in result_map]

    def _rerank_or_keep(
        self,
        query: str,
        candidates: list[dict[str, Any]],
        rrf_scores: dict[str, float],
        config: HybridSearchConfig,
    ) -> list[dict[str, Any]]:
        if config.reranker_backend == "none":
            return [
                {**candidate, "rerank_score": rrf_scores.get(candidate["id"], 0.0)}
                for candidate in candidates
            ]

        reranker = self._get_reranker(config)
        rerank_inputs = [
            make_rerank_text(
                candidate["document"],
                candidate["metadata"],
                use_metadata=config.use_metadata_for_rerank,
                max_chars=config.max_rerank_chars,
            )
            for candidate in candidates
        ]

        if config.reranker_backend == "bge":
            pairs = [[query, doc] for doc in rerank_inputs]
            scores = reranker.compute_score(pairs)
        elif config.reranker_backend == "qwen":
            pairs = [(query, doc) for doc in rerank_inputs]
            scores = reranker.predict(pairs, batch_size=config.rerank_batch_size)
        else:
            raise ValueError(f"Unsupported reranker backend: {config.reranker_backend}")

        ranked = [
            {**candidate, "rerank_score": float(score)}
            for candidate, score in zip(candidates, scores)
        ]
        ranked.sort(key=lambda item: item["rerank_score"], reverse=True)
        return ranked

    def _get_embedding_model(self):
        if self.embedding_model is None:
            from FlagEmbedding import BGEM3FlagModel

            self.embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
        return self.embedding_model

    def _get_reranker(self, config: HybridSearchConfig):
        model_name = config.reranker_model or default_reranker_model(config.reranker_backend)
        if (
            self.reranker is not None
            and self.reranker_backend == config.reranker_backend
            and self.reranker_model_name == model_name
        ):
            return self.reranker

        if config.reranker_backend == "bge":
            from FlagEmbedding import FlagReranker

            self.reranker = FlagReranker(model_name, use_fp16=True)
        elif config.reranker_backend == "qwen":
            from sentence_transformers import CrossEncoder

            self.reranker = CrossEncoder(model_name)
        else:
            raise ValueError(f"Unsupported reranker backend: {config.reranker_backend}")

        self.reranker_backend = config.reranker_backend
        self.reranker_model_name = model_name
        return self.reranker


def default_reranker_model(backend: RerankerBackend) -> str | None:
    if backend == "bge":
        return "BAAI/bge-reranker-v2-m3"
    if backend == "qwen":
        return "Qwen/Qwen3-Reranker-0.6B"
    return None


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def make_rerank_text(
    doc: str,
    meta: dict[str, Any],
    use_metadata: bool = True,
    max_chars: int = 1800,
) -> str:
    doc = doc[:max_chars]
    if not use_metadata:
        return doc
    return (
        f"제목: {meta.get('의결서제목', '')}\n"
        f"위반유형: {meta.get('위반유형', '')}\n"
        f"세부위반유형: {meta.get('세부위반유형', '')}\n"
        f"업종: {meta.get('업종', '')}\n"
        f"조치유형: {meta.get('조치유형', '')}\n"
        f"피심인기업명: {meta.get('피심인기업명', '')}\n"
        f"관련법령: {meta.get('관련법령', '')}\n"
        f"본문: {doc}"
    )


def aggregate_by_document(
    ranked_candidates: list[dict[str, Any]],
    top_k: int = 5,
) -> list[dict[str, Any]]:
    doc_best: dict[str, dict[str, Any]] = {}

    for candidate in ranked_candidates:
        parent_doc_id = candidate["id"].split("-CH-")[0]
        current_best = doc_best.get(parent_doc_id)
        if current_best is None or candidate["rerank_score"] > current_best["rerank_score"]:
            doc_best[parent_doc_id] = candidate

    return sorted(
        doc_best.values(),
        key=lambda item: item["rerank_score"],
        reverse=True,
    )[:top_k]


def format_chunk_result(
    candidate: dict[str, Any],
    rrf_scores: dict[str, float],
    dense_rank_map: dict[str, int],
    bm25_rank_map: dict[str, int],
) -> dict[str, Any]:
    doc_id = candidate["id"]
    meta = candidate["metadata"]
    return {
        "chunk_id": doc_id,
        "parent_doc_id": doc_id.split("-CH-")[0],
        "decision_id": meta.get("의결서관리번호", ""),
        "decision_title": meta.get("의결서제목", ""),
        "release_date": meta.get("공개일자", ""),
        "industry": meta.get("업종", ""),
        "violation_type": meta.get("위반유형", ""),
        "specific_violation_type": meta.get("세부위반유형", ""),
        "company_name": meta.get("피심인기업명", ""),
        "related_laws": meta.get("관련법령", ""),
        "rerank_score": round(float(candidate["rerank_score"]), 6),
        "rrf_score": round(float(rrf_scores.get(doc_id, 0.0)), 6),
        "dense_rank": dense_rank_map.get(doc_id),
        "bm25_rank": bm25_rank_map.get(doc_id),
        "snippet": candidate["document"][:240].replace("\n", " "),
    }


def elapsed(started: float) -> float:
    return round(time.perf_counter() - started, 3)

