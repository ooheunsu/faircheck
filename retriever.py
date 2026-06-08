import os
import pickle
from dataclasses import dataclass, field

os.environ["HF_HOME"] = r"D:\faircheck\hf_cache\huggingface"
os.environ["TRANSFORMERS_CACHE"] = r"D:\faircheck\hf_cache\transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"D:\faircheck\hf_cache\sentence_transformers"
os.environ["TORCH_HOME"] = r"D:\faircheck\hf_cache\torch"

import chromadb
import torch
from FlagEmbedding import BGEM3FlagModel
from transformers import AutoModelForSequenceClassification, AutoTokenizer


@dataclass
class RetrieverConfig:
    chroma_dir: str = r"D:\faircheck\db\chroma_bge"
    bm25_path: str = r"D:\faircheck\db\bm25_index.pkl"
    collection_name: str = "decisions_bge"

    embedding_model_name: str = "BAAI/bge-m3"
    reranker_model_name: str = "BAAI/bge-reranker-v2-m3"

    dense_weight: float = 1.0
    bm25_weight: float = 1.0
    rrf_k: int = 60
    candidate_size: int = 50

    top_k: int = 5
    doc_top_k: int = 5
    chunks_per_doc: int = 3
    max_chunks_per_doc_in_chunk_results: int = 2

    rerank_batch_size: int = 8
    max_rerank_tokens: int = 8192
    max_rerank_chars: int | None = None

    use_fp16: bool = True
    use_metadata_for_rerank: bool = True
    use_query_rewrite: bool = True
    query_rewrites: dict[str, str] = field(
        default_factory=lambda: {
            "배달앱 수수료 불공정거래": (
                "가맹본부가 배달앱 운영사에 지급할 수수료 또는 배달비를 "
                "가맹점사업자에게 전가하거나 부담하게 한 행위"
            ),
            "병원 의약품 리베이트": (
                "제약회사가 병원 또는 의료인에게 의약품 처방 유지나 처방 증대를 목적으로 "
                "리베이트, 수수료, 금품, 향응 등 경제적 이익을 제공한 부당한 고객유인행위"
            ),
        }
    )


class FaircheckRetriever:
    def __init__(self, config: RetrieverConfig | None = None):
        self.config = config or RetrieverConfig()

        print("검색기 로딩...")
        self.embedding_model = BGEM3FlagModel(
            self.config.embedding_model_name,
            use_fp16=self.config.use_fp16,
        )

        self.reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
        self.reranker_tokenizer = AutoTokenizer.from_pretrained(
            self.config.reranker_model_name,
            use_fast=True,
        )
        self.reranker = AutoModelForSequenceClassification.from_pretrained(
            self.config.reranker_model_name,
            trust_remote_code=True,
        )
        self.reranker.to(self.reranker_device)

        if self.reranker_device == "cuda" and self.config.use_fp16:
            self.reranker.half()

        self.reranker.eval()
        print(f"BGE reranker device: {self.reranker_device}")

        client = chromadb.PersistentClient(path=self.config.chroma_dir)
        self.collection = client.get_collection(self.config.collection_name)

        with open(self.config.bm25_path, "rb") as f:
            bm25_data = pickle.load(f)

        self.bm25 = bm25_data["bm25"]
        self.bm25_ids = bm25_data["ids"]
        print("검색기 로딩 완료")

    def search(self, query: str, top_k: int | None = None, candidate_size: int | None = None):
        top_k = top_k or self.config.top_k
        candidate_size = candidate_size or self.config.candidate_size
        search_query = self.rewrite_query(query)

        query_embedding = self.encode_query(search_query)
        dense_ids = self.search_dense(query_embedding, candidate_size)
        bm25_top_ids = self.search_bm25(search_query, candidate_size)

        rrf_scores, dense_rank_map, bm25_rank_map = self.merge_with_rrf(
            dense_ids,
            bm25_top_ids,
        )

        rrf_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        rrf_top_ids = [doc_id for doc_id, _ in rrf_sorted[:candidate_size]]

        candidate_ids, candidate_docs, candidate_metas = self.load_candidates(rrf_top_ids)
        rerank_inputs = [
            self.make_rerank_text(doc, meta)
            for doc, meta in zip(candidate_docs, candidate_metas)
        ]
        rerank_scores = self.rerank_score(search_query, rerank_inputs)

        rerank_sorted = sorted(
            zip(candidate_ids, candidate_docs, candidate_metas, rerank_scores),
            key=lambda x: x[3],
            reverse=True,
        )

        chunk_results = self.limit_chunks_per_document(
            rerank_sorted,
            top_k=top_k,
            max_chunks_per_doc=self.config.max_chunks_per_doc_in_chunk_results,
        )
        doc_results = self.aggregate_by_document(
            rerank_sorted,
            top_k=self.config.doc_top_k,
        )
        doc_contexts = self.build_document_contexts(
            rerank_sorted,
            top_k=self.config.doc_top_k,
            chunks_per_doc=self.config.chunks_per_doc,
        )

        return {
            "query": query,
            "search_query": search_query,
            "chunk_results": chunk_results,
            "doc_results": doc_results,
            "doc_contexts": doc_contexts,
            "rrf_scores": rrf_scores,
            "dense_rank_map": dense_rank_map,
            "bm25_rank_map": bm25_rank_map,
        }

    def rewrite_query(self, query: str) -> str:
        if not self.config.use_query_rewrite:
            return query

        return self.config.query_rewrites.get(query, query)

    def encode_query(self, query: str):
        output = self.embedding_model.encode(
            [query],
            batch_size=1,
            max_length=8192,
            return_dense=True,
            return_sparse=False,
            return_colbert_vecs=False,
        )
        return output["dense_vecs"][0].tolist()

    def search_dense(self, query_embedding, candidate_size: int) -> list[str]:
        dense_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=candidate_size,
            include=["documents", "metadatas", "distances"],
        )
        return dense_results["ids"][0]

    def search_bm25(self, query: str, candidate_size: int) -> list[str]:
        tokenized_query = query.split()
        bm25_scores_all = self.bm25.get_scores(tokenized_query)

        top_bm25_idx = sorted(
            range(len(bm25_scores_all)),
            key=lambda i: bm25_scores_all[i],
            reverse=True,
        )[:candidate_size]
        return [self.bm25_ids[i] for i in top_bm25_idx]

    def merge_with_rrf(self, dense_ids: list[str], bm25_top_ids: list[str]):
        rrf_scores = {}
        dense_rank_map = {}
        bm25_rank_map = {}

        for rank, doc_id in enumerate(dense_ids, start=1):
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0)
                + self.config.dense_weight * self.rrf_score(rank)
            )
            dense_rank_map[doc_id] = rank

        for rank, doc_id in enumerate(bm25_top_ids, start=1):
            rrf_scores[doc_id] = (
                rrf_scores.get(doc_id, 0)
                + self.config.bm25_weight * self.rrf_score(rank)
            )
            bm25_rank_map[doc_id] = rank

        return rrf_scores, dense_rank_map, bm25_rank_map

    def load_candidates(self, candidate_ids: list[str]):
        candidate_results = self.collection.get(
            ids=candidate_ids,
            include=["documents", "metadatas"],
        )

        result_map = {
            doc_id: (doc, meta)
            for doc_id, doc, meta in zip(
                candidate_results["ids"],
                candidate_results["documents"],
                candidate_results["metadatas"],
            )
        }

        ordered_ids = [doc_id for doc_id in candidate_ids if doc_id in result_map]
        docs = [result_map[doc_id][0] for doc_id in ordered_ids]
        metas = [result_map[doc_id][1] for doc_id in ordered_ids]
        return ordered_ids, docs, metas

    def make_rerank_text(self, doc: str, meta: dict) -> str:
        doc = self.trim_rerank_doc(doc)

        if not self.config.use_metadata_for_rerank:
            return doc

        return (
            f"제목: {meta.get('의결서제목', '')}\n"
            f"위반유형: {meta.get('위반유형', '')}\n"
            f"업종: {meta.get('업종', '')}\n"
            f"조치유형: {meta.get('조치유형', '')}\n"
            f"피심인기업명: {meta.get('피심인기업명', '')}\n"
            f"본문: {doc}"
        )

    def trim_rerank_doc(self, doc: str) -> str:
        if self.config.max_rerank_chars is None:
            return doc

        return doc[:self.config.max_rerank_chars]

    def rerank_score(self, query: str, documents: list[str]) -> list[float]:
        scores = []

        for start in range(0, len(documents), self.config.rerank_batch_size):
            batch_docs = documents[start:start + self.config.rerank_batch_size]
            batch_queries = [query] * len(batch_docs)

            inputs = self.reranker_tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self.config.max_rerank_tokens,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(self.reranker_device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = self.reranker(**inputs)
                logits = outputs.logits

                if logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1)
                else:
                    batch_scores = logits[:, 1]

            scores.extend(batch_scores.float().cpu().tolist())

        return scores

    def limit_chunks_per_document(self, rerank_sorted, top_k=5, max_chunks_per_doc=2):
        selected = []
        doc_counts = {}

        for item in rerank_sorted:
            doc_id = item[0]
            pid = self.parent_doc_id(doc_id)

            if doc_counts.get(pid, 0) >= max_chunks_per_doc:
                continue

            selected.append(item)
            doc_counts[pid] = doc_counts.get(pid, 0) + 1

            if len(selected) >= top_k:
                break

        return selected

    def aggregate_by_document(self, rerank_sorted, top_k=5):
        doc_best = {}

        for doc_id, doc, meta, score in rerank_sorted:
            pid = self.parent_doc_id(doc_id)

            if pid not in doc_best or score > doc_best[pid][3]:
                doc_best[pid] = (doc_id, doc, meta, score)

        return sorted(
            doc_best.values(),
            key=lambda x: x[3],
            reverse=True,
        )[:top_k]

    def build_document_contexts(self, rerank_sorted, top_k=5, chunks_per_doc=3):
        doc_contexts = []
        seen_doc_ids = set()

        for doc_id, doc, meta, score in rerank_sorted:
            pid = self.parent_doc_id(doc_id)

            if pid in seen_doc_ids:
                continue

            chunks = [
                {
                    "doc_id": chunk_doc_id,
                    "doc": chunk_doc,
                    "meta": chunk_meta,
                    "score": chunk_score,
                }
                for chunk_doc_id, chunk_doc, chunk_meta, chunk_score in rerank_sorted
                if self.parent_doc_id(chunk_doc_id) == pid
            ][:chunks_per_doc]

            doc_contexts.append(
                {
                    "parent_doc_id": pid,
                    "best_doc_id": doc_id,
                    "meta": meta,
                    "best_score": score,
                    "chunks": chunks,
                }
            )
            seen_doc_ids.add(pid)

            if len(doc_contexts) >= top_k:
                break

        return doc_contexts

    def rrf_score(self, rank: int) -> float:
        return 1.0 / (self.config.rrf_k + rank)

    @staticmethod
    def parent_doc_id(doc_id: str) -> str:
        return doc_id.split("-CH-")[0]
