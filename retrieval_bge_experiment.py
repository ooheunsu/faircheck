# retrieval_bge_experiment.py
import os

os.environ["HF_HOME"] = r"D:\faircheck\hf_cache\huggingface"
os.environ["TRANSFORMERS_CACHE"] = r"D:\faircheck\hf_cache\transformers"
os.environ["SENTENCE_TRANSFORMERS_HOME"] = r"D:\faircheck\hf_cache\sentence_transformers"
os.environ["TORCH_HOME"] = r"D:\faircheck\hf_cache\torch"

import pickle

import chromadb
from FlagEmbedding import BGEM3FlagModel


CHROMA_DIR = r"D:\faircheck\db\chroma_bge"
BM25_PATH = r"D:\faircheck\db\bm25_index.pkl"
COLLECTION_NAME = "decisions_bge"

DENSE_WEIGHT = 1.0
BM25_WEIGHT = 1.0
RRF_K = 60
CANDIDATE_SIZE = 50

TOP_K = 5
DOC_TOP_K = 5
CHUNKS_PER_DOC = 3
MAX_CHUNKS_PER_DOC_IN_CHUNK_RESULTS = 2

RERANK_BATCH_SIZE = 8
MAX_RERANK_TOKENS = 8192

USE_METADATA_FOR_RERANK = True
USE_QUERY_REWRITE = True
SHOW_CHUNK_LEVEL = True
SHOW_DOC_LEVEL = True
SHOW_DOC_CONTEXT = True

# BGE reranker는 전체 본문으로도 충분히 빠른 편이라 기본값은 None.
# 너무 느리거나 API reranker 비교 시에는 3000~5000 정도로 바꿔서 테스트.
MAX_RERANK_CHARS = None

RERANKER_BACKEND = "bge"
# "bge", "qwen", "cohere", "voyage" 중 선택

RERANKER_MODEL = {
    "bge": "BAAI/bge-reranker-v2-m3",
    "qwen": "Qwen/Qwen3-Reranker-0.6B",
    "cohere": "rerank-v3.5",
    "voyage": "rerank-2",
}[RERANKER_BACKEND]


QUERY_REWRITES = {
    "배달앱 수수료 불공정거래": (
        "가맹본부가 배달앱 운영사에 지급할 수수료 또는 배달비를 "
        "가맹점사업자에게 전가하거나 부담하게 한 행위"
    ),
    "병원 의약품 리베이트": (
        "제약회사가 병원 또는 의료인에게 의약품 처방 유지나 처방 증대를 목적으로 "
        "리베이트, 수수료, 금품, 향응 등 경제적 이익을 제공한 부당한 고객유인행위"
    ),
}


print("모델 로딩...")
embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

reranker = None
reranker_tokenizer = None
reranker_device = None
co = None
vo = None

if RERANKER_BACKEND == "bge":
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    reranker_device = "cuda" if torch.cuda.is_available() else "cpu"
    reranker_tokenizer = AutoTokenizer.from_pretrained(
        RERANKER_MODEL,
        use_fast=True,
    )
    reranker = AutoModelForSequenceClassification.from_pretrained(
        RERANKER_MODEL,
        trust_remote_code=True,
    )
    reranker.to(reranker_device)

    if reranker_device == "cuda":
        reranker.half()

    reranker.eval()
    print(f"BGE reranker device: {reranker_device}")

elif RERANKER_BACKEND == "qwen":
    from sentence_transformers import CrossEncoder

    reranker = CrossEncoder(RERANKER_MODEL)

elif RERANKER_BACKEND == "cohere":
    import cohere

    co = cohere.ClientV2()

elif RERANKER_BACKEND == "voyage":
    import voyageai

    vo = voyageai.Client()

else:
    raise ValueError(f"Unknown reranker backend: {RERANKER_BACKEND}")

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(COLLECTION_NAME)

with open(BM25_PATH, "rb") as f:
    bm25_data = pickle.load(f)

bm25 = bm25_data["bm25"]
bm25_ids = bm25_data["ids"]

print("로딩 완료!")
print(
    f"설정: dense_weight={DENSE_WEIGHT}, bm25_weight={BM25_WEIGHT}, "
    f"candidate_size={CANDIDATE_SIZE}, metadata_rerank={USE_METADATA_FOR_RERANK}, "
    f"query_rewrite={USE_QUERY_REWRITE}, doc_level={SHOW_DOC_LEVEL}, "
    f"doc_context={SHOW_DOC_CONTEXT}, reranker={RERANKER_BACKEND}:{RERANKER_MODEL}"
)


def rewrite_query(query):
    if not USE_QUERY_REWRITE:
        return query

    return QUERY_REWRITES.get(query, query)


def rrf_score(rank, k=60):
    return 1.0 / (k + rank)


def parent_doc_id(doc_id):
    return doc_id.split("-CH-")[0]


def trim_rerank_doc(doc):
    if MAX_RERANK_CHARS is None:
        return doc

    return doc[:MAX_RERANK_CHARS]


def make_rerank_text(doc, meta):
    doc = trim_rerank_doc(doc)

    if not USE_METADATA_FOR_RERANK:
        return doc

    return (
        f"제목: {meta.get('의결서제목', '')}\n"
        f"위반유형: {meta.get('위반유형', '')}\n"
        f"업종: {meta.get('업종', '')}\n"
        f"조치유형: {meta.get('조치유형', '')}\n"
        f"피심인기업명: {meta.get('피심인기업명', '')}\n"
        f"본문: {doc}"
    )


def as_float_list(scores):
    if isinstance(scores, (int, float)):
        return [float(scores)]

    return [float(score) for score in scores]


def rerank_score(query, documents):
    if RERANKER_BACKEND == "bge":
        scores = []

        for start in range(0, len(documents), RERANK_BATCH_SIZE):
            batch_docs = documents[start:start + RERANK_BATCH_SIZE]
            batch_queries = [query] * len(batch_docs)

            inputs = reranker_tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=MAX_RERANK_TOKENS,
                return_tensors="pt",
            )
            inputs = {
                key: value.to(reranker_device)
                for key, value in inputs.items()
            }

            with torch.no_grad():
                outputs = reranker(**inputs)
                logits = outputs.logits

                if logits.shape[-1] == 1:
                    batch_scores = logits.squeeze(-1)
                else:
                    batch_scores = logits[:, 1]

            scores.extend(batch_scores.float().cpu().tolist())

        return scores

    if RERANKER_BACKEND == "qwen":
        pairs = [(query, doc) for doc in documents]
        print(f"Qwen rerank 시작: {len(pairs)}개 후보")
        scores = reranker.predict(
            pairs,
            batch_size=1,
            show_progress_bar=True,
        )
        print("Qwen rerank 완료")
        return as_float_list(scores)

    if RERANKER_BACKEND == "cohere":
        response = co.rerank(
            model=RERANKER_MODEL,
            query=query,
            documents=documents,
            top_n=len(documents),
        )

        scores = [None] * len(documents)
        for item in response.results:
            scores[item.index] = float(item.relevance_score)
        return scores

    if RERANKER_BACKEND == "voyage":
        response = vo.rerank(
            query=query,
            documents=documents,
            model=RERANKER_MODEL,
            top_k=len(documents),
            truncation=True,
        )

        scores = [None] * len(documents)
        for item in response.results:
            score = getattr(item, "relevance_score", None)
            if score is None:
                score = getattr(item, "score")
            scores[item.index] = float(score)
        return scores

    raise ValueError(f"Unknown reranker backend: {RERANKER_BACKEND}")


def limit_chunks_per_document(rerank_sorted, top_k=5, max_chunks_per_doc=2):
    selected = []
    doc_counts = {}

    for item in rerank_sorted:
        doc_id = item[0]
        pid = parent_doc_id(doc_id)

        if doc_counts.get(pid, 0) >= max_chunks_per_doc:
            continue

        selected.append(item)
        doc_counts[pid] = doc_counts.get(pid, 0) + 1

        if len(selected) >= top_k:
            break

    return selected


def aggregate_by_document(rerank_sorted, top_k=5):
    doc_best = {}

    for doc_id, doc, meta, score in rerank_sorted:
        pid = parent_doc_id(doc_id)

        if pid not in doc_best or score > doc_best[pid][3]:
            doc_best[pid] = (doc_id, doc, meta, score)

    return sorted(
        doc_best.values(),
        key=lambda x: x[3],
        reverse=True,
    )[:top_k]


def build_document_contexts(rerank_sorted, top_k=5, chunks_per_doc=3):
    doc_contexts = []
    seen_doc_ids = set()

    for doc_id, doc, meta, score in rerank_sorted:
        pid = parent_doc_id(doc_id)

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
            if parent_doc_id(chunk_doc_id) == pid
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


def hybrid_search(query, top_k=5, candidate_size=50):
    search_query = rewrite_query(query)

    output = embedding_model.encode(
        [search_query],
        batch_size=1,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False,
    )
    query_embedding = output["dense_vecs"][0].tolist()

    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_size,
        include=["documents", "metadatas", "distances"],
    )
    dense_ids = dense_results["ids"][0]

    tokenized_query = search_query.split()
    bm25_scores_all = bm25.get_scores(tokenized_query)

    top_bm25_idx = sorted(
        range(len(bm25_scores_all)),
        key=lambda i: bm25_scores_all[i],
        reverse=True,
    )[:candidate_size]
    bm25_top_ids = [bm25_ids[i] for i in top_bm25_idx]

    rrf_scores = {}
    dense_rank_map = {}
    bm25_rank_map = {}

    for rank, doc_id in enumerate(dense_ids, start=1):
        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + DENSE_WEIGHT * rrf_score(rank, RRF_K)
        )
        dense_rank_map[doc_id] = rank

    for rank, doc_id in enumerate(bm25_top_ids, start=1):
        rrf_scores[doc_id] = (
            rrf_scores.get(doc_id, 0)
            + BM25_WEIGHT * rrf_score(rank, RRF_K)
        )
        bm25_rank_map[doc_id] = rank

    rrf_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    rrf_top_ids = [doc_id for doc_id, _ in rrf_sorted[:candidate_size]]

    candidate_results = collection.get(
        ids=rrf_top_ids,
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

    candidate_ids = [doc_id for doc_id in rrf_top_ids if doc_id in result_map]
    candidate_docs = [result_map[doc_id][0] for doc_id in candidate_ids]
    candidate_metas = [result_map[doc_id][1] for doc_id in candidate_ids]

    rerank_inputs = [
        make_rerank_text(doc, meta)
        for doc, meta in zip(candidate_docs, candidate_metas)
    ]

    rerank_scores = rerank_score(search_query, rerank_inputs)

    rerank_sorted = sorted(
        zip(candidate_ids, candidate_docs, candidate_metas, rerank_scores),
        key=lambda x: x[3],
        reverse=True,
    )

    chunk_results = limit_chunks_per_document(
        rerank_sorted,
        top_k=top_k,
        max_chunks_per_doc=MAX_CHUNKS_PER_DOC_IN_CHUNK_RESULTS,
    )
    doc_results = aggregate_by_document(rerank_sorted, DOC_TOP_K)
    doc_contexts = build_document_contexts(
        rerank_sorted,
        top_k=DOC_TOP_K,
        chunks_per_doc=CHUNKS_PER_DOC,
    )

    return (
        chunk_results,
        doc_results,
        doc_contexts,
        rrf_scores,
        dense_rank_map,
        bm25_rank_map,
        search_query,
    )


def print_results(title, results, rrf_scores, dense_rank_map, bm25_rank_map):
    print(f"\n[{title}]")

    for i, (doc_id, doc, meta, score) in enumerate(results):
        dense_rank = dense_rank_map.get(doc_id, "-")
        bm25_rank = bm25_rank_map.get(doc_id, "-")
        rrf = rrf_scores.get(doc_id, 0)

        print(
            f"\n{i+1}위 "
            f"(rerank: {score:.4f} | rrf: {rrf:.4f} | "
            f"dense: {dense_rank}위 | bm25: {bm25_rank}위)"
        )
        print(f"  ID: {doc_id}")
        print(f"  제목: {meta.get('의결서제목', '')}")
        print(f"  위반유형: {meta.get('위반유형', '')}")
        print(f"  업종: {meta.get('업종', '')}")
        print(f"  내용: {doc[:120]}...")


def print_document_contexts(title, doc_contexts, rrf_scores, dense_rank_map, bm25_rank_map):
    print(f"\n[{title}]")

    for i, context in enumerate(doc_contexts):
        meta = context["meta"]
        print(f"\n{i+1}위 문서 (best rerank: {context['best_score']:.4f})")
        print(f"  대표 ID: {context['best_doc_id']}")
        print(f"  제목: {meta.get('의결서제목', '')}")
        print(f"  위반유형: {meta.get('위반유형', '')}")
        print(f"  업종: {meta.get('업종', '')}")

        for j, chunk in enumerate(context["chunks"], start=1):
            doc_id = chunk["doc_id"]
            dense_rank = dense_rank_map.get(doc_id, "-")
            bm25_rank = bm25_rank_map.get(doc_id, "-")
            rrf = rrf_scores.get(doc_id, 0)

            print(
                f"    chunk {j}: {doc_id} "
                f"(rerank: {chunk['score']:.4f} | rrf: {rrf:.4f} | "
                f"dense: {dense_rank}위 | bm25: {bm25_rank}위)"
            )
            print(f"      내용: {chunk['doc'][:120]}...")


if __name__ == "__main__":
    queries = [
        "건설기계 임대단가 담합",
        "배달앱 수수료 불공정거래",
        "가맹점 식자재 구매 강제",
        "아파트 분양 담합",
        "병원 의약품 리베이트",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"질의: {query}")
        print(f"{'='*60}")

        (
            chunk_results,
            doc_results,
            doc_contexts,
            rrf_scores,
            dense_rank_map,
            bm25_rank_map,
            search_query,
        ) = hybrid_search(query, top_k=TOP_K, candidate_size=CANDIDATE_SIZE)

        if search_query != query:
            print(f"검색 질의 재작성: {search_query}")

        if SHOW_CHUNK_LEVEL:
            print_results(
                "Chunk 단위 결과",
                chunk_results,
                rrf_scores,
                dense_rank_map,
                bm25_rank_map,
            )

        if SHOW_DOC_LEVEL:
            print_results(
                "문서 단위 결과",
                doc_results,
                rrf_scores,
                dense_rank_map,
                bm25_rank_map,
            )

        if SHOW_DOC_CONTEXT:
            print_document_contexts(
                "문서별 RAG context 후보",
                doc_contexts,
                rrf_scores,
                dense_rank_map,
                bm25_rank_map,
            )