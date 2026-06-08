# retrieval_bge.py
import chromadb
from FlagEmbedding import BGEM3FlagModel, FlagReranker
import pickle

CHROMA_DIR = r"D:\faircheck\db\chroma_bge"
BM25_PATH = r"D:\faircheck\db\bm25_index.pkl"

print("모델 로딩...")
embedding_model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)
reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("decisions_bge")

with open(BM25_PATH, "rb") as f:
    bm25_data = pickle.load(f)
bm25 = bm25_data["bm25"]
bm25_ids = bm25_data["ids"]

print("로딩 완료!")

def rrf_score(rank, k=60):
    return 1.0 / (k + rank)

def hybrid_search(query, top_k=5, candidate_size=50):
    # 1. Dense 검색
    output = embedding_model.encode(
        [query],
        batch_size=1,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    query_embedding = output["dense_vecs"][0].tolist()

    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_size,
        include=["documents", "metadatas", "distances"]
    )
    dense_ids = dense_results["ids"][0]

    # 2. BM25 검색
    tokenized_query = query.split()
    bm25_scores_all = bm25.get_scores(tokenized_query)
    top_bm25_idx = sorted(
        range(len(bm25_scores_all)),
        key=lambda i: bm25_scores_all[i],
        reverse=True
    )[:candidate_size]
    bm25_top_ids = [bm25_ids[i] for i in top_bm25_idx]

    # 3. RRF 합산 (rank는 1부터 시작)
    rrf_scores = {}
    dense_rank_map = {}
    bm25_rank_map = {}

    for rank, doc_id in enumerate(dense_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score(rank)
        dense_rank_map[doc_id] = rank

    for rank, doc_id in enumerate(bm25_top_ids, start=1):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score(rank)
        bm25_rank_map[doc_id] = rank

    rrf_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    rrf_top_ids = [doc_id for doc_id, _ in rrf_sorted[:candidate_size]]

    # 4. 후보 가져오기 (순서 안전하게 map으로 처리)
    candidate_results = collection.get(
        ids=rrf_top_ids,
        include=["documents", "metadatas"]
    )

    result_map = {
        doc_id: (doc, meta)
        for doc_id, doc, meta in zip(
            candidate_results["ids"],
            candidate_results["documents"],
            candidate_results["metadatas"]
        )
    }

    candidate_ids = [doc_id for doc_id in rrf_top_ids if doc_id in result_map]
    candidate_docs = [result_map[doc_id][0] for doc_id in candidate_ids]
    candidate_metas = [result_map[doc_id][1] for doc_id in candidate_ids]

    # 5. Reranker (한국어 지원 bge-reranker-v2-m3)
    pairs = [[query, doc] for doc in candidate_docs]
    rerank_scores = reranker.compute_score(pairs)

    rerank_sorted = sorted(
        zip(candidate_ids, candidate_docs, candidate_metas, rerank_scores),
        key=lambda x: x[3],
        reverse=True
    )

    return rerank_sorted[:top_k], rrf_scores, dense_rank_map, bm25_rank_map


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

        results, rrf_scores, dense_rank_map, bm25_rank_map = hybrid_search(query)

        for i, (doc_id, doc, meta, score) in enumerate(results):
            dense_rank = dense_rank_map.get(doc_id, "-")
            bm25_rank = bm25_rank_map.get(doc_id, "-")
            rrf = rrf_scores.get(doc_id, 0)
            print(f"\n{i+1}위 (rerank: {score:.4f} | rrf: {rrf:.4f} | dense: {dense_rank}위 | bm25: {bm25_rank}위)")
            print(f"  ID: {doc_id}")
            print(f"  제목: {meta.get('의결서제목', '')}")
            print(f"  위반유형: {meta.get('위반유형', '')}")
            print(f"  업종: {meta.get('업종', '')}")
            print(f"  내용: {doc[:100]}...")