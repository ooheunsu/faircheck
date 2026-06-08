# retrieval.py
import chromadb
from sentence_transformers import SentenceTransformer, CrossEncoder
import pickle
import numpy as np

CHROMA_DIR = r"D:\faircheck\db\chroma"
BM25_PATH = r"D:\faircheck\db\bm25_index.pkl"

# 모델 로드
print("모델 로딩...")
embedding_model = SentenceTransformer("jhgan/ko-sroberta-multitask")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

# DB 로드
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("decisions")

with open(BM25_PATH, "rb") as f:
    bm25_data = pickle.load(f)
bm25 = bm25_data["bm25"]
bm25_ids = bm25_data["ids"]
bm25_texts = bm25_data["texts"]

print("로딩 완료!")

def rrf_score(rank, k=60):
    return 1.0 / (k + rank)

def hybrid_search(query, top_k=5, rrf_k=60, candidate_size=20):
    # 1. Dense 검색
    query_embedding = embedding_model.encode(query).tolist()
    dense_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=candidate_size
    )
    dense_ids = dense_results["ids"][0]
    dense_docs = dense_results["documents"][0]

    # 2. BM25 검색
    tokenized_query = query.split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:candidate_size]
    bm25_top_ids = [bm25_ids[i] for i in top_bm25_idx]

    # 3. RRF 합산
    rrf_scores = {}

    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score(rank, rrf_k)

    for rank, doc_id in enumerate(bm25_top_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + rrf_score(rank, rrf_k)

    # RRF 상위 후보 추출
    rrf_sorted = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    rrf_top_ids = [doc_id for doc_id, _ in rrf_sorted[:candidate_size]]

    # 4. 후보 텍스트 가져오기
    candidate_results = collection.get(
        ids=rrf_top_ids,
        include=["documents", "metadatas"]
    )
    candidate_docs = candidate_results["documents"]
    candidate_ids = candidate_results["ids"]

    # 5. Reranker
    pairs = [[query, doc] for doc in candidate_docs]
    rerank_scores = reranker.predict(pairs)
    rerank_sorted = sorted(
        zip(candidate_ids, candidate_docs, rerank_scores),
        key=lambda x: x[2],
        reverse=True
    )

    # Top 5 반환
    top5 = rerank_sorted[:top_k]
    result_ids = [x[0] for x in top5]
    result_docs = [x[1] for x in top5]
    result_scores = [float(x[2]) for x in top5]

    return result_ids, result_docs, result_scores


if __name__ == "__main__":
    # 테스트
    queries = [
        "건설기계 임대단가 담합",
        "배달앱 수수료 불공정거래",
        "가맹점 식자재 구매 강제",
    ]

    for query in queries:
        print(f"\n{'='*60}")
        print(f"질의: {query}")
        print(f"{'='*60}")

        ids, docs, scores = hybrid_search(query)

        for i, (doc_id, doc, score) in enumerate(zip(ids, docs, scores)):
            print(f"\n{i+1}위 (score: {score:.4f})")
            print(f"  ID: {doc_id}")
            print(f"  내용: {doc[:100]}...")