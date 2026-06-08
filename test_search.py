# test_search.py
import chromadb
from sentence_transformers import SentenceTransformer
import pickle

CHROMA_DIR = r"D:\faircheck\db\chroma"
BM25_PATH = r"D:\faircheck\db\bm25_index.pkl"

# ChromaDB 로드
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("decisions")
print(f"ChromaDB 총 chunk 수: {collection.count()}")

# 임베딩 모델
model = SentenceTransformer("jhgan/ko-sroberta-multitask")

# BM25 로드
with open(BM25_PATH, "rb") as f:
    bm25_data = pickle.load(f)
bm25 = bm25_data["bm25"]
ids = bm25_data["ids"]

# 테스트 질의
query = "건설기계 임대단가 담합"

# Dense 검색
query_embedding = model.encode(query).tolist()
dense_results = collection.query(
    query_embeddings=[query_embedding],
    n_results=5
)
print("\n[Dense 검색 결과]")
for i, (doc_id, doc) in enumerate(zip(
    dense_results["ids"][0],
    dense_results["documents"][0]
)):
    print(f"{i+1}. {doc_id}")
    print(f"   {doc[:80]}...")

# BM25 검색
tokenized_query = query.split()
bm25_scores = bm25.get_scores(tokenized_query)
top5_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]
print("\n[BM25 검색 결과]")
for i, idx in enumerate(top5_idx):
    print(f"{i+1}. {ids[idx]}")