# test_search_bge.py
import chromadb
from FlagEmbedding import BGEM3FlagModel
import pickle

CHROMA_DIR = r"D:\faircheck\db\chroma_bge"
BM25_PATH = r"D:\faircheck\db\bm25_index.pkl"

print("모델 로딩...")
model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection("decisions_bge")
print(f"ChromaDB 총 chunk 수: {collection.count()}")

with open(BM25_PATH, "rb") as f:
    bm25_data = pickle.load(f)
bm25 = bm25_data["bm25"]
bm25_ids = bm25_data["ids"]
bm25_texts = bm25_data["texts"]

# 한국어 형태소 분석기 (있으면 사용, 없으면 공백 분리)
try:
    from kiwipiepy import Kiwi
    kiwi = Kiwi()
    def tokenize(text):
        return [token.form for token in kiwi.tokenize(text)]
    print("형태소 분석기: kiwipiepy 사용")
except ImportError:
    def tokenize(text):
        return text.split()
    print("형태소 분석기: 공백 분리 사용 (kiwipiepy 없음)")

queries = [
    "건설기계 임대단가 담합",
    "배달앱 수수료 불공정거래",
    "가맹점 식자재 구매 강제",
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"질의: {query}")
    print(f"{'='*60}")

    # Dense 검색
    output = model.encode(
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
        n_results=5,
        include=["documents", "metadatas", "distances"]
    )

    print("\n[BGE-M3 Dense 검색]")
    for i, (doc_id, doc, meta, dist) in enumerate(zip(
        dense_results["ids"][0],
        dense_results["documents"][0],
        dense_results["metadatas"][0],
        dense_results["distances"][0],
    )):
        print(f"  {i+1}위: {doc_id} / distance={dist:.4f}")
        print(f"       제목: {meta.get('의결서제목', '')}")
        print(f"       위반유형: {meta.get('위반유형', '')}")
        print(f"       내용: {doc[:120]}...")

    # BM25 검색
    tokenized_query = tokenize(query)
    bm25_scores = bm25.get_scores(tokenized_query)
    top5_idx = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:5]

    print("\n[BM25 검색]")
    for i, idx in enumerate(top5_idx):
        print(f"  {i+1}위: {bm25_ids[idx]} / score={bm25_scores[idx]:.4f}")
        print(f"       내용: {bm25_texts[idx][:120]}...")