# build_chromadb_bge.py
import os
import json
import chromadb
import time
from datetime import timedelta
from FlagEmbedding import BGEM3FlagModel
from rank_bm25 import BM25Okapi
import pickle

HYBRID_DIR = r"D:\faircheck\data\hybrid"
ENRICHED_DIR = r"D:\faircheck\data\enriched"
CHROMA_DIR = r"D:\faircheck\db\chroma_bge"
BM25_PATH = r"D:\faircheck\db\bm25_index.pkl"  # BM25는 재사용

os.makedirs(CHROMA_DIR, exist_ok=True)
os.makedirs(r"D:\faircheck\db", exist_ok=True)

start_time = time.perf_counter()

print("BGE-M3 모델 로딩...")
model = BGEM3FlagModel(
    "BAAI/bge-m3",
    use_fp16=True  # 메모리 절약, GPU 있으면 더 빠름
)

print("ChromaDB 초기화...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name="decisions_bge",
    metadata={"hnsw:space": "cosine"}
)

# enriched 메타데이터 로드
print("enriched 메타데이터 로딩...")
enriched_map = {}
for f in os.listdir(ENRICHED_DIR):
    if not f.endswith("_enriched.json"):
        continue
    with open(os.path.join(ENRICHED_DIR, f), encoding="utf-8") as fp:
        data = json.load(fp)
    파일명 = data.get("의결서파일명", "")
    enriched_map[파일명] = data

print(f"enriched 메타데이터 {len(enriched_map)}개 로드됨")

# 전체 chunk 수집
print("hybrid.json 로딩...")
all_ids = []
all_texts = []
all_metadatas = []

for f in os.listdir(HYBRID_DIR):
    if not f.endswith("_hybrid.json"):
        continue
    with open(os.path.join(HYBRID_DIR, f), encoding="utf-8") as fp:
        chunks = json.load(fp)

    uuid = chunks[0]["metadata"]["chunk_id"].split("-CH-")[0].replace("DOC-", "")
    enriched = enriched_map.get(uuid, {})

    for chunk in chunks:
        chunk_id = chunk["metadata"]["chunk_id"]
        page_content = chunk["page_content"]

        metadata = {
            "의결서제목": enriched.get("의결서제목", ""),
            "의결서관리번호": enriched.get("의결서관리번호", ""),
            "공개일자": enriched.get("공개일자", ""),
            "의결서파일명": enriched.get("의결서파일명", ""),
            "업종": enriched.get("업종", ""),
            "위반유형": enriched.get("위반유형", ""),
            "세부위반유형": enriched.get("세부위반유형", ""),
            "조치유형": enriched.get("조치유형", ""),
            "피심인기업명": enriched.get("피심인기업명", ""),
            "관련법령": json.dumps(enriched.get("관련법령", []), ensure_ascii=False),
            "위반구체행위": json.dumps(enriched.get("위반구체행위", []), ensure_ascii=False),
            "section": chunk["metadata"].get("section", ""),
            "chunk_type": chunk["metadata"].get("chunk_type", ""),
            "chunk_index": chunk["metadata"].get("chunk_index", 0),
        }

        all_ids.append(chunk_id)
        all_texts.append(page_content)
        all_metadatas.append(metadata)

print(f"총 chunk 수: {len(all_texts)}개")

# 이미 적재된 chunk 확인
existing = set(collection.get()["ids"])
print(f"이미 적재됨: {len(existing)}개")

new_ids, new_texts, new_metadatas = [], [], []
for i, chunk_id in enumerate(all_ids):
    if chunk_id not in existing:
        new_ids.append(chunk_id)
        new_texts.append(all_texts[i])
        new_metadatas.append(all_metadatas[i])

print(f"새로 적재할 chunk: {len(new_ids)}개")

# 배치 임베딩 + ChromaDB 적재
BATCH_SIZE = 32  # bge-m3는 무거우니까 배치 작게
total = len(new_ids)

for i in range(0, total, BATCH_SIZE):
    batch_ids = new_ids[i:i+BATCH_SIZE]
    batch_texts = new_texts[i:i+BATCH_SIZE]
    batch_metas = new_metadatas[i:i+BATCH_SIZE]

    print(f"임베딩 중... [{i+len(batch_ids)}/{total}]")
    
    # BGE-M3 임베딩 (dense만 사용)
    output = model.encode(
        batch_texts,
        batch_size=BATCH_SIZE,
        max_length=8192,
        return_dense=True,
        return_sparse=False,
        return_colbert_vecs=False
    )
    embeddings = output["dense_vecs"].tolist()

    collection.add(
        ids=batch_ids,
        documents=batch_texts,
        embeddings=embeddings,
        metadatas=batch_metas
    )

print("✅ BGE-M3 ChromaDB 구축 완료!")

# BM25는 텍스트 기반이라 재사용 가능
# 이미 bm25_index.pkl 있으면 스킵
if not os.path.exists(BM25_PATH):
    print("\nBM25 인덱스 구축 중...")
    corpus = [text.split() for text in all_texts]
    bm25 = BM25Okapi(corpus)
    with open(BM25_PATH, "wb") as f:
        pickle.dump({
            "bm25": bm25,
            "ids": all_ids,
            "texts": all_texts
        }, f)
    print(f"✅ BM25 인덱스 완료!")
else:
    print("\nBM25 인덱스 이미 있음, 스킵!")


end_time = time.perf_counter()
elapsed = end_time - start_time

print("\n🎉 모든 DB 구축 완료!")
print(f"총 소요시간: {timedelta(seconds=int(elapsed))}")
print(f"총 소요시간(초): {elapsed:.2f}초")