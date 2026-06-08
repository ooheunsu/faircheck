import os
import json

data_dir = r"D:\공개본의결서압풀"

# 파일 개수 파악
pdf_files = []
hybrid_files = []
metadata_files = []

for filename in os.listdir(data_dir):
    if filename.endswith(".pdf"):
        pdf_files.append(filename)
    elif filename.endswith("_hybrid.json"):
        hybrid_files.append(filename)
    elif filename.endswith("_metadata.json"):
        metadata_files.append(filename)

print(f"PDF 파일 수: {len(pdf_files)}")
print(f"hybrid.json 수: {len(hybrid_files)}")
print(f"metadata.json 수: {len(metadata_files)}")

# hybrid.json 하나 열어서 총 chunk 수 파악
sample_hybrid = hybrid_files[0]
with open(os.path.join(data_dir, sample_hybrid), encoding="utf-8") as f:
    data = json.load(f)
print(f"\n샘플 파일: {sample_hybrid}")
print(f"chunk 수: {len(data)}")
print(f"첫 번째 chunk_id: {data[0]['metadata']['chunk_id']}")

# metadata.json 하나 열어서 구조 확인
sample_meta = metadata_files[0]
with open(os.path.join(data_dir, sample_meta), encoding="utf-8") as f:
    meta = json.load(f)
print(f"\n샘플 메타데이터:")
print(json.dumps(meta, ensure_ascii=False, indent=2))