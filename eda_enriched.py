# eda_enriched.py
import os
import json
from collections import Counter

OUTPUT_DIR = r"D:\faircheck\enriched_metadata_gemini"

업종_list = []
법령_list = []
행위_counts = []
관련법령_counts = []

for fname in os.listdir(OUTPUT_DIR):
    if not fname.endswith("_enriched.json"):
        continue
    with open(os.path.join(OUTPUT_DIR, fname), encoding="utf-8") as f:
        data = json.load(f)
    
    # 업종이 리스트로 들어온 경우 처리
    업종 = data.get("업종", "미분류")
    if isinstance(업종, list):
        업종 = 업종[0] if 업종 else "미분류"
    업종_list.append(업종)
    
    행위_counts.append(len(data.get("위반구체행위", [])))
    관련법령_counts.append(len(data.get("관련법령", [])))
    for law in data.get("관련법령", []):
        if isinstance(law, str):
            법령_list.append(law)

# 업종 분포
print("=" * 50)
print("【 업종별 의결서 수 】")
print("=" * 50)
업종_counter = Counter(업종_list)
for 업종, count in 업종_counter.most_common():
    bar = "█" * count
    print(f"{업종:<20} {count:>4}개  {bar}")

# 관련법령 수 분포
print("\n" + "=" * 50)
print("【 의결서당 관련법령 수 분포 】")
print("=" * 50)
법령수_counter = Counter(관련법령_counts)
for 수, count in sorted(법령수_counter.items()):
    print(f"{수}개: {count}건")

# 자주 등장하는 법령
print("\n" + "=" * 50)
print("【 자주 등장하는 관련법령 Top 20 】")
print("=" * 50)
법령_counter = Counter(법령_list)
for law, count in 법령_counter.most_common(20):
    print(f"{count:>4}회  {law}")