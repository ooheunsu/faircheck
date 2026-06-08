# copy_hybrid.py
## 공모전 zip파일 데이터에서 hybrid.json파일들만 faircheck프젝 폴더로 옮기는 코드
import os
import shutil

SRC = r"D:\공개본의결서압풀"
DST = r"D:\faircheck\data\hybrid"
os.makedirs(DST, exist_ok=True)

count = 0
for f in os.listdir(SRC):
    if f.endswith("_hybrid.json"):
        shutil.copy(os.path.join(SRC, f), os.path.join(DST, f))
        count += 1

print(f"복사 완료: {count}개")