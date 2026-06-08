# enrich_metadata_gemini.py
import os
import json
import time
from google import genai

# 설정
DATA_DIR = r"D:\공개본의결서압풀"
OUTPUT_DIR = r"D:\faircheck\enriched_metadata_gemini"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_KEY = os.environ.get("GOOGLE_API_KEY")
if not API_KEY:
    raise ValueError("GOOGLE_API_KEY 환경변수가 설정되지 않았어요!")
print(f"API 키 확인: {API_KEY[:8]}...")

client = genai.Client(api_key=API_KEY)

def get_full_text(hybrid_path):
    with open(hybrid_path, encoding="utf-8") as f:
        chunks = json.load(f)
    이유_text = ""
    전체_text = ""
    for chunk in chunks:
        if chunk["metadata"].get("chunk_type") != "text":
            continue
        content = chunk["page_content"]
        section = chunk["metadata"].get("section", "")
        if section == "이유":
            이유_text += content + " "
        전체_text += content + " "
    return 이유_text if 이유_text else 전체_text

def extract_metadata(title, text):
    prompt = f"""다음 공정거래 의결서를 분석해서 JSON으로만 답해줘. 다른 말은 절대 하지 마.

의결서 제목: {title}
의결서 내용: {text}

반환 형식:
{{
  "업종": "건설기계업",
  "위반구체행위": ["행위1", "행위2"],
  "관련법령": ["공정거래법 제40조 제1항 제1호"]
}}

업종은 피심인이 속한 산업 분야를 한국어로 짧게 써줘 (예: 식품제조업, 플랫폼업, 건설업, 유통업 등).
위반구체행위는 실제 위반한 구체적인 행동을 2~4개 써줘.
관련법령은 의결서에 언급된 정확한 모든 법 조항을 써줘 (예: 공정거래법 제40조 제1항 제1호)."""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash-preview-04-17",
                contents=prompt
            )
            result = response.text.strip()
            start = result.find("{")
            end = result.rfind("}") + 1
            if start != -1 and end != 0:
                return json.loads(result[start:end])
        except Exception as e:
            print(f"  오류({attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(10)
    return {"업종": "미분류", "위반구체행위": [], "관련법령": []}

def process_all():
    done = set()
    for f in os.listdir(OUTPUT_DIR):
        if not f.endswith("_enriched.json"):
            continue
        fpath = os.path.join(OUTPUT_DIR, f)
        with open(fpath, encoding="utf-8") as fp:
            data = json.load(fp)
        if data.get("업종") and data.get("업종") != "미분류":
            done.add(f.replace("_enriched.json", ""))

    meta_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_metadata.json")]
    total = len(meta_files)

    print(f"총 {total}개 처리 시작")
    print(f"이미 처리됨: {len(done)}개")
    print("-" * 50)

    for i, meta_file in enumerate(meta_files, 1):
        base = meta_file.replace("_metadata.json", "")
        if base in done:
            print(f"[{i}/{total}] 스킵: {base[:30]}...")
            continue

        meta_path = os.path.join(DATA_DIR, meta_file)
        hybrid_path = os.path.join(DATA_DIR, base + "_hybrid.json")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        title = meta.get("의결서제목", "")
        text = get_full_text(hybrid_path)

        print(f"[{i}/{total}] 처리 중: {title[:40]}...")

        llm_result = extract_metadata(title, text)

        피심인정보 = meta.get("피심인정보", [{}])
        첫번째 = 피심인정보[0] if 피심인정보 else {}

        enriched = {
            "의결서관리번호": meta.get("의결서관리번호", ""),
            "의결서제목": title,
            "공개일자": meta.get("공개일자", ""),
            "의결서파일명": meta.get("의결서파일명", ""),
            "위반유형": 첫번째.get("위반유형", ""),
            "세부위반유형": 첫번째.get("세부위반유형", ""),
            "피심인기업명": 첫번째.get("피심인기업명", ""),
            "조치유형": 첫번째.get("조치유형", ""),
            "업종": llm_result.get("업종", "미분류"),
            "위반구체행위": llm_result.get("위반구체행위", []),
            "관련법령": llm_result.get("관련법령", [])
        }

        out_path = os.path.join(OUTPUT_DIR, base + "_enriched.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

        print(f"  ✓ 업종: {enriched['업종']}")
        print(f"  ✓ 관련법령: {enriched['관련법령']}")

        time.sleep(6)

    print("\n✅ 완료!")

if __name__ == "__main__":
    process_all()