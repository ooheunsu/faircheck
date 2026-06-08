import os
import json
import time
from groq import Groq

# 설정
DATA_DIR = r"D:\공개본의결서압풀"
OUTPUT_DIR = r"D:\faircheck\enriched_metadata"
os.makedirs(OUTPUT_DIR, exist_ok=True)

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract_metadata_with_llm(title, text_preview):
    prompt = f"""다음 공정거래 의결서를 분석해서 JSON으로만 답해줘. 다른 말은 절대 하지 마.

의결서 제목: {title}
의결서 내용 (앞부분): {text_preview}

반환 형식:
{{
  "업종": "건설기계업",
  "위반구체행위": ["행위1", "행위2"],
  "관련법령": ["공정거래법 제OO조"]
}}

업종은 피심인이 속한 산업 분야를 한국어로 짧게 써줘 (예: 식품제조업, 플랫폼업, 건설업, 유통업 등).
위반구체행위는 실제 위반한 구체적인 행동을 2~4개 써줘.
관련법령은 의결서에 언급된 법 조항만 써줘."""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        result = response.choices[0].message.content.strip()
        # JSON만 추출
        start = result.find("{")
        end = result.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(result[start:end])
    except Exception as e:
        print(f"  LLM 오류: {e}")
    return {"업종": "미분류", "위반구체행위": [], "관련법령": []}

def get_text_preview(hybrid_path, max_chars=800):
    """hybrid.json에서 텍스트 앞부분 추출"""
    try:
        with open(hybrid_path, encoding="utf-8") as f:
            chunks = json.load(f)
        text = ""
        for chunk in chunks:
            if chunk["metadata"].get("chunk_type") == "text":
                text += chunk["page_content"] + " "
            if len(text) >= max_chars:
                break
        return text[:max_chars]
    except:
        return ""

def process_all():
    # 이미 처리된 파일 확인
    done = set(f.replace("_enriched.json", "") for f in os.listdir(OUTPUT_DIR))
    
    # metadata.json 파일 목록
    meta_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_metadata.json")]
    total = len(meta_files)
    
    print(f"총 {total}개 의결서 처리 시작")
    print(f"이미 처리됨: {len(done)}개")
    print("-" * 50)

    for i, meta_file in enumerate(meta_files, 1):
        base_name = meta_file.replace("_metadata.json", "")
        
        # 이미 처리된 것 스킵
        if base_name in done:
            print(f"[{i}/{total}] 스킵 (이미 처리됨): {base_name[:30]}...")
            continue

        meta_path = os.path.join(DATA_DIR, meta_file)
        hybrid_path = os.path.join(DATA_DIR, base_name + "_hybrid.json")

        # metadata.json 로드
        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        title = meta.get("의결서제목", "")
        text_preview = get_text_preview(hybrid_path)

        print(f"[{i}/{total}] 처리 중: {title[:40]}...")

        # LLM으로 추가 메타데이터 추출
        llm_result = extract_metadata_with_llm(title, text_preview)

        # 피심인정보 평탄화
        피심인정보 = meta.get("피심인정보", [{}])
        위반유형 = 피심인정보[0].get("위반유형", "") if 피심인정보 else ""
        세부위반유형 = 피심인정보[0].get("세부위반유형", "") if 피심인정보 else ""
        피심인기업명 = 피심인정보[0].get("피심인기업명", "") if 피심인정보 else ""
        조치유형 = 피심인정보[0].get("조치유형", "") if 피심인정보 else ""

        # 최종 enriched 메타데이터
        enriched = {
            "의결서관리번호": meta.get("의결서관리번호", ""),
            "의결서제목": title,
            "공개일자": meta.get("공개일자", ""),
            "의결서파일명": meta.get("의결서파일명", ""),
            "위반유형": 위반유형,
            "세부위반유형": 세부위반유형,
            "피심인기업명": 피심인기업명,
            "조치유형": 조치유형,
            # LLM이 추가한 것들
            "업종": llm_result.get("업종", "미분류"),
            "위반구체행위": llm_result.get("위반구체행위", []),
            "관련법령": llm_result.get("관련법령", [])
        }

        # 저장
        out_path = os.path.join(OUTPUT_DIR, base_name + "_enriched.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

        print(f"  ✓ 업종: {enriched['업종']}")
        print(f"  ✓ 위반행위: {enriched['위반구체행위']}")

        # API 속도 제한 방지 (초당 30 요청 제한)
        time.sleep(0.5)

    print("\n✅ 완료!")
    print(f"결과 저장 위치: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all()