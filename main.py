import os
import json
import time
import requests

# [수정] 맥북(M1 Air) 환경과 FairCheck 프로젝트 폴더 구조에 맞게 경로 재설정
BASE_DIR = os.path.dirname(os.path.abspath(__file__)) # 현재 main.py가 있는 폴더 경로
DATA_DIR = os.path.join(BASE_DIR, "docs")             # FairCheck/docs 폴더
OUTPUT_DIR = os.path.join(BASE_DIR, "enriched_metadata_ollama") # 결과가 저장될 폴더

# 결과 저장 폴더가 없으면 자동으로 생성
os.makedirs(OUTPUT_DIR, exist_ok=True)

SAMPLE_COUNT = 3  # 먼저 3개만 테스트로 돌려보기
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

def get_full_text(hybrid_path):
    # 만약 하이브리드 파일이 누락되어 있다면 예외 처리
    if not os.path.exists(hybrid_path):
        return ""
        
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
            response = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.1,
                    "num_predict": 500
                }
            }, timeout=300)
            result = response.json()["response"].strip()
            
            # JSON만 파싱하기 위한 슬라이싱 로직
            start = result.find("{")
            end = result.rfind("}") + 1
            if start != -1 and end != 0:
                return json.loads(result[start:end])
        except Exception as e:
            print(f"  오류({attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(5)
    return {"업종": "미분류", "위반구체행위": [], "관련법령": []}

def process_all():
    # 데이터 폴더가 비어있거나 없는 경우 점검
    if not os.path.exists(DATA_DIR):
        print(f"❌ 에러: {DATA_DIR} 폴더를 찾을 수 없습니다. docs 폴더에 데이터를 넣어주세요.")
        return

    # 맥북의 숨김파일(.DS_Store) 등을 제외하고 진짜 _metadata.json 파일만 수집
    meta_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_metadata.json") and not f.startswith(".")]
    
    if not meta_files:
        print("❌ 에러: docs 폴더 안에 '_metadata.json'으로 끝나는 파일이 하나도 없습니다.")
        return

    targets = meta_files[:SAMPLE_COUNT]

    print(f"🚀 FairCheck 프로젝트 메타데이터 확장 시작 (로컬 Ollama {MODEL})")
    print(f"📂 입력 경로: {DATA_DIR}")
    print(f"📂 출력 경로: {OUTPUT_DIR}")
    print("-" * 50)

    for i, meta_file in enumerate(targets, 1):
        base = meta_file.replace("_metadata.json", "")
        meta_path = os.path.join(DATA_DIR, meta_file)
        hybrid_path = os.path.join(DATA_DIR, base + "_hybrid.json")

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)

        title = meta.get("의결서제목", "제목 없음")
        text = get_full_text(hybrid_path)

        print(f"[{i}/{SAMPLE_COUNT}] 처리 중: {title[:30]}...")
        print(f"  텍스트 길이: {len(text)}자")

        llm_result = extract_metadata(title, text)

        피심인정보 = meta.get("피심인정보", [])
        첫번째 = 피심인정보[0] if 피심인정보 else {}

        enriched = {
            "의결서관리번호": meta.get("의결서관리번호", ""),
            "의결서제목": title,
            "공개일자": meta.get("공개일자", ""),
            "의결서파일명": meta.get("의결서파일명", ""),
            "파일경로": meta.get("파일경로", ""),
            "위반유형": 첫번째.get("위반유형", ""),
            "세부위반유형": 첫번째.get("세부위반유형", ""),
            "조치유형": 첫번째.get("조치유형", ""),
            "조치일자": 첫번째.get("조치일자", ""),
            "피심인기업명": 첫번째.get("피심인기업명", ""),
            "업종": llm_result.get("업종", "미분류"),
            "위반구체행위": llm_result.get("위반구체행위", []),
            "관련법령": llm_result.get("관련법령", [])
        }

        out_path = os.path.join(OUTPUT_DIR, base + "_enriched.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

        print(f"  ✓ 업종: {enriched['업종']}")
        print(f"  ✓ 관련법령: {enriched['관련법령']}")
        print(f"  ✓ 위반행위: {enriched['위반구체행위']}")
        print("-" * 30)

    print(f"\n✅ 샘플 처리 완료! 결과 저장 폴더: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all()