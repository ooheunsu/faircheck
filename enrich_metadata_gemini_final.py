# enrich_metadata_gemini_final.py
import os
import json
import time
from google import genai

DATA_DIR = r"D:\공개본의결서압풀"
OUTPUT_DIR = r"D:\faircheck\enriched_metadata_gemini_final"
os.makedirs(OUTPUT_DIR, exist_ok=True)

API_KEY = os.environ.get("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)

# 표준 업종 목록
INDUSTRY_LIST = """
건설업, 건설기계업, 식품제조업, 식품유통업, 유통업, 도소매업,
플랫폼업, IT서비스업, 소프트웨어업, 전자상거래업, 통신업,
금융업, 보험업, 제조업, 전자제조업, 자동차부품제조업, 화학제조업,
의약품제조업, 의료기기업, 교육업, 교육서비스업, 운송업, 물류업,
농업, 수산업, 축산업, 가맹사업업, 부동산업, 광고업, 에너지업, 기타
"""

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
    prompt = f"""다음 공정거래 의결서를 분석해서 JSON으로만 답해줘.
절대로 JSON 외에 다른 말, 설명, 마크다운 코드블록(```json)을 포함하지 마.
반드시 유효한 JSON 형식으로만 답해줘.

의결서 제목: {title}
의결서 내용: {text}

반환 형식:
{{
  "업종": "건설기계업",
  "위반구체행위": ["행위1", "행위2", "행위3"],
  "관련법령": ["독점규제 및 공정거래에 관한 법률 제40조 제1항 제1호", "독점규제 및 공정거래에 관한 법률 제51조 제1항"]
}}

[업종 작성 규칙]
- 목록은 참고용이야. 정확히 맞는 것이 있으면 선택하고, 없거나 애매하면 직접 더 적합한 업종명을 써줘. 억지로 맞추지 마.
- 목록: {INDUSTRY_LIST}
- 반드시 하나의 문자열로만 써줘. 리스트나 배열로 쓰지 마.
- 예: "건설기계업" (O), ["건설기계업"] (X)

[위반구체행위 작성 규칙]
- 실제로 위반한 구체적인 행동만 2~4개 써줘
- 시정명령 내용(~하여서는 아니된다)은 쓰지 마. 실제 위반행위만 써줘
- 각 항목은 간결하게 15자 이내로 요약해줘
- 예: "임대단가 결정 및 배포" (O), "건설기계 임대단가를 결정하고 구성사업자들에게 이를 준수하도록 함으로써 부당하게 경쟁을 제한하는 행위" (X)

[관련법령 작성 규칙]
- 피심인이 실제로 위반한 법률 조항만 써줘
- 아래 항목은 반드시 제외해줘:
  * 정의 조항 (제2조)
  * 고시, 훈령, 세부기준 (예: 과징금부과 세부기준 등에 관한 고시)
  * 과징금 산정 관련 조항 (예: 제55조의3, 시행령 제61조)
  * 절차 조항 (예: 시정명령 근거 조항인 제52조 단독)
- 법률명은 약칭 없이 정식 명칭으로 써줘
- 의결서 원문에 명시된 수준까지만 써줘. 원문이 제40조까지만 쓰면 제40조까지만, 제1항까지 쓰면 제1항까지, 제1호까지 쓰면 제1호까지 써줘.
- 원문에 없는 항이나 호를 추가로 붙이지 마.
- 예: "독점규제 및 공정거래에 관한 법률 제40조" (O)
- 예: "독점규제 및 공정거래에 관한 법률 제40조 제1항" (O)
- 예: "독점규제 및 공정거래에 관한 법률 제40조 제1항 제1호" (O)
- 예: "공정거래법 제40조" (X), "독점규제 및 공정거래에 관한 법률" (X)
- 예: "과징금부과 세부기준 등에 관한 고시 Ⅲ. 1. 가." (X)"""

    for attempt in range(3):
        try:
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt
            )
            result = response.text.strip()

            # 마크다운 코드블록 제거
            if "```" in result:
                result = result.split("```")[1]
                if result.startswith("json"):
                    result = result[4:]

            start = result.find("{")
            end = result.rfind("}") + 1
            if start != -1 and end != 0:
                parsed = json.loads(result[start:end])

                # 업종이 리스트로 온 경우 처리
                if isinstance(parsed.get("업종"), list):
                    parsed["업종"] = parsed["업종"][0] if parsed["업종"] else "기타"

                # 관련법령에서 정의조항(제2조) 필터링
                laws = parsed.get("관련법령", [])
                laws = [l for l in laws if "제2조" not in l or "제2조의" in l]
                parsed["관련법령"] = laws

                return parsed

        except json.JSONDecodeError as e:
            print(f"  JSON 파싱 오류({attempt+1}/3): {e}")
            print(f"  원본 응답: {result[:200]}")
            if attempt < 2:
                time.sleep(5)
        except Exception as e:
            print(f"  오류({attempt+1}/3): {e}")
            if attempt < 2:
                time.sleep(10)

    return {"업종": "기타", "위반구체행위": [], "관련법령": []}

def process_all():
    done = set()
    for f in os.listdir(OUTPUT_DIR):
        if not f.endswith("_enriched.json"):
            continue
        fpath = os.path.join(OUTPUT_DIR, f)
        with open(fpath, encoding="utf-8") as fp:
            data = json.load(fp)
        if data.get("업종") and data.get("업종") != "기타":
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

        print(f"\n[{i}/{total}] {title[:45]}...")
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
            "업종": llm_result.get("업종", "기타"),
            "위반구체행위": llm_result.get("위반구체행위", []),
            "관련법령": llm_result.get("관련법령", [])
        }

        out_path = os.path.join(OUTPUT_DIR, base + "_enriched.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(enriched, f, ensure_ascii=False, indent=2)

        print(f"  ✓ 업종: {enriched['업종']}")
        print(f"  ✓ 관련법령 ({len(enriched['관련법령'])}개): {enriched['관련법령']}")
        print(f"  ✓ 위반행위: {enriched['위반구체행위']}")

        time.sleep(1)

    print(f"\n✅ 완료! 결과: {OUTPUT_DIR}")

if __name__ == "__main__":
    process_all()