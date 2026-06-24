import os
import json
import time
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "docs")
OLLAMA_URL = "http://localhost:11434/api/generate"
#MODEL = "llama3.1:8b"
MODEL = "gemma2:9b"

# [테스트 대상 3개 파일 고정]
TARGET_FILES = [
    "(주)우리홈쇼핑의 전자상거래소비자보호법 위반행위에 대한 건",
    "경상북도개인택시운송사업조합 구미시지부의 사업자단체 금지행위에 대한 건",
    "명가토건(주)의 불공정하도급거래행위에 대한 건"
]

def get_full_text(hybrid_path):
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

def get_final_prompt(title, text):
    return f"""
    당신은 공정거래위원회 의결서를 분석하여 데이터베이스 구축용 메타데이터를 정제하는 전문 데이터 엔지니어입니다.
제공된 의결서 정보를 바탕으로, 하단의 [반환 형식]을 엄격히 준수하여 순수한 JSON 데이터만 출력하세요. 다른 설명이나 마크다운 코드블록(```json)은 절대 포함하지 마십시오.

[의결서 정보]
- 의결서 제목: {title}
- 의결서 내용: {text}

[작성 규칙]

1. 업종 (중요)
- '새마을금고중앙회' 같은 '발주처/기관'의 업종이 아닙니다. 법을 위반하여 심의를 받고 있는 주체인 '피심인 기업들(또는 피심인 단체)'이 실제로 영위하는 산업 분야를 한국어로 작성하십시오.
- 반드시 의결서 내 '피심인 일반현황'에 명시된 주요 업종 명칭을 팩트 기반으로 추출해야 합니다. 
- 피심인의 주요 업종이 여러 개라면, 절대로 생략하거나 임의로 축소하지 말고 모두 쉼표(,)로 연결하여 상세히 적으십시오.
- 반드시 하나의 문자열(String)로만 작성해야 합니다. (배열/리스트 형태 절대 금지)
- 예시: "건축공사업, 실내건축공사업" (O) / ["건축공사업", "실내건축공사업"] (X)

2. 위반구체행위
- 피심인이 실제로 저지른 구체적인 법 위반 행동만 2~4개 추출하여 리스트로 작성하십시오.
- 법률 조항 문구(예: 소비자를 유인한 행위, 위탁을 임의로 취소하는 행위)를 그대로 복사하지 말고, '무슨 제품/용역을 가지고 어떻게 행동했는지' 고유한 사건 팩트를 명시하십시오.
- 시정명령 내용 및 처분 결과(~행위를 다시 하여서는 아니 된다)는 위반 행위가 아니므로 절대 포함하지 마십시오.
- 각 항목은 데이터 시각화를 위해 반드시 '15자 이내'로 간결하게 요약된 명사형 키워드로 작성하십시오.
- 예시: ["개봉 시 환불 불가 허위 표시", "타일공사 임의 분리발주"]

3. 관련법령 (중요)
- 의결서 결과의 기반이 된, 피심인이 실제로 위반한 정확한 모든 법 조항을 리스트로 작성하십시오.
- 사건의 도메인(공정거래법, 전자상거래법, 하도급법 등)에 따라 본문에 언급된 '정확한 법률의 풀네임(정식 명칭)'을 명시해야 합니다. 절대 약칭을 쓰거나 다른 법률 명칭과 혼동하여 결합하지 마십시오. (예: '공정거래법' -> '독점규제 및 공정거래에 관한 법률')
- 조항 번호는 반드시 의결서 원문에 명시된 수준까지만 작성하십시오. 원문이 제40조까지만 쓰면 제40조까지만, 제1항까지 쓰면 제1항까지만 쓰고, 원문에 없는 항이나 호를 추측하여 추가하지 마십시오.
- [강력 제외 대상]: 아래 항목들은 데이터 노이즈이므로 관련법령 리스트에서 반드시 제외하십시오.
  * 정의 조항 (예: 제2조)
  * 각종 고시, 훈령, 지침, 세부기준 (예: 과징금부과 세부기준 등에 관한 고시 등)
  * 과징금 산정 및 부과 기준 조항 (예: 제55조의3, 시행령 제61조 등)
  * 처분 및 절차 근거 조항 단독 기재 (예: 시정명령 근거인 제52조 등)

[반환 형식]
{{
  "업종": "추출된 모든 업종을 쉼표로 연결한 하나의 문자열",
  "위반구체행위": ["15자 이내 요약 팩트 1", "15자 이내 요약 팩트 2"],
  "관련법령": ["정확한 법률 풀네임 제X조 제X항"]
}}"""

def run_final_test():
    print("==================================================")
    print(f"🚀 [최종 프롬프트] 3대 도메인 팩트 수집 및 속도 테스트")
    print("==================================================")

    for idx, base_name in enumerate(TARGET_FILES, 1):
        meta_path = os.path.join(DATA_DIR, base_name + "_metadata.json")
        hybrid_path = os.path.join(DATA_DIR, base_name + "_hybrid.json")

        if not os.path.exists(meta_path):
            print(f"\n❌ [{idx}/3] 파일을 찾을 수 없음: {base_name}_metadata.json")
            continue

        with open(meta_path, encoding="utf-8") as f:
            meta = json.load(f)
        title = meta.get("의결서제목", "제목 없음")
        text = get_full_text(hybrid_path)

        print(f"\n▶ [{idx}/3] 분석 중: {title[:35]}...")
        print(f"  - 문서 길이: {len(text)}자")
        print("  - 로컬 gemma2:9b 추론 시작...")

        # ⏱️ 소요 시간 측정 시작
        start_time = time.time()
        
        try:
            response = requests.post(OLLAMA_URL, json={
                "model": MODEL,
                "prompt": get_final_prompt(title, text),
                "stream": False,
                "format": "json",
                "options": {"temperature": 0.1, "num_predict": 500}
            }, timeout=300)
            result = response.json()["response"].strip()
        except Exception as e:
            result = f"❌ 오류 발생: {e}"

        # ⏱️ 소요 시간 측정 종료
        elapsed_time = time.time() - start_time

        print(f"\n[출력 결과]")
        print(result)
        print(f"⏱️ 해당 파일 분석 소요 시간: {elapsed_time:.2f}초")
        print("-" * 50)

    print("\n✅ 모든 최종 테스트 파일 구동 완료!")

if __name__ == "__main__":
    run_final_test()