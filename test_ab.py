import os
import json
import requests

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "docs")
OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL = "llama3.1:8b"

# [테스트 세팅] 방금 그 새마을금고 파일 이름 고정 (확장자 제외한 앞부분만)
# 폴더에 있는 실제 파일명과 매칭되도록 놔둡니다.
TARGET_BASE_NAME = "(주)인성정보의 불공정하도급거래행위에 대한 건"

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

# -------------------------------------------------------------
# [프롬프트 A] 친구분이 준 기존 프롬프트 (건설기계업 예시 포함)
# -------------------------------------------------------------
def get_prompt_A(title, text):
    return f"""다음 공정거래 의결서를 분석해서 JSON으로만 답해줘. 다른 말은 절대 하지 마.

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

# -------------------------------------------------------------
# [프롬프트 B] 피심인 기준 가이드를 보완한 새 프롬프트
# -------------------------------------------------------------
def get_prompt_B(title, text):
    return f"""당신은 공정거래위원회 의결서를 분석하여 핵심 메타데이터를 추출하는 전문 데이터 분석가입니다.
제공된 의결서 내용을 바탕으로 반드시 지정된 JSON 형식으로만 답변하세요. 생각이나 추가 설명은 절대 하지 마십시오.

[의결서 정보]
- 의결서 제목: {title}
- 의결서 내용: {text}

[작성 규칙]
1. 업종 (중요): 
   - '새마을금고중앙회' 같은 '발주처/기관'의 업종이 아닙니다.
   - 법을 위반하여 심의를 받고 있는 주체인 '피심인 기업들'이 실제로 영위하는 산업 분야를 한국어로 짧게 작성하세요.
   - (예: 입찰 대상이 보험 관련 용역이더라도, 피심인 기업들이 컨설팅이나 교육을 해주는 업체라면 업종은 '경영컨설팅업' 또는 '교육서비스업'이 되어야 합니다.)
2. 위반구체행위: 의결서 본문 및 주문에 나타난 피심인들의 실제 법 위반 행위를 구체적으로 2~4개 요약하여 리스트로 작성하세요. (예: ["사전 낙찰예정자 합의", "들러리 참여 및 투찰가격 공동 결정"])
3. 관련법령: 의결서에 언급된 정확한 모든 법 조항을 리스트로 작성하세요. (예: ["독점규제 및 공정거래에 관한 법률 제19조 제1항 제8호"])

[반환 형식]
{{
  "업종": "<피심인 기업의 실제 산업 분야를 명사형으로 입력>",
  "위반구체행위": ["구체적 행위1", "구체적 행위2"],
  "관련법령": ["관련 법 조항 1", "관련 법 조항 2"]
}}"""

def request_ollama(prompt):
    try:
        response = requests.post(OLLAMA_URL, json={
            "model": MODEL,
            "prompt": prompt,
            "stream": False,
            "options": {"temperature": 0.1, "num_predict": 500}
        }, timeout=300) # 타임아웃 5분으로 넉넉하게 고정
        return response.json()["response"].strip()
    except Exception as e:
        return f"오류 발생: {e}"

def run_ab_test():
    meta_path = os.path.join(DATA_DIR, TARGET_BASE_NAME + "_metadata.json")
    hybrid_path = os.path.join(DATA_DIR, TARGET_BASE_NAME + "_hybrid.json")
    
    if not os.path.exists(meta_path):
        print(f"❌ 파일을 찾을 수 없습니다.\ndocs 폴더에 다음 이름의 파일이 있는지 확인해주세요:\n{TARGET_BASE_NAME}_metadata.json")
        return

    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    title = meta.get("의결서제목", "제목 없음")
    text = get_full_text(hybrid_path)

    print("==================================================")
    print(f"🔬 프롬프트 AB 테스트 시작")
    print(f"📄 대상 문서: {title[:40]}...")
    print(f"📏 텍스트 길이: {len(text)}자")
    print("==================================================\n")

    # 1. 프롬프트 A 테스트
    print("⏳ [프롬프트 A - 기존 버전] 실행 중...")
    result_A = request_ollama(get_prompt_A(title, text))
    print("\n[결과 A]")
    print(result_A)
    print("\n" + "-"*50 + "\n")

    # 2. 프롬프트 B 테스트
    print("⏳ [프롬프트 B - 개선 버전] 실행 중...")
    result_B = request_ollama(get_prompt_B(title, text))
    print("\n[결과 B]")
    print(result_B)
    print("\n==================================================")

if __name__ == "__main__":
    run_ab_test()