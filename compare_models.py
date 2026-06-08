# compare_models.py
import os
import json
import time
from groq import Groq

DATA_DIR = r"D:\공개본의결서압풀"
client = Groq(api_key=os.environ.get("GROQ_API_KEY"))

def extract(title, text, model):
    prompt = f"""다음 공정거래 의결서를 분석해서 JSON으로만 답해줘. 다른 말은 절대 하지 마.

의결서 제목: {title}
의결서 내용: {text}

반환 형식:
{{
  "업종": "건설기계업",
  "위반구체행위": ["행위1", "행위2"],
  "관련법령": ["공정거래법 제OO조"]
}}

업종은 피심인이 속한 산업 분야를 한국어로 짧게 써줘 (예: 식품제조업, 플랫폼업, 건설업, 유통업 등).
위반구체행위는 실제 위반한 구체적인 행동을 2~4개 써줘.
관련법령은 의결서에 언급된 정확한 모든 법 조항을 써줘. (예: 공정거래법 제40조 제1항 제1호)."""
    try:    
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300
        )
        result = response.choices[0].message.content.strip()
        start = result.find("{")
        end = result.rfind("}") + 1
        if start != -1 and end != 0:
            return json.loads(result[start:end])
    except Exception as e:
        print(f"  오류: {e}")
    return {}

# 샘플 3개만 비교
meta_files = [f for f in os.listdir(DATA_DIR) if f.endswith("_metadata.json")][:3]
models = ["llama-3.1-8b-instant", "llama-3.3-70b-versatile"]

for meta_file in meta_files:
    base = meta_file.replace("_metadata.json", "")
    with open(os.path.join(DATA_DIR, meta_file), encoding="utf-8") as f:
        meta = json.load(f)
    
    hybrid_path = os.path.join(DATA_DIR, base + "_hybrid.json")
    with open(hybrid_path, encoding="utf-8") as f:
        chunks = json.load(f)
    text = " ".join([c["page_content"] for c in chunks if c["metadata"].get("chunk_type") == "text"])[:800]
    
    title = meta.get("의결서제목", "")
    print(f"\n{'='*60}")
    print(f"의결서: {title[:40]}")
    print(f"{'='*60}")
    
    for model in models:
        result = extract(title, text, model)
        print(f"\n[{model}]")
        print(f"  업종: {result.get('업종')}")
        print(f"  관련법령: {result.get('관련법령')}")
        print(f"  위반행위: {result.get('위반구체행위')}")
        time.sleep(1)  # API 속도 제한 방지