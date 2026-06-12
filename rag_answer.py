import argparse
import json
import textwrap
import time


DEFAULT_PROVIDER = "gemini"
DEFAULT_MODEL = "gemini-2.5-flash"
DEFAULT_ANALYZER_MODEL = "gemini-2.5-flash"


CLEAR_OUT_OF_SCOPE_KEYWORDS = {
    "최저임금/근로기준법 등 노동관계 법령": [
        "알바",
        "최저임금",
        "주휴수당",
        "퇴직금",
        "근로계약",
        "임금체불",
        "해고",
    ],
    "임대차": ["월세", "전세", "보증금", "임대차", "집주인", "상가임대"],
    "세금": ["세금", "부가세", "종합소득세", "법인세", "원천세"],
    "형사": ["폭행", "사기", "절도", "고소", "고발", "형사"],
}


SAMPLE_QUESTIONS = {
    "franchise_delivery_fee": (
        "가맹본부가 배달앱 주문에서 발생하는 중개수수료와 배달비 일부를 "
        "가맹점사업자에게 부담시키려고 하는데 공정거래상 문제가 될 수 있어?"
    ),
    "franchise_required_purchase": (
        "가맹본부가 특정 식자재와 소모품을 반드시 본사 또는 지정업체에서만 "
        "구매하도록 하고 외부 구매를 금지하면 위반 가능성이 있어?"
    ),
    "platform_fee_change": (
        "플랫폼 사업자가 입점업체에게 사전 협의 없이 광고비와 수수료 체계를 "
        "변경해서 부담을 늘리는 경우 불공정거래 문제가 될 수 있어?"
    ),
    "pharma_rebate": (
        "제약회사가 병원 의사에게 자사 의약품 처방 유지를 목적으로 강의료나 "
        "자문료 명목의 금전을 지급하면 리베이트로 볼 수 있어?"
    ),
    "bid_rigging": (
        "입찰에 참여하는 업체들이 낙찰 예정자와 투찰 가격을 미리 정하고 "
        "나머지 업체는 들러리로 참여했다면 담합 가능성이 높아?"
    ),
    "resale_price": (
        "제조사가 대리점에게 온라인 최저 판매가격을 정해주고 이를 지키지 않으면 "
        "공급을 중단하겠다고 하는 경우 재판매가격유지행위가 될 수 있어?"
    ),
}


SYSTEM_INSTRUCTION = """\
### 역할(Role)
당신은 '공정거래 위반 가능성 사전진단 서비스'의 RAG 답변 생성기입니다.
공정거래위원회 의결서 검색 결과를 근거로, 사용자가 설명한 거래 관행이나 사업 행위가
공정거래 관련 법령상 문제될 가능성이 있는지 초기 검토를 돕습니다.

### 청중(Audience)
주요 사용자는 가맹본부/가맹점사업자, 플랫폼 입점업체, 제조사/대리점,
입찰 참여 사업자, 제약/의료기기 영업 담당자, 사내 컴플라이언스 담당자입니다.
법률 전문가가 아닌 실무자도 이해할 수 있도록 설명하되, 근거와 한계는 정확히 구분하세요.

### 작업(Task)
제공된 [검색 근거]만 사용하여 사용자의 질문에 답하세요.
검색 근거에 포함된 유사 의결서, 위반유형, 사실관계, 판단 근거를 바탕으로
위반 가능성과 추가 검토 포인트를 구조화해 제시하세요.

### 정책과 제약(Policy & Constraints)
- 검색 근거에 없는 사실, 사건명, 법령 조문, 판단 이유를 만들어내지 마세요.
- 검색 근거 번호는 [문서 1], [문서 2], [문서 1-근거 1] 형식으로만 인용하세요.
- chunk ID는 답변 본문에 직접 쓰지 마세요. 실제 chunk ID 목록은 시스템이 별도로 출력합니다.
- 질문이 최저임금, 근로계약, 임대차, 세금, 형사, 가족관계, 일반 민원, 단순 상권 경쟁처럼
  공정거래위원회 의결서 기반 사전진단 범위를 벗어난 경우에는 억지로 의결서를 연결하지 마세요.
- 범위 밖 질문이거나 검색 근거가 단어만 일부 겹칠 뿐 직접 관련성이 낮은 경우,
  위반 가능성은 "판단 보류"로 표시하고 "직접 관련 있는 공정거래 의결서가 확인되지 않았습니다"라고 답하세요.
- 범위 밖 질문에서는 핵심 근거에 관련성이 낮은 의결서를 제시하지 말고,
  "직접 관련 근거 없음"이라고 작성하세요.
- 다만 공정거래 쟁점으로 이어질 수 있는 조건이 있다면, 그 조건을 확인 필요 사항에만 제시하세요.
  예: 인근 출점은 일반적으로 단순 경쟁일 수 있으나, 같은 가맹본부의 영업지역 침해라면 가맹사업법 쟁점이 될 수 있습니다.
- 단정적으로 "위반이다" 또는 "위반이 아니다"라고 결론내리지 마세요.
- "위반 가능성", "유사한 쟁점", "추가 검토 필요", "근거상 확인되는 범위" 같은 신중한 표현을 사용하세요.
- 사용자의 사실관계와 검색 근거의 사실관계가 다른 경우, 그 차이를 명확히 설명하세요.
- 근거가 부족하거나 직접 대응되는 의결서가 약하면 "판단 보류" 또는 "근거 부족"이라고 말하세요.
- 답변은 한국어로 작성하고, 실무자가 바로 읽을 수 있게 간결하게 쓰세요.
- 전체 답변은 반드시 1,200자 이내로 작성하세요.
- 상세한 법리 설명보다 사용자가 바로 이해할 수 있는 결론, 핵심 근거, 확인사항을 우선하세요.
- 질의 분석에서 사용자의 질문을 특정 공정거래 쟁점으로 해석했다면,
  1. 진단 요약에 "이 질문은 '...' 쟁점으로 보고 검토했습니다."라고 짧게 밝히세요.
"""


ANSWER_FORMAT = """\
### 출력 형식(Output Format)
아래 번호와 제목을 그대로 사용하세요.

1. 진단 요약
- 위반 가능성을 "높음 / 중간 / 낮음 / 판단 보류" 중 하나로 표시하세요.
- 핵심 이유를 2문장 이내로 요약하세요.

2. 핵심 근거
- 가장 관련 높은 의결서 최대 2개만 제시하세요.
- 각 의결서는 "의결서명 / 위반유형 / 핵심 근거 / 근거 번호" 형식으로 작성하세요.
- 각 의결서 설명은 2문장 이내로 제한하세요.
- 검색 근거와 사용자 질문의 사실관계가 다르면 차이를 짧게 적으세요.

3. 확인 필요 사항
- 최종 판단 전에 확인해야 할 사실을 최대 4개 bullet로 제시하세요.
- 예: 동의/협의 여부, 거부 시 불이익, 비용 부담 비율, 계약서 근거, 업계 관행

4. 주의
- 이 답변은 공정위 의결서 기반 사전 검토이며, 최종 법률 판단이나 법률 자문이 아니라고 명시하세요.

### 답변 예시(Style Example)
1. 진단 요약
- 위반 가능성: 중간
- 제공된 근거상 유사 의결서에서는 가맹본부가 가맹점사업자에게 비용을 일방적으로 부담시키거나 선택권을 제한한 점이 문제되었습니다. 다만 실제 위반 여부는 계약 내용, 비용 부담 방식, 가맹점의 선택 가능성 확인이 필요합니다.

2. 핵심 근거
- 의결서명: 예시 의결서명
  - 위반유형: 예시 위반유형
  - 관련 근거: 예시 근거 요약
  - 근거 번호: [문서 1-근거 1]
"""


def format_context(doc_contexts, max_chars_per_chunk=1200):
    sections = []

    for doc_index, context in enumerate(doc_contexts, start=1):
        meta = context["meta"]
        lines = [
            f"[문서 {doc_index}]",
            f"의결서제목: {meta.get('의결서제목', '')}",
            f"위반유형: {meta.get('위반유형', '')}",
            f"업종: {meta.get('업종', '')}",
            f"조치유형: {meta.get('조치유형', '')}",
            f"피심인기업명: {meta.get('피심인기업명', '')}",
            f"대표 rerank 점수: {context['best_score']:.4f}",
        ]

        for chunk_index, chunk in enumerate(context["chunks"], start=1):
            chunk_text = chunk["doc"][:max_chars_per_chunk]
            lines.extend(
                [
                    f"\n[문서 {doc_index}-근거 {chunk_index}]",
                    f"rerank 점수: {chunk['score']:.4f}",
                    f"내용: {chunk_text}",
                ]
            )

        sections.append("\n".join(lines))

    return "\n\n---\n\n".join(sections)


def build_analyzer_prompt(question):
    return f"""\
당신은 공정거래위원회 의결서 기반 RAG 서비스의 질의 분석기입니다.
사용자 질문을 보고 공정거래 사전진단 서비스에서 바로 검색/답변할 수 있는지 판단하세요.

서비스가 다루는 주요 범위:
- 가맹본부/가맹점사업자 관계의 필수품목 구매 강제, 비용 전가, 영업지역 침해
- 플랫폼/입점업체 관계의 수수료, 광고비, 불이익 제공
- 제조사/대리점 관계의 재판매가격유지, 공급중단, 거래상 지위 남용
- 입찰담합, 가격담합 등 부당한 공동행위
- 제약/의료기기 리베이트, 부당한 고객유인
- 하도급, 대규모유통업 등 공정거래위원회 의결서로 유사 사례를 찾을 수 있는 쟁점

범위 밖 예:
- 최저임금, 주휴수당, 퇴직금, 근로계약 등 노동법
- 임대차, 세금, 형사, 가족관계, 일반 민원
- 단순히 옆에 경쟁 가게가 생긴 경우처럼 공정거래 쟁점이 아직 드러나지 않은 일반 상권 경쟁

판단 기준:
- scope는 반드시 "in_scope", "needs_clarification", "out_of_scope" 중 하나입니다.
- in_scope: 공정거래 쟁점이 충분히 해석 가능하여 검색용 질의로 재작성할 수 있음
- needs_clarification: 공정거래 쟁점일 수도 있지만, 핵심 사실이 부족하여 바로 답변하기 위험함
- out_of_scope: 공정거래 의결서 기반 사전진단 범위를 벗어남
- search_query는 의결서 검색에 적합한 법적/도메인 표현으로 작성하세요.
- needs_clarification이면 suggested_question에 사용자가 다시 물어볼 수 있는 예시 질문을 작성하세요.
- suggested_question은 사용자에게 되묻는 확인 질문이 아니라,
  사용자가 서비스에 그대로 다시 입력할 수 있는 완성형 질문이어야 합니다.
- suggested_question에는 원래 사용자 질문의 핵심 사실을 반드시 보존하세요.
- 부족한 사실을 물어보는 문장은 suggested_question에 넣지 말고 missing_facts에 넣으세요.
- suggested_question은 "A라면 B가 공정거래상 문제가 될 수 있나요?" 형식으로 작성하세요.
- out_of_scope 또는 needs_clarification의 reason에서도
  "위반이 아니다", "해당하지 않는다", "문제 없다"처럼 최종 법률 판단처럼 보이는 표현을 피하세요.
- 대신 "현재 제공된 사실만으로는 공정거래 쟁점이 뚜렷하지 않다",
  "공정위 의결서 기반 검색으로 직접 검토하기 어렵다"처럼 표현하세요.
- 일반적인 가격 할인, 단골 할인, 거래처 우대는 곧바로 위반이라고 보지 말되,
  시장 지위, 차별 조건, 경쟁사업자 배제 효과, 부당지원 관계가 있으면 추가 검토가 필요하다고 설명하세요.

반드시 JSON만 출력하세요. 마크다운 코드블록은 쓰지 마세요.

JSON 형식:
{{
  "scope": "in_scope | needs_clarification | out_of_scope",
  "interpreted_issue": "해석한 공정거래 쟁점 또는 범위 밖 분야",
  "search_query": "검색용 질의. out_of_scope이면 null",
  "needs_clarification": true,
  "suggested_question": "사용자가 더 정확히 물어볼 수 있는 예시 질문 또는 null",
  "missing_facts": ["추가로 필요한 사실"],
  "reason": "판단 이유 한 문장"
}}

[사용자 질문]
{question}
"""


def format_analysis_for_prompt(search_query, analysis=None):
    if not analysis:
        return "별도 질의 분석 없이 원문 또는 검색 질의를 사용합니다."

    lines = [
        f"scope: {analysis.get('scope', '')}",
        f"interpreted_issue: {analysis.get('interpreted_issue', '')}",
        f"reason: {analysis.get('reason', '')}",
    ]

    suggested_question = analysis.get("suggested_question")
    if suggested_question:
        lines.append(f"suggested_question: {suggested_question}")

    missing_facts = analysis.get("missing_facts") or []
    if missing_facts:
        lines.append("missing_facts:")
        lines.extend(f"- {fact}" for fact in missing_facts)

    return "\n".join(lines)


def build_prompt(question, search_query, context_text, analysis=None):
    return f"""\
{SYSTEM_INSTRUCTION}

[사용자 질문]
{question}

### 질의 분석(Query Analysis)
{format_analysis_for_prompt(search_query, analysis)}

### 검색에 사용된 질의(Search Query)
{search_query}

### 검색 근거(Retrieved Evidence)
{context_text}

{ANSWER_FORMAT}

### 답변
"""


def call_gemini(
    prompt,
    model,
    temperature=0.2,
    max_output_tokens=1200,
    thinking_budget=0,
    show_metadata=True,
):
    from google import genai
    from google.genai import types

    thinking_config = None
    if thinking_budget is not None:
        thinking_config = types.ThinkingConfig(
            thinking_budget=thinking_budget,
            include_thoughts=False,
        )

    client = genai.Client()
    response = client.models.generate_content(
        model=model,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=temperature,
            max_output_tokens=max_output_tokens,
            thinking_config=thinking_config,
        ),
    )

    if show_metadata and getattr(response, "candidates", None):
        candidate = response.candidates[0]
        print(f"Gemini finish_reason: {getattr(candidate, 'finish_reason', None)}")
        print(f"Gemini safety_ratings: {getattr(candidate, 'safety_ratings', None)}")

    return response.text.strip()


def extract_json_object(text):
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`")
        text = text.removeprefix("json").strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        raise ValueError("No JSON object found in analyzer response.")

    return json.loads(text[start:end + 1])


def analyze_query_with_llm(question, model, thinking_budget=0):
    prompt = build_analyzer_prompt(question)
    text = call_gemini(
        prompt,
        model=model,
        temperature=0.0,
        max_output_tokens=700,
        thinking_budget=thinking_budget,
        show_metadata=False,
    )
    analysis = extract_json_object(text)
    return normalize_analysis(question, analysis)


def normalize_analysis(question, analysis):
    scope = analysis.get("scope")
    if scope not in {"in_scope", "needs_clarification", "out_of_scope"}:
        scope = "in_scope"

    search_query = analysis.get("search_query")
    if scope == "in_scope" and not search_query:
        search_query = question

    if scope != "in_scope":
        search_query = None

    missing_facts = analysis.get("missing_facts") or []
    if not isinstance(missing_facts, list):
        missing_facts = [str(missing_facts)]

    return {
        "scope": scope,
        "interpreted_issue": analysis.get("interpreted_issue") or "",
        "search_query": search_query,
        "needs_clarification": bool(analysis.get("needs_clarification")),
        "suggested_question": analysis.get("suggested_question"),
        "missing_facts": [str(fact) for fact in missing_facts if str(fact).strip()],
        "reason": analysis.get("reason") or "",
        "source": analysis.get("source", "llm"),
    }


def analyze_query_by_rule(question):
    for issue, keywords in CLEAR_OUT_OF_SCOPE_KEYWORDS.items():
        if any(keyword in question for keyword in keywords):
            return {
                "scope": "out_of_scope",
                "interpreted_issue": issue,
                "search_query": None,
                "needs_clarification": False,
                "suggested_question": None,
                "missing_facts": [],
                "reason": (
                    f"질문이 공정거래위원회 의결서 기반 사전진단보다 "
                    f"{issue} 쟁점에 가깝습니다."
                ),
                "source": "rule",
            }

    return None


def analyze_query(question, args):
    if args.query_analysis == "off" or args.provider == "dry-run":
        return {
            "scope": "in_scope",
            "interpreted_issue": "",
            "search_query": question,
            "needs_clarification": False,
            "suggested_question": None,
            "missing_facts": [],
            "reason": "질의 분석을 사용하지 않았습니다.",
            "source": "disabled",
        }

    rule_analysis = analyze_query_by_rule(question)
    if rule_analysis:
        return rule_analysis

    if args.query_analysis == "rule-only":
        return {
            "scope": "in_scope",
            "interpreted_issue": "",
            "search_query": question,
            "needs_clarification": False,
            "suggested_question": None,
            "missing_facts": [],
            "reason": "명백한 범위 밖 규칙에 해당하지 않아 원문 질의를 사용합니다.",
            "source": "rule",
        }

    try:
        return analyze_query_with_llm(
            question,
            model=args.analyzer_model,
            thinking_budget=args.thinking_budget,
        )
    except Exception as exc:
        return {
            "scope": "in_scope",
            "interpreted_issue": "",
            "search_query": question,
            "needs_clarification": False,
            "suggested_question": None,
            "missing_facts": [],
            "reason": f"질의 분석 실패로 원문 질의를 사용합니다: {exc}",
            "source": "fallback",
        }


def format_analysis_preview(analysis):
    lines = [
        f"scope: {analysis.get('scope', '')}",
        f"source: {analysis.get('source', '')}",
        f"interpreted_issue: {analysis.get('interpreted_issue', '')}",
        f"reason: {analysis.get('reason', '')}",
    ]

    if analysis.get("search_query"):
        lines.append(f"search_query: {analysis['search_query']}")

    if analysis.get("suggested_question"):
        lines.append(f"suggested_question: {analysis['suggested_question']}")

    missing_facts = analysis.get("missing_facts") or []
    if missing_facts:
        lines.append("missing_facts:")
        lines.extend(f"- {fact}" for fact in missing_facts)

    return "\n".join(lines)


def build_out_of_scope_answer(question, analysis):
    issue = analysis.get("interpreted_issue") or "서비스 범위 밖 쟁점"

    return f"""\
1. 진단 요약
- 위반 가능성: 판단 보류
- 이 질문은 '{issue}' 쟁점에 가까워 현재 서비스의 공정거래위원회 의결서 기반 사전진단 범위에서 직접 판단하기 어렵습니다.
- 따라서 공정거래 의결서를 근거로 무리하게 위반 가능성을 판단하지 않고, 관련 분야의 별도 검토가 필요합니다.

2. 핵심 근거
- 직접 관련 근거 없음
  - 현재 질문에 직접 대응하는 공정거래 의결서 검색은 수행하지 않았습니다.

3. 확인 필요 사항
- 해당 분야를 담당하는 기관이나 전문가에게 확인이 필요한 사안인지 검토하세요.
- 공정거래 쟁점으로 보려면 사업자 간 거래관계, 우월적 지위, 담합, 가맹/대리점/플랫폼 관계 등과 연결되는 사실이 있는지 확인해야 합니다.

4. 주의
- 이 답변은 공정위 의결서 기반 사전진단 서비스의 범위 안내이며, 최종 법률 판단이나 법률 자문이 아닙니다.
"""


def build_clarification_answer(question, analysis):
    suggested_question = analysis.get("suggested_question")
    missing_facts = analysis.get("missing_facts") or []
    interpreted_issue = analysis.get("interpreted_issue") or "공정거래 쟁점 가능성"

    fact_lines = "\n".join(f"- {fact}" for fact in missing_facts[:5])
    if not fact_lines:
        fact_lines = "- 문제되는 거래관계, 계약 내용, 거부 시 불이익 여부를 추가로 확인해야 합니다."

    question_line = ""
    if suggested_question:
        question_line = f'\n\n아래처럼 사실관계를 보완해 다시 질문하면 더 정확히 검색할 수 있습니다.\n"{suggested_question}"'

    return f"""\
1. 진단 요약
- 위반 가능성: 판단 보류
- 현재 질문은 '{interpreted_issue}'와 관련될 수 있지만, 공정거래 위반 가능성을 판단하기에는 핵심 사실이 부족합니다.{question_line}

2. 핵심 근거
- 직접 관련 근거 없음
  - 추가 사실이 확인되기 전에는 특정 공정위 의결서와 무리하게 연결하지 않는 것이 적절합니다.

3. 확인 필요 사항
{fact_lines}

4. 주의
- 이 답변은 공정위 의결서 기반 사전진단 서비스의 질문 보완 안내이며, 최종 법률 판단이나 법률 자문이 아닙니다.
"""


def format_context_preview(doc_contexts, max_chars_per_chunk=220):
    sections = []

    for doc_index, context in enumerate(doc_contexts, start=1):
        meta = context["meta"]
        lines = [
            f"[문서 {doc_index}] {meta.get('의결서제목', '')}",
            f"- 위반유형: {meta.get('위반유형', '')}",
            f"- 대표 chunk: {context['best_doc_id']} (score: {context['best_score']:.4f})",
        ]

        for chunk_index, chunk in enumerate(context["chunks"], start=1):
            snippet = " ".join(chunk["doc"].split())[:max_chars_per_chunk]
            lines.append(
                f"- chunk {chunk_index}: {chunk['doc_id']} "
                f"(score: {chunk['score']:.4f}) {snippet}..."
            )

        sections.append("\n".join(lines))

    return "\n\n".join(sections)


def format_reference_ids(doc_contexts):
    lines = []

    for doc_index, context in enumerate(doc_contexts, start=1):
        meta = context["meta"]
        title = meta.get("의결서제목", "")
        lines.append(f"[문서 {doc_index}] {title}")
        lines.append(
            f"- 대표 chunk: {context['best_doc_id']} "
            f"(score: {context['best_score']:.4f})"
        )

        for chunk_index, chunk in enumerate(context["chunks"], start=1):
            lines.append(
                f"- [문서 {doc_index}-근거 {chunk_index}] {chunk['doc_id']} "
                f"(score: {chunk['score']:.4f})"
            )

        lines.append("")

    return "\n".join(lines).rstrip()


def print_samples():
    print("사용 가능한 샘플 질의:")
    for key, question in SAMPLE_QUESTIONS.items():
        print(f"\n[{key}]")
        print(question)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--provider",
        default=DEFAULT_PROVIDER,
        choices=["gemini", "dry-run"],
        help="LLM provider. dry-run은 API 호출 없이 prompt만 출력합니다.",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Gemini model name. 기본값: gemini-2.5-flash",
    )
    parser.add_argument(
        "--analyzer-model",
        default=DEFAULT_ANALYZER_MODEL,
        help="질의 분석에 사용할 Gemini model name. 기본값: gemini-2.5-flash",
    )
    parser.add_argument(
        "--question",
        default=SAMPLE_QUESTIONS["franchise_delivery_fee"],
    )
    parser.add_argument(
        "--sample",
        choices=sorted(SAMPLE_QUESTIONS.keys()),
        help="미리 정의한 샘플 질의를 사용합니다.",
    )
    parser.add_argument(
        "--list-samples",
        action="store_true",
        help="샘플 질의 목록만 출력합니다.",
    )
    parser.add_argument("--candidate-size", type=int, default=20)
    parser.add_argument("--doc-top-k", type=int, default=3)
    parser.add_argument("--chunks-per-doc", type=int, default=3)
    parser.add_argument("--max-chars-per-chunk", type=int, default=1200)
    parser.add_argument("--max-output-tokens", type=int, default=1200)
    parser.add_argument(
        "--query-analysis",
        default="auto",
        choices=["auto", "rule-only", "off"],
        help=(
            "질의 분석 방식. auto는 명백한 범위 밖 rule 후 LLM analyzer를 사용하고, "
            "rule-only는 명백한 범위 밖 rule만 사용하며, off는 원문 질의로 검색합니다."
        ),
    )
    parser.add_argument(
        "--thinking-budget",
        type=int,
        default=0,
        help="Gemini 2.5 thinking token budget. 0이면 thinking을 최소화합니다.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    return parser.parse_args()


def main():
    total_start = time.perf_counter()
    args = parse_args()

    if args.list_samples:
        print_samples()
        return

    question = SAMPLE_QUESTIONS[args.sample] if args.sample else args.question

    analysis_start = time.perf_counter()
    analysis = analyze_query(question, args)
    analysis_elapsed = time.perf_counter() - analysis_start

    print("\n" + "=" * 60)
    print("사용자 질문")
    print("=" * 60)
    print(question)

    print("\n" + "=" * 60)
    print("질의 분석")
    print("=" * 60)
    print(format_analysis_preview(analysis))

    if analysis["scope"] == "out_of_scope":
        total_elapsed = time.perf_counter() - total_start
        print("\n" + "=" * 60)
        print("RAG 답변 (범위 밖)")
        print("=" * 60)
        print(build_out_of_scope_answer(question, analysis))
        print("\n" + "=" * 60)
        print("실행 시간")
        print("=" * 60)
        print(f"질의 분석: {analysis_elapsed:.2f}초")
        print(f"전체: {total_elapsed:.2f}초")
        return

    if analysis["scope"] == "needs_clarification":
        total_elapsed = time.perf_counter() - total_start
        print("\n" + "=" * 60)
        print("RAG 답변 (추가 확인 필요)")
        print("=" * 60)
        print(build_clarification_answer(question, analysis))
        print("\n" + "=" * 60)
        print("실행 시간")
        print("=" * 60)
        print(f"질의 분석: {analysis_elapsed:.2f}초")
        print(f"전체: {total_elapsed:.2f}초")
        return

    search_query = analysis.get("search_query") or question

    from retriever import FaircheckRetriever, RetrieverConfig

    config = RetrieverConfig(
        candidate_size=args.candidate_size,
        doc_top_k=args.doc_top_k,
        chunks_per_doc=args.chunks_per_doc,
    )
    load_start = time.perf_counter()
    retriever = FaircheckRetriever(config)
    load_elapsed = time.perf_counter() - load_start

    search_start = time.perf_counter()
    search_result = retriever.search(
        search_query,
        candidate_size=args.candidate_size,
    )
    search_result["query"] = question
    search_result["search_query"] = search_query
    search_elapsed = time.perf_counter() - search_start

    context_text = format_context(
        search_result["doc_contexts"],
        max_chars_per_chunk=args.max_chars_per_chunk,
    )
    prompt = build_prompt(
        question=search_result["query"],
        search_query=search_result["search_query"],
        context_text=context_text,
        analysis=analysis,
    )

    print("\n" + "=" * 60)
    print("검색 질의")
    print("=" * 60)
    print(search_result["search_query"])

    print("\n" + "=" * 60)
    print("LLM 입력 근거 요약")
    print("=" * 60)
    print(format_context_preview(search_result["doc_contexts"]))

    if args.provider == "dry-run":
        print("\n" + "=" * 60)
        print("생성된 Prompt")
        print("=" * 60)
        print(prompt)
        total_elapsed = time.perf_counter() - total_start
        print("\n" + "=" * 60)
        print("실행 시간")
        print("=" * 60)
        print(f"질의 분석: {analysis_elapsed:.2f}초")
        print(f"검색기 로딩: {load_elapsed:.2f}초")
        print(f"검색/Rerank: {search_elapsed:.2f}초")
        print(f"전체: {total_elapsed:.2f}초")
        print("\n" + "=" * 60)
        print("참조 근거 ID")
        print("=" * 60)
        print(format_reference_ids(search_result["doc_contexts"]))
        return

    llm_start = time.perf_counter()
    answer = call_gemini(
        prompt,
        model=args.model,
        temperature=args.temperature,
        max_output_tokens=args.max_output_tokens,
        thinking_budget=args.thinking_budget,
    )
    llm_elapsed = time.perf_counter() - llm_start
    total_elapsed = time.perf_counter() - total_start

    print("\n" + "=" * 60)
    print(f"RAG 답변 ({args.provider}:{args.model})")
    print("=" * 60)
    print(answer)

    print("\n" + "=" * 60)
    print("참조 근거 ID")
    print("=" * 60)
    print(format_reference_ids(search_result["doc_contexts"]))

    print("\n" + "=" * 60)
    print("실행 시간")
    print("=" * 60)
    print(f"질의 분석: {analysis_elapsed:.2f}초")
    print(f"검색기 로딩: {load_elapsed:.2f}초")
    print(f"검색/Rerank: {search_elapsed:.2f}초")
    print(f"LLM 답변 생성: {llm_elapsed:.2f}초")
    print(f"전체: {total_elapsed:.2f}초")


if __name__ == "__main__":
    main()
