# Faircheck

한국어 공정위 의결서 검색 및 RAG 실험 프로젝트입니다.

현재 목표는 공정위 의결서 chunk를 대상으로 hybrid retrieval pipeline을 구성하고, 이후 RAG 답변 생성 단계에서 재사용 가능한 검색기를 만드는 것입니다.

## Service Concept

서비스명: 공정거래 위반 가능성 사전진단 서비스

공정위 의결서 검색과 RAG를 활용해 사용자가 입력한 거래 관행이나 행위 설명에 대해 유사 의결서, 관련 위반유형, 판단 근거를 제공하는 사전진단 보조 서비스를 목표로 한다.

## 현재 Baseline

### 의결서 검색

- Dense embedding: `BAAI/bge-m3`
- Vector DB: ChromaDB
- ChromaDB path: `db/chroma_bge`
- ChromaDB collection: `decisions_bge`
- Keyword index: BM25
- BM25 index path: `db/bm25_index.pkl`
- Candidate merge: RRF
- Reranker: `BAAI/bge-reranker-v2-m3`
- Candidate size: `50`
- RRF K: `60`
- Dense weight: `1.0`
- BM25 weight: `1.0`

### 법령 검색

- Dense embedding: `BAAI/bge-m3`
- Vector DB: ChromaDB
- ChromaDB collection: `statutes_bge`
- Keyword index: statute BM25
- Candidate merge: RRF

의결서와 법령은 별도 저장소와 별도 검색기로 검색하고, 최종 RAG 분석 context에서 합칩니다. 즉, 의결서 검색 결과가 법령 검색 인덱스를 직접 오염시키지 않고, 법령 검색 결과도 의결서 chunk 검색 점수에 섞이지 않습니다.

## Repository Scope

이 저장소에는 코드, 실험 스크립트, 설정, 문서만 커밋합니다.

다음 파일과 디렉터리는 Git에 포함하지 않습니다.

- `data/`
- `db/`
- `hf_cache/`
- `*.pkl`
- `.env`

따라서 새 환경에서 실행하려면 별도로 데이터와 artifact를 준비해야 합니다.

필수 artifact:

```text
data/
db/chroma_bge/
db/bm25_index.pkl
db/statute_chroma_bge/
db/statute_bm25_index.pkl
```

## 주요 파일

```text
decision_retriever.py          # 의결서 BGE-M3 + BM25 + RRF + reranker 검색기
statute_retriever.py           # 법령 BGE-M3 + BM25 + RRF 검색기
risk_analysis_pipeline.py      # 의결서 검색 + 법령 검색 + Gemini 답변 생성 통합 실행 파일
rag_answer.py                  # 질의 분석, RAG 프롬프트, Gemini 호출 함수
api.py                         # FastAPI/Pydantic API endpoint
run_risk_analysis_batch.py     # 여러 사용자 쿼리를 한 번에 실행하고 txt 결과 저장
search_statutes.py             # 법령 검색기만 단독 확인하는 CLI
test_statute_retriever.py      # 법령 검색 보조 함수 테스트
```

## Setup

Python 3.11 기준으로 실험했습니다.

```powershell
python -m pip install -r requirements.txt
```

Hugging Face 모델 캐시 경로는 필요할 때 `.env`에서 `FAIRCHECK_HF_HOME` 등으로 지정할 수 있습니다. 생략하면 Hugging Face 기본 캐시를 사용합니다.

## Run Statute Retrieval

법령 검색용 ChromaDB와 BM25 pickle은 큰 산출물이므로 Git에 넣지 않고 `.env`로 위치만 지정합니다. `.env.example`을 참고해 `.env`를 만듭니다.

```powershell
# .env.example을 복사해서 개인 실행 설정 파일을 만듭니다.
Copy-Item .env.example .env
```

`.env`에는 실제 로컬 경로를 적습니다.

```powershell
# 법령 ChromaDB 폴더입니다. 이 폴더 안에 chroma.sqlite3가 있어야 합니다.
FAIRCHECK_STATUTE_CHROMA_DIR=C:\path\to\faircheck_chroma

# 법령 BM25 pickle 파일입니다.
FAIRCHECK_STATUTE_BM25_PATH=C:\path\to\bm25_index.pkl

# 예전 ko-sroberta 컬렉션이 아니라 BGE-M3 컬렉션을 지정합니다.
FAIRCHECK_STATUTE_COLLECTION=statutes_bge

# 의결서 원문 PDF 폴더입니다. 프론트에서 의결서 근거를 누르면 이 폴더의 PDF를 엽니다.
FAIRCHECK_DECISION_PDF_DIR=C:\path\to\faircheck-repo\db\decision_originals
```

검색 실행:

```powershell
# 법령 전용 검색기를 실행합니다.
python search_statutes.py "하도급 대금 지급 지연" --top-k 3
```

## Run Risk Analysis Pipeline

의결서 검색, 법령 검색, Gemini 답변 생성을 함께 실행합니다.

```powershell
# Gemini 호출 없이 통합 프롬프트까지만 확인합니다.
python risk_analysis_pipeline.py --provider dry-run --question "하도급 대금 지급 지연"

# Gemini까지 호출해 최종 답변을 생성합니다.
python risk_analysis_pipeline.py --provider gemini --question "하도급 대금 지급 지연"
```

여러 쿼리를 한 번에 실행하고 txt 파일로 저장하려면:

```powershell
python run_risk_analysis_batch.py --provider gemini --output risk_analysis_8q_results.txt
```

## Run FastAPI

프론트엔드 연결용 API 서버를 실행합니다.

```powershell
# 로컬 API 서버를 실행합니다.
uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

주요 endpoint:

```text
GET  /health
POST /api/risk-analysis
```

요청 예시:

```json
{
  "question": "커피 가맹점주인데 계약서상 보장된 영업지역 100m 안에 본사가 직영점을 열겠다고 합니다.",
  "provider": "gemini",
  "query_analysis": "auto",
  "include_prompt": false
}
```

응답은 세 가지 경우를 같은 JSON 구조로 반환합니다.

```text
result_type = "risk_analysis"         # 검색과 답변 생성까지 수행
result_type = "needs_clarification"   # 추가 사실 확인이 필요해 검색 전 보수적으로 중단
result_type = "out_of_scope"          # 공정거래 의결서 기반 서비스 범위 밖
```

`risk_analysis` 응답에서는 근거 목록을 두 방식으로 제공합니다.

```text
decision_references / statute_references
  -> 검색과 법령 조회로 확보한 전체 후보 근거입니다. 디버깅이나 추가 검토에 사용합니다.

used_decision_references / used_statute_references
  -> 최종 answer 본문에서 실제로 [문서 n], [법령 n] 형태로 인용된 핵심 근거입니다.

additional_decision_references / additional_statute_references
  -> 후보에는 있었지만 최종 answer 본문에서 직접 인용되지는 않은 추가 참고 근거입니다.
```

## Run Frontend

프론트엔드는 `frontend/` 폴더의 Vite + React 앱입니다. 백엔드 API 서버를 먼저 실행한 뒤 프론트엔드를 실행합니다.

```powershell
# 프론트엔드 폴더로 이동합니다.
cd frontend

# 최초 1회 의존성을 설치합니다.
npm install

# 로컬 프론트엔드 서버를 실행합니다.
npm run dev
```

브라우저에서 아래 주소를 엽니다.

```text
http://127.0.0.1:5173
```

프론트엔드는 기본적으로 아래 백엔드 API를 호출합니다.

```text
http://127.0.0.1:8000/api/risk-analysis
```

## Retrieval Pipeline

현재 검색 흐름은 다음과 같습니다.

```text
user query
  -> optional query rewrite
  -> BGE-M3 query embedding
  -> ChromaDB dense search
  -> BM25 keyword search
  -> RRF candidate merge
  -> BGE reranker
  -> chunk results / document results / document context candidates
```

의결서 검색의 RRF와 reranker는 별도 DB를 생성하지 않습니다. 검색 요청이 들어올 때마다 ChromaDB와 BM25 결과를 가져온 뒤 코드에서 계산됩니다.

법령 검색은 속도와 검색 품질 비교 결과를 반영해 reranker 없이 RRF까지만 적용합니다.

```text
user query
  -> BGE-M3 query embedding
  -> statutes_bge dense search
  -> statute BM25 keyword search
  -> RRF candidate merge
  -> top_k statute results
```

## Evaluation

검색 및 RAG 성능 평가는 정량 지표와 정성 평가를 함께 사용한다.

검토 중인 정량 지표:

- Recall@5
- MRR
- BERTScore
- F1

## Next Step

다음 단계에서는 프론트엔드에서 `/api/risk-analysis`를 호출해 입력창, 로딩 상태, 답변 영역, 의결서/법령 근거 카드를 구성합니다. 이후 의결서 카드 클릭 시 원문 PDF와 연결할 수 있도록 `decision_references`의 `pdf_source`와 chunk 정보를 활용합니다.

## 담당 역할
- RAG 검색 기능 및 공정거래 의결서 검색 기능 구현
- FastAPI 기반 API 개발 및 검색 결과 UI 연동
