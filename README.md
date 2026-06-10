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
- Reranker: `BAAI/bge-reranker-v2-m3` 또는 `none`

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
build_chromadb.py              # ko-sroberta 기반 ChromaDB/BM25 구축 실험
build_chromadb_bge.py          # BGE-M3 기반 ChromaDB/BM25 구축
retrieval.py                   # 초기 검색 실험
retrieval_bge.py               # BGE-M3 기본 검색 실험
retrieval_bge_experiment.py    # BGE-M3 + BM25 + RRF + reranker baseline
statute_retrieval.py           # 법령 BGE-M3 + BM25 + RRF + reranker 검색기
search_statutes.py             # 법령 검색 CLI 실행 스크립트
compare_models.py              # 임베딩 모델 비교 실험
compare_models_fulltext.py     # fulltext 기반 모델 비교 실험
enrich_*.py                    # 의결서 metadata 보강 실험
eda_enriched.py                # 보강 metadata EDA
```

## Setup

Python 3.11 기준으로 실험했습니다.

```powershell
python -m pip install -r requirements.txt
```

Hugging Face 모델 캐시는 C 드라이브가 아니라 프로젝트 하위 캐시 경로를 사용합니다.

```powershell
$env:HF_HOME="D:\faircheck\hf_cache\huggingface"
$env:TRANSFORMERS_CACHE="D:\faircheck\hf_cache\transformers"
$env:SENTENCE_TRANSFORMERS_HOME="D:\faircheck\hf_cache\sentence_transformers"
$env:TORCH_HOME="D:\faircheck\hf_cache\torch"
```

`retrieval_bge_experiment.py`에는 위 캐시 경로가 import 전에 코드로도 설정되어 있습니다.

## Run Baseline Retrieval

필수 artifact가 준비된 상태에서 실행합니다.

```powershell
python retrieval_bge_experiment.py
```

전체 로그를 파일로 남기려면:

```powershell
python retrieval_bge_experiment.py 2>&1 | Tee-Object -FilePath bge_baseline_result.txt
```

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
```

검색 실행:

```powershell
# 법령 전용 검색기를 실행합니다.
python search_statutes.py "하도급 대금 지급 지연" --top-k 3
```

reranker 없이 RRF 결과만 빠르게 확인할 수도 있습니다.

```powershell
# BGE reranker를 생략하고 dense+BM25+RRF 결과만 봅니다.
python search_statutes.py "하도급 대금 지급 지연" --top-k 3 --reranker none
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

RRF와 reranker는 별도 DB를 생성하지 않습니다. 검색 요청이 들어올 때마다 ChromaDB와 BM25 결과를 가져온 뒤 코드에서 계산됩니다.

법령 검색 흐름도 같은 원리입니다.

```text
user query
  -> BGE-M3 query embedding
  -> statutes_bge dense search
  -> statute BM25 keyword search
  -> RRF candidate merge
  -> optional reranker
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

검색 실험은 `BAAI/bge-reranker-v2-m3` baseline으로 마무리하고, 다음 단계에서는 검색 로직을 `retriever.py`로 분리한 뒤 RAG 답변 생성 코드를 작성합니다.
