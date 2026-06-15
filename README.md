# FairCheck

FairCheck는 공정거래위원회 의결서 검색과 RAG를 활용한 **공정거래 위반 가능성 사전진단 서비스** 실험 프로젝트입니다.

사용자가 거래 관행이나 사업 행위를 자연어로 입력하면, 시스템은 유사한 공정위 의결서를 검색하고 검색 근거를 바탕으로 위반 가능성, 유사 의결서, 확인해야 할 사실을 구조화해 제공합니다.

## Service Goal

본 프로젝트의 목표는 법률 전문가가 아닌 실무자도 공정거래 리스크를 초기에 검토할 수 있도록 돕는 것입니다.

주요 사용자는 다음과 같습니다.

- 가맹본부 / 가맹점사업자
- 플랫폼 입점업체
- 제조사 / 대리점
- 입찰 참여 사업자
- 제약 / 의료기기 영업 담당자
- 사내 컴플라이언스 담당자

이 서비스는 최종 법률 판단이나 법률 자문을 제공하지 않습니다. 공정위 의결서 기반의 유사 사례 검색과 사전 검토를 목적으로 합니다.

## Current Baseline

현재 기준 baseline은 다음과 같습니다.

| 구분 | 설정 |
|---|---|
| Embedding model | `BAAI/bge-m3` |
| Vector DB | ChromaDB |
| ChromaDB path | `db/chroma_bge` |
| ChromaDB collection | `decisions_bge` |
| Keyword retrieval | BM25 |
| BM25 index | `db/bm25_index.pkl` |
| Candidate merge | RRF |
| Reranker | `BAAI/bge-reranker-v2-m3` |
| RRF K | `60` |
| Dense weight | `1.0` |
| BM25 weight | `1.0` |
| Default candidate size | `20` for RAG, `50` for baseline experiment |
| Default document context | top 3 documents x 3 chunks x 1,200 chars |

## Pipeline

검색과 답변 생성은 다음 흐름으로 동작합니다.

```text
User Question
  -> Query Analysis
     - in_scope
     - needs_clarification
     - out_of_scope
  -> Query Rewrite for retrieval
  -> BGE-M3 dense search using ChromaDB
  -> BM25 keyword search
  -> RRF candidate merge
  -> BGE reranker
  -> Document-level aggregation
  -> Document context construction
  -> Gemini answer generation
```

`out_of_scope` 또는 `needs_clarification`으로 분류된 질문은 무리하게 의결서 검색을 수행하지 않습니다. 범위 밖 질문은 직접 관련 근거 없음으로 안내하고, 사실관계가 부족한 질문은 사용자가 다시 입력할 수 있는 보완 질문을 제안합니다.

## Repository Scope

이 저장소에는 코드, 설정 파일, 평가셋, 평가 결과만 커밋합니다. 원본 데이터, 모델 캐시, 대용량 DB 파일은 Git에 직접 포함하지 않습니다.

Git에 포함하지 않는 주요 항목:

```text
data/
db/*
hf_cache/
.env
*.pkl
__pycache__/
```

단, DVC 추적 파일은 Git에 포함합니다.

```text
db/chroma_bge.dvc
db/bm25_index.pkl.dvc
```

## Required Artifacts

실행을 위해 다음 artifact가 필요합니다.

```text
db/chroma_bge/
db/bm25_index.pkl
```

DVC remote 또는 별도 공유 경로에서 위 artifact를 받은 뒤 실행해야 합니다.

## Setup

Python 3.11 환경에서 실험했습니다.

```powershell
python -m pip install -r requirements.txt
```

Gemini API를 사용하는 답변 생성과 LLM-as-a-Judge 평가에는 API 키가 필요합니다. 실제 키는 `.env`에만 저장하고 커밋하지 않습니다.

```env
GOOGLE_API_KEY=your_google_api_key_here
```

Hugging Face 모델 캐시는 C 드라이브 사용량 증가를 막기 위해 프로젝트 하위 경로 사용을 권장합니다.

```powershell
$env:HF_HOME="D:\faircheck\hf_cache\huggingface"
$env:TRANSFORMERS_CACHE="D:\faircheck\hf_cache\transformers"
$env:SENTENCE_TRANSFORMERS_HOME="D:\faircheck\hf_cache\sentence_transformers"
$env:TORCH_HOME="D:\faircheck\hf_cache\torch"
```

## Main Files

| 파일 | 역할 |
|---|---|
| `retriever.py` | BGE-M3 + BM25 + RRF + BGE reranker 기반 의결서 검색기 |
| `rag_answer.py` | 질의 분석, 프롬프트 구성, Gemini 답변 생성 CLI |
| `run_retriever_demo.py` | 검색기 동작 확인용 demo |
| `evaluation_queries.json` | 검색/답변 평가용 15개 질의셋 |
| `evaluate_retrieval.py` | 검색 성능 평가 스크립트 |
| `evaluate_answers.py` | RAG 답변 품질 평가 스크립트 |
| `retrieval_bge_experiment.py` | BGE hybrid retrieval 실험용 baseline 스크립트 |
| `build_chromadb_bge.py` | BGE-M3 ChromaDB 구축 스크립트 |
| `compare_models*.py` | 임베딩 모델 비교 실험 |
| `enrich_*.py` | 의결서 metadata 보강 실험 |

`retrieval.py`, `retrieval_bge.py`, `test_search*.py` 등은 초기 실험용 파일입니다.

## Run Retrieval Demo

```powershell
python run_retriever_demo.py
```

## Run RAG Answer

샘플 질의 목록:

```powershell
python rag_answer.py --list-samples
```

샘플 질의 실행:

```powershell
python rag_answer.py --sample franchise_delivery_fee --max-output-tokens 1500
```

직접 질문 실행:

```powershell
python rag_answer.py --question "제조사가 대리점에게 온라인 최저 판매가격을 정해주고 지키지 않으면 공급을 중단하겠다고 하는 경우 문제가 될 수 있어?"
```

API 비용 없이 prompt만 확인하려면:

```powershell
python rag_answer.py --provider dry-run --sample resale_price
```

## Evaluation

평가는 검색 평가, 답변 품질 평가, 속도 평가로 나누어 수행했습니다.

### Retrieval Evaluation

검색 평가는 `evaluation_queries.json`의 15개 질의를 사용했습니다.

질의 유형:

| 유형 | 개수 | 의미 |
|---|---:|---|
| `in_scope` | 9 | 공정위 의결서 기반 RAG로 답변 가능한 질의 |
| `needs_clarification` | 3 | 공정거래 쟁점 가능성은 있으나 핵심 사실이 부족한 질의 |
| `out_of_scope` | 3 | 노동법, 소비자보상 등 서비스 범위 밖 질의 |

실행:

```powershell
python evaluate_retrieval.py
```

결과 파일:

```text
eval_results/retrieval_eval.csv
```

검색 평가 결과:

| 지표 | 결과 |
|---|---:|
| Scope Accuracy | 100% |
| Document Hit@5 | 100% |
| Document MRR | 0.759 |
| Avg query analysis time | 2.60 sec |
| Avg search/rerank time | 59.01 sec |

정답 chunk 라벨이 없기 때문에 공모전식 chunk Recall@5 대신, 팀 내부에서 구축한 gold 의결서 제목을 기준으로 문서 단위 Hit@5/MRR을 사용했습니다.

### Answer Evaluation

답변 품질 평가는 `in_scope` 9개 질의에 대해 수행했습니다.

평가 항목은 RAGAS의 주요 평가 관점인 Faithfulness, Answer Relevance, Context Precision을 참고하고, 서비스 목적에 맞게 Practical Usefulness를 추가했습니다.

| 항목 | 의미 |
|---|---|
| Faithfulness | 답변이 검색 근거 밖 내용을 만들지 않았는가 |
| Answer Relevance | 사용자 질문에 맞게 답했는가 |
| Practical Usefulness | 실무자가 확인할 사실과 리스크 포인트를 제시했는가 |
| Context Precision | 검색 근거가 질문과 실제로 관련 있는가 |

실행:

```powershell
python evaluate_answers.py --output eval_results/answer_eval_full.csv
```

결과 파일:

```text
eval_results/answer_eval_full.csv
```

답변 품질 평가 결과:

| 지표 | 평균 점수 |
|---|---:|
| Faithfulness | 4.33 / 5 |
| Answer Relevance | 4.89 / 5 |
| Practical Usefulness | 4.89 / 5 |
| Context Precision | 4.22 / 5 |
| Overall average | 4.58 / 5 |

### Speed Evaluation

속도 평가는 `answer_eval_full.csv`의 시간 열을 기준으로 정리했습니다.

| 항목 | 평균 시간 |
|---|---:|
| Query analysis | 2.38 sec |
| Search/Rerank | 39.55 sec |
| LLM answer generation | 4.94 sec |
| Service response time | about 46.87 sec |

LLM-as-a-Judge 시간은 평가용 후처리이므로 서비스 응답 시간에는 포함하지 않았습니다.

## Findings

검색 평가는 모든 `in_scope` 질의에서 정답 의결서가 상위 5개 안에 포함되어 Hit@5 100%를 기록했습니다. 다만 일부 질의에서는 정답 문서가 2~3위에 위치해 MRR은 0.759로 측정되었습니다.

답변 평가는 질문 적합성과 실무 유용성이 높게 나타났습니다. 반면 일부 질의에서는 검색 근거 중 직접 관련성이 낮은 문서가 포함되어 Context Precision과 Faithfulness가 상대적으로 낮아졌습니다.

대표적으로 배달앱 수수료 전가 질의는 관련 의결서를 찾았지만 일부 context가 직접적이지 않아 개선 여지가 있었습니다.

## Limitations

현재 실험에는 다음 한계가 있습니다.

- 평가셋이 15개로 작아 전체 서비스 성능을 일반화하기 어렵습니다.
- gold 문서는 팀 내부 수동 라벨이므로 주관성이 존재합니다.
- 정답 chunk 라벨이 없어 Context Recall과 chunk Recall@5를 엄밀히 계산하지 않았습니다.
- 답변 품질 평가는 LLM-as-a-Judge 기반이므로 평가 모델의 편향 가능성이 있습니다.
- CPU 환경에서 BGE reranker를 사용해 검색/Rerank 시간이 길게 측정되었습니다.
- 일부 질의는 검색 근거의 직접 관련성이 낮아 context filtering 개선이 필요합니다.

## Future Work

- FastAPI/프론트엔드 통합
- 검색 속도 개선: GPU 사용, 후보 수 조정, reranker 경량화
- gold chunk 라벨 구축 후 Context Recall 및 chunk Recall@5 평가
- 평가셋 확장
- 법령 DB 연동 안정화
- 약점 질의의 query rewriting 및 context filtering 개선
- PDF 원문 연결 및 근거 카드 UI 개선

## Branch Notes

현재 브랜치의 중심 작업은 RAG 검색/답변 평가입니다. FastAPI와 프론트엔드 통합은 별도 브랜치에서 진행 중입니다.
