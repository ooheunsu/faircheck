# FairCheck

FairCheck는 사용자가 자연어로 업종과 상황을 입력하면 공정거래위원회 의결서와 관련 법령을 검색해 위반 가능성 경고 근거를 보여주는 프로젝트입니다.

현재 이 저장소에는 SQLite DB 기반의 가벼운 FastAPI baseline이 들어 있습니다. BGE-M3 ChromaDB, BM25, RRF, reranker 파이프라인은 나중에 같은 API 응답 형태 뒤에 붙이면 됩니다.

## 실행 방법

```bash
python -m venv venv
./venv/bin/pip install -r requirements.txt
./venv/bin/uvicorn faircheck.api:app --reload
```

서버가 켜지면 브라우저에서 `http://127.0.0.1:8000/docs`를 열어 API를 테스트할 수 있습니다.

## 주요 API

- `GET /health`: DB 연결과 row 수 확인
- `POST /search/decisions`: 의결서 chunk를 검색하고 의결서 단위로 묶어서 반환
- `GET /decisions/{decision_id}`: 특정 의결서의 전체 chunk 조회
- `GET /search/statutes`: 법령 조문 검색
- `POST /risk-check`: 사용자 상황을 받아 유사 의결서와 관련 법령을 함께 반환

## 현재 검색 방식

현재 baseline은 모델을 쓰지 않고 SQLite의 `decisions`, `statutes` 테이블을 읽습니다. 질의를 키워드로 나누고 제목, 업종, 위반구체행위, 관련법령, 본문 chunk의 일치도를 점수화한 뒤 chunk 결과를 의결서 단위로 집계합니다.

이 방식은 설치와 실행이 가볍다는 장점이 있지만, 표현이 달라진 추상 질의에는 한계가 있습니다. 인수인계된 최종 목표 검색 파이프라인은 `BGE-M3 ChromaDB dense retrieval + BM25 sparse retrieval + RRF fusion + BGE reranker`입니다.

## 테스트

```bash
./venv/bin/python -m unittest discover -s tests -p 'test_*.py'
```
