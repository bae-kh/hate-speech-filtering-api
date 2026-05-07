# Real-time AI Hate Speech Filtering API Server

[![CI](https://github.com/baekh/mlops/actions/workflows/ci.yml/badge.svg)](https://github.com/baekh/mlops/actions/workflows/ci.yml)

HuggingFace 기반 한국어 혐오 표현 분류 모델을 FastAPI 서버로 서빙하기 위한 MLOps / Backend 포트폴리오 프로젝트입니다.

이 프로젝트의 목표는 단순히 AI 모델을 사용하는 것이 아니라, 무거운 딥러닝 모델을 실제 서비스 API에 안정적으로 연결하기 위한 서버 구조, 입력 방어선, 모델 로딩 최적화, Docker 기반 실행 환경, CI 자동 검증 파이프라인을 설계하는 것입니다.

---

## 1. Problem

HuggingFace 모델은 로컬에서는 쉽게 실행할 수 있지만, 실제 API 서버에 연결하면 다음 문제가 발생합니다.

- 요청마다 모델을 로딩할 경우 응답 지연과 메모리 낭비 발생
- 빈 문자열이나 비정상 입력이 모델까지 전달될 경우 불필요한 연산 발생
- 긴 텍스트 입력으로 인한 모델 입력 길이 초과 및 RuntimeError 가능성
- 테스트/CI 환경에서 모델 로딩으로 인한 병목
- 로컬 환경과 배포 환경의 차이로 인한 재현성 문제
- Docker 환경에서 PyTorch, NumPy, HuggingFace cache 관련 의존성 문제 발생 가능성

---

## 2. Service Policy

이 API의 1차 사용 시나리오는 **실시간 게임 채팅 moderation**입니다.

현재 사용하는 `smilegate-ai/kor_unsmile` 모델은 특정 단어 위치를 반환하는 token classification 모델이 아니라, 문장 전체를 분류하는 text-classification 모델입니다.

따라서 특정 욕설 단어만 `*`로 마스킹하는 방식보다는, 메시지 전체를 `allow` 또는 `block`으로 처리하는 정책이 현재 모델 구조에 더 적합하다고 판단했습니다.

처리 정책은 다음과 같습니다.

```text
action = "allow"
→ 채팅 메시지를 정상 전송

action = "block"
→ 채팅 메시지를 전송하지 않고 사용자에게 경고
```

입력 제한은 게임 채팅 시나리오에 맞춰 설정했습니다.

```text
API input limit:
- max_length = 500 characters

Model input limit:
- max_length = 256 tokens
- truncation = True
```

Pydantic의 `max_length=500`은 HTTP 요청으로 들어오는 원문 문자열 길이를 제한하는 API 레벨 방어선입니다.

반면 HuggingFace pipeline의 `max_length=256`은 tokenizer 이후 실제 모델에 들어가는 token 수를 제한하는 모델 레벨 방어선입니다.

---

## 3. Features

- FastAPI 기반 REST API 서버
- `/api/v1/health` 헬스체크 엔드포인트
- `/api/v1/detect` 혐오 표현 분석 엔드포인트
- Pydantic 기반 request/response schema
- 빈 문자열, 공백 문자열, 과도한 길이 입력 차단
- Request ID middleware를 통한 요청 추적성 확보
- FastAPI lifespan 기반 모델 1회 로딩
- HuggingFace `smilegate-ai/kor_unsmile` 모델 연동
- `run_in_threadpool` 기반 동기 추론 격리
- 메시지 전체 `allow/block` 정책 적용
- Docker 기반 CPU-only 컨테이너 실행 환경
- pytest 기반 API 및 model service layer 테스트
- GitHub Actions 기반 CI 자동 검증

---

## 4. Tech Stack

| Area | Tech |
|---|---|
| Language | Python |
| Web Framework | FastAPI |
| Validation | Pydantic |
| Model Serving | HuggingFace Transformers, PyTorch CPU |
| Testing | Pytest, FastAPI TestClient, MagicMock |
| Containerization | Docker |
| CI | GitHub Actions |
| Runtime | Uvicorn |

---

## 5. Architecture

```text
Game Client
  ↓
Game Server
  ↓
POST /api/v1/detect
  ↓
FastAPI Middleware
  - X-Request-ID
  ↓
Router
  - /api/v1/health
  - /api/v1/detect
  ↓
Pydantic Schema
  - DetectRequest
  - DetectResponse
  ↓
Service Layer
  - HateSpeechModel
  ↓
HuggingFace Pipeline
  - smilegate-ai/kor_unsmile
  ↓
Response
  - action: allow / block
```

---

## 6. Project Structure

```text
app/
├── main.py
├── api/
│   └── routes.py
├── schemas/
│   └── payload.py
└── services/
    └── model.py

tests/
├── test_api.py
└── test_model.py

.github/
└── workflows/
    └── ci.yml

Dockerfile
.dockerignore
requirements.txt
README.md
```

---

## 7. Phase Summary

### Phase 1. API MVP

AI 모델을 붙이기 전에 API 서버의 기본 구조와 입출력 계약을 먼저 고정했습니다.

Implemented:

- FastAPI 앱 구성
- `/api/v1/health` 엔드포인트 추가
- `/api/v1/detect` 엔드포인트 추가
- Pydantic 기반 Request/Response Schema 정의
- 빈 문자열, 공백 문자열, 과도한 길이 입력 차단
- Request ID middleware 추가
- Mock 기반 API 테스트 작성

Key idea:

> 실제 AI 모델 없이 API 계약과 입력 검증을 먼저 고정해, 모델 연동 전에도 서버 로직을 테스트할 수 있는 구조를 만들었습니다.

---

### Phase 2. Model Serving

HuggingFace `smilegate-ai/kor_unsmile` 모델을 FastAPI 서버에 연동하고, 모델 로딩과 추론 과정에서 발생할 수 있는 자원 문제를 방어했습니다.

Implemented:

- `HateSpeechModel` 클래스로 모델 로딩/추론 로직 분리
- FastAPI lifespan 기반 모델 1회 로딩
- `app.state.model`을 통한 모델 인스턴스 재사용
- `/api/v1/detect`에서 이미 로딩된 모델 사용
- `truncation=True`, `max_length=256` 적용
- RuntimeError 발생 시 HTTP 500 응답으로 변환
- 동기 추론 작업을 `run_in_threadpool`로 격리
- Mock 기반 model service layer 테스트 작성

Key idea:

> 모델을 요청마다 로딩하지 않고 서버 시작 시 한 번만 로딩하도록 구성했고, 긴 입력과 추론 예외로 인해 API 서버가 불안정해지지 않도록 입력 제한과 예외 처리를 추가했습니다.

---

### Phase 3. Dockerization

로컬 환경에 의존하던 FastAPI 모델 서빙 서버를 Docker 컨테이너로 패키징했습니다.

Implemented:

- `python:3.11.9-slim-bookworm` 기반 Docker 이미지 구성
- CPU-only PyTorch wheel 직접 설치
- `numpy==1.26.4` 고정
- non-root user 적용
- HuggingFace cache 경로 설정
- `.dockerignore` 작성
- Docker build/run 및 컨테이너 API 호출 검증

Key idea:

> 로컬 환경 의존성을 제거하기 위해 Docker로 실행 환경을 캡슐화했고, ML 의존성으로 인한 이미지 크기와 빌드 안정성 문제를 layer cache, 버전 고정, CPU-only 의존성 구성으로 관리했습니다.

---

### Phase 4. CI

GitHub Actions를 사용해 테스트와 Docker build 검증을 자동화했습니다.

Implemented:

- `.github/workflows/ci.yml` 작성
- push / pull_request 이벤트 기반 CI 실행
- Python 3.11 runner 환경 구성
- PyTorch CPU-only wheel 설치
- 프로젝트 의존성 설치
- `pytest -v` 자동 실행
- Docker image build 자동 검증

Key idea:

> 코드 변경이 API contract, validation, model service layer, Dockerfile을 깨뜨리지 않는지 GitHub Actions에서 자동으로 검증할 수 있게 했습니다.

---

## 8. API

### Health Check

```http
GET /api/v1/health
```

Response:

```json
{
  "status": "ok",
  "message": "API server is running"
}
```

---

### Detect Text

```http
POST /api/v1/detect
Content-Type: application/json
```

Request:

```json
{
  "text": "테스트 문장입니다."
}
```

Response:

```json
{
  "is_hate_speech": false,
  "confidence": 0.8855019807815552,
  "category": "clean",
  "action": "allow",
  "message": "Message allowed."
}
```

Blocked response example:

```json
{
  "is_hate_speech": true,
  "confidence": 0.91,
  "category": "hate",
  "action": "block",
  "message": "Message blocked due to harmful content."
}
```

Validation error example:

```json
{
  "text": ""
}
```

Response:

```http
422 Unprocessable Entity
```

---

## 9. Run Locally

```bash
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Health check:

```bash
curl http://localhost:8000/api/v1/health
```

Detect:

```bash
curl -X POST "http://localhost:8000/api/v1/detect" \
  -H "Content-Type: application/json" \
  -d '{"text":"테스트 문장입니다."}'
```

PowerShell:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/detect" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text":"테스트 문장입니다."}'
```

---

## 10. Run with Docker

Build image:

```bash
docker build -t hate-speech-api .
```

Run container:

```bash
docker run --rm -p 8000:8000 hate-speech-api
```

PowerShell test:

```powershell
Invoke-RestMethod `
  -Uri "http://localhost:8000/api/v1/detect" `
  -Method POST `
  -ContentType "application/json" `
  -Body '{"text":"테스트 문장입니다."}'
```

---

## 11. Test

Run tests locally:

```bash
pytest -v
```

The test suite includes:

- `/api/v1/health` 정상 응답
- `/api/v1/detect` 정상 응답
- 빈 문자열 validation
- 공백 문자열 validation
- text 필드 누락 validation
- max_length 초과 validation
- model service layer 결과 변환 로직
- RuntimeError 발생 시 HTTP 500 처리

Note:

`test_model.py`는 실제 HuggingFace 모델을 다운로드하거나 로딩하는 테스트가 아닙니다.  
`MagicMock`을 사용해 HuggingFace pipeline의 반환값을 대체하고, `HateSpeechModel`의 결과 변환 로직과 예외 처리만 빠르게 검증합니다.

실제 모델 로딩과 추론은 Docker 컨테이너 실행 후 `/api/v1/detect`를 호출하는 smoke test로 별도 검증합니다.

---

## 12. CI

GitHub Actions를 사용해 테스트와 Docker build 검증을 자동화했습니다.

CI는 `push` 또는 `pull_request` 이벤트가 발생하면 실행됩니다.

검증 항목은 다음과 같습니다.

- Python 3.11 환경 구성
- PyTorch CPU-only wheel 설치
- 프로젝트 의존성 설치
- `pytest -v` 실행
- Docker image build 검증

Workflow file:

```text
.github/workflows/ci.yml
```

Local equivalent:

```bash
pytest -v
docker build -t hate-speech-api .
```

현재 CI는 실제 HuggingFace 모델을 다운로드하지 않습니다.  
실제 모델 로딩과 추론은 Docker run 기반 smoke test로 별도 검증합니다.

---

## 13. Requirements

Current `requirements.txt`:

```txt
fastapi==0.110.1
uvicorn==0.29.0
pydantic==2.6.4
numpy==1.26.4
transformers==4.39.3
pytest==8.1.1
httpx==0.27.0
```

Note:

Docker와 CI 환경에서는 PyTorch CPU-only wheel을 직접 설치합니다.  
따라서 `requirements.txt`에는 `torch`를 포함하지 않습니다.

---

## 14. Trouble Shooting

### 1. PyTorch CUDA dependency issue

초기에는 `requirements.txt`에 `torch==2.2.2`를 포함했습니다.

하지만 Docker 빌드 시 CUDA 관련 패키지가 대량으로 설치되며 이미지 크기와 빌드 안정성 문제가 발생했습니다.

해결을 위해 `requirements.txt`에서 torch를 제거하고, Dockerfile에서 CPU-only PyTorch wheel을 직접 설치했습니다.

---

### 2. Base image instability

`python:3.11-slim` 기반 빌드 중 pip/Python 표준 라이브러리 단계에서 비정상 오류가 발생했습니다.

해결을 위해 base image를 다음과 같이 고정했습니다.

```dockerfile
FROM python:3.11.9-slim-bookworm
```

---

### 3. NumPy runtime compatibility issue

컨테이너 실행 후 실제 추론 요청 시 다음 오류가 발생했습니다.

```text
Numpy is not available
```

PyTorch 2.2.2 CPU wheel과 최신 NumPy 계열의 호환성 문제로 판단하고, `numpy==1.26.4`로 고정하여 해결했습니다.

---

### 4. Python bytecode cache issue

Docker build 또는 container run 과정에서 다음 오류가 발생한 적이 있습니다.

```text
ValueError: bad marshal data (invalid reference)
```

이는 Python bytecode cache 또는 Docker layer 상태 문제로 판단했습니다.

완화 방법:

- `.dockerignore`에 `__pycache__/`, `*.py[cod]` 포함
- `pip install` 단계에 `--no-compile` 옵션 추가
- 필요 시 `docker build --pull --no-cache`로 재빌드

---

### 5. CI contract mismatch

API 응답 구조에 `category`, `action` 필드를 추가한 뒤, 테스트에서 사용하는 FakeModel이 예전 응답 구조를 반환해 CI가 실패했습니다.

```text
KeyError: 'category'
```

FakeModel의 반환값을 실제 API contract에 맞게 수정하여 해결했습니다.

이 경험을 통해 CI가 API contract 변경에 따른 테스트 불일치를 조기에 감지하는 안전장치로 동작함을 확인했습니다.

---

## 15. Limitations

현재 구조에는 다음 한계가 있습니다.

- CPU-only 추론 환경
- GPU inference 미적용
- 단일 Uvicorn 프로세스 기반 실행
- worker 수 튜닝 미진행
- Docker run 기반 smoke test는 CI에 아직 포함하지 않음
- 클라우드 배포 미진행
- 부하 테스트 미진행
- 모니터링 미구성

---

## 16. Future Work

다음 작업을 진행할 예정입니다.

- Docker image registry push
- Cloud VM 배포
- docker run 기반 smoke test 자동화
- `/api/v1/health` 자동 smoke test
- `/api/v1/detect` 샘플 요청 자동 smoke test
- Locust 또는 k6 기반 부하 테스트
- worker 수에 따른 latency, error rate 비교
- structured logging 및 monitoring 추가

---

## 17. Interview Summary

이 프로젝트는 단순히 HuggingFace 모델을 FastAPI에 붙인 것이 아니라, 무거운 딥러닝 모델을 실제 API 서버로 안정적으로 서빙하기 위해 API 계약, 입력 검증, 모델 로딩 최적화, Docker 기반 실행 환경, CI 자동 검증 파이프라인을 단계적으로 설계한 MLOps / Backend 프로젝트입니다.