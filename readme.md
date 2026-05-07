# Real-time AI Hate Speech Filtering API Server

[![CI](https://github.com/baekh/hate-speech-filtering-api/actions/workflows/ci.yml/badge.svg)](https://github.com/baekh/hate-speech-filtering-api/actions/workflows/ci.yml)
HuggingFace 기반 한국어 혐오 표현 분류 모델을 FastAPI 서버로 서빙하기 위한 MLOps / Backend 포트폴리오 프로젝트입니다.

이 프로젝트의 목표는 단순히 AI 모델을 사용하는 것이 아니라, 무거운 딥러닝 모델을 실제 서비스 API에 안정적으로 연결하기 위한 서버 구조, 입력 방어선, 모델 로딩 최적화, Docker 기반 실행 환경을 설계하는 것입니다.

---

## 1. Problem

HuggingFace 모델은 로컬에서는 쉽게 실행할 수 있지만, 실제 API 서버에 연결하면 다음 문제가 발생합니다.

- 요청마다 모델을 로딩할 경우 응답 지연과 메모리 낭비 발생
- 빈 문자열이나 비정상 입력이 모델까지 전달될 경우 불필요한 연산 발생
- 긴 텍스트 입력으로 인한 OOM 가능성
- 테스트/CI 환경에서 모델 로딩으로 인한 병목
- 로컬 환경과 배포 환경의 차이로 인한 재현성 문제

---

## 2. Features

- FastAPI 기반 REST API 서버
- `/api/v1/health` 헬스체크 엔드포인트
- `/api/v1/detect` 혐오 표현 분석 엔드포인트
- Pydantic 기반 request/response schema
- 빈 문자열, 공백 문자열, 과도한 길이 입력 차단
- Request ID middleware를 통한 요청 추적성 확보
- FastAPI lifespan 기반 모델 1회 로딩
- HuggingFace `smilegate-ai/kor_unsmile` 모델 연동
- Docker 기반 CPU-only 컨테이너 실행 환경
- pytest 기반 API 및 모델 테스트

---

## 3. Tech Stack

| Area | Tech |
|---|---|
| Language | Python |
| Web Framework | FastAPI |
| Validation | Pydantic |
| Model Serving | HuggingFace Transformers, PyTorch |
| Testing | Pytest, FastAPI TestClient |
| Containerization | Docker |
| Runtime | Uvicorn |

---

## 4. Architecture

```text
Client
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
```

---

## 5. Project Structure

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

Dockerfile
.dockerignore
requirements.txt
README.md
```

---

## 6. Phase Summary

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

> Phase 1에서는 실제 AI 모델 없이 API 계약과 입력 검증을 먼저 고정해, 모델 연동 전에도 서버 로직을 테스트할 수 있는 구조를 만들었습니다.

---

### Phase 2. Model Serving

HuggingFace `smilegate-ai/kor_unsmile` 모델을 FastAPI 서버에 연동하고, 모델 로딩과 추론 과정에서 발생할 수 있는 자원 문제를 방어했습니다.

Implemented:

- HuggingFace `smilegate-ai/kor_unsmile` 모델 연동
- FastAPI lifespan 기반 모델 1회 로딩
- `app.state.model`을 통한 모델 인스턴스 재사용
- `truncation=True`, `max_length=512` 적용
- RuntimeError 발생 시 HTTP 500으로 변환
- 동기 추론 작업을 threadpool로 격리
- 모델 결과 해석 및 예외 처리 테스트 작성

Key idea:

> Phase 2에서는 모델을 요청마다 로딩하지 않고 서버 시작 시 한 번만 로딩하도록 구성했고, 긴 입력과 추론 예외로 인해 API 서버가 불안정해지지 않도록 입력 제한과 예외 처리를 추가했습니다.

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

> Phase 3에서는 로컬 환경 의존성을 제거하기 위해 Docker로 실행 환경을 캡슐화했고, ML 의존성으로 인한 이미지 크기와 빌드 안정성 문제를 layer cache, 버전 고정, CPU-only 의존성 구성으로 관리했습니다.

---

## 7. API

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
  "message": "Detected category: clean"
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

## 8. Run Locally

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

## 9. Run with Docker

Build image:

```bash
docker build --pull --no-cache -t hate-speech-api .
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

Example response:

```text
is_hate_speech: False
confidence: 0.8855
message: Detected category: clean
```

---

## 10. Test

Run tests:

```bash
pytest -v
```

Current test result:

```text
9 passed
```

Test coverage includes:

- `/api/v1/health` 정상 응답
- `/api/v1/detect` 정상 응답
- 빈 문자열 validation
- 공백 문자열 validation
- text 필드 누락 validation
- max_length 초과 validation
- 모델 결과 해석
- RuntimeError 발생 시 HTTP 500 처리

---

## 11. Requirements

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

> Docker 환경에서는 PyTorch CPU-only wheel을 Dockerfile에서 직접 설치합니다.  
> 따라서 `requirements.txt`에는 `torch`를 포함하지 않습니다.

Dockerfile에서 PyTorch CPU-only wheel을 직접 설치하는 이유는 일반 `torch==2.2.2` 설치 시 CUDA 관련 의존성이 대량으로 설치되어 이미지 크기와 빌드 안정성 문제가 발생했기 때문입니다.

---

## 12. Trouble Shooting

### 1. PyTorch CUDA dependency issue

초기에는 `requirements.txt`에 `torch==2.2.2`를 포함했습니다.

하지만 Docker 빌드 시 다음과 같은 CUDA 관련 패키지가 대량으로 설치되며 빌드가 실패했습니다.

```text
nvidia-cublas-cu12
nvidia-cudnn-cu12
nvidia-cusparse-cu12
triton
```

해결을 위해 `requirements.txt`에서 torch를 제거하고, Dockerfile에서 CPU-only PyTorch wheel을 직접 설치했습니다.

---

### 2. Base image instability

`python:3.11-slim` 기반 빌드 중 pip/Python 표준 라이브러리 단계에서 비정상 오류가 발생했습니다.

해결을 위해 base image를 다음과 같이 고정했습니다.

```dockerfile
FROM python:3.11.9-slim-bookworm
```

이를 통해 빌드 재현성을 높이고 PyTorch/Transformers 런타임 호환성을 안정화했습니다.

---

### 3. NumPy runtime compatibility issue

컨테이너 실행 후 실제 추론 요청 시 다음 오류가 발생했습니다.

```text
Numpy is not available
```

PyTorch 2.2.2 CPU wheel과 최신 NumPy 계열의 호환성 문제로 판단하고, `numpy==1.26.4`로 고정하여 해결했습니다.

---

## 13. Limitations

현재 구조에는 다음 한계가 있습니다.

- CPU-only 추론 환경
- 단일 Uvicorn 프로세스 기반 실행
- 모델은 컨테이너 시작 시 다운로드됨
- GPU inference 미적용
- worker 수 튜닝 미진행
- 부하 테스트 미진행
- 클라우드 배포 미진행
- 모니터링 미구성

---

## 14. Future Work

다음 Phase에서는 아래 작업을 진행할 예정입니다.

- GitHub Actions 기반 CI 구성
- push / pull request 시 pytest 자동 실행
- Docker build 검증 자동화
- Cloud VM 배포
- Locust 또는 k6 기반 부하 테스트
- worker 수에 따른 latency, error rate 비교
- structured logging 및 monitoring 추가

---

## 15. Interview Summary

이 프로젝트는 단순히 HuggingFace 모델을 FastAPI에 붙인 것이 아니라, 무거운 딥러닝 모델을 실제 API 서버로 안정적으로 서빙하기 위해 API 계약, 입력 검증, 모델 로딩 최적화, Docker 기반 실행 환경을 단계적으로 설계한 MLOps / Backend 프로젝트입니다.

## Service Policy

이 API의 1차 사용 시나리오는 실시간 게임 채팅 moderation입니다.

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

## PHASE 4. CI

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