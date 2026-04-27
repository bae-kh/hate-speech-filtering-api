import pytest
from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_detect_success() -> None:
    """정상적인 텍스트 입력 시 200 OK와 올바른 JSON 구조, 헤더를 반환하는지 검증합니다."""
    payload: dict[str, str] = {"text": "테스트 문장"}
    response = client.post("/api/v1/detect", json=payload)
    
    assert response.status_code == 200
    
    # 1. 헤더 검증 (대규모 트래픽 추적용 UUID)
    assert "x-request-id" in response.headers
    assert len(response.headers["x-request-id"]) > 0

    # 2. JSON 응답 구조 및 값 검증
    data: dict[str, any] = response.json()
    assert "is_hate_speech" in data
    assert "confidence" in data
    assert "message" in data
    
    assert data["is_hate_speech"] is False
    assert data["confidence"] == 0.0
    assert data["message"] == "Hello SKT - MVP Ready"

def test_detect_empty_text() -> None:
    """빈 문자열 입력 시 Pydantic 검증에 의해 422 Unprocessable Entity 에러가 발생하는지 검증합니다."""
    payload: dict[str, str] = {"text": ""}
    response = client.post("/api/v1/detect", json=payload)
    
    assert response.status_code == 422
    assert "x-request-id" in response.headers

def test_detect_missing_text() -> None:
    """필수 파라미터(text) 누락 시 422 에러가 발생하는지 검증합니다."""
    payload: dict[str, str] = {}
    response = client.post("/api/v1/detect", json=payload)
    
    assert response.status_code == 422
    assert "x-request-id" in response.headers
