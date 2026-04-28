import pytest
from unittest.mock import patch, MagicMock
from fastapi import HTTPException
from app.services.model import HateSpeechModel

@pytest.fixture
def model_instance():
    """테스트용 모델 인스턴스 픽스처"""
    return HateSpeechModel(model_name="test-model")

@patch("app.services.model.pipeline")
def test_model_load_and_predict_clean(mock_pipeline, model_instance):
    """
    정상 문장(clean) 추론을 모킹(Mocking)하여 가중치 다운로드 없이 테스트합니다.
    """
    # Mocking HF pipeline
    mock_pipe_instance = MagicMock()
    mock_pipe_instance.return_value = [{'label': 'clean', 'score': 0.95}]
    mock_pipeline.return_value = mock_pipe_instance

    # 모델 로드 (실제로는 Mock 파이프라인이 할당됨)
    model_instance.load()
    assert model_instance.pipeline is not None

    # 추론 수행
    result = model_instance.predict("좋은 하루 되세요!")

    # 검증
    assert result["is_hate_speech"] is False
    assert result["confidence"] == 0.95
    assert result["message"] == "Detected category: clean"
    mock_pipe_instance.assert_called_once_with("좋은 하루 되세요!", truncation=True, max_length=512)

@patch("app.services.model.pipeline")
def test_model_predict_hate_speech(mock_pipeline, model_instance):
    """
    혐오 표현 추론 결과를 모킹하여 테스트합니다.
    """
    mock_pipe_instance = MagicMock()
    mock_pipe_instance.return_value = [{'label': '악플/욕설', 'score': 0.88}]
    mock_pipeline.return_value = mock_pipe_instance

    model_instance.load()
    result = model_instance.predict("나쁜 말 테스트")

    assert result["is_hate_speech"] is True
    assert result["confidence"] == 0.88
    assert result["message"] == "Detected category: 악플/욕설"

@patch("app.services.model.pipeline")
def test_model_predict_runtime_error_triggers_500(mock_pipeline, model_instance):
    """
    추론 중 RuntimeError 발생 시 500 에러를 던지는지 확인합니다.
    (OOM 상황 등에 대한 방어 로직 검증)
    """
    mock_pipe_instance = MagicMock()
    # 파이프라인 호출 시 RuntimeError 발생 모사
    mock_pipe_instance.side_effect = RuntimeError("Out of memory simulated error")
    mock_pipeline.return_value = mock_pipe_instance

    model_instance.load()
    
    with pytest.raises(HTTPException) as exc_info:
        model_instance.predict("긴 텍스트...")
        
    assert exc_info.value.status_code == 500
    assert "Internal Server Error" in exc_info.value.detail
