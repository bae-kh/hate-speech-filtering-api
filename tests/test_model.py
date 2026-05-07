from typing import Any
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from app.services.model import HateSpeechModel


def test_model_load_and_predict_clean() -> None:
    model = HateSpeechModel()
    model.pipeline = MagicMock(return_value=[
        {
            "label": "clean",
            "score": 0.8855019807815552
        }
    ])

    result: dict[str, Any] = model.predict("테스트 문장입니다.")

    assert result["is_hate_speech"] is False
    assert result["confidence"] == 0.8855019807815552
    assert result["category"] == "clean"
    assert result["action"] == "allow"
    assert result["message"] == "Message allowed."


def test_model_predict_hate_speech() -> None:
    model = HateSpeechModel()
    model.pipeline = MagicMock(return_value=[
        {
            "label": "hate",
            "score": 0.91
        }
    ])

    result: dict[str, Any] = model.predict("나쁜 표현 예시")

    assert result["is_hate_speech"] is True
    assert result["confidence"] == 0.91
    assert result["category"] == "hate"
    assert result["action"] == "block"
    assert result["message"] == "Message blocked due to harmful content."


def test_model_predict_runtime_error_triggers_500() -> None:
    model = HateSpeechModel()
    model.pipeline = MagicMock(side_effect=RuntimeError("CPU inference error"))

    with pytest.raises(HTTPException) as exc_info:
        model.predict("긴 텍스트")

    assert exc_info.value.status_code == 500
    assert exc_info.value.detail == "Internal Server Error during model inference."