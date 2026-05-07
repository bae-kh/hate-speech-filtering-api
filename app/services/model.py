import logging
from typing import Any

from fastapi import HTTPException
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)

CLEAN_LABEL = "clean"
MODEL_MAX_TOKENS = 256


class HateSpeechModel:
    def __init__(self, model_name: str = "smilegate-ai/kor_unsmile"):
        self.model_name = model_name
        self.pipeline = None

    def load(self) -> None:
        """HuggingFace 모델을 메모리에 적재합니다."""
        logger.info(f"Loading model {self.model_name} into memory...")
        try:
            self.pipeline = hf_pipeline(
                "text-classification",
                model=self.model_name,
                truncation=True,
                max_length=MODEL_MAX_TOKENS
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def predict(self, text: str) -> dict[str, Any]:
        """적재된 모델을 통해 채팅 메시지의 혐오 표현 여부를 추론합니다."""
        if not self.pipeline:
            raise RuntimeError("Model is not loaded.")

        try:
            results = self.pipeline(
                text,
                truncation=True,
                max_length=MODEL_MAX_TOKENS
            )

            best_result = results[0]
            label = best_result["label"]
            score = best_result["score"]

            is_hate_speech = label != CLEAN_LABEL
            action = "block" if is_hate_speech else "allow"

            return {
                "is_hate_speech": is_hate_speech,
                "confidence": float(score),
                "category": label,
                "action": action,
                "message": (
                    "Message blocked due to harmful content."
                    if action == "block"
                    else "Message allowed."
                )
            }

        except RuntimeError as e:
            logger.error(f"Runtime inference error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Internal Server Error during model inference."
            )
        except Exception as e:
            logger.error(f"Unexpected inference error: {e}")
            raise HTTPException(
                status_code=500,
                detail="Unexpected error during model inference."
            )

    def unload(self) -> None:
        """모델 자원을 메모리에서 해제합니다."""
        logger.info("Unloading model and freeing resources...")
        self.pipeline = None