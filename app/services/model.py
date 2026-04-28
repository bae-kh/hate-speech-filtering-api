import logging
from transformers import pipeline
from fastapi import HTTPException

logger = logging.getLogger(__name__)

class HateSpeechModel:
    def __init__(self, model_name: str = "smilegate-ai/kor_unsmile"):
        self.model_name = model_name
        self.pipeline = None

    def load(self) -> None:
        """HuggingFace 모델을 메모리에 적재합니다."""
        logger.info(f"Loading model {self.model_name} into memory...")
        try:
            # text-classification pipeline 로드, truncation 설정
            self.pipeline = pipeline(
                "text-classification", 
                model=self.model_name, 
                truncation=True, 
                max_length=512
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise RuntimeError(f"Model initialization failed: {e}")

    def predict(self, text: str) -> dict:
        """적재된 모델을 통해 텍스트의 혐오 표현 여부를 추론합니다."""
        if not self.pipeline:
            raise RuntimeError("Model is not loaded.")
        
        try:
            # OOM 방어를 위한 truncation=True 강제 적용
            results = self.pipeline(text, truncation=True, max_length=512)
            
            # kor_unsmile은 여러 라벨에 대한 확률을 반환하거나, 가장 높은 것 하나를 반환
            best_result = results[0]
            label = best_result['label']
            score = best_result['score']
            
            # clean 라벨이 아니면 혐오 표현으로 간주
            is_hate_speech = label != 'clean'
            
            return {
                "is_hate_speech": is_hate_speech,
                "confidence": float(score),
                "message": f"Detected category: {label}"
            }
        except RuntimeError as e:
            # 모델 내부 추론 중 RuntimeError (ex. Out of Memory) 발생 시 방어
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
