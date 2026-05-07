from pydantic import BaseModel, Field, field_validator
from typing import Literal

class DetectRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=1, 
        max_length=500, 
        description="분석할 텍스트 입력값. 공백만 있는 문자열은 허용되지 않습니다."
    )

    @field_validator("text")
    @classmethod
    def text_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("text must not be blank")
        return value

class DetectResponse(BaseModel):
    is_hate_speech: bool = Field(..., description="혐오 표현 포함 여부")
    confidence: float = Field(..., description="예측 신뢰도")
    category: str = Field(..., description="모델이 예측한 카테고리")
    action: Literal["allow", "block"] = Field(..., description="서비스 처리 정책")
    message: str = Field(..., description="응답 메시지")