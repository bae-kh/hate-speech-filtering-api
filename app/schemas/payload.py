from pydantic import BaseModel, Field

class DetectRequest(BaseModel):
    text: str = Field(
        ..., 
        min_length=1, 
        description="분석할 텍스트 입력값. 빈 문자열은 허용되지 않습니다."
    )

class DetectResponse(BaseModel):
    is_hate_speech: bool = Field(..., description="혐오 표현 포함 여부")
    confidence: float = Field(..., description="예측 신뢰도")
    message: str = Field(..., description="응답 메시지")
