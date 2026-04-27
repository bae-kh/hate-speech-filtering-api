from fastapi import APIRouter
from app.schemas.payload import DetectRequest, DetectResponse

router = APIRouter()

@router.post("/detect", response_model=DetectResponse)
async def detect_text(request: DetectRequest) -> DetectResponse:
    """
    텍스트의 혐오 표현 여부를 분석합니다.
    (Phase 1: 클라이언트-서버 통신 안정성 확보를 위한 Mock 응답 반환)
    """
    return DetectResponse(
        is_hate_speech=False,
        confidence=0.0,
        message="Hello SKT - MVP Ready"
    )
