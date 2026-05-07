from fastapi import APIRouter, Request
from app.schemas.payload import DetectRequest, DetectResponse
from app.services.model import HateSpeechModel
from starlette.concurrency import run_in_threadpool

router = APIRouter()

@router.get("/health")
async def health_check() -> dict:
    return {
        "status": "ok",
        "message": "API server is running"
    }

@router.post("/detect", response_model=DetectResponse)
async def detect_text(request: DetectRequest, fastapi_req: Request) -> DetectResponse:
    """
    텍스트의 혐오 표현 여부를 분석합니다.
    동기 모델 추론이 이벤트 루프를 블로킹하지 않도록 threadpool로 격리하여 실행합니다.
    """
    model: HateSpeechModel = fastapi_req.app.state.model
    
    # 무거운 동기 추론 작업을 별도의 스레드에서 실행되도록 위임
    result = await run_in_threadpool(model.predict, request.text)
    

    return DetectResponse(
        is_hate_speech=result["is_hate_speech"],
        confidence=result["confidence"],
        category=result["category"],
        action=result["action"],
        message=result["message"]
    )