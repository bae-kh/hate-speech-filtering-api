import uuid
from typing import Callable, Awaitable
from fastapi import FastAPI, Request, Response
from app.api.routes import router as api_router

app = FastAPI(title="Text Filtering API", version="1.0.0-MVP")

@app.middleware("http")
async def add_request_id_header(
    request: Request, 
    call_next: Callable[[Request], Awaitable[Response]]
) -> Response:
    """
    대용량 트래픽 환경에서의 추적성 확보를 위한 미들웨어입니다.
    모든 요청에 UUID v4 기반의 고유 식별자를 부여하고 응답 헤더에 포함시킵니다.
    """
    request_id: str = str(uuid.uuid4())
    
    # 향후 애플리케이션 내부 로깅에서 활용할 수 있도록 state에 저장
    request.state.request_id = request_id
    
    # 다음 미들웨어 또는 라우터 처리
    response: Response = await call_next(request)
    
    # 클라이언트 및 인프라(APM 등)에서 추적 가능하도록 응답 헤더에 주입
    response.headers["X-Request-ID"] = request_id
    
    return response

# 라우터 등록 (경로 분리)
app.include_router(api_router, prefix="/api/v1")
