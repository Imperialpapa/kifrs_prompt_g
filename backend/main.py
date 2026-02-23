"""
K-IFRS 1019 DBO Validation System - FastAPI Backend
===================================================
설명: 확정급여채무(DBO) 평가를 위한 데이터 정합성 검증 시스템의 메인 API 서버입니다.

[시스템 아키텍처]
1. Presentation Layer (Frontend): HTML/Alpine.js 기반의 SPA
2. API Layer (Routers): FastAPI Router 기반 RESTful API
3. Service Layer: 비즈니스 로직 (dependencies.py에서 중앙 관리)
4. Domain Layer: AI 해석(AI Layer) 및 결정론적 검증(Rule Engine)
5. Data Layer: Supabase (PostgreSQL) 및 로컬 파일 처리
"""

import os
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from config import settings
from utils.logger import setup_logging, get_logger
from utils.exceptions import AppException

# =============================================================================
# 로깅 초기화
# =============================================================================
setup_logging()
logger = get_logger("main")

# =============================================================================
# FastAPI 앱 생성
# =============================================================================
app = FastAPI(
    title=settings.APP_NAME,
    description="AI-Powered Data Validation for Defined Benefit Obligations",
    version=settings.APP_VERSION,
)

# =============================================================================
# CORS 설정 (환경변수로 허용 도메인 관리)
# =============================================================================
_cors_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*")
_cors_origin_list = [o.strip() for o in _cors_origins.split(",") if o.strip()]
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origin_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# 정적 파일 서빙
# =============================================================================
if os.path.exists("../frontend"):
    app.mount("/static", StaticFiles(directory="../frontend"), name="static")
elif os.path.exists("frontend"):
    app.mount("/static", StaticFiles(directory="frontend"), name="static")

# =============================================================================
# Router 등록
# =============================================================================
from routers.core_router import router as core_router
from routers.rules_router import router as rules_router
from routers.validation_router import router as validation_router
from routers.session_router import router as session_router
from routers.feedback_router import router as feedback_router
from routers.learning_router import router as learning_router
from routers.ai_router import router as ai_router
from routers.kifrs_router import router as kifrs_router
from routers.export_router import router as export_router

app.include_router(core_router)
app.include_router(rules_router)
app.include_router(validation_router)
app.include_router(session_router)
app.include_router(feedback_router)
app.include_router(learning_router)
app.include_router(ai_router)
app.include_router(kifrs_router)
app.include_router(export_router)

# =============================================================================
# 예외 핸들러
# =============================================================================

@app.exception_handler(AppException)
async def app_exception_handler(request, exc: AppException):
    """애플리케이션 예외 핸들러"""
    logger.error(f"AppException: [{exc.error_code}] {exc.message}")
    return JSONResponse(
        status_code=exc.status_code,
        content=exc.to_response()
    )


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """전역 예외 핸들러"""
    logger.error(f"Unhandled exception: {type(exc).__name__}: {exc}")
    return JSONResponse(
        status_code=500,
        content={
            "error_code": "INTERNAL_ERROR",
            "message": str(exc),
            "detail": type(exc).__name__,
            "timestamp": datetime.now().isoformat()
        }
    )

# =============================================================================
# 서버 실행 (개발용)
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    logger.info("""
    =================================================================
      K-IFRS 1019 DBO Validation System
      AI-Powered Data Validation for Defined Benefit Obligations
    =================================================================

    Starting server...
    Mobile-optimized UI available at: http://localhost:8000
    API Documentation: http://localhost:8000/docs
    """)

    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
