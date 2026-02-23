"""
Core Router - 헬스체크, 버전, 기본 정보
"""

import os
from datetime import datetime
from fastapi import APIRouter
from fastapi.responses import FileResponse

from config import settings

router = APIRouter(tags=["Core"])


@router.get("/")
async def root():
    """프론트엔드 정적 파일(index.html) 제공"""
    if os.path.exists("../index.html"):
        return FileResponse("../index.html")
    elif os.path.exists("index.html"):
        return FileResponse("index.html")

    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational (Frontend file not found)"
    }


@router.get("/api")
async def api_info():
    """API 기본 정보 제공"""
    return {
        "service": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "status": "operational",
        "features": [
            "다중 시트 검증",
            "AI 규칙 해석",
            "개인정보 마스킹",
            "K-IFRS 특화 로직"
        ]
    }


@router.get("/health")
async def health_check():
    """시스템 상태 확인"""
    return {
        "status": "healthy",
        "ai_layer": "operational",
        "rule_engine": "operational"
    }


@router.get("/version")
async def get_version():
    """시스템 버전 정보 반환"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "system_version": settings.APP_VERSION,
        "build_time": now_str,
        "platform": "FastAPI/Python"
    }
