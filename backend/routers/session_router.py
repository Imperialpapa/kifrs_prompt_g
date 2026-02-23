"""
Session Router - 검증 세션 및 통계
"""

from fastapi import APIRouter, HTTPException

from dependencies import validation_service, statistics_service
from utils.logger import get_logger

logger = get_logger("session_router")
router = APIRouter(tags=["Sessions"])


@router.get("/sessions")
async def list_validation_sessions(limit: int = 50, offset: int = 0):
    """검증 세션 목록 조회"""
    try:
        sessions = await validation_service.list_sessions(limit, offset)
        return sessions
    except Exception as e:
        logger.error(f"Session list error: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/sessions/{session_id}")
async def get_session_details(session_id: str):
    """세션 상세 정보 및 에러 목록 조회"""
    try:
        details = await validation_service.get_session_details(session_id)
        if not details:
            raise HTTPException(status_code=404, detail="Session not found")
        return details
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Session detail error: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})


@router.get("/statistics/dashboard")
async def get_dashboard_statistics():
    """대시보드 통계 조회"""
    try:
        return await statistics_service.get_dashboard_statistics()
    except Exception as e:
        logger.error(f"Dashboard stats error: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
