"""
Learning Router - 학습 시스템 관리
"""

import traceback
from fastapi import APIRouter, HTTPException

from dependencies import learning_service
from utils.logger import get_logger

logger = get_logger("learning_router")
router = APIRouter(tags=["Learning"])


@router.get("/learning/statistics")
async def get_learning_statistics():
    """학습 시스템 통계 조회"""
    try:
        return await learning_service.get_learning_statistics()
    except Exception as e:
        logger.error(f"Learning stats error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to get learning statistics", "message": str(e)}
        )


@router.get("/learning/patterns/{pattern_id}/effectiveness")
async def get_pattern_effectiveness(pattern_id: str):
    """특정 학습 패턴의 효과성 분석"""
    try:
        result = await learning_service.get_pattern_effectiveness(pattern_id)
        if "error" in result:
            raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Pattern effectiveness error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to get pattern effectiveness", "message": str(e)}
        )


@router.post("/learning/maintenance")
async def run_learning_maintenance():
    """학습 시스템 유지보수 실행"""
    try:
        result = await learning_service.run_maintenance()
        return {
            "status": "success",
            "message": "학습 시스템 유지보수가 완료되었습니다.",
            **result
        }
    except Exception as e:
        logger.error(f"Learning maintenance error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to run learning maintenance", "message": str(e)}
        )
