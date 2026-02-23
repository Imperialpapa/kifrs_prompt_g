"""
Feedback Router - False Positive 피드백
"""

from fastapi import APIRouter, HTTPException

from models import FalsePositiveFeedback
from dependencies import feedback_service
from utils.logger import get_logger

logger = get_logger("feedback_router")
router = APIRouter(tags=["Feedback"])


@router.post("/feedback/false-positive")
async def submit_false_positive_feedback(feedback: FalsePositiveFeedback):
    """False Positive 피드백 제출"""
    try:
        result = await feedback_service.submit_false_positive_feedback(feedback)
        return {"status": "success", "data": result}
    except Exception as e:
        logger.error(f"Feedback submission error: {e}")
        raise HTTPException(status_code=500, detail={"error": str(e)})
