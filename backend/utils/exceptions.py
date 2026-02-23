"""
Application Exception & Error Response 표준화
=============================================
모든 API 에러 응답을 통일된 형식으로 반환
"""

from datetime import datetime
from typing import Optional
from pydantic import BaseModel


class ErrorResponse(BaseModel):
    """표준 에러 응답 모델"""
    error_code: str
    message: str
    detail: Optional[str] = None
    timestamp: str


class AppException(Exception):
    """
    애플리케이션 전용 예외

    Usage:
        raise AppException("VALIDATION_FAILED", "검증에 실패했습니다.", status_code=400)
        raise AppException("RULE_NOT_FOUND", "규칙을 찾을 수 없습니다.", status_code=404)
    """

    def __init__(
        self,
        error_code: str,
        message: str,
        status_code: int = 500,
        detail: Optional[str] = None
    ):
        self.error_code = error_code
        self.message = message
        self.status_code = status_code
        self.detail = detail
        super().__init__(message)

    def to_response(self) -> dict:
        return ErrorResponse(
            error_code=self.error_code,
            message=self.message,
            detail=self.detail,
            timestamp=datetime.now().isoformat()
        ).model_dump()
