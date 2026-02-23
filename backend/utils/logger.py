"""
System Logger - Python 표준 logging 모듈 기반
=============================================
콘솔 + 파일 핸들러, UTF-8 인코딩 지원
"""

import logging
import sys
import os
from pathlib import Path
from logging.handlers import RotatingFileHandler

# 로그 디렉토리
LOG_DIR = Path(__file__).parent.parent / "logs"
LOG_FILE = LOG_DIR / "system.log"

_initialized = False


def setup_logging():
    """
    로깅 시스템 초기화 (앱 시작 시 1회 호출)
    """
    global _initialized
    if _initialized:
        return
    _initialized = True

    LOG_DIR.mkdir(exist_ok=True)

    log_level = os.getenv("LOG_LEVEL", "DEBUG").upper()

    formatter = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )

    # 파일 핸들러 (5MB 로테이션, 최대 3개 백업)
    file_handler = RotatingFileHandler(
        LOG_FILE,
        maxBytes=5 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8"
    )
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.DEBUG)

    # 콘솔 핸들러
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    stream_handler.setLevel(logging.INFO)

    # 루트 로거 설정
    root = logging.getLogger("dbo")
    root.setLevel(getattr(logging, log_level, logging.DEBUG))
    root.addHandler(file_handler)
    root.addHandler(stream_handler)

    # 외부 라이브러리 로그 레벨 조정
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("uvicorn.access").setLevel(logging.WARNING)


def get_logger(name: str) -> logging.Logger:
    """
    모듈별 로거 생성

    Usage:
        from utils.logger import get_logger
        logger = get_logger(__name__)
        logger.info("Hello")
    """
    if not _initialized:
        setup_logging()
    return logging.getLogger(f"dbo.{name}")


# =============================================================================
# 하위 호환 함수 (기존 from utils.logger import debug, info, error 지원)
# =============================================================================
_compat_logger = None


def _get_compat_logger():
    global _compat_logger
    if _compat_logger is None:
        _compat_logger = get_logger("legacy")
    return _compat_logger


def debug(message: str, module: str = None):
    _get_compat_logger().debug(f"[{module}] {message}" if module else message)


def info(message: str, module: str = None):
    _get_compat_logger().info(f"[{module}] {message}" if module else message)


def warn(message: str, module: str = None):
    _get_compat_logger().warning(f"[{module}] {message}" if module else message)


def error(message: str, module: str = None):
    _get_compat_logger().error(f"[{module}] {message}" if module else message)
