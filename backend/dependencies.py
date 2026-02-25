"""
Dependency Injection - 전역 서비스 인스턴스 중앙 관리
===================================================
모든 라우터에서 공유하는 서비스 인스턴스를 한 곳에서 생성/관리합니다.
"""

from ai_layer import AIRuleInterpreter
from services.learning_service import LearningService
from services.ai_cache_service import AICacheService
from services.rule_service import RuleService
from services.validation_service import ValidationService
from services.feedback_service import FeedbackService
from services.statistics_service import StatisticsService
from services.fix_service import FixService
from services.kifrs_dbo_service import KifrsDboService
from database.supabase_client import supabase

# =============================================================================
# 서비스 레이어 초기화 (의존성 주입)
# =============================================================================

ai_interpreter = AIRuleInterpreter()
learning_service = LearningService(supabase_client=supabase)

ai_cache_service = AICacheService(
    interpreter=ai_interpreter,
    learning_service=learning_service
)

kifrs_dbo_service = KifrsDboService()

rule_service = RuleService(
    ai_cache_service=ai_cache_service,
    kifrs_dbo_service=kifrs_dbo_service
)

validation_service = ValidationService()
feedback_service = FeedbackService()
statistics_service = StatisticsService()
fix_service = FixService(ai_interpreter=ai_interpreter)
