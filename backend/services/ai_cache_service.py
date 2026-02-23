"""
AI Cache Service - AI 규칙 해석 및 캐싱
========================================
업로드된 규칙을 AI로 해석하고 데이터베이스에 캐싱
"""

from typing import List, Dict, Any, Tuple, Optional
from uuid import UUID, uuid4
from datetime import datetime

from database.rule_repository import RuleRepository
from ai_layer import AIRuleInterpreter
from models import ValidationRule, RuleSource
from utils.logger import get_logger

logger = get_logger("ai_cache_service")


class AICacheService:
    """
    AI 해석 결과를 데이터베이스에 캐싱하는 서비스
    """

    def __init__(self, repository=None, interpreter=None, learning_service=None):
        """Initialize service with dependencies"""
        self.repository = repository or RuleRepository()
        self.ai_interpreter = interpreter or AIRuleInterpreter()
        self.learning_service = learning_service

    async def interpret_and_cache_rules(
        self,
        file_id: str,
        force_reinterpret: bool = False,
        force_local: bool = False
    ) -> Dict[str, Any]:
        """
        규칙 파일의 모든 규칙을 AI로 해석하고 캐싱 (Smart Interpret 적용)
        """
        logger.info(f"Starting interpretation for file: {file_id}")
        start_time = datetime.now()

        # Step 1: 규칙 조회
        rules = await self.repository.get_rules_by_file(UUID(file_id), active_only=True)
        total_rules = len(rules)

        if total_rules == 0:
            return {"total_rules": 0, "interpreted_rules": 0, "processing_time_seconds": 0.0}

        # Step 2: 해석 대상 필터링
        rules_to_interpret = []
        skipped_count = 0
        for rule in rules:
            # 필수 필드(ai_rule_id, ai_rule_type, ai_error_message)가 하나라도 없으면 해석 대상
            if force_reinterpret or not rule.get('ai_rule_id') or not rule.get('ai_rule_type') or not rule.get('ai_error_message'):
                rules_to_interpret.append(rule)
            else:
                skipped_count += 1

        if not rules_to_interpret:
            return {"total_rules": total_rules, "interpreted_rules": 0, "skipped_rules": skipped_count}

        # Step 3: 통합된 Smart Interpret 로직 실행
        interpreted_results = []
        for db_rule in rules_to_interpret:
            interpretation, source = await self.smart_interpret_single(
                rule_text=db_rule.get('rule_text'),
                field_name=db_rule.get('field_name'),
                force_local=force_local
            )
            interpreted_results.append({
                "db_rule": db_rule,
                "interpretation": interpretation,
                "source": source
            })

        # Step 4: 결과 저장 (캐싱)
        interpreted_count = 0
        for res in interpreted_results:
            db_rule = res["db_rule"]
            interp = res["interpretation"]
            source = res["source"]
            
            actual_model = "learned-pattern" if source == "learned" else \
                          ("local-parser" if force_local or not self.ai_interpreter.use_cloud_ai else f"cloud-{self.ai_interpreter.default_provider}")

            ai_data = {
                "ai_rule_id": interp.get("rule_id", f"RULE_{uuid4().hex[:8]}"),
                "ai_rule_type": interp.get("rule_type"),
                "ai_parameters": interp.get("parameters"),
                "ai_error_message": interp.get("error_message"),
                "ai_interpretation_summary": interp.get("interpretation_summary"),
                "ai_confidence_score": float(interp.get("confidence_score", 0.8)),
                "ai_interpreted_at": datetime.now().isoformat(),
                "ai_model_version": actual_model
            }

            if await self.repository.update_rule_ai_interpretation(UUID(db_rule['id']), ai_data):
                interpreted_count += 1

        # Step 5: 로그 기록 및 요약
        processing_time = (datetime.now() - start_time).total_seconds()
        await self._log_interpretation(file_id, total_rules, interpreted_count, interpreted_results, processing_time)

        return {
            "total_rules": total_rules,
            "interpreted_rules": interpreted_count,
            "skipped_rules": skipped_count,
            "engine": "smart-hybrid",
            "processing_time_seconds": processing_time
        }

    async def smart_interpret_single(self, rule_text: str, field_name: str, force_local: bool = False):
        """
        단일 규칙에 대한 통합 해석 로직 (Smart Interpret)
        """
        if self.learning_service and not force_local:
            return await self.learning_service.smart_interpret(
                rule_text=rule_text,
                field_name=field_name,
                ai_interpreter=self.ai_interpreter,
                use_learning=True
            )
        else:
            interpretation = self.ai_interpreter.interpret_rule(rule_text, field_name)
            return interpretation, "ai"

    async def _log_interpretation(self, file_id, total_rules, interpreted_count, results, time_spent):
        """해석 로그 기록"""
        try:
            for res in results:
                rule = res["interpretation"]
                db_rule = res["db_rule"]
                log_data = {
                    "rule_file_id": file_id,
                    "natural_language_rule": db_rule.get('rule_text'),
                    "field_name": db_rule.get('field_name'),
                    "interpreted_rule_type": rule.get("rule_type"),
                    "interpreted_parameters": rule.get("parameters"),
                    "confidence_score": float(rule.get("confidence_score", 0.0)),
                    "ai_model_version": res["source"],
                    "processing_time_seconds": time_spent / len(results) if results else 0
                }
                self.repository.client.table('ai_interpretation_logs').insert(log_data).execute()
        except Exception as e:
            logger.error(f"Logging failed: {e}")

    async def get_cached_rules_as_validation_rules(self, file_id: str) -> List[ValidationRule]:
        """캐시된 AI 해석을 ValidationRule 객체로 변환"""
        rules = await self.repository.get_rules_by_file(UUID(file_id), active_only=True)
        validation_rules = []
        for rule in rules:
            if not rule.get('ai_rule_id'): continue
            try:
                # 필드 유효성 검사 및 기본값 설정
                ai_rule_type = rule.get('ai_rule_type')
                # Literal에 없는 타입인 경우 custom으로 전환
                valid_types = ["required", "no_duplicates", "format", "allowed_values", "range", "date_logic", "cross_field", "custom", "composite"]
                if ai_rule_type not in valid_types:
                    ai_rule_type = "custom"

                validation_rules.append(ValidationRule(
                    rule_id=rule.get('ai_rule_id') or f"RULE_{uuid4().hex[:8]}",
                    field_name=rule.get('field_name') or "(Unknown Field)",
                    rule_type=ai_rule_type,
                    parameters=rule.get('ai_parameters') or {},
                    error_message_template=rule.get('ai_error_message') or f"{rule.get('field_name')} 검증 실패",
                    source=RuleSource(
                        original_text=rule.get('rule_text') or "",
                        row_number=str(rule.get('row_number', '0'))
                    ),
                    ai_interpretation_summary=rule.get('ai_interpretation_summary') or "자동 로드된 규칙",
                    confidence_score=float(rule.get('ai_confidence_score') or 0.0),
                    is_common=rule.get('is_common', False)
                ))
            except Exception as e:
                logger.error(f"Failed to load rule {rule.get('id')}: {e}")
                continue
        return validation_rules