"""
AI Cache Service - AI 규칙 해석 및 캐싱
========================================
업로드된 규칙을 AI로 해석하고 데이터베이스에 캐싱
"""

from typing import List, Dict, Any
from uuid import UUID
from datetime import datetime

from database.rule_repository import RuleRepository
from ai_layer import AIRuleInterpreter
from models import ValidationRule, RuleSource


class AICacheService:
    """
    AI 해석 결과를 데이터베이스에 캐싱하는 서비스

    주요 기능:
    - 규칙 업로드 후 자동 AI 해석
    - 해석 결과를 rules 테이블에 캐싱
    - ai_interpretation_logs 테이블에 로그 기록
    - 재해석 기능
    """

    def __init__(self):
        """Initialize service"""
        self.repository = RuleRepository()
        self.ai_interpreter = AIRuleInterpreter()

    async def interpret_and_cache_rules(
        self,
        file_id: str,
        force_reinterpret: bool = False,
        force_local: bool = False
    ) -> Dict[str, Any]:
        """
        규칙 파일의 모든 규칙을 AI로 해석하고 캐싱

        Args:
            file_id: 규칙 파일 UUID
            force_reinterpret: True면 기존 해석 무시하고 재해석

        Returns:
            Dict: 해석 결과 통계
            {
                "total_rules": int,
                "interpreted_rules": int,
                "skipped_rules": int,
                "failed_rules": int,
                "processing_time_seconds": float
            }
        """
        print(f"[AICacheService] Starting interpretation for file: {file_id}")

        start_time = datetime.now()

        # Step 1: 규칙 조회
        rules = await self.repository.get_rules_by_file(UUID(file_id), active_only=True)
        total_rules = len(rules)

        print(f"[AICacheService] Found {total_rules} rules to process")

        if total_rules == 0:
            return {
                "total_rules": 0,
                "interpreted_rules": 0,
                "skipped_rules": 0,
                "failed_rules": 0,
                "processing_time_seconds": 0.0
            }

        # Step 2: 이미 해석된 규칙 필터링
        rules_to_interpret = []
        skipped_count = 0

        for rule in rules:
            if force_reinterpret or not rule.get('ai_rule_id'):
                # 자연어 규칙 형식으로 변환
                natural_rule = {
                    "sheet": rule.get('canonical_sheet_name'),
                    "display_sheet_name": rule.get('display_sheet_name'),
                    "row": rule.get('row_number'),
                    "column_letter": rule.get('column_letter'),
                    "field": rule.get('field_name'),
                    "rule_text": rule.get('rule_text'),
                    "condition": rule.get('condition', ''),
                    "note": rule.get('note', '')
                }
                rules_to_interpret.append({
                    "db_rule": rule,
                    "natural_rule": natural_rule
                })
            else:
                skipped_count += 1

        print(f"[AICacheService] To interpret: {len(rules_to_interpret)}, Skipped: {skipped_count}")

        if len(rules_to_interpret) == 0:
            print("[AICacheService] All rules already interpreted")
            return {
                "total_rules": total_rules,
                "interpreted_rules": 0,
                "skipped_rules": skipped_count,
                "failed_rules": 0,
                "processing_time_seconds": 0.0
            }

        # Step 3: AI 해석
        natural_rules = [r["natural_rule"] for r in rules_to_interpret]

        # force_local이면 로컬 파서만 사용
        provider = "local" if force_local else None

        print(f"[AICacheService] Calling AI interpreter... (force_local={force_local})")
        try:
            ai_response = await self.ai_interpreter.interpret_rules(natural_rules, provider=provider)
            print(f"[AICacheService] AI interpretation completed")
            print(f"   - Interpreted rules: {len(ai_response.rules)}")
            print(f"   - Conflicts: {len(ai_response.conflicts)}")
            print(f"   - Processing time: {ai_response.processing_time_seconds:.2f}s")
        except Exception as e:
            print(f"[AICacheService] AI interpretation failed: {str(e)}")
            return {
                "total_rules": total_rules,
                "interpreted_rules": 0,
                "skipped_rules": skipped_count,
                "failed_rules": len(rules_to_interpret),
                "processing_time_seconds": (datetime.now() - start_time).total_seconds(),
                "error": str(e)
            }

        # Step 4: 해석 결과를 DB에 캐싱
        print(f"[AICacheService] Caching interpretation results...")
        interpreted_count = 0
        failed_count = 0

        for idx, interpreted_rule in enumerate(ai_response.rules):
            if idx >= len(rules_to_interpret):
                print(f"[AICacheService] Warning: More interpreted rules than input rules")
                break

            db_rule = rules_to_interpret[idx]["db_rule"]
            rule_id = db_rule['id']

            # AI 해석 데이터 준비
            # 실제 사용된 엔진에 따라 model_version 설정
            actual_model = "local-parser" if force_local or not self.ai_interpreter.use_cloud_ai else f"cloud-{self.ai_interpreter.default_provider}"

            ai_data = {
                "ai_rule_id": interpreted_rule.rule_id,
                "ai_rule_type": interpreted_rule.rule_type,
                "ai_parameters": interpreted_rule.parameters,
                "ai_error_message": interpreted_rule.error_message_template,
                "ai_interpretation_summary": interpreted_rule.ai_interpretation_summary,
                "ai_confidence_score": float(interpreted_rule.confidence_score),
                "ai_interpreted_at": datetime.now().isoformat(),
                "ai_model_version": actual_model
            }

            try:
                success = await self.repository.update_rule_ai_interpretation(
                    UUID(rule_id),
                    ai_data
                )

                if success:
                    interpreted_count += 1
                else:
                    failed_count += 1
                    print(f"[AICacheService] Failed to update rule: {rule_id}")

            except Exception as e:
                failed_count += 1
                print(f"[AICacheService] Error updating rule {rule_id}: {str(e)}")

        # Step 5: 로그 테이블에 기록
        await self._log_interpretation(
            file_id=file_id,
            total_rules=len(rules_to_interpret),
            interpreted_count=interpreted_count,
            ai_response=ai_response
        )

        processing_time = (datetime.now() - start_time).total_seconds()

        print(f"[AICacheService] Caching completed")
        print(f"   - Cached: {interpreted_count}")
        print(f"   - Failed: {failed_count}")
        print(f"   - Total time: {processing_time:.2f}s")

        # 사용된 엔진 확인
        engine_used = "local" if force_local else ("cloud" if self.ai_interpreter.use_cloud_ai else "local")

        return {
            "total_rules": total_rules,
            "interpreted_rules": interpreted_count,
            "skipped_rules": skipped_count,
            "failed_rules": failed_count,
            "processing_time_seconds": processing_time,
            "conflicts": [c.dict() for c in ai_response.conflicts],
            "engine": engine_used
        }

    async def _log_interpretation(
        self,
        file_id: str,
        total_rules: int,
        interpreted_count: int,
        ai_response
    ):
        """
        AI 해석 로그를 ai_interpretation_logs 테이블에 기록

        Args:
            file_id: 규칙 파일 UUID
            total_rules: 전체 규칙 수
            interpreted_count: 해석된 규칙 수
            ai_response: AI 응답 객체
        """
        try:
            # 실제 사용된 엔진 확인
            actual_model = "local-parser" if not self.ai_interpreter.use_cloud_ai else f"cloud-{self.ai_interpreter.default_provider}"

            # 각 해석된 규칙에 대한 로그
            for rule in ai_response.rules:
                log_data = {
                    "rule_file_id": file_id,
                    "natural_language_rule": rule.source.original_text,
                    "sheet_name": rule.source.sheet_name,
                    "field_name": rule.field_name,
                    "interpreted_rule_type": rule.rule_type,
                    "interpreted_parameters": rule.parameters,
                    "confidence_score": float(rule.confidence_score),
                    "ai_model_version": actual_model,
                    "processing_time_seconds": ai_response.processing_time_seconds / len(ai_response.rules) if ai_response.rules else 0
                }

                # Insert to ai_interpretation_logs table
                self.repository.client.table('ai_interpretation_logs').insert(log_data).execute()

            print(f"[AICacheService] Logged {len(ai_response.rules)} interpretation records")

        except Exception as e:
            print(f"[AICacheService] Failed to log interpretation: {str(e)}")

    async def get_cached_rules_as_validation_rules(
        self,
        file_id: str
    ) -> List[ValidationRule]:
        """
        캐시된 AI 해석을 ValidationRule 객체로 변환

        Args:
            file_id: 규칙 파일 UUID

        Returns:
            List[ValidationRule]: 검증 엔진에서 사용 가능한 규칙 객체 리스트
        """
        print(f"[AICacheService] Loading cached rules for file: {file_id}")

        rules = await self.repository.get_rules_by_file(UUID(file_id), active_only=True)

        validation_rules = []
        not_interpreted_count = 0

        for rule in rules:
            # AI 해석이 없는 규칙은 스킵
            if not rule.get('ai_rule_id'):
                not_interpreted_count += 1
                continue

            try:
                # RuleSource 생성
                source = RuleSource(
                    original_text=rule.get('rule_text'),
                    sheet_name=rule.get('canonical_sheet_name'),
                    row_number=rule.get('row_number'),
                    kifrs_reference=None
                )

                # ValidationRule 생성
                validation_rule = ValidationRule(
                    rule_id=rule.get('ai_rule_id'),
                    field_name=rule.get('field_name'),
                    rule_type=rule.get('ai_rule_type'),
                    parameters=rule.get('ai_parameters') or {},
                    error_message_template=rule.get('ai_error_message'),
                    source=source,
                    ai_interpretation_summary=rule.get('ai_interpretation_summary') or "",
                    confidence_score=float(rule.get('ai_confidence_score') or 0.0),
                    is_common=rule.get('is_common', False)  # Map is_common
                )

                validation_rules.append(validation_rule)

            except Exception as e:
                print(f"[AICacheService] Error converting rule {rule.get('id')}: {str(e)}")
                continue

        print(f"[AICacheService] Loaded {len(validation_rules)} validation rules")
        if not_interpreted_count > 0:
            print(f"[AICacheService] Warning: {not_interpreted_count} rules not interpreted yet")

        return validation_rules


# =============================================================================
# Testing
# =============================================================================

if __name__ == "__main__":
    import asyncio

    async def test_service():
        """Test AI cache service"""
        print("=" * 70)
        print("AICacheService Test")
        print("=" * 70)

        service = AICacheService()

        # Test with a real file ID (replace with actual ID)
        file_id = "5ea221ce-8985-49cc-8412-4995e87e62b2"

        print("\n1. Testing interpretation and caching...")
        result = await service.interpret_and_cache_rules(file_id)

        print("\nResult:")
        for key, value in result.items():
            print(f"  {key}: {value}")

        print("\n2. Testing loading cached rules...")
        rules = await service.get_cached_rules_as_validation_rules(file_id)
        print(f"Loaded {len(rules)} validation rules")

        if len(rules) > 0:
            print(f"\nSample rule:")
            sample = rules[0]
            print(f"  Rule ID: {sample.rule_id}")
            print(f"  Field: {sample.field_name}")
            print(f"  Type: {sample.rule_type}")
            print(f"  Confidence: {sample.confidence_score}")

        print("\n" + "=" * 70)

    # Run test
    asyncio.run(test_service())
