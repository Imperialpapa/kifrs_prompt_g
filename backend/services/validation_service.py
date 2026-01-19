from models import ValidationRule, ValidationResponse, ValidationError, ValidationSummary, ValidationErrorGroup
from utils.common import convert_numpy_types, group_errors
import numpy as np
from datetime import datetime
from uuid import UUID, uuid4
from typing import Dict, List, Any, Optional
import json
import pandas as pd
import io

from database.rule_repository import RuleRepository
from database.validation_repository import ValidationRepository
from services.ai_cache_service import AICacheService
from rule_engine import RuleEngine, KIFRS_RuleEngine

class ValidationService:
    """
    DB 기반 검증을 수행하는 서비스
    """

    def __init__(self):
        """Initialize service"""
        self.rule_repository = RuleRepository()
        self.validation_repository = ValidationRepository()
        self.ai_cache_service = AICacheService()
        self.rule_engine = RuleEngine()

    async def validate_sheets(
        self,
        sheet_data_map: Dict[str, Dict[str, Any]],
        validation_rules: List[ValidationRule]
    ) -> ValidationResponse:
        """
        공통 시트 검증 로직
        
        Args:
            sheet_data_map: Canonical Name -> { display_name, original_name, df }
            validation_rules: 적용할 규칙 리스트
            
        Returns:
            ValidationResponse: 전체 검증 결과
        """
        all_errors = []
        all_sheets_summary = {}

        # 규칙을 시트별로 그룹화
        from collections import defaultdict
        rules_by_sheet = defaultdict(list)
        for rule in validation_rules:
            rules_by_sheet[rule.source.sheet_name].append(rule)

        # 각 시트 검증
        for canonical_name, data in sheet_data_map.items():
            display_name = data["display_name"]
            df = data["df"]
            
            sheet_rules = rules_by_sheet.get(canonical_name, [])
            if not sheet_rules:
                continue
            
            errors = self.rule_engine.validate(df, sheet_rules)
            summary = self.rule_engine.get_summary(len(df), len(sheet_rules))

            for error in errors:
                error.sheet = display_name
            
            all_errors.extend(errors)
            
            all_sheets_summary[display_name] = {
                "total_rows": len(df),
                "error_rows": summary.error_rows,
                "valid_rows": summary.valid_rows,
                "total_errors": len(errors),
                "rules_applied": len(sheet_rules)
            }

        # 데이터 정제 (Numpy 타입 변환)
        cleaned_errors = []
        for err in all_errors:
            err_dict = err.dict()
            err_dict['actual_value'] = convert_numpy_types(err.actual_value)
            cleaned_errors.append(ValidationError(**err_dict))

        # 전체 요약 계산
        total_rows = sum(s["total_rows"] for s in all_sheets_summary.values())
        total_error_rows = sum(s["error_rows"] for s in all_sheets_summary.values())
        
        # 원본 시트 순서 추출 (sheet_data_map은 삽입 순서가 보장된다고 가정하거나 별도 리스트 필요)
        # sheet_data_map은 위에서 excel_file.sheet_names 순서대로 생성되었으므로 keys() 순서가 원본 순서임
        original_sheet_order = [data["display_name"] for data in sheet_data_map.values()]

        overall_summary = ValidationSummary(
            total_rows=total_rows,
            valid_rows=total_rows - total_error_rows,
            error_rows=total_error_rows,
            total_errors=len(cleaned_errors),
            rules_applied=len(validation_rules),
            timestamp=datetime.now()
        )

        # 인지 내용 집계
        error_groups = group_errors(cleaned_errors)

        return ValidationResponse(
            validation_status="PASS" if len(cleaned_errors) == 0 else "FAIL",
            summary=overall_summary,
            errors=cleaned_errors,
            error_groups=error_groups,
            rules_applied=validation_rules,
            metadata={
                "sheets_summary": all_sheets_summary,
                "sheet_order": original_sheet_order  # 원본 시트 순서 추가
            }
        )

    async def validate_with_db_rules(
        self,
        rule_file_id: str,
        employee_file_content: bytes,
        employee_file_name: str
    ) -> Dict[str, Any]:
        """
        DB에 저장된 규칙을 사용하여 데이터 검증 수행
        """
        start_time = datetime.now()

        # Step 1: 규칙 로드
        print(f"[ValidationService] Loading rules from DB: {rule_file_id}")
        validation_rules = await self.ai_cache_service.get_cached_rules_as_validation_rules(rule_file_id)

        # Step 1.5: AI 해석이 없으면 자동으로 실행
        if not validation_rules:
            print(f"[ValidationService] No AI interpreted rules found. Running auto-interpretation...")
            interpret_result = await self.ai_cache_service.interpret_and_cache_rules(rule_file_id)
            print(f"[ValidationService] Auto-interpretation completed: {interpret_result}")

            # 다시 로드
            validation_rules = await self.ai_cache_service.get_cached_rules_as_validation_rules(rule_file_id)

            if not validation_rules:
                raise ValueError("규칙 해석에 실패했습니다. 규칙 파일을 확인해주세요.")

        # Step 2: 직원 데이터 파싱
        try:
            excel_file = pd.ExcelFile(io.BytesIO(employee_file_content))
            sheet_data_map = {}
            from utils.excel_parser import normalize_sheet_name, get_canonical_name
            
            for sheet_name in excel_file.sheet_names:
                df = pd.read_excel(io.BytesIO(employee_file_content), sheet_name=sheet_name)
                canonical_name = get_canonical_name(sheet_name)
                sheet_data_map[canonical_name] = {
                    "display_name": normalize_sheet_name(sheet_name),
                    "original_name": sheet_name,
                    "df": df
                }
        except Exception as e:
            raise ValueError(f"직원 데이터 파싱 실패: {str(e)}")

        # Step 3: 공통 규칙(is_common) 확장 적용
        # 공통 규칙은 모든 시트를 검사하여 동일한 필드명이 있으면 자동으로 적용됨
        print("[ValidationService] Expanding common rules...")
        common_rules = [r for r in validation_rules if r.is_common]
        expanded_rules = []

        if common_rules:
            for canonical_name, data in sheet_data_map.items():
                sheet_columns = set(data["df"].columns)
                display_name = data["display_name"]

                for common_rule in common_rules:
                    # 해당 시트에 공통 규칙의 필드가 존재하는지 확인
                    if common_rule.field_name in sheet_columns:
                        # 이미 해당 시트에 대해 정의된 규칙인지 확인 (중복 적용 방지 - 원본이 해당 시트인 경우)
                        if common_rule.source.sheet_name == canonical_name:
                            continue

                        # 새 규칙 인스턴스 생성 (해당 시트용으로 복제)
                        from copy import deepcopy
                        new_rule = deepcopy(common_rule)
                        new_rule.source.sheet_name = canonical_name
                        # 규칙 ID에 시트명을 붙여서 유니크하게 만듦 (디버깅 용이)
                        new_rule.rule_id = f"{common_rule.rule_id}_COMMON_{canonical_name}"
                        new_rule.source.original_text += f" (공통규칙 적용: {display_name})"
                        
                        expanded_rules.append(new_rule)
                        # print(f"  - Applied common rule '{common_rule.field_name}' to sheet '{display_name}'")

            if expanded_rules:
                print(f"[ValidationService] Added {len(expanded_rules)} expanded common rules")
                validation_rules.extend(expanded_rules)

        # Step 4: 공통 검증 로직 실행
        validation_res = await self.validate_sheets(sheet_data_map, validation_rules)
        engine_duration = (datetime.now() - start_time).total_seconds()

        # =============================================================================
        # K-IFRS 2단계 검증 실행
        # =============================================================================
        kifrs_errors = []
        main_employee_df = None
        main_sheet_name = None

        # 1. 메인 직원 데이터 시트 찾기 ( heuristic: 필수 필드를 가장 많이 포함하는 시트)
        required_kifrs_cols = {'employee_code', 'hire_date', 'birth_date', 'average_wage'}
        best_match_score = 0
        for canonical_name, data in sheet_data_map.items():
            # df.columns를 소문자로 변환하여 비교
            df_cols = {str(col).lower() for col in data["df"].columns}
            match_score = len(required_kifrs_cols.intersection(df_cols))
            if match_score > best_match_score:
                best_match_score = match_score
                main_employee_df = data["df"]
                main_sheet_name = data["display_name"]

        # 2. K-IFRS 검증 실행
        if main_employee_df is not None and best_match_score >= 3: # 최소 3개 이상의 필수 컬럼이 있어야 실행
            print(f"[ValidationService] Running K-IFRS Step 2 validation on sheet: {main_sheet_name}")
            kifrs_engine = KIFRS_RuleEngine(main_employee_df)
            
            # TODO: reconciliation_params를 외부에서 받아와야 함
            kifrs_errors = kifrs_engine.run_all_checks(reconciliation_params=None)
            
            if kifrs_errors:
                print(f"[ValidationService] Found {len(kifrs_errors)} K-IFRS validation errors.")
                # 에러에 시트 이름 추가
                for error in kifrs_errors:
                    error.sheet = main_sheet_name
                
                # 3. 결과 병합
                # 기존 에러와 K-IFRS 에러 병합
                all_errors = validation_res.errors + kifrs_errors
                
                # 요약 정보 업데이트
                total_rows = validation_res.summary.total_rows
                
                # 시트별 오류 행 재계산
                error_rows_set = set()
                for err in all_errors:
                    if err.sheet and err.row > 0:
                        error_rows_set.add((err.sheet, err.row))

                # 시트별 요약 업데이트
                from collections import Counter
                error_counts_by_sheet = Counter(err.sheet for err in all_errors)

                if "sheets_summary" in validation_res.metadata:
                    for sheet_name, summary in validation_res.metadata["sheets_summary"].items():
                        summary["total_errors"] = error_counts_by_sheet.get(sheet_name, 0)

                # 전체 요약 업데이트
                validation_res.summary.total_errors = len(all_errors)
                validation_res.summary.error_rows = len(error_rows_set)
                validation_res.summary.valid_rows = validation_res.summary.total_rows - len(error_rows_set)
                
                # 상태 업데이트
                validation_res.validation_status = "FAIL"
                
                # 그룹 재계산 및 에러 목록 업데이트
                validation_res.errors = all_errors
                validation_res.error_groups = group_errors(all_errors)

        # Step 4: 매칭 통계 추가
        file_record = await self.rule_repository.get_rule_file(UUID(rule_file_id))
        matching_stats = {
            "matched_sheets": len(validation_res.metadata["sheets_summary"]),
            "total_rule_sheets": file_record.get('sheet_count', 0) if file_record else 0,
            "all_data_sheets": [data['display_name'] for data in sheet_data_map.values()],
            "all_rule_sheets": list(set(r.source.sheet_name for r in validation_rules))
        }
        validation_res.metadata.update({
            "employee_file_name": employee_file_name,
            "rule_file_id": rule_file_id,
            "matching_stats": matching_stats
        })

        # Step 5: 세션 저장
        session_id = str(uuid4())
        session_token = f"V-{datetime.now().strftime('%Y%m%d')}-{session_id[:8].upper()}"
        
        full_results_json = json.loads(validation_res.json())

        session_data = {
            "id": session_id,
            "session_token": session_token,
            "employee_file_name": employee_file_name,
            "rule_source_type": "database",
            "rule_file_id": rule_file_id,
            "total_rows": validation_res.summary.total_rows,
            "valid_rows": validation_res.summary.valid_rows,
            "error_rows": validation_res.summary.error_rows,
            "total_errors": validation_res.summary.total_errors,
            "rules_applied_count": validation_res.summary.rules_applied,
            "validation_status": validation_res.validation_status,
            "validation_processing_time_seconds": engine_duration,
            "full_results": full_results_json,
            "created_at": datetime.now().isoformat()
        }

        await self.validation_repository.create_session(session_data)

        # Step 6: 개별 에러 저장
        if validation_res.errors:
            error_records = [{
                "session_id": session_id,
                "sheet_name": err.sheet or "Sheet1",
                "row_number": err.row,
                "column_name": err.column,
                "rule_id": err.rule_id,
                "error_message": err.message,
                "actual_value": str(err.actual_value) if err.actual_value is not None else None,
                "expected_value": err.expected,
                "source_rule_text": err.source_rule
            } for err in validation_res.errors]
            
            await self.validation_repository.create_errors_batch(error_records)

        return {
            "status": "success",
            "session_id": session_id,
            "session_token": session_token,
            "summary": validation_res.summary.dict(),
            "total_processing_time_seconds": engine_duration
        }

    async def get_session_details(self, session_id: str) -> Dict[str, Any]:
        """
        세션 상세 정보 조회
        """
        session = await self.validation_repository.get_session(UUID(session_id))
        if not session:
            return None
            
        errors = await self.validation_repository.get_session_errors(UUID(session_id))
        
        return {
            "session": session,
            "errors": errors
        }

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        세션 목록 조회
        """
        return await self.validation_repository.list_sessions(limit, offset)
