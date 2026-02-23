from models import ValidationRule, ValidationResponse, ValidationError, ValidationSummary, ValidationErrorGroup
from utils.common import convert_numpy_types, group_errors, filter_garbage_rows
from utils.logger import get_logger
import numpy as np

logger = get_logger("validation_service")
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
from utils.field_matcher import FieldMatcher

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
        self.field_matcher = FieldMatcher()

    async def validate_sheets(
        self,
        sheet_data_map: Dict[str, Dict[str, Any]],
        validation_rules: List[ValidationRule]
    ) -> ValidationResponse:
        """
        공통 시트 검증 로직 (필드명 기반 규칙 적용)

        규칙은 시트에 종속되지 않고, 해당 필드가 존재하는 모든 시트에 적용됩니다.
        FieldMatcher를 통해 유사한 항목명에도 규칙을 적용합니다.

        Args:
            sheet_data_map: Canonical Name -> { display_name, original_name, df }
            validation_rules: 적용할 규칙 리스트

        Returns:
            ValidationResponse: 전체 검증 결과
        """
        all_errors = []
        all_sheets_summary = {}

        # 각 시트에 적용 가능한 규칙 필터링 (필드명 기반)
        for canonical_name, data in sheet_data_map.items():
            display_name = data["display_name"]
            df = data["df"]
            sheet_columns = [str(col) for col in df.columns]

            # FieldMatcher를 사용하여 규칙 필드명과 실제 컬럼 매핑
            field_mapping = self.field_matcher.match_rules_to_columns(validation_rules, sheet_columns)
            
            # 매핑된 컬럼을 기준으로 규칙 재구성 (필드명 변경)
            applicable_rules = []
            for rule in validation_rules:
                if rule.field_name in field_mapping:
                    # 원본 규칙을 복사하여 필드명만 매핑된 컬럼명으로 변경
                    mapped_col = field_mapping[rule.field_name]
                    rule_copy = rule.copy()
                    rule_copy.field_name = mapped_col
                    applicable_rules.append(rule_copy)

            if not applicable_rules:
                continue

            errors = self.rule_engine.validate(df, applicable_rules)
            summary = self.rule_engine.get_summary(len(df), len(applicable_rules))

            for error in errors:
                error.sheet = display_name

            all_errors.extend(errors)

            all_sheets_summary[display_name] = {
                "total_rows": len(df),
                "error_rows": summary.error_rows,
                "valid_rows": summary.valid_rows,
                "total_errors": len(errors),
                "rules_applied": len(applicable_rules)
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
        DB에 저장된 규칙을 사용하여 데이터 검증 수행 (Main Workflow)

        이 메서드는 업로드된 직원 데이터 파일에 대해 다음 절차를 수행합니다:
        1. **규칙 로드:** DB에서 AI가 해석한 검증 규칙을 불러옵니다.
        2. **파일 파싱:** 엑셀 파일을 읽고, 숨겨진 시트는 자동으로 제외합니다.
        3. **전처리 (Garbage Filtering):** 사번/입사일 등 핵심 정보가 없는 무의미한 행을 제거합니다.
        4. **규칙 확장 (Common Rules):** '공통 규칙'을 해당 컬럼이 존재하는 모든 시트에 적용합니다.
        5. **검증 실행 (Rule Engine):** 결정론적 엔진을 통해 데이터를 전수 검사합니다.
        6. **K-IFRS 심화 검증:** 보험수리적 가정, 연령 정합성 등 고급 로직을 체크합니다.
        7. **결과 정렬:** 사용자가 보기 편하도록 엑셀 컬럼 순서대로 결과를 정렬합니다.
        8. **세션 저장:** 검증 이력을 DB에 기록하고 세션 ID를 발급합니다.

        Args:
            rule_file_id: 적용할 규칙 파일의 UUID
            employee_file_content: 업로드된 직원 데이터 파일 (bytes)
            employee_file_name: 파일명

        Returns:
            Dict: 검증 결과 요약, 세션 ID, 처리 시간 등을 포함한 응답 객체
        """
        start_time = datetime.now()

        # Step 1: 규칙 로드
        logger.info(f"Loading rules from DB: {rule_file_id}")
        validation_rules = await self.ai_cache_service.get_cached_rules_as_validation_rules(rule_file_id)

        # Step 1.5: AI 해석이 없으면 자동으로 실행
        if not validation_rules:
            logger.info("No AI interpreted rules found. Running auto-interpretation...")
            interpret_result = await self.ai_cache_service.interpret_and_cache_rules(rule_file_id)
            logger.info(f"Auto-interpretation completed: {interpret_result}")

            # 다시 로드
            validation_rules = await self.ai_cache_service.get_cached_rules_as_validation_rules(rule_file_id)

            if not validation_rules:
                raise ValueError("규칙 해석에 실패했습니다. 규칙 파일을 확인해주세요.")

        # Step 2: 직원 데이터 파싱
        try:
            from utils.excel_parser import normalize_sheet_name, get_canonical_name, get_visible_sheet_names
            
            # 숨겨진 시트 제외
            visible_sheets = get_visible_sheet_names(employee_file_content)
            logger.info(f"Visible sheets: {visible_sheets}")
            
            sheet_data_map = {}
            
            for sheet_name in visible_sheets:
                df = pd.read_excel(io.BytesIO(employee_file_content), sheet_name=sheet_name)
                canonical_name = get_canonical_name(sheet_name)
                sheet_data_map[canonical_name] = {
                    "display_name": normalize_sheet_name(sheet_name),
                    "original_name": sheet_name,
                    "df": df
                }
        except Exception as e:
            raise ValueError(f"직원 데이터 파싱 실패: {str(e)}")

        # Step 2.5: 유효하지 않은 행(Garbage Rows) 필터링
        logger.info("Filtering garbage rows...")
        for canonical_name, data in sheet_data_map.items():
            original_len = len(data["df"])
            data["df"] = filter_garbage_rows(data["df"])
            dropped = original_len - len(data["df"])
            if dropped > 0:
                logger.debug(f"Sheet '{data['display_name']}': Dropped {dropped} garbage rows.")

        # Step 3: 필드명 기반 규칙 적용
        # 시트명 제거됨 - 모든 규칙은 해당 필드가 존재하는 모든 시트에 자동 적용됩니다.
        # 별도의 공통 규칙 확장이 필요 없음 (validate_sheets에서 필드 기반 매칭 수행)
        logger.info(f"Applying {len(validation_rules)} field-based rules to all matching sheets")

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
            logger.info(f"Running K-IFRS Step 2 validation on sheet: {main_sheet_name}")
            kifrs_engine = KIFRS_RuleEngine(main_employee_df)
            
            # TODO: reconciliation_params를 외부에서 받아와야 함
            kifrs_errors = kifrs_engine.run_all_checks(reconciliation_params=None)
            
            if kifrs_errors:
                logger.info(f"Found {len(kifrs_errors)} K-IFRS validation errors.")
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

        # Step 4: 매칭 통계 추가 (필드 기반)
        file_record = await self.rule_repository.get_rule_file(UUID(rule_file_id))

        # 고유 필드 목록
        unique_rule_fields = set(r.field_name for r in validation_rules)

        matching_stats = {
            "matched_sheets": len(validation_res.metadata["sheets_summary"]),
            "total_rule_fields": len(unique_rule_fields),
            "all_data_sheets": [data['display_name'] for data in sheet_data_map.values()],
            "all_rule_fields": list(unique_rule_fields)
        }

        # --- Rule-specific Status Calculation (Sheet-specific) ---
        from collections import Counter
        # Get sheet-specific error counts
        error_counts_by_sheet_rule = Counter((err.sheet, err.rule_id) for err in validation_res.errors)

        rules_by_sheet = {}
        column_order_map = {}
        for c_name, data in sheet_data_map.items():
            sheet_name = data['display_name']
            column_order_map[sheet_name] = {col: idx for idx, col in enumerate(data['df'].columns)}
            
            sheet_rules = []
            # FieldMatcher를 사용하여 이 시트에 실제로 적용된 매핑 확인
            sheet_columns = [str(col) for col in data["df"].columns]
            field_mapping = self.field_matcher.match_rules_to_columns(validation_rules, sheet_columns)
            
            for rule in validation_rules:
                # 규칙의 field_name이 이 시트의 컬럼과 매핑되는지 확인
                if rule.field_name in field_mapping:
                    mapped_col = field_mapping[rule.field_name]
                    err_count = error_counts_by_sheet_rule.get((sheet_name, rule.rule_id), 0)
                    status_msg = "검증 100% 완료!" if err_count == 0 else f"{err_count}건의 수정 필요사항 발견"
                    
                    sheet_rules.append({
                        "rule_id": rule.rule_id,
                        "field_name": mapped_col,
                        "rule_text": rule.source.original_text,
                        "error_count": err_count,
                        "status_message": status_msg,
                        "column_index": column_order_map[sheet_name].get(mapped_col, 999)
                    })
            
            # Sort by column index
            sheet_rules.sort(key=lambda x: x['column_index'])
            rules_by_sheet[sheet_name] = sheet_rules

        # 시트 순서 보장을 위해 metadata에 명시
        sheet_order = [data['display_name'] for data in sheet_data_map.values()]

        # --- AI Role Summary Generation ---
        ai_summary_text = (
            f"AI는 저장된 규칙 {len(validation_rules)}개를 로드하여 {matching_stats['matched_sheets']}개의 시트에 적용했습니다. "
            f"총 {validation_res.summary.total_rows}행의 데이터를 검증하고 "
            f"{validation_res.summary.total_errors}건의 데이터 오류를 식별했습니다."
        )

        validation_res.metadata.update({
            "employee_file_name": employee_file_name,
            "rule_file_id": rule_file_id,
            "matching_stats": matching_stats,
            "rules_by_sheet": rules_by_sheet,
            "sheet_order": sheet_order,
            "column_order_map": column_order_map,
            "ai_role_summary": ai_summary_text
        })

        # Step 5: 세션 저장
        session_id = str(uuid4())
        session_token = f"V-{datetime.now().strftime('%Y%m%d')}-{session_id[:8].upper()}"
        
        full_results_json = json.loads(validation_res.json())

        # full_results에서 개별 에러 목록 제거 (validation_errors 테이블에 별도 저장됨)
        # Supabase 응답 크기 제한(502) 방지
        full_results_slim = {k: v for k, v in full_results_json.items() if k != "errors"}

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
            "full_results": full_results_slim,
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
        full_results에서 제거된 errors를 validation_errors 테이블에서 재합성
        """
        session = await self.validation_repository.get_session(UUID(session_id))
        if not session:
            return None

        errors = await self.validation_repository.get_session_errors(UUID(session_id))

        # full_results에 errors가 없으면 DB에서 재합성
        full_results = session.get("full_results")
        if full_results and "errors" not in full_results and errors:
            full_results["errors"] = [
                {
                    "sheet": e.get("sheet_name"),
                    "row": e.get("row_number"),
                    "column": e.get("column_name"),
                    "rule_id": e.get("rule_id"),
                    "message": e.get("error_message"),
                    "actual_value": e.get("actual_value"),
                    "expected": e.get("expected_value"),
                    "source_rule": e.get("source_rule_text", "")
                }
                for e in errors
            ]
            session["full_results"] = full_results

        return {
            "session": session,
            "errors": errors
        }

    async def list_sessions(self, limit: int = 50, offset: int = 0) -> List[Dict[str, Any]]:
        """
        세션 목록 조회
        """
        return await self.validation_repository.list_sessions(limit, offset)
