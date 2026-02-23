"""
Validation Router - 검증 관련 엔드포인트
========================================
Legacy 파일 기반 검증, DB 규칙 기반 검증, 규칙 해석,
오류 설명, 수정 제안/적용 등을 담당합니다.
"""

import io
import json
import traceback
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID

import pandas as pd
from pydantic import BaseModel
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse

from dependencies import (
    ai_interpreter,
    validation_service,
    rule_service,
    learning_service,
    fix_service,
    ai_cache_service,
)
from utils.logger import get_logger
from utils.excel_parser import (
    parse_rules_from_excel,
    normalize_sheet_name,
    get_canonical_name,
    sanitize_sheet_name,
    get_visible_sheet_names,
)
from utils.common import group_errors, filter_garbage_rows
from utils.field_matcher import FieldMatcher
from config import settings
from models import (
    ValidationResponse,
    ValidationError,
    AIInterpretationResponse,
    BatchFixRequest,
    FixSuggestion,
)

logger = get_logger("validation_router")

router = APIRouter(tags=["Validation"])


# =============================================================================
# Request Models (이 라우터 전용)
# =============================================================================

class ExplainErrorRequest(BaseModel):
    error: ValidationError
    ai_provider: str = "openai"


class FixSuggestRequest(BaseModel):
    session_id: str
    error_ids: Optional[List[str]] = None
    ai_provider: str = "openai"


# =============================================================================
# 1. Legacy File-based Validation
# =============================================================================

@router.post("/validate", response_model=ValidationResponse)
async def validate_data(
    employee_file: UploadFile = File(..., description="직원 데이터 파일 (Excel A)"),
    rules_file: UploadFile = File(..., description="검증 규칙 파일 (Excel B)"),
    ai_provider: str = Form("openai", description="AI Provider (openai, anthropic, gemini)")
):
    """
    [Legacy] 파일 기반 즉시 검증 엔드포인트

    규칙 파일을 DB에 저장하지 않고, 업로드된 두 파일(데이터, 규칙)을 즉시 분석하여 결과를 반환합니다.

    Process:
    1. 직원 데이터(.xlsx) 로드 및 숨겨진 시트 필터링
    2. 무의미한 행(Garbage Row) 자동 감지 및 제거
    3. 규칙 파일 로드 및 AI 해석 (Local/Cloud Hybrid)
    4. Rule Engine을 통한 검증 실행
    5. 결과 리턴 (메타데이터 및 통계 포함)
    """
    try:
        # Step 1: Excel A 읽기 (직원 데이터)
        logger.info("[Step 1] Reading employee data...")
        employee_content = await employee_file.read()

        # 숨겨진 시트 제외하고 로드
        visible_sheets = get_visible_sheet_names(employee_content)
        logger.info(f"[Step 1] Visible sheets: {visible_sheets}")

        sheet_data_map = {}
        sheet_mapping_info = {}

        for sheet_name in visible_sheets:
            df = pd.read_excel(io.BytesIO(employee_content), sheet_name=sheet_name)
            norm_name = normalize_sheet_name(sheet_name)
            canonical_name = get_canonical_name(sheet_name)

            sheet_data_map[canonical_name] = {
                "display_name": norm_name,
                "original_name": sheet_name,
                "df": df
            }
            sheet_mapping_info[canonical_name] = sheet_name

        # Step 1.5: 유효하지 않은 행(Garbage Rows) 필터링
        logger.info("[Step 1.5] Filtering garbage rows...")
        for canonical_name, data in sheet_data_map.items():
            data["df"] = filter_garbage_rows(data["df"])

        # Step 2: Excel B 읽기 (자연어 규칙)
        logger.info("[Step 2] Reading validation rules...")
        rules_content = await rules_file.read()
        natural_language_rules, field_rule_counts, total_raw_rows, reported_max_row = parse_rules_from_excel(rules_content)

        # 필드명 기반 규칙 관리 (시트명 제거됨)
        all_rule_fields = sorted(list(field_rule_counts.keys()))

        # Step 3: AI 규칙 해석
        logger.info(f"[Step 3] AI interpreting rules using {ai_provider}...")
        ai_response: AIInterpretationResponse = await ai_interpreter.interpret_rules(
            natural_language_rules,
            provider=ai_provider
        )

        # Step 4: 결정론적 검증 실행 (필드 기반 - 모든 시트에 적용)
        logger.info("[Step 4] Running deterministic validation...")
        validation_res = await validation_service.validate_sheets(sheet_data_map, ai_response.rules)

        # Step 5: 응답 생성 및 메타데이터 추가 (필드 기반)
        all_data_sheets = sorted(list(sheet_mapping_info.values()))

        # 필드별 규칙 개수 표시
        display_list = []
        for field_name in all_rule_fields:
            rule_count = field_rule_counts.get(field_name, 0)
            display_list.append(f"{field_name} ({rule_count}개 규칙)")

        matching_stats = {
            "total_rule_fields": len(all_rule_fields),
            "matched_sheets": len(validation_res.metadata.get("sheets_summary", {})),
            "all_rule_fields": display_list,
            "all_data_sheets": all_data_sheets,
            "total_raw_rows": total_raw_rows,
            "reported_max_row": reported_max_row,
            "total_rules_count": len(ai_response.rules)
        }

        validation_res.conflicts = ai_response.conflicts

        # 실제 사용된 엔진 확인
        actual_model = "local-parser" if not ai_interpreter.use_cloud_ai else f"cloud-{ai_provider}"

        # --- Rule-specific Status Calculation (Sheet-specific) ---
        error_counts_by_sheet_rule = Counter((err.sheet, err.rule_id) for err in validation_res.errors)

        rules_by_sheet = {}
        for c_name, data in sheet_data_map.items():
            sheet_name = data['display_name']
            sheet_rules = []

            # FieldMatcher를 사용하여 이 시트에 실제로 적용된 매핑 확인
            sheet_columns = [str(col) for col in data["df"].columns]
            matcher = FieldMatcher()
            field_mapping = matcher.match_rules_to_columns(ai_response.rules, sheet_columns)

            # 컬럼 순서 맵 생성
            col_order = {str(col): idx for idx, col in enumerate(data["df"].columns)}

            for rule in ai_response.rules:
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
                        "column_index": col_order.get(mapped_col, 999)
                    })

            sheet_rules.sort(key=lambda x: x['column_index'])
            rules_by_sheet[sheet_name] = sheet_rules

        # --- AI Role Summary Generation ---
        matched_fields_count = len(all_rule_fields)
        ai_summary_text = (
            f"AI는 {len(ai_response.rules)}개의 자연어 규칙을 해석하여 {matched_fields_count}개의 필드에 적용했습니다. "
            f"총 {validation_res.summary.total_rows}행의 데이터를 검증하는 과정에서 "
            f"{len(ai_response.conflicts)}건의 규칙 충돌 가능성을 분석하고, "
            f"{validation_res.summary.total_errors}건의 데이터 오류를 식별했습니다."
        )

        validation_res.metadata.update({
            "employee_file_name": employee_file.filename,
            "rules_file_name": rules_file.filename,
            "ai_model_version": actual_model,
            "system_version": settings.APP_VERSION,
            "ai_processing_time_seconds": ai_response.processing_time_seconds,
            "total_errors": validation_res.summary.total_errors,
            "errors_shown": min(validation_res.summary.total_errors, 200),
            "error_groups_count": len(validation_res.error_groups),
            "matching_stats": matching_stats,
            "sheet_order": [data["display_name"] for data in sheet_data_map.values()],
            "rules_by_sheet": rules_by_sheet,
            "ai_role_summary": ai_summary_text
        })

        logger.info("[OK] Response ready")
        return validation_res

    except Exception as e:
        logger.error(f"Validation error: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Validation failed",
                "message": str(e),
                "type": type(e).__name__
            }
        )


# =============================================================================
# 2. DB Rule-based Validation
# =============================================================================

@router.post("/validate-with-db-rules")
async def validate_with_db_rules(
    rule_file_id: str,
    employee_file: UploadFile = File(..., description="직원 데이터 파일 (Excel A)")
):
    """
    DB에 저장된 규칙을 사용하여 데이터 검증 수행

    Args:
        rule_file_id: 규칙 파일 UUID
        employee_file: 직원 데이터 파일

    Returns:
        Dict: 검증 결과 요약 및 세션 ID
    """
    try:
        logger.info(f"Validating with DB rules: {rule_file_id}")

        # Read file content
        content = await employee_file.read()

        result = await validation_service.validate_with_db_rules(
            rule_file_id=rule_file_id,
            employee_file_content=content,
            employee_file_name=employee_file.filename
        )

        # 학습: 검증 결과 피드백 기록
        try:
            # 1. 세션 상세 정보 조회 (오류 내역 확인용)
            session_id = result.get("session_id")
            session_details = await validation_service.get_session_details(session_id)

            if session_details:
                errors = session_details.get("errors", [])

                # 규칙별 오류 횟수 집계
                error_counts = Counter(e['rule_id'] for e in errors)

                # 2. 해당 파일의 모든 규칙 조회 (패턴 ID 확인용)
                db_rules = await rule_service.repository.get_rules_by_file(UUID(rule_file_id), active_only=True)

                # 3. 각 규칙별로 피드백 기록
                total_rows = result.get("summary", {}).get("total_rows", 0)

                for rule in db_rules:
                    rule_id = str(rule['id'])
                    pattern_id = None

                    # AI Rule ID가 'LEARNED_'로 시작하면 패턴 ID로 간주
                    ai_rule_id = rule.get('ai_rule_id', '')
                    if ai_rule_id and ai_rule_id.startswith('LEARNED_'):
                        pattern_id = ai_rule_id.replace('LEARNED_', '')

                    rule_error_count = error_counts.get(rule_id, 0)

                    if pattern_id:
                        # 기존 패턴에 대한 피드백 기록
                        await learning_service.record_validation_result(
                            rule_id=rule_id,
                            pattern_id=pattern_id,
                            total_rows=total_rows,
                            error_count=rule_error_count
                        )
                    else:
                        # AI 해석 규칙 (아직 학습되지 않음) - 자동 학습 시도
                        # 규칙별 성공률 계산
                        rule_success_rate = 1.0 - (rule_error_count / total_rows) if total_rows > 0 else 0

                        rule_text = rule.get('rule_text', '')
                        field_name = rule.get('field_name', '')

                        if rule_text and field_name:
                            ai_interpretation = {
                                "rule_type": rule.get('ai_rule_type'),
                                "parameters": rule.get('ai_parameters', {}),
                                "error_message": rule.get('ai_error_message', ''),
                                "confidence_score": rule.get('ai_confidence_score', 0.8)
                            }

                            await learning_service.auto_learn_from_validation(
                                rule_id=rule_id,
                                rule_text=rule_text,
                                field_name=field_name,
                                ai_interpretation=ai_interpretation,
                                validation_success_rate=rule_success_rate,
                                total_rows=total_rows
                            )

                logger.info(f"Recorded validation feedback for session: {session_id}")

        except Exception as e:
            logger.error(f"Failed to record learning feedback: {e}")
            # 피드백 실패는 무시 (메인 로직에 영향 주지 않음)

        return result

    except ValueError as ve:
        logger.error(f"Validation value error: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Validation failed",
                "message": str(ve)
            }
        )
    except Exception as e:
        logger.error(f"Unexpected validation error: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Validation failed",
                "message": str(e)
            }
        )


# =============================================================================
# 3. Rules-only Interpretation
# =============================================================================

@router.post("/interpret-rules")
async def interpret_rules_only(
    rules_file: UploadFile = File(..., description="검증 규칙 파일 (Excel B)"),
    ai_provider: str = Form("openai", description="AI Provider")
):
    """
    규칙만 해석 (검증 실행 없이)
    """
    try:
        rules_content = await rules_file.read()
        natural_language_rules, _, _, _ = parse_rules_from_excel(rules_content)
        ai_response = await ai_interpreter.interpret_rules(natural_language_rules, provider=ai_provider)

        return {
            "status": "success",
            "rules_count": len(ai_response.rules),
            "conflicts_count": len(ai_response.conflicts),
            "rules": [rule.dict() for rule in ai_response.rules],
            "conflicts": [conflict.dict() for conflict in ai_response.conflicts],
            "summary": ai_response.ai_summary
        }
    except Exception as e:
        logger.error(f"Rule interpretation failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Rule interpretation failed", "message": str(e)}
        )


# =============================================================================
# 4. Error Explanation
# =============================================================================

@router.post("/errors/explain")
async def explain_error(request: ExplainErrorRequest):
    """
    단일 검증 오류에 대한 AI의 상세 설명 및 조치 권고를 받습니다.
    """
    try:
        explanation = await ai_interpreter.get_error_explanation(request.error, provider=request.ai_provider)
        return explanation
    except Exception as e:
        logger.error(f"Error getting explanation: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get error explanation",
                "message": str(e)
            }
        )


# =============================================================================
# 5. Fix Suggestions
# =============================================================================

@router.post("/fix/suggest", response_model=List[FixSuggestion])
async def suggest_fixes(request: FixSuggestRequest):
    """
    오류에 대한 AI 수정 제안 생성
    """
    try:
        logger.info(f"Suggest fixes for session: {request.session_id}, errors: {len(request.error_ids) if request.error_ids else 'all'}")
        suggestions = await fix_service.suggest_fixes(
            request.session_id,
            request.error_ids,
            provider=request.ai_provider
        )
        logger.info(f"Generated {len(suggestions)} fix suggestions")
        return suggestions
    except Exception as e:
        logger.error(f"Suggest fixes failed: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to generate fix suggestions", "message": str(e)}
        )


# =============================================================================
# 6. Apply Fixes
# =============================================================================

@router.post("/fix/apply")
async def apply_fixes(
    fix_request_json: str = Form(..., description="BatchFixRequest JSON string"),
    original_file: UploadFile = File(..., description="Original Excel file")
):
    """
    수정 사항을 적용하여 엑셀 파일 다운로드
    """
    try:
        # Parse JSON payload
        request_data = json.loads(fix_request_json)
        # Validate with Pydantic
        fix_request = BatchFixRequest(**request_data)

        logger.info(f"Applying {len(fix_request.fixes)} fixes to file: {original_file.filename}")

        # Read file content
        content = await original_file.read()

        # Apply fixes
        modified_excel = fix_service.apply_fixes_to_excel(content, fix_request.fixes)

        # Generate filename
        base_name = original_file.filename.rsplit('.', 1)[0]
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_fixed_{timestamp}.xlsx"

        return StreamingResponse(
            io.BytesIO(modified_excel),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(modified_excel)),
                "Cache-Control": "no-cache"
            }
        )

    except Exception as e:
        logger.error(f"Apply fixes failed: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to apply fixes", "message": str(e)}
        )
