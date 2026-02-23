"""
K-IFRS Router - K-IFRS 1019 DBO 전문 검증
"""

import json
import traceback
import io
import numpy as np
import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from starlette.responses import Response

from dependencies import kifrs_dbo_service, rule_service
from utils.common import filter_garbage_rows
from utils.logger import get_logger


class _NumpyEncoder(json.JSONEncoder):
    """numpy 타입 → Python 네이티브 타입 JSON 인코더"""
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super().default(obj)


def _numpy_json_response(data: dict, status_code: int = 200) -> Response:
    """numpy 타입을 안전하게 직렬화하여 JSON Response 반환 (jsonable_encoder 우회)"""
    body = json.dumps(data, ensure_ascii=False, cls=_NumpyEncoder)
    return Response(content=body, status_code=status_code, media_type="application/json")

logger = get_logger("kifrs_router")
router = APIRouter(tags=["K-IFRS"])


@router.get("/api/kifrs/ping")
async def kifrs_ping():
    """코드 리로드 확인용"""
    return {"status": "ok", "version": "numpy-fix-v2"}


@router.get("/kifrs-references")
async def get_kifrs_references():
    """K-IFRS 1019 참조 정보 조회"""
    from models import KIFRS_1019_REFERENCES
    return KIFRS_1019_REFERENCES


@router.post("/api/kifrs/dbo-validate")
async def kifrs_dbo_validate(
    employee_file: UploadFile = File(..., description="직원 데이터 파일"),
    base_date: str = Form("", description="평가기준일 (YYYYMMDD)"),
    discount_rate: str = Form("", description="할인율 (%)"),
    salary_growth: str = Form("", description="급여상승률 (%)"),
    turnover_rate: str = Form("", description="퇴직률 (%)"),
    retirement_age: str = Form("", description="정년 (세)")
):
    """K-IFRS 1019 DBO 전문 검증"""
    try:
        content = await employee_file.read()

        from utils.excel_parser import get_visible_sheet_names
        visible_sheets = get_visible_sheet_names(content)

        sheet_data = {}
        column_names = {}
        for sheet_name in visible_sheets:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)
            df = filter_garbage_rows(df)
            sheet_data[sheet_name] = df
            column_names[sheet_name] = [str(c) for c in df.columns]

        assumptions = {}
        if discount_rate:
            assumptions["discount_rate"] = discount_rate
        if salary_growth:
            assumptions["salary_growth"] = salary_growth
        if turnover_rate:
            assumptions["turnover_rate"] = turnover_rate
        if retirement_age:
            assumptions["retirement_age"] = retirement_age

        result = kifrs_dbo_service.validate_dbo_data(
            sheet_data=sheet_data,
            column_names=column_names,
            base_date=base_date if base_date else None,
            assumptions=assumptions if assumptions else None
        )

        return _numpy_json_response(result)

    except Exception as e:
        logger.error(f"DBO validation error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"error": "DBO validation failed", "message": str(e)}
        )


@router.get("/api/kifrs/standard-rules")
async def get_standard_rule_templates():
    """K-IFRS 1019 DBO 표준 검증 규칙 템플릿 목록 반환"""
    try:
        templates = kifrs_dbo_service.get_standard_rule_templates()
        return {
            "total": len(templates),
            "templates": templates
        }
    except Exception as e:
        logger.error(f"Template retrieval error: {e}")
        raise HTTPException(
            status_code=500,
            detail={"error": "Template retrieval failed", "message": str(e)}
        )


@router.post("/api/kifrs/apply-standard-rules")
async def apply_standard_rules(rule_file_id: str = Form(..., description="대상 규칙 파일 ID")):
    """표준 K-IFRS 규칙 템플릿을 기존 규칙 파일에 일괄 적용"""
    try:
        templates = kifrs_dbo_service.get_standard_rule_templates()
        created_count = 0

        for tmpl in templates:
            try:
                rule_data = {
                    "rule_file_id": rule_file_id,
                    "row_number": "0",
                    "column_name": tmpl["field_name"],
                    "rule_text": tmpl["rule_text"],
                    "ai_rule_type": tmpl["ai_rule_type"],
                    "ai_parameters": tmpl.get("ai_parameters", {}),
                    "ai_error_message": tmpl.get("ai_error_message", tmpl["rule_text"]),
                    "ai_confidence_score": 1.0,
                    "is_common": True,
                }
                await rule_service.repository.create_rule(rule_data)
                created_count += 1
            except Exception as inner_e:
                logger.warning(f"Rule template apply error: {inner_e}")

        return {
            "status": "success",
            "total_templates": len(templates),
            "created_count": created_count,
            "message": f"{created_count}개 표준 규칙이 적용되었습니다."
        }

    except Exception as e:
        logger.error(f"Apply standard rules error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"error": "Standard rules application failed", "message": str(e)}
        )
