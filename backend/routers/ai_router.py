"""
AI Router - AI 분석 엔드포인트
==============================
크로스필드 분석, 데이터 프로파일링, 자연어 질의/수정,
규칙 자동 생성, K-IFRS 컴플라이언스, 데이터 완전성 점수
"""

import io
import traceback
from datetime import datetime
from urllib.parse import quote

import pandas as pd
from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse

from dependencies import ai_interpreter, fix_service
from utils.logger import get_logger
from utils.excel_parser import get_visible_sheet_names

logger = get_logger("ai_router")
router = APIRouter(tags=["AI Analysis"])


# =============================================================================
# Phase 10: AI Smart Analysis Endpoints
# =============================================================================


@router.post("/ai/cross-field-analysis")
async def cross_field_analysis(
    employee_file: UploadFile = File(..., description="직원 데이터 파일"),
    ai_provider: str = Form("openai", description="AI Provider")
):
    """
    크로스필드 논리 모순 탐지

    데이터의 필드 간 관계를 AI가 추론하여 논리적 모순을 자동 발견합니다.
    규칙 파일 없이 데이터만으로 동작합니다.
    """
    try:
        logger.info(f"Cross-field analysis requested (provider: {ai_provider})")
        content = await employee_file.read()

        visible_sheets = get_visible_sheet_names(content)

        sheet_data_samples = {}
        column_names = {}

        for sheet_name in visible_sheets:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)

            # 빈 행 제거
            df = df.dropna(how='all')

            cols = [str(c) for c in df.columns]
            column_names[sheet_name] = cols

            # 샘플 데이터 (최대 50행)
            samples = []
            for idx, row in df.head(50).iterrows():
                row_dict = {}
                for col in cols:
                    val = row[col]
                    if pd.notna(val):
                        row_dict[col] = val
                row_dict["__row_number__"] = idx + 2  # Excel 1-based + header
                samples.append(row_dict)

            sheet_data_samples[sheet_name] = samples

        result = await ai_interpreter.analyze_cross_field(
            sheet_data_samples, column_names, provider=ai_provider
        )

        logger.info(f"Cross-field analysis complete: {result.get('total_issues', 0)} issues found")
        return result

    except Exception as e:
        logger.error(f"Cross-field analysis error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "Cross-field analysis failed", "message": str(e)}
        )


@router.post("/ai/data-profile")
async def data_profile(
    employee_file: UploadFile = File(..., description="직원 데이터 파일"),
    ai_provider: str = Form("openai", description="AI Provider")
):
    """
    AI 데이터 프로파일링 (Zero-Rule Anomaly Scan)

    규칙 파일 없이 데이터만 업로드하면 이상치, 형식 불일치, 결측 패턴, 중복 의심 등을 자동 탐지합니다.
    """
    try:
        logger.info(f"Data profiling requested (provider: {ai_provider})")
        content = await employee_file.read()

        visible_sheets = get_visible_sheet_names(content)

        sheet_data_samples = {}
        column_names = {}
        sheet_stats = {}

        for sheet_name in visible_sheets:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)
            df = df.dropna(how='all')

            cols = [str(c) for c in df.columns]
            column_names[sheet_name] = cols

            # 통계 정보 수집
            null_counts = {}
            for col in cols:
                null_count = int(df[col].isna().sum())
                empty_count = int((df[col].astype(str).str.strip().isin(['', 'None', 'nan', 'NaT'])).sum())
                null_counts[col] = null_count + empty_count

            sheet_stats[sheet_name] = {
                "total_rows": len(df),
                "columns_count": len(cols),
                "null_counts": null_counts
            }

            # 샘플 데이터 (최대 50행)
            samples = []
            for idx, row in df.head(50).iterrows():
                row_dict = {}
                for col in cols:
                    val = row[col]
                    if pd.notna(val):
                        row_dict[col] = val
                row_dict["__row_number__"] = idx + 2
                samples.append(row_dict)

            sheet_data_samples[sheet_name] = samples

        result = await ai_interpreter.analyze_data_profile(
            sheet_data_samples, column_names, sheet_stats, provider=ai_provider
        )

        logger.info(f"Data profiling complete: health_score={result.get('health_score')}, findings={len(result.get('findings', []))}")
        return result

    except Exception as e:
        logger.error(f"Data profiling error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "Data profiling failed", "message": str(e)}
        )


# =============================================================================
# Phase 11: Advanced AI Features
# =============================================================================


@router.post("/api/ai/natural-query")
async def natural_language_query(
    employee_file: UploadFile = File(..., description="직원 데이터 파일"),
    query: str = Form(..., description="자연어 질의"),
    ai_provider: str = Form("local", description="AI Provider")
):
    """
    자연어 질의 검증 - 한국어로 데이터에 질문하면 AI가 해당 행을 찾아 답변합니다.

    예시: "퇴직일이 2025년인 직원 중 급여가 5000만원 이상인 사람은?"
    """
    try:
        logger.info(f"Natural language query: '{query}' (provider: {ai_provider})")
        content = await employee_file.read()

        visible_sheets = get_visible_sheet_names(content)

        sheet_data = {}
        column_names = {}

        for sheet_name in visible_sheets:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)
            df = df.dropna(how='all')
            sheet_data[sheet_name] = df
            column_names[sheet_name] = [str(c) for c in df.columns]

        result = await ai_interpreter.query_data_natural_language(
            query=query,
            sheet_data_samples=sheet_data,
            column_names=column_names,
            provider=ai_provider
        )

        logger.info(f"Query result: {result.get('total_matches', 0)} matches")
        return result

    except Exception as e:
        logger.error(f"Natural query error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "Natural language query failed", "message": str(e)}
        )


@router.post("/api/ai/auto-rules")
async def auto_generate_rules(
    employee_file: UploadFile = File(..., description="직원 데이터 파일"),
    ai_provider: str = Form("local", description="AI Provider")
):
    """
    규칙 자동 생성 - AI가 데이터 패턴을 분석하여 암묵적 규칙을 역추론 제안합니다.

    예시: "사번 컬럼은 항상 8자리 숫자" -> format 규칙 자동 제안
    """
    try:
        logger.info(f"Auto rule generation (provider: {ai_provider})")
        content = await employee_file.read()

        visible_sheets = get_visible_sheet_names(content)

        sheet_data = {}
        column_names = {}

        for sheet_name in visible_sheets:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)
            df = df.dropna(how='all')
            sheet_data[sheet_name] = df
            column_names[sheet_name] = [str(c) for c in df.columns]

        result = await ai_interpreter.auto_generate_rules(
            sheet_data=sheet_data,
            column_names=column_names,
            provider=ai_provider
        )

        logger.info(f"Auto-generated {result.get('total_suggestions', 0)} rules")
        return result

    except Exception as e:
        logger.error(f"Auto rule generation error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "Auto rule generation failed", "message": str(e)}
        )


@router.post("/api/ai/kifrs-compliance")
async def kifrs_compliance_check(
    employee_file: UploadFile = File(..., description="직원 데이터 파일"),
    ai_provider: str = Form("local", description="AI Provider")
):
    """
    K-IFRS 1019 컴플라이언스 어드바이저 - DBO 필수 항목 누락 및 계리적 합리성을 검토합니다.
    """
    try:
        logger.info(f"K-IFRS compliance check (provider: {ai_provider})")
        content = await employee_file.read()

        visible_sheets = get_visible_sheet_names(content)

        sheet_data = {}
        column_names = {}

        for sheet_name in visible_sheets:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)
            df = df.dropna(how='all')
            sheet_data[sheet_name] = df
            column_names[sheet_name] = [str(c) for c in df.columns]

        result = await ai_interpreter.check_kifrs_compliance(
            sheet_data=sheet_data,
            column_names=column_names,
            provider=ai_provider
        )

        logger.info(f"Compliance score: {result.get('overall_score')}%")
        return result

    except Exception as e:
        logger.error(f"Compliance check error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "K-IFRS compliance check failed", "message": str(e)}
        )


@router.post("/api/ai/natural-fix")
async def natural_language_fix(
    employee_file: UploadFile = File(..., description="직원 데이터 파일"),
    instruction: str = Form(..., description="자연어 수정 지시"),
    ai_provider: str = Form("local", description="AI Provider"),
    preview_only: bool = Form(True, description="미리보기만 (True) 또는 실제 적용 (False)")
):
    """
    자연어 수정 지시 - "퇴직일 비어있는 퇴직자는 2025-12-31로 채워줘" 같은 지시를 처리합니다.
    """
    try:
        logger.info(f"Natural fix instruction: '{instruction}' (preview={preview_only})")
        content = await employee_file.read()

        visible_sheets = get_visible_sheet_names(content)
        column_names = {}
        for sheet_name in visible_sheets:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)
            column_names[sheet_name] = [str(c) for c in df.columns]

        # 지시 파싱
        parsed_fix = await ai_interpreter.parse_fix_instruction(
            instruction=instruction,
            column_names=column_names,
            provider=ai_provider
        )

        if parsed_fix.get("confidence", 0) < 0.3:
            return {
                "status": "parse_failed",
                "instruction": instruction,
                "interpretation": parsed_fix.get("interpretation", "파싱 실패"),
                "confidence": parsed_fix.get("confidence", 0),
                "message": "지시를 정확히 이해하지 못했습니다. 더 구체적으로 입력해주세요."
            }

        if preview_only:
            # 미리보기: 영향 받는 행 수만 계산
            preview = fix_service.apply_natural_language_fix(
                file_content=content,
                instruction=instruction,
                parsed_fix=parsed_fix,
                filename=employee_file.filename or ""
            )
            return {
                "status": "preview",
                "instruction": instruction,
                "interpretation": parsed_fix.get("interpretation", ""),
                "confidence": parsed_fix.get("confidence", 0),
                "target_field": parsed_fix.get("target_field", ""),
                "new_value": parsed_fix.get("new_value", ""),
                "affected_count": preview["affected_count"],
                "changes_preview": preview["changes"][:10],
                "summary": preview["summary"]
            }
        else:
            # 실제 적용
            result = fix_service.apply_natural_language_fix(
                file_content=content,
                instruction=instruction,
                parsed_fix=parsed_fix,
                filename=employee_file.filename or ""
            )

            base_name = (employee_file.filename or "data").rsplit('.', 1)[0]
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"{base_name}_nlfix_{timestamp}.xlsx"
            filename_encoded = quote(filename, safe='')

            return StreamingResponse(
                io.BytesIO(result["modified_file"]),
                media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                headers={
                    "Content-Disposition": f"attachment; filename*=UTF-8''{filename_encoded}",
                    "Content-Length": str(len(result["modified_file"])),
                    "Cache-Control": "no-cache"
                }
            )

    except Exception as e:
        logger.error(f"Natural fix error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "Natural language fix failed", "message": str(e)}
        )


@router.post("/api/data/completeness")
async def data_completeness_score(
    employee_file: UploadFile = File(..., description="직원 데이터 파일")
):
    """
    데이터 완전성 점수 - 각 시트/컬럼별 필드 채움 비율을 점수화합니다.
    """
    try:
        content = await employee_file.read()
        visible_sheets = get_visible_sheet_names(content)

        sheet_data = {}
        column_names = {}
        for sheet_name in visible_sheets:
            df = pd.read_excel(io.BytesIO(content), sheet_name=sheet_name)
            df = df.dropna(how='all')
            sheet_data[sheet_name] = df
            column_names[sheet_name] = [str(c) for c in df.columns]

        result = ai_interpreter.calculate_completeness_score(sheet_data, column_names)
        return result

    except Exception as e:
        logger.error(f"Completeness score error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "Completeness score calculation failed", "message": str(e)}
        )
