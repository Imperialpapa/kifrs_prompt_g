"""
K-IFRS 1019 DBO Validation System - FastAPI Backend
===================================================
작성일: 2024-05-22
설명: 확정급여채무(DBO) 평가를 위한 데이터 정합성 검증 시스템의 메인 API 서버입니다.

[시스템 아키텍처]
1. Presentation Layer (Frontend): HTML/Alpine.js 기반의 SPA
2. API Layer (This File): FastAPI를 이용한 RESTful API 제공
3. Service Layer: 비즈니스 로직 처리 (Validation, Rule, Fix, Statistics)
4. Domain Layer: AI 해석(AI Layer) 및 결정론적 검증(Rule Engine)
5. Data Layer: Supabase (PostgreSQL) 및 로컬 파일 처리

[주요 기능]
- Excel 파일(.xlsx) 업로드 및 파싱
- 자연어 규칙의 AI 해석 및 DB 저장
- 결정론적 규칙 엔진을 통한 데이터 검증
- K-IFRS 1019 특화 검증 (보험수리적 가정 등)
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
import pandas as pd
import io
import os
from typing import List, Dict, Any, Optional
from uuid import UUID
import traceback
from datetime import datetime
from pydantic import BaseModel

from models import (
    ValidationResponse,
    ValidationError,
    AIInterpretationResponse,
    RuleConflict,
    ValidationErrorGroup,
    RuleFileUpload,
    RuleFileResponse,
    RuleUpdate,
    RuleCreate,
    RuleDetail,
    FalsePositiveFeedback,
    BatchFixRequest,
    FixSuggestion
)
from ai_layer import AIRuleInterpreter
from rule_engine import RuleEngine
from services.rule_service import RuleService
from services.ai_cache_service import AICacheService
from services.validation_service import ValidationService
from services.feedback_service import FeedbackService
from services.statistics_service import StatisticsService
from services.fix_service import FixService
from services.learning_service import LearningService
from database.supabase_client import supabase
from utils.excel_parser import parse_rules_from_excel, normalize_sheet_name, get_canonical_name
from utils.common import group_errors

app = FastAPI(
    title="K-IFRS 1019 DBO Validator",
    description="AI-Powered Data Validation for Defined Benefit Obligations",
    version="1.2.0"
)

# CORS 설정 (모바일 및 다양한 클라이언트 접근 지원)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# =============================================================================
# 전역 서비스 레이어 초기화
# =============================================================================

ai_interpreter = AIRuleInterpreter()
rule_service = RuleService()
ai_cache_service = AICacheService()
validation_service = ValidationService()
feedback_service = FeedbackService()
statistics_service = StatisticsService()
fix_service = FixService()
learning_service = LearningService(supabase_client=supabase)

# =============================================================================
# 유틸리티 및 헬스체크 엔드포인트
# =============================================================================

@app.get("/")
async def root():
    """
    프론트엔드 정적 파일(index.html) 제공
    """
    # 루트 디렉토리 또는 현재 디렉토리에서 index.html 탐색
    if os.path.exists("../index.html"):
        return FileResponse("../index.html")
    elif os.path.exists("index.html"):
        return FileResponse("index.html")
    
    return {
        "service": "K-IFRS 1019 DBO Validation System",
        "version": "1.4.1",
        "status": "operational (Frontend file not found)"
    }


@app.get("/api")
async def api_info():
    """API 기본 정보 제공"""
    return {
        "service": "K-IFRS 1019 DBO Validation System",
        "version": "1.4.1",
        "status": "operational",
        "features": [
            "다중 시트 검증",
            "AI 규칙 해석",
            "개인정보 마스킹",
            "K-IFRS 특화 로직"
        ]
    }


@app.get("/health")
async def health_check():
    """시스템 상태 확인"""
    return {
        "status": "healthy",
        "ai_layer": "operational",
        "rule_engine": "operational"
    }


@app.get("/version")
async def get_version():
    """시스템 버전 정보 반환"""
    now_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    return {
        "system_version": "1.2.6",
        "build_time": now_str,
        "platform": "FastAPI/Python"
    }

# =============================================================================
# 1. Validation Endpoints (검증 관련)
# =============================================================================

@app.post("/validate", response_model=ValidationResponse)
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
        print("[Step 1] Reading employee data...")
        employee_content = await employee_file.read()
        
        # 숨겨진 시트 제외하고 로드
        from utils.excel_parser import get_visible_sheet_names
        visible_sheets = get_visible_sheet_names(employee_content)
        print(f"[Step 1] Visible sheets: {visible_sheets}")
        
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
        print("[Step 1.5] Filtering garbage rows...")
        for canonical_name, data in sheet_data_map.items():
            df = data["df"]
            
            # 사번/입사일 컬럼 식별 (부분 일치)
            id_keywords = ['사번', '사원번호', 'employee_id', 'id', '코드', 'code']
            date_keywords = ['입사일', '입사일자', 'hire_date']
            df_cols_lower = {str(col).lower(): col for col in df.columns}
            
            id_col = None
            for kw in id_keywords:
                for col_lower, original in df_cols_lower.items():
                    if kw in col_lower:
                        id_col = original
                        break
                if id_col: break

            date_col = None
            for kw in date_keywords:
                for col_lower, original in df_cols_lower.items():
                    if kw in col_lower:
                        date_col = original
                        break
                if date_col: break
            
            # 빈 값 체크 헬퍼
            def is_row_empty(series):
                import numpy as np
                return series.astype(str).str.strip().replace(['nan', 'None', 'NaT', ''], np.nan).isna()

            if id_col and date_col:
                mask = is_row_empty(df[id_col]) & is_row_empty(df[date_col])
                df = df[~mask]
            elif id_col or date_col:
                target = id_col or date_col
                mask = is_row_empty(df[target])
                df = df[~mask]
            else:
                valid_counts = df.apply(lambda x: (~is_row_empty(x)).sum(), axis=1)
                df = df[valid_counts >= 2]
            
            data["df"] = df

        # Step 2: Excel B 읽기 (자연어 규칙)
        print("[Step 2] Reading validation rules...")
        rules_content = await rules_file.read()
        natural_language_rules, sheet_row_counts, total_raw_rows, reported_max_row = parse_rules_from_excel(rules_content)

        all_rule_sheets_display_unfiltered = sorted(list(sheet_row_counts.keys()))
        all_rule_sheets_canonical_unfiltered = [get_canonical_name(name) for name in all_rule_sheets_display_unfiltered]

        # Step 3: AI 규칙 해석
        print(f"[Step 3] AI interpreting rules using {ai_provider}...")
        ai_response: AIInterpretationResponse = await ai_interpreter.interpret_rules(
            natural_language_rules,
            provider=ai_provider
        )

        # Step 4: 결정론적 검증 실행
        print("[Step 4] Running deterministic validation...")
        validation_res = await validation_service.validate_sheets(sheet_data_map, ai_response.rules)

        # Step 5: 응답 생성 및 메타데이터 추가
        from collections import Counter
        rule_counts_by_canonical = Counter(rule.source.sheet_name for rule in ai_response.rules)

        data_sheets_set = set(sheet_data_map.keys())
        rule_sheets_set = set(all_rule_sheets_canonical_unfiltered)
        
        matched_sheets_set = data_sheets_set.intersection(rule_sheets_set)
        unmatched_sheets_set = rule_sheets_set - data_sheets_set
        
        # 매칭 안 된 시트들의 Display Name 찾기
        unmatched_sheet_names = []
        for c_name in unmatched_sheets_set:
            d_name = next((d for d in all_rule_sheets_display_unfiltered if get_canonical_name(d) == c_name), c_name)
            unmatched_sheet_names.append(d_name)

        all_data_sheets = sorted(list(sheet_mapping_info.values()))

        display_list = []
        for d_name in all_rule_sheets_display_unfiltered:
            c_name = get_canonical_name(d_name)
            raw_rows_in_sheet = sheet_row_counts.get(d_name, 0)
            rule_count_in_sheet = rule_counts_by_canonical.get(c_name, 0)
            display_list.append(f"{d_name} ({raw_rows_in_sheet}행 / {rule_count_in_sheet}규칙)")

        matching_stats = {
            "total_rule_sheets": len(all_rule_sheets_canonical_unfiltered),
            "matched_sheets": len(matched_sheets_set),
            "unmatched_sheet_names": unmatched_sheet_names,
            "all_rule_sheets": display_list,
            "all_data_sheets": all_data_sheets,
            "total_raw_rows": total_raw_rows, 
            "reported_max_row": reported_max_row,
            "total_rules_count": len(ai_response.rules)
        }

        validation_res.conflicts = ai_response.conflicts

        # 실제 사용된 엔진 확인
        actual_model = "local-parser" if not ai_interpreter.use_cloud_ai else f"cloud-{ai_provider}"

        # --- Rule-specific Status Calculation ---
        from collections import Counter
        error_counts_by_rule = Counter(err.rule_id for err in validation_res.errors)
        
        # Create column order map for each sheet
        column_order_map = {}
        for c_name, data in sheet_data_map.items():
            column_order_map[c_name] = {col: idx for idx, col in enumerate(data['df'].columns)}

        rules_validation_status = []
        for rule in ai_response.rules:
            err_count = error_counts_by_rule.get(rule.rule_id, 0)
            status_msg = "검증 100% 완료!" if err_count == 0 else f"{err_count}건의 수정 필요사항 발견"
            
            # Frontend uses display_name (from sheet_data_map values) for tabs.
            # We must ensure rule.sheet_name matches that.
            rule_canonical = get_canonical_name(rule.source.sheet_name)
            display_sheet_name = sheet_data_map.get(rule_canonical, {}).get("display_name", rule.source.sheet_name)

            # Get column index for sorting
            col_idx = 9999
            if rule_canonical in column_order_map:
                col_idx = column_order_map[rule_canonical].get(rule.field_name, 9999)

            rules_validation_status.append({
                "rule_id": rule.rule_id,
                "field_name": rule.field_name,
                "rule_text": rule.source.original_text,
                "sheet_name": display_sheet_name,
                "error_count": err_count,
                "status_message": status_msg,
                "column_index": col_idx
            })
        
        # Sort by sheet name then column index
        rules_validation_status.sort(key=lambda x: (x['sheet_name'], x['column_index']))
            
        # --- AI Role Summary Generation ---
        ai_summary_text = (
            f"AI는 {len(ai_response.rules)}개의 자연어 규칙을 해석하여 {len(matched_sheets_set)}개의 시트에 자동 매핑했습니다. "
            f"총 {validation_res.summary.total_rows}행의 데이터를 검증하는 과정에서 "
            f"{len(ai_response.conflicts)}건의 규칙 충돌 가능성을 분석하고, "
            f"{validation_res.summary.total_errors}건의 데이터 오류를 식별했습니다."
        )

        validation_res.metadata.update({
            "employee_file_name": employee_file.filename,
            "rules_file_name": rules_file.filename,
            "ai_model_version": actual_model,
            "system_version": "1.0.0",
            "ai_processing_time_seconds": ai_response.processing_time_seconds,
            "total_errors": validation_res.summary.total_errors,
            "errors_shown": min(validation_res.summary.total_errors, 200),
            "error_groups_count": len(validation_res.error_groups),
            "matching_stats": matching_stats,
            "sheet_order": [data["display_name"] for data in sheet_data_map.values()],
            "rules_validation_status": rules_validation_status,
            "ai_role_summary": ai_summary_text
        })

        print("\n[OK] Response ready")
        return validation_res

    except Exception as e:
        print(f"\n[ERROR] Error: {str(e)}")
        print(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Validation failed",
                "message": str(e),
                "type": type(e).__name__
            }
        )


@app.post("/interpret-rules")
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
        raise HTTPException(
            status_code=500,
            detail={"error": "Rule interpretation failed", "message": str(e)}
        )


# Removed duplicate - see line 754



@app.get("/kifrs-references")
async def get_kifrs_references():
    """
    K-IFRS 1019 참조 정보 조회
    """
    from models import KIFRS_1019_REFERENCES
    return KIFRS_1019_REFERENCES


# =============================================================================
# Rule Management Endpoints (Phase 2)
# =============================================================================

@app.post("/rules/upload-to-db", response_model=RuleFileResponse)
async def upload_rule_file_to_db(
    rules_file: UploadFile = File(..., description="검증 규칙 파일 (Excel B)"),
    file_version: str = "1.0",
    uploaded_by: str = "system",
    notes: str = None
):
    """
    규칙 파일을 데이터베이스에 업로드

    Process:
    1. Excel B 파일 파싱
    2. rule_files 테이블에 메타데이터 저장
    3. rules 테이블에 개별 규칙 배치 저장
    4. 저장된 파일 정보 반환

    Args:
        rules_file: Excel 규칙 파일
        file_version: 파일 버전 (기본값: "1.0")
        uploaded_by: 업로드한 사용자 (기본값: "system")
        notes: 추가 메모

    Returns:
        RuleFileResponse: 저장된 규칙 파일 메타데이터
    """
    try:
        print(f"[API] Uploading rule file: {rules_file.filename}")

        # Read file content
        content = await rules_file.read()

        # Create metadata
        metadata = RuleFileUpload(
            file_name=rules_file.filename,
            file_version=file_version,
            uploaded_by=uploaded_by,
            notes=notes
        )

        # Upload using service
        response = await rule_service.upload_rule_file(content, metadata)

        print(f"[API] Successfully uploaded rule file: {response.id}")
        return response

    except Exception as e:
        print(f"[API] Error uploading rule file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to upload rule file",
                "message": str(e)
            }
        )


@app.get("/rules/files", response_model=List[RuleFileResponse])
async def list_rule_files(
    status: str = "active",
    limit: int = 50,
    offset: int = 0
):
    """
    저장된 규칙 파일 목록 조회

    Args:
        status: 필터링할 상태 (기본값: "active")
        limit: 최대 결과 수 (기본값: 50)
        offset: 페이지네이션 오프셋 (기본값: 0)

    Returns:
        List[RuleFileResponse]: 규칙 파일 목록
    """
    try:
        print(f"[API] Listing rule files (status={status}, limit={limit}, offset={offset})")
        files = await rule_service.list_rule_files(status, limit, offset)
        return files

    except Exception as e:
        print(f"[API] Error listing rule files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to list rule files",
                "message": str(e)
            }
        )


@app.get("/rules/files/{file_id}")
async def get_rule_file_details(file_id: str):
    """
    규칙 파일 상세 정보 조회

    Args:
        file_id: 규칙 파일 UUID

    Returns:
        Dict: 파일 메타데이터, 통계, 시트별 규칙 정보
    """
    try:
        print(f"[API] Getting rule file details: {file_id}")
        details = await rule_service.get_rule_file_details(file_id)

        if not details:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Rule file not found",
                    "file_id": file_id
                }
            )

        return details

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error getting rule file details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get rule file details",
                "message": str(e)
            }
        )


@app.get("/rules/files/{file_id}/mappings")
async def get_rule_mappings(file_id: str):
    """
    규칙 파일의 AI 매핑 현황 상세 조회

    원본 규칙과 AI 해석 결과를 비교하여 매핑 상태를 반환합니다.

    Args:
        file_id: 규칙 파일 UUID

    Returns:
        Dict: 매핑 통계 및 모든 규칙의 매핑 상세 정보
    """
    try:
        print(f"[API] Getting rule mappings for file: {file_id}")
        mappings = await rule_service.get_rule_mappings(file_id)

        if not mappings:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Rule file not found",
                    "file_id": file_id
                }
            )

        return mappings

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error getting rule mappings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get rule mappings",
                "message": str(e)
            }
        )


@app.put("/rules/{rule_id}/mapping")
async def update_rule_mapping(rule_id: str, mapping_data: dict):
    """
    개별 규칙의 원본 정보 및 AI 매핑 수동 설정

    사용자가 원본 규칙 정보(시트명, 필드명, 규칙 원문 등)와
    AI 해석을 수동으로 설정하거나 수정할 수 있습니다.

    Args:
        rule_id: 규칙 UUID
        mapping_data: 규칙 및 AI 매핑 데이터
            {
                // 원본 규칙 정보 (optional)
                "sheet_name": str,
                "field_name": str,
                "rule_text": str,
                "row_number": int,
                "column_letter": str,
                // AI 매핑 데이터 (optional)
                "ai_rule_type": str,
                "ai_parameters": dict,
                "ai_error_message": str,
                "ai_confidence_score": float
            }

    Returns:
        Dict: 업데이트 결과
    """
    try:
        print(f"[API] Updating rule mapping: {rule_id}")

        # AI 설정이 포함된 경우에만 수동 설정 처리
        if "ai_rule_type" in mapping_data:
            # 수동 설정임을 명시
            mapping_data["ai_model_version"] = "manual"
            mapping_data["ai_interpreted_at"] = datetime.now().isoformat()
            mapping_data["ai_interpretation_summary"] = mapping_data.get("ai_interpretation_summary", "사용자 수동 설정")

            # 신뢰도가 없으면 1.0 (수동 설정은 100% 신뢰)
            if "ai_confidence_score" not in mapping_data:
                mapping_data["ai_confidence_score"] = 1.0

            # ai_rule_id 생성 (없으면)
            if not mapping_data.get("ai_rule_id"):
                import uuid
                mapping_data["ai_rule_id"] = f"RULE_MANUAL_{str(uuid.uuid4())[:8].upper()}"

        success = await rule_service.update_rule(rule_id, mapping_data)

        if not success:
            raise HTTPException(
                status_code=404,
                detail={"error": "Rule not found or update failed"}
            )

        # 학습: 사용자가 직접 확정한 패턴을 학습 시스템에 저장
        # AI 설정이 포함되어 있고, 원본 텍스트/필드명이 있는 경우 학습
        try:
            # 필요한 정보가 mapping_data에 없으면 DB에서 조회
            rule_text = mapping_data.get("rule_text")
            field_name = mapping_data.get("field_name")
            
            if not rule_text or not field_name:
                rule = await rule_service.get_rule(rule_id)
                if rule:
                    rule_text = rule_text or rule.get("rule_text")
                    field_name = field_name or rule.get("field_name")

            if "ai_rule_type" in mapping_data and rule_text and field_name:
                # 비동기로 학습 데이터 저장 (사용자 응답 지연 방지 위해 await 사용 최소화 가능하나, 
                # 여기서는 데이터 무결성을 위해 await 사용)
                await learning_service.save_learned_pattern(
                    rule_text=rule_text,
                    field_name=field_name,
                    ai_rule_type=mapping_data["ai_rule_type"],
                    ai_parameters=mapping_data.get("ai_parameters", {}),
                    ai_error_message=mapping_data.get("ai_error_message", ""),
                    source_rule_id=rule_id,
                    confidence_boost=0.1  # 사용자 확정은 신뢰도 부스트
                )
                print(f"[Learning] Learned pattern from rule: {rule_id}")
        except Exception as e:
            # 학습 실패가 API 응답을 막으면 안 됨
            print(f"[Learning] Failed to learn pattern: {e}")

        return {
            "status": "success",
            "message": "규칙이 성공적으로 업데이트되었습니다.",
            "rule_id": rule_id
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error updating rule mapping: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to update rule mapping",
                "message": str(e)
            }
        )


@app.post("/rules/{rule_id}/reinterpret")
async def reinterpret_single_rule(rule_id: str, use_local_parser: bool = True):
    """
    개별 규칙의 rule_text를 기반으로 AI 재해석 수행

    Args:
        rule_id: 규칙 UUID
        use_local_parser: True면 로컬 파서, False면 Cloud AI 사용

    Returns:
        Dict: 새로운 AI 해석 결과
    """
    try:
        print(f"[API] Reinterpreting single rule: {rule_id}")

        # 규칙 정보 조회
        rule = await rule_service.get_rule(rule_id)
        if not rule:
            raise HTTPException(
                status_code=404,
                detail={"error": "Rule not found"}
            )

        # AI 재해석 수행 (Smart Interpret: 학습된 패턴 우선)
        interpretation, source = await learning_service.smart_interpret(
            rule_text=rule.get("rule_text", ""),
            field_name=rule.get("field_name", ""),
            ai_interpreter=ai_interpreter,
            use_learning=True  # 학습된 패턴 활용
        )
        
        # 로컬 파서 강제 사용 시 덮어쓰기 (단, 학습된 패턴이 정확히 매칭된 경우는 유지 가능하나, 
        # 여기서는 use_local_parser 요청이 오면 보통 로컬 로직을 테스트하려는 의도이므로 
        # 학습 패턴이 'ai' 소스인 경우에만 로컬 파서 로직이 적용되도록 ai_interpreter 내부에서 처리됨)
        # 하지만 smart_interpret은 interpreter.interpret_rule을 호출하므로, 
        # interpreter의 설정을 바꿔야 할 수도 있음.
        # 현재 smart_interpret 구현상 ai_interpreter를 그대로 쓰므로,
        # use_local_parser=True일 경우 AI 호출을 안 하게 됨.

        # 해석 결과 업데이트
        update_data = {
            "ai_rule_type": interpretation.get("rule_type"),
            "ai_rule_id": interpretation.get("rule_id"),
            "ai_parameters": interpretation.get("parameters", {}),
            "ai_error_message": interpretation.get("error_message", ""),
            "ai_confidence_score": interpretation.get("confidence_score", 0.8),
            "ai_interpretation_summary": interpretation.get("interpretation_summary", "") + f" (Source: {source})",
            "ai_model_version": "local-parser" if use_local_parser else interpretation.get("model_version", "unknown"),
            "ai_interpreted_at": datetime.now().isoformat()
        }

        success = await rule_service.update_rule(rule_id, update_data)
        if not success:
            raise HTTPException(
                status_code=500,
                detail={"error": "Failed to save interpretation"}
            )

        return {
            "status": "success",
            "rule_id": rule_id,
            "ai_rule_type": update_data["ai_rule_type"],
            "ai_parameters": update_data["ai_parameters"],
            "ai_error_message": update_data["ai_error_message"],
            "ai_confidence_score": update_data["ai_confidence_score"],
            "ai_interpretation_summary": update_data["ai_interpretation_summary"]
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error reinterpreting rule: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to reinterpret rule",
                "message": str(e)
            }
        )


@app.get("/rules/download/{file_id}")
async def download_rule_file(file_id: str):
    """
    데이터베이스에서 규칙을 Excel 파일로 다운로드

    Args:
        file_id: 규칙 파일 UUID

    Returns:
        Excel 파일 (StreamingResponse)
    """
    try:
        print(f"[API] Downloading rule file: {file_id}")

        # Export rules to Excel
        excel_bytes = await rule_service.export_rules_to_excel(file_id)
        print(f"[API] Excel generated: {len(excel_bytes)} bytes")

        # Get file metadata for filename
        try:
            details = await rule_service.get_rule_file_details(file_id)
            original_filename = details['file_name'] if details else 'rules.xlsx'
        except Exception as e:
            print(f"[API] Warning: Could not get file details, using default filename: {e}")
            original_filename = 'rules.xlsx'

        # Remove extension if exists
        base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename

        # Create download filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_exported_{timestamp}.xlsx"

        print(f"[API] Sending file: {filename}")

        # Create response with proper headers
        return StreamingResponse(
            io.BytesIO(excel_bytes),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Content-Length": str(len(excel_bytes)),
                "Cache-Control": "no-cache"
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error downloading rule file: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to download rule file",
                "message": str(e),
                "file_id": file_id
            }
        )


@app.delete("/rules/files/{file_id}")
async def archive_rule_file(file_id: str):
    """
    규칙 파일 아카이브 (소프트 삭제)

    Args:
        file_id: 규칙 파일 UUID

    Returns:
        Dict: 삭제 결과
    """
    try:
        print(f"[API] Archiving rule file: {file_id}")

        # Use rule_service to archive the file
        success = await rule_service.archive_rule_file(file_id)

        if not success:
            raise HTTPException(
                status_code=404,
                detail={
                    "error": "Rule file not found or could not be archived",
                    "file_id": file_id
                }
            )

        return {
            "status": "success",
            "message": "규칙 파일이 성공적으로 삭제되었습니다.",
            "file_id": file_id
        }

    except HTTPException:
        raise
    except Exception as e:
        print(f"[API] Error archiving rule file: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to archive rule file",
                "message": str(e),
                "file_id": file_id
            }
        )


@app.post("/rules/interpret/{file_id}")
async def interpret_rules(
    file_id: str,
    force_reinterpret: bool = False,
    use_local_parser: bool = False
):
    """
    규칙 파일의 AI 해석 실행 또는 재해석

    Args:
        file_id: 규칙 파일 UUID
        force_reinterpret: True면 기존 해석 무시하고 재해석
        use_local_parser: True면 로컬 파서만 사용 (AI 오류 방지)

    Returns:
        Dict: 해석 결과 통계
    """
    try:
        print(f"[API] Starting interpretation for file: {file_id} (force={force_reinterpret}, local={use_local_parser})")

        result = await ai_cache_service.interpret_and_cache_rules(
            file_id,
            force_reinterpret,
            force_local=use_local_parser
        )

        print(f"[API] Interpretation completed")
        return {
            "status": "success",
            "file_id": file_id,
            **result
        }

    except Exception as e:
        print(f"[API] Error during interpretation: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to interpret rules",
                "message": str(e),
                "file_id": file_id
            }
        )


@app.post("/rules/reinterpret/{file_id}")
async def reinterpret_rules_from_original(
    file_id: str,
    use_local_parser: bool = True
):
    """
    저장된 원본 파일로 규칙 재해석

    기존 AI 해석을 모두 초기화하고 원본 파일로 재해석합니다.
    use_local_parser=True (기본값)이면 로컬 파서를 사용하여 AI 오류를 방지합니다.

    Args:
        file_id: 규칙 파일 UUID
        use_local_parser: True면 로컬 파서만 사용 (권장)

    Returns:
        Dict: 재해석 결과 통계
    """
    try:
        print(f"[API] Starting re-interpretation for file: {file_id} (local={use_local_parser})")

        result = await rule_service.reinterpret_rules(file_id, use_local_parser)

        print(f"[API] Re-interpretation completed")
        return {
            "status": "success",
            "file_id": file_id,
            "message": "규칙이 성공적으로 재해석되었습니다.",
            **result
        }

    except Exception as e:
        print(f"[API] Error during re-interpretation: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to re-interpret rules",
                "message": str(e),
                "file_id": file_id
            }
        )


@app.post("/rules/", status_code=201)
async def create_rule(rule: RuleCreate):
    """
    개별 규칙 수동 생성
    """
    try:
        result = await rule_service.create_single_rule(rule)
        return {"status": "success", "message": "Rule created successfully", "id": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/rules/{rule_id}", response_model=RuleDetail)
async def get_rule_detail(rule_id: str):
    """
    개별 규칙 상세 정보 조회
    """
    try:
        rule = await rule_service.get_rule(rule_id)
        if not rule:
            raise HTTPException(status_code=404, detail="Rule not found")
        return rule
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.put("/rules/{rule_id}")
async def update_rule(rule_id: str, updates: RuleUpdate):
    """
    개별 규칙 수정
    """
    try:
        success = await rule_service.update_rule(rule_id, updates.dict(exclude_unset=True))
        if not success:
            raise HTTPException(status_code=404, detail="Rule not found or no changes made")
        return {"status": "success", "message": "Rule updated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/rules/{rule_id}")
async def delete_rule(rule_id: str, permanent: bool = False):
    """
    개별 규칙 삭제 (기본값: 비활성화)
    """
    try:
        success = await rule_service.delete_rule(rule_id, permanent)
        if not success:
            raise HTTPException(status_code=404, detail="Rule not found")
        return {"status": "success", "message": "Rule deleted/deactivated successfully"}
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Validation Session Endpoints (Phase 3)
# =============================================================================

@app.post("/validate-with-db-rules")
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
        print(f"[API] Validating with DB rules: {rule_file_id}")
        
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
                from collections import Counter
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

                print(f"[Learning] Recorded validation feedback for session: {session_id}")

        except Exception as e:
            print(f"[Learning] Failed to record feedback: {e}")
            # 피드백 실패는 무시 (메인 로직에 영향 주지 않음)

        return result

    except ValueError as ve:
        print(f"[API] Validation error: {str(ve)}")
        raise HTTPException(
            status_code=400,
            detail={
                "error": "Validation failed",
                "message": str(ve)
            }
        )
    except Exception as e:
        print(f"[API] Unexpected error: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Validation failed",
                "message": str(e)
            }
        )


@app.get("/sessions")
async def list_validation_sessions(
    limit: int = 50,
    offset: int = 0
):
    """
    검증 세션 목록 조회
    """
    try:
        sessions = await validation_service.list_sessions(limit, offset)
        return sessions
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)}
        )


@app.get("/sessions/{session_id}")
async def get_session_details(session_id: str):
    """
    세션 상세 정보 및 에러 목록 조회
    """
    try:
        details = await validation_service.get_session_details(session_id)
        if not details:
            raise HTTPException(status_code=404, detail="Session not found")
        return details
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)}
        )


@app.post("/feedback/false-positive")
async def submit_false_positive_feedback(feedback: FalsePositiveFeedback):
    """
    False Positive 피드백 제출
    """
    try:
        result = await feedback_service.submit_false_positive_feedback(feedback)
        return {"status": "success", "data": result}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)}
        )


@app.get("/statistics/dashboard")
async def get_dashboard_statistics():
    """
    대시보드 통계 조회
    """
    try:
        return await statistics_service.get_dashboard_statistics()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": str(e)}
        )


@app.post("/errors/explain")
async def explain_error(
    error: ValidationError,
    ai_provider: str = Form("openai", description="AI Provider (openai, anthropic, gemini)")
):
    """
    단일 검증 오류에 대한 AI의 상세 설명 및 조치 권고를 받습니다.
    """
    try:
        explanation = await ai_interpreter.get_error_explanation(error, provider=ai_provider)
        return explanation
    except Exception as e:
        print(f"[API] Error getting explanation: {str(e)}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get error explanation",
                "message": str(e)
            }
        )


# =============================================================================
# Phase 5: Smart Fix Endpoints
# =============================================================================

class FixSuggestRequest(BaseModel):
    session_id: str
    error_ids: Optional[List[str]] = None
    ai_provider: str = "openai"

@app.post("/fix/suggest", response_model=List[FixSuggestion])
async def suggest_fixes(request: FixSuggestRequest):
    """
    오류에 대한 AI 수정 제안 생성
    """
    try:
        print(f"[API] Suggest fixes for session: {request.session_id}, errors: {len(request.error_ids) if request.error_ids else 'all'}")
        suggestions = await fix_service.suggest_fixes(
            request.session_id,
            request.error_ids,
            provider=request.ai_provider
        )
        print(f"[API] Generated {len(suggestions)} fix suggestions")
        return suggestions
    except Exception as e:
        print(f"[API] Suggest fixes failed: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to generate fix suggestions", "message": str(e)}
        )

@app.post("/fix/apply")
async def apply_fixes(
    fix_request_json: str = Form(..., description="BatchFixRequest JSON string"),
    original_file: UploadFile = File(..., description="Original Excel file")
):
    """
    수정 사항을 적용하여 엑셀 파일 다운로드
    """
    try:
        # Parse JSON payload
        import json
        request_data = json.loads(fix_request_json)
        # Validate with Pydantic
        fix_request = BatchFixRequest(**request_data)
        
        print(f"[API] Applying {len(fix_request.fixes)} fixes to file: {original_file.filename}")
        
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
        print(f"[API] Apply fixes failed: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to apply fixes", "message": str(e)}
        )


@app.post("/api/fix/download")
async def bulk_fix_download(
    cells_to_fix_json: str = Form(..., description="JSON array of cells to fix"),
    original_file: UploadFile = File(..., description="Original Excel file")
):
    """
    오류 항목 일괄 수정 후 엑셀 파일 다운로드

    - cells_to_fix: [{sheet, row, column, currentValue, fixType}, ...]
    - original_file: 원본 엑셀 파일
    """
    try:
        import json
        cells_to_fix = json.loads(cells_to_fix_json)

        print(f"[API] Bulk fix download: {len(cells_to_fix)} cells from {original_file.filename}")

        # Read original file
        content = await original_file.read()

        # Apply bulk fixes (filename 전달하여 xls/xlsx 구분)
        modified_excel = fix_service.apply_bulk_fixes_to_excel(
            content,
            cells_to_fix,
            filename=original_file.filename or ""
        )

        # Generate filename (한글 인코딩 처리)
        from urllib.parse import quote
        base_name = original_file.filename.rsplit('.', 1)[0] if original_file.filename else "data"
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_fixed_{timestamp}.xlsx"
        filename_encoded = quote(filename, safe='')

        return StreamingResponse(
            io.BytesIO(modified_excel),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename*=UTF-8''{filename_encoded}",
                "Content-Length": str(len(modified_excel)),
                "Cache-Control": "no-cache"
            }
        )

    except Exception as e:
        print(f"[API] Bulk fix download failed: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to create fixed file", "message": str(e)}
        )


def _safe_str(value, max_length=32000):
    """엑셀 안전 문자열 변환 (셀 크기 제한 고려)"""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    s = str(value)
    if len(s) > max_length:
        return s[:max_length] + "..."
    return s

@app.post("/download-results")
async def download_validation_results(validation_response: ValidationResponse):
    """
    검증 결과를 Excel 파일로 다운로드

    Args:
        validation_response: 검증 결과 데이터

    Returns:
        Excel 파일 (StreamingResponse)
    """
    try:
        output = io.BytesIO()

        with pd.ExcelWriter(output, engine='openpyxl') as writer:
            # 1. 요약 시트
            summary_data = {
                "항목": ["검증 상태", "전체 행 수", "정상 행 수", "오류 행 수", "총 오류 수", "적용된 규칙 수", "검증 시각"],
                "값": [
                    validation_response.validation_status,
                    validation_response.summary.total_rows,
                    validation_response.summary.valid_rows,
                    validation_response.summary.error_rows,
                    validation_response.summary.total_errors,
                    validation_response.summary.rules_applied,
                    validation_response.summary.timestamp.strftime("%Y-%m-%d %H:%M:%S")
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name=sanitize_sheet_name("검증 요약"), index=False)

            # 2. 인지 항목 집계 시트
            if validation_response.error_groups:
                groups_data = []
                for group in validation_response.error_groups:
                    rows_str = ', '.join(map(str, group.affected_rows[:20]))
                    if len(group.affected_rows) > 20:
                        rows_str += f" 외 {len(group.affected_rows) - 20}개"

                    groups_data.append({
                        "시트명": _safe_str(group.sheet),
                        "열": _safe_str(group.column),
                        "규칙ID": _safe_str(group.rule_id),
                        "인지 메시지": _safe_str(group.message, 1000),
                        "인지 횟수": group.count,
                        "영향받은 행": _safe_str(rows_str, 500),
                        "샘플 값": _safe_str(", ".join(map(str, group.sample_values[:10])), 500),
                        "예상 값": _safe_str(group.expected),
                        "원본 규칙": _safe_str(group.source_rule, 500)
                    })
                df_groups = pd.DataFrame(groups_data)
                df_groups.to_excel(writer, sheet_name=sanitize_sheet_name("인지 항목 집계"), index=False)

            # 3. 개별 인지 목록 시트
            if validation_response.errors:
                errors_data = []
                for error in validation_response.errors:
                    errors_data.append({
                        "시트명": _safe_str(error.sheet),
                        "행": error.row,
                        "열": _safe_str(error.column),
                        "규칙ID": _safe_str(error.rule_id),
                        "인지 메시지": _safe_str(error.message, 1000),
                        "실제 값": _safe_str(error.actual_value, 500),
                        "예상 값": _safe_str(error.expected),
                        "원본 규칙": _safe_str(error.source_rule, 500)
                    })
                df_errors = pd.DataFrame(errors_data)
                df_errors.to_excel(writer, sheet_name=sanitize_sheet_name("개별 인지 목록"), index=False)

            # 4. 규칙 충돌 시트
            if validation_response.conflicts:
                conflicts_data = []
                for conflict in validation_response.conflicts:
                    conflicts_data.append({
                        "규칙ID": _safe_str(conflict.rule_id),
                        "충돌 유형": _safe_str(conflict.conflict_type),
                        "설명": _safe_str(conflict.description, 1000),
                        "K-IFRS 1019 참조": _safe_str(conflict.kifrs_reference),
                        "영향받는 규칙": _safe_str(", ".join(conflict.affected_rules), 500),
                        "권장사항": _safe_str(conflict.recommendation, 1000),
                        "심각도": _safe_str(conflict.severity)
                    })
                df_conflicts = pd.DataFrame(conflicts_data)
                df_conflicts.to_excel(writer, sheet_name=sanitize_sheet_name("규칙 충돌"), index=False)

            # 5. 적용된 규칙 시트
            if validation_response.rules_applied:
                rules_data = []
                for rule in validation_response.rules_applied:
                    rules_data.append({
                        "규칙ID": _safe_str(rule.rule_id),
                        "시트명": _safe_str(rule.source.sheet_name),
                        "필드명": _safe_str(rule.field_name),
                        "규칙 유형": _safe_str(rule.rule_type),
                        "파라미터": _safe_str(rule.parameters, 500),
                        "오류 메시지": _safe_str(rule.error_message_template, 500),
                        "원본 규칙": _safe_str(rule.source.original_text, 500),
                        "신뢰도": rule.confidence_score
                    })
                df_rules = pd.DataFrame(rules_data)
                df_rules.to_excel(writer, sheet_name=sanitize_sheet_name("적용된 규칙"), index=False)

        # 파일명 생성
        timestamp = datetime.now().strftime("%Y-%m-%d")
        filename = f"DBO_Validation_Result_{timestamp}.xlsx"

        # CRITICAL: output을 다시 처음으로 이동
        output.seek(0)

        # StreamingResponse는 file-like object를 직접 받아야 함
        return StreamingResponse(
            output,
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f'attachment; filename="{filename}"',
                "Cache-Control": "no-cache"
            }
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to generate Excel file",
                "message": str(e)
            }
        )


# =============================================================================
# Phase 9: Learning System Endpoints
# =============================================================================

@app.get("/learning/statistics")
async def get_learning_statistics():
    """
    학습 시스템 통계 조회
    """
    try:
        return await learning_service.get_learning_statistics()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to get learning statistics", "message": str(e)}
        )

@app.get("/learning/patterns/{pattern_id}/effectiveness")
async def get_pattern_effectiveness(pattern_id: str):
    """
    특정 학습 패턴의 효과성 분석
    """
    try:
        result = await learning_service.get_pattern_effectiveness(pattern_id)
        if "error" in result:
             raise HTTPException(status_code=404, detail=result["error"])
        return result
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to get pattern effectiveness", "message": str(e)}
        )


@app.post("/learning/maintenance")
async def run_learning_maintenance():
    """
    학습 시스템 유지보수 실행

    - 저신뢰 패턴 자동 비활성화 (성공률 < 60%, 피드백 5회 이상)
    - 고성공률 패턴 확정 (성공률 >= 98%, 연속 성공 10회 이상)

    Returns:
        Dict: 유지보수 결과 통계
    """
    try:
        result = await learning_service.run_maintenance()
        return {
            "status": "success",
            "message": "학습 시스템 유지보수가 완료되었습니다.",
            **result
        }
    except Exception as e:
        print(f"[API] Learning maintenance error: {e}")
        traceback.print_exc()
        raise HTTPException(
            status_code=500,
            detail={"error": "Failed to run learning maintenance", "message": str(e)}
        )


# =============================================================================
# 예외 핸들러
# =============================================================================

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """
    전역 예외 핸들러
    """
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "type": type(exc).__name__
        }
    )


# =============================================================================
# 서버 실행 (개발용)
# =============================================================================

if __name__ == "__main__":
    import uvicorn

    print("""
    =================================================================
      K-IFRS 1019 DBO Validation System
      AI-Powered Data Validation for Defined Benefit Obligations
    =================================================================

    Starting server...
    Mobile-optimized UI available at: http://localhost:8000
    API Documentation: http://localhost:8000/docs

    """)
    
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )