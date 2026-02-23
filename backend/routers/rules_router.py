"""
Rules Router - 규칙 파일 관리 및 AI 해석
========================================
규칙 업로드, 조회, 수정, 삭제, AI 해석/재해석, 다운로드, 충돌 감지 등
"""

import io
import uuid
import traceback
from datetime import datetime
from typing import List, Dict, Any, Optional
from uuid import UUID

from fastapi import APIRouter, File, UploadFile, HTTPException, Form
from fastapi.responses import StreamingResponse

from dependencies import rule_service, ai_cache_service, ai_interpreter, learning_service
from models import RuleFileUpload, RuleFileResponse, RuleUpdate, RuleCreate, RuleDetail
from utils.excel_parser import _should_split_rule
from utils.logger import get_logger

logger = get_logger("rules_router")
router = APIRouter(tags=["Rules"])


# =============================================================================
# 1. 규칙 파일 업로드
# =============================================================================

@router.post("/rules/upload-to-db", response_model=RuleFileResponse)
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
        logger.info(f"Uploading rule file: {rules_file.filename}")

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

        logger.info(f"Successfully uploaded rule file: {response.id}")
        return response

    except Exception as e:
        logger.error(f"Error uploading rule file: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to upload rule file",
                "message": str(e)
            }
        )


# =============================================================================
# 2. 규칙 파일 목록 조회
# =============================================================================

@router.get("/rules/files", response_model=List[RuleFileResponse])
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
        logger.info(f"Listing rule files (status={status}, limit={limit}, offset={offset})")
        files = await rule_service.list_rule_files(status, limit, offset)
        return files

    except Exception as e:
        logger.error(f"Error listing rule files: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to list rule files",
                "message": str(e)
            }
        )


# =============================================================================
# 3. 규칙 파일 상세 정보 조회
# =============================================================================

@router.get("/rules/files/{file_id}")
async def get_rule_file_details(file_id: str):
    """
    규칙 파일 상세 정보 조회

    Args:
        file_id: 규칙 파일 UUID

    Returns:
        Dict: 파일 메타데이터, 통계, 시트별 규칙 정보
    """
    try:
        logger.info(f"Getting rule file details: {file_id}")
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
        logger.error(f"Error getting rule file details: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get rule file details",
                "message": str(e)
            }
        )


# =============================================================================
# 4. 규칙 파일의 AI 매핑 현황 조회
# =============================================================================

@router.get("/rules/files/{file_id}/mappings")
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
        logger.info(f"Getting rule mappings for file: {file_id}")
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
        logger.error(f"Error getting rule mappings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to get rule mappings",
                "message": str(e)
            }
        )


# =============================================================================
# 5. 개별 규칙 매핑 수동 설정
# =============================================================================

@router.put("/rules/{rule_id}/mapping")
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
                "row_number": str,  # 서브 인덱스 지원: "5", "5.1", "5.2"
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
        logger.info(f"Updating rule mapping: {rule_id}")

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
                logger.info(f"Learned pattern from rule: {rule_id}")
        except Exception as e:
            # 학습 실패가 API 응답을 막으면 안 됨
            logger.warning(f"Failed to learn pattern: {e}")

        return {
            "status": "success",
            "message": "규칙이 성공적으로 업데이트되었습니다.",
            "rule_id": rule_id
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error updating rule mapping: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to update rule mapping",
                "message": str(e)
            }
        )


# =============================================================================
# 6. 개별 규칙 AI 재해석
# =============================================================================

@router.post("/rules/{rule_id}/reinterpret")
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
        logger.info(f"Reinterpreting single rule: {rule_id}")

        # 규칙 정보 조회
        rule = await rule_service.get_rule(rule_id)
        if not rule:
            raise HTTPException(
                status_code=404,
                detail={"error": "Rule not found"}
            )

        rule_text = rule.get("rule_text", "")

        # 복합 규칙 감지 → 자동 분리 후 각각 해석
        if _should_split_rule(rule_text):
            logger.info(f"Composite rule detected, splitting: {rule_text}")
            split_result = await rule_service.split_rule(rule_id)
            return {
                "status": "split",
                "rule_id": rule_id,
                "message": f"복합 규칙이 {split_result['created_count']}개로 분리되었습니다.",
                "created_count": split_result["created_count"],
                "created_rules": split_result["created_rules"],
            }

        # 단일 규칙: AI 재해석 수행
        interpretation, source = await learning_service.smart_interpret(
            rule_text=rule_text,
            field_name=rule.get("field_name", ""),
            ai_interpreter=ai_interpreter,
            use_learning=False
        )

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
        logger.error(f"Error reinterpreting rule: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to reinterpret rule",
                "message": str(e)
            }
        )


# =============================================================================
# 7. 규칙 파일 Excel 다운로드
# =============================================================================

@router.get("/rules/download/{file_id}")
async def download_rule_file(file_id: str):
    """
    데이터베이스에서 규칙을 Excel 파일로 다운로드

    Args:
        file_id: 규칙 파일 UUID

    Returns:
        Excel 파일 (StreamingResponse)
    """
    try:
        logger.info(f"Downloading rule file: {file_id}")

        # Export rules to Excel
        excel_bytes = await rule_service.export_rules_to_excel(file_id)
        logger.info(f"Excel generated: {len(excel_bytes)} bytes")

        # Get file metadata for filename
        try:
            details = await rule_service.get_rule_file_details(file_id)
            original_filename = details['file_name'] if details else 'rules.xlsx'
        except Exception as e:
            logger.warning(f"Could not get file details, using default filename: {e}")
            original_filename = 'rules.xlsx'

        # Remove extension if exists
        base_name = original_filename.rsplit('.', 1)[0] if '.' in original_filename else original_filename

        # Create download filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{base_name}_exported_{timestamp}.xlsx"

        logger.info(f"Sending file: {filename}")

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
        logger.error(f"Error downloading rule file: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to download rule file",
                "message": str(e),
                "file_id": file_id
            }
        )


# =============================================================================
# 8. 규칙 파일 아카이브 (소프트 삭제)
# =============================================================================

@router.delete("/rules/files/{file_id}")
async def archive_rule_file(file_id: str):
    """
    규칙 파일 아카이브 (소프트 삭제)

    Args:
        file_id: 규칙 파일 UUID

    Returns:
        Dict: 삭제 결과
    """
    try:
        logger.info(f"Archiving rule file: {file_id}")

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
        logger.error(f"Error archiving rule file: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to archive rule file",
                "message": str(e),
                "file_id": file_id
            }
        )


# =============================================================================
# 9. 규칙 파일 AI 해석 실행
# =============================================================================

@router.post("/rules/interpret/{file_id}")
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
        logger.info(f"Starting interpretation for file: {file_id} (force={force_reinterpret}, local={use_local_parser})")

        result = await ai_cache_service.interpret_and_cache_rules(
            file_id,
            force_reinterpret,
            force_local=use_local_parser
        )

        logger.info("Interpretation completed")
        return {
            "status": "success",
            "file_id": file_id,
            **result
        }

    except Exception as e:
        logger.error(f"Error during interpretation: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to interpret rules",
                "message": str(e),
                "file_id": file_id
            }
        )


# =============================================================================
# 10. 저장된 원본 파일로 규칙 재해석
# =============================================================================

@router.post("/rules/reinterpret/{file_id}")
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
        logger.info(f"Starting re-interpretation for file: {file_id} (local={use_local_parser})")

        result = await rule_service.reinterpret_rules(file_id, use_local_parser)

        logger.info("Re-interpretation completed")
        return {
            "status": "success",
            "file_id": file_id,
            "message": "규칙이 성공적으로 재해석되었습니다.",
            **result
        }

    except Exception as e:
        logger.error(f"Error during re-interpretation: {str(e)}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={
                "error": "Failed to re-interpret rules",
                "message": str(e),
                "file_id": file_id
            }
        )


# =============================================================================
# 11. 개별 규칙 수동 생성
# =============================================================================

@router.post("/rules/", status_code=201)
async def create_rule(rule: RuleCreate):
    """
    개별 규칙 수동 생성
    """
    try:
        result = await rule_service.create_single_rule(rule)
        return {"status": "success", "message": "Rule created successfully", "id": result}
    except Exception as e:
        logger.error(f"Error creating rule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 12. 개별 규칙 상세 정보 조회
# =============================================================================

@router.get("/rules/{rule_id}", response_model=RuleDetail)
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
        logger.error(f"Error getting rule detail: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 13. 개별 규칙 수정
# =============================================================================

@router.put("/rules/{rule_id}")
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
        logger.error(f"Error updating rule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 14. 개별 규칙 삭제
# =============================================================================

@router.delete("/rules/{rule_id}")
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
        logger.error(f"Error deleting rule: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# 15. 복합 규칙 분리
# =============================================================================

@router.post("/rules/{rule_id}/split")
async def split_rule(rule_id: str):
    """
    복합 규칙을 개별 규칙으로 분리

    예: "공백, 중복" → "공백" + "중복" 2개의 독립 규칙 생성, 원본 비활성화

    Args:
        rule_id: 분리할 규칙 UUID

    Returns:
        Dict: 분리 결과 (생성된 규칙 개수 및 목록)
    """
    try:
        logger.info(f"Splitting composite rule: {rule_id}")
        result = await rule_service.split_rule(rule_id)
        return {
            "status": "success",
            "message": f"규칙이 {result['created_count']}개로 분리되었습니다.",
            "created_count": result["created_count"],
            "created_rules": result["created_rules"],
        }
    except Exception as e:
        logger.error(f"Error splitting rule: {str(e)}")
        raise HTTPException(
            status_code=400,
            detail={"error": "Failed to split rule", "message": str(e)}
        )


# =============================================================================
# 16. 규칙 충돌 감지
# =============================================================================

@router.post("/api/rules/check-conflicts")
async def check_rule_conflicts(rule_file_id: str):
    """
    규칙 충돌 감지 - 서로 모순되는 규칙을 자동 탐지합니다.
    """
    try:
        rules = await rule_service.repository.get_rules_by_file(UUID(rule_file_id), active_only=True)
        result = ai_interpreter.detect_rule_conflicts(rules)
        return result

    except Exception as e:
        logger.error(f"Rule conflict check error: {e}")
        logger.debug(traceback.format_exc())
        raise HTTPException(
            status_code=500,
            detail={"error": "Rule conflict check failed", "message": str(e)}
        )
