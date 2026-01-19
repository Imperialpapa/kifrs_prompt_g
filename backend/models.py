"""
K-IFRS 1019 DBO Validation System - Data Models
================================================
감사 대응을 위한 엄격한 데이터 모델 정의

모든 AI 판단은 추적 가능하며, 결정론적 검증과 명확히 분리됨
"""

from typing import List, Dict, Any, Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


# =============================================================================
# Input Data Models
# =============================================================================

class EmployeeDataRow(BaseModel):
    """
    K-IFRS 검증 대상이 되는 직원 데이터 한 행의 구조화된 모델
    - Excel 파일의 한 행에 해당하며, 파싱 후 초기 데이터를 담음
    - 각 필드는 Optional로 정의하여, 유효성 검사 단계에서 누락/형식 오류를 잡아내도록 설계
    """
    row_number: int = Field(..., description="원본 Excel 파일의 행 번호")
    employee_code: Optional[str] = Field(None, description="사원코드")
    employee_name: Optional[str] = Field(None, description="사원명")
    birth_date: Optional[str] = Field(None, description="생년월일 (YYYYMMDD)")
    hire_date: Optional[str] = Field(None, description="입사일")
    termination_date: Optional[str] = Field(None, description="퇴사일 (재직 시 빈 값)")
    evaluation_date: Optional[str] = Field(None, description="평가기준일")
    average_wage: Optional[float] = Field(None, description="평균임금 (최근 3개월)")
    plan_type: Optional[str] = Field(None, description="제도유형 (DB/DC/퇴직금 등)")
    payment_rate: Optional[float] = Field(None, description="지급배수/지급률")
    first_hire_date_affiliated: Optional[str] = Field(None, description="관계사 전입 최초 입사일")
    
    # 추가적으로 원본 데이터를 그대로 저장할 필드
    original_data: Dict[str, Any] = Field(default_factory=dict, description="파싱된 원본 행 데이터")


class RuleSource(BaseModel):
    """
    규칙의 출처 추적
    - AI가 해석한 규칙의 원본과 근거를 명확히 기록
    """
    original_text: str = Field(..., description="Excel B의 원본 자연어 규칙")
    sheet_name: str = Field(..., description="시트명")
    row_number: int = Field(..., description="행 번호")
    kifrs_reference: Optional[str] = Field(None, description="K-IFRS 1019 관련 조항")


class ValidationRule(BaseModel):
    """
    AI가 해석한 구조화된 규칙
    - 이 구조는 AI 출력이지만, 실행은 결정론적 엔진이 담당
    """
    rule_id: str = Field(..., description="고유 규칙 ID (예: RULE_001)")
    field_name: str = Field(..., description="검증 대상 필드명")
    rule_type: Literal[
        "required",
        "no_duplicates", 
        "format",
        "range",
        "date_logic",
        "cross_field",
        "custom",
        "composite"
    ] = Field(..., description="규칙 유형")
    
    # 규칙 파라미터 (타입별로 다름)
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="규칙 실행에 필요한 파라미터",
        examples=[
            {"format": "YYYYMMDD"},
            {"min_value": 0, "max_value": 150},
            {"compare_field": "birth_date", "operator": "greater_than"}
        ]
    )
    
    error_message_template: str = Field(
        ..., 
        description="오류 발생 시 사용할 메시지 템플릿"
    )
    
    source: RuleSource = Field(..., description="규칙 출처 정보")
    
    ai_interpretation_summary: str = Field(
        ..., 
        description="AI가 이 규칙을 어떻게 해석했는지 요약"
    )
    
    confidence_score: float = Field(
        ..., 
        ge=0.0, 
        le=1.0,
        description="AI 해석의 신뢰도 (0~1)"
    )
    
    is_common: bool = Field(False, description="공통 규칙 여부")


class RuleConflict(BaseModel):
    """
    규칙 충돌 보고서
    - AI가 감지한 규칙 간 또는 K-IFRS 1019와의 충돌
    """
    rule_id: str
    conflict_type: Literal[
        "rule_contradiction",
        "kifrs_mismatch", 
        "ambiguous_interpretation"
    ]
    description: str = Field(..., description="충돌 내용 설명")
    kifrs_reference: Optional[str] = Field(None, description="관련 K-IFRS 1019 조항")
    affected_rules: List[str] = Field(default_factory=list, description="영향받는 다른 규칙들")
    recommendation: str = Field(..., description="해결 방안 제안")
    severity: Literal["high", "medium", "low"] = Field(..., description="심각도")


class ValidationError(BaseModel):
    """
    개별 검증 인지
    - 결정론적 엔진이 생성한 인지 레코드
    """
    sheet: Optional[str] = Field(None, description="인지 발생 시트명")
    row: int = Field(..., description="인지 발생 행 (1-based)")
    column: str = Field(..., description="인지 발생 컬럼명")
    rule_id: str = Field(..., description="위반된 규칙 ID")
    message: str = Field(..., description="인지 메시지")
    actual_value: Any = Field(..., description="실제 입력된 값")
    expected: Optional[str] = Field(None, description="예상 값 또는 형식")
    source_rule: str = Field(..., description="원본 자연어 규칙 (추적용)")


class ValidationErrorGroup(BaseModel):
    """
    동일한 인지 내용의 집계
    - 같은 규칙 위반이 여러 행에서 발생한 경우 그룹화
    """
    sheet: str = Field(..., description="시트명")
    column: str = Field(..., description="컬럼명")
    rule_id: str = Field(..., description="규칙 ID")
    message: str = Field(..., description="인지 메시지")
    affected_rows: List[int] = Field(..., description="영향받은 행 번호 목록")
    count: int = Field(..., description="인지 발생 횟수")
    sample_values: List[Any] = Field(default_factory=list, description="샘플 값 (최대 3개)")
    expected: Optional[str] = Field(None, description="예상 값 또는 형식")
    source_rule: str = Field(..., description="원본 자연어 규칙")


class ValidationSummary(BaseModel):
    """
    검증 결과 요약 통계
    """
    total_rows: int = Field(..., description="전체 데이터 행 수")
    valid_rows: int = Field(..., description="정상 행 수")
    error_rows: int = Field(..., description="인지가 있는 행 수")
    total_errors: int = Field(..., description="총 인지 개수")
    rules_applied: int = Field(..., description="적용된 규칙 수")
    timestamp: datetime = Field(default_factory=datetime.now, description="검증 실행 시각")


class ValidationResponse(BaseModel):
    """
    최종 검증 결과 응답
    - 감사인에게 제출 가능한 포맷
    """
    validation_status: Literal["PASS", "FAIL"] = Field(
        ...,
        description="전체 검증 결과"
    )

    summary: ValidationSummary = Field(..., description="요약 통계")

    errors: List[ValidationError] = Field(
        default_factory=list,
        description="모든 검증 인지 목록 (개별)"
    )

    error_groups: List[ValidationErrorGroup] = Field(
        default_factory=list,
        description="인지 내용 집계 (동일 규칙 위반을 그룹화)"
    )

    conflicts: List[RuleConflict] = Field(
        default_factory=list,
        description="규칙 충돌 목록 (AI 감지)"
    )

    rules_applied: List[ValidationRule] = Field(
        default_factory=list,
        description="적용된 규칙 전체 목록 (감사추적용)"
    )

    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="기타 메타데이터",
        examples=[{
            "employee_file_name": "employees_2024.xlsx",
            "rules_file_name": "validation_rules_v2.xlsx",
            "ai_model_version": "claude-sonnet-4-20250514",
            "system_version": "1.0.0"
        }]
    )


class AIInterpretationRequest(BaseModel):
    """
    AI 해석 요청
    - Excel B의 자연어 규칙을 구조화된 JSON으로 변환 요청
    """
    natural_language_rules: List[Dict[str, Any]] = Field(
        ...,
        description="Excel B에서 읽은 자연어 규칙들",
        examples=[[
            {
                "sheet": "validation_rules",
                "row": 2,
                "field": "employee_id",
                "rule_text": "사번은 공백이 없어야 하며, 중복이 없어야 함"
            }
        ]]
    )


class AIInterpretationResponse(BaseModel):
    """
    AI 해석 결과
    - AI가 생성한 구조화된 규칙 + 충돌 보고서
    """
    rules: List[ValidationRule] = Field(..., description="해석된 구조화 규칙")
    conflicts: List[RuleConflict] = Field(default_factory=list, description="감지된 충돌")
    ai_summary: str = Field(..., description="전체 해석 요약")
    processing_time_seconds: float = Field(..., description="처리 소요 시간")


# =============================================================================
# K-IFRS 1019 참조 데이터 (상수)
# =============================================================================

KIFRS_1019_REFERENCES = {
    "employee_eligibility": {
        "paragraph": "28-31",
        "description": "확정급여제도의 급여 대상 직원 범위 정의",
        "key_points": [
            "과거 근무용역을 제공한 임직원",
            "퇴직, 사망, 장애 등으로 급여를 받을 자격이 있는 자"
        ]
    },
    "measurement_date": {
        "paragraph": "57-59",
        "description": "보고기간 말 현재 측정",
        "key_points": [
            "보고기간 말 현재의 현재가치로 측정",
            "보험수리적 가정 적용"
        ]
    },
    "actuarial_assumptions": {
        "paragraph": "75-98",
        "description": "보험수리적 가정",
        "key_points": [
            "할인율, 임금상승률, 사망률 등",
            "최선의 추정치 사용"
        ]
    },
    "data_requirements": {
        "paragraph": "관련 실무",
        "description": "DBO 계산을 위한 직원 데이터 요구사항",
        "key_points": [
            "생년월일: 정확한 연령 계산 필요",
            "입사일: 근속연수 계산",
            "급여 정보: 현재 및 예상 급여",
            "성별: 사망률 테이블 적용",
            "퇴직 예정일: 급여 지급 시점 추정"
        ]
    }
}


# =============================================================================
# 예시 데이터
# =============================================================================

EXAMPLE_VALIDATION_RESPONSE = {
    "validation_status": "FAIL",
    "summary": {
        "total_rows": 1250,
        "valid_rows": 1180,
        "error_rows": 70,
        "total_errors": 95,
        "rules_applied": 12,
        "timestamp": "2025-01-12T10:30:00Z"
    },
    "errors": [
        {
            "row": 15,
            "column": "birth_date",
            "rule_id": "RULE_003",
            "message": "생년월일 형식이 잘못되었습니다. YYYYMMDD 형식이어야 합니다.",
            "actual_value": "1985-03-15",
            "expected": "YYYYMMDD (예: 19850315)",
            "source_rule": "생년월일은 YYYYMMDD 형식으로 입력"
        },
        {
            "row": 42,
            "column": "hire_date",
            "rule_id": "RULE_005",
            "message": "입사일이 생년월일보다 이전입니다.",
            "actual_value": "20000101",
            "expected": "생년월일 이후",
            "source_rule": "입사일은 생년월일보다 이후여야 함"
        }
    ],
    "conflicts": [
        {
            "rule_id": "RULE_008",
            "conflict_type": "kifrs_mismatch",
            "description": "Excel B에서 '60세 이상은 제외'라고 명시했으나, K-IFRS 1019는 모든 재직 중인 임직원을 포함해야 함",
            "kifrs_reference": "K-IFRS 1019 문단 28-31",
            "affected_rules": ["RULE_007", "RULE_008"],
            "recommendation": "K-IFRS 1019 기준을 따르거나, 회계법인과 협의 필요",
            "severity": "high"
        }
    ],
    "rules_applied": [
        {
            "rule_id": "RULE_001",
            "field_name": "employee_id",
            "rule_type": "required",
            "parameters": {},
            "error_message_template": "사번이 비어있습니다.",
            "source": {
                "original_text": "사번: 공백 없음, 중복 없음",
                "sheet_name": "validation_rules",
                "row_number": 2,
                "kifrs_reference": None
            },
            "ai_interpretation_summary": "사번 필드는 필수이며 중복이 없어야 함",
            "confidence_score": 0.98
        }
    ],
    "metadata": {
        "employee_file_name": "employees_2024Q4.xlsx",
        "rules_file_name": "rules_v3.xlsx",
        "ai_model_version": "claude-sonnet-4-20250514",
        "system_version": "1.0.0"
    }
}


# =============================================================================
# Database-Related Models (Supabase Integration)
# =============================================================================

class RuleFileUpload(BaseModel):
    """Request model for uploading rule file to database"""
    file_name: str
    file_version: Optional[str] = None
    uploaded_by: Optional[str] = None
    notes: Optional[str] = None


class RuleFileResponse(BaseModel):
    """Response model for rule file metadata"""
    id: str  # UUID as string
    file_name: str
    file_version: Optional[str] = None
    uploaded_by: Optional[str] = None
    uploaded_at: datetime
    sheet_count: int
    total_rules_count: int
    status: str


class RuleUpdate(BaseModel):
    """Request model for updating an individual rule"""
    field_name: Optional[str] = None
    rule_text: Optional[str] = None
    condition: Optional[str] = None
    note: Optional[str] = None
    is_active: Optional[bool] = None
    is_common: Optional[bool] = None  # Common rule flag
    
    # AI-related fields (if user wants to manually adjust AI interpretation)
    ai_rule_type: Optional[str] = None
    ai_parameters: Optional[Dict[str, Any]] = None
    ai_error_message: Optional[str] = None


class RuleDetail(BaseModel):
    """Detailed rule information for editing"""
    id: str
    rule_file_id: str
    sheet_name: str
    display_sheet_name: Optional[str]
    canonical_sheet_name: Optional[str]
    row_number: Optional[int]
    column_letter: Optional[str]
    field_name: str
    rule_text: str
    condition: Optional[str]
    note: Optional[str]
    is_common: bool = False  # Common rule flag
    
    # AI Interpretation
    ai_rule_id: Optional[str]
    ai_rule_type: Optional[str]
    ai_parameters: Optional[Dict[str, Any]]
    ai_error_message: Optional[str]
    ai_interpretation_summary: Optional[str]
    ai_confidence_score: Optional[float]
    ai_interpreted_at: Optional[datetime]
    
    is_active: bool
    created_at: datetime
    updated_at: datetime


class RuleCreate(BaseModel):
    """Request model for creating a single rule manually"""
    rule_file_id: str
    sheet_name: str
    row_number: int = 0
    column_name: str  # mapped to field_name
    rule_text: str
    condition: Optional[str] = None
    is_common: bool = False
    ai_rule_type: Optional[str] = "custom"
    ai_parameters: Optional[Dict[str, Any]] = Field(default_factory=dict)


class RuleSourceType(BaseModel):
    """Request to specify rule source"""
    source_type: Literal["file_upload", "database"]
    rule_file_id: Optional[str] = None  # Required if source_type == "database"


class ValidationSessionResponse(BaseModel):
    """Response for validation session"""
    session_id: str  # UUID as string
    session_token: str
    validation_results: ValidationResponse
    stored_at: datetime


class UserCorrectionRequest(BaseModel):
    """Request to record user correction"""
    session_id: str  # UUID as string
    error_id: Optional[str] = None  # UUID as string
    correction_action: Literal["mark_false_positive", "adjust_rule", "confirm_error"]
    correction_reason: str
    old_value: Optional[str] = None
    new_value: Optional[str] = None
    suggested_rule_change: Optional[str] = None
    corrected_by: str


class FalsePositiveFeedback(BaseModel):
    """Feedback on false positives"""
    error_id: str  # UUID as string
    is_false_positive: bool
    user_explanation: str
    suggested_rule_adjustment: Optional[str] = None
    feedback_by: str


class AILearningStats(BaseModel):
    """AI learning statistics"""
    total_interpretations: int
    average_confidence: float
    false_positive_rate: float
    most_problematic_rules: List[Dict[str, Any]] = Field(default_factory=list)
    improvement_suggestions: List[str] = Field(default_factory=list)


# =============================================================================
# Phase 5: Smart Fix Models
# =============================================================================

class FixSuggestion(BaseModel):
    """AI 제안 수정안"""
    error_id: Optional[str] = Field(None, description="오류 ID (DB에 있는 경우)")
    sheet_name: str
    row: int
    column: str
    original_value: Any
    fixed_value: Any
    confidence_score: float = Field(..., ge=0.0, le=1.0)
    reason: str
    is_auto_fixable: bool = Field(False, description="신뢰도가 높아 자동 적용 가능한지")

class FixRequest(BaseModel):
    """사용자가 승인한 단일 수정 요청"""
    sheet_name: str
    row: int
    column: str
    original_value: Any
    fixed_value: Any
    error_id: Optional[str] = None

class BatchFixRequest(BaseModel):
    """일괄 수정 요청 (파일 생성용)"""
    session_id: str
    fixes: List[FixRequest]
    generate_file: bool = True  # True면 엑셀 파일 생성하여 반환

