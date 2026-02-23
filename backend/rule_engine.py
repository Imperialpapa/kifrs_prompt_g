"""
K-IFRS 1019 DBO Validation System - Rule Engine
================================================
AI가 해석한 규칙을 결정론적으로 실행하는 검증 엔진

🎯 핵심 원칙:
1. 100% 결정론적 실행 (동일 입력 → 동일 출력)
2. AI는 관여하지 않음 (규칙만 실행)
3. 감사 추적 가능 (모든 오류에 출처 명시)
4. 타입 안정성 (명확한 데이터 구조)
"""

import re
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Optional
from datetime import datetime
from utils.logger import get_logger
from models import (
    ValidationRule,
    ValidationError,
    ValidationSummary,
    ValidationResponse
)


class RuleEngine:
    """
    [결정론적 검증 엔진]
    
    AI가 해석한 JSON 형태의 규칙을 받아, Pandas DataFrame에 실제로 적용하고 오류를 검출합니다.
    이 클래스는 100% 결정론적(Deterministic)으로 동작하며, 동일한 입력에 대해 항상 동일한 결과를 보장합니다.
    
    [주요 기능]
    - 필수값, 중복, 형식(Regex), 범위(Range), 날짜 논리 등 기본 검증 수행
    - 오류 메시지 생성 시 필드명 자동 치환
    - 데이터 타입(숫자, 문자, 날짜)에 따른 유연한 처리
    """

    logger = get_logger("rule_engine")

    def __init__(self):
        """초기화"""
        self.errors: List[ValidationError] = []
        self.row_error_flags: set = set()  # 오류가 있는 행 번호 추적
    
    def validate(
        self,
        data: pd.DataFrame,
        rules: List[ValidationRule]
    ) -> List[ValidationError]:
        """
        데이터프레임에 규칙 적용
        
        Args:
            data: 검증할 데이터프레임 (Excel A)
            rules: AI가 해석한 규칙들
            
        Returns:
            List[ValidationError]: 발견된 모든 오류
        """
        self.errors = []
        self.row_error_flags = set()
        
        for rule in rules:
            self._apply_rule(data, rule)
        
        return self.errors
    
    def _apply_rule(self, data: pd.DataFrame, rule: ValidationRule):
        """
        개별 규칙 적용
        """
        # 행 검증 항목(axis='row')은 향후 추가 예정이므로 현재는 열(column) 검증만 수행
        if hasattr(rule, 'validation_axis') and rule.validation_axis == "row":
            self.logger.debug(f"Skipping row-based rule for now: {rule.rule_id}")
            return

        if rule.rule_type == "required":
            self._validate_required(data, rule)
        
        elif rule.rule_type == "no_duplicates":
            self._validate_no_duplicates(data, rule)
        
        elif rule.rule_type == "format":
            self._validate_format(data, rule)
        
        elif rule.rule_type == "allowed_values":
            self._validate_allowed_values(data, rule)
        
        elif rule.rule_type == "range":
            self._validate_range(data, rule)
        
        elif rule.rule_type == "date_logic":
            self._validate_date_logic(data, rule)
        
        elif rule.rule_type == "cross_field":
            self._validate_cross_field(data, rule)
        
        elif rule.rule_type == "custom":
            self._validate_custom(data, rule)

        elif rule.rule_type == "composite":
            self._validate_composite(data, rule)

        else:
            raise ValueError(f"알 수 없는 규칙 타입: {rule.rule_type}")
    
    # =========================================================================
    # 규칙 타입별 검증 메서드
    # =========================================================================
    
    def _validate_required(self, data: pd.DataFrame, rule: ValidationRule):
        """
        필수 필드 검증
        """
        field = rule.field_name
        
        if field not in data.columns:
            # 보조 매칭: 정규화된 이름으로 한 번 더 확인
            normalized_field = re.sub(r'[^a-zA-Z0-9가-힣]', '', str(field)).lower()
            found_col = None
            for col in data.columns:
                if re.sub(r'[^a-zA-Z0-9가-힣]', '', str(col)).lower() == normalized_field:
                    found_col = col
                    break
            
            if found_col:
                field = found_col
            else:
                # 컬럼 자체가 없으면 모든 행에 대해 오류 (단, 데이터가 있는 행에 대해서만)
                for idx in data.index:
                    self._add_error(
                        row=idx + 2,
                        column=field,
                        rule=rule,
                        message=f"필수 컬럼 '{field}'가 존재하지 않습니다.",
                        actual_value=None
                    )
                return
        
        # Null, NaN, 빈 문자열 체크 (인덱스 유지)
        for idx, value in data[field].items():
            if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                self._add_error(
                    row=idx + 2,
                    column=field,
                    rule=rule,
                    message=rule.error_message_template,
                    actual_value=value
                )
    
    def _validate_no_duplicates(self, data: pd.DataFrame, rule: ValidationRule):
        """
        중복 금지 검증
        """
        field = rule.field_name
        
        if field not in data.columns:
            return
        
        # 유효한 값(비어있지 않은 값)만 대상으로 중복 검사 수행
        # NaN, None, 빈 문자열, 공백만 있는 문자열 제외
        valid_mask = data[field].astype(str).str.strip().replace(['nan', 'None', 'NaT', ''], np.nan).notna()
        valid_data = data[valid_mask]
        
        if valid_data.empty:
            return

        # 중복 찾기
        duplicates = valid_data[valid_data.duplicated(subset=[field], keep=False)]
        
        for idx in duplicates.index:
            value = data.loc[idx, field]
            self._add_error(
                row=idx + 2,
                column=field,
                rule=rule,
                message=rule.error_message_template,
                actual_value=value,
                expected="고유값"
            )
    
    def _validate_format(self, data: pd.DataFrame, rule: ValidationRule):
        """
        형식 검증 (regex, allowed_values 등)
        """
        field = rule.field_name
        params = rule.parameters
        
        if field not in data.columns:
            return
        
        # Helper to normalize value for format checks
        def normalize_val(v):
            if pd.isna(v): return v
            # If float ending in .0, convert to int then str
            if isinstance(v, float) and v.is_integer():
                return str(int(v))
            return str(v)

        # allowed_values 검증
        if "allowed_values" in params:
            allowed = params["allowed_values"]
            for idx, value in data[field].items():
                if pd.notna(value):
                    # Check both raw value and normalized string value
                    norm_val = normalize_val(value)
                    if value not in allowed and norm_val not in [str(a) for a in allowed]:
                        self._add_error(
                            row=idx + 2,
                            column=field,
                            rule=rule,
                            message=rule.error_message_template,
                            actual_value=value,
                            expected=f"{allowed} 중 하나"
                        )
        
        # regex 검증
        elif "regex" in params:
            pattern = re.compile(params["regex"])
            for idx, value in data[field].items():
                if pd.notna(value):
                    value_str = normalize_val(value)
                    if not pattern.match(value_str):
                        self._add_error(
                            row=idx + 2,
                            column=field,
                            rule=rule,
                            message=rule.error_message_template,
                            actual_value=value,
                            expected=params.get("format", "정규식 패턴 일치")
                        )
        
        # format 검증 (예: YYYYMMDD)
        elif "format" in params:
            fmt = params["format"]
            for idx, value in data[field].items():
                if pd.notna(value):
                    value_str = normalize_val(value)
                    if not self._check_date_format(value_str, fmt):
                        self._add_error(
                            row=idx + 2,
                            column=field,
                            rule=rule,
                            message=rule.error_message_template,
                            actual_value=value,
                            expected=fmt
                        )
    
    def _validate_allowed_values(self, data: pd.DataFrame, rule: ValidationRule):
        """
        허용값 목록(나열형) 검증
        """
        field = rule.field_name
        params = rule.parameters
        
        if field not in data.columns:
            return
            
        allowed = params.get("allowed_values", [])
        if not allowed:
            return

        def normalize_val(v):
            if pd.isna(v): return v
            if isinstance(v, float) and v.is_integer():
                return str(int(v))
            return str(v)

        for idx, value in data[field].items():
            if pd.notna(value):
                norm_val = normalize_val(value)
                # 원본값 또는 문자열 변환값이 허용 목록에 있는지 확인
                if value not in allowed and norm_val not in [str(a) for a in allowed]:
                    self._add_error(
                        row=idx + 2,
                        column=field,
                        rule=rule,
                        message=rule.error_message_template,
                        actual_value=value,
                        expected=f"{allowed} 중 하나"
                    )

    def _validate_range(self, data: pd.DataFrame, rule: ValidationRule):
        """
        범위 검증 (숫자 또는 날짜)
        """
        field = rule.field_name
        params = rule.parameters
        
        if field not in data.columns:
            return
        
        # 날짜 범위 검증
        if "min_date" in params or "max_date" in params:
            min_date = params.get("min_date")
            max_date = params.get("max_date")
            
            for idx, value in data[field].items():
                if pd.notna(value):
                    value_str = str(value)
                    if min_date and value_str < min_date:
                        self._add_error(
                            row=idx + 2,
                            column=field,
                            rule=rule,
                            message=rule.error_message_template,
                            actual_value=value,
                            expected=f">= {min_date}"
                        )
                    if max_date and value_str > max_date:
                        self._add_error(
                            row=idx + 2,
                            column=field,
                            rule=rule,
                            message=rule.error_message_template,
                            actual_value=value,
                            expected=f"<= {max_date}"
                        )
        
        # 숫자 범위 검증
        elif "min_value" in params or "max_value" in params:
            min_val = params.get("min_value")
            max_val = params.get("max_value")
            
            for idx, value in data[field].items():
                if pd.notna(value):
                    try:
                        num_val = float(value)
                        if min_val is not None and num_val < min_val:
                            self._add_error(
                                row=idx + 2,
                                column=field,
                                rule=rule,
                                message=rule.error_message_template,
                                actual_value=value,
                                expected=f">= {min_val}"
                            )
                        if max_val is not None and num_val > max_val:
                            self._add_error(
                                row=idx + 2,
                                column=field,
                                rule=rule,
                                message=rule.error_message_template,
                                actual_value=value,
                                expected=f"<= {max_val}"
                            )
                    except (ValueError, TypeError):
                        self._add_error(
                            row=idx + 2,
                            column=field,
                            rule=rule,
                            message=f"숫자 형식이 아닙니다: {value}",
                            actual_value=value,
                            expected="숫자"
                        )
    
    def _validate_date_logic(self, data: pd.DataFrame, rule: ValidationRule):
        """
        날짜 논리 검증 (예: 입사일 > 생년월일, 중간정산기준일 <= 입사일)

        지원 연산자:
        - greater_than: >
        - less_than: <
        - greater_than_or_equal: >=
        - less_than_or_equal: <=
        - equal: =
        - not_equal: <>
        """
        field = rule.field_name
        params = rule.parameters

        if field not in data.columns:
            return

        compare_field = params.get("compare_field")
        operator = params.get("operator")

        if not compare_field or compare_field not in data.columns:
            # 비교 대상 필드가 없으면 경고 로그만 출력
            self.logger.warning(f"compare_field '{compare_field}' not found in data columns for date_logic rule on '{field}'")
            return

        # 연산자 표시 맵
        op_display = {
            "greater_than": ">",
            "less_than": "<",
            "greater_than_or_equal": ">=",
            "less_than_or_equal": "<=",
            "equal": "=",
            "not_equal": "<>"
        }

        for idx, row in data.iterrows():
            value1 = row[field]
            value2 = row[compare_field]

            if pd.isna(value1) or pd.isna(value2):
                continue

            # 문자열로 변환하여 비교 (날짜 형식 YYYYMMDD는 문자열 비교로 정확함)
            str_val1 = str(value1).strip()
            str_val2 = str(value2).strip()

            # 숫자형 날짜 처리 (예: 20230101.0 -> 20230101)
            if '.' in str_val1:
                try:
                    str_val1 = str(int(float(str_val1)))
                except (ValueError, TypeError):
                    pass
            if '.' in str_val2:
                try:
                    str_val2 = str(int(float(str_val2)))
                except (ValueError, TypeError):
                    pass

            is_valid = True

            # 날짜 비교
            if operator == "greater_than":
                if str_val1 <= str_val2:
                    is_valid = False

            elif operator == "less_than":
                if str_val1 >= str_val2:
                    is_valid = False

            elif operator == "greater_than_or_equal":
                if str_val1 < str_val2:
                    is_valid = False

            elif operator == "less_than_or_equal":
                if str_val1 > str_val2:
                    is_valid = False

            elif operator == "equal":
                if str_val1 != str_val2:
                    is_valid = False

            elif operator == "not_equal":
                if str_val1 == str_val2:
                    is_valid = False

            if not is_valid:
                self._add_error(
                    row=idx + 2,
                    column=field,
                    rule=rule,
                    message=rule.error_message_template,
                    actual_value=f"{field}={str_val1}, {compare_field}={str_val2}",
                    expected=f"{field} {op_display.get(operator, operator)} {compare_field}"
                )

            # 최소 나이 체크 (입사 시)
            if "min_age_at_hire" in params:
                min_age = params["min_age_at_hire"]
                try:
                    birth_year = int(str(value2)[:4])
                    hire_year = int(str(value1)[:4])
                    age_at_hire = hire_year - birth_year

                    if age_at_hire < min_age:
                        self._add_error(
                            row=idx + 2,
                            column=field,
                            rule=rule,
                            message=f"입사 시 만 {age_at_hire}세로, 최소 만 {min_age}세 미만입니다.",
                            actual_value=f"만 {age_at_hire}세",
                            expected=f"만 {min_age}세 이상"
                        )
                except (ValueError, TypeError):
                    pass
    
    def _validate_cross_field(self, data: pd.DataFrame, rule: ValidationRule):
        """
        필드 간 교차 검증
        """
        field = rule.field_name
        params = rule.parameters
        
        if field not in data.columns:
            return
        
        reference_field = params.get("reference_field")
        condition = params.get("condition")
        
        if not reference_field or reference_field not in data.columns:
            return
        
        for idx, row in data.iterrows():
            value = row[field]
            ref_value = row[reference_field]

            if condition == "required_if_not_null":
                if pd.notna(ref_value) and pd.isna(value):
                    self._add_error(
                        row=idx + 2,
                        column=field,
                        rule=rule,
                        message=rule.error_message_template,
                        actual_value=value,
                        expected=f"{reference_field}이(가) 있을 때 필수"
                    )
    
    def _validate_custom(self, data: pd.DataFrame, rule: ValidationRule):
        """
        사용자 정의 검증
        - 복잡한 비즈니스 로직
        """
        # 예시: eval을 사용한 동적 검증 (실제로는 보안 고려 필요)
        # 여기서는 간단히 pass
        pass

    def _validate_composite(self, data: pd.DataFrame, rule: ValidationRule):
        """
        복합 검증 (Composite Validation)
        - 여러 검증 조건을 순차적으로 적용
        - parameters.validations 배열에 각 검증 조건 포함

        validations 배열 구조:
        [
            {"type": "required", "parameters": {}, "error_message": "..."},
            {"type": "format", "parameters": {"format": "YYYYMMDD", ...}, "error_message": "..."},
            ...
        ]
        """
        validations = rule.parameters.get("validations", [])

        if not validations:
            return

        field = rule.field_name

        # 컬럼 존재 여부 확인
        if field not in data.columns:
            # 컬럼이 없으면 required 검증만 실패로 처리
            for v in validations:
                if v.get("type") == "required":
                    for idx in data.index:
                        self._add_error(
                            row=idx + 2,
                            column=field,
                            rule=rule,
                            message=v.get("error_message", f"필수 컬럼 '{field}'가 존재하지 않습니다."),
                            actual_value=None
                        )
            return

        # 각 행에 대해 모든 검증 조건 적용 (인덱스 유지)
        for idx, value in data[field].items():
            row_num = idx + 2  # Excel 행 번호

            for v in validations:
                v_type = v.get("type")
                v_params = v.get("parameters", {})
                v_error_msg = v.get("error_message", f"{field} 검증 실패")

                # 1. Required 검증
                if v_type == "required":
                    if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                        self._add_error(
                            row=row_num,
                            column=field,
                            rule=rule,
                            message=v_error_msg,
                            actual_value=value
                        )
                        # required 실패 시 다른 검증은 의미 없음
                        break

                # 빈 값이면 나머지 검증 스킵 (required 아닌 경우)
                if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                    continue

                # Helper to normalize value for format checks
                def normalize_val(v):
                    if pd.isna(v): return v
                    if isinstance(v, float) and v.is_integer():
                        return str(int(v))
                    return str(v)

                str_value = str(value).strip()
                norm_value = normalize_val(value) # Normalized (e.g. "19881115")

                # 2. Format 검증
                if v_type == "format":
                    is_valid = True

                    # 정규식 검증
                    if "regex" in v_params:
                        import re
                        # Check both original str_value and normalized value
                        # (Original might be "19881115.0", Normalized "19881115")
                        if not re.match(v_params["regex"], str_value) and not re.match(v_params["regex"], norm_value):
                            is_valid = False

                    # 허용값 목록 검증
                    elif "allowed_values" in v_params:
                        allowed = v_params["allowed_values"]
                        # Check original, normalized, and numeric conversion
                        if str_value not in allowed and value not in allowed and norm_value not in [str(a) for a in allowed]:
                            is_valid = False

                    if not is_valid:
                        self._add_error(
                            row=row_num,
                            column=field,
                            rule=rule,
                            message=v_error_msg,
                            actual_value=value
                        )

                # 3. Range 검증
                elif v_type == "range":
                    try:
                        num_value = float(value)
                        is_valid = True

                        if "min_value" in v_params:
                            min_val = v_params["min_value"]
                            if v_params.get("exclusive_min"):
                                if num_value <= min_val:
                                    is_valid = False
                            else:
                                if num_value < min_val:
                                    is_valid = False

                        if "max_value" in v_params:
                            max_val = v_params["max_value"]
                            if v_params.get("exclusive_max"):
                                if num_value >= max_val:
                                    is_valid = False
                            else:
                                if num_value > max_val:
                                    is_valid = False

                        if not is_valid:
                            self._add_error(
                                row=row_num,
                                column=field,
                                rule=rule,
                                message=v_error_msg,
                                actual_value=value
                            )
                    except (ValueError, TypeError):
                        # 숫자 변환 실패
                        if v_params.get("numeric_only"):
                            self._add_error(
                                row=row_num,
                                column=field,
                                rule=rule,
                                message=v_error_msg,
                                actual_value=value
                            )

                # 4. No Duplicates 검증은 별도 처리 필요 (행 단위가 아닌 컬럼 전체 대상)

                # 5. Date Logic 검증 (필드 간 비교)
                elif v_type == "date_logic":
                    compare_field = v_params.get("compare_field")
                    operator = v_params.get("operator")

                    if compare_field and compare_field in data.columns:
                        compare_value = data.loc[idx, compare_field]

                        if pd.notna(compare_value):
                            # 문자열로 변환
                            str_val1 = str(value).strip()
                            str_val2 = str(compare_value).strip()

                            # 숫자형 날짜 처리
                            if '.' in str_val1:
                                try:
                                    str_val1 = str(int(float(str_val1)))
                                except (ValueError, TypeError):
                                    pass
                            if '.' in str_val2:
                                try:
                                    str_val2 = str(int(float(str_val2)))
                                except (ValueError, TypeError):
                                    pass

                            is_valid = True

                            if operator == "greater_than":
                                if str_val1 <= str_val2:
                                    is_valid = False
                            elif operator == "less_than":
                                if str_val1 >= str_val2:
                                    is_valid = False
                            elif operator == "greater_than_or_equal":
                                if str_val1 < str_val2:
                                    is_valid = False
                            elif operator == "less_than_or_equal":
                                if str_val1 > str_val2:
                                    is_valid = False
                            elif operator == "equal":
                                if str_val1 != str_val2:
                                    is_valid = False
                            elif operator == "not_equal":
                                if str_val1 == str_val2:
                                    is_valid = False

                            if not is_valid:
                                self._add_error(
                                    row=row_num,
                                    column=field,
                                    rule=rule,
                                    message=v_error_msg,
                                    actual_value=f"{field}={str_val1}, {compare_field}={str_val2}"
                                )

        # No Duplicates 검증 (컬럼 전체 대상)
        for v in validations:
            if v.get("type") == "no_duplicates":
                v_error_msg = v.get("error_message", f"{field}이(가) 중복되었습니다.")
                seen = {}
                for idx, value in data[field].items():
                    if pd.isna(value) or (isinstance(value, str) and value.strip() == ""):
                        continue

                    str_value = str(value).strip()
                    if str_value in seen:
                        # 중복 발견
                        self._add_error(
                            row=idx + 2,
                            column=field,
                            rule=rule,
                            message=v_error_msg,
                            actual_value=value
                        )
                    else:
                        seen[str_value] = idx + 2

    # =========================================================================
    # 유틸리티 메서드
    # =========================================================================
    
    def _check_date_format(self, value: str, format_str: str) -> bool:
        """
        날짜 형식 체크
        """
        if format_str == "YYYYMMDD":
            if len(value) != 8:
                return False
            try:
                year = int(value[:4])
                month = int(value[4:6])
                day = int(value[6:8])
                # 간단한 유효성 체크
                if not (1900 <= year <= 2100):
                    return False
                if not (1 <= month <= 12):
                    return False
                if not (1 <= day <= 31):
                    return False
                # 실제 날짜 유효성
                datetime(year, month, day)
                return True
            except (ValueError, TypeError):
                return False
        
        return True
    
    def _add_error(
        self,
        row: int,
        column: str,
        rule: ValidationRule,
        message: str,
        actual_value: Any,
        expected: Optional[str] = None
    ):
        """
        오류 추가
        - 메시지에 {field_name} 플레이스홀더를 실제 필드명으로 치환
        - 잘못된 필드명이 하드코딩된 경우 제거
        """
        # 1. {field_name} 플레이스홀더 치환
        if "{field_name}" in message:
            message = message.replace("{field_name}", column)

        # 2. 현재 필드명과 다른 필드명이 메시지에 포함된 경우 제거
        # (예: "생년월일" 필드인데 "사원번호이(가)" 라는 텍스트가 있는 경우)
        if column and column not in message:
            # 일반적인 필드명 패턴 (한글, 영문, 숫자, _)을 찾아서 제거
            # "XXX이(가)", "XXX은(는)", "XXX을(를)" 같은 패턴 감지
            wrong_field_patterns = [
                r'([가-힣a-zA-Z0-9_]+)이\(가\)',
                r'([가-힣a-zA-Z0-9_]+)은\(는\)',
                r'([가-힣a-zA-Z0-9_]+)을\(를\)',
                r'([가-힣a-zA-Z0-9_]+)\s*값',
                r'([가-힣a-zA-Z0-9_]+)\s*형식'
            ]

            for pattern in wrong_field_patterns:
                match = re.search(pattern, message)
                if match:
                    found_field = match.group(1)
                    # 발견된 필드명이 현재 컬럼명과 다르면 현재 컬럼명으로 교체
                    if found_field != column:
                        message = message.replace(found_field, column)
                        break

        # 3. 여전히 현재 필드명이 메시지에 없으면 앞에 추가
        if column and column not in message:
            if any(keyword in message for keyword in ["중복", "비어있습니다", "필수", "형식", "범위", "값", "올바르지"]):
                message = f"{column}: {message}"

        error = ValidationError(
            row=row,
            column=column,
            rule_id=rule.rule_id,
            message=message,
            actual_value=actual_value,
            expected=expected,
            source_rule=rule.source.original_text
        )

        self.errors.append(error)
        self.row_error_flags.add(row)
    
    def get_summary(self, total_rows: int, rules_count: int) -> ValidationSummary:
        """
        검증 요약 통계 생성
        """
        error_rows = len(self.row_error_flags)
        
        return ValidationSummary(
            total_rows=total_rows,
            valid_rows=total_rows - error_rows,
            error_rows=error_rows,
            total_errors=len(self.errors),
            rules_applied=rules_count,
            timestamp=datetime.now()
        )


# =============================================================================
# 편의 함수
# =============================================================================

def validate_data(
    data: pd.DataFrame,
    rules: List[ValidationRule]
) -> ValidationResponse:
    """
    데이터 검증 실행 및 응답 생성
    
    Args:
        data: 검증할 데이터프레임
        rules: AI가 해석한 규칙들
        
    Returns:
        ValidationResponse: 전체 검증 결과
    """
    engine = RuleEngine()
    errors = engine.validate(data, rules)
    summary = engine.get_summary(len(data), len(rules))
    
    return ValidationResponse(
        validation_status="PASS" if len(errors) == 0 else "FAIL",
        summary=summary,
        errors=errors,
        conflicts=[],  # 규칙 충돌은 AI 레이어에서 처리
        rules_applied=rules
    )


# =============================================================================
# K-IFRS 1019 검증 엔진 (2단계)
# =============================================================================

class KIFRS_RuleEngine:
    """
    [K-IFRS 1019 특화 검증 엔진]
    
    기본적인 데이터 정합성 검증(1단계) 이후, 퇴직급여부채 평가를 위한 
    회계적/보험수리적 관점의 심층 검증을 수행합니다.
    
    [검증 항목]
    1. 완전성(Completeness): 필수 평가 요소(사번, 입사일, 급여 등) 누락 확인
    2. 유효성(Validity): 평가기준일 대비 미래 입사일 등 논리적 모순 확인
    3. 일관성(Consistency): 
       - 입사일 vs 퇴사일 역전
       - 관계사 전입일 vs 현사 입사일 역전
       - **입사 시 연령 체크 (만 15세 미만 등)**
    4. 가정 검증(Actuarial Assumptions):
       - 할인율 (0.5% ~ 15% 범위 체크)
       - 임금상승률 (0% ~ 20% 범위 체크)
    5. 이상치(Outliers): 평균임금 3시그마 벗어나는 값 탐지
    """
    
    def __init__(self, data: pd.DataFrame):
        self.data = data.copy()
        self.errors: List[ValidationError] = []
        self._preprocess_data()

    def _preprocess_data(self):
        """데이터 전처리 (날짜 변환, 숫자 변환 등)"""
        date_cols = ['birth_date', 'hire_date', 'termination_date', 'first_hire_date_affiliated', 'evaluation_date']
        for col in date_cols:
            if col in self.data.columns:
                # format을 지정하여 다양한 형식의 날짜 문자열을 파싱
                self.data[col] = pd.to_datetime(self.data[col], errors='coerce', format='%Y%m%d')

        numeric_cols = ['average_wage', 'payment_rate']
        for col in numeric_cols:
             if col in self.data.columns:
                self.data[col] = pd.to_numeric(self.data[col], errors='coerce')
    
    def run_all_checks(self, reconciliation_params: Optional[Dict[str, Any]] = None) -> List[ValidationError]:
        """K-IFRS 6대 검증 축을 모두 실행"""
        self.errors = []
        
        self._check_completeness()
        self._check_validity()
        self._check_consistency()
        self._check_actuarial_assumptions() # 신규 추가
        if reconciliation_params:
            self._check_reconciliation(reconciliation_params)
        self._check_outliers()
        self._check_roll_forward_skeleton() # 골격만 구현

        return self.errors

    def _check_actuarial_assumptions(self):
        """보험수리적 가정(할인율, 임금상승률 등)의 합리적 범위 검증"""
        # 1. 할인율 (Discount Rate) 검증 (예: 1% ~ 10%)
        discount_cols = [c for c in self.data.columns if any(kw in c.lower() for kw in ['discount', '할인율'])]
        for col in discount_cols:
            invalid_rates = self.data[self.data[col].notna()]
            for idx, row in invalid_rates.iterrows():
                val = float(row[col])
                # 보통 백분율(2.5) 또는 소수점(0.025)으로 입력됨
                rate = val if val < 1 else val / 100
                if rate < 0.005 or rate > 0.15: # 0.5% 미만 또는 15% 초과 시 경고
                    self._add_kifrs_error(
                        row=idx + 2, column=col,
                        message=f"보험수리적 가정(할인율: {val}%)이 통상적인 범위를 벗어납니다 (권장: 1%~10%).",
                        actual_value=val, rule_id="KIFRS_ASSUMPTION_DISCOUNT",
                        expected="1% ~ 10%"
                    )

        # 2. 임금상승률 (Salary Increase Rate) 검증 (예: 0% ~ 15%)
        salary_inc_cols = [c for c in self.data.columns if any(kw in c.lower() for kw in ['salary_increase', '임금상승률', '급여인상률'])]
        for col in salary_inc_cols:
            invalid_rates = self.data[self.data[col].notna()]
            for idx, row in invalid_rates.iterrows():
                val = float(row[col])
                rate = val if val < 1 else val / 100
                if rate < 0 or rate > 0.20: # 0% 미만 또는 20% 초과 시 경고
                    self._add_kifrs_error(
                        row=idx + 2, column=col,
                        message=f"보험수리적 가정(임금상승률: {val}%)이 통상적인 범위를 벗어납니다 (권장: 0%~15%).",
                        actual_value=val, rule_id="KIFRS_ASSUMPTION_SALARY",
                        expected="0% ~ 15%"
                    )

    def _add_kifrs_error(self, row: int, column: str, message: str, actual_value: Any, rule_id: str, expected: Optional[str] = None):
        """K-IFRS 검증 오류 추가"""
        error = ValidationError(
            row=row,
            column=column,
            rule_id=rule_id,
            message=message,
            actual_value=str(actual_value),
            expected=expected,
            source_rule="K-IFRS 1019 Guideline"
        )
        self.errors.append(error)

    # (1) 완전성 검증
    def _check_completeness(self):
        """필수 필드의 NULL/빈값 카운트 등"""
        required_fields = ['employee_code', 'employee_name', 'hire_date', 'birth_date', 'average_wage']
        for field in required_fields:
            if field not in self.data.columns:
                self._add_kifrs_error(row=0, column=field, message=f"필수 컬럼 '{field}'가 누락되었습니다.", actual_value="N/A", rule_id="KIFRS_COMPLETENESS_COL_MISSING")
                continue

            null_count = self.data[field].isnull().sum()
            if null_count > 0:
                 self._add_kifrs_error(
                    row=0, # 특정 행이 아닌 전체에 대한 오류
                    column=field,
                    message=f"필수 필드 '{field}'에 {null_count}개의 누락된 값이 있습니다.",
                    actual_value=f"{null_count} nulls",
                    rule_id="KIFRS_COMPLETENESS_NULL"
                )

    # (2) 형식/유효성 검증
    def _check_validity(self):
        """날짜 유효성, 평균임금 > 0 등"""
        if 'evaluation_date' in self.data.columns and not self.data['evaluation_date'].isnull().all():
            eval_date = self.data['evaluation_date'].dropna().iloc[0]
            if pd.notna(eval_date):
                for col in ['birth_date', 'hire_date']:
                    if col in self.data.columns:
                        future_dates = self.data[self.data[col] > eval_date]
                        for idx, row_data in future_dates.iterrows():
                            self._add_kifrs_error(
                                row=idx + 2, column=col,
                                message=f"날짜({row_data[col].date()})가 평가기준일({eval_date.date()})보다 미래일 수 없습니다.",
                                actual_value=row_data[col].date(), rule_id="KIFRS_VALIDITY_FUTURE_DATE"
                            )

        if 'average_wage' in self.data.columns:
            invalid_wage = self.data[self.data['average_wage'] <= 0]
            for idx, row_data in invalid_wage.iterrows():
                self._add_kifrs_error(
                    row=idx + 2, column='average_wage',
                    message="평균임금은 0보다 커야 합니다.",
                    actual_value=row_data['average_wage'], rule_id="KIFRS_VALIDITY_WAGE"
                )

    # (3) 논리 일관성 검증
    def _check_consistency(self):
        """입사일 <= 퇴사일 등"""
        if 'hire_date' in self.data.columns and 'termination_date' in self.data.columns:
            inconsistent_dates = self.data.dropna(subset=['hire_date', 'termination_date'])
            inconsistent_dates = inconsistent_dates[inconsistent_dates['hire_date'] > inconsistent_dates['termination_date']]
            for idx, row_data in inconsistent_dates.iterrows():
                self._add_kifrs_error(
                    row=idx + 2, column='termination_date',
                    message=f"퇴사일({row_data['termination_date'].date()})이 입사일({row_data['hire_date'].date()})보다 빠릅니다.",
                    actual_value=f"Hire: {row_data['hire_date'].date()}, Term: {row_data['termination_date'].date()}",
                    rule_id="KIFRS_CONSISTENCY_DATES"
                )

        if 'first_hire_date_affiliated' in self.data.columns and 'hire_date' in self.data.columns:
            inconsistent_first_hire = self.data.dropna(subset=['first_hire_date_affiliated', 'hire_date'])
            inconsistent_first_hire = inconsistent_first_hire[inconsistent_first_hire['first_hire_date_affiliated'] > inconsistent_first_hire['hire_date']]
            for idx, row_data in inconsistent_first_hire.iterrows():
                self._add_kifrs_error(
                    row=idx + 2, column='hire_date',
                    message=f"현재 회사 입사일({row_data['hire_date'].date()})이 관계사 최초 입사일({row_data['first_hire_date_affiliated'].date()})보다 빠릅니다.",
                    actual_value=f"First Hire: {row_data['first_hire_date_affiliated'].date()}, Current Hire: {row_data['hire_date'].date()}",
                    rule_id="KIFRS_CONSISTENCY_FIRST_HIRE"
                )

        # (신규) 입사 시 연령 검증 (만 15세 미만 등 체크)
        if 'hire_date' in self.data.columns and 'birth_date' in self.data.columns:
            for idx, row_data in self.data.dropna(subset=['hire_date', 'birth_date']).iterrows():
                try:
                    age_at_hire = row_data['hire_date'].year - row_data['birth_date'].year
                    if age_at_hire < 15: # 법정 취업 가능 연령 기준 경고
                        self._add_kifrs_error(
                            row=idx + 2, column='hire_date',
                            message=f"입사 시 연령이 만 {age_at_hire}세로 매우 낮습니다. 데이터 확인이 필요합니다.",
                            actual_value=f"만 {age_at_hire}세",
                            rule_id="KIFRS_CONSISTENCY_AGE_AT_HIRE",
                            expected="만 15세 이상"
                        )
                except:
                    pass
    
    # (4) 집계 리콘 검증
    def _check_reconciliation(self, params: Dict[str, Any]):
        """총 인원수, 총 평균임금 합계 등 비교"""
        if 'total_employee_count' in params:
            count_in_data = len(self.data)
            count_from_source = params['total_employee_count']
            if count_in_data != count_from_source:
                self._add_kifrs_error(
                    row=0, column='(Summary)',
                    message=f"총 인원 수가 일치하지 않습니다. (데이터: {count_in_data}, 원천: {count_from_source})",
                    actual_value=count_in_data, expected=str(count_from_source),
                    rule_id="KIFRS_RECON_TOTAL_COUNT"
                )

        if 'total_average_wage' in params and 'average_wage' in self.data.columns:
            sum_in_data = self.data['average_wage'].sum()
            sum_from_source = params['total_average_wage']
            tolerance = params.get('tolerance', 0.001) # 0.1%
            if abs(sum_in_data - sum_from_source) / sum_from_source > tolerance:
                self._add_kifrs_error(
                    row=0, column='average_wage',
                    message=f"총 평균임금 합계가 허용 오차({tolerance*100}%)를 벗어났습니다. (데이터 합계: {sum_in_data:,.0f}, 원천 합계: {sum_from_source:,.0f})",
                    actual_value=f"{sum_in_data:,.0f}", expected=f"~{sum_from_source:,.0f}",
                    rule_id="KIFRS_RECON_WAGE_SUM"
                )

    # (5) 이상치 탐지
    def _check_outliers(self):
        """
        수치 필드에 대한 이상치 탐지 (Z-score + Modified Z-score + IQR 병행)
        평균임금, 근속연수 등 수치 필드에 자동 적용
        """
        import numpy as np

        # 이상치 탐지 대상 컬럼 (키워드 기반)
        numeric_keywords = {
            'average_wage': ['평균임금', '급여', '임금', 'wage', 'salary'],
            'service_years': ['근속연수', '근무연수', 'service_year', 'tenure'],
            'payment_rate': ['지급배수', '지급률', 'payment_rate']
        }

        for label, keywords in numeric_keywords.items():
            target_col = None
            for col in self.data.columns:
                col_lower = str(col).lower()
                if any(kw in col_lower for kw in keywords):
                    target_col = col
                    break

            if not target_col or self.data[target_col].notna().sum() < 5:
                continue

            numeric_data = pd.to_numeric(self.data[target_col], errors='coerce')
            valid_data = numeric_data.dropna()

            if len(valid_data) < 5:
                continue

            # Method 1: Z-score (평균±3σ)
            mean = valid_data.mean()
            std = valid_data.std()
            if std > 0:
                z_lower = mean - 3 * std
                z_upper = mean + 3 * std
            else:
                z_lower = mean
                z_upper = mean

            # Method 2: Modified Z-score (MAD 기반, 더 로버스트)
            median = valid_data.median()
            mad = np.median(np.abs(valid_data - median))
            if mad > 0:
                modified_z_threshold = 3.5
                # Modified Z-score = 0.6745 * (x - median) / MAD
                # |Modified Z| > 3.5 이면 이상치
                mad_lower = median - modified_z_threshold * mad / 0.6745
                mad_upper = median + modified_z_threshold * mad / 0.6745
            else:
                mad_lower = median
                mad_upper = median

            # Method 3: IQR
            q1 = valid_data.quantile(0.25)
            q3 = valid_data.quantile(0.75)
            iqr = q3 - q1
            iqr_lower = q1 - 1.5 * iqr
            iqr_upper = q3 + 1.5 * iqr

            # 종합: 두 가지 이상 방법에서 이상치로 판정된 경우만 보고
            outlier_mask = (
                ((numeric_data < z_lower) | (numeric_data > z_upper)) &
                ((numeric_data < iqr_lower) | (numeric_data > iqr_upper))
            )

            outliers = self.data[outlier_mask & numeric_data.notna()]

            if not outliers.empty:
                self._add_kifrs_error(
                    row=0, column=str(target_col),
                    message=(
                        f"{len(outliers)}개의 {target_col} 이상치가 탐지되었습니다. "
                        f"(Z-score 범위: [{z_lower:,.0f}~{z_upper:,.0f}], "
                        f"IQR 범위: [{iqr_lower:,.0f}~{iqr_upper:,.0f}])"
                    ),
                    actual_value=f"{len(outliers)} outliers",
                    expected="Within normal range",
                    rule_id=f"KIFRS_OUTLIER_{label.upper()}_SUMMARY"
                )
                for idx, row_data in outliers.iterrows():
                    val = numeric_data.loc[idx]
                    self._add_kifrs_error(
                        row=idx + 2, column=str(target_col),
                        message=f"{target_col} 이상치 탐지 (Z-score + IQR 기준).",
                        actual_value=f"{val:,.0f}" if pd.notna(val) else "N/A",
                        expected=f"Range [{z_lower:,.0f}, {z_upper:,.0f}]",
                        rule_id=f"KIFRS_OUTLIER_{label.upper()}_ROW"
                    )

    # (6) 회계 리콘 (롤포워드)
    def _check_roll_forward_skeleton(self):
        """롤포워드 검증 함수 골격"""
        # "전기말 부채 + 당기 서비스원가 + 순이자비용 - 지급액 ± 재측정 = 당기말 부채"
        # 이 검증은 여러 데이터 소스가 필요하므로, 여기서는 골격만 만듭니다.
        # 실제 구현 시에는 financial_data 같은 별도의 파라미터를 받아야 합니다.
        pass


# =============================================================================
# 테스트 코드
# =============================================================================

if __name__ == "__main__":
    import json
    from models import ValidationRule, RuleSource
    
    # 샘플 데이터
    sample_data = pd.DataFrame({
        "employee_id": ["E001", "E002", "E003", "E002", ""],
        "birth_date": ["19850315", "1990-05-20", "19920708", "19880101", "19950101"],
        "hire_date": ["20100101", "20150601", "20200101", "20120101", "20180101"],
        "gender": ["M", "F", "M", "X", "F"]
    })
    
    # 샘플 규칙
    sample_rules = [
        ValidationRule(
            rule_id="RULE_001",
            field_name="employee_id",
            rule_type="required",
            parameters={},
            error_message_template="사번이 비어있습니다.",
            source=RuleSource(
                original_text="사번: 공백 없음",
                row_number="2"
            ),
            ai_interpretation_summary="사번 필수",
            confidence_score=0.99
        ),
        ValidationRule(
            rule_id="RULE_002",
            field_name="employee_id",
            rule_type="no_duplicates",
            parameters={},
            error_message_template="사번이 중복되었습니다.",
            source=RuleSource(
                original_text="사번: 중복 없음",
                row_number="2"
            ),
            ai_interpretation_summary="사번 고유",
            confidence_score=0.99
        ),
        ValidationRule(
            rule_id="RULE_003",
            field_name="birth_date",
            rule_type="format",
            parameters={"format": "YYYYMMDD", "regex": "^[0-9]{8}$"},
            error_message_template="생년월일 형식이 잘못되었습니다.",
            source=RuleSource(
                original_text="생년월일: YYYYMMDD",
                row_number="3"
            ),
            ai_interpretation_summary="날짜 형식",
            confidence_score=0.99
        ),
        ValidationRule(
            rule_id="RULE_004",
            field_name="gender",
            rule_type="format",
            parameters={"allowed_values": ["M", "F", "남", "여"]},
            error_message_template="성별 값이 올바르지 않습니다.",
            source=RuleSource(
                original_text="성별: M/F/남/여",
                row_number="4"
            ),
            ai_interpretation_summary="성별 허용값",
            confidence_score=0.99
        )
    ]
    
    # 검증 실행
    result = validate_data(sample_data, sample_rules)
    print(json.dumps(result.dict(), indent=2, ensure_ascii=False, default=str))
