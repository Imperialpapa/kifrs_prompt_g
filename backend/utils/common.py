"""
Common Utility Functions
========================
Shared helper functions for the DBO Validation System
"""

from typing import List, Dict, Any, Set
from collections import defaultdict
import pandas as pd
import numpy as np
from models import ValidationErrorGroup

def convert_numpy_types(obj):
    """
    현장 투입용 고성능 데이터 변환기.
    Numpy, Pandas 타입을 표준 Python 타입으로 변환하고 JSON 비호환 값(NaN, Inf)을 처리합니다.
    """
    if obj is None:
        return None
        
    if isinstance(obj, dict):
        return {str(k): convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple, set)):
        return [convert_numpy_types(i) for i in obj]
    elif isinstance(obj, (np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.float64, np.float32, np.float16, float)):
        val = float(obj)
        if np.isnan(val) or np.isinf(val):
            return None
        return val
    elif isinstance(obj, (np.bool_)):
        return bool(obj)
    elif pd.isna(obj):
        return None
    
    # 그 외 타입은 문자열로 변환하여 안전하게 반환
    if hasattr(obj, 'isoformat'): # datetime 등 처리
        return obj.isoformat()
        
    return obj

def filter_garbage_rows(df: pd.DataFrame) -> pd.DataFrame:
    """
    유효하지 않은 행(Garbage Rows) 필터링.
    사번/입사일이 모두 비어있는 행을 주석/메모로 간주하여 제거합니다.

    Args:
        df: 원본 DataFrame

    Returns:
        pd.DataFrame: 필터링된 DataFrame
    """
    id_keywords = ['사번', '사원번호', 'employee_id', 'emp_id', 'id', '코드', 'code']
    date_keywords = ['입사일', '입사일자', 'hire_date', 'hire_dt']

    df_cols_lower = {str(col).lower(): col for col in df.columns}

    id_col = None
    for kw in id_keywords:
        for col_lower, original in df_cols_lower.items():
            if kw in col_lower:
                id_col = original
                break
        if id_col:
            break

    date_col = None
    for kw in date_keywords:
        for col_lower, original in df_cols_lower.items():
            if kw in col_lower:
                date_col = original
                break
        if date_col:
            break

    def is_row_empty(series):
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

    return df


def group_errors(errors: list) -> List[ValidationErrorGroup]:
    """
    동일한 인지 내용을 그룹화하여 집계

    Args:
        errors: ValidationError 리스트

    Returns:
        List[ValidationErrorGroup]: 그룹화된 인지 목록
    """
    # (시트, 컬럼, 규칙ID, 메시지)를 키로 그룹화
    groups = defaultdict(list)

    for error in errors:
        key = (
            error.sheet or "",
            error.column,
            error.rule_id,
            error.message
        )
        groups[key].append(error)

    # ValidationErrorGroup 객체 생성
    error_groups = []
    for (sheet, column, rule_id, message), error_list in groups.items():
        # 행 번호 수집
        affected_rows = [e.row for e in error_list]

        # 샘플 값 수집 (최대 3개, 중복 제거)
        sample_values = []
        seen_values = set()
        for e in error_list:
            val_str = str(e.actual_value)
            if val_str not in seen_values and len(sample_values) < 3:
                sample_values.append(e.actual_value)
                seen_values.add(val_str)

        error_group = ValidationErrorGroup(
            sheet=sheet,
            column=column,
            rule_id=rule_id,
            message=message,
            affected_rows=sorted(affected_rows),
            count=len(error_list),
            sample_values=sample_values,
            expected=error_list[0].expected if error_list else None,
            source_rule=error_list[0].source_rule if error_list else ""
        )
        error_groups.append(error_group)

    # 인지 개수 많은 순으로 정렬
    error_groups.sort(key=lambda x: x.count, reverse=True)

    return error_groups
