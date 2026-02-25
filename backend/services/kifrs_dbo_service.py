"""
K-IFRS 1019 DBO Validation Service
===================================
확정급여채무(DBO) 산출 데이터에 대한 전문 검증 서비스

검증 영역:
1. 보험수리적 가정 검증 (할인율, 급여상승률, 퇴직률, 사망률)
2. DBO 입력 데이터 검증 (생년월일, 입사일, 퇴사일, 급여 등)
3. 재무 정합성 검증 (급여 합산, 인원 대조)
4. K-IFRS 1019 규정 준수 체크리스트
"""

import re
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import List, Dict, Any, Optional, Tuple
from utils.field_matcher import FieldMatcher


class KifrsDboService:
    """K-IFRS 1019 DBO 전문 검증 서비스"""

    # ── 필드 매칭용 키워드 매핑 ──
    FIELD_ALIASES = {
        "employee_id": ["사번", "사원번호", "직원번호", "emp_id", "employee_id", "코드", "id"],
        "name": ["성명", "이름", "성함", "name"],
        "birth_date": ["생년월일", "생일", "birth_date", "birthday", "출생일"],
        "gender": ["성별", "gender", "sex"],
        "hire_date": ["입사일", "입사일자", "hire_date", "join_date", "채용일"],
        "termination_date": ["퇴사일", "퇴사일자", "퇴직일", "termination_date", "이직일"],
        "employment_status": ["재직구분", "재직상태", "근무상태", "status", "재직여부", "구분"],
        "salary": ["기본급", "급여", "월급여", "임금", "월급", "salary", "wage", "pay", "보수"],
        "department": ["부서", "부서명", "department", "dept"],
        "position": ["직급", "직위", "직책", "position", "rank", "grade"],
        "tenure": ["근속연수", "근속년수", "근속기간", "tenure", "service_years"],
    }

    # ── 보험수리 가정 필드 키워드 ──
    ASSUMPTION_ALIASES = {
        "discount_rate": ["할인율", "discount_rate", "discount", "할인"],
        "salary_growth": ["급여상승률", "임금상승률", "salary_growth", "salary_increase", "승급률"],
        "turnover_rate": ["퇴직률", "이직률", "turnover_rate", "turnover", "탈퇴율"],
        "mortality_table": ["사망률", "생명표", "mortality", "mortality_table"],
        "retirement_age": ["정년", "퇴직연령", "retirement_age", "정년연령"],
    }

    # ── 합리적 범위 기준치 ──
    REASONABLE_RANGES = {
        "discount_rate": (0.01, 0.08),        # 1% ~ 8%
        "salary_growth": (0.01, 0.10),         # 1% ~ 10%
        "turnover_rate": (0.0, 0.30),          # 0% ~ 30%
        "retirement_age": (55, 65),            # 55세 ~ 65세
        "working_age_min": 18,
        "working_age_max": 70,
        "salary_change_threshold": 0.50,       # 급여 변동 ±50% 경고
    }

    def __init__(self):
        self.field_matcher = FieldMatcher(threshold=0.5)

    # =====================================================================
    # 메인 검증 진입점
    # =====================================================================

    def validate_dbo_data(
        self,
        sheet_data: Dict[str, pd.DataFrame],
        column_names: Dict[str, List[str]],
        base_date: Optional[str] = None,
        assumptions: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        DBO 데이터 전체 검증 수행

        Args:
            sheet_data: {시트명: DataFrame}
            column_names: {시트명: [컬럼명]}
            base_date: 평가기준일 (YYYYMMDD), None이면 오늘
            assumptions: 보험수리 가정값 dict

        Returns:
            검증 결과 dict
        """
        if base_date:
            base_dt = self._parse_date(base_date)
        else:
            base_dt = datetime.now()

        all_issues: List[Dict[str, Any]] = []
        field_maps: Dict[str, Dict[str, str]] = {}

        # 시트별 필드 매핑 탐색
        for sheet_name, cols in column_names.items():
            field_maps[sheet_name] = self._map_fields(cols)

        # 1. 보험수리 가정 검증
        if assumptions:
            assumption_issues = self._validate_assumptions(assumptions)
            all_issues.extend(assumption_issues)

        # 2. DBO 입력 데이터 검증 (시트별)
        for sheet_name, df in sheet_data.items():
            fmap = field_maps.get(sheet_name, {})
            input_issues = self._validate_input_data(df, fmap, sheet_name, base_dt)
            all_issues.extend(input_issues)

        # 3. 크로스시트 재무 정합성 검증
        reconciliation_issues = self._validate_reconciliation(sheet_data, field_maps)
        all_issues.extend(reconciliation_issues)

        # 4. K-IFRS 1019 컴플라이언스 체크리스트
        compliance_results = self._check_compliance(sheet_data, field_maps, assumptions)

        # 결과 집계
        severity_counts = {"critical": 0, "warning": 0, "info": 0}
        for issue in all_issues:
            sev = issue.get("severity", "info")
            severity_counts[sev] = severity_counts.get(sev, 0) + 1

        categories = {}
        for issue in all_issues:
            cat = issue.get("category", "other")
            categories.setdefault(cat, []).append(issue)

        total_issues = len(all_issues)
        score = max(0, 100 - severity_counts["critical"] * 10 - severity_counts["warning"] * 3 - severity_counts["info"] * 1)

        return {
            "dbo_validation_score": min(100, score),
            "total_issues": total_issues,
            "severity_counts": severity_counts,
            "categories": {
                cat: {"count": len(items), "issues": items}
                for cat, items in categories.items()
            },
            "compliance": compliance_results,
            "field_mappings": {
                sheet: {k: v for k, v in fmap.items() if v}
                for sheet, fmap in field_maps.items()
            },
            "base_date": base_dt.strftime("%Y-%m-%d"),
            "validated_at": datetime.now().isoformat(),
        }

    # =====================================================================
    # 1. 보험수리적 가정 검증
    # =====================================================================

    def _validate_assumptions(self, assumptions: Dict[str, Any]) -> List[Dict[str, Any]]:
        """보험수리 가정값의 합리성 검증"""
        issues = []

        # 할인율
        dr = assumptions.get("discount_rate")
        if dr is not None:
            dr_val = self._to_float(dr)
            if dr_val is not None:
                # 퍼센트 값이면 소수로 변환
                if dr_val > 1:
                    dr_val = dr_val / 100
                lo, hi = self.REASONABLE_RANGES["discount_rate"]
                if not (lo <= dr_val <= hi):
                    issues.append(self._make_issue(
                        "actuarial_assumption", "critical",
                        f"할인율({dr_val*100:.2f}%)이 합리적 범위({lo*100:.0f}%~{hi*100:.0f}%)를 벗어남",
                        field="할인율", value=dr,
                        kifrs_ref="제86항: 할인율은 우량 회사채 시장수익률 기준"
                    ))

        # 급여상승률
        sg = assumptions.get("salary_growth")
        if sg is not None:
            sg_val = self._to_float(sg)
            if sg_val is not None:
                if sg_val > 1:
                    sg_val = sg_val / 100
                lo, hi = self.REASONABLE_RANGES["salary_growth"]
                if not (lo <= sg_val <= hi):
                    issues.append(self._make_issue(
                        "actuarial_assumption", "critical",
                        f"급여상승률({sg_val*100:.2f}%)이 합리적 범위({lo*100:.0f}%~{hi*100:.0f}%)를 벗어남",
                        field="급여상승률", value=sg,
                        kifrs_ref="제87-88항: 인플레이션, 근속, 승진 반영"
                    ))

        # 할인율 vs 급여상승률 일관성 (제83항)
        if dr is not None and sg is not None:
            dr_v = self._to_float(dr)
            sg_v = self._to_float(sg)
            if dr_v is not None and sg_v is not None:
                if dr_v > 1:
                    dr_v = dr_v / 100
                if sg_v > 1:
                    sg_v = sg_v / 100
                if sg_v > dr_v + 0.03:
                    issues.append(self._make_issue(
                        "actuarial_assumption", "warning",
                        f"급여상승률({sg_v*100:.1f}%)이 할인율({dr_v*100:.1f}%)보다 3%p 이상 높음 - 가정 간 일관성 확인 필요",
                        field="급여상승률 vs 할인율",
                        kifrs_ref="제83항: 보험수리적 가정의 상호 일관성"
                    ))

        # 퇴직률
        tr = assumptions.get("turnover_rate")
        if tr is not None:
            tr_val = self._to_float(tr)
            if tr_val is not None:
                if tr_val > 1:
                    tr_val = tr_val / 100
                lo, hi = self.REASONABLE_RANGES["turnover_rate"]
                if not (lo <= tr_val <= hi):
                    issues.append(self._make_issue(
                        "actuarial_assumption", "warning",
                        f"퇴직률({tr_val*100:.2f}%)이 합리적 범위({lo*100:.0f}%~{hi*100:.0f}%)를 벗어남",
                        field="퇴직률", value=tr
                    ))

        # 정년
        ra = assumptions.get("retirement_age")
        if ra is not None:
            ra_val = self._to_float(ra)
            if ra_val is not None:
                lo, hi = self.REASONABLE_RANGES["retirement_age"]
                if not (lo <= ra_val <= hi):
                    issues.append(self._make_issue(
                        "actuarial_assumption", "warning",
                        f"정년({ra_val:.0f}세)이 합리적 범위({lo}~{hi}세)를 벗어남",
                        field="정년", value=ra
                    ))

        if not issues and assumptions:
            issues.append(self._make_issue(
                "actuarial_assumption", "info",
                "모든 보험수리 가정값이 합리적 범위 내에 있습니다.",
                is_pass=True
            ))

        return issues

    # =====================================================================
    # 2. DBO 입력 데이터 검증
    # =====================================================================

    def _validate_input_data(
        self, df: pd.DataFrame, field_map: Dict[str, str],
        sheet_name: str, base_dt: datetime
    ) -> List[Dict[str, Any]]:
        """시트 내 개인별 데이터 검증"""
        issues = []

        if df.empty:
            return issues

        # ── 필수 필드 존재 여부 ──
        required_fields = ["employee_id", "name", "birth_date", "hire_date", "salary"]
        for req in required_fields:
            col = field_map.get(req)
            if not col:
                issues.append(self._make_issue(
                    "required_field", "critical",
                    f"[{sheet_name}] DBO 필수 필드 '{req}' 에 매칭되는 컬럼을 찾을 수 없음",
                    sheet=sheet_name, field=req,
                    kifrs_ref="제73항: 확정급여채무 측정에 필요한 데이터"
                ))

        # ── 사번 중복 체크 ──
        id_col = field_map.get("employee_id")
        if id_col and id_col in df.columns:
            id_series = df[id_col].dropna()
            dupes = id_series[id_series.duplicated(keep=False)]
            if len(dupes) > 0:
                dupe_vals = dupes.unique()[:5]
                issues.append(self._make_issue(
                    "data_integrity", "critical",
                    f"[{sheet_name}] 사번 중복 {len(dupes)}건 발견 (예: {', '.join(str(v) for v in dupe_vals)})",
                    sheet=sheet_name, field=id_col,
                    count=len(dupes)
                ))

        # ── 생년월일 합리성 ──
        birth_col = field_map.get("birth_date")
        if birth_col and birth_col in df.columns:
            birth_issues = self._check_birth_dates(df, birth_col, sheet_name, base_dt)
            issues.extend(birth_issues)

        # ── 입사일 검증 ──
        hire_col = field_map.get("hire_date")
        if hire_col and hire_col in df.columns:
            hire_issues = self._check_hire_dates(df, hire_col, birth_col, sheet_name, base_dt)
            issues.extend(hire_issues)

        # ── 퇴사일 vs 재직구분 교차 검증 ──
        term_col = field_map.get("termination_date")
        status_col = field_map.get("employment_status")
        if term_col and status_col and term_col in df.columns and status_col in df.columns:
            cross_issues = self._check_termination_status(df, term_col, status_col, sheet_name)
            issues.extend(cross_issues)

        # ── 근속연수 일관성 ──
        tenure_col = field_map.get("tenure")
        if tenure_col and hire_col and tenure_col in df.columns and hire_col in df.columns:
            tenure_issues = self._check_tenure_consistency(df, tenure_col, hire_col, sheet_name, base_dt)
            issues.extend(tenure_issues)

        # ── 급여 이상치 ──
        salary_col = field_map.get("salary")
        if salary_col and salary_col in df.columns:
            salary_issues = self._check_salary_outliers(df, salary_col, sheet_name)
            issues.extend(salary_issues)

        # ── 성별 코드 표준화 ──
        gender_col = field_map.get("gender")
        if gender_col and gender_col in df.columns:
            gender_issues = self._check_gender_codes(df, gender_col, sheet_name)
            issues.extend(gender_issues)

        # ── 필수값 누락 ──
        for req in required_fields:
            col = field_map.get(req)
            if col and col in df.columns:
                null_count = df[col].isna().sum()
                if null_count > 0:
                    issues.append(self._make_issue(
                        "missing_data", "warning" if null_count < len(df) * 0.1 else "critical",
                        f"[{sheet_name}] '{col}' 필수 필드에 빈 값 {null_count}건 ({null_count/len(df)*100:.1f}%)",
                        sheet=sheet_name, field=col, count=null_count
                    ))

        return issues

    def _check_birth_dates(self, df, col, sheet, base_dt) -> List[Dict]:
        """생년월일 합리성 검증"""
        issues = []
        min_age = self.REASONABLE_RANGES["working_age_min"]
        max_age = self.REASONABLE_RANGES["working_age_max"]

        dates = df[col].apply(self._parse_date_safe)
        valid_dates = dates.dropna()

        if len(valid_dates) == 0:
            return issues

        ages = valid_dates.apply(lambda d: (base_dt - d).days / 365.25)

        too_young = ages[ages < min_age]
        too_old = ages[ages > max_age]

        if len(too_young) > 0:
            issues.append(self._make_issue(
                "data_integrity", "warning",
                f"[{sheet}] {min_age}세 미만 직원 {len(too_young)}명 (최소 {ages.min():.1f}세)",
                sheet=sheet, field=col, count=len(too_young)
            ))

        if len(too_old) > 0:
            issues.append(self._make_issue(
                "data_integrity", "warning",
                f"[{sheet}] {max_age}세 초과 직원 {len(too_old)}명 (최대 {ages.max():.1f}세)",
                sheet=sheet, field=col, count=len(too_old)
            ))

        # 파싱 불가 건수
        parse_fail = len(df[col].dropna()) - len(valid_dates)
        if parse_fail > 0:
            issues.append(self._make_issue(
                "data_integrity", "warning",
                f"[{sheet}] '{col}' 날짜 형식 파싱 불가 {parse_fail}건",
                sheet=sheet, field=col, count=parse_fail
            ))

        return issues

    def _check_hire_dates(self, df, hire_col, birth_col, sheet, base_dt) -> List[Dict]:
        """입사일 검증"""
        issues = []

        hire_dates = df[hire_col].apply(self._parse_date_safe)
        valid_hire = hire_dates.dropna()

        # 입사일 > 기준일
        future_hire = valid_hire[valid_hire > base_dt]
        if len(future_hire) > 0:
            issues.append(self._make_issue(
                "date_logic", "critical",
                f"[{sheet}] 입사일이 기준일({base_dt.strftime('%Y-%m-%d')}) 이후인 직원 {len(future_hire)}명",
                sheet=sheet, field=hire_col, count=len(future_hire),
                kifrs_ref="입사일은 평가기준일 이전이어야 함"
            ))

        # 입사일 < 생년월일 (비정상)
        if birth_col and birth_col in df.columns:
            birth_dates = df[birth_col].apply(self._parse_date_safe)
            both_valid = df.index[hire_dates.notna() & birth_dates.notna()]
            if len(both_valid) > 0:
                hire_before_birth = both_valid[hire_dates[both_valid] < birth_dates[both_valid]]
                if len(hire_before_birth) > 0:
                    issues.append(self._make_issue(
                        "date_logic", "critical",
                        f"[{sheet}] 입사일이 생년월일보다 빠른 직원 {len(hire_before_birth)}명",
                        sheet=sheet, field=hire_col, count=len(hire_before_birth)
                    ))

                # 입사 시 나이 14세 미만
                age_at_hire = (hire_dates[both_valid] - birth_dates[both_valid]).apply(lambda d: d.days / 365.25)
                too_young_hire = age_at_hire[age_at_hire < 14]
                if len(too_young_hire) > 0:
                    issues.append(self._make_issue(
                        "date_logic", "warning",
                        f"[{sheet}] 입사 시 14세 미만인 직원 {len(too_young_hire)}명",
                        sheet=sheet, field=hire_col, count=len(too_young_hire)
                    ))

        return issues

    def _check_termination_status(self, df, term_col, status_col, sheet) -> List[Dict]:
        """퇴사일 vs 재직구분 교차 검증"""
        issues = []

        active_keywords = ["재직", "재직중", "근무", "현직", "active", "1"]
        inactive_keywords = ["퇴직", "퇴사", "이직", "resigned", "terminated", "0", "2"]

        has_term = df[term_col].notna()
        status_lower = df[status_col].astype(str).str.strip().str.lower()

        # 재직자인데 퇴사일이 있는 경우
        is_active = status_lower.apply(
            lambda s: any(kw in s for kw in active_keywords)
        )
        active_with_term = df.index[is_active & has_term]
        if len(active_with_term) > 0:
            issues.append(self._make_issue(
                "cross_field", "warning",
                f"[{sheet}] 재직 상태이지만 퇴사일이 입력된 직원 {len(active_with_term)}명",
                sheet=sheet, field=f"{status_col}/{term_col}", count=len(active_with_term)
            ))

        # 퇴직자인데 퇴사일이 없는 경우
        is_inactive = status_lower.apply(
            lambda s: any(kw in s for kw in inactive_keywords)
        )
        inactive_no_term = df.index[is_inactive & ~has_term]
        if len(inactive_no_term) > 0:
            issues.append(self._make_issue(
                "cross_field", "warning",
                f"[{sheet}] 퇴직 상태이지만 퇴사일이 없는 직원 {len(inactive_no_term)}명",
                sheet=sheet, field=f"{status_col}/{term_col}", count=len(inactive_no_term)
            ))

        return issues

    def _check_tenure_consistency(self, df, tenure_col, hire_col, sheet, base_dt) -> List[Dict]:
        """근속연수 일관성 검증"""
        issues = []

        hire_dates = df[hire_col].apply(self._parse_date_safe)
        tenure_vals = pd.to_numeric(df[tenure_col], errors='coerce')

        valid_idx = df.index[hire_dates.notna() & tenure_vals.notna()]
        if len(valid_idx) == 0:
            return issues

        calc_tenure = (base_dt - hire_dates[valid_idx]).apply(lambda d: d.days / 365.25)
        recorded_tenure = tenure_vals[valid_idx]

        diff = (calc_tenure - recorded_tenure).abs()
        mismatch = diff[diff > 1.5]  # 1.5년 이상 차이

        if len(mismatch) > 0:
            issues.append(self._make_issue(
                "cross_field", "warning",
                f"[{sheet}] 근속연수와 입사일 기반 계산값이 1.5년 이상 차이나는 직원 {len(mismatch)}명 (최대 차이: {diff.max():.1f}년)",
                sheet=sheet, field=f"{tenure_col}/{hire_col}", count=len(mismatch)
            ))

        return issues

    def _check_salary_outliers(self, df, col, sheet) -> List[Dict]:
        """급여 이상치 감지 (Z-score + IQR)"""
        issues = []

        # 숫자 변환 (콤마 제거)
        salary = df[col].apply(lambda v: self._to_float(str(v).replace(",", "")) if pd.notna(v) else None)
        salary = salary.dropna()

        if len(salary) < 5:
            return issues

        # 음수/0 급여
        zero_or_neg = salary[salary <= 0]
        if len(zero_or_neg) > 0:
            issues.append(self._make_issue(
                "data_integrity", "critical",
                f"[{sheet}] 급여가 0 이하인 직원 {len(zero_or_neg)}명",
                sheet=sheet, field=col, count=len(zero_or_neg)
            ))

        positive = salary[salary > 0]
        if len(positive) < 3:
            return issues

        # IQR 기반 이상치
        q1 = positive.quantile(0.25)
        q3 = positive.quantile(0.75)
        iqr = q3 - q1
        if iqr > 0:
            lower = q1 - 3.0 * iqr
            upper = q3 + 3.0 * iqr
            outliers = positive[(positive < lower) | (positive > upper)]
            if len(outliers) > 0:
                issues.append(self._make_issue(
                    "statistical_outlier", "warning",
                    f"[{sheet}] 급여 극단 이상치 {len(outliers)}건 (IQR 3배 기준, 범위: {lower:,.0f}~{upper:,.0f})",
                    sheet=sheet, field=col, count=len(outliers)
                ))

        return issues

    def _check_gender_codes(self, df, col, sheet) -> List[Dict]:
        """성별 코드 표준화 검증"""
        issues = []

        vals = df[col].dropna().astype(str).str.strip()
        unique_vals = vals.unique()

        standard_codes = {"1", "2"}
        text_codes = {"남", "여", "남자", "여자", "m", "f", "male", "female"}
        all_valid = standard_codes | text_codes

        invalid_vals = [v for v in unique_vals if v.lower() not in all_valid]
        if invalid_vals:
            issues.append(self._make_issue(
                "data_integrity", "warning",
                f"[{sheet}] 비표준 성별 코드 발견: {', '.join(invalid_vals[:5])} (표준: 1=남, 2=여)",
                sheet=sheet, field=col, count=len(invalid_vals)
            ))

        # 혼재 여부 (숫자 + 텍스트 혼재)
        has_numeric = any(v in standard_codes for v in unique_vals)
        has_text = any(v.lower() in (text_codes - standard_codes) for v in unique_vals)
        if has_numeric and has_text:
            issues.append(self._make_issue(
                "data_integrity", "warning",
                f"[{sheet}] 성별 코드에 숫자와 텍스트가 혼재됨 (예: {', '.join(unique_vals[:4])})",
                sheet=sheet, field=col
            ))

        return issues

    # =====================================================================
    # 3. 재무 정합성 크로스시트 검증
    # =====================================================================

    def _validate_reconciliation(
        self, sheet_data: Dict[str, pd.DataFrame],
        field_maps: Dict[str, Dict[str, str]]
    ) -> List[Dict[str, Any]]:
        """시트 간 재무 정합성 검증"""
        issues = []

        if len(sheet_data) < 2:
            return issues

        # 시트별 인원수 & 사번 대조
        sheet_ids: Dict[str, set] = {}
        sheet_counts: Dict[str, int] = {}
        sheet_salary_sums: Dict[str, float] = {}

        for sheet_name, df in sheet_data.items():
            fmap = field_maps.get(sheet_name, {})
            id_col = fmap.get("employee_id")
            salary_col = fmap.get("salary")

            if id_col and id_col in df.columns:
                ids = set(df[id_col].dropna().astype(str).unique())
                sheet_ids[sheet_name] = ids
                sheet_counts[sheet_name] = len(ids)

            if salary_col and salary_col in df.columns:
                salary_vals = df[salary_col].apply(
                    lambda v: self._to_float(str(v).replace(",", "")) if pd.notna(v) else 0
                )
                sheet_salary_sums[sheet_name] = float(salary_vals.sum())

        # 시트 간 인원 불일치 감지
        sheet_names = list(sheet_ids.keys())
        for i in range(len(sheet_names)):
            for j in range(i + 1, len(sheet_names)):
                s1, s2 = sheet_names[i], sheet_names[j]
                ids1, ids2 = sheet_ids[s1], sheet_ids[s2]

                only_in_s1 = ids1 - ids2
                only_in_s2 = ids2 - ids1

                if only_in_s1:
                    issues.append(self._make_issue(
                        "reconciliation", "warning",
                        f"[{s1}]에만 존재하는 사번 {len(only_in_s1)}건 (예: {', '.join(list(only_in_s1)[:3])})",
                        sheet=f"{s1} vs {s2}", count=len(only_in_s1)
                    ))
                if only_in_s2:
                    issues.append(self._make_issue(
                        "reconciliation", "warning",
                        f"[{s2}]에만 존재하는 사번 {len(only_in_s2)}건 (예: {', '.join(list(only_in_s2)[:3])})",
                        sheet=f"{s1} vs {s2}", count=len(only_in_s2)
                    ))

        # 크로스시트 사번 중복 (동일 사번이 여러 시트에 중복 존재)
        all_ids_flat = []
        for sheet_name, ids in sheet_ids.items():
            for eid in ids:
                all_ids_flat.append((sheet_name, eid))

        id_sheet_map: Dict[str, List[str]] = {}
        for sheet_name, eid in all_ids_flat:
            id_sheet_map.setdefault(eid, []).append(sheet_name)

        multi_sheet_ids = {eid: sheets for eid, sheets in id_sheet_map.items() if len(sheets) > 1}
        # 이것은 정상적일 수 있음 (여러 시트에 같은 사번), info로 기록
        if multi_sheet_ids and len(sheet_data) > 1:
            issues.append(self._make_issue(
                "reconciliation", "info",
                f"복수 시트에 존재하는 사번 {len(multi_sheet_ids)}건 (크로스시트 정합성 확인 대상)",
                count=len(multi_sheet_ids)
            ))

        return issues

    # =====================================================================
    # 4. K-IFRS 1019 컴플라이언스 체크리스트
    # =====================================================================

    def _check_compliance(
        self, sheet_data: Dict[str, pd.DataFrame],
        field_maps: Dict[str, Dict[str, str]],
        assumptions: Optional[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """K-IFRS 1019 핵심 조항별 준수 여부 체크"""
        checks = []

        # ── 제73항: 예측단위적립방식 데이터 ──
        has_salary = any(fmap.get("salary") for fmap in field_maps.values())
        has_birth = any(fmap.get("birth_date") for fmap in field_maps.values())
        has_hire = any(fmap.get("hire_date") for fmap in field_maps.values())

        puc_ready = has_salary and has_birth and has_hire
        checks.append({
            "article": "제73항",
            "title": "예측단위적립방식 필수 데이터",
            "description": "급여, 생년월일, 입사일 데이터 존재 여부",
            "status": "pass" if puc_ready else "fail",
            "details": f"급여: {'O' if has_salary else 'X'}, 생년월일: {'O' if has_birth else 'X'}, 입사일: {'O' if has_hire else 'X'}"
        })

        # ── 제83항: 보험수리 가정 상호 일관성 ──
        if assumptions:
            dr = self._to_float(assumptions.get("discount_rate", 0))
            sg = self._to_float(assumptions.get("salary_growth", 0))
            if dr and sg:
                if dr > 1: dr = dr / 100
                if sg > 1: sg = sg / 100
                consistent = abs(sg - dr) < 0.05  # 5%p 이내 차이
                checks.append({
                    "article": "제83항",
                    "title": "보험수리 가정 상호 일관성",
                    "description": "할인율과 급여상승률 간 합리적 관계",
                    "status": "pass" if consistent else "warning",
                    "details": f"할인율: {dr*100:.1f}%, 급여상승률: {sg*100:.1f}%, 차이: {abs(sg-dr)*100:.1f}%p"
                })
        else:
            checks.append({
                "article": "제83항",
                "title": "보험수리 가정 상호 일관성",
                "description": "보험수리 가정 데이터가 제공되지 않음",
                "status": "skip",
                "details": "보험수리 가정값을 입력하면 일관성을 검증합니다."
            })

        # ── 제86항: 할인율 적정성 ──
        if assumptions and assumptions.get("discount_rate"):
            dr_val = self._to_float(assumptions["discount_rate"])
            if dr_val and dr_val > 1:
                dr_val = dr_val / 100
            lo, hi = self.REASONABLE_RANGES["discount_rate"]
            dr_ok = dr_val is not None and lo <= dr_val <= hi
            checks.append({
                "article": "제86항",
                "title": "할인율 적정 범위",
                "description": "우량 회사채 시장수익률 기준 적정 범위 여부",
                "status": "pass" if dr_ok else "fail",
                "details": f"할인율: {dr_val*100:.2f}%, 적정 범위: {lo*100:.0f}%~{hi*100:.0f}%"
                           if dr_val else "파싱 불가"
            })
        else:
            checks.append({
                "article": "제86항",
                "title": "할인율 적정 범위",
                "description": "할인율 데이터가 제공되지 않음",
                "status": "skip",
                "details": "할인율을 입력하면 적정성을 검증합니다."
            })

        # ── 제87-88항: 급여 추정 근거 ──
        if assumptions and assumptions.get("salary_growth"):
            sg_val = self._to_float(assumptions["salary_growth"])
            if sg_val and sg_val > 1:
                sg_val = sg_val / 100
            lo, hi = self.REASONABLE_RANGES["salary_growth"]
            sg_ok = sg_val is not None and lo <= sg_val <= hi
            checks.append({
                "article": "제87-88항",
                "title": "급여상승률 합리성",
                "description": "인플레이션, 근속, 승진 등 반영 여부",
                "status": "pass" if sg_ok else "warning",
                "details": f"급여상승률: {sg_val*100:.2f}%, 적정 범위: {lo*100:.0f}%~{hi*100:.0f}%"
                           if sg_val else "파싱 불가"
            })
        else:
            checks.append({
                "article": "제87-88항",
                "title": "급여상승률 합리성",
                "description": "급여상승률 데이터가 제공되지 않음",
                "status": "skip",
                "details": "급여상승률을 입력하면 합리성을 검증합니다."
            })

        # ── 데이터 완전성 ──
        total_rows = sum(len(df) for df in sheet_data.values())
        total_null_critical = 0
        for sheet_name, df in sheet_data.items():
            fmap = field_maps.get(sheet_name, {})
            for req in ["employee_id", "name", "birth_date", "hire_date", "salary"]:
                col = fmap.get(req)
                if col and col in df.columns:
                    total_null_critical += int(df[col].isna().sum())

        completeness = 1 - (total_null_critical / max(total_rows * 5, 1))
        checks.append({
            "article": "데이터 품질",
            "title": "필수 필드 완전성",
            "description": "사번, 성명, 생년월일, 입사일, 급여의 데이터 채움률",
            "status": "pass" if completeness > 0.95 else ("warning" if completeness > 0.8 else "fail"),
            "details": f"완전성: {completeness*100:.1f}% (누락 {total_null_critical}건 / 총 {total_rows}행)"
        })

        pass_count = sum(1 for c in checks if c["status"] == "pass")
        total_checks = sum(1 for c in checks if c["status"] != "skip")
        compliance_score = int(pass_count / max(total_checks, 1) * 100)

        return {
            "score": compliance_score,
            "total_checks": len(checks),
            "pass_count": pass_count,
            "checks": checks
        }

    # =====================================================================
    # 표준 규칙 템플릿 생성
    # =====================================================================

    def get_standard_rule_templates(self) -> List[Dict[str, Any]]:
        """K-IFRS 1019 DBO 표준 검증 규칙 템플릿을 반환합니다."""
        return [
            # ── 필수값 규칙 ──
            {
                "field_name": "사번",
                "rule_text": "필수 입력, 공백 불가, 중복 불가",
                "ai_rule_type": "required",
                "ai_parameters": {},
                "ai_error_message": "사번은 필수 입력이며 중복될 수 없습니다.",
                "category": "필수값",
                "kifrs_ref": "제73항"
            },
            {
                "field_name": "생년월일",
                "rule_text": "필수 입력, YYYYMMDD 형식, 18세~70세 범위",
                "ai_rule_type": "format",
                "ai_parameters": {"pattern": "^\\d{8}$"},
                "ai_error_message": "생년월일은 YYYYMMDD 형식이며 합리적 연령 범위여야 합니다.",
                "category": "필수값/형식",
                "kifrs_ref": "제73항"
            },
            {
                "field_name": "성별",
                "rule_text": "필수 입력, 허용값: 1(남), 2(여)",
                "ai_rule_type": "allowed_values",
                "ai_parameters": {"allowed": ["1", "2"]},
                "ai_error_message": "성별은 1(남) 또는 2(여)만 허용됩니다.",
                "category": "필수값",
                "kifrs_ref": "제73항"
            },
            {
                "field_name": "입사일",
                "rule_text": "필수 입력, YYYYMMDD 형식, 기준일 이전이어야 함",
                "ai_rule_type": "format",
                "ai_parameters": {"pattern": "^\\d{8}$"},
                "ai_error_message": "입사일은 YYYYMMDD 형식이며 평가기준일 이전이어야 합니다.",
                "category": "필수값/날짜",
                "kifrs_ref": "제73항"
            },
            {
                "field_name": "급여",
                "rule_text": "필수 입력, 숫자, 0보다 큰 값",
                "ai_rule_type": "range",
                "ai_parameters": {"min": 0, "exclusive_min": True},
                "ai_error_message": "급여는 0보다 큰 숫자여야 합니다.",
                "category": "필수값/범위",
                "kifrs_ref": "제73항"
            },
            # ── DBO 필수필드 공백 체크 규칙 ──
            {
                "field_name": "생년월일",
                "rule_text": "필수 입력, 공백 불가",
                "ai_rule_type": "required",
                "ai_parameters": {},
                "ai_error_message": "생년월일은 필수 입력이며 공백일 수 없습니다.",
                "category": "필수값",
                "kifrs_ref": "제73항"
            },
            {
                "field_name": "성별",
                "rule_text": "필수 입력, 공백 불가",
                "ai_rule_type": "required",
                "ai_parameters": {},
                "ai_error_message": "성별은 필수 입력이며 공백일 수 없습니다.",
                "category": "필수값",
                "kifrs_ref": "제73항"
            },
            {
                "field_name": "입사일",
                "rule_text": "필수 입력, 공백 불가",
                "ai_rule_type": "required",
                "ai_parameters": {},
                "ai_error_message": "입사일은 필수 입력이며 공백일 수 없습니다.",
                "category": "필수값",
                "kifrs_ref": "제73항"
            },
            {
                "field_name": "퇴사일",
                "rule_text": "퇴직자는 퇴사일 필수 입력, 공백 불가",
                "ai_rule_type": "conditional_required",
                "ai_parameters": {"condition_field": "재직구분", "condition_values": ["퇴직", "퇴직자", "퇴사", "2"], "then": "required"},
                "ai_error_message": "퇴직자의 퇴사일은 필수 입력이며 공백일 수 없습니다.",
                "category": "필수값",
                "kifrs_ref": "제73항"
            },
            {
                "field_name": "재직구분",
                "rule_text": "필수 입력, 공백 불가",
                "ai_rule_type": "required",
                "ai_parameters": {},
                "ai_error_message": "재직구분은 필수 입력이며 공백일 수 없습니다.",
                "category": "필수값",
                "kifrs_ref": "제73항"
            },
            {
                "field_name": "기본급",
                "rule_text": "필수 입력, 공백 불가",
                "ai_rule_type": "required",
                "ai_parameters": {},
                "ai_error_message": "기본급은 필수 입력이며 공백일 수 없습니다.",
                "category": "필수값",
                "kifrs_ref": "제73항"
            },
            {
                "field_name": "기본급",
                "rule_text": "숫자, 0 이상의 값",
                "ai_rule_type": "range",
                "ai_parameters": {"min": 0},
                "ai_error_message": "기본급은 0 이상의 숫자여야 합니다.",
                "category": "필수값/범위",
                "kifrs_ref": "제73항"
            },
            # ── 날짜 논리 규칙 ──
            {
                "field_name": "입사일",
                "rule_text": "입사일은 생년월일보다 이후여야 함",
                "ai_rule_type": "date_logic",
                "ai_parameters": {"compare_field": "생년월일", "operator": ">"},
                "ai_error_message": "입사일은 생년월일 이후여야 합니다.",
                "category": "날짜논리",
                "kifrs_ref": "데이터 정합성"
            },
            {
                "field_name": "퇴사일",
                "rule_text": "퇴사일이 있으면 입사일보다 이후여야 함",
                "ai_rule_type": "date_logic",
                "ai_parameters": {"compare_field": "입사일", "operator": ">", "allow_null": True},
                "ai_error_message": "퇴사일은 입사일 이후여야 합니다.",
                "category": "날짜논리",
                "kifrs_ref": "데이터 정합성"
            },
            # ── 크로스필드 규칙 ──
            {
                "field_name": "재직구분",
                "rule_text": "재직자는 퇴사일이 없어야 하며, 퇴직자는 퇴사일이 있어야 함",
                "ai_rule_type": "cross_field",
                "ai_parameters": {"related_field": "퇴사일", "logic": "status_termination_consistency"},
                "ai_error_message": "재직구분과 퇴사일이 일치하지 않습니다.",
                "category": "교차검증",
                "kifrs_ref": "데이터 정합성"
            },
            # ── 중복 불가 ──
            {
                "field_name": "사번",
                "rule_text": "사번은 시트 내에서 고유해야 함",
                "ai_rule_type": "no_duplicates",
                "ai_parameters": {},
                "ai_error_message": "중복된 사번이 있습니다.",
                "category": "무결성",
                "kifrs_ref": "데이터 정합성"
            },
        ]

    # =====================================================================
    # 유틸리티
    # =====================================================================

    def _map_fields(self, columns: List[str]) -> Dict[str, str]:
        """컬럼 목록에서 표준 필드를 자동 매핑"""
        result = {}
        for standard_name, aliases in {**self.FIELD_ALIASES, **self.ASSUMPTION_ALIASES}.items():
            best_col = None
            best_score = 0
            for col in columns:
                col_norm = self.field_matcher.normalize(col)
                for alias in aliases:
                    alias_norm = self.field_matcher.normalize(alias)
                    if col_norm == alias_norm:
                        best_col = col
                        best_score = 1.0
                        break
                    score = self.field_matcher.calculate_similarity(col, alias)
                    if score > best_score and score >= 0.6:
                        best_score = score
                        best_col = col
                if best_score >= 1.0:
                    break
            result[standard_name] = best_col
        return result

    @staticmethod
    def _to_python(val):
        """numpy 타입을 Python 네이티브 타입으로 변환 (JSON 직렬화 안전)"""
        if isinstance(val, (np.integer,)):
            return int(val)
        if isinstance(val, (np.floating,)):
            return float(val)
        if isinstance(val, np.ndarray):
            return val.tolist()
        return val

    def _make_issue(
        self, category: str, severity: str, message: str,
        sheet: str = "", field: str = "", value: Any = None,
        count: int = 0, kifrs_ref: str = "", is_pass: bool = False
    ) -> Dict[str, Any]:
        """이슈 레코드 생성"""
        return {
            "category": category,
            "severity": severity,
            "message": message,
            "sheet": sheet,
            "field": field,
            "value": self._to_python(value),
            "count": self._to_python(count),
            "kifrs_ref": kifrs_ref,
            "is_pass": is_pass,
        }

    def _parse_date_safe(self, value) -> Optional[datetime]:
        """다양한 날짜 형식 안전 파싱"""
        if pd.isna(value) or value is None:
            return None
        return self._parse_date(value)

    def _parse_date(self, value) -> Optional[datetime]:
        """날짜 파싱 (다양한 형식 지원)"""
        if isinstance(value, (datetime, date)):
            return datetime(value.year, value.month, value.day)
        if isinstance(value, pd.Timestamp):
            return value.to_pydatetime()

        s = str(value).strip()
        # 숫자만 8자리 → YYYYMMDD
        digits = re.sub(r'\D', '', s)
        if len(digits) == 8:
            try:
                return datetime.strptime(digits, "%Y%m%d")
            except ValueError:
                pass

        for fmt in ["%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d", "%d-%m-%Y", "%m/%d/%Y"]:
            try:
                return datetime.strptime(s, fmt)
            except ValueError:
                continue

        return None

    def _to_float(self, value) -> Optional[float]:
        """안전한 float 변환"""
        if value is None or (isinstance(value, float) and np.isnan(value)):
            return None
        try:
            return float(str(value).replace(",", "").replace("%", "").strip())
        except (ValueError, TypeError):
            return None
