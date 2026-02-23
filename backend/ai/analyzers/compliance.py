"""
K-IFRS Compliance Mixin
========================
K-IFRS 1019 준수 여부 검사 및 계리적 가정 합리성 검토
"""

from typing import Dict, List, Any

from utils.logger import get_logger

logger = get_logger("ai.analyzers.compliance")


class ComplianceMixin:
    """
    K-IFRS 1019 컴플라이언스 어드바이저 메서드 모음 (Mixin)
    """

    async def check_kifrs_compliance(
        self,
        sheet_data: Dict[str, Any],
        column_names: Dict[str, List[str]],
        provider: str = None
    ) -> Dict[str, Any]:
        """
        K-IFRS 1019 준수 여부를 검사합니다.

        DBO 계산에 필수인 항목 누락 탐지 및 계리적 가정의 합리성을 검토합니다.

        Args:
            sheet_data: 시트별 DataFrame
            column_names: 시트별 컬럼명
            provider: AI provider

        Returns:
            Dict: compliance_items, overall_score, recommendations
        """
        import pandas as pd

        compliance_items = []

        # K-IFRS 1019 필수 공시 항목 체크리스트
        required_fields = {
            "사번/사원코드": {
                "keywords": ["사번", "사원번호", "employee_id", "emp_id", "코드"],
                "category": "기본정보",
                "importance": "high",
                "description": "DBO 계산 대상 식별을 위한 고유 식별자"
            },
            "생년월일": {
                "keywords": ["생년월일", "생일", "birth_date", "birth"],
                "category": "기본정보",
                "importance": "high",
                "description": "사망률 테이블 적용 및 연령 산출에 필수"
            },
            "입사일": {
                "keywords": ["입사일", "입사일자", "hire_date", "hire"],
                "category": "기본정보",
                "importance": "high",
                "description": "근속연수 산출 및 퇴직금 적립 기간 산정에 필수"
            },
            "퇴직일": {
                "keywords": ["퇴직일", "퇴사일", "termination", "退職"],
                "category": "기본정보",
                "importance": "medium",
                "description": "퇴직자 구분 및 DBO 제외 판단에 필요"
            },
            "평균임금/급여": {
                "keywords": ["평균임금", "급여", "연봉", "salary", "wage", "임금"],
                "category": "보수정보",
                "importance": "high",
                "description": "퇴직급여 산정의 핵심 변수 (K-IFRS 1019.67)"
            },
            "성별": {
                "keywords": ["성별", "gender", "sex"],
                "category": "기본정보",
                "importance": "medium",
                "description": "성별 사망률 테이블 적용에 필요"
            },
            "제도유형": {
                "keywords": ["제도유형", "제도", "plan_type", "plan", "DB", "DC"],
                "category": "제도정보",
                "importance": "high",
                "description": "DB/DC 구분에 따른 DBO 산출 범위 결정"
            },
            "지급배수/지급률": {
                "keywords": ["지급배수", "지급률", "payment_rate", "배수"],
                "category": "제도정보",
                "importance": "medium",
                "description": "퇴직급여 산정 시 적용되는 지급 배수"
            },
            "평가기준일": {
                "keywords": ["평가기준일", "기준일", "evaluation_date", "valuation"],
                "category": "평가정보",
                "importance": "high",
                "description": "DBO 평가 시점 (보고기간 말일 기준, K-IFRS 1019.70)"
            }
        }

        # 모든 시트의 컬럼을 수집
        all_columns = set()
        for sheet, cols in column_names.items():
            all_columns.update([str(c).lower() for c in cols])

        met_count = 0
        total_count = len(required_fields)

        for field_label, field_info in required_fields.items():
            found = False
            found_column = None
            for col in all_columns:
                for kw in field_info["keywords"]:
                    if kw.lower() in col:
                        found = True
                        found_column = col
                        break
                if found:
                    break

            status = "met" if found else "not_met"
            if found:
                met_count += 1

            compliance_items.append({
                "field": field_label,
                "category": field_info["category"],
                "importance": field_info["importance"],
                "status": status,
                "found_column": found_column,
                "description": field_info["description"],
                "kifrs_reference": "K-IFRS 1019"
            })

        # 계리적 가정 합리성 검증
        actuarial_checks = self._check_actuarial_assumptions(sheet_data, column_names)
        compliance_items.extend(actuarial_checks)

        # 전체 점수 계산
        total_items = len(compliance_items)
        met_items = sum(1 for item in compliance_items if item["status"] == "met")
        caution_items = sum(1 for item in compliance_items if item["status"] == "caution")
        not_met_items = sum(1 for item in compliance_items if item["status"] == "not_met")

        overall_score = round(met_items / total_items * 100, 1) if total_items > 0 else 0

        # 권고사항 생성
        recommendations = []
        for item in compliance_items:
            if item["status"] == "not_met" and item["importance"] == "high":
                recommendations.append(f"[필수] {item['field']}: {item['description']}")
            elif item["status"] == "not_met" and item["importance"] == "medium":
                recommendations.append(f"[권장] {item['field']}: {item['description']}")
            elif item["status"] == "caution":
                recommendations.append(f"[주의] {item['field']}: {item.get('caution_reason', '')}")

        return {
            "overall_score": overall_score,
            "total_items": total_items,
            "met_count": met_items,
            "caution_count": caution_items,
            "not_met_count": not_met_items,
            "compliance_items": compliance_items,
            "recommendations": recommendations,
            "summary": f"K-IFRS 1019 준수율: {overall_score}% ({met_items}/{total_items} 항목 충족)"
        }

    def _check_actuarial_assumptions(
        self,
        sheet_data: Dict[str, Any],
        column_names: Dict[str, List[str]]
    ) -> List[Dict[str, Any]]:
        """계리적 가정의 합리성 검증"""
        import pandas as pd
        checks = []

        for sheet_name, df in sheet_data.items():
            if not isinstance(df, pd.DataFrame):
                continue

            cols = {str(c).lower(): str(c) for c in df.columns}

            # 할인율 검증 (보통 1%~10% 범위)
            for kw in ["할인율", "discount_rate", "할인"]:
                for col_lower, col_original in cols.items():
                    if kw in col_lower:
                        numeric_data = pd.to_numeric(df[col_original], errors='coerce').dropna()
                        if len(numeric_data) > 0:
                            avg_val = float(numeric_data.mean())
                            if avg_val < 0.01 or avg_val > 0.15:
                                checks.append({
                                    "field": "할인율",
                                    "category": "계리적 가정",
                                    "importance": "high",
                                    "status": "caution",
                                    "found_column": col_original,
                                    "description": f"할인율 평균 {avg_val:.2%}은 일반적 범위(1%~10%)에서 벗어납니다.",
                                    "caution_reason": f"할인율 평균: {avg_val:.2%}",
                                    "kifrs_reference": "K-IFRS 1019.83-86"
                                })
                            else:
                                checks.append({
                                    "field": "할인율",
                                    "category": "계리적 가정",
                                    "importance": "high",
                                    "status": "met",
                                    "found_column": col_original,
                                    "description": f"할인율 평균 {avg_val:.2%}은 적정 범위입니다.",
                                    "kifrs_reference": "K-IFRS 1019.83-86"
                                })

            # 급여상승률 검증 (보통 2%~8% 범위)
            for kw in ["급여상승률", "salary_increase", "임금상승"]:
                for col_lower, col_original in cols.items():
                    if kw in col_lower:
                        numeric_data = pd.to_numeric(df[col_original], errors='coerce').dropna()
                        if len(numeric_data) > 0:
                            avg_val = float(numeric_data.mean())
                            if avg_val < 0.01 or avg_val > 0.12:
                                checks.append({
                                    "field": "급여상승률",
                                    "category": "계리적 가정",
                                    "importance": "high",
                                    "status": "caution",
                                    "found_column": col_original,
                                    "description": f"급여상승률 평균 {avg_val:.2%}은 일반적 범위(2%~8%)에서 벗어납니다.",
                                    "caution_reason": f"급여상승률 평균: {avg_val:.2%}",
                                    "kifrs_reference": "K-IFRS 1019.87"
                                })
                            else:
                                checks.append({
                                    "field": "급여상승률",
                                    "category": "계리적 가정",
                                    "importance": "high",
                                    "status": "met",
                                    "found_column": col_original,
                                    "description": f"급여상승률 평균 {avg_val:.2%}은 적정 범위입니다.",
                                    "kifrs_reference": "K-IFRS 1019.87"
                                })

        return checks
