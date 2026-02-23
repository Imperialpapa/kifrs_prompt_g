"""
Cross-Field Analysis Mixin
==========================
크로스필드 논리 모순 탐지 (AI 기반 + 로컬 결정론적 체크)
"""

import json
import re
from typing import Dict, List, Any

from utils.logger import get_logger

logger = get_logger("ai.analyzers.cross_field")


class CrossFieldMixin:
    """
    크로스필드 논리 모순 탐지 메서드 모음 (Mixin)
    """

    async def analyze_cross_field(
        self,
        sheet_data_samples: Dict[str, List[Dict[str, Any]]],
        column_names: Dict[str, List[str]],
        provider: str = None
    ) -> Dict[str, Any]:
        """
        크로스필드 논리 모순 탐지 (AI 기반)

        데이터의 필드 간 관계를 AI가 추론하여 논리적 모순을 자동 발견합니다.
        규칙 없이도 입사일/퇴직일, 나이/근속, 성별/출산휴가 등 다중 필드 조합 오류를 탐지합니다.

        Args:
            sheet_data_samples: {시트명: [행 데이터 dict, ...]} (시트별 최대 50행 샘플)
            column_names: {시트명: [컬럼명 리스트]}
            provider: AI 프로바이더

        Returns:
            Dict: {
                "contradictions": [{sheet, rows, fields, description, severity, suggestion}],
                "analysis_summary": str,
                "total_issues": int
            }
        """
        target_provider = (provider or self.default_provider).lower()
        use_cloud = self._check_provider_availability(target_provider)

        if not use_cloud:
            # 로컬 폴백: 결정론적 크로스필드 체크
            return self._local_cross_field_check(sheet_data_samples, column_names)

        try:
            prompt = self._build_cross_field_prompt(sheet_data_samples, column_names)
            ai_response = await self._call_cloud_ai(prompt, target_provider)
            return self._parse_cross_field_response(ai_response)
        except Exception as e:
            logger.error("Cross-field analysis failed (%s): %s", target_provider, e)
            return self._local_cross_field_check(sheet_data_samples, column_names)

    def _build_cross_field_prompt(
        self,
        sheet_data_samples: Dict[str, List[Dict[str, Any]]],
        column_names: Dict[str, List[str]]
    ) -> str:
        """크로스필드 분석 프롬프트 생성"""
        data_description = []
        for sheet_name, samples in sheet_data_samples.items():
            cols = column_names.get(sheet_name, [])
            data_description.append(f"[시트: {sheet_name}] 컬럼: {', '.join(cols)}")
            # 샘플 데이터 (최대 15행)
            for i, row in enumerate(samples[:15]):
                row_str = json.dumps(row, ensure_ascii=False, default=str)
                data_description.append(f"  Row {i+1}: {row_str}")

        data_text = "\n".join(data_description)

        return f"""You are a K-IFRS 1019 Data Quality Expert analyzing employee benefit data.

Analyze the following dataset and find CROSS-FIELD LOGICAL CONTRADICTIONS.
Look for cases where field values are mutually inconsistent.

IMPORTANT DETECTION TARGETS:
1. Date contradictions: hire_date before birth_date, termination before hire, etc.
2. Age vs tenure: impossible combinations (e.g., age 25 but 20 years of service)
3. Status contradictions: terminated status but no termination date, or active with termination date
4. Gender-related: biological impossibilities
5. Salary anomalies: negative values, zero salary for active employees
6. Code consistency: mismatched codes between related fields
7. Any other logical impossibility between two or more fields

DATA:
{data_text}

OUTPUT ONLY the following JSON structure (no markdown, no extra text):
{{
    "contradictions": [
        {{
            "sheet": "시트명",
            "rows": [행번호1, 행번호2],
            "fields": ["필드1", "필드2"],
            "description": "모순 설명 (한국어)",
            "severity": "high|medium|low",
            "suggestion": "수정 제안 (한국어)"
        }}
    ],
    "analysis_summary": "전체 분석 요약 (한국어, 2-3문장)"
}}"""

    def _parse_cross_field_response(self, response: str) -> Dict[str, Any]:
        """AI 크로스필드 분석 결과 파싱"""
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(response)
            contradictions = data.get("contradictions", [])
            return {
                "contradictions": contradictions,
                "analysis_summary": data.get("analysis_summary", "분석 완료"),
                "total_issues": len(contradictions)
            }
        except Exception as e:
            logger.error("Failed to parse cross-field response: %s", e)
            return {"contradictions": [], "analysis_summary": "AI 응답 파싱 실패", "total_issues": 0}

    def _local_cross_field_check(
        self,
        sheet_data_samples: Dict[str, List[Dict[str, Any]]],
        column_names: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """로컬 결정론적 크로스필드 체크 (AI 없이)"""
        contradictions = []

        # 날짜 관련 컬럼 키워드
        date_keywords = {
            "birth": ["생년월일", "birth", "생일"],
            "hire": ["입사일", "입사", "hire", "입사일자"],
            "term": ["퇴사일", "퇴직일", "퇴사", "termination", "퇴직일자", "퇴사일자"],
            "eval": ["평가기준일", "기준일", "evaluation", "산정일"]
        }

        def find_col(cols, keywords):
            for col in cols:
                col_lower = col.lower()
                for kw in keywords:
                    if kw in col_lower or kw in col:
                        return col
            return None

        def parse_date_val(val):
            if not val or str(val).strip() in ('', 'None', 'nan', 'NaT'):
                return None
            s = str(val).strip()
            # YYYYMMDD
            if re.match(r'^(19|20)\d{6}$', s):
                try:
                    return int(s[:4]), int(s[4:6]), int(s[6:8])
                except Exception:
                    return None
            # YYYY-MM-DD or YYYY/MM/DD or YYYY.MM.DD
            m = re.match(r'^(\d{4})[-/.](\d{1,2})[-/.](\d{1,2})', s)
            if m:
                return int(m.group(1)), int(m.group(2)), int(m.group(3))
            # Excel serial number (float)
            try:
                num = float(s)
                if 20000 < num < 60000:
                    from datetime import datetime, timedelta
                    d = datetime(1899, 12, 30) + timedelta(days=int(num))
                    return d.year, d.month, d.day
            except Exception:
                pass
            return None

        def date_tuple_to_days(t):
            if not t:
                return None
            y, m, d = t
            return y * 365 + m * 30 + d  # 간이 비교용

        for sheet_name, samples in sheet_data_samples.items():
            cols = column_names.get(sheet_name, [])

            birth_col = find_col(cols, date_keywords["birth"])
            hire_col = find_col(cols, date_keywords["hire"])
            term_col = find_col(cols, date_keywords["term"])

            for i, row in enumerate(samples):
                row_num = row.get("__row_number__", i + 2)

                birth_val = parse_date_val(row.get(birth_col)) if birth_col else None
                hire_val = parse_date_val(row.get(hire_col)) if hire_col else None
                term_val = parse_date_val(row.get(term_col)) if term_col else None

                birth_days = date_tuple_to_days(birth_val)
                hire_days = date_tuple_to_days(hire_val)
                term_days = date_tuple_to_days(term_val)

                # 1. 입사일 < 생년월일
                if birth_days and hire_days and hire_days < birth_days:
                    contradictions.append({
                        "sheet": sheet_name,
                        "rows": [row_num],
                        "fields": [birth_col, hire_col],
                        "description": f"입사일({row.get(hire_col)})이 생년월일({row.get(birth_col)})보다 이전입니다.",
                        "severity": "high",
                        "suggestion": "입사일 또는 생년월일을 확인하세요."
                    })

                # 2. 퇴사일 < 입사일
                if hire_days and term_days and term_days < hire_days:
                    contradictions.append({
                        "sheet": sheet_name,
                        "rows": [row_num],
                        "fields": [hire_col, term_col],
                        "description": f"퇴사일({row.get(term_col)})이 입사일({row.get(hire_col)})보다 이전입니다.",
                        "severity": "high",
                        "suggestion": "퇴사일 또는 입사일을 확인하세요."
                    })

                # 3. 나이 < 15세에 입사 (비현실적)
                if birth_val and hire_val:
                    age_at_hire = hire_val[0] - birth_val[0]
                    if 0 < age_at_hire < 15:
                        contradictions.append({
                            "sheet": sheet_name,
                            "rows": [row_num],
                            "fields": [birth_col, hire_col],
                            "description": f"입사 시 나이가 {age_at_hire}세로 비현실적입니다.",
                            "severity": "medium",
                            "suggestion": "생년월일 또는 입사일을 확인하세요."
                        })

        summary_parts = []
        if contradictions:
            high = sum(1 for c in contradictions if c["severity"] == "high")
            med = sum(1 for c in contradictions if c["severity"] == "medium")
            summary_parts.append(f"총 {len(contradictions)}건의 논리 모순 발견")
            if high:
                summary_parts.append(f"(심각: {high}건)")
            if med:
                summary_parts.append(f"(주의: {med}건)")
        else:
            summary_parts.append("크로스필드 논리 모순이 발견되지 않았습니다.")

        return {
            "contradictions": contradictions,
            "analysis_summary": " ".join(summary_parts),
            "total_issues": len(contradictions)
        }
