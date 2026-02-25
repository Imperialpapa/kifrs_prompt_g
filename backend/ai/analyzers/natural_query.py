"""
Natural Language Query Mixin
=============================
한국어 자연어 질의로 데이터 검색
"""

import json
import os
import re
from typing import Dict, List, Any, Optional

from ai.providers.cloud import parse_json_response
from utils.logger import get_logger

logger = get_logger("ai.analyzers.natural_query")

AI_PROMPT_MAX_CHARS = int(os.getenv("AI_PROMPT_MAX_CHARS", "8000"))


class NaturalQueryMixin:
    """
    자연어 질의 검증 메서드 모음 (Mixin)
    """

    async def query_data_natural_language(
        self,
        query: str,
        sheet_data_samples: Dict[str, Any],
        column_names: Dict[str, List[str]],
        provider: str = None
    ) -> Dict[str, Any]:
        """
        한국어 자연어 질의로 데이터를 검색합니다.

        Args:
            query: 사용자 자연어 질의 (예: "퇴직일이 2025년인 직원은?")
            sheet_data_samples: 시트별 데이터 (DataFrame dict)
            column_names: 시트별 컬럼명 리스트
            provider: AI provider

        Returns:
            Dict: 질의 결과 (matched_rows, summary, query_interpretation)
        """
        target_provider = (provider or self.default_provider).lower()
        use_cloud = target_provider != "local" and self._check_provider_availability(target_provider)

        if use_cloud:
            try:
                return await self._query_data_cloud(query, sheet_data_samples, column_names, target_provider)
            except Exception as e:
                logger.error("Cloud query failed, falling back to local: %s", e)

        return self._query_data_local(query, sheet_data_samples, column_names)

    def _query_data_local(
        self,
        query: str,
        sheet_data: Dict[str, Any],
        column_names: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """로컬 키워드 기반 자연어 질의 처리"""
        import pandas as pd

        results = []
        query_lower = query.lower().strip()

        # 키워드 기반 필터 조건 추출
        conditions = self._extract_query_conditions(query_lower)

        for sheet_name, df in sheet_data.items():
            if not isinstance(df, pd.DataFrame):
                continue

            cols = [str(c) for c in df.columns]
            matched_mask = pd.Series([True] * len(df), index=df.index)

            for cond in conditions:
                col_match = self._find_matching_column(cond["field"], cols)
                if not col_match:
                    continue

                col_data = df[col_match].astype(str).str.strip()

                if cond["op"] == "contains":
                    matched_mask &= col_data.str.contains(str(cond["value"]), case=False, na=False)
                elif cond["op"] == "equals":
                    matched_mask &= col_data == str(cond["value"])
                elif cond["op"] == "greater_than":
                    try:
                        matched_mask &= pd.to_numeric(col_data, errors='coerce') > float(cond["value"])
                    except (ValueError, TypeError):
                        matched_mask &= col_data > str(cond["value"])
                elif cond["op"] == "less_than":
                    try:
                        matched_mask &= pd.to_numeric(col_data, errors='coerce') < float(cond["value"])
                    except (ValueError, TypeError):
                        matched_mask &= col_data < str(cond["value"])
                elif cond["op"] == "is_empty":
                    matched_mask &= col_data.isin(['', 'nan', 'None', 'NaT', 'NaN'])
                elif cond["op"] == "is_not_empty":
                    matched_mask &= ~col_data.isin(['', 'nan', 'None', 'NaT', 'NaN'])
                elif cond["op"] == "starts_with":
                    matched_mask &= col_data.str.startswith(str(cond["value"]), na=False)

            matched_df = df[matched_mask]
            if len(matched_df) > 0:
                for idx, row in matched_df.head(100).iterrows():
                    row_dict = {"__sheet__": sheet_name, "__row__": idx + 2}
                    for col in cols:
                        val = row[col]
                        if pd.notna(val):
                            row_dict[col] = str(val)
                    results.append(row_dict)

        return {
            "query": query,
            "query_interpretation": f"조건 {len(conditions)}개 추출 (로컬 파서)",
            "conditions": conditions,
            "total_matches": len(results),
            "matched_rows": results[:100],
            "summary": f"총 {len(results)}건의 데이터가 질의 조건에 부합합니다." if results else "조건에 맞는 데이터를 찾지 못했습니다.",
            "engine": "local-parser"
        }

    def _extract_query_conditions(self, query: str) -> List[Dict[str, Any]]:
        """자연어 질의에서 필터 조건을 추출"""
        conditions = []

        # 패턴 1: "X이/가 Y인" -> field=X, value=Y, op=contains
        pattern_is = re.findall(r'([가-힣a-zA-Z_]+)[이가]\s+(.+?)(?:인|인\s)', query)
        for field, value in pattern_is:
            conditions.append({"field": field.strip(), "op": "contains", "value": value.strip()})

        # 패턴 2: "X이/가 Y 이상" -> field=X, value=Y, op=greater_than
        pattern_gte = re.findall(r'([가-힣a-zA-Z_]+)[이가]\s+([\d,.]+)\s*(?:만원\s*)?이상', query)
        for field, value in pattern_gte:
            val = value.replace(',', '').replace('만원', '')
            conditions.append({"field": field.strip(), "op": "greater_than", "value": val})

        # 패턴 3: "X이/가 Y 이하" -> field=X, value=Y, op=less_than
        pattern_lte = re.findall(r'([가-힣a-zA-Z_]+)[이가]\s+([\d,.]+)\s*(?:만원\s*)?이하', query)
        for field, value in pattern_lte:
            val = value.replace(',', '').replace('만원', '')
            conditions.append({"field": field.strip(), "op": "less_than", "value": val})

        # 패턴 4: "X이/가 비어있는 / 없는" -> field=X, op=is_empty
        pattern_empty = re.findall(r'([가-힣a-zA-Z_]+)[이가]\s*(?:비어\s*있|없|빈|공백)', query)
        for field in pattern_empty:
            conditions.append({"field": field.strip(), "op": "is_empty", "value": ""})

        # 패턴 5: "X 중" or "X 중에서" (컨텍스트) - 무시
        # 패턴 6: "YYYY년" -> 연도 필터
        year_match = re.findall(r'(\d{4})년', query)
        if year_match and not conditions:
            # 연도만 지정된 경우: 날짜 관련 컬럼에서 검색
            for year in year_match:
                conditions.append({"field": "날짜", "op": "starts_with", "value": year})

        # 패턴 7: "퇴직자" -> 퇴직일이 있는 사람
        if '퇴직자' in query and not any(c["field"] == "퇴직일" for c in conditions):
            conditions.append({"field": "퇴직일", "op": "is_not_empty", "value": ""})

        # 패턴 8: "재직자" -> 퇴직일이 비어있는 사람
        if '재직자' in query and not any(c["field"] == "퇴직일" for c in conditions):
            conditions.append({"field": "퇴직일", "op": "is_empty", "value": ""})

        return conditions

    def _find_matching_column(self, keyword: str, columns: List[str]) -> Optional[str]:
        """키워드와 가장 잘 매칭되는 컬럼을 찾음"""
        keyword_lower = keyword.lower()

        # 유사어 맵핑
        synonyms = {
            "급여": ["급여", "임금", "연봉", "salary", "wage", "평균임금"],
            "퇴직일": ["퇴직일", "퇴사일", "termination", "퇴직일자"],
            "입사일": ["입사일", "입사일자", "hire_date", "hire"],
            "사번": ["사번", "사원번호", "employee_id", "emp_id"],
            "이름": ["이름", "성명", "사원명", "name"],
            "생년월일": ["생년월일", "생일", "birth_date", "birth"],
            "성별": ["성별", "gender", "sex"],
            "날짜": ["퇴직일", "입사일", "평가기준일", "기준일", "date"],
        }

        # 정확 매칭
        for col in columns:
            if keyword_lower in col.lower():
                return col

        # 유사어 매칭
        for key, syns in synonyms.items():
            if keyword_lower in [s.lower() for s in syns]:
                for col in columns:
                    for syn in syns:
                        if syn.lower() in col.lower():
                            return col

        return None

    async def _query_data_cloud(
        self,
        query: str,
        sheet_data: Dict[str, Any],
        column_names: Dict[str, List[str]],
        provider: str
    ) -> Dict[str, Any]:
        """Cloud AI를 사용한 자연어 질의"""
        import pandas as pd

        # 데이터 요약 생성
        data_summary = {}
        for sheet, cols in column_names.items():
            data_summary[sheet] = {"columns": cols}
            if sheet in sheet_data and isinstance(sheet_data[sheet], pd.DataFrame):
                data_summary[sheet]["row_count"] = len(sheet_data[sheet])
                data_summary[sheet]["sample"] = sheet_data[sheet].head(5).to_dict('records')

        data_json = json.dumps(data_summary, ensure_ascii=False, default=str)
        if len(data_json) > AI_PROMPT_MAX_CHARS:
            logger.warning("Natural query 데이터가 %d자로 %d자 제한 초과, 잘림 발생",
                           len(data_json), AI_PROMPT_MAX_CHARS)
            data_json = data_json[:AI_PROMPT_MAX_CHARS] + "..."

        prompt = f"""당신은 한국어 데이터 질의 전문가입니다.
사용자의 자연어 질문을 분석하여 데이터에서 조건에 맞는 행을 찾아주세요.

데이터 구조:
{data_json}

사용자 질문: {query}

다음 JSON 형식으로 응답하세요:
{{
    "query_interpretation": "질문 해석 설명",
    "conditions": [
        {{"field": "컬럼명", "op": "contains|equals|greater_than|less_than|is_empty|is_not_empty|starts_with", "value": "값"}}
    ],
    "summary": "결과 요약"
}}"""

        response = await self._call_cloud_ai_async(prompt, provider)
        parsed = parse_json_response(response)

        # 파싱된 조건으로 로컬 필터링 실행
        conditions = parsed.get("conditions", [])
        results = []
        for sheet_name, df in sheet_data.items():
            if not isinstance(df, pd.DataFrame):
                continue
            cols = [str(c) for c in df.columns]
            matched_mask = pd.Series([True] * len(df), index=df.index)

            for cond in conditions:
                col_match = self._find_matching_column(cond.get("field", ""), cols)
                if not col_match:
                    continue
                col_data = df[col_match].astype(str).str.strip()
                op = cond.get("op", "contains")
                val = str(cond.get("value", ""))

                if op == "contains":
                    matched_mask &= col_data.str.contains(val, case=False, na=False)
                elif op == "equals":
                    matched_mask &= col_data == val
                elif op == "greater_than":
                    try:
                        matched_mask &= pd.to_numeric(col_data, errors='coerce') > float(val)
                    except (ValueError, TypeError):
                        pass
                elif op == "less_than":
                    try:
                        matched_mask &= pd.to_numeric(col_data, errors='coerce') < float(val)
                    except (ValueError, TypeError):
                        pass
                elif op == "is_empty":
                    matched_mask &= col_data.isin(['', 'nan', 'None', 'NaT', 'NaN'])
                elif op == "is_not_empty":
                    matched_mask &= ~col_data.isin(['', 'nan', 'None', 'NaT', 'NaN'])
                elif op == "starts_with":
                    matched_mask &= col_data.str.startswith(val, na=False)

            matched_df = df[matched_mask]
            for idx, row in matched_df.head(100).iterrows():
                row_dict = {"__sheet__": sheet_name, "__row__": idx + 2}
                for col in cols:
                    val = row[col]
                    if pd.notna(val):
                        row_dict[col] = str(val)
                results.append(row_dict)

        return {
            "query": query,
            "query_interpretation": parsed.get("query_interpretation", "AI 분석 완료"),
            "conditions": conditions,
            "total_matches": len(results),
            "matched_rows": results[:100],
            "summary": parsed.get("summary", f"총 {len(results)}건 매칭"),
            "engine": f"cloud-{provider}"
        }
