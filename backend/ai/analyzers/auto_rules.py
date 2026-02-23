"""
Auto Rule Generation Mixin
===========================
데이터 패턴 분석을 통한 암묵적 규칙 역추론 제안
"""

import json
import re
from typing import Dict, List, Any

from utils.logger import get_logger

logger = get_logger("ai.analyzers.auto_rules")


class AutoRulesMixin:
    """
    규칙 자동 생성 메서드 모음 (Mixin)
    """

    async def auto_generate_rules(
        self,
        sheet_data: Dict[str, Any],
        column_names: Dict[str, List[str]],
        provider: str = None
    ) -> Dict[str, Any]:
        """
        데이터 패턴 분석을 통한 암묵적 규칙 역추론 제안

        Args:
            sheet_data: 시트별 DataFrame
            column_names: 시트별 컬럼명
            provider: AI provider

        Returns:
            Dict: suggested_rules 리스트
        """
        import pandas as pd

        all_suggestions = []

        for sheet_name, df in sheet_data.items():
            if not isinstance(df, pd.DataFrame) or len(df) == 0:
                continue

            cols = [str(c) for c in df.columns]

            for col in cols:
                col_data = df[col]
                suggestions = self._analyze_column_for_rules(col, col_data, len(df))
                for s in suggestions:
                    s["sheet_name"] = sheet_name
                all_suggestions.extend(suggestions)

        # Cloud AI 보강
        target_provider = (provider or self.default_provider).lower()
        use_cloud = target_provider != "local" and self._check_provider_availability(target_provider)

        if use_cloud:
            try:
                ai_suggestions = await self._auto_rules_cloud(sheet_data, column_names, target_provider)
                all_suggestions.extend(ai_suggestions)
            except Exception as e:
                logger.error("Cloud auto-rule generation failed: %s", e)

        # 중복 제거 (같은 필드 + 같은 rule_type)
        seen = set()
        unique_suggestions = []
        for s in all_suggestions:
            key = (s.get("field_name"), s.get("rule_type"))
            if key not in seen:
                seen.add(key)
                unique_suggestions.append(s)

        return {
            "total_suggestions": len(unique_suggestions),
            "suggested_rules": unique_suggestions,
            "engine": f"cloud-{target_provider}" if use_cloud else "local-parser"
        }

    def _analyze_column_for_rules(self, col_name: str, col_data, total_rows: int) -> List[Dict[str, Any]]:
        """단일 컬럼 통계 분석 -> 규칙 제안"""
        import pandas as pd
        suggestions = []

        col_str = col_name.lower()

        # 비결측 데이터
        non_null = col_data.dropna()
        non_null_str = non_null.astype(str).str.strip()
        non_null_str = non_null_str[~non_null_str.isin(['', 'nan', 'None', 'NaT'])]

        if len(non_null_str) == 0:
            return suggestions

        null_ratio = 1.0 - len(non_null_str) / total_rows
        unique_ratio = non_null_str.nunique() / len(non_null_str) if len(non_null_str) > 0 else 0

        # 규칙 1: 필수 필드 탐지 (null 비율이 0%이면 -> required 제안)
        if null_ratio == 0 and total_rows >= 5:
            suggestions.append({
                "field_name": col_name,
                "rule_type": "required",
                "rule_text": f"{col_name}: 공백 없음 (필수 항목)",
                "parameters": {},
                "confidence": 0.95,
                "reason": f"전체 {total_rows}행에서 빈 값이 없습니다."
            })

        # 규칙 2: 고유값 탐지 (unique ratio == 1.0 -> no_duplicates 제안)
        if unique_ratio == 1.0 and len(non_null_str) >= 5:
            id_keywords = ['사번', '코드', 'id', 'code', '번호']
            is_id_like = any(kw in col_str for kw in id_keywords)
            if is_id_like:
                suggestions.append({
                    "field_name": col_name,
                    "rule_type": "no_duplicates",
                    "rule_text": f"{col_name}: 중복 없음",
                    "parameters": {},
                    "confidence": 0.90,
                    "reason": f"모든 {len(non_null_str)}개 값이 고유합니다."
                })

        # 규칙 3: 날짜 형식 탐지
        date_pattern = re.compile(r'^\d{8}$')
        date_matches = non_null_str.apply(lambda x: bool(date_pattern.match(str(x))))
        if date_matches.sum() / len(non_null_str) >= 0.9:
            suggestions.append({
                "field_name": col_name,
                "rule_type": "format",
                "rule_text": f"{col_name}: YYYYMMDD 형식",
                "parameters": {"format": "YYYYMMDD", "regex": "^[0-9]{8}$"},
                "confidence": 0.90,
                "reason": f"{date_matches.sum()}/{len(non_null_str)}개 값이 8자리 날짜 형식입니다."
            })

        # 규칙 4: 허용값 탐지 (unique 값이 적으면 -> allowed_values 제안)
        unique_vals = non_null_str.unique()
        if 2 <= len(unique_vals) <= 10 and unique_ratio < 0.5:
            suggestions.append({
                "field_name": col_name,
                "rule_type": "allowed_values",
                "rule_text": f"{col_name}: {'/'.join(sorted(unique_vals))} 중 하나",
                "parameters": {"allowed_values": sorted(unique_vals.tolist())},
                "confidence": 0.85,
                "reason": f"값이 {len(unique_vals)}종류로 제한됩니다: {', '.join(sorted(unique_vals)[:5])}"
            })

        # 규칙 5: 숫자 범위 탐지
        numeric_data = pd.to_numeric(non_null_str, errors='coerce').dropna()
        if len(numeric_data) >= 5 and len(numeric_data) / len(non_null_str) >= 0.9:
            min_val = float(numeric_data.min())
            max_val = float(numeric_data.max())
            if min_val >= 0:
                suggestions.append({
                    "field_name": col_name,
                    "rule_type": "range",
                    "rule_text": f"{col_name}: {min_val} ~ {max_val} 범위",
                    "parameters": {"min_value": min_val, "max_value": max_val},
                    "confidence": 0.75,
                    "reason": f"숫자 데이터 범위: {min_val} ~ {max_val}"
                })

        # 규칙 6: 패턴 탐지 (정규식)
        if len(non_null_str) >= 5:
            sample = non_null_str.head(20)
            lengths = sample.str.len()
            if lengths.nunique() == 1:
                fixed_len = int(lengths.iloc[0])
                if fixed_len <= 20:
                    # 모든 값이 같은 길이이면 고정 길이 패턴
                    if non_null_str.str.len().nunique() == 1:
                        if non_null_str.str.match(r'^\d+$').all():
                            suggestions.append({
                                "field_name": col_name,
                                "rule_type": "format",
                                "rule_text": f"{col_name}: {fixed_len}자리 숫자",
                                "parameters": {"regex": f"^\\d{{{fixed_len}}}$"},
                                "confidence": 0.80,
                                "reason": f"모든 값이 {fixed_len}자리 숫자 형식입니다."
                            })

        return suggestions

    async def _auto_rules_cloud(
        self,
        sheet_data: Dict[str, Any],
        column_names: Dict[str, List[str]],
        provider: str
    ) -> List[Dict[str, Any]]:
        """Cloud AI를 사용한 규칙 자동 생성"""
        import pandas as pd

        data_info = {}
        for sheet, cols in column_names.items():
            if sheet in sheet_data and isinstance(sheet_data[sheet], pd.DataFrame):
                df = sheet_data[sheet]
                col_stats = {}
                for col in cols:
                    col_data = df[col].dropna().astype(str)
                    col_stats[col] = {
                        "sample_values": col_data.head(5).tolist(),
                        "unique_count": int(col_data.nunique()),
                        "null_count": int(df[col].isna().sum()),
                        "total": len(df)
                    }
                data_info[sheet] = col_stats

        prompt = f"""당신은 K-IFRS 1019 DBO 데이터 검증 전문가입니다.
다음 데이터의 컬럼별 통계를 분석하여 암묵적인 데이터 검증 규칙을 제안해주세요.

데이터 통계:
{json.dumps(data_info, ensure_ascii=False, default=str)[:4000]}

다음 JSON 형식으로 응답하세요:
{{
    "suggested_rules": [
        {{
            "field_name": "컬럼명",
            "rule_type": "required|no_duplicates|format|allowed_values|range|date_logic|cross_field",
            "rule_text": "규칙 설명 (한국어)",
            "parameters": {{}},
            "confidence": 0.0~1.0,
            "reason": "제안 근거"
        }}
    ]
}}"""

        response = self._call_cloud_ai_sync(prompt, provider)
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            parsed = json.loads(match.group(0)) if match else {}
            return parsed.get("suggested_rules", [])
        except Exception:
            return []
