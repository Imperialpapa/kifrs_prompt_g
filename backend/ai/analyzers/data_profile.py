"""
Data Profile Analysis Mixin
============================
Zero-Rule Anomaly Scan - 규칙 없이 데이터만으로 이상치 자동 탐지
"""

import json
import re
from typing import Dict, List, Any

from ai.providers.cloud import parse_json_response
from utils.logger import get_logger

logger = get_logger("ai.analyzers.data_profile")


class DataProfileMixin:
    """
    데이터 프로파일링 메서드 모음 (Mixin)
    """

    async def analyze_data_profile(
        self,
        sheet_data_samples: Dict[str, List[Dict[str, Any]]],
        column_names: Dict[str, List[str]],
        sheet_stats: Dict[str, Dict[str, Any]],
        provider: str = None
    ) -> Dict[str, Any]:
        """
        AI 데이터 프로파일링 (Zero-Rule Anomaly Scan)

        규칙 파일 없이 데이터만으로 이상치, 형식 불일치, 결측 패턴, 중복 의심 등을 자동 탐지합니다.

        Args:
            sheet_data_samples: {시트명: [행 데이터 dict, ...]}
            column_names: {시트명: [컬럼명 리스트]}
            sheet_stats: {시트명: {total_rows, columns_count, null_counts: {col: count}, ...}}
            provider: AI 프로바이더

        Returns:
            Dict: {
                "health_score": float (0-100),
                "findings": [{category, sheet, column, description, severity, affected_count, examples}],
                "summary": str,
                "category_scores": {category: score}
            }
        """
        target_provider = (provider or self.default_provider).lower()
        use_cloud = self._check_provider_availability(target_provider)

        # 로컬 통계 분석 (항상 실행)
        local_findings = self._local_data_profile(sheet_data_samples, column_names, sheet_stats)

        if not use_cloud:
            local_findings["engine"] = "local-parser"
            return local_findings

        try:
            prompt = self._build_profile_prompt(sheet_data_samples, column_names, sheet_stats, local_findings)
            ai_response = await self._call_cloud_ai(prompt, target_provider)
            ai_result = self._parse_profile_response(ai_response)

            merged = self._merge_profile_results(local_findings, ai_result)
            merged["engine"] = f"cloud-{target_provider}"
            return merged
        except Exception as e:
            logger.error("Data profiling failed (%s): %s", target_provider, e)
            local_findings["engine"] = f"cloud-{target_provider}→local"
            return local_findings

    def _local_data_profile(
        self,
        sheet_data_samples: Dict[str, List[Dict[str, Any]]],
        column_names: Dict[str, List[str]],
        sheet_stats: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """로컬 결정론적 데이터 프로파일링"""
        findings = []

        for sheet_name, samples in sheet_data_samples.items():
            cols = column_names.get(sheet_name, [])
            stats = sheet_stats.get(sheet_name, {})
            total_rows = stats.get("total_rows", len(samples))
            null_counts = stats.get("null_counts", {})

            for col in cols:
                values = [str(row.get(col, '')).strip() for row in samples if row.get(col) is not None]
                non_empty = [v for v in values if v and v not in ('', 'None', 'nan', 'NaT')]

                if not non_empty:
                    continue

                # 1. 결측 패턴 (>50% 비어있음)
                null_count = null_counts.get(col, 0)
                if total_rows > 0 and null_count > 0:
                    null_rate = null_count / total_rows
                    if null_rate > 0.5:
                        findings.append({
                            "category": "missing_data",
                            "sheet": sheet_name,
                            "column": col,
                            "description": f"결측률이 {null_rate*100:.0f}%로 매우 높습니다. ({null_count}/{total_rows}건)",
                            "severity": "high" if null_rate > 0.8 else "medium",
                            "affected_count": null_count,
                            "examples": []
                        })
                    elif null_rate > 0.2:
                        findings.append({
                            "category": "missing_data",
                            "sheet": sheet_name,
                            "column": col,
                            "description": f"결측률이 {null_rate*100:.0f}%입니다. ({null_count}/{total_rows}건)",
                            "severity": "low",
                            "affected_count": null_count,
                            "examples": []
                        })

                # 2. 형식 불일치 감지
                format_groups = {"date_hyphen": 0, "date_dot": 0, "date_slash": 0,
                                 "date_plain": 0, "numeric": 0, "text": 0}
                for v in non_empty:
                    if re.match(r'^\d{4}-\d{2}-\d{2}', v):
                        format_groups["date_hyphen"] += 1
                    elif re.match(r'^\d{4}\.\d{2}\.\d{2}', v):
                        format_groups["date_dot"] += 1
                    elif re.match(r'^\d{4}/\d{2}/\d{2}', v):
                        format_groups["date_slash"] += 1
                    elif re.match(r'^(19|20)\d{6}$', v):
                        format_groups["date_plain"] += 1
                    elif re.match(r'^-?[\d,]+\.?\d*$', v.replace(',', '')):
                        format_groups["numeric"] += 1
                    else:
                        format_groups["text"] += 1

                # 날짜 형식이 2가지 이상 혼재
                date_formats_used = {k: v for k, v in format_groups.items() if k.startswith("date_") and v > 0}
                if len(date_formats_used) >= 2:
                    format_names = {"date_hyphen": "YYYY-MM-DD", "date_dot": "YYYY.MM.DD",
                                    "date_slash": "YYYY/MM/DD", "date_plain": "YYYYMMDD"}
                    mixed_str = ", ".join(f"{format_names.get(k, k)}({v}건)" for k, v in date_formats_used.items())
                    findings.append({
                        "category": "format_inconsistency",
                        "sheet": sheet_name,
                        "column": col,
                        "description": f"날짜 형식이 혼재되어 있습니다: {mixed_str}",
                        "severity": "medium",
                        "affected_count": sum(date_formats_used.values()),
                        "examples": [v for v in non_empty[:5]]
                    })

                # 3. 숫자 컬럼 이상치 탐지
                numeric_vals = []
                for v in non_empty:
                    try:
                        numeric_vals.append(float(v.replace(',', '')))
                    except Exception:
                        pass

                if len(numeric_vals) >= 5:
                    sorted_vals = sorted(numeric_vals)
                    q1_idx = len(sorted_vals) // 4
                    q3_idx = 3 * len(sorted_vals) // 4
                    q1 = sorted_vals[q1_idx]
                    q3 = sorted_vals[q3_idx]
                    iqr = q3 - q1

                    if iqr > 0:
                        lower_bound = q1 - 3 * iqr
                        upper_bound = q3 + 3 * iqr
                        outliers = [v for v in numeric_vals if v < lower_bound or v > upper_bound]

                        if outliers:
                            findings.append({
                                "category": "statistical_outlier",
                                "sheet": sheet_name,
                                "column": col,
                                "description": f"통계적 이상치 {len(outliers)}건 발견 (범위: {q1:,.0f}~{q3:,.0f}, 이상치: {min(outliers):,.0f}~{max(outliers):,.0f})",
                                "severity": "medium",
                                "affected_count": len(outliers),
                                "examples": [f"{v:,.0f}" for v in outliers[:5]]
                            })

                # 4. 중복 의심
                if len(non_empty) >= 2:
                    from collections import Counter
                    val_counts = Counter(non_empty)
                    duplicates = {v: c for v, c in val_counts.items() if c > 1}

                    # 식별자성 컬럼(사번, ID 등)에서만 중복 경고
                    id_keywords = ["사번", "사원번호", "id", "코드", "code", "주민", "번호"]
                    is_id_col = any(kw in col.lower() or kw in col for kw in id_keywords)

                    if is_id_col and duplicates:
                        dup_examples = list(duplicates.items())[:3]
                        dup_str = ", ".join(f"'{v}'({c}회)" for v, c in dup_examples)
                        findings.append({
                            "category": "duplicate_suspect",
                            "sheet": sheet_name,
                            "column": col,
                            "description": f"식별자 컬럼에서 중복 값 발견: {dup_str}",
                            "severity": "high",
                            "affected_count": sum(c - 1 for c in duplicates.values()),
                            "examples": [v for v, _ in dup_examples]
                        })

        # 건강 점수 계산
        total_severity_score = sum(
            3 if f["severity"] == "high" else (2 if f["severity"] == "medium" else 1)
            for f in findings
        )
        health_score = max(0, 100 - total_severity_score * 5)

        # 카테고리별 점수
        category_scores = {}
        categories = set(f["category"] for f in findings)
        for cat in categories:
            cat_findings = [f for f in findings if f["category"] == cat]
            cat_severity = sum(3 if f["severity"] == "high" else 2 if f["severity"] == "medium" else 1 for f in cat_findings)
            category_scores[cat] = max(0, 100 - cat_severity * 10)

        # 카테고리 이름 매핑
        cat_labels = {
            "missing_data": "결측 데이터",
            "format_inconsistency": "형식 일관성",
            "statistical_outlier": "통계적 이상치",
            "duplicate_suspect": "중복 의심"
        }
        category_scores_labeled = {cat_labels.get(k, k): v for k, v in category_scores.items()}

        summary = f"데이터 건강 점수: {health_score}점. "
        if findings:
            summary += f"총 {len(findings)}건의 잠재적 이슈 발견."
        else:
            summary += "특별한 이슈가 발견되지 않았습니다."

        return {
            "health_score": health_score,
            "findings": findings,
            "summary": summary,
            "category_scores": category_scores_labeled
        }

    def _build_profile_prompt(
        self,
        sheet_data_samples: Dict[str, List[Dict[str, Any]]],
        column_names: Dict[str, List[str]],
        sheet_stats: Dict[str, Dict[str, Any]],
        local_findings: Dict[str, Any]
    ) -> str:
        """데이터 프로파일링 AI 프롬프트"""
        data_desc = []
        for sheet_name, samples in sheet_data_samples.items():
            cols = column_names.get(sheet_name, [])
            stats = sheet_stats.get(sheet_name, {})
            data_desc.append(f"\n[시트: {sheet_name}] 총 {stats.get('total_rows', '?')}행, 컬럼: {', '.join(cols)}")
            for i, row in enumerate(samples[:10]):
                row_str = json.dumps(row, ensure_ascii=False, default=str)
                data_desc.append(f"  Row {i+1}: {row_str}")

        local_issues = json.dumps(local_findings.get("findings", [])[:10], ensure_ascii=False, default=str)

        return f"""You are a K-IFRS 1019 Data Quality Expert performing a ZERO-RULE DATA PROFILING scan.
Analyze the employee benefit dataset below WITHOUT any predefined rules.
Find anomalies, patterns, and potential data quality issues.

ALREADY DETECTED by local analysis:
{local_issues}

DATA:
{''.join(data_desc)}

Find ADDITIONAL issues NOT already detected above. Focus on:
1. Semantic anomalies (values that don't make business sense)
2. Cross-column pattern breaks
3. Unusual distributions
4. Business logic violations specific to K-IFRS 1019 / DBO data

OUTPUT ONLY JSON:
{{
    "findings": [
        {{
            "category": "semantic_anomaly|business_logic|pattern_break|other",
            "sheet": "시트명",
            "column": "컬럼명 (or 'multiple')",
            "description": "설명 (한국어)",
            "severity": "high|medium|low",
            "affected_count": 0,
            "examples": ["예시값1"]
        }}
    ],
    "ai_summary": "AI가 발견한 추가 소견 (한국어, 2-3문장)"
}}"""

    def _parse_profile_response(self, response: str) -> Dict[str, Any]:
        """AI 프로파일링 결과 파싱 (단계적 JSON 파서 사용)"""
        try:
            data = parse_json_response(response)
            if not data:
                return {"findings": [], "ai_summary": ""}
            return {
                "findings": data.get("findings", []),
                "ai_summary": data.get("ai_summary", "")
            }
        except Exception as e:
            logger.error("Failed to parse profile response: %s", e)
            return {"findings": [], "ai_summary": ""}

    def _merge_profile_results(
        self, local: Dict[str, Any], ai: Dict[str, Any]
    ) -> Dict[str, Any]:
        """로컬 + AI 프로파일링 결과 병합"""
        all_findings = local.get("findings", []) + ai.get("findings", [])

        # 건강 점수 재계산
        total_severity = sum(
            3 if f["severity"] == "high" else 2 if f["severity"] == "medium" else 1
            for f in all_findings
        )
        health_score = max(0, 100 - total_severity * 5)

        summary = local.get("summary", "")
        ai_summary = ai.get("ai_summary", "")
        if ai_summary:
            summary += f" {ai_summary}"

        return {
            "health_score": health_score,
            "findings": all_findings,
            "summary": summary,
            "category_scores": local.get("category_scores", {})
        }
