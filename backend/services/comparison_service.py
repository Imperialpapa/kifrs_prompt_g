"""
Comparison Service - 시계열 파일 비교 검증
==========================================
다기간(예: 2024년 vs 2025년) 파일 비교, 비정상 변동 탐지
"""

import io
import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional


class ComparisonService:
    """두 시점의 데이터 파일을 비교하여 비정상 변동을 탐지합니다."""

    # 사번 관련 키워드
    ID_KEYWORDS = ['사번', '사원번호', 'employee_id', 'emp_id', 'id', '코드', 'code']

    def compare_files(
        self,
        file1_content: bytes,
        file2_content: bytes,
        file1_name: str = "이전",
        file2_name: str = "현재"
    ) -> Dict[str, Any]:
        """
        두 파일을 비교하여 변동 요약 및 이상 항목을 반환합니다.

        Args:
            file1_content: 이전 시점 파일 바이트
            file2_content: 현재 시점 파일 바이트
            file1_name: 이전 파일명
            file2_name: 현재 파일명

        Returns:
            Dict: comparison_summary, anomalies, details
        """
        from utils.excel_parser import get_visible_sheet_names

        # 파일 로드
        sheets1 = get_visible_sheet_names(file1_content)
        sheets2 = get_visible_sheet_names(file2_content)

        results = {
            "file1_name": file1_name,
            "file2_name": file2_name,
            "sheets_compared": [],
            "summary": {},
            "anomalies": [],
            "details": []
        }

        # 공통 시트 기반 비교
        common_sheets = set(sheets1) & set(sheets2)
        if not common_sheets:
            # 시트 이름이 다르면 첫 번째 시트끼리 비교
            if sheets1 and sheets2:
                common_sheets = [(sheets1[0], sheets2[0])]
            else:
                results["summary"] = {"error": "비교 가능한 시트가 없습니다."}
                return results
        else:
            common_sheets = [(s, s) for s in common_sheets]

        total_added = 0
        total_removed = 0
        total_changed = 0
        total_unchanged = 0

        for sheet1_name, sheet2_name in common_sheets:
            df1 = pd.read_excel(io.BytesIO(file1_content), sheet_name=sheet1_name)
            df2 = pd.read_excel(io.BytesIO(file2_content), sheet_name=sheet2_name)

            # 빈 행 제거
            df1 = df1.dropna(how='all')
            df2 = df2.dropna(how='all')

            # ID 컬럼 찾기
            id_col1 = self._find_id_column(df1)
            id_col2 = self._find_id_column(df2)

            if not id_col1 or not id_col2:
                results["anomalies"].append({
                    "type": "no_key",
                    "severity": "medium",
                    "sheet": sheet1_name,
                    "description": "키 컬럼(사번)을 찾을 수 없어 행 단위 비교가 불가합니다.",
                    "details": f"시트 '{sheet1_name}' 컬럼: {list(df1.columns)[:5]}"
                })
                continue

            # ID를 문자열로 통일
            df1[id_col1] = df1[id_col1].astype(str).str.strip()
            df2[id_col2] = df2[id_col2].astype(str).str.strip()

            # 빈 ID 제거
            df1 = df1[df1[id_col1].notna() & (df1[id_col1] != '') & (df1[id_col1] != 'nan')]
            df2 = df2[df2[id_col2].notna() & (df2[id_col2] != '') & (df2[id_col2] != 'nan')]

            ids1 = set(df1[id_col1].tolist())
            ids2 = set(df2[id_col2].tolist())

            added = ids2 - ids1
            removed = ids1 - ids2
            common = ids1 & ids2

            total_added += len(added)
            total_removed += len(removed)

            # 신규 입사자 이상 탐지
            if added:
                results["details"].append({
                    "type": "added",
                    "sheet": sheet2_name,
                    "count": len(added),
                    "ids": sorted(list(added))[:20],
                    "description": f"신규 추가된 인원: {len(added)}명"
                })

            # 퇴직자/삭제 이상 탐지
            if removed:
                results["details"].append({
                    "type": "removed",
                    "sheet": sheet1_name,
                    "count": len(removed),
                    "ids": sorted(list(removed))[:20],
                    "description": f"삭제/퇴직 인원: {len(removed)}명"
                })

                # 퇴직자 재등장 체크 (이전에 제거되었다가 다시 나타나는 경우)
                # 여기서는 단순히 removed를 기록

            # 공통 인원 필드별 변동 비교
            common_cols = list(set(df1.columns) & set(df2.columns))
            # ID 컬럼 제외
            compare_cols = [c for c in common_cols if c != id_col1 and c != id_col2]

            df1_indexed = df1.set_index(id_col1)
            df2_indexed = df2.set_index(id_col2)

            changed_count = 0
            unchanged_count = 0

            for emp_id in common:
                if emp_id not in df1_indexed.index or emp_id not in df2_indexed.index:
                    continue

                row1 = df1_indexed.loc[emp_id]
                row2 = df2_indexed.loc[emp_id]

                # 중복 인덱스 처리 (첫 번째 행만 사용)
                if isinstance(row1, pd.DataFrame):
                    row1 = row1.iloc[0]
                if isinstance(row2, pd.DataFrame):
                    row2 = row2.iloc[0]

                has_change = False
                for col in compare_cols:
                    if col not in row1.index or col not in row2.index:
                        continue

                    val1 = row1[col]
                    val2 = row2[col]

                    # NaN 비교
                    if pd.isna(val1) and pd.isna(val2):
                        continue

                    str1 = str(val1).strip() if pd.notna(val1) else ""
                    str2 = str(val2).strip() if pd.notna(val2) else ""

                    if str1 != str2:
                        has_change = True

                        # 수치 변동률 계산
                        anomaly = self._check_field_anomaly(
                            emp_id, col, val1, val2, sheet1_name
                        )
                        if anomaly:
                            results["anomalies"].append(anomaly)

                if has_change:
                    changed_count += 1
                else:
                    unchanged_count += 1

            total_changed += changed_count
            total_unchanged += unchanged_count

            results["sheets_compared"].append({
                "sheet": sheet1_name,
                "file1_rows": len(df1),
                "file2_rows": len(df2),
                "added": len(added),
                "removed": len(removed),
                "changed": changed_count,
                "unchanged": unchanged_count
            })

        results["summary"] = {
            "total_sheets_compared": len(results["sheets_compared"]),
            "total_added": total_added,
            "total_removed": total_removed,
            "total_changed": total_changed,
            "total_unchanged": total_unchanged,
            "total_anomalies": len(results["anomalies"]),
            "description": (
                f"비교 결과: 신규 {total_added}명, 삭제 {total_removed}명, "
                f"변동 {total_changed}명, 미변동 {total_unchanged}명, "
                f"이상 항목 {len(results['anomalies'])}건"
            )
        }

        # 이상 항목 정렬 (severity 기준)
        severity_order = {"high": 0, "medium": 1, "low": 2}
        results["anomalies"].sort(key=lambda x: severity_order.get(x.get("severity", "low"), 3))

        return results

    def _find_id_column(self, df: pd.DataFrame) -> Optional[str]:
        """DataFrame에서 ID(사번) 컬럼을 찾습니다."""
        cols_lower = {str(col).lower(): str(col) for col in df.columns}
        for kw in self.ID_KEYWORDS:
            for col_lower, original in cols_lower.items():
                if kw in col_lower:
                    return original
        return None

    def _check_field_anomaly(
        self,
        emp_id: str,
        col: str,
        val1: Any,
        val2: Any,
        sheet: str
    ) -> Optional[Dict[str, Any]]:
        """필드 변동의 이상 여부를 판단합니다."""
        col_lower = str(col).lower()

        # 수치 필드 급변 탐지
        salary_keywords = ['급여', '임금', '연봉', 'salary', 'wage', '평균임금']
        is_salary = any(kw in col_lower for kw in salary_keywords)

        if is_salary:
            try:
                n1 = float(str(val1).replace(',', ''))
                n2 = float(str(val2).replace(',', ''))

                if n1 > 0:
                    change_rate = abs(n2 - n1) / n1

                    # 50% 이상 급변
                    if change_rate >= 0.5:
                        direction = "급등" if n2 > n1 else "급락"
                        return {
                            "type": "salary_spike",
                            "severity": "high",
                            "sheet": sheet,
                            "employee_id": emp_id,
                            "field": col,
                            "old_value": str(val1),
                            "new_value": str(val2),
                            "change_rate": round(change_rate * 100, 1),
                            "description": f"급여 {direction}: {val1} → {val2} ({change_rate:.0%} 변동)"
                        }
                    # 20%~50% 변동
                    elif change_rate >= 0.2:
                        direction = "증가" if n2 > n1 else "감소"
                        return {
                            "type": "salary_change",
                            "severity": "medium",
                            "sheet": sheet,
                            "employee_id": emp_id,
                            "field": col,
                            "old_value": str(val1),
                            "new_value": str(val2),
                            "change_rate": round(change_rate * 100, 1),
                            "description": f"급여 {direction}: {val1} → {val2} ({change_rate:.0%} 변동)"
                        }
            except (ValueError, TypeError):
                pass

        # 근속연수 역전 탐지
        seniority_keywords = ['근속', '근무연수', 'service_year', 'tenure']
        if any(kw in col_lower for kw in seniority_keywords):
            try:
                n1 = float(str(val1).replace(',', ''))
                n2 = float(str(val2).replace(',', ''))
                if n2 < n1:
                    return {
                        "type": "seniority_reversal",
                        "severity": "high",
                        "sheet": sheet,
                        "employee_id": emp_id,
                        "field": col,
                        "old_value": str(val1),
                        "new_value": str(val2),
                        "description": f"근속연수 역전: {val1} → {val2} (감소함)"
                    }
            except (ValueError, TypeError):
                pass

        # 생년월일 변경 탐지 (변경되면 안 되는 필드)
        immutable_keywords = ['생년월일', 'birth', '주민']
        if any(kw in col_lower for kw in immutable_keywords):
            return {
                "type": "immutable_change",
                "severity": "high",
                "sheet": sheet,
                "employee_id": emp_id,
                "field": col,
                "old_value": str(val1),
                "new_value": str(val2),
                "description": f"불변 필드 변경: {col} {val1} → {val2}"
            }

        return None
