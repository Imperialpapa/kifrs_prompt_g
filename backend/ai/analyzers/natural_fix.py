"""
Natural Language Fix Mixin
===========================
자연어 수정 지시를 구조화된 수정 명령으로 변환
"""

import json
import re
from typing import Dict, List, Any

from ai.providers.cloud import parse_json_response
from utils.logger import get_logger

logger = get_logger("ai.analyzers.natural_fix")


class NaturalFixMixin:
    """
    자연어 수정 지시 파싱 메서드 모음 (Mixin)
    """

    async def parse_fix_instruction(
        self,
        instruction: str,
        column_names: Dict[str, List[str]],
        provider: str = None
    ) -> Dict[str, Any]:
        """
        자연어 수정 지시를 구조화된 수정 명령으로 변환합니다.

        예시:
        - "퇴직일 비어있는 퇴직자는 2025-12-31로 채워줘"
        - "성별이 M인 것을 1로 바꿔줘"
        - "급여가 0인 행의 급여를 100으로 수정해줘"

        Returns:
            Dict: condition, target_field, new_value, target_condition_op
        """
        target_provider = (provider or self.default_provider).lower()
        use_cloud = target_provider != "local" and self._check_provider_availability(target_provider)

        if use_cloud:
            try:
                return await self._parse_fix_cloud(instruction, column_names, target_provider)
            except Exception as e:
                logger.error("Cloud fix parse failed, falling back to local: %s", e)

        return self._parse_fix_local(instruction, column_names)

    def _parse_fix_local(self, instruction: str, column_names: Dict[str, List[str]]) -> Dict[str, Any]:
        """로컬 패턴 매칭으로 수정 지시 파싱"""
        instruction_lower = instruction.lower().strip()

        # 모든 컬럼명 수집
        all_columns = set()
        for cols in column_names.values():
            all_columns.update(cols)

        result = {
            "instruction": instruction,
            "condition": {},
            "target_field": "",
            "new_value": "",
            "target_condition_op": "",
            "confidence": 0.0,
            "interpretation": ""
        }

        # 패턴 1: "X 비어있는 Y를/은/는 Z로 채워줘/바꿔줘"
        p1 = re.search(
            r'([가-힣a-zA-Z_]+)\s*(?:이|가)?\s*비어\s*있는\s*.*?(?:을|를|은|는)?\s*(.+?)(?:으로|로)\s*(?:채워|바꿔|변경|수정)',
            instruction
        )
        if p1:
            target = p1.group(1).strip()
            new_val = p1.group(2).strip()
            result["target_field"] = target
            result["new_value"] = new_val
            result["target_condition_op"] = "is_empty"
            result["confidence"] = 0.85
            result["interpretation"] = f"{target}이(가) 비어있는 행의 {target}을(를) {new_val}(으)로 수정"
            return result

        # 패턴 2: "X이/가 Y인 것/행의 Z를 W로 바꿔/수정"
        p2 = re.search(
            r'([가-힣a-zA-Z_]+)\s*(?:이|가)?\s*(.+?)(?:인|인\s)\s*.*?([가-힣a-zA-Z_]+)\s*(?:을|를)?\s*(.+?)(?:으로|로)\s*(?:바꿔|변경|수정|채워)',
            instruction
        )
        if p2:
            cond_field = p2.group(1).strip()
            cond_value = p2.group(2).strip()
            target = p2.group(3).strip()
            new_val = p2.group(4).strip()
            result["condition"] = {"field": cond_field, "op": "equals", "value": cond_value}
            result["target_field"] = target
            result["new_value"] = new_val
            result["confidence"] = 0.80
            result["interpretation"] = f"{cond_field}이(가) {cond_value}인 행의 {target}을(를) {new_val}(으)로 수정"
            return result

        # 패턴 3: "X를/을 Y로 바꿔/변경"
        p3 = re.search(
            r'([가-힣a-zA-Z_]+)\s*(?:을|를)\s*(.+?)(?:으로|로)\s*(?:바꿔|변경|수정|채워)',
            instruction
        )
        if p3:
            target = p3.group(1).strip()
            new_val = p3.group(2).strip()
            result["target_field"] = target
            result["new_value"] = new_val
            result["confidence"] = 0.65
            result["interpretation"] = f"{target}을(를) {new_val}(으)로 수정"
            return result

        # 패턴 4: 퇴직자 + 퇴직일 특수 처리
        if '퇴직자' in instruction and '퇴직일' in instruction:
            date_match = re.search(r'(\d{4}[-/]?\d{2}[-/]?\d{2})', instruction)
            new_val = date_match.group(1).replace('-', '').replace('/', '') if date_match else ""
            result["condition"] = {"field": "퇴직일", "op": "is_not_empty", "value": ""}
            result["target_field"] = "퇴직일"
            result["target_condition_op"] = "is_empty"
            result["new_value"] = new_val
            result["confidence"] = 0.80
            result["interpretation"] = f"퇴직일이 비어있는 퇴직자의 퇴직일을 {new_val}(으)로 수정"
            return result

        result["confidence"] = 0.0
        result["interpretation"] = "지시를 파싱할 수 없습니다. 더 구체적으로 입력해주세요."
        return result

    async def _parse_fix_cloud(
        self,
        instruction: str,
        column_names: Dict[str, List[str]],
        provider: str
    ) -> Dict[str, Any]:
        """Cloud AI를 사용한 수정 지시 파싱"""
        prompt = f"""당신은 한국어 데이터 수정 지시를 분석하는 전문가입니다.
사용자의 수정 지시를 구조화된 명령으로 변환해주세요.

사용 가능한 컬럼명: {json.dumps(list(set(c for cols in column_names.values() for c in cols)), ensure_ascii=False)}

사용자 지시: {instruction}

다음 JSON 형식으로 응답하세요:
{{
    "condition": {{"field": "조건 컬럼명", "op": "equals|contains|is_empty|is_not_empty|greater_than|less_than", "value": "조건값"}},
    "target_field": "수정할 컬럼명",
    "new_value": "새로운 값",
    "target_condition_op": "is_empty|is_not_empty (타겟 필드에 대한 추가 조건, 선택)",
    "confidence": 0.0~1.0,
    "interpretation": "해석 설명"
}}"""

        response = await self._call_cloud_ai_async(prompt, provider)
        try:
            parsed = parse_json_response(response)
            if not parsed:
                return self._parse_fix_local(instruction, column_names)
            parsed["instruction"] = instruction
            return parsed
        except Exception:
            return self._parse_fix_local(instruction, column_names)
