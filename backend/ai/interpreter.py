"""
K-IFRS 1019 DBO Validation System - AI Rule Interpreter
========================================================
Main class combining all mixin modules for hybrid AI/local rule interpretation.

Feature:
1. Multi-Provider: Anthropic, OpenAI, Gemini support (env-var selected)
2. Hybrid Engine: Cloud AI fallback to Regex Parser (Local)
3. Auto-Fix: Deterministic fix suggestions for data cleansing
"""

import json
import re
import os
import time
from typing import List, Dict, Any, Optional

from models import (
    AIInterpretationResponse,
    ValidationRule,
    RuleConflict,
    KIFRS_1019_REFERENCES,
    FixSuggestion
)
from utils.logger import get_logger

from ai.providers.cloud import CloudProviderMixin
from ai.analyzers.cross_field import CrossFieldMixin
from ai.analyzers.data_profile import DataProfileMixin
from ai.analyzers.natural_query import NaturalQueryMixin
from ai.analyzers.auto_rules import AutoRulesMixin
from ai.analyzers.compliance import ComplianceMixin
from ai.analyzers.natural_fix import NaturalFixMixin

logger = get_logger("ai.interpreter")


class AIRuleInterpreter(
    CloudProviderMixin,
    CrossFieldMixin,
    DataProfileMixin,
    NaturalQueryMixin,
    AutoRulesMixin,
    ComplianceMixin,
    NaturalFixMixin,
):
    """
    Multi-Provider AI 규칙 해석기

    Inherits from all mixin classes to compose the full feature set:
    - CloudProviderMixin: cloud API calls (OpenAI, Anthropic, Gemini)
    - CrossFieldMixin: cross-field logical contradiction detection
    - DataProfileMixin: zero-rule anomaly scanning
    - NaturalQueryMixin: natural language data queries
    - AutoRulesMixin: automatic rule generation from data patterns
    - ComplianceMixin: K-IFRS 1019 compliance checking
    - NaturalFixMixin: natural language fix instruction parsing
    """

    def __init__(self):
        """
        초기화: 기본 설정 로드
        """
        self.default_provider = os.getenv("AI_PROVIDER", "openai").lower()
        self.use_cloud_ai = False  # Track whether cloud AI was used
        logger.info("Default Provider: %s", self.default_provider.upper())

    # =========================================================================
    # Core Rule Interpretation
    # =========================================================================

    async def interpret_rules(
        self,
        natural_language_rules: List[Dict[str, Any]],
        provider: str = None
    ) -> AIInterpretationResponse:
        """
        자연어 규칙을 구조화된 JSON으로 변환
        """
        start_time = time.time()

        target_provider = (provider or self.default_provider).lower()

        # "local" provider는 항상 로컬 파서 사용
        if target_provider == "local":
            use_cloud = False
        else:
            use_cloud = self._check_provider_availability(target_provider)

        rules = []
        conflicts = []

        if use_cloud:
            try:
                logger.info("Interpreting rules using %s...", target_provider.upper())
                prompt = self._build_interpretation_prompt(natural_language_rules)
                ai_response = await self._call_cloud_ai(prompt, target_provider)
                rules, conflicts = self._parse_ai_response(ai_response)
                self.use_cloud_ai = True
            except Exception as e:
                logger.info("Cloud inference (%s) failed, falling back to local engine: %s", target_provider, e)
                rules, conflicts = self._local_rule_parser(natural_language_rules)
                self.use_cloud_ai = False
        else:
            logger.info("Provider %s not available/configured. Using Local Engine.", target_provider)
            rules, conflicts = self._local_rule_parser(natural_language_rules)
            self.use_cloud_ai = False

        processing_time = time.time() - start_time

        return AIInterpretationResponse(
            rules=rules,
            conflicts=conflicts,
            ai_summary=self._generate_summary(rules, conflicts),
            processing_time_seconds=processing_time
        )

    async def suggest_corrections(
        self,
        errors: List[Dict[str, Any]],
        past_corrections: List[Dict[str, Any]] = None,
        provider: str = None
    ) -> List[FixSuggestion]:
        """
        오류에 대한 수정 제안 생성 (하이브리드: 로컬 우선 -> 필요 시 클라우드 AI)
        """
        if not errors:
            return []

        # 1. 로컬 휴리스틱 엔진 실행 (즉각적인 수정 제안)
        local_suggestions = self._local_fix_engine(errors)

        # 컬럼별 규칙 매핑 생성 (Cloud 결과 검증용)
        column_rule_map = {}
        for err in errors:
            col = err.get('column')
            if col:
                column_rule_map[col] = {
                    "type": err.get('rule_type'),
                    "params": err.get('rule_params', {})
                }

        target_provider = (provider or self.default_provider).lower()
        use_cloud = self._check_provider_availability(target_provider)

        if use_cloud:
            # 로컬 엔진이 처리하지 못한 항목이나 신뢰도 낮은 항목에 대해 AI 호출 고려 가능
            # 현재는 일관성을 위해 클라우드 AI에게 전체 문맥을 전달하여 제안을 정교화함
            try:
                prompt = self._build_correction_prompt(errors, past_corrections)
                ai_response = await self._call_cloud_ai(prompt, target_provider)
                cloud_suggestions = self._parse_correction_response(ai_response)

                # 클라우드 제안에 대해서도 안전 검증 적용
                valid_cloud_suggestions = self._filter_invalid_suggestions(cloud_suggestions, column_rule_map)

                # 클라우드 제안이 있으면 우선 사용 (병합)
                return valid_cloud_suggestions if valid_cloud_suggestions else local_suggestions
            except Exception as e:
                logger.error("Cloud correction failed (%s), using local engine: %s", target_provider, e)
                return local_suggestions

        return local_suggestions

    def _filter_invalid_suggestions(self, suggestions: List[FixSuggestion], column_rule_map: Dict[str, Any] = None) -> List[FixSuggestion]:
        """
        AI가 생성한 제안 중 논리적으로 맞지 않는 항목 필터링

        필터링 조건:
        1. 금액/숫자 필드인데 '날짜 형식'으로 수정하려는 경우
        2. 금액/숫자 필드인데 수정된 값이 숫자가 아닌 경우
        """
        valid_suggestions = []
        column_rule_map = column_rule_map or {}

        # 금액/숫자 관련 필드 키워드 (로컬 엔진과 동일하게 유지)
        numeric_field_keywords = [
            "급여", "금액", "수당", "원", "임금", "보수", "연봉", "월급",
            "salary", "amount", "wage", "pay", "bonus", "income",
            "기준급", "평균급", "통상급", "퇴직금", "retirement"
        ]

        for sugg in suggestions:
            field = sugg.column
            fixed_val = str(sugg.fixed_value)

            field_lower = field.lower()

            # 규칙 정보 확인
            rule_info = column_rule_map.get(field, {})
            rule_type = rule_info.get("type")
            rule_params = rule_info.get("params", {}) or {}

            # 숫자형 필드 판단 (규칙 + 키워드)
            is_numeric_rule = rule_type == 'range' or \
                              (rule_type == 'format' and 'numeric' in str(rule_params)) or \
                              (rule_type == 'custom' and any(kw in str(rule_params) for kw in ['number', 'amount', '금액']))

            is_numeric_keyword = any(kw in field for kw in numeric_field_keywords) or \
                                 any(kw in field_lower for kw in ["salary", "amount", "wage", "pay"])

            is_numeric_field = is_numeric_rule or is_numeric_keyword

            # 숫자 필드 안전 검증
            if is_numeric_field:
                # 1. 수정된 값이 날짜 형식(YYYYMMDD)인 경우 거부
                is_fixed_date_format = bool(re.match(r'^(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$', fixed_val))

                # 원본이 날짜 구분자가 있는 형태였는데(2023-01-01), 결과가 8자리 숫자라면 -> 날짜 포맷팅으로 간주하고 제거
                is_original_date_like = bool(re.match(r'^\d{4}[-./]\d{2}[-./]\d{2}$', str(sugg.original_value)))

                if is_original_date_like and is_fixed_date_format:
                    logger.info("Filtered unsafe suggestion for numeric field '%s': %s -> %s (Date format detected)",
                                field, sugg.original_value, fixed_val)
                    continue

                # 2. 수정된 값이 숫자가 아닌 경우 거부
                cleaned_val = re.sub(r'[,\s]', '', fixed_val)
                try:
                    float(cleaned_val)
                except ValueError:
                    logger.info("Filtered non-numeric suggestion for numeric field '%s': %s", field, fixed_val)
                    continue

            valid_suggestions.append(sugg)

        return valid_suggestions

    async def get_error_explanation(
        self,
        error: 'ValidationError',
        provider: str = None
    ) -> Dict[str, str]:
        """
        검증 오류에 대한 AI 기반의 설명과 권장 조치를 생성합니다.
        """
        target_provider = (provider or self.default_provider).lower()
        use_cloud = self._check_provider_availability(target_provider)

        if not use_cloud:
            return {
                "explanation": "AI 설명 기능을 사용할 수 없습니다. (설정 필요)",
                "recommendation": "관리자에게 문의하여 AI Provider 설정을 확인하세요."
            }

        try:
            prompt = self._build_explanation_prompt(error)
            ai_response_str = await self._call_cloud_ai(prompt, target_provider)

            # AI 응답 파싱
            match = re.search(r'\{.*\}', ai_response_str, re.DOTALL)
            if not match:
                return {"explanation": ai_response_str, "recommendation": "AI가 생성한 설명을 참고하여 데이터를 직접 수정하세요."}

            response_json = json.loads(match.group(0))

            return {
                "explanation": response_json.get("explanation", "AI가 설명을 생성하지 못했습니다."),
                "recommendation": response_json.get("recommendation", "데이터를 직접 확인하고 수정하세요.")
            }
        except Exception as e:
            logger.error("Error getting explanation: %s", e)
            return {
                "explanation": "오류 설명을 생성하는 중 문제가 발생했습니다.",
                "recommendation": "오류 메시지를 참고하여 데이터를 수정하세요."
            }

    # =========================================================================
    # Prompt Builders & Response Parsers
    # =========================================================================

    def _build_explanation_prompt(self, error: 'ValidationError') -> str:
        """오류 설명을 생성하기 위한 프롬프트 구성"""

        kifrs_context = ""
        # K-IFRS 관련 규칙 ID 형식 (예: KIFRS_CONSISTENCY_DATES) 에 따라 컨텍스트 추가
        if error.rule_id.startswith("KIFRS_"):
            rule_type = error.rule_id.split('_')[1].lower()
            # 모델에 정의된 참조 정보와 매핑 시도
            ref_key = next((key for key in KIFRS_1019_REFERENCES if rule_type in key), None)
            if ref_key and ref_key in KIFRS_1019_REFERENCES:
                 kifrs_context = f'''
                 [Relevant K-IFRS 1019 Guideline: {ref_key}]
                 Description: {KIFRS_1019_REFERENCES[ref_key]['description']}
                 Key Points: {', '.join(KIFRS_1019_REFERENCES[ref_key]['key_points'])}
                 '''

        prompt = f"""
        You are an expert accounting assistant specializing in K-IFRS 1019 (Defined Benefit Obligations).
        A data validation error was found. Your task is to explain it clearly to a user in HR or accounting who may not be a data expert.

        [Validation Error Details]
        - Rule ID: "{error.rule_id}"
        - Error Message: "{error.message}"
        - Sheet: "{error.sheet}"
        - Row: {error.row}
        - Column: "{error.column}"
        - Erroneous Value: "{error.actual_value}"
        {kifrs_context}

        [Your Task]
        Provide a concise explanation and a recommended action in KOREAN.
        1.  **Explanation**: Clearly explain WHY this is a problem from a practical, accounting perspective. Avoid technical jargon.
        2.  **Recommendation**: Suggest a concrete, actionable next step for the user.

        Output ONLY the following JSON structure:
        {{
            "explanation": "...",
            "recommendation": "..."
        }}
        """
        return prompt

    def _build_correction_prompt(self, errors: List[Dict[str, Any]], past_corrections: List[Dict[str, Any]]) -> str:
        """수정 제안을 위한 상세 RAG 프롬프트"""
        return f"""
        You are a Data Quality Expert. Fix the following validation errors in K-IFRS 1019 employee data.

        [Past Correction Examples (Learning Context)]
        {json.dumps(past_corrections, ensure_ascii=False)}

        [Current Errors to Fix]
        {json.dumps(errors, ensure_ascii=False)}

        Guidelines:
        1. Fix format issues (dates to YYYYMMDD, gender to M/F).
        2. Reference past examples if similar patterns exist.
        3. Provide a clear reason for each fix.
        4. Output JSON with "suggestions" list.
        """

    def _parse_correction_response(self, response: str) -> List[FixSuggestion]:
        """AI의 수정 제안 응답 파싱"""
        try:
            match = re.search(r'\{.*\}', response, re.DOTALL)
            data = json.loads(match.group(0)) if match else json.loads(response)
            return [FixSuggestion(**s) for s in data.get("suggestions", [])]
        except Exception:
            return []

    def _build_interpretation_prompt(self, rules: List[Dict[str, Any]]) -> str:
        # 기존과 동일한 프롬프트 로직 사용
        rules_text = json.dumps(rules, indent=2, ensure_ascii=False)
        return f"""
        You are a K-IFRS 1019 Data Validation Expert.
        Parse the following natural language rules into structured JSON.

        CRITICAL REQUIREMENTS:
        1. ALWAYS use 'field_name' from the input - NEVER change or substitute it
        2. In 'error_message_template', ALWAYS use "{{{{field_name}}}}" placeholder instead of hardcoding field names
        3. NEVER mention other field names in the error message

        CORRECT example:
        - field_name: "생년월일"
        - error_message_template: "{{{{field_name}}}}이(가) 중복되었습니다"

        WRONG example (DO NOT DO THIS):
        - field_name: "생년월일"
        - error_message_template: "사원번호이(가) 중복되었습니다"

        Input Rules:
        {rules_text}

        Output Format (JSON): {{ "rules": [...], "conflicts": [...] }}
        """

    def _parse_ai_response(self, ai_response: str) -> tuple:
        """JSON 추출 및 파싱"""
        try:
            # JSON 블록 찾기 (Markdown ```json ... ``` 제거)
            match = re.search(r'\{.*\}', ai_response, re.DOTALL)
            json_str = match.group(0) if match else ai_response

            data = json.loads(json_str)
            rules = [ValidationRule(**r) for r in data.get("rules", [])]
            conflicts = [RuleConflict(**c) for c in data.get("conflicts", [])]
            return rules, conflicts
        except Exception as e:
            logger.error("Failed to parse JSON response: %s", e)
            raise

    # =========================================================================
    # Local Rule Engine (Robust Regex Parser)
    # =========================================================================

    def _split_compound_rule(self, rule_text: str, field_name: str) -> List[Dict[str, str]]:
        """
        쉼표로 구분된 복합 규칙을 개별 규칙으로 분리

        핵심 로직:
        - "YYYYMMDD, 중간정산기준일<= 입사일" -> 두 개의 규칙으로 분리
        - "1, 3, 4" -> 허용값 목록으로 인식 (분리하지 않음)

        조건 구분 쉼표 vs 단순 나열 쉼표 판별:
        - 단순 나열: 모든 세그먼트가 짧은 값(숫자, 짧은 코드)인 경우
        - 조건 구분: 키워드(YYYYMMDD, 공백, 중복 등)나 비교연산자가 포함된 경우

        Args:
            rule_text: 원본 규칙 텍스트
            field_name: 필드명 (필드 간 비교에서 사용)

        Returns:
            List[Dict]: 분리된 규칙 목록 [{text, type, ...}, ...]
        """
        if not rule_text:
            return []

        result = []
        original_rule_text = rule_text

        # 콜론(:) 뒤의 내용만 추출 (예: "중간정산기준일 : YYYYMMDD, ...")
        if ':' in rule_text:
            parts = rule_text.split(':', 1)
            if len(parts) > 1:
                rule_text = parts[1].strip()

        # 먼저 단순 허용값 나열인지 확인
        # "1, 3, 4" 또는 "A, B, C" 같은 패턴 (각 값이 짧고 특수 키워드가 없음)
        is_simple_value_list = self._is_simple_value_list(rule_text)
        if is_simple_value_list:
            # 허용값 목록은 분리하지 않고 전체를 하나의 "allowed_values" 타입으로 반환
            return [{"text": rule_text, "original": original_rule_text, "type": "allowed_values"}]

        # 쉼표로 분리 (단, 괄호 안의 쉼표는 무시)
        segments = []
        current = ""
        paren_depth = 0

        for char in rule_text:
            if char == '(':
                paren_depth += 1
                current += char
            elif char == ')':
                paren_depth -= 1
                current += char
            elif char == ',' and paren_depth == 0:
                if current.strip():
                    segments.append(current.strip())
                current = ""
            else:
                current += char

        if current.strip():
            segments.append(current.strip())

        # 각 세그먼트 분류
        for seg in segments:
            seg_lower = seg.lower().strip()
            seg_info = {"text": seg, "original": seg}

            # 필드 간 비교 규칙 감지 (<=, >=, <, >, =)
            comparison_match = re.search(
                r'([가-힣a-zA-Z0-9_]+)\s*(<=|>=|<>|<|>|=)\s*([가-힣a-zA-Z0-9_]+)',
                seg
            )
            if comparison_match:
                seg_info["type"] = "comparison"
                seg_info["left_field"] = comparison_match.group(1).strip()
                seg_info["operator"] = comparison_match.group(2).strip()
                seg_info["right_field"] = comparison_match.group(3).strip()
            # 날짜 형식 감지
            elif "yyyymmdd" in seg_lower or "yyyy-mm-dd" in seg_lower:
                seg_info["type"] = "date_format"
            # 필수/공백 감지
            elif any(kw in seg for kw in ["공백", "필수", "빈값"]):
                seg_info["type"] = "required"
            # 중복 감지
            elif any(kw in seg for kw in ["중복", "유일", "unique"]):
                seg_info["type"] = "no_duplicates"
            # 범위 감지
            elif any(kw in seg for kw in ["이상", "이하", "초과", "미만"]) or re.search(r'[<>]=?', seg):
                # 필드 간 비교가 아닌 경우에만 범위로 처리
                if not comparison_match:
                    seg_info["type"] = "range"
            else:
                seg_info["type"] = "other"

            result.append(seg_info)

        return result

    def _is_simple_value_list(self, text: str) -> bool:
        """
        텍스트가 단순 허용값 나열인지 판별

        단순 허용값 나열 조건:
        - 쉼표나 슬래시로 구분된 짧은 값들 (각 10자 이하)
        - 특수 키워드(YYYYMMDD, 공백, 중복, 필수 등)가 없음
        - 비교 연산자(<=, >=, <, >, =)가 없음

        예시:
        - "1, 3, 4" -> True
        - "A, B, C" -> True
        - "YYYYMMDD, 중간정산기준일 <= 입사일" -> False
        - "공백없음, 중복없음" -> False
        """
        if not text:
            return False

        # 특수 키워드가 포함되어 있으면 단순 나열이 아님
        special_keywords = [
            "yyyymmdd", "yyyy-mm-dd", "yyyy", "날짜",
            "공백", "필수", "중복", "유일", "unique",
            "이상", "이하", "초과", "미만",
            "형식", "format", "패턴", "regex"
        ]
        text_lower = text.lower()
        if any(kw in text_lower for kw in special_keywords):
            return False

        # 비교 연산자가 포함되어 있으면 단순 나열이 아님
        if re.search(r'<=|>=|<>|<|>', text):
            return False

        # 쉼표나 슬래시로 분리했을 때 모든 값이 짧은 코드인지 확인
        parts = re.split(r'[,/\s]+', text)
        parts = [p.strip() for p in parts if p.strip()]

        if len(parts) < 2:
            return False

        # 모든 값이 10자 이하의 짧은 코드인지 확인
        for part in parts:
            if len(part) > 10:
                return False
            # 콜론이 포함된 "코드:라벨" 형식도 허용
            if ':' in part:
                code_part = part.split(':')[0]
                if len(code_part) > 10:
                    return False

        return True

    def _parse_field_comparison(self, comparison_info: Dict[str, str], field_name: str) -> Optional[Dict[str, Any]]:
        """
        필드 간 비교 규칙을 date_logic 규칙으로 변환

        Args:
            comparison_info: {left_field, operator, right_field, text}
            field_name: 현재 규칙이 정의된 필드명

        Returns:
            Dict: date_logic 규칙 파라미터 또는 None (현재 필드와 무관한 비교면 None)
        """
        left = comparison_info.get("left_field", "")
        op = comparison_info.get("operator", "")
        right = comparison_info.get("right_field", "")

        if not left or not op or not right:
            return None

        # 현재 필드가 비교에 포함되어 있는지 확인
        field_in_left = (left == field_name or left in field_name or field_name in left)
        field_in_right = (right == field_name or right in field_name or field_name in right)

        if not field_in_left and not field_in_right:
            # 현재 필드가 비교와 무관하면 규칙 생성하지 않음
            logger.debug("Skipping comparison '%s %s %s' - not related to field '%s'", left, op, right, field_name)
            return None

        # 연산자 변환
        operator_map = {
            "<=": "less_than_or_equal",
            ">=": "greater_than_or_equal",
            "<": "less_than",
            ">": "greater_than",
            "=": "equal",
            "<>": "not_equal"
        }

        mapped_op = operator_map.get(op)
        if not mapped_op:
            return None

        # 현재 필드와 비교 대상 필드 결정
        if field_in_left:
            compare_field = right
        else:
            compare_field = left
            # 연산자 방향 반전
            reverse_map = {
                "less_than_or_equal": "greater_than_or_equal",
                "greater_than_or_equal": "less_than_or_equal",
                "less_than": "greater_than",
                "greater_than": "less_than",
                "equal": "equal",
                "not_equal": "not_equal"
            }
            mapped_op = reverse_map.get(mapped_op, mapped_op)

        return {
            "compare_field": compare_field,
            "operator": mapped_op,
            "original_expression": comparison_info.get("text", "")
        }

    def _local_rule_parser(self, natural_language_rules: List[Dict[str, Any]]) -> tuple:
        """
        강력한 정규식 기반 로컬 규칙 파서
        현장에서 자주 사용되는 패턴을 사전 정의하여 AI 없이도 높은 정확도 제공

        복합 규칙 처리:
        - 쉼표로 구분된 여러 조건을 개별 규칙으로 분리
        - 필드 간 비교 규칙 (<=, >=, <, >) 지원
        """
        rules = []
        conflicts = []
        rule_counter = 1

        logger.info("Processing %d natural language rules", len(natural_language_rules))

        for nat_rule in natural_language_rules:
            field = nat_rule.get('field', '')
            rule_text = str(nat_rule.get('rule_text', '')).strip()
            sheet = nat_rule.get('sheet', '')
            row = nat_rule.get('row', 0)

            if not field:
                continue

            # 필드명에 줄바꿈이 있고 허용값 패턴이 포함된 경우 처리
            field_allowed_values = []
            if '\n' in field:
                field_parts = field.split('\n', 1)
                field_name_clean = field_parts[0].strip()
                field_extra = field_parts[1].strip() if len(field_parts) > 1 else ""

                # 괄호 안에 "숫자: 설명" 패턴이 있는지 확인
                if field_extra:
                    code_pattern = re.findall(r'(\d+)\s*[:\-]\s*[가-힣A-Za-z]+', field_extra)
                    if code_pattern:
                        field_allowed_values = code_pattern
                        # 규칙 텍스트가 비어있으면 필드 설명을 규칙으로 사용
                        if not rule_text:
                            rule_text = field_extra

                # 필드명을 정리된 이름으로 업데이트 (선택적)
                field = field_name_clean

            if not rule_text and not field_allowed_values:
                continue

            # Track if any rule was created for this nat_rule
            initial_counter = rule_counter

            # CRITICAL: Check if rule_text explicitly contains format patterns FIRST
            has_format_pattern = any(kw in rule_text for kw in ["형식", "format", "YYYYMMDD", "YYYY-MM-DD", "regex", "패턴"])

            # 1. 필수/중복 (Required & Unique)
            if ("공백" in rule_text or "필수" in rule_text or "missing" in rule_text.lower()) and not has_format_pattern:
                rules.append(self._create_rule(
                    rule_counter, field, "required", {},
                    "{field_name}은(는) 필수 입력 항목입니다.", nat_rule, "필수값 체크"
                ))
                rule_counter += 1

            # CRITICAL: Only check for duplicates if NOT a format/date rule
            if ("중복" in rule_text or "unique" in rule_text.lower() or "유일" in rule_text) and not has_format_pattern:
                rules.append(self._create_rule(
                    rule_counter, field, "no_duplicates", {},
                    "{field_name}이(가) 중복되었습니다.", nat_rule, "중복 체크"
                ))
                rule_counter += 1

            # 2. 날짜 형식 (Date)
            if "yyyy" in rule_text.lower() or "날짜" in rule_text or "date" in field.lower():
                # YYYYMMDD
                if "yyyymmdd" in rule_text.lower().replace("-", "").replace("/", ""):
                    rules.append(self._create_rule(
                        rule_counter, field, "format",
                        {"format": "YYYYMMDD", "regex": r"^(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$"},
                        "{field_name} 형식이 올바르지 않습니다. (YYYYMMDD)", nat_rule, "날짜 형식(8자리)"
                    ))
                    rule_counter += 1
                # YYYY-MM-DD
                elif "-" in rule_text:
                    rules.append(self._create_rule(
                        rule_counter, field, "format",
                        {"format": "YYYY-MM-DD", "regex": r"^\d{4}-\d{2}-\d{2}$"},
                        "{field_name} 형식이 올바르지 않습니다. (YYYY-MM-DD)", nat_rule, "날짜 형식(하이픈)"
                    ))
                    rule_counter += 1

            # 3. 주민등록번호
            if "주민" in field or "resident" in field.lower() or "jumin" in field.lower():
                rules.append(self._create_rule(
                    rule_counter, field, "format",
                    {"regex": r"^\d{6}-?[1-4]\d{6}$"},
                    "{field_name} 형식이 올바르지 않습니다.", nat_rule, "주민번호 패턴"
                ))
                rule_counter += 1

            # 4. 성별 (Gender)
            if "성별" in field or "gender" in field.lower():
                allowed = []

                # 텍스트에서 허용값 추출 (예: "1:남자, 2:여자" -> ["1", "2"])
                code_pattern = re.findall(r'(\d+)\s*[:\-]\s*[가-힣]+', rule_text)
                if code_pattern:
                    allowed = code_pattern

                # 패턴2: 괄호 안의 값 (예: "(M/F)" 또는 "(남/여)")
                if not allowed:
                    paren_match = re.search(r'\(([^)]+)\)', rule_text)
                    if paren_match:
                        inner = paren_match.group(1)
                        parts = re.split(r'[/,\s]+', inner)
                        allowed = [p.strip() for p in parts if p.strip() and ':' not in p]

                # 추출 실패 시 규칙 생성 스킵
                if not allowed:
                    rules.append(self._create_rule(
                        rule_counter, field, "format",
                        {"raw_rule": rule_text},
                        f"{{field_name}} 규칙을 확인하세요: {rule_text}", nat_rule, "성별 검증"
                    ))
                else:
                    allowed_preview = ', '.join(allowed[:4])
                    rules.append(self._create_rule(
                        rule_counter, field, "format",
                        {"allowed_values": allowed},
                        f"{{field_name}} 값이 올바르지 않습니다. (허용: {allowed_preview})", nat_rule, "성별 코드 검증"
                    ))
                rule_counter += 1

            # 4-1. 일반 허용값 목록 (성별 외 필드)
            special_keywords_in_rule = any(kw in rule_text for kw in [
                "공백", "필수", "중복", "유일", "형식", "날짜", "YYYY", "이상", "이하"
            ])

            if "성별" not in field and "gender" not in field.lower() and not special_keywords_in_rule:
                allowed_values = []

                # 패턴1: "1:정규직, 3:임원" 형태
                code_pattern = re.findall(r'([A-Za-z0-9]+)\s*[:\-]\s*[가-힣]+', rule_text)
                if code_pattern:
                    allowed_values = code_pattern

                # 패턴2: 괄호 안의 값 "(1/3/4)"
                if not allowed_values:
                    paren_match = re.search(r'\(([^)]+)\)', rule_text)
                    if paren_match:
                        inner = paren_match.group(1)
                        if '/' in inner or ',' in inner:
                            parts = re.split(r'[/,\s]+', inner)
                            allowed_values = [p.strip() for p in parts if p.strip() and ':' not in p]

                # 패턴3: "허용: 1, 3, 4"
                if not allowed_values:
                    allowed_match = re.search(r'(?:허용|allowed)[:\s]*([^\.]+)', rule_text, re.IGNORECASE)
                    if allowed_match:
                        parts = re.split(r'[,\s]+', allowed_match.group(1))
                        allowed_values = [p.strip() for p in parts if p.strip()]

                # 패턴4: 단순 나열 "1, 3, 4"
                if not allowed_values:
                    simple_list_match = re.match(r'^[\s]*([A-Za-z0-9가-힣]{1,10})(?:\s*[,/]\s*([A-Za-z0-9가-힣]{1,10}))+[\s]*$', rule_text)
                    if simple_list_match:
                        parts = re.split(r'[,/\s]+', rule_text)
                        allowed_values = [p.strip() for p in parts if p.strip()]

                if allowed_values and len(allowed_values) >= 2:
                    allowed_preview = ', '.join(allowed_values[:5])
                    rules.append(self._create_rule(
                        rule_counter, field, "format",
                        {"allowed_values": allowed_values},
                        f"{{field_name}} 값이 올바르지 않습니다. (허용: {allowed_preview})", nat_rule, f"허용값({allowed_preview})"
                    ))
                    rule_counter += 1

            # 5. 숫자/금액 범위 또는 타입
            is_numeric_rule = any(kw in rule_text for kw in ["금액", "숫자", "원", "수치", "amount", "number", "numeric"])
            has_range = ">" in rule_text or "<" in rule_text or "이상" in rule_text or "이하" in rule_text

            # 필드 간 비교인지 확인
            is_field_comparison = bool(re.search(r'[가-힣a-zA-Z_]+\s*[<>=]+\s*[가-힣a-zA-Z_]+', rule_text))

            if (has_range or is_numeric_rule) and not is_field_comparison:
                nums = re.findall(r'\d+', rule_text)

                if nums and "이상" in rule_text:
                    rules.append(self._create_rule(
                        rule_counter, field, "range",
                        {"min_value": float(nums[0])},
                        f"{{{{field_name}}}} 값은 {nums[0]} 이상이어야 합니다.", nat_rule, "최소값 검증"
                    ))
                    rule_counter += 1
                elif nums and "이하" in rule_text:
                    rules.append(self._create_rule(
                        rule_counter, field, "range",
                        {"max_value": float(nums[0])},
                        f"{{{{field_name}}}} 값은 {nums[0]} 이하이어야 합니다.", nat_rule, "최대값 검증"
                    ))
                    rule_counter += 1
                elif is_numeric_rule:
                    rules.append(self._create_rule(
                        rule_counter, field, "range",
                        {"min_value": 0},
                        f"{{{{field_name}}}}은(는) 숫자여야 합니다.", nat_rule, "숫자 타입 검증"
                    ))
                    rule_counter += 1

            # 6. 필드 간 비교 규칙 (date_logic)
            split_rules = self._split_compound_rule(rule_text, field)
            for split_info in split_rules:
                if split_info.get("type") == "comparison":
                    comparison_params = self._parse_field_comparison(split_info, field)
                    if comparison_params:
                        compare_field = comparison_params.get("compare_field", "")
                        operator = comparison_params.get("operator", "")

                        op_display = {
                            "less_than_or_equal": "<=",
                            "greater_than_or_equal": ">=",
                            "less_than": "<",
                            "greater_than": ">",
                            "equal": "=",
                            "not_equal": "<>"
                        }.get(operator, operator)

                        error_msg = "{field_name}은(는) " + compare_field + " 조건을 만족해야 합니다. (" + field + " " + op_display + " " + compare_field + ")"

                        rules.append(self._create_rule(
                            rule_counter, field, "date_logic",
                            comparison_params,
                            error_msg, nat_rule,
                            f"필드비교({field}{op_display}{compare_field})"
                        ))
                        rule_counter += 1
                        logger.debug("Created date_logic rule: %s %s %s", field, op_display, compare_field)

            # Fallback: 규칙이 하나도 생성되지 않은 경우
            if rule_counter == initial_counter:
                if field_allowed_values:
                    allowed_preview = ', '.join(field_allowed_values[:5])
                    logger.debug("Created format rule from field name for '%s': allowed_values=%s", field, field_allowed_values)
                    rules.append(self._create_rule(
                        rule_counter, field, "format",
                        {"allowed_values": field_allowed_values},
                        f"{{field_name}} 값이 올바르지 않습니다. (허용: {allowed_preview})", nat_rule, f"허용값({allowed_preview})"
                    ))
                    rule_counter += 1
                else:
                    logger.debug("No specific rule matched for field '%s', creating custom rule", field)
                    rules.append(self._create_rule(
                        rule_counter, field, "custom",
                        {"description": rule_text},
                        f"{{{{field_name}}}} 검증 실패: {rule_text}", nat_rule, "사용자 정의 규칙 (Manual Check)", confidence=0.7
                    ))
                    rule_counter += 1

        logger.info("Generated %d rules total", len(rules))
        for i, rule in enumerate(rules[:5]):
            logger.debug("  Rule %d: %s on field '%s' - %s", i+1, rule.rule_type, rule.field_name, rule.error_message_template[:50])

        return rules, conflicts

    def _create_rule(self, id_num, field, rtype, params, msg, source_dict, summary, confidence=0.95):
        """규칙 객체 생성 헬퍼 (필드명 기반 - 시트명 제거됨)"""
        # row_number를 문자열로 변환 (서브 인덱스 지원: "5", "5.1", "5.2")
        row_num = source_dict.get('row', '0')
        if not isinstance(row_num, str):
            row_num = str(row_num)

        return ValidationRule(
            rule_id=f"RULE_LOCAL_{id_num:03d}",
            field_name=field,
            rule_type=rtype,
            parameters=params,
            error_message_template=msg,
            source={
                "original_text": source_dict.get('rule_text', ''),
                "row_number": row_num,
                "kifrs_reference": None
            },
            ai_interpretation_summary=summary,
            confidence_score=confidence
        )

    def interpret_rule(self, rule_text: str, column_name: str = "", use_local_parser: bool = True) -> Dict[str, Any]:
        """
        단일 규칙 텍스트를 해석하여 검증 설정 반환 (복합 조건 지원)

        복합 조건이 감지되면 composite 타입으로 반환하고,
        validations 배열에 각 검증 조건을 포함합니다.

        Args:
            rule_text: 규칙 원문 (자연어)
            column_name: 필드명
            use_local_parser: True면 로컬 파서 사용

        Returns:
            Dict: {
                "rule_type": str,
                "rule_id": str,
                "parameters": dict,
                "error_message": str,
                "confidence_score": float,
                "interpretation_summary": str
            }
        """
        if not rule_text:
            return {
                "rule_type": "custom",
                "rule_id": "RULE_EMPTY",
                "parameters": {},
                "error_message": "{field_name} 검증 실패",
                "confidence_score": 0.5,
                "interpretation_summary": "규칙 텍스트 없음"
            }

        # 복합 조건 감지를 위한 검증 목록
        validations = []
        summaries = []

        rule_text_lower = rule_text.lower()

        # ===== 1. 필수 입력 (Required) =====
        if any(kw in rule_text for kw in ["공백", "필수", "빈값", "비어있으면"]) or "missing" in rule_text_lower:
            validations.append({
                "type": "required",
                "parameters": {},
                "error_message": "{field_name}은(는) 필수 입력 항목입니다."
            })
            summaries.append("필수값")

        # ===== 2. 중복 검증 (No Duplicates) =====
        has_format_pattern = any(kw in rule_text for kw in ["형식", "format", "YYYYMMDD", "YYYY-MM-DD"])
        if any(kw in rule_text for kw in ["중복", "유일"]) or "unique" in rule_text_lower:
            if not has_format_pattern:
                validations.append({
                    "type": "no_duplicates",
                    "parameters": {},
                    "error_message": "{field_name}이(가) 중복되었습니다."
                })
                summaries.append("중복불가")

        # ===== 3. 날짜 형식 (Date Format) =====
        if "yyyy" in rule_text_lower or "날짜" in rule_text or "date" in column_name.lower():
            if "yyyymmdd" in rule_text_lower.replace("-", "").replace("/", ""):
                validations.append({
                    "type": "format",
                    "parameters": {
                        "format": "YYYYMMDD",
                        "regex": r"^(19|20)\d{2}(0[1-9]|1[0-2])(0[1-9]|[12]\d|3[01])$"
                    },
                    "error_message": "{field_name} 형식이 올바르지 않습니다. (YYYYMMDD)"
                })
                summaries.append("YYYYMMDD형식")
            elif "yyyy-mm-dd" in rule_text_lower or "-" in rule_text:
                validations.append({
                    "type": "format",
                    "parameters": {
                        "format": "YYYY-MM-DD",
                        "regex": r"^\d{4}-\d{2}-\d{2}$"
                    },
                    "error_message": "{field_name} 형식이 올바르지 않습니다. (YYYY-MM-DD)"
                })
                summaries.append("YYYY-MM-DD형식")

        # ===== 4. 허용값 목록 (Allowed Values) =====
        allowed_values = []

        # 패턴1: "1:남자, 2:여자" 형태
        code_pattern = re.findall(r'(\d+)\s*[:\-]\s*[가-힣]+', rule_text)
        if code_pattern:
            allowed_values = code_pattern

        # 패턴2: 괄호 안의 값 "(M/F)" 또는 "(남/여)"
        if not allowed_values:
            paren_match = re.search(r'\(([^)]+)\)', rule_text)
            if paren_match:
                inner = paren_match.group(1)
                if '/' in inner or ',' in inner:
                    parts = re.split(r'[/,\s]+', inner)
                    allowed_values = [p.strip() for p in parts if p.strip() and ':' not in p]

        # 패턴3: "허용:" 또는 "allowed:" 뒤의 값
        allowed_match = re.search(r'(?:허용|allowed)[:\s]*([^\.]+)', rule_text, re.IGNORECASE)
        if allowed_match and not allowed_values:
            parts = re.split(r'[,\s]+', allowed_match.group(1))
            allowed_values = [p.strip() for p in parts if p.strip()]

        # 패턴4: 단순 나열 "1, 3, 4" 또는 "1,3,4"
        if not allowed_values:
            simple_list_match = re.match(r'^[\s]*([A-Za-z0-9가-힣]{1,10})(?:\s*[,/]\s*([A-Za-z0-9가-힣]{1,10}))+[\s]*$', rule_text)
            if simple_list_match:
                parts = re.split(r'[,/\s]+', rule_text)
                allowed_values = [p.strip() for p in parts if p.strip()]

        if allowed_values:
            validations.append({
                "type": "allowed_values",
                "parameters": {"allowed_values": allowed_values},
                "error_message": "{field_name} 값이 올바르지 않습니다. (허용: " + ", ".join(allowed_values[:4]) + ")"
            })
            summaries.append(f"나열형({','.join(allowed_values[:3])})")

        # ===== 5. 숫자 범위 (Range) =====
        has_range = any(kw in rule_text for kw in ["이상", "이하", "초과", "미만"]) or \
                    ">" in rule_text or "<" in rule_text

        if has_range:
            nums = re.findall(r'[\d.]+', rule_text)
            range_params = {}
            range_msgs = []

            if nums:
                if "이상" in rule_text or ">=" in rule_text:
                    range_params["min_value"] = float(nums[0])
                    range_msgs.append(f"{nums[0]} 이상")
                if "이하" in rule_text or "<=" in rule_text:
                    idx = 1 if "이상" in rule_text and len(nums) > 1 else 0
                    if idx < len(nums):
                        range_params["max_value"] = float(nums[idx])
                        range_msgs.append(f"{nums[idx]} 이하")
                if "초과" in rule_text or ">" in rule_text and ">=" not in rule_text:
                    range_params["min_value"] = float(nums[0])
                    range_params["exclusive_min"] = True
                    range_msgs.append(f"{nums[0]} 초과")
                if "미만" in rule_text or "<" in rule_text and "<=" not in rule_text:
                    idx = 1 if len(nums) > 1 else 0
                    range_params["max_value"] = float(nums[idx])
                    range_params["exclusive_max"] = True
                    range_msgs.append(f"{nums[idx]} 미만")

            if range_params:
                validations.append({
                    "type": "range",
                    "parameters": range_params,
                    "error_message": "{field_name} 값은 " + ", ".join(range_msgs) + "이어야 합니다."
                })
                summaries.append("범위(" + ", ".join(range_msgs) + ")")

        # ===== 6. 숫자 타입 검증 =====
        is_numeric_rule = any(kw in rule_text for kw in ["금액", "숫자", "원", "수치", "정수"]) or \
                          any(kw in rule_text_lower for kw in ["amount", "number", "numeric", "integer"])

        # 이미 range 검증이 추가되지 않은 경우에만
        if is_numeric_rule and not has_range:
            validations.append({
                "type": "range",
                "parameters": {"numeric_only": True},
                "error_message": "{field_name}은(는) 숫자여야 합니다."
            })
            summaries.append("숫자타입")

        # ===== 7. 필드 간 비교 (Date Logic / Cross Field) =====
        split_rules = self._split_compound_rule(rule_text, column_name)
        for split_info in split_rules:
            if split_info.get("type") == "comparison":
                comparison_params = self._parse_field_comparison(split_info, column_name)
                if comparison_params:
                    compare_field = comparison_params.get("compare_field", "")
                    operator = comparison_params.get("operator", "")

                    op_display = {
                        "less_than_or_equal": "<=",
                        "greater_than_or_equal": ">=",
                        "less_than": "<",
                        "greater_than": ">",
                        "equal": "=",
                        "not_equal": "<>"
                    }.get(operator, operator)

                    validations.append({
                        "type": "date_logic",
                        "parameters": comparison_params,
                        "error_message": f"{{field_name}}은(는) {compare_field} 조건({op_display})을 만족해야 합니다."
                    })
                    summaries.append(f"필드비교({column_name}{op_display}{compare_field})")

        # ===== 결과 생성 =====
        if len(validations) == 0:
            return {
                "rule_type": "custom",
                "rule_id": "RULE_CUSTOM_001",
                "parameters": {"description": rule_text},
                "error_message": "{field_name} 검증 실패: " + rule_text[:50],
                "confidence_score": 0.6,
                "interpretation_summary": "사용자 정의 규칙 (수동 확인 필요)"
            }
        elif len(validations) == 1:
            v = validations[0]
            return {
                "rule_type": v["type"],
                "rule_id": f"RULE_{v['type'].upper()}_001",
                "parameters": v["parameters"],
                "error_message": v["error_message"],
                "confidence_score": 0.9,
                "interpretation_summary": summaries[0]
            }
        else:
            return {
                "rule_type": "composite",
                "rule_id": "RULE_COMPOSITE_001",
                "parameters": {
                    "validations": validations
                },
                "error_message": "{field_name} 검증 실패: " + ", ".join(summaries),
                "confidence_score": 0.85,
                "interpretation_summary": " + ".join(summaries)
            }

    # =========================================================================
    # Local Fix Engine (Smart Cleaner)
    # =========================================================================

    def _local_fix_engine(self, errors: List[Dict[str, Any]]) -> List[FixSuggestion]:
        """
        현장 데이터 최적화된 스마트 수정 엔진

        핵심 원칙:
        - 필드 타입과 값 타입이 일치하는 경우에만 자동 수정 제안
        - 금액 필드에 날짜가 들어간 경우 -> 자동수정 불가 (완전히 잘못된 데이터)
        - 날짜 필드에 날짜 형식이 다른 경우 -> 형식 변환 제안
        """
        suggestions = []

        # 금액/숫자 관련 필드 키워드
        numeric_field_keywords = [
            "급여", "금액", "수당", "원", "임금", "보수", "연봉", "월급",
            "salary", "amount", "wage", "pay", "bonus", "income",
            "기준급", "평균급", "통상급", "퇴직금", "retirement"
        ]

        # 날짜 관련 필드 키워드
        date_field_keywords = [
            "일", "일자", "date", "날짜", "기준일", "입사", "퇴사", "생년월일",
            "정산", "산정", "기산", "만료", "시작", "종료"
        ]

        for err in errors:
            val = str(err.get('actual_value', ''))
            field = str(err.get('column', ''))
            msg = str(err.get('message', ''))

            # 규칙 문맥 활용
            rule_type = err.get('rule_type', '')
            rule_params = err.get('rule_params', {}) or {}

            # Skip invalid values
            if val == 'None' or val == 'nan' or not val:
                continue

            fixed = val
            reason = ""
            score = 0.0
            auto = False

            field_lower = field.lower()

            # 필드 타입 판별 (규칙 우선)
            is_numeric_rule = rule_type == 'range' or \
                              (rule_type == 'format' and 'numeric' in str(rule_params)) or \
                              (rule_type == 'custom' and any(kw in str(rule_params) for kw in ['number', 'amount', '금액']))

            is_numeric_keyword = any(kw in field for kw in numeric_field_keywords) or \
                                 any(kw in field_lower for kw in ["salary", "amount", "wage", "pay"])

            is_numeric_field = is_numeric_rule or is_numeric_keyword

            is_date_rule = rule_type == 'date_logic' or \
                           (rule_type == 'format' and ('YYYY' in str(rule_params) or 'date' in str(rule_params)))

            is_date_keyword = any(kw in field for kw in date_field_keywords) or \
                              any(kw in field_lower for kw in ["date"])

            is_date_field = is_date_rule or is_date_keyword

            # 입력값이 날짜 형식인지 확인
            is_date_value = bool(re.match(r'^\d{4}[-./]\d{2}[-./]\d{2}$', val))

            # 1. 금액/숫자 필드에 날짜가 들어간 경우 -> 자동수정 불가
            if is_numeric_field and is_date_value:
                continue

            # 2. 날짜 필드에서 날짜 형식 표준화 (YYYYMMDD)
            is_date_format_error = any(kw in msg for kw in ["YYYYMMDD", "날짜", "형식"])

            if is_date_field and is_date_format_error:
                # 2023-01-01 -> 20230101
                if re.match(r'^\d{4}-\d{2}-\d{2}$', val):
                    fixed = val.replace("-", "")
                    reason = "표준 포맷으로 변환 (- 제거)"
                    score = 0.99
                    auto = True
                # 2023.01.01 -> 20230101
                elif re.match(r'^\d{4}\.\d{2}\.\d{2}$', val):
                    fixed = val.replace(".", "")
                    reason = "표준 포맷으로 변환 (. 제거)"
                    score = 0.99
                    auto = True
                # 2023/01/01 -> 20230101
                elif re.match(r'^\d{4}/\d{2}/\d{2}$', val):
                    fixed = val.replace("/", "")
                    reason = "표준 포맷으로 변환 (/ 제거)"
                    score = 0.99
                    auto = True

            # 2. 성별 표준화
            elif "성별" in field or "gender" in field.lower():
                val_clean = val.strip().lower()
                if val_clean in ['남', '남자', 'man', 'male']:
                    fixed = 'M'
                    reason = "성별 코드 표준화 (남 -> M)"
                    score = 0.98
                    auto = True
                elif val_clean in ['여', '여자', 'woman', 'female']:
                    fixed = 'F'
                    reason = "성별 코드 표준화 (여 -> F)"
                    score = 0.98
                    auto = True

            # 3. 주민번호 표준화
            elif "주민" in field:
                if "-" in val and len(val) == 14:
                    pass

            # 4. 공백 제거 (범용)
            if fixed == val and " " in val:
                if val.strip() != val:
                    fixed = val.strip()
                    reason = "앞뒤 공백 제거"
                    score = 0.95
                    auto = True
                elif "사번" in field or "id" in field.lower() or "주민" in field:
                    fixed = val.replace(" ", "")
                    reason = "식별자 내 불필요한 공백 제거"
                    score = 0.90
                    auto = False

            # 수정사항이 있으면 추가
            if fixed != val:
                suggestions.append(FixSuggestion(
                    error_id=err.get('id'),
                    sheet_name=err.get('sheet', ''),
                    row=err.get('row', 0),
                    column=err.get('column', ''),
                    original_value=val,
                    fixed_value=fixed,
                    confidence_score=score,
                    reason=reason,
                    is_auto_fixable=auto
                ))

        return suggestions

    # =========================================================================
    # Summary & Utility
    # =========================================================================

    def _generate_summary(self, rules, conflicts):
        return f"해석 완료: 규칙 {len(rules)}개, 충돌 {len(conflicts)}건 (Engine: {'Cloud AI' if self.use_cloud_ai else 'Local Regex'})"

    def calculate_completeness_score(
        self,
        sheet_data: Dict[str, Any],
        column_names: Dict[str, List[str]]
    ) -> Dict[str, Any]:
        """
        데이터 완전성 점수를 계산합니다.

        각 행/시트별 필수 필드 채움 비율을 점수화합니다.

        Returns:
            Dict: overall_score, sheet_scores, field_scores
        """
        import pandas as pd

        sheet_scores = {}
        field_scores = {}
        total_cells = 0
        filled_cells = 0

        for sheet_name, df in sheet_data.items():
            if not isinstance(df, pd.DataFrame) or len(df) == 0:
                continue

            cols = [str(c) for c in df.columns]
            sheet_total = 0
            sheet_filled = 0
            col_scores = {}

            for col in cols:
                col_total = len(df)
                col_str = df[col].astype(str).str.strip()
                col_empty = col_str.isin(['', 'nan', 'None', 'NaT', 'NaN']).sum() + df[col].isna().sum()
                col_filled = col_total - int(col_empty)

                col_scores[col] = {
                    "total": col_total,
                    "filled": col_filled,
                    "empty": int(col_empty),
                    "score": round(col_filled / col_total * 100, 1) if col_total > 0 else 0
                }

                sheet_total += col_total
                sheet_filled += col_filled

            total_cells += sheet_total
            filled_cells += sheet_filled

            sheet_score = round(sheet_filled / sheet_total * 100, 1) if sheet_total > 0 else 0
            sheet_scores[sheet_name] = {
                "score": sheet_score,
                "total_cells": sheet_total,
                "filled_cells": sheet_filled,
                "rows": len(df),
                "columns": len(cols),
                "field_scores": col_scores
            }

            for col, score_data in col_scores.items():
                if col not in field_scores or score_data["score"] < field_scores[col]["score"]:
                    field_scores[col] = {**score_data, "sheet": sheet_name}

        overall_score = round(filled_cells / total_cells * 100, 1) if total_cells > 0 else 0

        # 최저 점수 필드 Top 10
        worst_fields = sorted(field_scores.items(), key=lambda x: x[1]["score"])[:10]

        return {
            "overall_score": overall_score,
            "total_cells": total_cells,
            "filled_cells": filled_cells,
            "empty_cells": total_cells - filled_cells,
            "sheet_scores": sheet_scores,
            "worst_fields": [{"field": f, **s} for f, s in worst_fields],
            "summary": f"전체 데이터 완전성: {overall_score}% ({filled_cells}/{total_cells} 셀)"
        }

    def detect_rule_conflicts(self, rules: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        서로 모순되는 규칙을 자동 탐지합니다.

        예: "A > 100" AND "A < 50"는 동시에 만족 불가

        Args:
            rules: 규칙 리스트 (DB에서 조회한 규칙들)

        Returns:
            Dict: conflicts 리스트
        """
        conflicts = []
        rules_by_field = {}

        # 필드별 규칙 그룹화
        for rule in rules:
            field = rule.get("field_name", "")
            if field not in rules_by_field:
                rules_by_field[field] = []
            rules_by_field[field].append(rule)

        for field, field_rules in rules_by_field.items():
            if len(field_rules) < 2:
                continue

            # 범위 충돌 탐지
            range_rules = [r for r in field_rules if r.get("ai_rule_type") == "range"]
            if len(range_rules) >= 2:
                for i in range(len(range_rules)):
                    for j in range(i + 1, len(range_rules)):
                        r1 = range_rules[i].get("ai_parameters", {})
                        r2 = range_rules[j].get("ai_parameters", {})

                        min1 = r1.get("min_value")
                        max1 = r1.get("max_value")
                        min2 = r2.get("min_value")
                        max2 = r2.get("max_value")

                        if min1 is not None and max2 is not None and min1 > max2:
                            conflicts.append({
                                "type": "range_contradiction",
                                "severity": "high",
                                "field": field,
                                "rule1_id": range_rules[i].get("id", ""),
                                "rule1_text": range_rules[i].get("rule_text", ""),
                                "rule2_id": range_rules[j].get("id", ""),
                                "rule2_text": range_rules[j].get("rule_text", ""),
                                "description": f"'{field}' 범위 모순: 최솟값({min1}) > 다른 규칙의 최댓값({max2})"
                            })
                        if max1 is not None and min2 is not None and max1 < min2:
                            conflicts.append({
                                "type": "range_contradiction",
                                "severity": "high",
                                "field": field,
                                "rule1_id": range_rules[i].get("id", ""),
                                "rule1_text": range_rules[i].get("rule_text", ""),
                                "rule2_id": range_rules[j].get("id", ""),
                                "rule2_text": range_rules[j].get("rule_text", ""),
                                "description": f"'{field}' 범위 모순: 최댓값({max1}) < 다른 규칙의 최솟값({min2})"
                            })

            # 허용값 충돌 탐지
            allowed_rules = [r for r in field_rules if r.get("ai_rule_type") == "allowed_values"]
            if len(allowed_rules) >= 2:
                sets = []
                for r in allowed_rules:
                    vals = r.get("ai_parameters", {}).get("allowed_values", [])
                    sets.append(set(vals))

                for i in range(len(sets)):
                    for j in range(i + 1, len(sets)):
                        overlap = sets[i] & sets[j]
                        if not overlap:
                            conflicts.append({
                                "type": "allowed_values_no_overlap",
                                "severity": "high",
                                "field": field,
                                "rule1_id": allowed_rules[i].get("id", ""),
                                "rule1_text": allowed_rules[i].get("rule_text", ""),
                                "rule2_id": allowed_rules[j].get("id", ""),
                                "rule2_text": allowed_rules[j].get("rule_text", ""),
                                "description": f"'{field}' 허용값 모순: 두 규칙의 허용값에 겹치는 값이 없습니다."
                            })

            # 유형 충돌 탐지 (required + allowed_values에 빈값 없음 등)
            has_required = any(r.get("ai_rule_type") == "required" for r in field_rules)
            format_rules = [r for r in field_rules if r.get("ai_rule_type") == "format"]

            if has_required and format_rules:
                for fr in format_rules:
                    params = fr.get("ai_parameters", {})
                    if params.get("allow_empty", False):
                        conflicts.append({
                            "type": "required_vs_allow_empty",
                            "severity": "medium",
                            "field": field,
                            "rule1_text": "필수 입력 규칙",
                            "rule2_text": fr.get("rule_text", ""),
                            "description": f"'{field}' 필수 규칙과 빈값 허용 형식 규칙이 충돌합니다."
                        })

        return {
            "total_conflicts": len(conflicts),
            "conflicts": conflicts,
            "fields_checked": len(rules_by_field),
            "summary": f"{len(conflicts)}건의 규칙 충돌이 감지되었습니다." if conflicts else "규칙 충돌이 없습니다."
        }
