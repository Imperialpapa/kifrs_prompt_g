"""
K-IFRS 1019 DBO Validation System - AI Interpretation Layer
============================================================
AI 및 로컬 엔진을 활용한 하이브리드 규칙 해석/수정 시스템

Feature:
1. Multi-Provider: Anthropic, OpenAI, Gemini 지원 (환경변수로 선택)
2. Hybrid Engine: Cloud AI 실패 시 Regex Parser(Local) 자동 전환
3. Auto-Fix: 데이터 클렌징을 위한 결정론적 수정 제안 로직
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

# Optional Imports with Graceful Fallback
try:
    import anthropic
    ANTHROPIC_AVAILABLE = True
except ImportError:
    ANTHROPIC_AVAILABLE = False

try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


class AIRuleInterpreter:
    """
    Multi-Provider AI 규칙 해석기
    """
    
    def __init__(self):
        """
        초기화: 기본 설정 로드
        """
        self.default_provider = os.getenv("AI_PROVIDER", "openai").lower()
        self.use_cloud_ai = False  # Track whether cloud AI was used
        print(f"[AIRuleInterpreter] Default Provider: {self.default_provider.upper()}")

    def _check_provider_availability(self, provider: str) -> bool:
        """
        지정된 Provider가 사용 가능한지 확인

        Args:
            provider: "openai", "anthropic", "gemini"

        Returns:
            bool: 사용 가능 여부
        """
        if provider == "openai":
            return OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY")
        elif provider in ["anthropic", "claude"]:
            return ANTHROPIC_AVAILABLE and os.getenv("ANTHROPIC_API_KEY")
        elif provider == "gemini":
            return GEMINI_AVAILABLE and os.getenv("GEMINI_API_KEY")
        return False
    
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
                print(f"[AI] Interpreting rules using {target_provider.upper()}...")
                prompt = self._build_interpretation_prompt(natural_language_rules)
                ai_response = await self._call_cloud_ai(prompt, target_provider)
                rules, conflicts = self._parse_ai_response(ai_response)
                self.use_cloud_ai = True
            except Exception as e:
                print(f"[AI] Cloud inference ({target_provider}) failed, falling back to local engine: {e}")
                rules, conflicts = self._local_rule_parser(natural_language_rules)
                self.use_cloud_ai = False
        else:
            print(f"[AI] Provider {target_provider} not available/configured. Using Local Engine.")
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
        
        target_provider = (provider or self.default_provider).lower()
        use_cloud = self._check_provider_availability(target_provider)

        if use_cloud:
            # 로컬 엔진이 처리하지 못한 항목이나 신뢰도 낮은 항목에 대해 AI 호출 고려 가능
            # 현재는 일관성을 위해 클라우드 AI에게 전체 문맥을 전달하여 제안을 정교화함
            try:
                prompt = self._build_correction_prompt(errors, past_corrections)
                ai_response = await self._call_cloud_ai(prompt, target_provider)
                cloud_suggestions = self._parse_correction_response(ai_response)
                
                # 클라우드 제안이 있으면 우선 사용 (병합)
                return cloud_suggestions if cloud_suggestions else local_suggestions
            except Exception as e:
                print(f"[AI] Cloud correction failed ({target_provider}), using local engine: {e}")
                return local_suggestions
        
        return local_suggestions

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
            print(f"[AI] Error getting explanation: {e}")
            return {
                "explanation": "오류 설명을 생성하는 중 문제가 발생했습니다.",
                "recommendation": "오류 메시지를 참고하여 데이터를 수정하세요."
            }

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
        except:
            return []

    async def _call_cloud_ai(self, prompt: str, provider: str) -> str:
        """선택된 Provider의 API 호출 (OpenAI JSON 모드 적극 활용)"""
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            client = openai.OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model=os.getenv("AI_MODEL_VERSION_OPENAI", "gpt-4o"),
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )
            return response.choices[0].message.content
        
        # Anthropic, Gemini 로직 생략 (기존과 동일)
        return await getattr(self, f"_call_{provider}_api")(prompt)

    async def _call_claude_api(self, prompt: str) -> str:
        """Anthropic Claude API"""
        api_key = os.getenv("ANTHROPIC_API_KEY")
        model = os.getenv("AI_MODEL_VERSION_ANTHROPIC", "claude-3-haiku-20240307")
        
        client = anthropic.Anthropic(api_key=api_key)
        message = client.messages.create(
            model=model,
            max_tokens=4000,
            temperature=0.0,
            system="You are a strict data validation rule parser. Output JSON only.",
            messages=[{"role": "user", "content": prompt}]
        )
        return message.content[0].text

    async def _call_openai_api(self, prompt: str) -> str:
        """OpenAI GPT API"""
        api_key = os.getenv("OPENAI_API_KEY")
        model = os.getenv("AI_MODEL_VERSION_OPENAI", "gpt-4o")
        
        client = openai.OpenAI(api_key=api_key)
        response = client.chat.completions.create(
            model=model,
            temperature=0.0,
            messages=[
                {"role": "system", "content": "You are a strict data validation rule parser. Output JSON only."},
                {"role": "user", "content": prompt}
            ],
            response_format={"type": "json_object"}
        )
        return response.choices[0].message.content

    async def _call_gemini_api(self, prompt: str) -> str:
        """Google Gemini API"""
        api_key = os.getenv("GEMINI_API_KEY")
        model = os.getenv("AI_MODEL_VERSION_GEMINI", "gemini-1.5-flash")
        
        genai.configure(api_key=api_key)
        gemini_model = genai.GenerativeModel(
            model,
            generation_config={"response_mime_type": "application/json"}
        )
        response = gemini_model.generate_content(prompt)
        return response.text

    # =========================================================================
    # 🏗️ Common Logic
    # =========================================================================

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
        - error_message_template: "사원번호이(가) 중복되었습니다"  ❌ WRONG!

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
            print(f"[AI] Failed to parse JSON response: {e}")
            raise

    # =========================================================================
    # 💻 Local Rule Engine (Robust Regex Parser)
    # =========================================================================

    def _local_rule_parser(self, natural_language_rules: List[Dict[str, Any]]) -> tuple:
        """
        강력한 정규식 기반 로컬 규칙 파서
        현장에서 자주 사용되는 패턴을 사전 정의하여 AI 없이도 높은 정확도 제공
        """
        rules = []
        conflicts = []
        rule_counter = 1

        print(f"[LocalParser] Processing {len(natural_language_rules)} natural language rules")

        for nat_rule in natural_language_rules:
            field = nat_rule.get('field', '')
            rule_text = str(nat_rule.get('rule_text', '')).strip()
            sheet = nat_rule.get('sheet', '')
            row = nat_rule.get('row', 0)

            if not field or not rule_text:
                continue

            # Track if any rule was created for this nat_rule
            initial_counter = rule_counter

            # CRITICAL: Check if rule_text explicitly contains format patterns FIRST
            # This prevents "YYYYMMDD 형식" from being misclassified as duplicate
            has_format_pattern = any(kw in rule_text for kw in ["형식", "format", "YYYYMMDD", "YYYY-MM-DD", "regex", "패턴"])

            # 1. 필수/중복 (Required & Unique)
            # "공백, 중복" 처럼 콤마로 구분된 경우 처리
            # BUT: Only apply if not a format rule
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

                # 텍스트에서 허용값 추출 (예: "1:남자, 2:여자" → ["1", "2"])
                # 패턴1: "1:남자" 형태
                code_pattern = re.findall(r'(\d+)\s*[:\-]\s*[가-힣]+', rule_text)
                if code_pattern:
                    allowed = code_pattern

                # 패턴2: 괄호 안의 값 (예: "(M/F)" 또는 "(남/여)")
                if not allowed:
                    paren_match = re.search(r'\(([^)]+)\)', rule_text)
                    if paren_match:
                        inner = paren_match.group(1)
                        # 슬래시, 쉼표, 또는 공백으로 분리
                        parts = re.split(r'[/,\s]+', inner)
                        allowed = [p.strip() for p in parts if p.strip() and ':' not in p]

                # 추출 실패 시 규칙 생성 스킵 (원본 규칙 텍스트로 안내)
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

            # 5. 숫자/금액 범위 또는 타입
            is_numeric_rule = any(kw in rule_text for kw in ["금액", "숫자", "원", "수치", "amount", "number", "numeric"])
            has_range = ">" in rule_text or "<" in rule_text or "이상" in rule_text or "이하" in rule_text

            if has_range or is_numeric_rule:
                nums = re.findall(r'\d+', rule_text)

                # 범위가 있는 경우 (예: "0 이상")
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
                    # 숫자/금액 타입 검증 (범위 없이 숫자인지만 확인)
                    rules.append(self._create_rule(
                        rule_counter, field, "range",
                        {"min_value": 0},  # 0 이상으로 설정하면 숫자 타입 검증됨
                        f"{{{{field_name}}}}은(는) 숫자여야 합니다.", nat_rule, "숫자 타입 검증"
                    ))
                    rule_counter += 1

            # Fallback: 규칙이 하나도 생성되지 않은 경우 Custom 규칙 생성
            if rule_counter == initial_counter:
                print(f"[LocalParser] No specific rule matched for field '{field}', creating custom rule")
                rules.append(self._create_rule(
                    rule_counter, field, "custom",
                    {"description": rule_text},
                    f"{{{{field_name}}}} 검증 실패: {rule_text}", nat_rule, "사용자 정의 규칙 (Manual Check)", confidence=0.7
                ))
                rule_counter += 1

        print(f"[LocalParser] Generated {len(rules)} rules total")
        for i, rule in enumerate(rules[:5]):  # 처음 5개만 출력
            print(f"  Rule {i+1}: {rule.rule_type} on field '{rule.field_name}' - {rule.error_message_template[:50]}")

        return rules, conflicts

    def _create_rule(self, id_num, field, rtype, params, msg, source_dict, summary, confidence=0.95):
        """규칙 객체 생성 헬퍼"""
        return ValidationRule(
            rule_id=f"RULE_LOCAL_{id_num:03d}",
            field_name=field,
            rule_type=rtype,
            parameters=params,
            error_message_template=msg,
            source={
                "original_text": source_dict.get('rule_text', ''),
                "sheet_name": source_dict.get('sheet', ''),
                "row_number": source_dict.get('row', 0),
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
        # 날짜 형식 규칙에서 잘못 감지되지 않도록 주의
        has_format_pattern = any(kw in rule_text for kw in ["형식", "format", "YYYYMMDD", "YYYY-MM-DD"])
        if any(kw in rule_text for kw in ["중복", "유일"]) or "unique" in rule_text_lower:
            if not has_format_pattern:  # 형식 규칙이 아닌 경우에만
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
        # 패턴: "M/F", "1:남, 2:여", "(허용: A, B, C)"
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

        if allowed_values:
            validations.append({
                "type": "format",
                "parameters": {"allowed_values": allowed_values},
                "error_message": "{field_name} 값이 올바르지 않습니다. (허용: " + ", ".join(allowed_values[:4]) + ")"
            })
            summaries.append(f"허용값({','.join(allowed_values[:3])})")

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

        # ===== 결과 생성 =====
        if len(validations) == 0:
            # 해석 실패 - custom 규칙
            return {
                "rule_type": "custom",
                "rule_id": "RULE_CUSTOM_001",
                "parameters": {"description": rule_text},
                "error_message": "{field_name} 검증 실패: " + rule_text[:50],
                "confidence_score": 0.6,
                "interpretation_summary": "사용자 정의 규칙 (수동 확인 필요)"
            }
        elif len(validations) == 1:
            # 단일 검증
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
            # 복합 검증 (composite)
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
    # 🛠️ Local Fix Engine (Smart Cleaner)
    # =========================================================================

    def _local_fix_engine(self, errors: List[Dict[str, Any]]) -> List[FixSuggestion]:
        """
        현장 데이터 최적화된 스마트 수정 엔진
        """
        suggestions = []
        
        for err in errors:
            val = str(err.get('actual_value', ''))
            field = str(err.get('column', ''))
            msg = str(err.get('message', ''))
            
            # Skip invalid values
            if val == 'None' or val == 'nan' or not val:
                continue

            fixed = val
            reason = ""
            score = 0.0
            auto = False

            # 1. 날짜 표준화 (YYYYMMDD)
            if "형식" in msg and ("YYYYMMDD" in msg or "날짜" in field):
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
                # Excel Serial Date (예: 45000 -> 날짜)
                elif re.match(r'^\d{5}$', val):
                    try:
                        # 엑셀 기준일 1899-12-30 처리 로직은 복잡하므로 간단히 언급
                        # 여기서는 단순 숫자 포맷 오류인 경우만 처리
                        pass 
                    except: pass

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
                # 123456-1234567 -> 1234561234567 (규칙이 하이픈 제거인 경우)
                if "-" in val and len(val) == 14:
                    # 규칙에 따라 다름. 여기서는 하이픈 있는게 표준이면 제안 안함.
                    # 만약 규칙이 "형식 오류"라면 포맷팅을 시도
                    pass

            # 4. 공백 제거 (범용)
            if fixed == val and " " in val:
                # 앞뒤 공백
                if val.strip() != val:
                    fixed = val.strip()
                    reason = "앞뒤 공백 제거"
                    score = 0.95
                    auto = True
                # 중간 공백 (사번, 주민번호 등 식별자 컬럼인 경우)
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

    def _generate_summary(self, rules, conflicts):
        return f"해석 완료: 규칙 {len(rules)}개, 충돌 {len(conflicts)}건 (Engine: {'Cloud AI' if self.use_cloud_ai else 'Local Regex'})"
